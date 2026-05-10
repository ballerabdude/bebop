//! 100 Hz policy inference loop for [`crate::mode::Mode::RunPolicy`].
//!
//! Owns:
//!
//! - A [`PolicyController`] (ONNX session + last-action cache + history
//!   buffer; with `HISTORY_STEPS = 1` the buffer is just the latest frame).
//! - An [`ObservationBuilder`] that holds IMU / cmd_vel / joint state and
//!   emits the 36-element observation in the layout fixed by
//!   `bebop_v2_base_cfg.py::PolicyCfg`.
//! - An `Arc<Supervisor>` it consults for joint feedback (read) and pushes
//!   PD commands through (`safe_send_ctrl`, which already enforces the
//!   per-joint hard-limit clamp + slew limit).
//!
//! Threading: the tick is synchronous and intended to be called from the
//! same 100 Hz tokio task that runs the watchdog and the DialIn hold cycle.
//! ONNX inference is sub-millisecond on CPU for our `[512, 256, 128]` MLP,
//! so blocking the executor briefly is fine.
//!
//! ## "No IMU yet" mode
//!
//! Bebop V2 ships without an IMU wired in (yet). We feed the policy
//! synthetic upright-at-rest observations — exactly what the simulator
//! presents at the start of every standing episode (modulo training
//! noise):
//!
//! - `quaternion = [0, 0, 0, 1]` (XYZW identity) ⇒ `projected_gravity = (0, 0, -1)`,
//! - `angular_velocity = (0, 0, 0)`,
//! - `base_lin_vel = (0, 0, 0)`,
//! - `velocity_commands = (0, 0, 0)` — matches `BebopV2FlatBalanceCfg`'s
//!   forced-zero command range.
//!
//! Joint positions and velocities still come from the *real* motor
//! feedback, so the policy does see the real robot's state on the eight
//! joint slots. With the standing-task policy this should produce
//! near-zero actions (the policy was rewarded for staying still while
//! upright), so the practical effect is "hold current pose".
//!
//! When an IMU lands, replace the synthetic block in [`PolicyRunner::tick`]
//! with the live readout. Convention is XYZW per [`crate::observation::ImuState`].

use anyhow::{anyhow, Context, Result};
use std::panic::AssertUnwindSafe;
use std::path::Path;
use std::sync::Arc;
use tracing::{debug, info, warn};

use crate::config::dims;
use crate::mode::Mode;
use crate::observation::{
    scale_actions_to_targets, ImuState, ObservationBuilder, VelocityCommand, JOINT_NAMES,
    NUM_JOINTS,
};
use crate::policy::PolicyController;
use crate::safety::{BreachReason, Supervisor};

/// Sentinel observation values for "no IMU attached". Mirrors what training
/// presents at episode start in the standing task — see module docs.
const SYNTHETIC_IMU_QUATERNION_XYZW: [f32; 4] = [0.0, 0.0, 0.0, 1.0];
const SYNTHETIC_BASE_LIN_VEL: [f32; 3] = [0.0, 0.0, 0.0];

pub struct PolicyRunner {
    controller: PolicyController,
    obs_builder: ObservationBuilder,
    supervisor: Arc<Supervisor>,
    /// `joint_indices[slot]` = index into `Supervisor::cfg().joints` of
    /// the joint occupying policy slot `slot` (0..8 in [`JOINT_NAMES`] order).
    joint_indices: [usize; NUM_JOINTS],
    /// Default joint positions in policy slot order. Used both as the
    /// `obs_builder.default_positions` (for `joint_pos_rel`) and as the
    /// offset in `target = default + scale * action`.
    default_positions: [f32; NUM_JOINTS],
    /// Edge-detect entering RunPolicy so we can clear the policy's history
    /// buffer + last_action cache.
    was_running: bool,
}

impl PolicyRunner {
    /// Load the ONNX policy and resolve the policy-slot ↔ supervisor-joint
    /// mapping by name.
    ///
    /// Errors out if any joint named in [`JOINT_NAMES`] is missing from the
    /// loaded `RobotConfig`. We refuse to silently swap or drop joints —
    /// a misconfigured YAML there would silently break the policy I/O
    /// contract.
    pub fn new<P: AsRef<Path>>(supervisor: Arc<Supervisor>, model_path: P) -> Result<Self> {
        let model_path = model_path.as_ref();
        let cfg = supervisor.cfg();

        let mut joint_indices = [0usize; NUM_JOINTS];
        let mut default_positions = [0.0_f32; NUM_JOINTS];
        for (slot, name) in JOINT_NAMES.iter().enumerate() {
            let joint = cfg.get_joint(name).ok_or_else(|| {
                anyhow!(
                    "policy expects joint {name:?} but it is not present in the loaded \
                     config. Either restore the joint in bebop_v2.yaml or retrain \
                     against the current joint set."
                )
            })?;
            joint_indices[slot] = joint.index;
            default_positions[slot] = joint.default_position;
        }

        // `PolicyController::new` -> `Session::builder()` triggers `ort`'s
        // lazy dylib lookup. With `feature = "load-dynamic"`, that lookup
        // calls `.expect("Failed to load ONNX Runtime dylib")` if
        // libonnxruntime.so cannot be dlopen'd (see ort/src/lib.rs:191).
        // We intercept that panic so bebop-linux can still come up in
        // Idle/DialIn modes when the dylib is missing on the Jetson —
        // RunPolicy will simply be a no-op until the operator installs
        // the lib (or sets `ORT_DYLIB_PATH`) and restarts the service.
        let controller = match std::panic::catch_unwind(AssertUnwindSafe(|| {
            PolicyController::new(model_path)
        })) {
            Ok(result) => {
                result.with_context(|| format!("load policy ONNX from {}", model_path.display()))?
            }
            Err(panic_payload) => {
                let msg = panic_payload
                    .downcast_ref::<&str>()
                    .map(|s| (*s).to_string())
                    .or_else(|| panic_payload.downcast_ref::<String>().cloned())
                    .unwrap_or_else(|| "unknown panic".to_string());
                return Err(anyhow!(
                    "ORT panicked loading policy from {}: {msg}. \
                     Most likely libonnxruntime.so cannot be dlopen'd; install \
                     it (e.g. via Microsoft's aarch64 prebuilt) or set \
                     ORT_DYLIB_PATH. RunPolicy mode will be unavailable.",
                    model_path.display()
                ));
            }
        };
        let mut obs_builder = ObservationBuilder::new();
        obs_builder.set_default_positions(&default_positions);

        info!(
            model = %model_path.display(),
            obs_dim = dims::OBS_DIM,
            action_dim = dims::ACTION_DIM,
            "policy runner ready"
        );

        Ok(Self {
            controller,
            obs_builder,
            supervisor,
            joint_indices,
            default_positions,
            was_running: false,
        })
    }

    /// Run one inference + TX cycle. No-ops outside RunPolicy or while
    /// E-STOP is latched; both branches also clear the on-entry state so
    /// re-entering RunPolicy starts fresh.
    pub fn tick(&mut self) {
        let sup = self.supervisor.clone();
        let in_run_policy = sup.mode() == Mode::RunPolicy && !sup.estop_active();

        if !in_run_policy {
            if self.was_running {
                self.controller.reset();
                self.obs_builder
                    .update_last_action(&[0.0_f32; dims::ACTION_DIM]);
                debug!("policy controller reset on RunPolicy exit");
            }
            self.was_running = false;
            return;
        }

        if !self.was_running {
            // Entering RunPolicy: clear any stale history / last_action so
            // the first observation matches a fresh-episode condition.
            self.controller.reset();
            self.obs_builder
                .update_last_action(&[0.0_f32; dims::ACTION_DIM]);
            info!("RunPolicy entered; policy controller reset");
            self.was_running = true;
        }

        // 1) Pull real joint feedback from the supervisor and lay it out in
        //    policy-slot order.
        let snapshots = sup.snapshot_motors();
        let mut joint_pos = [0.0_f32; NUM_JOINTS];
        let mut joint_vel = [0.0_f32; NUM_JOINTS];
        for (slot, &idx) in self.joint_indices.iter().enumerate() {
            let s = &snapshots[idx];
            joint_pos[slot] = s.position;
            joint_vel[slot] = s.velocity;
        }

        // 2) Synthetic "no IMU" observation block. Replace with live IMU
        //    when the sensor is wired in (XYZW quaternion convention,
        //    body-frame angular velocity, body-frame linear velocity from
        //    an external estimator).
        self.obs_builder.update_imu(ImuState {
            quaternion: SYNTHETIC_IMU_QUATERNION_XYZW,
            angular_velocity: [0.0; 3],
            linear_acceleration: [0.0; 3],
        });
        self.obs_builder
            .update_base_velocity(SYNTHETIC_BASE_LIN_VEL);

        // 3) Velocity command. Isaac-BebopV2-Flat-v0 forces (0, 0, 0)
        //    during training; locomotion checkpoints will want a real
        //    UDP / WS feed plumbed in here.
        self.obs_builder.update_cmd_vel(VelocityCommand::default());

        // 4) Joint state.
        self.obs_builder.joint_positions = joint_pos;
        self.obs_builder.joint_velocities = joint_vel;

        // 5) Build the 36-dim observation, run inference.
        let obs = self.obs_builder.build();
        let action = match self.controller.step(&obs) {
            Ok(a) => a,
            Err(e) => {
                warn!(error = %e, "policy inference failed; latching E-STOP");
                sup.trigger_estop(BreachReason::Operator(format!(
                    "policy inference error: {e}"
                )));
                return;
            }
        };

        if action.len() != dims::ACTION_DIM {
            warn!(
                got = action.len(),
                expected = dims::ACTION_DIM,
                "policy returned wrong-shape action; latching E-STOP"
            );
            sup.trigger_estop(BreachReason::Operator(format!(
                "policy action shape mismatch: got {}, expected {}",
                action.len(),
                dims::ACTION_DIM
            )));
            return;
        }

        // 6) Mirror the action into ObservationBuilder.last_action so it
        //    appears in *next* tick's obs[25..33]. (PolicyController stores
        //    its own copy too, but the obs is built externally here.)
        self.obs_builder.update_last_action(&action);

        // 7) Convert raw action into 8 joint-position targets:
        //       target[i] = default[i] + 0.8 * clip(action[i], -1, 1)
        //    The supervisor's `safe_send_ctrl` will clamp again to
        //    per-joint pos_min..pos_max and slew-limit per tick.
        let targets = scale_actions_to_targets(&action, &self.default_positions);

        // 8) Push to motors. Use per-joint hold_gains for kp/kd; these
        //    should ideally match the gains baked into the training-time
        //    actuator config (see `BEBOP_V2_CFG.actuators` in
        //    `bebop_v2_base_cfg.py`). They currently differ — known
        //    sim-to-real gap.
        for (slot, &idx) in self.joint_indices.iter().enumerate() {
            let cfg = &sup.cfg().joints[idx];
            let kp = cfg.hold_gains.kp;
            let kd = cfg.hold_gains.kd;
            if let Err(e) = sup.safe_send_ctrl(idx, targets[slot], kp, kd, 0.0, 0.0) {
                debug!(joint = %cfg.name, error = %e, "policy TX failed");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthetic_imu_is_xyzw_identity() {
        // Sanity: if anyone reverts the WXYZ -> XYZW migration, this test
        // prevents the no-IMU sentinel from silently flipping the gravity
        // vector.
        let imu = ImuState {
            quaternion: SYNTHETIC_IMU_QUATERNION_XYZW,
            ..Default::default()
        };
        let g = imu.projected_gravity();
        assert!(g[0].abs() < 1e-5);
        assert!(g[1].abs() < 1e-5);
        assert!((g[2] - (-1.0)).abs() < 1e-5);
    }

    #[test]
    fn joint_names_count_matches_action_dim() {
        assert_eq!(JOINT_NAMES.len(), NUM_JOINTS);
        assert_eq!(JOINT_NAMES.len(), dims::ACTION_DIM);
    }
}
