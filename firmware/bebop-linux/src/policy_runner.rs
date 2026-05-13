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
//! - An [`ImuShared`] handle that the [`crate::imu`] thread fills with the
//!   latest body-frame BNO085 quaternion + calibrated gyroscope reading.
//!
//! Threading: the tick is synchronous and intended to be called from the
//! same 100 Hz tokio task that runs the watchdog and the DialIn hold cycle.
//! ONNX inference is sub-millisecond on CPU for our `[512, 256, 128]` MLP,
//! so blocking the executor briefly is fine.
//!
//! ## IMU sourcing
//!
//! Each tick we lock [`ImuShared`] and try to copy the latest body-frame
//! `quaternion` + `angular_velocity_body` into the
//! [`crate::observation::ImuState`] that feeds the observation builder. The
//! values are body-frame FLU (`+x forward`, `+y left`, `+z up`) — the
//! [`crate::imu`] loop already post-multiplies by `mount_quat_sensor_body`
//! and rotates the gyro by the same rotation, so we never apply a frame
//! transform here. That matches what `mdp.imu_ang_vel` and
//! `mdp.imu_projected_gravity` produce in
//! `sim/bebop_training/envs/bebop_v2_base_cfg.py`, so the trained policy
//! sees the same observation pipeline at deploy time as during training.
//!
//! When the IMU is **stale** (no fresh report for `3 × report_period_ms`)
//! or **never received** (no `imu:` block in the YAML, dead BNO, failed
//! SHTP boot), we fall back to synthetic upright-at-rest observations —
//! the same values the simulator presents at the start of every standing
//! episode:
//!
//! - `quaternion = [0, 0, 0, 1]` (XYZW identity) ⇒ `projected_gravity = (0, 0, -1)`,
//! - `angular_velocity = (0, 0, 0)`.
//!
//! That fallback only ever fires when the sensor isn't actually present,
//! which we surface as a `warn!` once per state transition to avoid log
//! spam. Joint positions and velocities still come from real motor
//! feedback, and `base_lin_vel` stays at zero until an estimator is wired
//! in (see [`SYNTHETIC_BASE_LIN_VEL`]).

use anyhow::{anyhow, Context, Result};
use std::panic::AssertUnwindSafe;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

use crate::config::dims;
use crate::imu::ImuShared;
use crate::mode::Mode;
use crate::observation::{
    scale_actions_to_targets, ImuState, ObservationBuilder, VelocityCommand, JOINT_NAMES,
    NUM_JOINTS,
};
use crate::policy::PolicyController;
use crate::safety::{BreachReason, Supervisor};

/// Sentinel observation values for "IMU not present / stale". Mirrors what
/// training presents at episode start in the standing task — see module
/// docs.
const SYNTHETIC_IMU_QUATERNION_XYZW: [f32; 4] = [0.0, 0.0, 0.0, 1.0];
const SYNTHETIC_BASE_LIN_VEL: [f32; 3] = [0.0, 0.0, 0.0];

pub struct PolicyRunner {
    controller: PolicyController,
    obs_builder: ObservationBuilder,
    supervisor: Arc<Supervisor>,
    /// Shared handle on the latest BNO085 reading. Filled by the
    /// [`crate::imu`] thread; consumed here on every tick. Always
    /// present even when no IMU is configured (`imu:` block omitted
    /// from the YAML) — in that case the snapshot stays at its
    /// default and we use synthetic observations.
    imu_shared: ImuShared,
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
    /// Edge-detect on the IMU live/synthetic boundary so we log
    /// transitions once instead of every tick.
    imu_was_live: bool,
    /// Last time we emitted the human-readable I/O summary at `info!`.
    /// Per-tick logs would be 100 lines/s; we rate-limit to ~1 Hz.
    last_io_log_at: Option<Instant>,
}

/// How often to emit the human-readable observation/action summary at
/// `info!`. Every tick is still available at `debug!`.
const IO_LOG_INTERVAL: Duration = Duration::from_secs(1);

impl PolicyRunner {
    /// Load the ONNX policy and resolve the policy-slot ↔ supervisor-joint
    /// mapping by name.
    ///
    /// Errors out if any joint named in [`JOINT_NAMES`] is missing from the
    /// loaded `RobotConfig`. We refuse to silently swap or drop joints —
    /// a misconfigured YAML there would silently break the policy I/O
    /// contract.
    pub fn new<P: AsRef<Path>>(
        supervisor: Arc<Supervisor>,
        imu_shared: ImuShared,
        model_path: P,
    ) -> Result<Self> {
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
            imu_shared,
            joint_indices,
            default_positions,
            was_running: false,
            imu_was_live: false,
            last_io_log_at: None,
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
        //    policy-slot order. Capture each joint's armed state too so we
        //    can skip TX for joints the operator hasn't enabled yet.
        let snapshots = sup.snapshot_motors();
        let mut joint_pos = [0.0_f32; NUM_JOINTS];
        let mut joint_vel = [0.0_f32; NUM_JOINTS];
        let mut armed = [false; NUM_JOINTS];
        for (slot, &idx) in self.joint_indices.iter().enumerate() {
            let s = &snapshots[idx];
            joint_pos[slot] = s.position;
            joint_vel[slot] = s.velocity;
            armed[slot] = s.armed;
        }

        // 2) IMU. Pull from the shared snapshot if it's fresh, else
        //    fall back to synthetic upright-at-rest. The mount
        //    rotation has already been applied by `imu::spawn_imu_thread`
        //    (both to the quaternion and to the gyro vector), so the
        //    values are body-frame FLU and ready to drop straight into
        //    `ImuState`. See the module docs and
        //    `bebop_v2_base_cfg.py::ObservationsCfg` for the matching
        //    sim-side pipeline.
        let now = Instant::now();
        let imu_state = match self.imu_shared.lock() {
            Ok(g) if !g.is_stale(now) => {
                let quaternion = g.quaternion.unwrap_or(SYNTHETIC_IMU_QUATERNION_XYZW);
                let angular_velocity = g.angular_velocity_body.unwrap_or([0.0; 3]);
                if !self.imu_was_live {
                    info!(
                        ?quaternion,
                        ?angular_velocity,
                        "IMU live: switching PolicyRunner from synthetic to BNO085 observations"
                    );
                    self.imu_was_live = true;
                }
                ImuState {
                    quaternion,
                    angular_velocity,
                    linear_acceleration: [0.0; 3],
                }
            }
            _ => {
                if self.imu_was_live {
                    warn!(
                        "IMU stale / unavailable: PolicyRunner falling back to synthetic \
                         upright observations (was using live BNO085)"
                    );
                    self.imu_was_live = false;
                }
                ImuState {
                    quaternion: SYNTHETIC_IMU_QUATERNION_XYZW,
                    angular_velocity: [0.0; 3],
                    linear_acceleration: [0.0; 3],
                }
            }
        };
        self.obs_builder.update_imu(imu_state);

        // Body-frame linear velocity. Bebop V2 has no wheel odometry
        // and no visual-inertial estimator wired in yet, so we still
        // feed zeros here. The trained policy was exposed to wide
        // uniform noise on this channel (see
        // `BebopV2BaseEnvCfg.ObservationsCfg.base_lin_vel`), so a
        // constant zero is well within distribution for the standing
        // task. Wire in an estimator before relying on it for
        // locomotion.
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

        // Per-tick raw dump for offline replay / debugging. Enable with
        // `RUST_LOG=bebop_linux::policy_runner=debug` (or the binary's
        // equivalent prefix).
        debug!(
            observation = ?obs.as_slice(),
            raw_action = ?action.as_slice(),
            position_targets_rad = ?targets.as_slice(),
            "policy tick: 36-dim obs → raw action → scaled joint targets (rad)"
        );

        // Rate-limited human-readable summary at info!. We break the
        // 36-element observation into the same named slices as
        // `ObservationBuilder::build` so it's actually parseable in a
        // running terminal. `imu_live` makes it obvious whether the
        // policy is currently consuming the BNO085 or the synthetic
        // upright fallback.
        let should_log_info = self
            .last_io_log_at
            .is_none_or(|t| now.duration_since(t) >= IO_LOG_INTERVAL);
        if should_log_info {
            self.last_io_log_at = Some(now);
            info!(
                imu_live = self.imu_was_live,
                base_lin_vel = ?&obs[0..3],
                base_ang_vel = ?&obs[3..6],
                projected_gravity = ?&obs[6..9],
                joint_pos_rel = ?&obs[9..17],
                joint_vel = ?&obs[17..25],
                last_action = ?&obs[25..33],
                cmd_vel = ?&obs[33..36],
                raw_action = ?action.as_slice(),
                position_targets_rad = ?targets.as_slice(),
                "policy I/O"
            );
        }

        // 8) Push to motors. Skip joints the operator hasn't armed: a
        //    disabled motor ignores PD commands at the bus level, and
        //    flooding the bus with TX traffic for not-yet-armed joints
        //    starves the *armed* joints' feedback frames (the watchdog
        //    only allows ~100 ms before latching E-STOP, and a sequential
        //    `arm_all` over 8 motors takes ~160 ms; without this filter
        //    every re-arm in RunPolicy mode trips the feedback watchdog
        //    on whichever joint was armed first).
        //
        //    Use per-joint hold_gains for kp/kd; these should ideally
        //    match the gains baked into the training-time actuator
        //    config (see `BEBOP_V2_CFG.actuators` in
        //    `bebop_v2_base_cfg.py`). They currently differ — known
        //    sim-to-real gap.
        for (slot, &idx) in self.joint_indices.iter().enumerate() {
            if !armed[slot] {
                continue;
            }
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
