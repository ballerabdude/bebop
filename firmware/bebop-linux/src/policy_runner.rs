//! 100 Hz policy inference loop for [`crate::mode::Mode::RunPolicy`].
//!
//! Owns:
//!
//! - A [`PolicyController`] (ONNX session + last-action cache + history
//!   buffer; with `HISTORY_STEPS = 1` the buffer is just the latest frame).
//! - An [`ObservationBuilder`] that holds IMU / cmd_vel / joint state and
//!   emits the 52-element observation in the layout fixed by
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
//! feedback.
//!
//! ## IMU liveness
//!
//! Rotation vector and calibrated gyro arrive on the same SH-2 data
//! channel at the same cadence, so a single freshness clock
//! (`ImuSnapshot::is_stale`, surfaced as `imu_live`) covers both. When
//! it's true we use the live BNO reading for both attitude and
//! angular velocity; when false the policy falls back to synthetic
//! upright (quaternion identity + zero angular velocity).

use anyhow::{anyhow, Context, Result};
use std::panic::AssertUnwindSafe;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use tracing::{debug, info, warn};

use crate::policy_capture::{CaptureHandle, TickSample};

use crate::config::{dims, PolicyGainClamps};
use crate::imu::ImuShared;
use crate::mode::Mode;
use crate::observation::{
    decode_policy_action, DecodedAction, GainEma, ImuState, ObservationBuilder, VelocityCommand,
    JOINT_NAMES, NUM_JOINTS,
};
use crate::policy::PolicyController;
use crate::policy_capture::{self};
use crate::policy_control::PolicyControlShared;
use crate::policy_io::PolicyIoShared;
use crate::safety::{BreachReason, Supervisor};

/// Sentinel observation values for "IMU not present / stale". Mirrors what
/// training presents at episode start in the standing task — see module
/// docs.
const SYNTHETIC_IMU_QUATERNION_XYZW: [f32; 4] = [0.0, 0.0, 0.0, 1.0];

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
    /// Latest observation/action vectors for the WS telemetry pump.
    policy_io: PolicyIoShared,
    /// `joint_indices[slot]` = index into `Supervisor::cfg().joints` of
    /// the joint occupying policy slot `slot` (0..8 in [`JOINT_NAMES`] order).
    joint_indices: [usize; NUM_JOINTS],
    /// Default joint positions in policy slot order. Used both as the
    /// `obs_builder.default_positions` (for `joint_pos_rel`) and as the
    /// offset in `target = default + scale * action`.
    default_positions: [f32; NUM_JOINTS],
    /// Per-joint policy kp/kd clamps in policy slot order. Cached at
    /// startup from `JointConfig::policy_gain_clamps` so we don't have
    /// to re-resolve the joint mapping on every tick.
    policy_gain_clamps: [PolicyGainClamps; NUM_JOINTS],
    /// Low-pass filter on the decoded kp/kd (the firmware mirror of the
    /// sim action term's `gain_ema_tau_s`). Re-seeded at the midpoint
    /// gains on every RunPolicy entry; applied after
    /// [`decode_policy_action`] every tick so the gains sent to the
    /// motors can never snap tick-to-tick.
    gain_ema: GainEma,
    /// Edge-detect entering RunPolicy so we can clear the policy's history
    /// buffer + last_action cache.
    was_running: bool,
    /// Edge-detect on the IMU live/synthetic boundary so we log
    /// transitions once instead of every tick.
    imu_was_live: bool,
    /// Operator-driven dry-run flag. The runner reads this every tick;
    /// the WS server (`handlers.rs`) is the writer.
    policy_control: PolicyControlShared,
    /// Handle on the MCAP writer thread. The runner calls
    /// `request_open` / `request_close` to manage the capture file's
    /// lifecycle and `send_sample` once per tick while capture is
    /// active. The writer thread owns the file descriptor; we never
    /// touch disk on this thread.
    capture: CaptureHandle,
    /// Mirrors the writer-thread's "currently has a file open" state so
    /// the runner can detect when its requested open/close has actually
    /// taken effect (without locking `policy_io` from inside the tick
    /// for read-modify-write loops).
    capture_was_active: bool,
    /// Monotonic per-capture sample counter. Reset to 0 each time a new
    /// capture file opens (detected via the `capture_was_active` edge in
    /// `reconcile_capture`) and written into
    /// [`TickSample::tick`]. Independent of the MCAP `sequence`
    /// header so readers still get a usable tick axis.
    capture_tick: u64,
    /// Wall/monotonic origin of the current capture file, set lazily on
    /// the first sample after a file opens. Drives
    /// `TickSample::sim_time_s` so a single file plots against
    /// a self-contained, 0-based time axis. `None` between captures.
    capture_started_at: Option<Instant>,
    /// Last time we asked the writer thread to open a capture file
    /// while one wasn't yet active. Acts as a back-off on retry: if the
    /// writer can't actually open the file (e.g. read-only filesystem,
    /// missing permissions), the runner would otherwise re-issue the
    /// request on every 10 ms tick, swamping the writer channel and
    /// the journal. With the back-off we retry at most every
    /// [`CAPTURE_OPEN_RETRY`] until the writer eventually publishes
    /// `capture_active=true`. Cleared as soon as a file is actually
    /// open so the next legitimate close→open transition is immediate.
    last_open_request_at: Option<Instant>,
    /// Edge-detect on the dry-run flag so we log transitions once.
    dry_run_was_on: bool,
}

/// Back-off between successive [`crate::policy_capture::CaptureHandle::request_open`]
/// calls when the writer thread isn't (yet) reporting an active file.
/// Short enough that a brief transient (e.g. SD card just (re-)mounted)
/// recovers within a couple of seconds; long enough that a persistent
/// failure (read-only fs, permissions) doesn't produce 100 retries per
/// second. Chosen to roughly match the WS auto-reconnect cadence on
/// the operator app so the two stay visually in sync.
const CAPTURE_OPEN_RETRY: std::time::Duration = std::time::Duration::from_secs(5);

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
        policy_io: PolicyIoShared,
        policy_control: PolicyControlShared,
        capture: CaptureHandle,
        model_path: P,
    ) -> Result<Self> {
        let model_path = model_path.as_ref();
        let cfg = supervisor.cfg();

        let mut joint_indices = [0usize; NUM_JOINTS];
        let mut default_positions = [0.0_f32; NUM_JOINTS];
        let mut policy_gain_clamps = [PolicyGainClamps::FALLBACK; NUM_JOINTS];
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
            policy_gain_clamps[slot] = joint.policy_gain_clamps;
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

        let gain_ema = GainEma::new(cfg.policy.gain_ema_tau_s, &policy_gain_clamps);
        info!(
            gain_ema_tau_s = cfg.policy.gain_ema_tau_s,
            "gain low-pass filter configured (must mirror sim gain_ema_tau_s)"
        );

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
            policy_io,
            joint_indices,
            default_positions,
            policy_gain_clamps,
            gain_ema,
            was_running: false,
            imu_was_live: false,
            policy_control,
            capture,
            capture_was_active: false,
            capture_tick: 0,
            capture_started_at: None,
            last_open_request_at: None,
            dry_run_was_on: false,
        })
    }

    /// Run one inference + TX cycle. The flow is now:
    ///
    /// 1. Reconcile capture state: capture is always-on whenever the
    ///    runtime is in `DialIn` or `RunPolicy` (and not E-STOPped). The
    ///    writer thread opens a fresh MCAP file on entry and closes it
    ///    on exit; size-based rotation happens transparently inside the
    ///    writer (see [`crate::policy_capture`]). `Idle` stays quiet so
    ///    long-running idle sessions don't fill the disk with zeros.
    /// 2. If we're in `DialIn` (with capture active), build the observation
    ///    from real feedback and submit an obs-only sample to the writer
    ///    thread — no inference, no motor TX.
    /// 3. If we're in `RunPolicy` (and not E-STOPped), run the existing
    ///    inference path; in `dry_run` mode skip the per-joint
    ///    `safe_send_ctrl` (everything else still runs).
    /// 4. Otherwise no-op, modulo the on-exit reset that re-enters
    ///    RunPolicy with a clean controller.
    pub fn tick(&mut self) {
        let sup = self.supervisor.clone();
        let mode = sup.mode();
        let estop = sup.estop_active();

        // --- 0. Reconcile operator control flags + capture lifecycle ----
        // Capture is now always-on in the "active" modes (DialIn obs-only,
        // RunPolicy obs + action). Idle stays quiet because joints are
        // disabled then and the observation would be a long run of zeros
        // that swamps the file. E-STOP also closes the file: there's no
        // useful telemetry to record when the supervisor has latched.
        let dry_run = match self.policy_control.lock() {
            Ok(g) => g.dry_run,
            Err(_) => false,
        };
        let should_capture = !estop && (mode == Mode::DialIn || mode == Mode::RunPolicy);
        self.reconcile_capture(should_capture);
        self.publish_dry_run(dry_run);

        if dry_run != self.dry_run_was_on {
            if dry_run {
                info!(
                    "policy DRY-RUN enabled: RunPolicy will still infer + publish + capture, \
                     but instead of applying the policy action it sends a hold-gains \
                     keepalive (kp/kd from hold_gains, target = last_target_pos) so the \
                     robot freezes in its current posture and the feedback watchdog stays \
                     happy"
                );
            } else {
                info!("policy DRY-RUN disabled: RunPolicy will resume sending PD commands");
            }
            self.dry_run_was_on = dry_run;
        }

        let in_run_policy = mode == Mode::RunPolicy && !estop;

        if !in_run_policy {
            if self.was_running {
                self.controller.reset();
                self.obs_builder
                    .update_last_action(&[0.0_f32; dims::ACTION_DIM]);
                if let Ok(mut g) = self.policy_io.lock() {
                    g.clear_tick();
                }
                debug!("policy controller reset on RunPolicy exit");
            }
            self.was_running = false;

            // DialIn obs-only capture: build the observation from real
            // feedback (without running inference) and submit a sample
            // to the writer thread. We deliberately gate on
            // `mode == DialIn` (rather than "anything not RunPolicy")
            // so Idle captures stay quiet — joints are disabled in Idle
            // and the obs would be a long run of zeros, swamping the file.
            if mode == Mode::DialIn && !estop && self.capture_was_active {
                self.capture_dial_in_observation(&sup, dry_run);
            }
            return;
        }

        if !self.was_running {
            // Entering RunPolicy: clear any stale history / last_action so
            // the first observation matches a fresh-episode condition.
            self.controller.reset();
            self.obs_builder
                .update_last_action(&[0.0_f32; dims::ACTION_DIM]);
            // Re-seed the gain low-pass at the midpoint gains (what the
            // robot physically holds at arm) so engagement starts
            // transient-free — mirrors the sim action term's reset().
            self.gain_ema.reset(&self.policy_gain_clamps);
            info!("RunPolicy entered; policy controller reset");
            self.was_running = true;
        }

        // 1) Pull real joint feedback from the supervisor and lay it out in
        //    policy-slot order. Capture each joint's armed state too so we
        //    can skip TX for joints the operator hasn't enabled yet.
        let snapshots = sup.snapshot_motors();
        let mut joint_pos = [0.0_f32; NUM_JOINTS];
        let mut joint_vel = [0.0_f32; NUM_JOINTS];
        let mut joint_trq = [0.0_f32; NUM_JOINTS];
        let mut armed = [false; NUM_JOINTS];
        for (slot, &idx) in self.joint_indices.iter().enumerate() {
            let s = &snapshots[idx];
            joint_pos[slot] = s.position;
            joint_vel[slot] = s.velocity;
            joint_trq[slot] = s.torque;
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
        // Rotation vector and gyro arrive on the same SH-2 data channel
        // at the same cadence, so `ImuSnapshot::is_stale` is the single
        // liveness signal for both. When the IMU is fresh we use the
        // live quaternion + angular velocity; otherwise we fall back to
        // synthetic upright.
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

        // 3) Velocity command. Isaac-BebopV2-Flat-v0 forces (0, 0, 0)
        //    during training; locomotion checkpoints will want a real
        //    UDP / WS feed plumbed in here.
        self.obs_builder.update_cmd_vel(VelocityCommand::default());

        // 4) Joint state.
        self.obs_builder.joint_positions = joint_pos;
        self.obs_builder.joint_velocities = joint_vel;

        // 5) Build the 52-dim observation, run inference.
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

        // 7) Decode the 24-dim MIT-mode action into (8 position targets,
        //    8 kp, 8 kd). The decoder clips raw channels to [-1, 1],
        //    applies the position scale + default offset, and affine-maps
        //    each per-joint kp/kd to its `policy_gain_clamps` range. The
        //    decoded kp/kd then pass through the gain low-pass
        //    (`gain_ema_tau_s`, mirrors the sim action term) so the gains
        //    sent to the motors can never snap tick-to-tick. The
        //    supervisor's `safe_send_ctrl` then additionally clamps
        //    position to per-joint `pos_min..pos_max` and slew-limits
        //    per tick before pushing the MIT-mode CAN frame.
        let mut decoded = decode_policy_action(
            &action,
            &self.default_positions,
            &self.policy_gain_clamps,
        );
        self.gain_ema.apply(&mut decoded);

        // Per-tick policy I/O is now captured to MCAP (see
        // `crate::policy_capture`); no console log here. Open the latest
        // file under the capture dir for Foxglove playback / plotting.

        if let Ok(mut g) = self.policy_io.lock() {
            g.publish_tick(
                self.imu_was_live,
                dry_run,
                &obs,
                &action,
                &decoded.targets,
                &decoded.kp,
                &decoded.kd,
            );
        }

        // 7b) Capture (RunPolicy: obs + raw action + decoded action). We
        // build the proto sample here and hand it off to the writer
        // thread via a try_send. The runner is back on the hot loop in
        // a few µs regardless of how slow the disk is — backlog is
        // surfaced as `capture_dropped` in the next telemetry frame.
        if self.capture_was_active {
            // Hoist the Copy field out so it isn't read while `self` is
            // mutably borrowed as the `build_capture_sample` receiver.
            let imu_was_live = self.imu_was_live;
            let sample = self.build_capture_sample(
                now,
                Mode::RunPolicy,
                dry_run,
                imu_was_live,
                &joint_pos,
                &joint_vel,
                &joint_trq,
                &armed,
                &obs,
                Some((&action, &decoded)),
            );
            self.capture.send_sample(sample);
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
        //    kp/kd come from the policy on every tick (MIT-mode variable
        //    impedance). The decode path has already clamped them to the
        //    per-joint `policy_gain_clamps` envelope, so no further
        //    clipping is needed here. `hold_gains` in the YAML is used
        //    for the pre-RunPolicy idle hold AND for the dry-run
        //    keepalive below.
        //
        //    Dry-run path: we still produced + published + captured the
        //    policy action above, but we deliberately don't apply it.
        //    Robstride MIT-mode motors only emit feedback in response to
        //    a control frame, so silently skipping TX would starve the
        //    supervisor's 100 ms feedback watchdog and immediately latch
        //    E-STOP on the first armed joint (this was the failure mode
        //    operators were hitting on the bench). Instead, send a
        //    hold-gains keepalive at the joint's last target position so
        //    the robot freezes in whatever posture it had when dry-run
        //    began. The keepalive is identical in shape to DialIn's
        //    hold cycle.
        if dry_run {
            sup.tick_hold_armed();
            return;
        }
        for (slot, &idx) in self.joint_indices.iter().enumerate() {
            if !armed[slot] {
                continue;
            }
            let cfg = &sup.cfg().joints[idx];
            let kp = decoded.kp[slot];
            let kd = decoded.kd[slot];
            if let Err(e) = sup.safe_send_ctrl(idx, decoded.targets[slot], kp, kd, 0.0, 0.0) {
                debug!(joint = %cfg.name, error = %e, "policy TX failed");
            }
        }
    }

    /// Edge-detect the always-on capture predicate (DialIn/RunPolicy +
    /// !estop) and ask the writer thread to open / close as needed. The
    /// writer thread is the source of truth for "file actually open" —
    /// we track its progress via the shared
    /// `PolicyIoSnapshot.capture_active` field that the writer publishes
    /// itself, mirrored here into `self.capture_was_active` so the tick
    /// body has a cheap local check.
    ///
    /// Note: the writer may also rotate the active file underneath us
    /// when it crosses the size cap. From the runner's perspective that
    /// looks like `capture_active` staying `true` across ticks — the
    /// path / row counter just change in the published snapshot.
    fn reconcile_capture(&mut self, should_capture: bool) {
        let active_on_disk = self
            .policy_io
            .lock()
            .map(|g| g.capture_active)
            .unwrap_or(false);
        // Fresh file just opened: restart the per-capture tick counter and
        // clear the time origin so `tick` / `sim_time_s` are 0-based within
        // each session. The origin is set lazily on the first sample.
        // Mid-session rotation keeps `capture_was_active` true, so this
        // edge only fires on a real open (mode entry), not on each
        // rotated file.
        if active_on_disk && !self.capture_was_active {
            self.capture_tick = 0;
            self.capture_started_at = None;
            // Writer accepted the file; drop the open-retry throttle so
            // the next close→open transition can re-arm cleanly.
            self.last_open_request_at = None;
        }
        self.capture_was_active = active_on_disk;

        match (active_on_disk, should_capture) {
            (true, true) | (false, false) => {
                // Steady state — nothing for the runner to do.
            }
            (false, true) => {
                // Want capture, but the writer doesn't have a file open
                // yet. Send a single open request and then wait up to
                // CAPTURE_OPEN_RETRY before trying again. Without this
                // throttle a persistent open failure (read-only fs,
                // perms) makes this branch fire every 10 ms tick and
                // produces a flood of identical errors in the journal
                // (one per writer attempt) plus needless channel
                // pressure.
                let now = Instant::now();
                let due = self
                    .last_open_request_at
                    .is_none_or(|t| now.duration_since(t) >= CAPTURE_OPEN_RETRY);
                if due {
                    self.capture.request_open();
                    self.last_open_request_at = Some(now);
                }
            }
            (true, false) => {
                self.capture.request_close();
                // Cancel any pending throttle so a subsequent re-entry
                // can immediately issue a fresh open request.
                self.last_open_request_at = None;
            }
        }
    }

    /// Push the latest dry-run flag into the shared snapshot. Capture
    /// state (`active` / `path` / `rows` / `dropped`) is owned by the
    /// writer thread, which publishes directly into `PolicyIoShared`.
    fn publish_dry_run(&self, dry_run: bool) {
        if let Ok(mut g) = self.policy_io.lock() {
            g.set_dry_run(dry_run);
        }
    }

    /// DialIn capture path: gather real joint + IMU feedback, build the
    /// observation, submit a sample to the writer thread. Does NOT run
    /// the ONNX session and does NOT send any motor commands. The
    /// action-related proto fields stay empty (length-0 `repeated`s)
    /// so the schema is identical to RunPolicy samples.
    fn capture_dial_in_observation(&mut self, sup: &Arc<Supervisor>, dry_run: bool) {
        let snapshots = sup.snapshot_motors();
        let mut joint_pos = [0.0_f32; NUM_JOINTS];
        let mut joint_vel = [0.0_f32; NUM_JOINTS];
        let mut joint_trq = [0.0_f32; NUM_JOINTS];
        let mut armed = [false; NUM_JOINTS];
        for (slot, &idx) in self.joint_indices.iter().enumerate() {
            let s = &snapshots[idx];
            joint_pos[slot] = s.position;
            joint_vel[slot] = s.velocity;
            joint_trq[slot] = s.torque;
            armed[slot] = s.armed;
        }

        // IMU — same pull as RunPolicy. We don't repeat the live/synthetic
        // edge logging here: that's a RunPolicy diagnostic and would
        // otherwise double-fire when the operator hops between modes
        // mid-capture.
        let now = Instant::now();
        let mut imu_live = false;
        let imu_state = match self.imu_shared.lock() {
            Ok(g) if !g.is_stale(now) => {
                imu_live = true;
                let quaternion = g.quaternion.unwrap_or(SYNTHETIC_IMU_QUATERNION_XYZW);
                let angular_velocity = g.angular_velocity_body.unwrap_or([0.0; 3]);
                ImuState {
                    quaternion,
                    angular_velocity,
                    linear_acceleration: [0.0; 3],
                }
            }
            _ => ImuState {
                quaternion: SYNTHETIC_IMU_QUATERNION_XYZW,
                angular_velocity: [0.0; 3],
                linear_acceleration: [0.0; 3],
            },
        };
        self.obs_builder.update_imu(imu_state);
        // DialIn has no command source today (no policy = no cmd_vel
        // teleop); train-time defaults to (0, 0, 0) so we do the same.
        self.obs_builder.update_cmd_vel(VelocityCommand::default());
        self.obs_builder.joint_positions = joint_pos;
        self.obs_builder.joint_velocities = joint_vel;
        // `last_action` stays at its previous value (cleared to zeros on
        // every RunPolicy exit by `clear_tick` -> `update_last_action`),
        // so DialIn rows always have last_action = 0 — matching the start
        // of a training episode.

        let obs = self.obs_builder.build();
        let sample = self.build_capture_sample(
            now,
            Mode::DialIn,
            dry_run,
            imu_live,
            &joint_pos,
            &joint_vel,
            &joint_trq,
            &armed,
            &obs,
            None,
        );
        self.capture.send_sample(sample);
    }

    /// Build a [`TickSample`] from the current tick's state.
    /// Hot path — keep it allocation-light (a few small `Vec`s for the
    /// `repeated` fields, no formatting).
    #[allow(clippy::too_many_arguments)]
    fn build_capture_sample(
        &mut self,
        now: Instant,
        mode: Mode,
        dry_run: bool,
        imu_live: bool,
        joint_pos: &[f32; NUM_JOINTS],
        joint_vel: &[f32; NUM_JOINTS],
        joint_trq: &[f32; NUM_JOINTS],
        armed: &[bool; NUM_JOINTS],
        observation: &[f32],
        action: Option<(&[f32], &DecodedAction)>,
    ) -> TickSample {
        let origin = *self.capture_started_at.get_or_insert(now);
        let (wall_time_ns, sim_time_s) = policy_capture::timestamps(now, Some(origin));

        let tick = self.capture_tick;
        self.capture_tick += 1;

        let mode_str = match mode {
            Mode::Idle => "IDLE",
            Mode::DialIn => "DIAL_IN",
            Mode::RunPolicy => "RUN_POLICY",
        }
        .to_string();

        let quat = self.obs_builder.imu.quaternion;
        let ang_vel = self.obs_builder.imu.angular_velocity;

        let (raw_action, position_targets_rad, kp, kd) = match action {
            Some((raw, decoded)) => (
                raw.to_vec(),
                decoded.targets.to_vec(),
                decoded.kp.to_vec(),
                decoded.kd.to_vec(),
            ),
            None => (Vec::new(), Vec::new(), Vec::new(), Vec::new()),
        };

        TickSample {
            tick,
            wall_time_ns,
            sim_time_s,
            mode: mode_str,
            dry_run,
            imu_live,
            quaternion: quat,
            angular_velocity: ang_vel,
            joint_pos_rad: joint_pos.to_vec(),
            joint_vel_rad_s: joint_vel.to_vec(),
            joint_torque_nm: joint_trq.to_vec(),
            joint_armed: armed.to_vec(),
            observation: observation.to_vec(),
            raw_action,
            position_targets_rad,
            kp,
            kd,
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
        // MIT-mode action: 3 channels (pos, kp, kd) per joint.
        assert_eq!(3 * JOINT_NAMES.len(), dims::ACTION_DIM);
    }
}

// Round-trip tests for the MCAP capture writer live next to the writer
// itself (see `crate::policy_capture::tests`). Header / schema drift is
// guarded there because that module owns both the open path and the
// schema-name constants.
