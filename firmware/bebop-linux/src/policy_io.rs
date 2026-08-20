//! Shared snapshot of the latest policy observation + action vectors.
//!
//! The [`crate::policy_runner::PolicyRunner`] writes this on every 100 Hz
//! tick while RunPolicy is active; the WS telemetry builder reads it at
//! ~30 Hz and packs it into [`bebop_proto::runtime::v1::PolicyIoStats`].

use std::sync::{Arc, Mutex};

use crate::observation::{JOINT_NAMES, NUM_JOINTS};

/// Latest policy I/O from the inference loop.
#[derive(Debug, Clone, Default)]
pub struct PolicyIoSnapshot {
    /// True when `policy.onnx` loaded at boot.
    pub present: bool,
    /// True while RunPolicy is active and E-STOP is not latched.
    pub active: bool,
    /// Whether observations use live BNO085 readings vs synthetic fallback.
    /// Driven by the rotation-vector freshness clock; rotation vector and
    /// gyro arrive on the same data channel at the same cadence, so a
    /// fresh rotation vector implies a fresh gyro too.
    pub imu_live: bool,
    /// Mirrors [`crate::policy_control::PolicyControl::dry_run`]: when
    /// `true`, RUN_POLICY's PD output is suppressed and a hold-gains
    /// keepalive is sent instead (robot stays put, telemetry / capture
    /// still run). See the field doc on
    /// [`crate::policy_control::PolicyControl`] for the rationale.
    pub dry_run: bool,
    /// True while the writer thread currently has an MCAP capture file
    /// open. Distinct from
    /// [`crate::policy_control::PolicyControl::capture_requested`]:
    /// `capture_active` only flips to `true` once the file has actually
    /// been opened, and back to `false` once it's been flushed + closed.
    pub capture_active: bool,
    /// Absolute filesystem path of the active capture file on the robot,
    /// or empty when no capture is open. Useful so the operator UI can
    /// tell the user which file to scp off the Jetson later.
    pub capture_path: String,
    /// Number of samples appended to the current capture file. Resets
    /// to 0 each time a new capture is opened.
    pub capture_rows: u64,
    /// Cumulative number of samples the tick thread tried to submit but
    /// the writer-thread channel was full for. Monotonic across the
    /// process lifetime — the operator UI compares deltas to flag bursts
    /// of data loss (e.g. SD-card stall). `0` means the capture has been
    /// keeping up.
    pub capture_dropped: u64,
    /// Full 49-dim observation vector fed to ONNX.
    pub observation: Vec<f32>,
    /// Full 24-dim raw NN output.
    pub raw_action: Vec<f32>,
    /// Decoded position targets (rad), policy slot order.
    pub position_targets_rad: [f32; NUM_JOINTS],
    /// Decoded kp gains, policy slot order.
    pub kp: [f32; NUM_JOINTS],
    /// Decoded kd gains, policy slot order.
    pub kd: [f32; NUM_JOINTS],
}

pub type PolicyIoShared = Arc<Mutex<PolicyIoSnapshot>>;

/// Allocate a fresh shared snapshot. Call [`PolicyIoSnapshot::set_present`]
/// after a successful policy load.
pub fn new_shared() -> PolicyIoShared {
    Arc::new(Mutex::new(PolicyIoSnapshot::default()))
}

impl PolicyIoSnapshot {
    /// Mark whether a policy was loaded at boot.
    pub fn set_present(&mut self, present: bool) {
        self.present = present;
        if !present {
            self.clear_tick();
        }
    }

    /// Clear per-tick fields when leaving RunPolicy or on policy unload.
    /// Capture-state fields are deliberately *not* cleared here: a capture
    /// can outlive a single RunPolicy session (DialIn captures obs-only,
    /// and the operator may toggle modes mid-capture). The runner is the
    /// sole writer of those fields via [`PolicyIoSnapshot::set_capture_state`].
    pub fn clear_tick(&mut self) {
        self.active = false;
        self.imu_live = false;
        self.observation.clear();
        self.raw_action.clear();
        self.position_targets_rad = [0.0; NUM_JOINTS];
        self.kp = [0.0; NUM_JOINTS];
        self.kd = [0.0; NUM_JOINTS];
    }

    /// Publish one inference cycle to the shared snapshot.
    pub fn publish_tick(
        &mut self,
        imu_live: bool,
        dry_run: bool,
        observation: &[f32],
        raw_action: &[f32],
        decoded: &crate::observation::DecodedAction,
    ) {
        self.active = true;
        self.imu_live = imu_live;
        self.dry_run = dry_run;
        self.observation.clear();
        self.observation.extend_from_slice(observation);
        self.raw_action.clear();
        self.raw_action.extend_from_slice(raw_action);
        self.position_targets_rad = decoded.targets;
        self.kp = decoded.kp;
        self.kd = decoded.kd;
    }

    /// Update capture-state fields without touching the per-tick I/O
    /// (so the operator sees a stable "Recording N rows -> path"
    /// indicator regardless of mode). Called by the MCAP writer thread
    /// itself on every open / close transition and periodically while a
    /// file is open, so `rows` / `dropped` reflect what's actually on
    /// disk + the channel's pressure rather than what the tick thread
    /// *tried* to send.
    pub fn set_capture_state(&mut self, active: bool, path: &str, rows: u64, dropped: u64) {
        self.capture_active = active;
        self.capture_path.clear();
        self.capture_path.push_str(path);
        self.capture_rows = rows;
        self.capture_dropped = dropped;
    }

    /// Update the dry-run flag without touching per-tick I/O. Used so
    /// the operator pill stays in sync even when the policy isn't
    /// ticking (e.g. while in IDLE).
    pub fn set_dry_run(&mut self, dry_run: bool) {
        self.dry_run = dry_run;
    }
}

/// Policy slot joint names for UI labels.
pub fn joint_names() -> &'static [&'static str; NUM_JOINTS] {
    &JOINT_NAMES
}
