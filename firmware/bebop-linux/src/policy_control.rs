//! Shared operator-controlled flags for the policy runner.
//!
//! The WS server's [`crate::server::handlers`] writes here when the
//! operator app toggles a control flag; the 100 Hz
//! [`crate::policy_runner::PolicyRunner`] reads (and reconciles) on every
//! tick. Keeping this state behind a small [`Arc<Mutex<_>>`] (mirroring
//! [`crate::policy_io::PolicyIoShared`]) means:
//!
//! - the WS task never touches a capture file handle directly (all
//!   file I/O lives on the dedicated MCAP writer thread spawned by
//!   [`crate::policy_capture`]; the runner just forwards the flag);
//! - flag changes are visible to the next tick within ≤10 ms, which is
//!   tight enough that the operator perceives the toggle as immediate;
//! - if the runner thread is wedged, the WS handler still acks the
//!   request (the change just won't be acted on until the runner runs
//!   again — exactly what we want for a debug knob).
//!
//! The flags live separately from [`crate::policy_io::PolicyIoSnapshot`]
//! (which is the read-only telemetry view) so the runner is always the
//! single source of truth for "what actually happened on the last tick"
//! while the WS handler is the single source of truth for "what the
//! operator currently wants".

use std::sync::{Arc, Mutex};

/// Mutable operator-driven knobs read by [`crate::policy_runner::PolicyRunner`]
/// on every tick.
#[derive(Debug, Clone, Default)]
pub struct PolicyControl {
    /// When `true`, RUN_POLICY still infers, publishes telemetry, and
    /// writes capture samples, but the policy's PD command is NOT sent
    /// to the motors. Instead the supervisor's hold-gains keepalive
    /// frame is sent each tick so the robot freezes in its current
    /// posture (and so the Robstride feedback watchdog stays happy —
    /// armed motors only respond to a control frame). Used for bench
    /// observation / noise capture without physically driving the
    /// actuators.
    pub dry_run: bool,
    /// When `true`, the runner asks
    /// [`crate::policy_capture`]'s writer thread to open (if not already)
    /// an MCAP capture file under the firmware's configured
    /// `--capture-dir` and submits one sample per tick while in DIAL_IN
    /// (obs only) or RUN_POLICY (obs + action). When set back to
    /// `false`, the runner asks the writer thread to flush and close
    /// the file on its next iteration.
    pub capture_requested: bool,
    /// Optional operator-supplied label folded into the timestamped
    /// capture filename for later identification (e.g. "noise_floor",
    /// "policy_v17_dry"). Empty when no label was provided. The runner
    /// sanitizes it to `[A-Za-z0-9_-]` before using it in a path.
    pub capture_label: String,
}

/// Shared handle on the operator control flags. Cloned into the WS
/// handler (writer) and the policy runner (reader / file-state owner).
pub type PolicyControlShared = Arc<Mutex<PolicyControl>>;

/// Allocate a fresh shared control block. All flags start `false` /
/// empty so RUN_POLICY drives motors by default and capture is off.
pub fn new_shared() -> PolicyControlShared {
    Arc::new(Mutex::new(PolicyControl::default()))
}
