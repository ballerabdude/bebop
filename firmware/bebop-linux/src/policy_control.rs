//! Shared operator-controlled flags for the policy runner.
//!
//! The WS server's [`crate::server::handlers`] writes here when the
//! operator app toggles a control flag; the 100 Hz
//! [`crate::policy_runner::PolicyRunner`] reads on every tick. Keeping
//! this state behind a small [`Arc<Mutex<_>>`] (mirroring
//! [`crate::policy_io::PolicyIoShared`]) means:
//!
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
//!
//! MCAP capture used to live here as an operator-requested toggle; it
//! is now always-on whenever the runtime is in DIAL_IN or RUN_POLICY
//! (the writer auto-rotates the file on size — see
//! [`crate::policy_capture`]), so the only remaining operator knob is
//! the dry-run flag.

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
}

/// Shared handle on the operator control flags. Cloned into the WS
/// handler (writer) and the policy runner (reader / file-state owner).
pub type PolicyControlShared = Arc<Mutex<PolicyControl>>;

/// Allocate a fresh shared control block. `dry_run` starts `false` so
/// RUN_POLICY drives motors by default.
pub fn new_shared() -> PolicyControlShared {
    Arc::new(Mutex::new(PolicyControl::default()))
}
