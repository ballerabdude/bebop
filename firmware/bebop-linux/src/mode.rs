//! Operating mode state machine.
//!
//! ```text
//! Idle ──set_mode(DialIn)──▶ DialIn ──set_mode(RunPolicy)──▶ RunPolicy
//!  ▲                          │                              │
//!  │                          │                              │
//!  └──────────set_mode(Idle)──┴──────────set_mode(Idle)──────┘
//! ```
//!
//! Leaving DialIn or RunPolicy disarms every motor before the transition.
//! Leaving any mode while E-STOP is latched is forbidden — the operator
//! must `ResetEStop` first, which drops the mode to `Idle`.

use bebop_proto::runtime::v1 as proto;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Mode {
    /// Motors disabled, telemetry streams. Default at boot.
    Idle = 1,
    /// Per-motor enable/disable, slew-limited hold cycle, watchdog. Used
    /// for bench bring-up and safety-limit dial-in.
    DialIn = 2,
    /// ONNX policy drives the joints. The runtime loads `policy.onnx`
    /// (sibling of the joint YAML by default; override with `--policy`)
    /// and runs the 36-dim observation -> 8-dim action MLP at 100 Hz via
    /// [`crate::policy_runner::PolicyRunner`].
    RunPolicy = 3,
}

impl Mode {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            x if x == Mode::Idle as u8 => Some(Mode::Idle),
            x if x == Mode::DialIn as u8 => Some(Mode::DialIn),
            x if x == Mode::RunPolicy as u8 => Some(Mode::RunPolicy),
            _ => None,
        }
    }

    pub fn as_proto(self) -> proto::Mode {
        match self {
            Mode::Idle => proto::Mode::Idle,
            Mode::DialIn => proto::Mode::DialIn,
            Mode::RunPolicy => proto::Mode::RunPolicy,
        }
    }

    pub fn from_proto(m: proto::Mode) -> Option<Self> {
        match m {
            proto::Mode::Idle => Some(Mode::Idle),
            proto::Mode::DialIn => Some(Mode::DialIn),
            proto::Mode::RunPolicy => Some(Mode::RunPolicy),
            proto::Mode::Unspecified => None,
        }
    }
}
