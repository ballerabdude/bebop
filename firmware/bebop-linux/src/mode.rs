//! Operating mode state machine.
//!
//! ```text
//! Idle ──set_mode(DialIn)──▶ DialIn
//!  ▲                          │
//!  └──────set_mode(Idle)──────┘
//! ```
//!
//! Leaving DialIn for Idle disarms every motor and wheel before the
//! transition. While E-STOP is latched the only allowed transition is to
//! Idle; `ResetEStop` clears the latch without changing the mode, and
//! everything stays disarmed until it is armed again.

use bebop_proto::runtime::v1 as proto;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Mode {
    /// Nothing can be armed; telemetry streams. Default at boot.
    Idle = 1,
    /// The armed mode: wheels and joints may be armed. Armed wheels track
    /// the operator twist, armed joints get the slew-limited hold cycle,
    /// and the feedback watchdog is enforced. Used for driving and for
    /// joint bring-up / safety-limit dial-in.
    DialIn = 2,
}

impl Mode {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            x if x == Mode::Idle as u8 => Some(Mode::Idle),
            x if x == Mode::DialIn as u8 => Some(Mode::DialIn),
            _ => None,
        }
    }

    pub fn as_proto(self) -> proto::Mode {
        match self {
            Mode::Idle => proto::Mode::Idle,
            Mode::DialIn => proto::Mode::DialIn,
        }
    }

    pub fn from_proto(m: proto::Mode) -> Option<Self> {
        match m {
            proto::Mode::Idle => Some(Mode::Idle),
            proto::Mode::DialIn => Some(Mode::DialIn),
            proto::Mode::Unspecified => None,
        }
    }
}
