//! Motor safety supervisor.
//!
//! Defense in depth — every motor TX flows through [`Supervisor`], which
//! enforces:
//!
//! 1. **Outgoing clamp.** Commanded position is clamped to the joint's
//!    `pos_min..=pos_max`. Velocity / torque feed-forward are clamped
//!    against `vel_max` / `tau_max`.
//! 2. **Slew limit.** Setpoint can never step by more than
//!    `slew.max_pos_step_per_tick` between consecutive TX cycles.
//! 3. **Incoming check.** Every feedback frame is validated against the
//!    same limits; breach latches E-STOP and disables every motor on every
//!    bus.
//! 4. **Watchdog.** If no feedback frame arrives within
//!    `feedback_timeout_ms`, E-STOP latches.
//! 5. **Drop-disables-everything.** When the supervisor is dropped (process
//!    exit / panic), the destructor sends `Disable` to every motor on every
//!    bus three times before letting the sockets close.

pub mod bus_pool;
pub mod limits;
pub mod power_monitor;
pub mod supervisor;

pub use bus_pool::{read_can_state, BusPool};
pub use limits::{BreachReason, MotorRuntimeState, MotorSnapshot};
pub use power_monitor::{spawn_power_monitor, PowerBoardSnapshot, PowerMonitor};
pub use supervisor::{Supervisor, SupervisorEvent};
