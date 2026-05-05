//! Per-motor runtime state and limit-breach types.

use crate::config::JointConfig;
use crate::robstride::RobstrideMotor;
use std::time::Instant;

/// Reason an E-STOP was latched. Surfaced over the WS API so the operator
/// can see *why* without scraping logs.
#[derive(Debug, Clone)]
pub enum BreachReason {
    Operator(String),
    PositionOutOfRange {
        joint: String,
        value: f32,
        min: f32,
        max: f32,
    },
    VelocityExceeded {
        joint: String,
        value: f32,
        max: f32,
    },
    TorqueExceeded {
        joint: String,
        value: f32,
        max: f32,
    },
    TemperatureExceeded {
        joint: String,
        value: f32,
        max: f32,
    },
    MotorFault {
        joint: String,
        bits: u8,
        description: String,
    },
    FeedbackWatchdog {
        joint: String,
        elapsed_ms: f32,
        timeout_ms: f32,
    },
    BusError {
        can_interface: String,
        message: String,
    },
    ProcessExit,
}

impl BreachReason {
    pub fn human(&self) -> String {
        match self {
            BreachReason::Operator(reason) => format!("operator: {reason}"),
            BreachReason::PositionOutOfRange {
                joint,
                value,
                min,
                max,
            } => format!(
                "{joint}: position {value:+.3} outside [{min:+.3}, {max:+.3}]"
            ),
            BreachReason::VelocityExceeded { joint, value, max } => {
                format!("{joint}: |velocity| {:.2} > {:.2}", value.abs(), max)
            }
            BreachReason::TorqueExceeded { joint, value, max } => {
                format!("{joint}: |torque| {:.2} > {:.2} Nm", value.abs(), max)
            }
            BreachReason::TemperatureExceeded { joint, value, max } => {
                format!("{joint}: temperature {value:.1} > {max:.1} °C")
            }
            BreachReason::MotorFault {
                joint,
                bits,
                description,
            } => format!("{joint}: motor fault 0x{bits:02X} ({description})"),
            BreachReason::FeedbackWatchdog {
                joint,
                elapsed_ms,
                timeout_ms,
            } => format!(
                "{joint}: feedback watchdog: {elapsed_ms:.0} ms since last RX (timeout {timeout_ms:.0} ms)"
            ),
            BreachReason::BusError {
                can_interface,
                message,
            } => format!("CAN bus {can_interface}: {message}"),
            BreachReason::ProcessExit => "process exit".to_string(),
        }
    }
}

/// Runtime state owned by the supervisor for a single motor. One per joint.
///
/// Wraps [`RobstrideMotor`] (for protocol formatting + cached feedback) and
/// adds supervisor-only fields like the slew tracker and watchdog timestamp.
#[derive(Debug)]
pub struct MotorRuntimeState {
    /// Static configuration (cloned from the global config so the supervisor
    /// owns it for the lifetime of the run).
    pub joint_cfg: JointConfig,
    /// Robstride driver: holds protocol formatter + most recently parsed
    /// feedback in `motor.state`.
    pub motor: RobstrideMotor,
    /// `true` once the supervisor has accepted an Enable for this motor.
    pub armed: bool,
    /// Most recent commanded position; used by the slew limiter.
    pub last_target_pos: f32,
    /// Wall-clock timestamp of the last feedback frame parsed for this
    /// motor. `None` until the first frame arrives.
    pub last_rx: Option<Instant>,
}

/// Read-only snapshot of motor state, suitable for telemetry. Cheap to
/// clone; produced by [`MotorRuntimeState::snapshot`].
#[derive(Debug, Clone)]
pub struct MotorSnapshot {
    pub joint_name: String,
    pub can_interface: String,
    pub motor_id: u8,
    pub model: &'static str,
    pub armed: bool,
    pub feedback_stale: bool,
    pub fault_bits: u8,
    pub position: f32,
    pub velocity: f32,
    pub torque: f32,
    pub temperature: f32,
    /// Last commanded position target (post-clamp, post-slew). Meaningful
    /// only while `armed`; reset to the live position on each new arm.
    pub target_position: f32,
    pub pos_min: f32,
    pub pos_max: f32,
    pub vel_max: f32,
    pub tau_max: f32,
    pub temp_max: f32,
}

impl MotorRuntimeState {
    pub fn new(joint_cfg: JointConfig) -> Self {
        let motor = RobstrideMotor::new(joint_cfg.can_id, joint_cfg.model);
        Self {
            joint_cfg,
            motor,
            armed: false,
            last_target_pos: 0.0,
            last_rx: None,
        }
    }

    /// Whether feedback hasn't been received within
    /// `joint_cfg.hard_limits.feedback_timeout_ms`. Returns `false` until
    /// the first frame arrives so the operator's first arm doesn't trip
    /// the watchdog instantly.
    pub fn feedback_stale(&self, now: Instant) -> bool {
        match self.last_rx {
            None => false,
            Some(t) => {
                let elapsed_ms = now.duration_since(t).as_secs_f32() * 1000.0;
                elapsed_ms > self.joint_cfg.hard_limits.feedback_timeout_ms
            }
        }
    }

    pub fn snapshot(&self, now: Instant) -> MotorSnapshot {
        let h = &self.joint_cfg.hard_limits;
        MotorSnapshot {
            joint_name: self.joint_cfg.name.clone(),
            can_interface: self.joint_cfg.can_bus.clone(),
            motor_id: self.joint_cfg.can_id,
            model: self.joint_cfg.model.as_str(),
            armed: self.armed,
            feedback_stale: self.feedback_stale(now),
            fault_bits: self.motor.state.error_code as u8,
            position: self.motor.state.position,
            velocity: self.motor.state.velocity,
            torque: self.motor.state.torque,
            temperature: self.motor.state.temperature,
            target_position: self.last_target_pos,
            pos_min: h.pos_min,
            pos_max: h.pos_max,
            vel_max: h.vel_max,
            tau_max: h.tau_max,
            temp_max: h.temp_max,
        }
    }
}
