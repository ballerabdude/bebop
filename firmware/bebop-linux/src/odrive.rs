//! ODrive S1 "CANSimple" driver for velocity-controlled wheels.
//!
//! Drives ODrive BotWheels (ODrive S1 single-axis controllers) over
//! SocketCAN for differential-drive locomotion. This is the wheel analog
//! of [`crate::robstride`]: that module speaks MIT-mode position/velocity
//! control to Robstride RS0x actuators, whereas a wheel is a continuous-
//! rotation, velocity-controlled actuator with no meaningful position
//! envelope.
//!
//! # Protocol (firmware 0.6.x "simple" CAN)
//!
//! ODrive uses **standard 11-bit** CAN IDs (in contrast to Robstride's
//! 29-bit extended IDs) with the arbitration ID split as:
//!
//! ```text
//!     id = (node_id << 5) | cmd_id
//! ```
//!
//! * `node_id` — bits 5..=10 (0..=63; `0x3F` = broadcast/unaddressed).
//! * `cmd_id`  — bits 0..=4 (0..=31; the CANSimple message ID).
//!
//! All multi-byte payload fields are **little-endian**. `Set_Input_Vel` /
//! `Get_Encoder_Estimates` carry IEEE-754 `float32` in units of **turns**
//! (`rev`) and **turns/second** (`rev/s`).
//!
//! This is the "classic" ODrive CANSimple layout (verified against live S1
//! traffic: node 46's encoder is `(46 << 5) | 0x09 == 0x5C9`, node 36's is
//! `0x489`, node 36's heartbeat `0x481`). Some documentation diagrams show
//! `(cmd_id << 6) | node_id`; do NOT use that — the firmware gates cyclic
//! messages and command routing on the `node_id << 5` split.
//!
//! # Velocity-control startup
//!
//! 1. (once) assign a node id via `Address` (`0x06`) from the broadcast
//!    id — only needed if the S1 wasn't pre-configured over USB.
//! 2. `Set_Controller_Mode` (`0x0B`): `VELOCITY_CONTROL` (2) +
//!    `PASSTHROUGH` (1).
//! 3. `Set_Axis_State` (`0x07`): request `CLOSED_LOOP_CONTROL` (8).
//! 4. stream `Set_Input_Vel` (`0x0D`) with `Input_Vel` in `rev/s`.
//!
//! Feedback arrives cyclically: `Heartbeat` (`0x01`, default 100 ms) and
//! `Get_Encoder_Estimates` (`0x09`, default 10 ms) — no explicit request
//! needed once the axis is addressed.

use crate::can_interface::{CanInterface, ODriveEncoderFeedback, ODriveHeartbeat};
use crate::config::JointState;
use anyhow::Result;

/// CANSimple command IDs (bits 6..=10 of the 11-bit arbitration ID).
pub mod cmd {
    /// Cyclic heartbeat: axis error + state, 100 ms default.
    pub const HEARTBEAT: u8 = 0x01;
    /// Assign / enumerate a node id (a.k.a. `Set_Axis_Node_ID`).
    pub const ADDRESS: u8 = 0x06;
    /// Request an axis state (IDLE / CLOSED_LOOP_CONTROL / …).
    pub const SET_AXIS_STATE: u8 = 0x07;
    /// Cyclic encoder position/velocity feedback (turns, turns/s).
    pub const GET_ENCODER_ESTIMATES: u8 = 0x09;
    /// Select control mode (velocity/torque/…) + input mode.
    pub const SET_CONTROLLER_MODE: u8 = 0x0B;
    /// Position setpoint (+ vel/torque FF).
    pub const SET_INPUT_POS: u8 = 0x0C;
    /// Velocity setpoint (+ torque FF). The wheel command.
    pub const SET_INPUT_VEL: u8 = 0x0D;
    /// Clear latched axis errors.
    pub const CLEAR_ERRORS: u8 = 0x18;
}

/// Axis state enum values carried in the Heartbeat message.
pub mod axis_state {
    pub const IDLE: u32 = 1;
    pub const FULL_CALIBRATION_SEQUENCE: u32 = 3;
    pub const CLOSED_LOOP_CONTROL: u32 = 8;
}

/// Controller `control_mode` values (from `Set_Controller_Mode`).
pub mod control {
    pub const VOLTAGE: u32 = 0;
    pub const TORQUE: u32 = 1;
    pub const VELOCITY: u32 = 2;
    pub const POSITION: u32 = 3;
}

/// Controller `input_mode` values.
pub mod input {
    pub const PASSTHROUGH: u32 = 1;
}

/// ODrive `AxisError` bit definitions. The heartbeat's byte 0-3 field is
/// `<axis>.active_errors | <axis>.disarm_reason`, decoded as a single
/// u32. In current firmware (0.6.x, S1) this is the `ODrive.Error` enum —
/// a flat set of flags replacing the older split Axis/Motor/Controller
/// errors. These definitions come from the API reference
/// (`com.odriverobotics.ODrive.Error`).
/// Used by [`ODriveWheel::fault_description`] to decode the heartbeat's
/// `axis_error` word. Recognised bits are named; unrecognised bits fall
/// through to a hex dump appended as `UNKNOWN`.
pub mod axis_error {
    pub const INITIALIZING: u32 = 1 << 0;
    pub const SYSTEM_LEVEL: u32 = 1 << 1;
    pub const TIMING_ERROR: u32 = 1 << 2;
    pub const MISSING_ESTIMATE: u32 = 1 << 3;
    pub const BAD_CONFIG: u32 = 1 << 4;
    pub const DRV_FAULT: u32 = 1 << 5;
    pub const MISSING_INPUT: u32 = 1 << 6;
    pub const DC_BUS_OVER_VOLTAGE: u32 = 1 << 8;
    pub const DC_BUS_UNDER_VOLTAGE: u32 = 1 << 9;
    pub const DC_BUS_OVER_CURRENT: u32 = 1 << 10;
    pub const DC_BUS_OVER_REGEN_CURRENT: u32 = 1 << 11;
    pub const CURRENT_LIMIT_VIOLATION: u32 = 1 << 12;
    pub const MOTOR_OVER_TEMP: u32 = 1 << 13;
    pub const INVERTER_OVER_TEMP: u32 = 1 << 14;
    pub const VELOCITY_LIMIT_VIOLATION: u32 = 1 << 15;
    pub const POSITION_LIMIT_VIOLATION: u32 = 1 << 16;
    pub const REQUESTED_CURRENT_TOO_HIGH: u32 = 1 << 17;
    pub const WATCHDOG_TIMER_EXPIRED: u32 = 1 << 24;
    pub const ESTOP_REQUESTED: u32 = 1 << 25;
    pub const SPINOUT_DETECTED: u32 = 1 << 26;
    pub const BRAKE_RESISTOR_DISARMED: u32 = 1 << 27;
    pub const THERMISTOR_DISCONNECTED: u32 = 1 << 28;
    pub const CALIBRATION_ERROR: u32 = 1 << 30;
}

/// Broadcast / unaddressed node id.
pub const BROADCAST_NODE_ID: u8 = 0x3F;

/// 2π — converts ODrive turns/s (`rev/s`) into radians/s.
pub const TWO_PI: f32 = 2.0 * std::f32::consts::PI;

/// Compose an ODrive standard CAN arbitration ID from a command id and a
/// node id. The ODrive CANSimple protocol packs `node_id` in bits 5..=10
/// and `cmd_id` in bits 0..=4 — i.e. `(node_id << 5) | cmd_id`. `node_id`
/// is masked to 6 bits, `cmd_id` to 5 bits.
#[inline]
pub fn make_can_id(cmd_id: u8, node_id: u8) -> u16 {
    (((node_id & 0x3F) as u16) << 5) | ((cmd_id & 0x1F) as u16)
}

/// A single ODrive axis (one BotWheel). Holds the protocol node id and the
/// latest decoded feedback in `state`. `axis_state` mirrors the raw
/// [`axis_state`] enum value from the last heartbeat so telemetry can
/// distinguish "armed but axis still IDLE" (the silent-enable-fail mode
/// of an uncalibrated encoder) from a healthy CLOSED_LOOP axis.
#[derive(Debug)]
pub struct ODriveWheel {
    pub node_id: u8,
    pub state: JointState,
    /// Raw ODrive AxisState from the heartbeat; 0 = UNSPECIFIED,
    /// 1 = IDLE, 8 = CLOSED_LOOP_CONTROL, 3..=11 = calibration states.
    pub axis_state: u8,
}

impl ODriveWheel {
    pub fn new(node_id: u8) -> Self {
        Self {
            node_id,
            state: JointState::default(),
            axis_state: 0,
        }
    }

    /// Send a standard-frame command to this axis.
    fn send_standard(&self, can: &CanInterface, cmd_id: u8, data: &[u8]) -> Result<()> {
        can.send_standard(make_can_id(cmd_id, self.node_id), data)
    }

    /// Assign `new_node_id` to the axis whose 48-bit `serial_number`
    /// matches. `serial_number = 0` is a wildcard that always matches, so
    /// the same helper re-addresses an already-addressed axis (send
    /// addressed to its current id) or a factory-fresh one (send to
    /// [`BROADCAST_NODE_ID`]).
    ///
    /// Payload: `[0] = node_id`, `[1..=6] = serial_number` (le48),
    /// `[7] = connection_id` (0).
    pub fn assign_node(
        &self,
        can: &CanInterface,
        new_node_id: u8,
        serial_number: u64,
    ) -> Result<()> {
        let mut data = [0u8; 8];
        data[0] = new_node_id;
        let serial = serial_number & 0x0000_FFFF_FFFF_FFFF; // 48-bit
        data[1..7].copy_from_slice(&serial.to_le_bytes()[..6]);
        can.send_standard(make_can_id(cmd::ADDRESS, BROADCAST_NODE_ID), &data)
    }

    /// Request an axis state (`IDLE`, `CLOSED_LOOP_CONTROL`, …).
    pub fn request_axis_state(&self, can: &CanInterface, state: u32) -> Result<()> {
        self.send_standard(can, cmd::SET_AXIS_STATE, &state.to_le_bytes())
    }

    /// Select `control_mode` / `input_mode`.
    pub fn set_controller_mode(&self, can: &CanInterface, control: u32, input: u32) -> Result<()> {
        let mut data = [0u8; 8];
        data[0..4].copy_from_slice(&control.to_le_bytes());
        data[4..8].copy_from_slice(&input.to_le_bytes());
        self.send_standard(can, cmd::SET_CONTROLLER_MODE, &data)
    }

    /// Convenience: velocity control, passthrough input (the wheel mode).
    pub fn set_velocity_mode(&self, can: &CanInterface) -> Result<()> {
        self.set_controller_mode(can, control::VELOCITY, input::PASSTHROUGH)
    }

    /// Enter closed-loop control (enable the axis).
    pub fn enable(&self, can: &CanInterface) -> Result<()> {
        self.request_axis_state(can, axis_state::CLOSED_LOOP_CONTROL)
    }

    /// Return the axis to IDLE (disable the axis, motor coast/free).
    pub fn disable(&self, can: &CanInterface) -> Result<()> {
        self.request_axis_state(can, axis_state::IDLE)
    }

    /// Request FULL_CALIBRATION_SEQUENCE (motor + encoder offset). The
    /// axis spins for ~20-30 s and must be supervisor-side disarmed.
    /// Used to recover a wheel whose encoder calibration was lost
    /// (symptom: `pos_estimate = NaN` on CAN, arm ack succeeds, axis
    /// stays IDLE and ignores velocity commands). CAN has no
    /// "save configuration" message, so the result dies on power cycle.
    pub fn calibrate(&self, can: &CanInterface) -> Result<()> {
        self.request_axis_state(can, axis_state::FULL_CALIBRATION_SEQUENCE)
    }

    /// Clear latched axis errors (no payload).
    pub fn clear_errors(&self, can: &CanInterface) -> Result<()> {
        self.send_standard(can, cmd::CLEAR_ERRORS, &[0u8; 8])
    }

    /// Command a wheel velocity in `turns/s` with an optional torque
    /// feed-forward in Nm. `Set_Input_Vel` payload is two little-endian
    /// `float32`: `Input_Vel` (rev/s) then `Input_Torque_FF` (Nm).
    pub fn send_velocity(
        &self,
        can: &CanInterface,
        turns_per_s: f32,
        torque_ff_nm: f32,
    ) -> Result<()> {
        let mut data = [0u8; 8];
        data[0..4].copy_from_slice(&turns_per_s.to_le_bytes());
        data[4..8].copy_from_slice(&torque_ff_nm.to_le_bytes());
        self.send_standard(can, cmd::SET_INPUT_VEL, &data)
    }

    /// Decode an encoder-estimate frame into `state.position` / `velocity`.
    ///
    /// The parser already converted `rev` / `rev/s` into `rad` / `rad/s`
    /// (see [`crate::can_interface::parse_odrive_encoder`]), so this just
    /// stamps the fields and refreshes the feedback clock. Wheel angle is
    /// therefore cumulative in radians (unbounded — wheels spin freely).
    pub fn process_encoder(&mut self, fb: &ODriveEncoderFeedback) {
        if fb.node_id != self.node_id {
            return;
        }
        self.state.position = fb.position;
        self.state.velocity = fb.velocity;
        self.state.last_update_ms = Self::now_ms();
    }

    /// Decode a heartbeat frame into `state.is_enabled` / `has_error` /
    /// `error_code` / `axis_state`, and refresh the feedback clock
    /// (heartbeat is part of the watchdog liveness signal too).
    pub fn process_heartbeat(&mut self, hb: &ODriveHeartbeat) {
        if hb.node_id != self.node_id {
            return;
        }
        self.axis_state = hb.axis_state;
        self.state.is_enabled = hb.axis_state == axis_state::CLOSED_LOOP_CONTROL as u8;
        self.state.has_error = hb.axis_error != 0;
        self.state.error_code = hb.axis_error;
        self.state.last_update_ms = Self::now_ms();
    }

    /// Whether feedback (encoder or heartbeat) arrived within `max_age_ms`.
    pub fn is_alive(&self, max_age_ms: u64) -> bool {
        let now = Self::now_ms();
        now.saturating_sub(self.state.last_update_ms) < max_age_ms
    }

    /// Human-readable axis-error description. Decodes the ODrive
    /// `ODrive.Error` bit field from the heartbeat (`.active_errors |
    /// .disarm_reason`); recognised bits are named, any remaining bits
    /// are surfaced as `UNKNOWN` so nothing is silently dropped.
    pub fn fault_description(&self) -> Option<String> {
        let e = self.state.error_code;
        if e == 0 {
            return None;
        }
        const KNOWN_BITS: &[(u32, &str)] = &[
            (axis_error::INITIALIZING, "INITIALIZING"),
            (axis_error::SYSTEM_LEVEL, "SYSTEM_LEVEL"),
            (axis_error::TIMING_ERROR, "TIMING_ERROR"),
            (axis_error::MISSING_ESTIMATE, "MISSING_ESTIMATE"),
            (axis_error::BAD_CONFIG, "BAD_CONFIG"),
            (axis_error::DRV_FAULT, "DRV_FAULT"),
            (axis_error::MISSING_INPUT, "MISSING_INPUT"),
            (axis_error::DC_BUS_OVER_VOLTAGE, "DC_BUS_OVER_VOLTAGE"),
            (axis_error::DC_BUS_UNDER_VOLTAGE, "DC_BUS_UNDER_VOLTAGE"),
            (axis_error::DC_BUS_OVER_CURRENT, "DC_BUS_OVER_CURRENT"),
            (axis_error::DC_BUS_OVER_REGEN_CURRENT, "DC_BUS_OVER_REGEN_CURRENT"),
            (axis_error::CURRENT_LIMIT_VIOLATION, "CURRENT_LIMIT_VIOLATION"),
            (axis_error::MOTOR_OVER_TEMP, "MOTOR_OVER_TEMP"),
            (axis_error::INVERTER_OVER_TEMP, "INVERTER_OVER_TEMP"),
            (axis_error::VELOCITY_LIMIT_VIOLATION, "VELOCITY_LIMIT_VIOLATION"),
            (axis_error::POSITION_LIMIT_VIOLATION, "POSITION_LIMIT_VIOLATION"),
            (
                axis_error::REQUESTED_CURRENT_TOO_HIGH,
                "REQUESTED_CURRENT_TOO_HIGH",
            ),
            (axis_error::WATCHDOG_TIMER_EXPIRED, "WATCHDOG_TIMER_EXPIRED"),
            (axis_error::ESTOP_REQUESTED, "ESTOP_REQUESTED"),
            (axis_error::SPINOUT_DETECTED, "SPINOUT_DETECTED"),
            (axis_error::BRAKE_RESISTOR_DISARMED, "BRAKE_RESISTOR_DISARMED"),
            (axis_error::THERMISTOR_DISCONNECTED, "THERMISTOR_DISCONNECTED"),
            (axis_error::CALIBRATION_ERROR, "CALIBRATION_ERROR"),
        ];
        let mut names: Vec<&str> = KNOWN_BITS
            .iter()
            .filter(|(bit, _)| e & bit != 0)
            .map(|(_, n)| *n)
            .collect();
        let known_mask: u32 = KNOWN_BITS.iter().map(|(b, _)| *b).fold(0, |a, b| a | b);
        if e & !known_mask != 0 {
            names.push("UNKNOWN");
        }
        if names.is_empty() {
            Some(format!("axis_error 0x{e:08X}"))
        } else {
            Some(format!("{} (0x{e:08X})", names.join(", ")))
        }
    }

    fn now_ms() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0)
    }
}

impl Default for ODriveWheel {
    fn default() -> Self {
        Self::new(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn can_id_encodes_node_in_high_bits_cmd_in_low_five() {
        // Golden vectors from live S1 traffic (node 46/36 on the robot).
        assert_eq!(make_can_id(cmd::HEARTBEAT, 36), 0x481);
        assert_eq!(make_can_id(cmd::GET_ENCODER_ESTIMATES, 36), 0x489);
        assert_eq!(make_can_id(cmd::GET_ENCODER_ESTIMATES, 46), 0x5C9);
        // Broadcast address (node 0x3F) for ID 0x7E6.
        assert_eq!(make_can_id(cmd::ADDRESS, BROADCAST_NODE_ID), 0x7E6);
        // node_id masked to 6 bits, cmd_id to 5 bits.
        assert_eq!(make_can_id(0x21, 0x40), make_can_id(0x01, 0x00));
    }

    #[test]
    fn can_id_low_11_bits() {
        // The composed id must always fit a standard 11-bit frame.
        for cmd_id in 0..=31u8 {
            for node_id in 0..=63u8 {
                let id = make_can_id(cmd_id, node_id);
                assert!(id < 0x800, "id {id:#x} exceeds 11 bits");
            }
        }
    }

    #[test]
    fn heartbeat_sets_enabled_and_error() {
        let mut wheel = ODriveWheel::new(2);
        wheel.process_heartbeat(&ODriveHeartbeat {
            node_id: 2,
            axis_error: 0,
            axis_state: axis_state::CLOSED_LOOP_CONTROL as u8,
            procedure_result: 0,
            trajectory_done: false,
        });
        assert!(wheel.state.is_enabled);
        assert!(!wheel.state.has_error);

        wheel.process_heartbeat(&ODriveHeartbeat {
            node_id: 2,
            axis_error: axis_error::ESTOP_REQUESTED,
            axis_state: axis_state::IDLE as u8,
            procedure_result: 0,
            trajectory_done: false,
        });
        assert!(!wheel.state.is_enabled);
        assert!(wheel.state.has_error);
        assert_eq!(wheel.state.error_code, axis_error::ESTOP_REQUESTED);
        assert!(wheel
            .fault_description()
            .unwrap()
            .contains("ESTOP_REQUESTED"));
    }

    #[test]
    fn heartbeat_ignores_other_nodes() {
        let mut wheel = ODriveWheel::new(3);
        wheel.process_heartbeat(&ODriveHeartbeat {
            node_id: 7,
            axis_error: 0,
            axis_state: axis_state::CLOSED_LOOP_CONTROL as u8,
            procedure_result: 0,
            trajectory_done: false,
        });
        assert!(!wheel.state.is_enabled);
    }

    #[test]
    fn encoder_updates_position_and_velocity() {
        let mut wheel = ODriveWheel::new(1);
        wheel.process_encoder(&ODriveEncoderFeedback {
            node_id: 1,
            position: 1.5,
            velocity: -0.25,
        });
        assert_eq!(wheel.state.position, 1.5);
        assert_eq!(wheel.state.velocity, -0.25);
    }

    #[test]
    fn fault_description_none_when_healthy() {
        let wheel = ODriveWheel::new(1);
        assert!(wheel.fault_description().is_none());
    }

    #[test]
    fn fault_description_decodes_spinout_detected() {
        // 0x04000000 = bit 26 = SPINOUT_DETECTED (firmware 0.6.x;
        // previously mis-decoded as "MOTOR_DISARMED_ABS_POSITION").
        let mut wheel = ODriveWheel::new(1);
        wheel.state.error_code = axis_error::SPINOUT_DETECTED;
        let desc = wheel.fault_description().unwrap();
        assert!(
            desc.contains("SPINOUT_DETECTED"),
            "expected SPINOUT_DETECTED in {desc}"
        );
        assert!(desc.contains("0x04000000"));
    }

    #[test]
    fn fault_description_decodes_thermistor_disconnected() {
        // 0x10000000 = bit 28 = THERMISTOR_DISCONNECTED (persistent system
        // error — can't be cleared via clear_errors() while the hardware/config
        // issue remains).
        let mut wheel = ODriveWheel::new(1);
        wheel.state.error_code = axis_error::THERMISTOR_DISCONNECTED;
        let desc = wheel.fault_description().unwrap();
        assert!(
            desc.contains("THERMISTOR_DISCONNECTED"),
            "expected THERMISTOR_DISCONNECTED in {desc}"
        );
        assert!(desc.contains("0x10000000"));
    }

    #[test]
    fn fault_description_decodes_multiple_bits() {
        let mut wheel = ODriveWheel::new(1);
        wheel.state.error_code = axis_error::MISSING_ESTIMATE | axis_error::DRV_FAULT;
        let desc = wheel.fault_description().unwrap();
        assert!(desc.contains("MISSING_ESTIMATE"));
        assert!(desc.contains("DRV_FAULT"));
    }
}
