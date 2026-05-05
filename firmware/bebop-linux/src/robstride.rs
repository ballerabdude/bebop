//! Robstride motor driver (RS01, RS02, RS03, RS04)
//!
//! Implements the Robstride CAN protocol for motor control.
//! Uses CAN 2.0 Extended Frame (29-bit ID) at 1 Mbps.

use crate::can_interface::{CanInterface, ReceivedFrame, RobstrideFeedback};
use crate::config::{JointCommand, JointState, RobstrideModel, RobstrideSpecs};
use anyhow::Result;
use tracing::{debug, trace};

/// Robstride communication types (bits 28-24 of extended frame ID)
mod cmd {
    pub const GET_ID: u8 = 0x00;
    pub const MOTOR_CTRL: u8 = 0x01;    // Operation control mode
    pub const FEEDBACK: u8 = 0x02;       // Motor feedback response
    pub const ENABLE: u8 = 0x03;
    pub const STOP: u8 = 0x04;
    pub const SET_ZERO: u8 = 0x06;
    pub const PARAM_READ: u8 = 0x11;
    pub const PARAM_WRITE: u8 = 0x12;
    pub const FAULT_FEEDBACK: u8 = 0x15;
    pub const ACTIVE_REPORT: u8 = 0x18;
}

/// Robstride parameter indices
mod param {
    pub const RUN_MODE: u16 = 0x7005;
    pub const IQ_REF: u16 = 0x7006;
    pub const SPD_REF: u16 = 0x700A;
    pub const LIMIT_TORQUE: u16 = 0x700B;
    pub const LOC_REF: u16 = 0x7016;
    pub const LIMIT_SPD: u16 = 0x7017;
    pub const LIMIT_CUR: u16 = 0x7018;
    pub const MECH_POS: u16 = 0x7019;
    pub const IQF: u16 = 0x701A;
    pub const MECH_VEL: u16 = 0x701B;
    pub const VBUS: u16 = 0x701C;
}

/// Run modes
mod mode {
    pub const OPERATION: u8 = 0x00;      // MIT-like control
    pub const POSITION_PP: u8 = 0x01;    // Profile Position
    pub const VELOCITY: u8 = 0x02;
    pub const CURRENT: u8 = 0x03;
    pub const POSITION_CSP: u8 = 0x05;   // Cyclic Synchronous Position
}

/// Protocol constants (same for all models)
const P_MIN: f32 = -12.57;  // -4π rad
const P_MAX: f32 = 12.57;   // +4π rad
const KP_MIN: f32 = 0.0;
const KP_MAX: f32 = 5000.0;
const KD_MIN: f32 = 0.0;
const KD_MAX: f32 = 100.0;

/// Host CAN ID (master controller)
const HOST_ID: u8 = 0xFD;

/// Robstride motor driver
#[derive(Debug)]
pub struct RobstrideMotor {
    pub can_id: u8,
    pub model: RobstrideModel,
    specs: RobstrideSpecs,
    pub state: JointState,
    run_mode: u8,
}

impl RobstrideMotor {
    /// Create a new Robstride motor driver
    pub fn new(can_id: u8, model: RobstrideModel) -> Self {
        Self {
            can_id,
            model,
            specs: model.specs(),
            state: JointState::default(),
            run_mode: mode::OPERATION,
        }
    }

    /// Build the 29-bit extended CAN ID
    fn make_can_id(&self, cmd_type: u8, data_area2: u16) -> u32 {
        let id = ((cmd_type as u32) << 24)
            | ((data_area2 as u32) << 8)
            | (self.can_id as u32);
        id
    }

    /// Build CAN ID for operation control mode
    fn make_ctrl_can_id(&self, torque_raw: u16) -> u32 {
        ((cmd::MOTOR_CTRL as u32) << 24)
            | ((torque_raw as u32) << 8)
            | (self.can_id as u32)
    }

    /// Convert float to uint16 with given range
    fn float_to_uint16(value: f32, min: f32, max: f32) -> u16 {
        let clamped = value.clamp(min, max);
        let proportion = (clamped - min) / (max - min);
        (proportion * 65535.0) as u16
    }

    /// Convert uint16 to float with given range
    fn uint16_to_float(value: u16, min: f32, max: f32) -> f32 {
        let proportion = value as f32 / 65535.0;
        min + proportion * (max - min)
    }

    /// Enable the motor
    pub fn enable(&self, can: &CanInterface) -> Result<()> {
        let can_id = self.make_can_id(cmd::ENABLE, HOST_ID as u16);
        let data = [0u8; 8];
        can.send_extended(can_id, &data)?;
        debug!("Enabled Robstride motor {}", self.can_id);
        Ok(())
    }

    /// Disable the motor
    pub fn disable(&self, can: &CanInterface) -> Result<()> {
        let can_id = self.make_can_id(cmd::STOP, HOST_ID as u16);
        let data = [0u8; 8];
        can.send_extended(can_id, &data)?;
        debug!("Disabled Robstride motor {}", self.can_id);
        Ok(())
    }

    /// Set mechanical zero position
    pub fn set_zero(&self, can: &CanInterface) -> Result<()> {
        let can_id = self.make_can_id(cmd::SET_ZERO, HOST_ID as u16);
        let mut data = [0u8; 8];
        data[0] = 1;
        can.send_extended(can_id, &data)?;
        debug!("Set zero position for Robstride motor {}", self.can_id);
        Ok(())
    }

    /// Enable active reporting at specified interval (ms)
    pub fn enable_active_reporting(&self, can: &CanInterface, interval_ms: u8) -> Result<()> {
        let can_id = self.make_can_id(cmd::ACTIVE_REPORT, HOST_ID as u16);
        let mut data = [0u8; 8];
        data[0] = 1;  // Enable
        data[1] = interval_ms;
        can.send_extended(can_id, &data)?;
        debug!(
            "Enabled active reporting for Robstride motor {} at {}ms",
            self.can_id, interval_ms
        );
        Ok(())
    }

    /// Send operation control mode command (MIT-like control)
    ///
    /// τ = Kp × (p_target - p_actual) + Kd × (v_target - v_actual) + τ_ff
    pub fn send_command(&self, can: &CanInterface, cmd: &JointCommand) -> Result<()> {
        // Convert to raw values
        let position_raw = Self::float_to_uint16(cmd.position, P_MIN, P_MAX);
        let velocity_raw = Self::float_to_uint16(
            cmd.velocity,
            self.specs.velocity_min,
            self.specs.velocity_max,
        );
        let kp_raw = Self::float_to_uint16(cmd.kp, KP_MIN, KP_MAX);
        let kd_raw = Self::float_to_uint16(cmd.kd, KD_MIN, KD_MAX);
        let torque_raw = Self::float_to_uint16(
            cmd.torque,
            self.specs.torque_min,
            self.specs.torque_max,
        );

        // Build CAN frame
        let can_id = self.make_ctrl_can_id(torque_raw);
        let mut data = [0u8; 8];

        // Pack data (big-endian)
        data[0..2].copy_from_slice(&position_raw.to_be_bytes());
        data[2..4].copy_from_slice(&velocity_raw.to_be_bytes());
        data[4..6].copy_from_slice(&kp_raw.to_be_bytes());
        data[6..8].copy_from_slice(&kd_raw.to_be_bytes());

        can.send_extended(can_id, &data)?;

        trace!(
            "RS{} CMD: pos={:.3} vel={:.3} kp={:.1} kd={:.2} τ={:.2}",
            self.can_id,
            cmd.position,
            cmd.velocity,
            cmd.kp,
            cmd.kd,
            cmd.torque
        );

        Ok(())
    }

    /// Send position-only command (uses configured Kp/Kd)
    pub fn send_position(&self, can: &CanInterface, position: f32, kp: f32, kd: f32) -> Result<()> {
        let cmd = JointCommand {
            position,
            velocity: 0.0,
            torque: 0.0,
            kp,
            kd,
        };
        self.send_command(can, &cmd)
    }

    /// Process feedback frame
    pub fn process_feedback(&mut self, feedback: &RobstrideFeedback) {
        if feedback.motor_id != self.can_id {
            return;
        }

        self.state.position = feedback.position;
        self.state.velocity = feedback.velocity;
        self.state.torque = feedback.torque;
        self.state.temperature = feedback.temperature;
        self.state.has_error = feedback.fault_bits != 0;
        self.state.error_code = feedback.fault_bits as u32;
        self.state.is_enabled = feedback.mode_status == 0x02;  // Motor mode
        self.state.last_update_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
    }

    /// Check if motor is receiving feedback
    pub fn is_alive(&self, max_age_ms: u64) -> bool {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        
        (now - self.state.last_update_ms) < max_age_ms
    }

    /// Get fault description
    pub fn fault_description(&self) -> Option<String> {
        let bits = self.state.error_code as u8;
        if bits == 0 {
            return None;
        }

        let mut faults = Vec::new();
        if bits & 0x01 != 0 { faults.push("Undervoltage"); }
        if bits & 0x02 != 0 { faults.push("Overcurrent"); }
        if bits & 0x04 != 0 { faults.push("Overtemperature"); }
        if bits & 0x08 != 0 { faults.push("Magnetic encoding fault"); }
        if bits & 0x10 != 0 { faults.push("Gridlock overload"); }
        if bits & 0x20 != 0 { faults.push("Uncalibrated"); }

        Some(faults.join(", "))
    }
}

/// Collection of Robstride motors on a single bus
pub struct RobstrideMotorBus {
    pub motors: Vec<RobstrideMotor>,
}

impl RobstrideMotorBus {
    pub fn new() -> Self {
        Self { motors: Vec::new() }
    }

    pub fn add_motor(&mut self, can_id: u8, model: RobstrideModel) {
        self.motors.push(RobstrideMotor::new(can_id, model));
    }

    pub fn get_motor(&self, can_id: u8) -> Option<&RobstrideMotor> {
        self.motors.iter().find(|m| m.can_id == can_id)
    }

    pub fn get_motor_mut(&mut self, can_id: u8) -> Option<&mut RobstrideMotor> {
        self.motors.iter_mut().find(|m| m.can_id == can_id)
    }

    /// Enable all motors
    pub fn enable_all(&self, can: &CanInterface) -> Result<()> {
        for motor in &self.motors {
            motor.enable(can)?;
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        Ok(())
    }

    /// Disable all motors
    pub fn disable_all(&self, can: &CanInterface) -> Result<()> {
        for motor in &self.motors {
            motor.disable(can)?;
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        Ok(())
    }

    /// Process received frame and update motor state
    pub fn process_frame(&mut self, frame: &ReceivedFrame) {
        if let Some(feedback) = frame.parse_robstride() {
            if let Some(motor) = self.get_motor_mut(feedback.motor_id) {
                motor.process_feedback(&feedback);
            }
        }
    }
}

impl Default for RobstrideMotorBus {
    fn default() -> Self {
        Self::new()
    }
}
