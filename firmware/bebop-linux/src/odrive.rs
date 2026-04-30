//! ODrive motor driver (S1)
//!
//! Implements the ODrive CANSimple protocol.
//! Uses CAN 2.0 Standard Frame (11-bit ID).

use crate::can_interface::{CanInterface, ODriveEncoderFeedback, ODriveHeartbeat, ReceivedFrame};
use crate::config::JointState;
use anyhow::Result;
use tracing::{debug, trace};

/// ODrive CAN command IDs
mod cmd {
    pub const HEARTBEAT: u8 = 0x01;
    pub const ESTOP: u8 = 0x02;
    pub const GET_ERROR: u8 = 0x03;
    pub const SET_AXIS_STATE: u8 = 0x07;
    pub const GET_ENCODER_EST: u8 = 0x09;
    pub const SET_CONTROLLER_MODE: u8 = 0x0B;
    pub const SET_INPUT_POS: u8 = 0x0C;
    pub const SET_INPUT_VEL: u8 = 0x0D;
    pub const SET_INPUT_TORQUE: u8 = 0x0E;
    pub const SET_LIMITS: u8 = 0x0F;
    pub const GET_IQ: u8 = 0x14;
    pub const GET_TEMPERATURE: u8 = 0x15;
    pub const CLEAR_ERRORS: u8 = 0x18;
}

/// Axis states
mod axis_state {
    pub const UNDEFINED: u8 = 0;
    pub const IDLE: u8 = 1;
    pub const STARTUP_SEQUENCE: u8 = 2;
    pub const FULL_CALIBRATION: u8 = 3;
    pub const CLOSED_LOOP_CONTROL: u8 = 8;
}

/// Control modes
mod control_mode {
    pub const VOLTAGE: u8 = 0;
    pub const TORQUE: u8 = 1;
    pub const VELOCITY: u8 = 2;
    pub const POSITION: u8 = 3;
}

/// Input modes
mod input_mode {
    pub const INACTIVE: u8 = 0;
    pub const PASSTHROUGH: u8 = 1;
    pub const VEL_RAMP: u8 = 2;
    pub const POS_FILTER: u8 = 3;
    pub const TRAP_TRAJ: u8 = 5;
    pub const TORQUE_RAMP: u8 = 6;
}

/// Unit conversion constants
const RAD_TO_REV: f32 = 1.0 / (2.0 * std::f32::consts::PI);
const REV_TO_RAD: f32 = 2.0 * std::f32::consts::PI;

/// ODrive motor driver
pub struct ODriveMotor {
    pub node_id: u8,
    pub state: JointState,
    pub axis_state: u8,
    pub axis_error: u32,
    pub control_mode: u8,
    pub input_mode: u8,
    torque_constant: f32,
}

impl ODriveMotor {
    /// Create a new ODrive motor driver
    pub fn new(node_id: u8) -> Self {
        Self {
            node_id,
            state: JointState::default(),
            axis_state: axis_state::IDLE,
            axis_error: 0,
            control_mode: control_mode::VELOCITY,
            input_mode: input_mode::PASSTHROUGH,
            torque_constant: 0.083,  // Default for M8325s 100KV
        }
    }

    /// Set torque constant (Nm/A)
    pub fn set_torque_constant(&mut self, kt: f32) {
        self.torque_constant = kt;
    }

    /// Build 11-bit CAN ID
    fn make_can_id(&self, cmd_id: u8) -> u16 {
        ((self.node_id as u16) << 5) | (cmd_id as u16)
    }

    /// Set axis state (e.g., enable closed-loop control)
    pub fn set_axis_state(&self, can: &CanInterface, state: u8) -> Result<()> {
        let can_id = self.make_can_id(cmd::SET_AXIS_STATE);
        let mut data = [0u8; 8];
        data[0..4].copy_from_slice(&(state as u32).to_le_bytes());
        can.send_standard(can_id, &data)?;
        debug!("ODrive {} set axis state to {}", self.node_id, state);
        Ok(())
    }

    /// Enable motor (enter closed-loop control)
    pub fn enable(&self, can: &CanInterface) -> Result<()> {
        self.set_axis_state(can, axis_state::CLOSED_LOOP_CONTROL)
    }

    /// Disable motor (enter idle)
    pub fn disable(&self, can: &CanInterface) -> Result<()> {
        self.set_axis_state(can, axis_state::IDLE)
    }

    /// Emergency stop
    pub fn estop(&self, can: &CanInterface) -> Result<()> {
        let can_id = self.make_can_id(cmd::ESTOP);
        let data = [0u8; 8];
        can.send_standard(can_id, &data)?;
        debug!("ODrive {} E-STOP", self.node_id);
        Ok(())
    }

    /// Clear errors
    pub fn clear_errors(&self, can: &CanInterface) -> Result<()> {
        let can_id = self.make_can_id(cmd::CLEAR_ERRORS);
        let data = [0u8; 8];
        can.send_standard(can_id, &data)?;
        debug!("ODrive {} clear errors", self.node_id);
        Ok(())
    }

    /// Set controller mode
    pub fn set_controller_mode(
        &mut self,
        can: &CanInterface,
        control_mode: u8,
        input_mode: u8,
    ) -> Result<()> {
        let can_id = self.make_can_id(cmd::SET_CONTROLLER_MODE);
        let mut data = [0u8; 8];
        data[0..4].copy_from_slice(&(control_mode as u32).to_le_bytes());
        data[4..8].copy_from_slice(&(input_mode as u32).to_le_bytes());
        can.send_standard(can_id, &data)?;

        self.control_mode = control_mode;
        self.input_mode = input_mode;

        debug!(
            "ODrive {} set control mode {} input mode {}",
            self.node_id, control_mode, input_mode
        );
        Ok(())
    }

    /// Set velocity and current limits
    pub fn set_limits(&self, can: &CanInterface, vel_limit: f32, current_limit: f32) -> Result<()> {
        let can_id = self.make_can_id(cmd::SET_LIMITS);
        let mut data = [0u8; 8];
        data[0..4].copy_from_slice(&vel_limit.to_le_bytes());
        data[4..8].copy_from_slice(&current_limit.to_le_bytes());
        can.send_standard(can_id, &data)?;
        debug!(
            "ODrive {} set limits: vel={} rev/s current={}A",
            self.node_id, vel_limit, current_limit
        );
        Ok(())
    }

    /// Set input velocity (rad/s)
    pub fn set_velocity(&self, can: &CanInterface, velocity_rad_s: f32, torque_ff: f32) -> Result<()> {
        let velocity_rev_s = velocity_rad_s * RAD_TO_REV;
        
        let can_id = self.make_can_id(cmd::SET_INPUT_VEL);
        let mut data = [0u8; 8];
        data[0..4].copy_from_slice(&velocity_rev_s.to_le_bytes());
        data[4..8].copy_from_slice(&torque_ff.to_le_bytes());
        can.send_standard(can_id, &data)?;

        trace!(
            "OD{} VEL: {:.2} rad/s ({:.3} rev/s) τff={:.2}",
            self.node_id,
            velocity_rad_s,
            velocity_rev_s,
            torque_ff
        );
        Ok(())
    }

    /// Set input position (rad)
    pub fn set_position(
        &self,
        can: &CanInterface,
        position_rad: f32,
        velocity_ff: f32,
        torque_ff: f32,
    ) -> Result<()> {
        let position_rev = position_rad * RAD_TO_REV;
        let velocity_ff_rev = velocity_ff * RAD_TO_REV;

        let can_id = self.make_can_id(cmd::SET_INPUT_POS);
        let mut data = [0u8; 8];
        data[0..4].copy_from_slice(&position_rev.to_le_bytes());

        // Velocity and torque feedforward are packed as int16
        let vel_ff_int = (velocity_ff_rev * 1000.0) as i16;
        let torque_ff_int = (torque_ff * 1000.0) as i16;
        data[4..6].copy_from_slice(&vel_ff_int.to_le_bytes());
        data[6..8].copy_from_slice(&torque_ff_int.to_le_bytes());

        can.send_standard(can_id, &data)?;

        trace!(
            "OD{} POS: {:.3} rad ({:.4} rev)",
            self.node_id,
            position_rad,
            position_rev
        );
        Ok(())
    }

    /// Set input torque (Nm)
    pub fn set_torque(&self, can: &CanInterface, torque: f32) -> Result<()> {
        let can_id = self.make_can_id(cmd::SET_INPUT_TORQUE);
        let mut data = [0u8; 8];
        data[0..4].copy_from_slice(&torque.to_le_bytes());
        can.send_standard(can_id, &data)?;

        trace!("OD{} TORQUE: {:.2} Nm", self.node_id, torque);
        Ok(())
    }

    /// Process encoder feedback
    pub fn process_encoder_feedback(&mut self, feedback: &ODriveEncoderFeedback) {
        if feedback.node_id != self.node_id {
            return;
        }

        self.state.position = feedback.position;
        self.state.velocity = feedback.velocity;
        self.state.last_update_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
    }

    /// Process heartbeat
    pub fn process_heartbeat(&mut self, heartbeat: &ODriveHeartbeat) {
        if heartbeat.node_id != self.node_id {
            return;
        }

        self.axis_error = heartbeat.axis_error;
        self.axis_state = heartbeat.axis_state;
        self.state.has_error = heartbeat.axis_error != 0;
        self.state.error_code = heartbeat.axis_error;
        self.state.is_enabled = heartbeat.axis_state == axis_state::CLOSED_LOOP_CONTROL;
    }

    /// Check if motor is receiving feedback
    pub fn is_alive(&self, max_age_ms: u64) -> bool {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        (now - self.state.last_update_ms) < max_age_ms
    }

    /// Check if motor is in closed-loop control
    pub fn is_enabled(&self) -> bool {
        self.axis_state == axis_state::CLOSED_LOOP_CONTROL
    }
}

/// Collection of ODrive motors
pub struct ODriveMotorBus {
    pub motors: Vec<ODriveMotor>,
}

impl ODriveMotorBus {
    pub fn new() -> Self {
        Self { motors: Vec::new() }
    }

    pub fn add_motor(&mut self, node_id: u8) {
        self.motors.push(ODriveMotor::new(node_id));
    }

    pub fn get_motor(&self, node_id: u8) -> Option<&ODriveMotor> {
        self.motors.iter().find(|m| m.node_id == node_id)
    }

    pub fn get_motor_mut(&mut self, node_id: u8) -> Option<&mut ODriveMotor> {
        self.motors.iter_mut().find(|m| m.node_id == node_id)
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
        // Try to parse as encoder feedback
        if let Some(feedback) = frame.parse_odrive_encoder() {
            if let Some(motor) = self.get_motor_mut(feedback.node_id) {
                motor.process_encoder_feedback(&feedback);
            }
        }

        // Try to parse as heartbeat
        if let Some(heartbeat) = frame.parse_odrive_heartbeat() {
            if let Some(motor) = self.get_motor_mut(heartbeat.node_id) {
                motor.process_heartbeat(&heartbeat);
            }
        }
    }
}

impl Default for ODriveMotorBus {
    fn default() -> Self {
        Self::new()
    }
}

