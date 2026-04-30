//! Robot configuration and joint definitions
//!
//! This module contains all hardware-specific configuration including:
//! - Joint definitions (CAN IDs, limits, gains)
//! - Motor model specifications
//! - Timing parameters


/// Control loop timing configuration
pub mod timing {
    pub const POLICY_RATE_HZ: u64 = 50;  // Main policy rate (Hz)
    pub const POLICY_INTERVAL_MS: u64 = 1000 / POLICY_RATE_HZ;
    pub const FEEDBACK_PUBLISH_HZ: u64 = 100;
    pub const WATCHDOG_TIMEOUT_MS: u64 = 500;
    pub const UDP_PORT: u16 = 10000;
}

/// Observation and action dimensions (must match training!)
pub mod dims {
    pub const OBS_DIM: usize = 30;
    pub const ACTION_DIM: usize = 6;
    pub const HISTORY_STEPS: usize = 1;
    pub const TOTAL_OBS_DIM: usize = OBS_DIM * HISTORY_STEPS;
}

/// Scaling factors (must match training!)
pub mod scales {
    pub const SCALE_LIN_VEL: f32 = 1.0;
    pub const SCALE_ANG_VEL: f32 = 1.0;
    pub const SCALE_DOF_POS: f32 = 1.0;
    pub const SCALE_DOF_VEL: f32 = 1.0;
    pub const SCALE_ACTION_LEGS: f32 = 0.8;
    pub const SCALE_ACTION_WHEELS: f32 = 20.0;

    pub const CLIP_LIN_VEL: f32 = 3.0;
    pub const CLIP_ANG_VEL: f32 = 10.0;
    pub const CLIP_DOF_VEL: f32 = 15.0;
}

/// Safety limits
pub mod limits {
    pub const MAX_LEG_POS_RAD: f32 = 0.8;
    pub const MAX_WHEEL_VEL_RAD_S: f32 = 20.0;
}

/// Robstride motor model specifications
#[derive(Debug, Clone, Copy)]
pub struct RobstrideSpecs {
    pub torque_min: f32,
    pub torque_max: f32,
    pub velocity_min: f32,
    pub velocity_max: f32,
    pub kp_max: f32,
    pub kd_max: f32,
}

impl RobstrideSpecs {
    pub const RS01: Self = Self {
        torque_min: -12.0,
        torque_max: 12.0,
        velocity_min: -45.0,
        velocity_max: 45.0,
        kp_max: 500.0,
        kd_max: 5.0,
    };

    pub const RS02: Self = Self {
        torque_min: -25.0,
        torque_max: 25.0,
        velocity_min: -30.0,
        velocity_max: 30.0,
        kp_max: 500.0,
        kd_max: 5.0,
    };

    pub const RS03: Self = Self {
        torque_min: -60.0,
        torque_max: 60.0,
        velocity_min: -20.0,
        velocity_max: 20.0,
        kp_max: 5000.0,
        kd_max: 100.0,
    };

    pub const RS04: Self = Self {
        torque_min: -120.0,
        torque_max: 120.0,
        velocity_min: -15.0,
        velocity_max: 15.0,
        kp_max: 5000.0,
        kd_max: 100.0,
    };
}

/// Motor type enumeration
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MotorType {
    Robstride(RobstrideModel),
    ODrive,
}

/// Robstride model variants
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RobstrideModel {
    RS01,
    RS02,
    RS03,
    RS04,
}

impl RobstrideModel {
    pub fn specs(&self) -> RobstrideSpecs {
        match self {
            RobstrideModel::RS01 => RobstrideSpecs::RS01,
            RobstrideModel::RS02 => RobstrideSpecs::RS02,
            RobstrideModel::RS03 => RobstrideSpecs::RS03,
            RobstrideModel::RS04 => RobstrideSpecs::RS04,
        }
    }
}

/// Joint configuration
#[derive(Debug, Clone)]
pub struct JointConfig {
    pub name: String,
    pub index: usize,
    pub can_id: u8,
    pub can_bus: String,
    pub motor_type: MotorType,
    pub default_position: f32,
    pub position_min: f32,
    pub position_max: f32,
    pub kp: f32,
    pub kd: f32,
}

/// Complete robot configuration
#[derive(Debug, Clone)]
pub struct RobotConfig {
    pub joints: Vec<JointConfig>,
    pub can_interfaces: Vec<String>,
}

impl Default for RobotConfig {
    fn default() -> Self {
        Self::bebop_wheeled()
    }
}

impl RobotConfig {
    /// Configuration for Bebop wheeled robot (4 legs + 2 wheels)
    pub fn bebop_wheeled() -> Self {
        let can_bus = "can0".to_string();

        let joints = vec![
            // Leg joints (Robstride RS04 - position control)
            JointConfig {
                name: "left_hip_pitch".to_string(),
                index: 0,
                can_id: 31,
                can_bus: can_bus.clone(),
                motor_type: MotorType::Robstride(RobstrideModel::RS04),
                default_position: 0.0,
                position_min: -0.8,
                position_max: 0.8,
                kp: 50.0,
                kd: 2.0,
            },
            JointConfig {
                name: "right_hip_pitch".to_string(),
                index: 1,
                can_id: 41,
                can_bus: can_bus.clone(),
                motor_type: MotorType::Robstride(RobstrideModel::RS04),
                default_position: 0.0,
                position_min: -0.8,
                position_max: 0.8,
                kp: 50.0,
                kd: 2.0,
            },
            JointConfig {
                name: "left_knee_pitch".to_string(),
                index: 2,
                can_id: 34,
                can_bus: can_bus.clone(),
                motor_type: MotorType::Robstride(RobstrideModel::RS04),
                default_position: 0.0,
                position_min: -0.8,
                position_max: 0.8,
                kp: 50.0,
                kd: 2.0,
            },
            JointConfig {
                name: "right_knee_pitch".to_string(),
                index: 3,
                can_id: 44,
                can_bus: can_bus.clone(),
                motor_type: MotorType::Robstride(RobstrideModel::RS04),
                default_position: 0.0,
                position_min: -0.8,
                position_max: 0.8,
                kp: 50.0,
                kd: 2.0,
            },
            // Wheel joints (ODrive - velocity control)
            JointConfig {
                name: "left_wheel".to_string(),
                index: 4,
                can_id: 35,  // ODrive node ID
                can_bus: can_bus.clone(),
                motor_type: MotorType::ODrive,
                default_position: 0.0,
                position_min: f32::NEG_INFINITY,
                position_max: f32::INFINITY,
                kp: 0.0,  // Not used for velocity control
                kd: 0.0,
            },
            JointConfig {
                name: "right_wheel".to_string(),
                index: 5,
                can_id: 45,  // ODrive node ID
                can_bus: can_bus.clone(),
                motor_type: MotorType::ODrive,
                default_position: 0.0,
                position_min: f32::NEG_INFINITY,
                position_max: f32::INFINITY,
                kp: 0.0,
                kd: 0.0,
            },
        ];

        Self {
            joints,
            can_interfaces: vec![can_bus],
        }
    }

    /// Get joint by name
    pub fn get_joint(&self, name: &str) -> Option<&JointConfig> {
        self.joints.iter().find(|j| j.name == name)
    }

    /// Get joint by index
    pub fn get_joint_by_index(&self, index: usize) -> Option<&JointConfig> {
        self.joints.iter().find(|j| j.index == index)
    }

    /// Get all Robstride joints
    pub fn robstride_joints(&self) -> Vec<&JointConfig> {
        self.joints
            .iter()
            .filter(|j| matches!(j.motor_type, MotorType::Robstride(_)))
            .collect()
    }

    /// Get all ODrive joints
    pub fn odrive_joints(&self) -> Vec<&JointConfig> {
        self.joints
            .iter()
            .filter(|j| j.motor_type == MotorType::ODrive)
            .collect()
    }

    /// Number of joints
    pub fn num_joints(&self) -> usize {
        self.joints.len()
    }
}

/// Joint state (feedback from motor)
#[derive(Debug, Clone, Default)]
pub struct JointState {
    pub position: f32,      // radians
    pub velocity: f32,      // rad/s
    pub torque: f32,        // Nm
    pub temperature: f32,   // Celsius
    pub is_enabled: bool,
    pub has_error: bool,
    pub error_code: u32,
    pub last_update_ms: u64,
}

/// Joint command (sent to motor)
#[derive(Debug, Clone, Default)]
pub struct JointCommand {
    pub position: f32,      // radians (for position control)
    pub velocity: f32,      // rad/s (for velocity control)
    pub torque: f32,        // Nm (feedforward torque)
    pub kp: f32,            // Position gain
    pub kd: f32,            // Velocity gain
}
