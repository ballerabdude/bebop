//! Observation building for policy inference
//!
//! This module constructs the observation vector from robot state.
//! The observation format must match exactly what was used during training.

use crate::config::{dims, limits, scales, JointState};
use nalgebra::{Quaternion, UnitQuaternion, Vector3};

/// IMU state
#[derive(Debug, Clone, Default)]
pub struct ImuState {
    pub quaternion: [f32; 4],          // [w, x, y, z]
    pub angular_velocity: [f32; 3],    // [x, y, z] rad/s
    pub linear_acceleration: [f32; 3], // [x, y, z] m/s²
}

impl ImuState {
    /// Compute projected gravity vector in body frame
    pub fn projected_gravity(&self) -> [f32; 3] {
        let quat = UnitQuaternion::from_quaternion(Quaternion::new(
            self.quaternion[0], // w
            self.quaternion[1], // x
            self.quaternion[2], // y
            self.quaternion[3], // z
        ));

        // Gravity in world frame (pointing down)
        let gravity_world = Vector3::new(0.0, 0.0, -1.0);

        // Transform to body frame
        let gravity_body = quat.inverse() * gravity_world;

        [gravity_body.x, gravity_body.y, gravity_body.z]
    }
}

/// Velocity command (from UDP or ROS)
#[derive(Debug, Clone, Default)]
pub struct VelocityCommand {
    pub linear_x: f32,  // m/s (forward)
    pub linear_y: f32,  // m/s (lateral)
    pub angular_z: f32, // rad/s (yaw)
}

/// Estimated base velocity
#[derive(Debug, Clone, Default)]
pub struct BaseVelocity {
    pub linear: [f32; 3], // [x, y, z] m/s in body frame
}

impl BaseVelocity {
    /// Update velocity estimate from wheel odometry and IMU
    pub fn update(
        &mut self,
        wheel_velocities: &[f32], // [left, right] rad/s
        imu: &ImuState,
        dt: f32,
        wheel_radius: f32,
    ) {
        const VEL_FUSION: f32 = 0.05;
        const VEL_DECAY: f32 = 0.90;

        if wheel_velocities.len() < 2 {
            return;
        }

        // Wheel odometry (average wheel velocity)
        let avg_wheel_vel = (wheel_velocities[0] + wheel_velocities[1]) / 2.0;
        let odom_vel_x = avg_wheel_vel * wheel_radius;

        // Static switch - if wheels aren't moving, assume stationary
        if avg_wheel_vel.abs() < 0.1 {
            self.linear = [0.0, 0.0, 0.0];
            return;
        }

        // IMU integration with yaw rotation
        let yaw_rate = imu.angular_velocity[2];
        let cos_y = (yaw_rate * dt).cos();
        let sin_y = (yaw_rate * dt).sin();

        let vx = self.linear[0];
        let vy = self.linear[1];

        // Rotate velocity by yaw
        self.linear[0] = vx * cos_y + vy * sin_y;
        self.linear[1] = -vx * sin_y + vy * cos_y;

        // Integrate acceleration
        self.linear[0] += imu.linear_acceleration[0] * dt;
        self.linear[1] += imu.linear_acceleration[1] * dt;
        self.linear[2] = 0.0;

        // Fuse with odometry
        self.linear[0] = (1.0 - VEL_FUSION) * self.linear[0] + VEL_FUSION * odom_vel_x;

        // Decay lateral velocity
        self.linear[1] *= VEL_DECAY;
    }
}

/// Observation builder
pub struct ObservationBuilder {
    pub imu: ImuState,
    pub base_velocity: BaseVelocity,
    pub cmd_vel: VelocityCommand,
    pub joint_positions: Vec<f32>,
    pub joint_velocities: Vec<f32>,
    pub default_positions: Vec<f32>,
    pub last_action: Vec<f32>,
    wheel_radius: f32,
}

impl ObservationBuilder {
    /// Create a new observation builder
    pub fn new(num_joints: usize) -> Self {
        Self {
            imu: ImuState::default(),
            base_velocity: BaseVelocity::default(),
            cmd_vel: VelocityCommand::default(),
            joint_positions: vec![0.0; num_joints],
            joint_velocities: vec![0.0; num_joints],
            default_positions: vec![0.0; num_joints],
            last_action: vec![0.0; dims::ACTION_DIM],
            wheel_radius: 0.05, // Default wheel radius
        }
    }

    /// Set wheel radius for velocity estimation
    pub fn set_wheel_radius(&mut self, radius: f32) {
        self.wheel_radius = radius;
    }

    /// Set default joint positions (should match training)
    pub fn set_default_positions(&mut self, positions: &[f32]) {
        self.default_positions.copy_from_slice(positions);
    }

    /// Update IMU state
    pub fn update_imu(&mut self, imu: ImuState) {
        self.imu = imu;
    }

    /// Update joint states from motor feedback
    pub fn update_joints(&mut self, joints: &[JointState]) {
        for (i, joint) in joints.iter().enumerate() {
            if i < self.joint_positions.len() {
                self.joint_positions[i] = joint.position;
                self.joint_velocities[i] = joint.velocity;
            }
        }
    }

    /// Update velocity command
    pub fn update_cmd_vel(&mut self, cmd: VelocityCommand) {
        self.cmd_vel = cmd;
    }

    /// Update last action (for observation)
    pub fn update_last_action(&mut self, action: &[f32]) {
        self.last_action.copy_from_slice(action);
    }

    /// Update velocity estimate
    pub fn update_velocity_estimate(&mut self, dt: f32) {
        // Get wheel velocities (indices 4 and 5 for Bebop)
        let wheel_vels = if self.joint_velocities.len() >= 6 {
            &self.joint_velocities[4..6]
        } else {
            return;
        };

        self.base_velocity
            .update(wheel_vels, &self.imu, dt, self.wheel_radius);
    }

    /// Build the observation vector
    ///
    /// Observation format (must match training!):
    /// [0-2]   Base linear velocity (scaled)
    /// [3-5]   Base angular velocity (scaled)
    /// [6-8]   Projected gravity
    /// [9-11]  Command velocity (scaled)
    /// [12-17] Joint positions (relative to default, scaled)
    /// [18-23] Joint velocities (scaled)
    /// [24-29] Last action
    pub fn build(&self) -> Vec<f32> {
        let mut obs = vec![0.0; dims::OBS_DIM];
        let mut idx = 0;

        // 1. Base linear velocity (3)
        for i in 0..3 {
            let vel =
                self.base_velocity.linear[i].clamp(-scales::CLIP_LIN_VEL, scales::CLIP_LIN_VEL);
            obs[idx] = vel * scales::SCALE_LIN_VEL;
            idx += 1;
        }

        // 2. Base angular velocity (3) - from IMU
        for i in 0..3 {
            let vel =
                self.imu.angular_velocity[i].clamp(-scales::CLIP_ANG_VEL, scales::CLIP_ANG_VEL);
            obs[idx] = vel * scales::SCALE_ANG_VEL;
            idx += 1;
        }

        // 3. Projected gravity (3)
        let gravity = self.imu.projected_gravity();
        for &g in &gravity {
            obs[idx] = g;
            idx += 1;
        }

        // 4. Command velocity (3)
        obs[idx] = self.cmd_vel.linear_x * scales::SCALE_LIN_VEL;
        idx += 1;
        obs[idx] = self.cmd_vel.linear_y * scales::SCALE_LIN_VEL;
        idx += 1;
        obs[idx] = self.cmd_vel.angular_z * scales::SCALE_ANG_VEL;
        idx += 1;

        // 5. Joint positions relative to default (6)
        // Legs first (4), then wheels (2 - always 0 for continuous joints)
        for i in 0..4 {
            let pos = self.joint_positions.get(i).unwrap_or(&0.0);
            let default = self.default_positions.get(i).unwrap_or(&0.0);
            obs[idx] = (pos - default) * scales::SCALE_DOF_POS;
            idx += 1;
        }
        // Wheels (position doesn't matter, use 0)
        obs[idx] = 0.0;
        idx += 1;
        obs[idx] = 0.0;
        idx += 1;

        // 6. Joint velocities (6)
        for i in 0..6 {
            let vel = self.joint_velocities.get(i).unwrap_or(&0.0);
            let clipped = vel.clamp(-scales::CLIP_DOF_VEL, scales::CLIP_DOF_VEL);
            obs[idx] = clipped * scales::SCALE_DOF_VEL;
            idx += 1;
        }

        // 7. Last action (6)
        for i in 0..dims::ACTION_DIM {
            obs[idx] = self.last_action.get(i).copied().unwrap_or(0.0);
            idx += 1;
        }

        assert_eq!(idx, dims::OBS_DIM, "Observation size mismatch");

        obs
    }

    /// Check if robot is upright (for safety)
    pub fn is_upright(&self) -> bool {
        let gravity = self.imu.projected_gravity();
        // Gravity should point down (negative z in body frame)
        gravity[2] < -0.5
    }
}

/// Scale and limit actions for motor commands
pub fn scale_actions(actions: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let mut leg_commands = Vec::with_capacity(4);
    let mut wheel_commands = Vec::with_capacity(2);

    // Legs (position control) - indices 0-3
    for i in 0..4 {
        let action = actions.get(i).copied().unwrap_or(0.0);
        let scaled = action * scales::SCALE_ACTION_LEGS;
        // Add to default position and clamp
        let cmd = scaled.clamp(-limits::MAX_LEG_POS_RAD, limits::MAX_LEG_POS_RAD);
        leg_commands.push(cmd);
    }

    // Wheels (velocity control) - indices 4-5
    for i in 4..6 {
        let action = actions.get(i).copied().unwrap_or(0.0);
        let scaled = action * scales::SCALE_ACTION_WHEELS;
        let cmd = scaled.clamp(-limits::MAX_WHEEL_VEL_RAD_S, limits::MAX_WHEEL_VEL_RAD_S);
        wheel_commands.push(cmd);
    }

    (leg_commands, wheel_commands)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_projected_gravity_upright() {
        let imu = ImuState {
            quaternion: [1.0, 0.0, 0.0, 0.0], // Identity quaternion
            ..Default::default()
        };

        let gravity = imu.projected_gravity();
        assert!((gravity[2] - (-1.0)).abs() < 0.01);
    }

    #[test]
    fn test_observation_size() {
        let builder = ObservationBuilder::new(6);
        let obs = builder.build();
        assert_eq!(obs.len(), dims::OBS_DIM);
    }

    #[test]
    fn test_scale_actions() {
        let actions = vec![0.5, -0.5, 0.5, -0.5, 0.5, -0.5];
        let (legs, wheels) = scale_actions(&actions);

        assert_eq!(legs.len(), 4);
        assert_eq!(wheels.len(), 2);

        // Check scaling
        assert!((legs[0] - 0.4).abs() < 0.01); // 0.5 * 0.8 = 0.4
        assert!((wheels[0] - 10.0).abs() < 0.01); // 0.5 * 20 = 10
    }
}
