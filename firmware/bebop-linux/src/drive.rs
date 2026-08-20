//! Differential-drive kinematics and odometry for the wheeled chassis.
//!
//! Operates in **robot-frame** wheel quantities: [`crate::odrive`] reports
//! raw motor rotation, and the `direction` sign on each [`WheelConfig`] is
//! applied at the wheel⇄robot boundary (in [`Supervisor`]'s wheel path) so
//! that everything here sees "left/right wheel forward = positive" in
//! radians. Angles are radians, linear speeds m/s, distances metres.

use crate::config::DiffDriveConfig;

/// A body-frame motion command: `vx` (forward m/s), `wz` (yaw rate rad/s,
/// + = left turn / CCW).
#[derive(Debug, Clone, Copy, Default)]
pub struct Twist {
    pub vx: f32,
    pub wz: f32,
}

/// Inverse kinematics: twist → left/right wheel angular velocities (rad/s).
///
/// ```text
///   v_l = (vx - wz * half_track) / radius
///   v_r = (vx + wz * half_track) / radius
/// ```
pub fn twist_to_wheel_angular(vx: f32, wz: f32, cfg: &DiffDriveConfig) -> (f32, f32) {
    let v_l = (vx - wz * cfg.half_track) / cfg.wheel_radius;
    let v_r = (vx + wz * cfg.half_track) / cfg.wheel_radius;
    (v_l, v_r)
}

/// Forward kinematics: left/right wheel angular velocities (rad/s) → twist.
///
/// ```text
///   vx = radius * (v_l + v_r) / 2
///   wz = radius * (v_r - v_l) / (2 * half_track)
/// ```
pub fn wheel_angular_to_twist(v_l: f32, v_r: f32, cfg: &DiffDriveConfig) -> Twist {
    let vx = cfg.wheel_radius * (v_l + v_r) * 0.5;
    let wz = cfg.wheel_radius * (v_r - v_l) / (2.0 * cfg.half_track);
    Twist { vx, wz }
}

/// Dead-reckoning pose integrator (wheel-encoder-only; no IMU fusion yet).
#[derive(Debug, Clone, Copy, Default)]
pub struct Odometry {
    /// World-frame position (m). `theta` = yaw (rad), 0 = start heading.
    pub x: f32,
    pub y: f32,
    pub theta: f32,
}

impl Odometry {
    /// Advance the pose by one timestep given wheel angular velocities
    /// (`rad/s`) and `dt` (seconds). Uses the midpoint yaw for a first-
    /// order accurate integration that stays stable at 100 Hz.
    pub fn step(&mut self, v_l: f32, v_r: f32, dt: f32, cfg: &DiffDriveConfig) {
        let t = wheel_angular_to_twist(v_l, v_r, cfg);
        let half_dtheta = 0.5 * t.wz * dt;
        let cos = (self.theta + half_dtheta).cos();
        let sin = (self.theta + half_dtheta).sin();
        self.x += t.vx * dt * cos;
        self.y += t.vx * dt * sin;
        self.theta += t.wz * dt;
    }

    /// Reset to the origin (used on odometry re-zero).
    pub fn reset(&mut self) {
        self.x = 0.0;
        self.y = 0.0;
        self.theta = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> DiffDriveConfig {
        DiffDriveConfig {
            left_wheel: "left".into(),
            right_wheel: "right".into(),
            wheel_radius: 0.05,
            half_track: 0.15,
        }
    }

    #[test]
    fn straight_line_maps_equal_wheel_speeds() {
        let c = cfg();
        let (vl, vr) = twist_to_wheel_angular(1.0, 0.0, &c);
        assert_eq!(vl, vr);
        assert!((vl - 20.0).abs() < 1e-4); // 1 m/s / 0.05 m
    }

    #[test]
    fn turn_in_place_maps_opposite_wheel_speeds() {
        let c = cfg();
        let (vl, vr) = twist_to_wheel_angular(0.0, 1.0, &c);
        assert!((vl + vr).abs() < 1e-4); // opposite signs
        assert!((vr - 3.0).abs() < 1e-3); // wz*half_track / radius = 1*0.15/0.05
    }

    #[test]
    fn round_trip_twist_wheel_twist() {
        let c = cfg();
        let (vl, vr) = twist_to_wheel_angular(0.8, 0.3, &c);
        let t = wheel_angular_to_twist(vl, vr, &c);
        assert!((t.vx - 0.8).abs() < 1e-4);
        assert!((t.wz - 0.3).abs() < 1e-4);
    }

    #[test]
    fn odometry_straight_line_accumulates_distance() {
        let c = cfg();
        let mut odom = Odometry::default();
        let (vl, vr) = twist_to_wheel_angular(1.0, 0.0, &c);
        for _ in 0..100 {
            odom.step(vl, vr, 0.01, &c);
        }
        assert!((odom.x - 1.0).abs() < 1e-3);
        assert!(odom.y.abs() < 1e-4);
        assert!(odom.theta.abs() < 1e-4);
    }

    #[test]
    fn odometry_turn_in_place_rotates_in_place() {
        let c = cfg();
        let mut odom = Odometry::default();
        let (vl, vr) = twist_to_wheel_angular(0.0, 0.5, &c);
        for _ in 0..100 {
            odom.step(vl, vr, 0.01, &c);
        }
        // 1 s at 0.5 rad/s -> 0.5 rad of heading.
        assert!((odom.theta - 0.5).abs() < 1e-3);
        assert!(odom.x.abs() < 1e-3);
        assert!(odom.y.abs() < 1e-3);
    }
}
