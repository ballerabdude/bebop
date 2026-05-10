//! Observation building and action scaling for Bebop V2 policy inference.
//!
//! The 36-element observation vector and 8-element action vector layouts
//! are owned by [`crate::config::dims`] (see the docstring there for the
//! authoritative spec). Anything in this file is a Rust mirror of what
//! `sim/bebop_training/envs/bebop_v2_base_cfg.py` is doing during training.
//!
//! ## Joint order
//!
//! All 8-element joint slots in both observations and actions use this
//! order, matching `JOINT_NAMES_ALL` in
//! `sim/bebop_training/envs/bebop_v2_base_cfg.py:34-43`. It is the
//! left/right-interleaved BFS-from-root order Newton physics produces from
//! the URDF.
//!
//! | idx | joint                         |
//! |-----|-------------------------------|
//! |  0  | `hip_abduction_left_joint`    |
//! |  1  | `hip_abduction_right_joint`   |
//! |  2  | `femur_left_joint`            |
//! |  3  | `femur_right_joint`           |
//! |  4  | `shin_left_joint`             |
//! |  5  | `shin_right_joint`            |
//! |  6  | `foot_left_joint`             |
//! |  7  | `foot_right_joint`            |

use crate::config::{dims, scales, JointState};
use nalgebra::{Quaternion, UnitQuaternion, Vector3};

/// Number of revolute joints driven by the policy on Bebop V2.
pub const NUM_JOINTS: usize = 8;

/// Joint names in policy order. Index into the observation/action vector.
pub const JOINT_NAMES: [&str; NUM_JOINTS] = [
    "hip_abduction_left_joint",
    "hip_abduction_right_joint",
    "femur_left_joint",
    "femur_right_joint",
    "shin_left_joint",
    "shin_right_joint",
    "foot_left_joint",
    "foot_right_joint",
];

// ---------------------------------------------------------------------------
// IMU + body-frame quantities
// ---------------------------------------------------------------------------

/// IMU state in body frame, sourced from the BNO085 (or whichever IMU the
/// vehicle ships with) over the CAN gateway / ROS bridge.
///
/// ## Quaternion convention: XYZW (scalar last)
///
/// `quaternion = [x, y, z, w]`. Aligned with:
///
/// - ROS 2 `geometry_msgs/Quaternion` (`x, y, z, w`),
/// - Isaac Lab 3.0 / Warp / Newton / PhysX (all migrated to XYZW),
/// - rsl_rl observation pipeline (the trained policy was exposed to gravity
///   already projected, so its training is convention-invariant — but every
///   *new* code path in this stack should be XYZW).
///
/// **Producer responsibility**: any code feeding `ImuState` must convert
/// from its source convention before populating this field:
///
/// - BNO085 raw output is WXYZ → reorder before storing here,
/// - ROS `sensor_msgs/Imu.orientation` is already XYZW (`x, y, z, w`),
/// - Isaac Lab `asset.data.root_quat_w` is XYZW since IL 3.0.
#[derive(Debug, Clone, Default)]
pub struct ImuState {
    /// Body-to-world rotation as a unit quaternion in `[x, y, z, w]` order.
    pub quaternion: [f32; 4],
    /// Body-frame angular velocity (rad/s).
    pub angular_velocity: [f32; 3],
    /// Body-frame linear acceleration (m/s²). Currently unused for the
    /// policy observation but kept on the struct for downstream estimators.
    pub linear_acceleration: [f32; 3],
}

impl ImuState {
    /// Compute the world-frame gravity vector `(0, 0, -1)` projected into
    /// body frame. The policy uses this as a "which way is up" cue.
    ///
    /// `nalgebra::Quaternion::new` takes `(w, i, j, k)`, so we re-index out
    /// of our XYZW storage explicitly.
    pub fn projected_gravity(&self) -> [f32; 3] {
        let [x, y, z, w] = self.quaternion;
        let quat = UnitQuaternion::from_quaternion(Quaternion::new(w, x, y, z));
        let gravity_world = Vector3::new(0.0, 0.0, -1.0);
        let gravity_body = quat.inverse() * gravity_world;
        [gravity_body.x, gravity_body.y, gravity_body.z]
    }
}

/// Pilot-supplied velocity command in body frame.
#[derive(Debug, Clone, Default)]
pub struct VelocityCommand {
    pub linear_x: f32,
    pub linear_y: f32,
    pub angular_z: f32,
}

/// Estimated base linear velocity in body frame (m/s).
///
/// Bebop V2 has no wheels, so this cannot be derived from joint
/// instrumentation alone. Feed it from an external estimator (EKF over
/// IMU + foot-contact, visual odometry, T265, ...). For first bring-up
/// publishing zeros is acceptable — see
/// `ros2/src/bebop_pilot/config/policy_runner.yaml::lin_vel_source`.
#[derive(Debug, Clone, Default)]
pub struct BaseVelocity {
    pub linear: [f32; 3],
}

// ---------------------------------------------------------------------------
// Observation builder
// ---------------------------------------------------------------------------

/// Stateful builder that assembles the 36-element observation vector once
/// per control tick. Update the individual fields whenever fresh data
/// arrives (IMU 200 Hz, joint feedback ~100 Hz, cmd_vel async); call
/// [`ObservationBuilder::build`] at the policy rate.
pub struct ObservationBuilder {
    pub imu: ImuState,
    pub base_velocity: BaseVelocity,
    pub cmd_vel: VelocityCommand,
    pub joint_positions: [f32; NUM_JOINTS],
    pub joint_velocities: [f32; NUM_JOINTS],
    pub default_positions: [f32; NUM_JOINTS],
    pub last_action: [f32; dims::ACTION_DIM],
}

impl Default for ObservationBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ObservationBuilder {
    pub fn new() -> Self {
        Self {
            imu: ImuState::default(),
            base_velocity: BaseVelocity::default(),
            cmd_vel: VelocityCommand::default(),
            joint_positions: [0.0; NUM_JOINTS],
            joint_velocities: [0.0; NUM_JOINTS],
            default_positions: [0.0; NUM_JOINTS],
            last_action: [0.0; dims::ACTION_DIM],
        }
    }

    /// Set the joint "home" positions the policy was trained against.
    /// For the stock V2 training (`bebop_v2_base_cfg.py`) all defaults are
    /// 0.0 — but if a future curriculum trains around a non-zero stance
    /// this is the knob.
    pub fn set_default_positions(&mut self, positions: &[f32; NUM_JOINTS]) {
        self.default_positions.copy_from_slice(positions);
    }

    pub fn update_imu(&mut self, imu: ImuState) {
        self.imu = imu;
    }

    pub fn update_base_velocity(&mut self, linear: [f32; 3]) {
        self.base_velocity.linear = linear;
    }

    /// Update joint state from the supervisor's per-joint feedback.
    ///
    /// `joints` is indexed by [`JOINT_NAMES`] / [`crate::config::JointConfig::index`].
    /// Slices shorter than [`NUM_JOINTS`] leave the trailing entries unchanged;
    /// slices longer than [`NUM_JOINTS`] silently ignore the extras.
    pub fn update_joints(&mut self, joints: &[JointState]) {
        for (i, joint) in joints.iter().take(NUM_JOINTS).enumerate() {
            self.joint_positions[i] = joint.position;
            self.joint_velocities[i] = joint.velocity;
        }
    }

    pub fn update_cmd_vel(&mut self, cmd: VelocityCommand) {
        self.cmd_vel = cmd;
    }

    /// Record the previous tick's raw policy output so it can become part
    /// of the next observation (`last_action` term).
    pub fn update_last_action(&mut self, action: &[f32]) {
        for (i, slot) in self.last_action.iter_mut().enumerate() {
            *slot = action.get(i).copied().unwrap_or(0.0);
        }
    }

    /// Assemble the 36-element observation vector. Layout matches
    /// `bebop_v2_base_cfg.py::PolicyCfg`:
    ///
    /// ```text
    ///   [ 0.. 3)  base_lin_vel
    ///   [ 3.. 6)  base_ang_vel
    ///   [ 6.. 9)  projected_gravity
    ///   [ 9..17)  joint_pos_rel       (q - q_default)
    ///   [17..25)  joint_vel_rel       (q_dot - q_dot_default; q_dot_default = 0)
    ///   [25..33)  last_action
    ///   [33..36)  velocity_commands   (vx, vy, wz)
    /// ```
    pub fn build(&self) -> Vec<f32> {
        let mut obs = vec![0.0_f32; dims::OBS_DIM];
        let mut idx = 0;

        for &v in &self.base_velocity.linear {
            obs[idx] = v.clamp(-scales::CLIP_LIN_VEL, scales::CLIP_LIN_VEL) * scales::SCALE_LIN_VEL;
            idx += 1;
        }

        for &v in &self.imu.angular_velocity {
            obs[idx] = v.clamp(-scales::CLIP_ANG_VEL, scales::CLIP_ANG_VEL) * scales::SCALE_ANG_VEL;
            idx += 1;
        }

        for &g in &self.imu.projected_gravity() {
            obs[idx] = g;
            idx += 1;
        }

        for i in 0..NUM_JOINTS {
            obs[idx] =
                (self.joint_positions[i] - self.default_positions[i]) * scales::SCALE_DOF_POS;
            idx += 1;
        }

        for i in 0..NUM_JOINTS {
            let v = self.joint_velocities[i].clamp(-scales::CLIP_DOF_VEL, scales::CLIP_DOF_VEL);
            obs[idx] = v * scales::SCALE_DOF_VEL;
            idx += 1;
        }

        for i in 0..dims::ACTION_DIM {
            obs[idx] = self.last_action[i];
            idx += 1;
        }

        obs[idx] = self.cmd_vel.linear_x * scales::SCALE_LIN_VEL;
        idx += 1;
        obs[idx] = self.cmd_vel.linear_y * scales::SCALE_LIN_VEL;
        idx += 1;
        obs[idx] = self.cmd_vel.angular_z * scales::SCALE_ANG_VEL;
        idx += 1;

        debug_assert_eq!(idx, dims::OBS_DIM, "observation builder index mismatch");
        obs
    }

    /// Whether the projected-gravity z-component looks roughly upright.
    /// Used as a coarse safety gate before arming RunPolicy.
    pub fn is_upright(&self) -> bool {
        self.imu.projected_gravity()[2] < -0.5
    }
}

// ---------------------------------------------------------------------------
// Action scaling
// ---------------------------------------------------------------------------

/// Convert a raw 8-element policy action into 8 joint-position targets in
/// [`JOINT_NAMES`] order.
///
/// Mirrors training's `JointPositionActionCfg(scale=0.8, use_default_offset=True)`:
///
/// ```text
///   target[i] = default_pos[i] + SCALE_ACTION * clip(action[i], -1, 1)
/// ```
///
/// The `[-1, 1]` clip is a defense-in-depth: rsl_rl's Gaussian head can
/// occasionally emit outliers above 1, and we'd rather pull those back to
/// the trained envelope than have the supervisor clamp them silently.
/// The supervisor will *additionally* clamp to per-joint `pos_min`/`pos_max`
/// before the wire — both clamps are intentional.
pub fn scale_actions_to_targets(
    action: &[f32],
    default_positions: &[f32; NUM_JOINTS],
) -> [f32; NUM_JOINTS] {
    let mut targets = [0.0_f32; NUM_JOINTS];
    for i in 0..NUM_JOINTS {
        let a = action.get(i).copied().unwrap_or(0.0).clamp(-1.0, 1.0);
        targets[i] = default_positions[i] + scales::SCALE_ACTION * a;
    }
    targets
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projected_gravity_upright() {
        let imu = ImuState {
            // XYZW identity = body aligned with world; w (scalar) at index 3.
            quaternion: [0.0, 0.0, 0.0, 1.0],
            ..Default::default()
        };
        let gravity = imu.projected_gravity();
        assert!((gravity[0]).abs() < 1e-5);
        assert!((gravity[1]).abs() < 1e-5);
        assert!((gravity[2] - (-1.0)).abs() < 1e-5);
    }

    #[test]
    fn projected_gravity_180_about_x_flips_z() {
        // 180° rotation about body x-axis: world (0,0,-1) -> body (0,0,+1).
        // Quaternion in XYZW: (sin(90°), 0, 0, cos(90°)) = (1, 0, 0, 0).
        // This guards against an accidental WXYZ regression — under the
        // old (wrong) interpretation, [1,0,0,0] would be the *identity*
        // quat and gravity_z would still be -1.
        let imu = ImuState {
            quaternion: [1.0, 0.0, 0.0, 0.0],
            ..Default::default()
        };
        let gravity = imu.projected_gravity();
        assert!((gravity[0]).abs() < 1e-5);
        assert!((gravity[1]).abs() < 1e-5);
        assert!((gravity[2] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn observation_size_matches_dims() {
        let builder = ObservationBuilder::new();
        let obs = builder.build();
        assert_eq!(obs.len(), dims::OBS_DIM);
        assert_eq!(obs.len(), 36);
    }

    #[test]
    fn observation_layout_is_v2() {
        let mut builder = ObservationBuilder::new();
        builder.update_base_velocity([0.1, 0.2, 0.3]);
        builder.update_imu(ImuState {
            // XYZW identity (scalar last).
            quaternion: [0.0, 0.0, 0.0, 1.0],
            angular_velocity: [0.4, 0.5, 0.6],
            linear_acceleration: [0.0; 3],
        });
        builder.joint_positions = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        builder.joint_velocities = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
        builder.update_last_action(&[0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18]);
        builder.update_cmd_vel(VelocityCommand {
            linear_x: 0.7,
            linear_y: 0.8,
            angular_z: 0.9,
        });

        let obs = builder.build();
        assert_eq!(&obs[0..3], &[0.1, 0.2, 0.3]); // base_lin_vel
        assert_eq!(&obs[3..6], &[0.4, 0.5, 0.6]); // base_ang_vel
                                                  // projected_gravity in body frame for identity quat -> (0, 0, -1)
        assert!(obs[6].abs() < 1e-5 && obs[7].abs() < 1e-5);
        assert!((obs[8] - (-1.0)).abs() < 1e-5);
        assert_eq!(&obs[9..17], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]); // joint_pos
                                                                            // joint_vel: 80.0 exceeds CLIP_DOF_VEL=15.0, so it should clamp.
        assert!((obs[17] - 10.0).abs() < 1e-5);
        assert!((obs[24] - scales::CLIP_DOF_VEL).abs() < 1e-5);
        assert_eq!(
            &obs[25..33],
            &[0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18]
        ); // last_action
        assert_eq!(&obs[33..36], &[0.7, 0.8, 0.9]); // velocity_commands
    }

    #[test]
    fn scale_actions_clips_and_offsets() {
        let defaults = [0.0_f32; NUM_JOINTS];
        let raw = [0.5, -0.5, 1.5, -1.5, 0.0, 0.25, -0.25, 1.0];
        let targets = scale_actions_to_targets(&raw, &defaults);

        assert!((targets[0] - 0.4).abs() < 1e-5); //  0.5 * 0.8
        assert!((targets[1] - (-0.4)).abs() < 1e-5); // -0.5 * 0.8
        assert!((targets[2] - 0.8).abs() < 1e-5); // clipped to 1.0 -> 0.8
        assert!((targets[3] - (-0.8)).abs() < 1e-5); // clipped to -1.0 -> -0.8
        assert!(targets[4].abs() < 1e-5);
        assert!((targets[5] - 0.2).abs() < 1e-5);
        assert!((targets[6] - (-0.2)).abs() < 1e-5);
        assert!((targets[7] - 0.8).abs() < 1e-5);
    }

    #[test]
    fn scale_actions_respects_default_positions() {
        let defaults = [0.1_f32, -0.1, 0.2, -0.2, 0.3, -0.3, 0.4, -0.4];
        let raw = [0.0_f32; NUM_JOINTS];
        let targets = scale_actions_to_targets(&raw, &defaults);
        assert_eq!(targets, defaults);
    }

    #[test]
    fn joint_names_table_has_no_typos() {
        // Cheap guard: protects against future renames silently breaking
        // the policy <-> firmware contract. If you legitimately rename a
        // joint, update both this table and `bebop_v2.yaml` keys.
        assert_eq!(JOINT_NAMES.len(), NUM_JOINTS);
        assert_eq!(JOINT_NAMES.len(), dims::ACTION_DIM);
        assert!(JOINT_NAMES.iter().all(|n| n.ends_with("_joint")));
    }
}
