//! Observation building and action scaling for Bebop V2 policy inference.
//!
//! The 52-element observation vector and 24-element MIT-mode action
//! vector layouts are owned by [`crate::config::dims`] (see the docstring
//! there for the authoritative spec). Anything in this file is a Rust
//! mirror of what `sim/bebop_training/envs/bebop_v2_base_cfg.py` is
//! doing during training.
//!
//! ## Joint order
//!
//! All 8-slot joint groups in both observations and actions use this
//! order, matching `JOINT_NAMES_ALL` in
//! `sim/bebop_training/envs/bebop_v2_base_cfg.py`. It is the
//! left/right-interleaved BFS-from-root order Newton physics produces from
//! the URDF. The 24-dim MIT-mode action stacks three of these 8-slot
//! groups in the order: positions, kp, kd.
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

use crate::config::{dims, scales, JointState, PolicyGainClamps};
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

/// Stateful builder that assembles the 52-element observation vector once
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

    /// Assemble the 52-element observation vector. Layout matches
    /// `bebop_v2_base_cfg.py::PolicyCfg`:
    ///
    /// ```text
    ///   [ 0.. 3)  base_lin_vel
    ///   [ 3.. 6)  base_ang_vel
    ///   [ 6.. 9)  projected_gravity
    ///   [ 9..17)  joint_pos_rel       (q - q_default)
    ///   [17..25)  joint_vel_rel       (q_dot - q_dot_default; q_dot_default = 0)
    ///   [25..49)  last_action         (24-dim raw NN output: pos | kp | kd)
    ///   [49..52)  velocity_commands   (vx, vy, wz)
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
// Action decoding (MIT-mode variable impedance)
// ---------------------------------------------------------------------------

/// Decoded MIT-mode action: 8 position targets + 8 kp + 8 kd, in
/// [`JOINT_NAMES`] order.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct DecodedAction {
    pub targets: [f32; NUM_JOINTS],
    pub kp: [f32; NUM_JOINTS],
    pub kd: [f32; NUM_JOINTS],
}

/// Decode the raw 24-element policy action into per-joint position
/// targets, kp, and kd values.
///
/// Layout (mirrors `bebop_v2_actions.py::VariableImpedanceJointAction`):
///
/// - `action[ 0.. 8)` raw position commands
/// - `action[ 8..16)` raw kp commands
/// - `action[16..24)` raw kd commands
///
/// Per-channel transform with `a = clamp(raw, -1, 1)`:
///
/// ```text
///   target[i] = default_pos[i] + SCALE_ACTION * a_pos
///   kp[i]     = kp_min[i] + (a_kp + 1) / 2 * (kp_max[i] - kp_min[i])
///   kd[i]     = kd_min[i] + (a_kd + 1) / 2 * (kd_max[i] - kd_min[i])
/// ```
///
/// The `[-1, 1]` clip is defense-in-depth — rsl_rl's Gaussian head can
/// emit outliers above 1, and we'd rather pull those back to the trained
/// envelope than let the per-joint clamps absorb the difference silently.
/// The supervisor additionally clamps the position to `pos_min`/`pos_max`
/// before TX; all clamps are intentional.
///
/// `clamps[i]` MUST come from `JointConfig::policy_gain_clamps` for the
/// joint at policy slot `i` (which is the joint whose name is
/// `JOINT_NAMES[i]`).
pub fn decode_policy_action(
    action: &[f32],
    default_positions: &[f32; NUM_JOINTS],
    clamps: &[PolicyGainClamps; NUM_JOINTS],
) -> DecodedAction {
    let mut out = DecodedAction::default();
    for i in 0..NUM_JOINTS {
        let a_pos = action.get(i).copied().unwrap_or(0.0).clamp(-1.0, 1.0);
        let a_kp = action
            .get(NUM_JOINTS + i)
            .copied()
            .unwrap_or(0.0)
            .clamp(-1.0, 1.0);
        let a_kd = action
            .get(2 * NUM_JOINTS + i)
            .copied()
            .unwrap_or(0.0)
            .clamp(-1.0, 1.0);

        out.targets[i] = default_positions[i] + scales::SCALE_ACTION * a_pos;

        let c = clamps[i];
        out.kp[i] = c.kp_min + 0.5 * (a_kp + 1.0) * (c.kp_max - c.kp_min);
        out.kd[i] = c.kd_min + 0.5 * (a_kd + 1.0) * (c.kd_max - c.kd_min);
    }
    out
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
        assert_eq!(obs.len(), 52);
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
        // 24-dim last_action: 8 positions, 8 kp, 8 kd.
        let last_action: Vec<f32> = (0..dims::ACTION_DIM).map(|i| 0.01 * i as f32).collect();
        builder.update_last_action(&last_action);
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
        assert_eq!(&obs[25..49], last_action.as_slice()); // last_action (24 dims)
        assert_eq!(&obs[49..52], &[0.7, 0.8, 0.9]); // velocity_commands
    }

    fn default_clamps() -> [PolicyGainClamps; NUM_JOINTS] {
        let c = PolicyGainClamps {
            kp_min: 10.0,
            kp_max: 110.0,
            kd_min: 1.0,
            kd_max: 5.0,
        };
        [c; NUM_JOINTS]
    }

    #[test]
    fn decode_action_clips_and_offsets_position_channel() {
        let defaults = [0.0_f32; NUM_JOINTS];
        let clamps = default_clamps();
        // 24-dim raw action: first 8 = position, next 16 = gains at midpoint (0).
        let mut raw = [0.0_f32; dims::ACTION_DIM];
        raw[0..NUM_JOINTS].copy_from_slice(&[0.5, -0.5, 1.5, -1.5, 0.0, 0.25, -0.25, 1.0]);

        let decoded = decode_policy_action(&raw, &defaults, &clamps);

        assert!((decoded.targets[0] - 0.4).abs() < 1e-5); //  0.5 * 0.8
        assert!((decoded.targets[1] - (-0.4)).abs() < 1e-5); // -0.5 * 0.8
        assert!((decoded.targets[2] - 0.8).abs() < 1e-5); // clipped to  1.0 -> 0.8
        assert!((decoded.targets[3] - (-0.8)).abs() < 1e-5); // clipped to -1.0 -> -0.8
        assert!(decoded.targets[4].abs() < 1e-5);
        assert!((decoded.targets[5] - 0.2).abs() < 1e-5);
        assert!((decoded.targets[6] - (-0.2)).abs() < 1e-5);
        assert!((decoded.targets[7] - 0.8).abs() < 1e-5);
    }

    #[test]
    fn decode_action_maps_gains_with_midpoint_at_raw_zero() {
        let defaults = [0.0_f32; NUM_JOINTS];
        let clamps = default_clamps();
        let raw = [0.0_f32; dims::ACTION_DIM];

        let decoded = decode_policy_action(&raw, &defaults, &clamps);

        let kp_mid = 0.5 * (clamps[0].kp_min + clamps[0].kp_max);
        let kd_mid = 0.5 * (clamps[0].kd_min + clamps[0].kd_max);
        for i in 0..NUM_JOINTS {
            assert!((decoded.kp[i] - kp_mid).abs() < 1e-5);
            assert!((decoded.kd[i] - kd_mid).abs() < 1e-5);
        }
    }

    #[test]
    fn decode_action_maps_gains_at_extremes() {
        let defaults = [0.0_f32; NUM_JOINTS];
        let clamps = default_clamps();
        let mut raw = [0.0_f32; dims::ACTION_DIM];
        // raw_kp = +1 for all -> kp = kp_max; raw_kd = -1 for all -> kd = kd_min.
        for i in 0..NUM_JOINTS {
            raw[NUM_JOINTS + i] = 1.0;
            raw[2 * NUM_JOINTS + i] = -1.0;
        }
        // Also test clipping: send raw_kp = 2.5 on slot 0 and raw_kd = -3.0 on slot 1.
        raw[NUM_JOINTS] = 2.5;
        raw[2 * NUM_JOINTS + 1] = -3.0;

        let decoded = decode_policy_action(&raw, &defaults, &clamps);
        for i in 0..NUM_JOINTS {
            assert!((decoded.kp[i] - clamps[i].kp_max).abs() < 1e-5);
            assert!((decoded.kd[i] - clamps[i].kd_min).abs() < 1e-5);
        }
    }

    #[test]
    fn decode_action_respects_default_positions() {
        let defaults = [0.1_f32, -0.1, 0.2, -0.2, 0.3, -0.3, 0.4, -0.4];
        let clamps = default_clamps();
        let raw = [0.0_f32; dims::ACTION_DIM];
        let decoded = decode_policy_action(&raw, &defaults, &clamps);
        assert_eq!(decoded.targets, defaults);
    }

    #[test]
    fn joint_names_table_has_no_typos() {
        // Cheap guard: protects against future renames silently breaking
        // the policy <-> firmware contract. If you legitimately rename a
        // joint, update both this table and `bebop_v2.yaml` keys.
        assert_eq!(JOINT_NAMES.len(), NUM_JOINTS);
        // The MIT-mode action has 3 channels per joint (position, kp, kd).
        assert_eq!(3 * JOINT_NAMES.len(), dims::ACTION_DIM);
        assert!(JOINT_NAMES.iter().all(|n| n.ends_with("_joint")));
    }
}
