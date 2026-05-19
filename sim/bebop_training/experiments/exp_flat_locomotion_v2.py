"""Locomotion ("flat walk") experiment for Bebop V2.

Velocity-tracking task with gentle command ranges and light push disturbances.
Warm-start this from a balance checkpoint with ``train_bebop.py --resume <path>``
for fastest convergence.
"""

import isaaclab.envs.mdp as mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass

from ..envs.bebop_v2_base_cfg import BebopV2BaseEnvCfg


@configclass
class BebopV2FlatLocomotionCfg(BebopV2BaseEnvCfg):
    """Velocity-tracking locomotion task config for the Bebop V2 articulation."""

    def __post_init__(self):
        super().__post_init__()

        # Gentle command ranges so the policy can keep balancing while it
        # learns to walk. Most envs receive non-zero commands so the policy is
        # forced to attempt motion (only 10% standing baseline).
        self.commands.base_velocity.resampling_time_range = (6.0, 10.0)
        self.commands.base_velocity.rel_standing_envs = 0.1
        self.commands.base_velocity.ranges = mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.3, 0.5),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-0.5, 0.5),
        )

        # Reward shaping: bias HEAVILY toward velocity tracking so the policy
        # is incentivised to attempt motion rather than just collect the
        # standing alive bonus.
        self.rewards.track_lin_vel_xy.weight = 4.0
        self.rewards.track_ang_vel_z.weight = 2.0
        # Drop the alive bonus so standing-in-place stops being a
        # comfortable local optimum that out-earns attempting motion.
        self.rewards.alive.weight = 0.25

        # Disable the "hold still" rewards entirely during locomotion so they
        # don't punish stepping motions.
        self.rewards.leg_action_when_stable.weight = 0.0
        self.rewards.leg_hold_reward.weight = 0.0
        self.rewards.undesired_yaw.weight = -0.5

        # Posture: keep torso upright and pull joints back to neutral so the
        # policy walks tall instead of crouching against joint limits.
        self.rewards.flat_orientation_l2.weight = -2.0
        self.rewards.joint_deviation.weight = -0.1
        self.rewards.base_height.weight = -3.0
        self.rewards.joint_pos_limits.weight = -2.0

        # Light random pushes during locomotion for sim-to-real robustness.
        self.events.push_robot = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(8.0, 12.0),
            params={"velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}},
        )
