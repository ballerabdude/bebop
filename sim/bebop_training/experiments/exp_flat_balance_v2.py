"""Stand-only ("flat balance") experiment for Bebop V2.

The robot is asked to stand in place; all velocity commands are zero and the
"hold still when stable" rewards are strengthened. Use this for the initial
balance training stage. Once the policy survives episodes reliably, switch to
``BebopV2FlatLocomotionCfg`` to learn walking.
"""

import isaaclab.envs.mdp as mdp
from isaaclab.utils import configclass

from ..envs.bebop_v2_base_cfg import BebopV2BaseEnvCfg


@configclass
class BebopV2FlatBalanceCfg(BebopV2BaseEnvCfg):
    """Stand-in-place task config for the Bebop V2 USD articulation."""

    def __post_init__(self):
        super().__post_init__()

        # Force every command to zero -> pure standing task.
        self.commands.base_velocity.rel_standing_envs = 1.0
        self.commands.base_velocity.ranges = mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
        )

        # Strongly reward "hold still when balanced" while standing.
        # self.rewards.leg_action_when_stable.weight = -3.0
        self.rewards.leg_hold_reward.weight = 1.0
        # self.rewards.undesired_yaw.weight = -2.0

        # Anti-splay: femur is the actual lateral hip-abduction axis.
        # During pure standing the policy has no reason to widen the
        # stance, so crank the deviation penalty hard. Joint position
        # limits (USD / YAML hard_limits) are deliberately left wide for
        # future use — this term shrinks only the *operating* envelope
        # the policy chooses to live in, via the reward landscape.
        self.rewards.femur_deviation.weight = -3.0

        # Feet flat: while standing there's no legitimate reason for
        # either sole to tilt off the ground. Crank the base ``foot_flat``
        # reward up so it's on par with ``torso_upright_via_legs``; the
        # two together force a straight-leg / flat-foot / upright-torso
        # standing pose rather than the policy's previous "splay + bent
        # knees" cheat. Tighten the shaping width too — at rest we want
        # genuinely flat feet (<5° tilt), not "approximately level".
        self.rewards.foot_flat.weight = 2.0
        self.rewards.foot_flat.params["std"] = 0.10

        # Disable push disturbances during the balance stage.
        if hasattr(self.events, "push_robot"):
            self.events.push_robot = None
