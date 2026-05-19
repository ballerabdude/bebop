"""Stand + push-recovery ("flat balance") experiment for Bebop V2.

Single merged task that trains both standing in place AND recovering
from random lateral pushes. Previously this was split across two
stages (``Isaac-BebopV2-Flat-v0`` for pure standing, then
``Isaac-BebopV2-FlatRobust-v0`` for push recovery as a warm-start);
the new ``BebopV2BaseEnvCfg.EventCfg`` already wires in both
wide-range initial-condition randomisation *and* periodic mid-episode
pushes, so a single training stage produces a policy that does both.

The robot is asked to hold zero velocity command. Reward shaping
biases toward:

* upright torso (``torso_upright_via_legs``),
* flat feet (``foot_flat`` — relaxed shaping width here so dynamic
  recovery is not punished for a transient toe-down),
* legs near zero, especially the femur ("anti-splay"
  ``femur_deviation``),
* symmetric left/right configuration (the four ``*_symmetry`` terms
  inherited from the base cfg).

Knee bend is **not** prescribed any more — the previous
``knee_bend_reward`` was forcing a specific crouch angle, which the
policy learned as part of a single static "safe" posture. Removing it
lets the policy discover its own preferred standing pose under the
combined torso / foot / symmetry / deviation constraints. In
practice this should converge to near-straight legs, flat feet,
upright torso, with whatever small flex the policy finds useful for
push absorption.
"""

import isaaclab.envs.mdp as mdp
from isaaclab.utils import configclass

from ..envs.bebop_v2_base_cfg import BebopV2BaseEnvCfg


@configclass
class BebopV2FlatBalanceCfg(BebopV2BaseEnvCfg):
    """Stand-in-place + push-recovery task for the Bebop V2 articulation."""

    def __post_init__(self):
        super().__post_init__()

        # Force every command to zero -> pure standing task. Pushes from
        # the inherited ``push_robot`` event are what generate the
        # base-velocity transients the policy must recover from.
        self.commands.base_velocity.rel_standing_envs = 1.0
        self.commands.base_velocity.ranges = mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
        )

        # Reward shaping overrides for the merged balance task.
        self.rewards.leg_hold_reward.weight = 1.0

        # Anti-splay: femur is the actual lateral hip-abduction axis on
        # this articulation. Keep the deviation penalty strong so the
        # policy never resorts to widening the stance for a cheap
        # support polygon. Joint position limits in the USD/firmware
        # YAML are deliberately left wide for future use; this term
        # constrains only the *operating* envelope.
        self.rewards.femur_deviation.weight = -3.0

        # Feet flat: in steady state we want soles parallel to the
        # ground, but during a push recovery the policy must be allowed
        # to plantarflex / dorsiflex transiently. ``std=0.20`` widens
        # the basin so a recovery toe-down (~11° tilt) drops the reward
        # only to ~0.37 of max instead of essentially zero. Weight
        # stays meaningful but not dominant.
        self.rewards.foot_flat.weight = 1.5
        self.rewards.foot_flat.params["std"] = 0.20
