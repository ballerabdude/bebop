"""Robust-balance ("stand under push") experiment for Bebop V2.

Same task as :class:`BebopV2FlatBalanceCfg` (zero velocity commands), but with
external push disturbances enabled so the policy learns push recovery while
standing in place. Warm-start from the balance checkpoint
(``model_4000.pt``) so the policy only has to learn the recovery skill, not
re-discover standing.

Recommended command::

    python train_bebop.py \
        --task Isaac-BebopV2-FlatRobust-v0 \
        --resume logs/rsl_rl/Isaac-BebopV2-Flat-v0/<balance-run>/model_4000.pt

After this stage converges, use the resulting checkpoint as the warm-start
for ``Isaac-BebopV2-Locomotion-v0``.
"""

import isaaclab.envs.mdp as mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass

from .exp_flat_balance_v2 import BebopV2FlatBalanceCfg


@configclass
class BebopV2FlatBalanceRobustCfg(BebopV2FlatBalanceCfg):
    """Stand-in-place while absorbing random base-velocity pushes."""

    def __post_init__(self):
        super().__post_init__()

        # ------------------------------------------------------------------
        # Recovery freedom: the policy must be allowed to actuate fast and
        # asymmetrically to step out from under a shove. The base cfg's
        # symmetry penalties are tuned for *standing still*; here we keep
        # them as a soft prior toward a symmetric resting pose, but lower
        # them enough that asymmetric stepping is the better local optimum
        # when actually disturbed.
        # ------------------------------------------------------------------
        self.rewards.leg_action_when_stable.weight = -0.5  # was -3.0 in pure balance
        self.rewards.leg_hold_reward.weight = 1.0
        self.rewards.undesired_yaw.weight = -2.0
        self.rewards.action_rate_l2.weight = -0.03  # was -0.05; allow faster leg swings

        # Symmetry: soft prior, not a hard constraint. Lateral abduction
        # gets the smallest penalty so a side-step is cheap.
        self.rewards.hip_abduction_symmetry.weight = -0.3
        self.rewards.femur_symmetry.weight = -0.5
        self.rewards.shin_symmetry.weight = -0.5
        self.rewards.foot_symmetry.weight = -0.3

        # Slightly relax base-height penalty: a recovering robot may briefly
        # squat as it loads a leg.
        self.rewards.base_height.weight = -1.5

        # ------------------------------------------------------------------
        # Push disturbances: tuned to force stepping, not just ankle torque.
        # Lateral (y) push is biased larger because biped support polygons
        # are narrower in y, so a 1.2 m/s lateral shove typically displaces
        # the CoM outside the support polygon and a step becomes mechanically
        # necessary to recover.
        # ------------------------------------------------------------------
        self.events.push_robot = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(2.0, 4.0),
            params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.2, 1.2)}},
        )
