"""Robust-balance ("stand under push") experiment for Bebop V2.

Same task as :class:`BebopV2FlatBalanceCfg` (zero velocity commands), but
with external push disturbances enabled so the policy learns push recovery
while standing in place. Warm-start from the balance checkpoint
(``model_4000.pt``) so the policy only has to learn the recovery skill,
not re-discover standing.

## Observation pipeline

Inherits the IMU-based observation pipeline from
:class:`BebopV2BaseEnvCfg`: ``base_ang_vel`` and ``projected_gravity``
flow through the explicit :class:`isaaclab.sensors.ImuCfg` sensor
mounted on ``base_link``. That pipeline is byte-compatible with the
on-robot path through ``firmware/bebop-linux/src/imu.rs`` (BNO085 SH-2
report 0x28 + calibrated gyro report 0x02, both rotated into the body
frame by ``mount_quat_sensor_body``). Both ends use **XYZW**
quaternion order — Isaac Lab 3.0's new default, which already matches
the firmware's ``ImuState`` convention — so the same trained policy
can be exported to ONNX and run on the real robot without an
observation schema mismatch. (The 3.0 migration guide describes a
forthcoming Imu→Pva sensor split; in our installed build ``Imu`` is
still the full-state sensor — see the docstring on
``BebopV2BaseEnvCfg.ObservationsCfg`` for the upgrade path.)

## Training procedure

Two-stage curriculum:

1. **Stand-only** (``Isaac-BebopV2-Flat-v0``): teaches the policy to
   keep the torso upright with the support polygon centred under it.

       /workspace/isaaclab/isaaclab.sh -p train_bebop.py \\
           --task Isaac-BebopV2-Flat-v0 --num_envs 4096 --headless

   Train until ``model_~4000.pt`` (≈45 min on a 5090).

2. **Robust** (this file): adds randomised pushes that periodically
   displace the CoM outside the support polygon, forcing the policy to
   learn stepping recoveries.

       /workspace/isaaclab/isaaclab.sh -p train_bebop.py \\
           --task Isaac-BebopV2-FlatRobust-v0 --num_envs 4096 \\
           --headless \\
           --resume logs/rsl_rl/Isaac-BebopV2-Flat-v0/<balance-run>/model_4000.pt \\
           --reset_action_std 0.4

   ``--reset_action_std`` re-inflates the actor's exploration noise so
   the warm-started policy actually explores stepping motions instead
   of collapsing back onto the pure-balance optimum.

After this stage converges, use the resulting checkpoint as the
warm-start for ``Isaac-BebopV2-Locomotion-v0``.
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
        # Robust-start randomisation: in pure-balance the robot resets
        # near-perfectly upright; here we add a small initial tilt so the
        # policy never gets the luxury of "do nothing because gravity is
        # already pointing down". A ±5° roll/pitch envelope is small
        # enough that the warm-started balance policy can still recover
        # (so the early-training collapse rate stays manageable), but
        # large enough that the stable-action penalty bites only when
        # the policy has genuinely re-centred — not because the episode
        # was reset into a trivial configuration.
        # ------------------------------------------------------------------
        self.events.reset_base.params["pose_range"]["roll"] = (-0.08, 0.08)
        self.events.reset_base.params["pose_range"]["pitch"] = (-0.08, 0.08)
        self.events.reset_base.params["pose_range"]["yaw"] = (-0.5, 0.5)
        self.events.reset_base.params["velocity_range"]["roll"] = (-0.3, 0.3)
        self.events.reset_base.params["velocity_range"]["pitch"] = (-0.3, 0.3)

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
