"""Minimal standing-balance experiment for Bebop V2.

Train a policy that keeps the torso upright, then deploy the exported ONNX
to the real robot and debug before adding the next feature. This file is the
single source of truth for the standing task config; add one knob at a time
and validate on hardware after each training run.

Kept because the real robot needs them:
  * IMU (BNO085) — projected gravity and gyro for balance.
  * Joint encoders — ``joint_pos`` / ``joint_vel`` observations.
  * MIT-mode variable impedance — 24-dim action (8 pos + 8 kp + 8 kd), same
    decode as ``firmware/bebop-linux/config/bebop_v2.yaml``.
  * 49-dim observation layout matching ``observation.rs`` (base_ang_vel,
    projected_gravity, joint_pos, joint_vel, last_action, cmd_vel — note
    no base_lin_vel; the robot cannot observe it).
  * DCMotor actuators — explicit PD + torque-speed saturation (not implicit
    PhysX drives, which can stand at zero pose without a policy).
  * IMU sample rate (200 Hz) and action delay (20 ms) matching the real
    loop (``bebop_v2.yaml`` ``rotation_vector_period_ms = 5``).

Reward set:
  * ``alive``               — +1/tick while base_link is above the
                              ground-contact termination threshold.
  * ``torso_pitch``         — asymmetric IMU pitch: reward balancing
                              anywhere in a back-lean *band* (~10°–17°,
                              ``proj_grav[0]`` in ``[-0.29, -0.17]``) as a
                              flat-top plateau, so the torso can settle at
                              a range of pitches instead of one angle;
                              penalize forward pitch (hardware falls on any
                              forward lean).
  * ``joint_pos_limits``    — quadratic penalty for exceeding the URDF
                              joint soft limits.
  * ``joint_effort``        — L2 penalty on applied joint torques, a
                              proxy for motor electrical/thermal power
                              (current ∝ torque). Drives "stay alive with
                              the least effort": a deep crouch costs
                              continuous holding torque and is pushed
                              away, while a low-torque (typically
                              straight-leg) stand is favoured — without
                              mandating any fixed pose. Replaces the old
                              fixed ``leg_straightness`` target.
  * ``joint_motion``        — small L2 penalty on joint velocity to damp
                              high-frequency knee/joint jitter directly.
  * ``track_lin_vel_xy`` /
    ``track_ang_vel_z``     — bounded ``exp(-err²/σ²)`` reward tracking
                              the (zero) ``base_velocity`` command, i.e.
                              "be still." Saturates to 0 (not punitive)
                              during legitimate recovery transients, so
                              it composes with ``alive`` and
                              ``torso_pitch`` instead of fighting them.
  * ``lin_vel_z_l2`` /
    ``ang_vel_xy_l2``       — small L2 penalties on the off-command
                              DOFs (vertical bounce, roll/pitch rate)
                              that should be ~0 in every regime.
  * ``action_rate_l2`` / ``action_l2`` — action smoothness + magnitude.

Terminations:
  * ``imu_pitch_out_of_bounds`` — end episode when ``|pitch| > 20°`` from
                              vertical (``|proj_grav[0]| > sin(20°)``), before
                              the torso hits the ground.
  * ``base_link_ground_contact`` — torso height near floor (fallen).

Reset randomization (this version):
  * Joints sampled uniformly per-joint via
    ``reset_joints_uniform_within_limits`` with
    ``range_fraction = RESET_JOINT_RANGE_FRACTION`` (0.25), i.e. within
    25% of the distance from the default pose to each joint's soft
    limit, preserving the asymmetric knee / hip-abduction shape. This is
    a deliberately tight band around the nominal pose so most episodes
    survive and the policy gets dense standing signal. Set the fraction
    to 1.0 to recover the full-configuration-box behaviour (many
    unrecoverable inits from ``base_link.z = 0.8 m``, diluted signal);
    widen it gradually once the robot reliably stands. Watch
    ``mean_episode_length`` and the ``base_link_ground_contact``
    termination rate when changing it.
  * Base roll ±0.30 rad; pitch biased toward back lean ``(-0.26, +0.10)``
    rad (~−15°..+6°) — forward pitch samples are rare because hardware
    cannot recover from forward tilt, and the deep edge leaves a ~5°
    margin below the ±20° fall limit so resets don't terminate on the
    cliff. Yaw full ±π.
  * Initial angular velocity perturbed by ±0.3 rad/s on all three axes.

Deliberately off (add back one at a time after hardware validation):
  * Mid-episode pushes, observation noise, contact sensors, stepping rewards.
  * ``flat_orientation``, ``base_height``, symmetry penalties beyond the
    asymmetric pitch term above. (Fixed straight-leg shaping via
    ``joint_deviation_l1`` was removed in favour of the ``joint_effort``
    power objective above — the legs may now bend if it is genuinely
    cheaper.)

Deployment checklist (every run):
  1. Export ONNX from the training run.
  2. Confirm ``pos_scale`` (0.5), ``max_pos_step_per_tick`` (0.020), and
     ``POLICY_*`` clamps match ``bebop_v2.yaml``.
  3. Pose the robot near the trained init (joints ≈ 0, torso ~15–20° back)
     before RunPolicy.
  4. Log raw actions + decoded targets on hardware; compare to sim play mode.
"""

import math

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
import isaaclab.envs.mdp as mdp

from isaaclab.actuators import DCMotorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as TermTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ImuCfg
from isaaclab.utils import configclass

from ..envs.bebop_v2_actions import VariableImpedanceJointActionCfg
from ..envs.bebop_v2_events import reset_joints_uniform_within_limits
from ..envs.bebop_v2_rewards import torso_pitch_asymmetric_reward
from ..envs.bebop_v2_terminations import base_link_on_ground, imu_pitch_out_of_bounds


# IMU pitch convention (body FLU, same as obs / firmware):
#   proj_grav[0] < 0  => torso pitched back (stable band on hardware)
#   proj_grav[0] > 0  => torso pitched forward (falls)
PITCH_FALL_LIMIT_DEG = 20.0
PITCH_FALL_LIMIT_GX = math.sin(math.radians(PITCH_FALL_LIMIT_DEG))

# Back-lean *band* the torso is rewarded to balance anywhere within (a
# flat-top plateau, not a single target — so the policy can settle at a
# range of pitches instead of collapsing onto one angle in playback).
# proj_grav[0] = -sin(pitch): more negative => deeper back lean. The band
# is centered on the known-good 17° back lean and extended toward upright
# (down to 10°). Both edges stay inside the ±20° imu_pitch_out_of_bounds
# termination so the policy is never rewarded for sitting on the fall
# cliff (~3° margin at the deep edge). Widen the band for a larger range
# of stable angles; if you push the deep edge past ~17° also raise
# PITCH_FALL_LIMIT_DEG (and validate the deep lean on hardware first).
PITCH_BAND_DEEP_DEG = 17.0
PITCH_BAND_SHALLOW_DEG = 10.0
PITCH_BAND_DEEP_GX = -math.sin(math.radians(PITCH_BAND_DEEP_DEG))
PITCH_BAND_SHALLOW_GX = -math.sin(math.radians(PITCH_BAND_SHALLOW_DEG))

# Initial-pose back-lean range (radians, for reset_root_state_uniform's
# pose_range["pitch"], which is an Euler angle in rad — NOT a projected
# gravity component). Previously this used -PITCH_FALL_LIMIT_GX (a sine,
# ~0.342) as if it were radians, spawning episodes at ~-19.6°, i.e. right
# on the ±20° imu_pitch_out_of_bounds cliff; combined with the init
# angular velocity those episodes terminated almost immediately and
# inflated the pitch-out termination rate without measuring real
# balance. We now seed within the recoverable back-lean band with a
# margin below the fall limit. Deep init (~15°) sits just inside the
# 10–17° torso_pitch reward band; the small forward edge (+6°) makes the
# policy practice leaning back into the band.
PITCH_INIT_BACK_RAD = -math.radians(15.0)
PITCH_INIT_FWD_RAD = math.radians(6.0)

# Fraction of each joint's soft range (measured from the default pose to
# each limit) used when sampling reset poses. 1.0 = full configuration
# box (many unrecoverable inits, diluted signal); a small value like 0.25
# keeps resets near the nominal pose so most episodes survive and the
# policy gets dense standing signal. Widen toward 1.0 once it stands.
RESET_JOINT_RANGE_FRACTION = 0.25


# Joint order must match firmware/bebop-linux/src/observation.rs::JOINT_NAMES.
JOINT_NAMES_ALL = [
    "hip_flexion_left_joint",
    "hip_flexion_right_joint",
    "hip_abduction_left_joint",
    "hip_abduction_right_joint",
    "knee_flexion_left_joint",
    "knee_flexion_right_joint",
    "foot_left_joint",
    "foot_right_joint",
]

# Per-joint kp/kd clamps for the 24-dim MIT action. Must mirror
# POLICY_KP_MIN/MAX and POLICY_KD_MIN/MAX in bebop_v2_base_cfg.py and
# policy_gain_clamps in firmware/bebop-linux/config/bebop_v2.yaml.
POLICY_KP_MIN = [20.0, 20.0, 40.0, 40.0, 30.0, 30.0, 30.0, 30.0]
POLICY_KP_MAX = [100.0, 100.0, 300.0, 300.0, 250.0, 250.0, 250.0, 250.0]
POLICY_KD_MIN = [1.5, 1.5, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0]
POLICY_KD_MAX = [5.0, 5.0, 8.0, 8.0, 8.0, 8.0, 5.0, 5.0]

# Robstride motor friction / armature estimates and T-N curve corners.
JOINT_FRICTION_HIP_FLEX = 0.5
JOINT_FRICTION_HIP_ABD = 0.3
JOINT_FRICTION_KNEE_FLEX = 0.5
JOINT_FRICTION_FOOT = 0.1

JOINT_ARMATURE_HIP_FLEX = 0.025
JOINT_ARMATURE_HIP_ABD = 0.012
JOINT_ARMATURE_KNEE_FLEX = 0.025
JOINT_ARMATURE_FOOT = 0.004

MOTOR_STALL_TORQUE_RS04 = 120.0
MOTOR_STALL_TORQUE_RS03 = 60.0
MOTOR_STALL_TORQUE_RS02 = 17.0

MOTOR_NOLOAD_VEL_RS04 = 20.9
MOTOR_NOLOAD_VEL_RS03 = 20.4
MOTOR_NOLOAD_VEL_RS02 = 42.9


def _midpoint(lo: list[float], hi: list[float]) -> list[float]:
    return [0.5 * (a + b) for a, b in zip(lo, hi)]


_KP_MID = _midpoint(POLICY_KP_MIN, POLICY_KP_MAX)
_KD_MID = _midpoint(POLICY_KD_MIN, POLICY_KD_MAX)


BEBOP_V2_STANDING_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/workspace/bebop_bot/sim/usd/bebopv2/bebopv2.usda",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.8),
        joint_pos={joint_name: 0.0 for joint_name in JOINT_NAMES_ALL},
        joint_vel={joint_name: 0.0 for joint_name in JOINT_NAMES_ALL},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "hip_flexion": DCMotorCfg(
            joint_names_expr=[
                "hip_flexion_left_joint",
                "hip_flexion_right_joint",
            ],
            saturation_effort=MOTOR_STALL_TORQUE_RS04,
            effort_limit_sim=84.0,
            velocity_limit_sim=12.0,
            velocity_limit=MOTOR_NOLOAD_VEL_RS04,
            stiffness=_KP_MID[0],
            damping=_KD_MID[0],
            armature=JOINT_ARMATURE_HIP_FLEX,
            friction=JOINT_FRICTION_HIP_FLEX,
        ),
        "hip_abduction": DCMotorCfg(
            joint_names_expr=[
                "hip_abduction_left_joint",
                "hip_abduction_right_joint",
            ],
            saturation_effort=MOTOR_STALL_TORQUE_RS03,
            effort_limit_sim=42.0,
            velocity_limit_sim=12.0,
            velocity_limit=MOTOR_NOLOAD_VEL_RS03,
            stiffness=_KP_MID[2],
            damping=_KD_MID[2],
            armature=JOINT_ARMATURE_HIP_ABD,
            friction=JOINT_FRICTION_HIP_ABD,
        ),
        "knee_flexion": DCMotorCfg(
            joint_names_expr=[
                "knee_flexion_left_joint",
                "knee_flexion_right_joint",
            ],
            saturation_effort=MOTOR_STALL_TORQUE_RS04,
            effort_limit_sim=84.0,
            velocity_limit_sim=12.0,
            velocity_limit=MOTOR_NOLOAD_VEL_RS04,
            stiffness=_KP_MID[4],
            damping=_KD_MID[4],
            armature=JOINT_ARMATURE_KNEE_FLEX,
            friction=JOINT_FRICTION_KNEE_FLEX,
        ),
        "foot": DCMotorCfg(
            joint_names_expr=["foot_left_joint", "foot_right_joint"],
            saturation_effort=MOTOR_STALL_TORQUE_RS02,
            effort_limit_sim=17.0,
            velocity_limit_sim=20.0,
            velocity_limit=MOTOR_NOLOAD_VEL_RS02,
            stiffness=_KP_MID[6],
            damping=_KD_MID[6],
            armature=JOINT_ARMATURE_FOOT,
            friction=JOINT_FRICTION_FOOT,
        ),
    },
)


@configclass
class ActionsCfg:
    joint_pos = VariableImpedanceJointActionCfg(
        asset_name="robot",
        joint_names=JOINT_NAMES_ALL,
        pos_scale=0.5,
        use_default_offset=True,
        max_pos_step_per_tick=0.020,
        action_delay_steps=2,
        kp_min=POLICY_KP_MIN,
        kp_max=POLICY_KP_MAX,
        kd_min=POLICY_KD_MIN,
        kd_max=POLICY_KD_MAX,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # NOTE: base_lin_vel is intentionally NOT an observation. Bebop V2
        # has no wheel odometry or VIO estimator, so the real robot can only
        # ever feed zeros for it (see firmware policy_runner.rs). Training
        # against the sim's privileged ground-truth base velocity created a
        # 3-dim sim-to-real gap that destabilized deployment. The standing
        # task is at zero command, so base linear velocity carries no policy
        # signal anyway — it lives only in the (privileged) tracking rewards.
        base_ang_vel = ObsTerm(
            func=mdp.imu_ang_vel,
            params={"asset_cfg": SceneEntityCfg("imu")},
        )
        projected_gravity = ObsTerm(
            func=mdp.imu_projected_gravity,
            params={"asset_cfg": SceneEntityCfg("imu")},
        )
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )

    def __post_init__(self):
        self.policy = self.PolicyCfg()


@configclass
class EventCfg:
    # Per-joint uniform sampling across each joint's soft limits. With
    # soft_joint_pos_limit_factor = 0.9, this covers ~90% of the URDF
    # range for every joint, including the asymmetric knee
    # (-1.41, +0.71 rad) and hip-abduction (-0.31, +0.16 rad on the
    # right, mirror on the left) ranges that a symmetric offset can't
    # represent. The VariableImpedanceJointAction's reset() reads the
    # post-event joint pose into the slew tracker, so whatever we
    # sample here becomes the policy's true initial condition (it
    # won't get yanked back to zero on tick 1).
    reset_joints = EventTerm(
        func=reset_joints_uniform_within_limits,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES_ALL),
            "velocity_range": (-0.5, 0.5),
            "range_fraction": RESET_JOINT_RANGE_FRACTION,
        },
    )
    # Wide base-orientation randomization. roll / pitch directly enter
    # projected_gravity, so the policy actually sees these. yaw is
    # invariant to projected_gravity and shows up only via imu_ang_vel
    # over the recovery transient — at zero command the robot is
    # yaw-symmetric, so full ±pi yaw is a free robustness probe.
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (-0.30, 0.30),
                # Bias toward back lean; rare forward samples (hardware
                # cannot recover from forward pitch). Both edges leave a
                # margin below the ±20° fall limit so resets measure
                # balance rather than terminating on the cliff.
                "pitch": (PITCH_INIT_BACK_RAD, PITCH_INIT_FWD_RAD),
                "yaw": (-math.pi, math.pi),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (-0.3, 0.3),
                "pitch": (-0.3, 0.3),
                "yaw": (-0.3, 0.3),
            },
        },
    )


@configclass
class RewardsCfg:
    """Standing balance rewards aligned with hardware pitch behaviour."""

    alive = RewTerm(func=mdp.is_alive, weight=1.0)

    # Hardware-stable band: torso ~15–20° pitched back. Forward pitch
    # is penalized inside the reward; ``imu_pitch_out_of_bounds`` ends
    # the episode at ±20° from vertical.
    torso_pitch = RewTerm(
        func=torso_pitch_asymmetric_reward,
        weight=0.5,
        params={
            "band_gx_min": PITCH_BAND_DEEP_GX,
            "band_gx_max": PITCH_BAND_SHALLOW_GX,
            "edge_std": 0.12,
            "roll_std": 0.15,
            "forward_penalty_gain": 5.0,
            "forward_deadband": 0.0,
        },
    )

    joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)

    # Least-effort posture, via a power objective instead of a fixed
    # straight-leg target.
    #
    # The old ``leg_straightness`` term pinned the knees and hip-abduction
    # joints to their zero default with an L1 penalty. The intent was
    # right (straight legs stack the load through the joint, so they hold
    # the body with little torque), but mandating an exact pose fights
    # the balance controller: when the genuinely cheapest standing
    # configuration is a hair off zero, the policy is torn between the
    # posture target and balance and chatters around it — a plausible
    # source of the residual knee jitter.
    #
    # Replace the hand-picked pose with the physical quantity we actually
    # care about: the effort needed to stay alive. ``joint_torques_l2``
    # penalizes the sum of squared applied joint torques — a proxy for
    # motor electrical/thermal power (current ∝ torque, I²R heating).
    # The policy is now free to bend the knee if that is genuinely
    # cheaper, but a deep crouch (knee/abduction motors fighting gravity
    # continuously) is expensive and gets pushed away on its own. If the
    # straight-leg rationale holds, the policy rediscovers straight legs
    # because they minimize holding torque — but it is no longer forced.
    #
    # Weight is small because torques are tens of N·m (τ² is in the
    # hundreds–thousands per tick summed over 8 joints). Start at -1e-5
    # (≈ -0.02..-0.05/tick at a quiet stand, comparable to the action
    # regularizers) and raise toward -2.5e-5 if play mode still shows a
    # high-effort crouch; lower it if the policy goes limp (drops kp to
    # dodge the penalty) and sags.
    joint_effort = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)

    # Direct jitter damping: penalize squared joint velocity. A buzzing
    # knee carries velocity even when its mean position is steady, so
    # this term taxes the chatter the torque/action-rate penalties may
    # miss. Kept small so it damps high-frequency motion without
    # over-damping a legitimate balance-recovery sweep.
    joint_motion = RewTerm(func=mdp.joint_vel_l2, weight=-2.5e-4)

    # Zero-command tracking → "be still" reward. CommandsCfg.base_velocity
    # is pinned to (0, 0, 0) for this standing task, so the exponential
    # tracking rewards reduce to exp(-||v||² / std²) on base linear-xy
    # and base angular-z. Bounded in [0, 1] per tick, which is why we use
    # them as the *carrot* instead of an unbounded L2 velocity penalty:
    # during a legitimate balance-recovery transient the reward just
    # saturates toward 0, it does not actively fight `alive` or
    # `torso_pitch` the way a quadratic stick would.
    #
    # std picked so the reward decays to ~1/e at typical drift speeds
    # but is still close to 1 inside the noise floor of a quiet stand.
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.0,
        params={"std": 0.25, "command_name": "base_velocity"},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.5,
        params={"std": 0.5, "command_name": "base_velocity"},
    )

    # Off-command DOFs: vertical velocity and roll/pitch rates are not in
    # the command at all, and a healthy stand has them ~0 in every
    # regime. Small L2 penalties act as regularizers here — same role as
    # action_rate_l2 — without competing with the recovery reward.
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)

    # Smoothness regularizer: penalize tick-to-tick change in any of the
    # 24 raw action channels. Catches the high-frequency *flipping* mode
    # (e.g. foot kp 250 -> 107 -> 250 -> 250 -> 250 across consecutive
    # ticks). Does NOT catch a policy that camps at the saturation rails
    # without flipping — that's what action_l2 below is for.
    #
    # Bumped -0.02 -> -0.05 to kill the knee/gain jitter seen in playback.
    # This term is the main lever on the variable-impedance *gain*
    # channels: a fluttering knee kp (30<->250) modulates joint torque
    # even when the position target is steady, which reads as visible
    # knee jitter. Raising the action-rate cost disciplines all 24
    # channels — position and gains alike. If the legs start to feel
    # over-damped / sluggish to recover, back off toward -0.03.
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.05)

    # Physical smoothness regularizer at the joints (not the policy
    # output). Penalizes joint acceleration directly, so it catches
    # high-frequency chatter that survives a smooth raw-action stream —
    # e.g. when steady-looking kp/kd commands still drive a buzzing
    # torque response. Weight is tiny because accelerations are large in
    # rad/s²; -2.5e-7 is the standard Isaac Lab locomotion value and
    # contributes only a small fraction per tick at a quiet stand.
    joint_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)

    # Magnitude regularizer: penalize raw action magnitude directly.
    # Each of the 24 channels is clipped to [-1, 1], so a channel parked
    # at ±1 contributes 1.0/tick to the L2 sum; a policy fully saturated
    # across all 24 channels pays |w| * 24 = -0.24/tick — comparable in
    # magnitude to the +1/tick `alive` bonus, so the optimizer is
    # strongly pushed to leave the rails on every channel that doesn't
    # earn its keep. With this penalty the policy is pushed toward
    # raw ≈ 0, which decodes to (default_joint_pos, kp_mid, kd_mid) —
    # a healthy quiet-standing prior.
    action_l2 = RewTerm(func=mdp.action_l2, weight=-0.01)


@configclass
class TerminationsCfg:
    time_out = TermTerm(func=mdp.time_out, time_out=True)
    imu_pitch_out_of_bounds = TermTerm(
        func=imu_pitch_out_of_bounds,
        params={
            "pitch_forward_gx_max": PITCH_FALL_LIMIT_GX,
            "pitch_back_gx_min": -PITCH_FALL_LIMIT_GX,
        },
    )
    base_link_ground_contact = TermTerm(
        func=base_link_on_ground,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "ground_height_threshold": 0.30,
        },
    )


@configclass
class CommandsCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 12.0),
        debug_vis=False,
        rel_standing_envs=1.0,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
        ),
    )


@configclass
class BebopV2StandingCfg(ManagerBasedRLEnvCfg):
    decimation = 2
    episode_length_s = 20.0

    scene = InteractiveSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=True)
    observations = ObservationsCfg()
    actions = ActionsCfg()
    commands = CommandsCfg()
    rewards = RewardsCfg()
    terminations = TerminationsCfg()
    events = EventCfg()

    def __post_init__(self):
        self.viewer.eye = [2.5, 2.5, 2.5]
        self.viewer.lookat = [0.0, 0.0, 0.0]

        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = True

        self.scene.robot = BEBOP_V2_STANDING_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.scene.terrain = terrain_gen.TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="plane",
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="average",
                restitution_combine_mode="average",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
        )

        self.scene.light = AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
        )

        self.scene.imu = ImuCfg(
            prim_path="{ENV_REGEX_NS}/Robot/Geometry/base_link",
            update_period=0.005,
            debug_vis=False,
            offset=ImuCfg.OffsetCfg(
                pos=(0.0, 0.0, 0.0),
                rot=(0.0, 0.0, 0.0, 1.0),
            ),
        )
