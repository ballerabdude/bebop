"""Minimal quiet-standing experiment for Bebop V2.

Trains the simplest standing policy (zero velocity command, upright torso),
deploys the exported ONNX to the real robot, and debugs before adding the
next feature. Add one knob at a time and validate on hardware after each run.

Joint order and per-joint gain clamps MUST mirror
``firmware/bebop-linux/config/bebop_v2.yaml`` and
``firmware/bebop-linux/src/observation.rs::JOINT_NAMES``.
"""

import math

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
import isaaclab.envs.mdp as mdp

from isaaclab.actuators import DCMotorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as TermTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ImuCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from ..envs.bebop_v2_actions import VariableImpedanceJointActionCfg
from ..envs.bebop_v2_events import (
    push_magnitude_curriculum,
    randomize_torso_com_uniform,
    reset_joints_uniform_within_limits,
    torso_com_curriculum,
)
from ..envs.bebop_v2_rewards import (
    action_gain_rate_l2,
    action_position_rate_l2,
    bilateral_joint_symmetry_l2,
    com_over_support_reward,
    feet_flat_orientation_l2,
    joint_deviation_l1_balance_gated,
    joint_vel_l2,
    torso_pitch_asymmetric_reward,
)
from ..envs.bebop_v2_terminations import (
    base_link_on_ground,
    imu_pitch_out_of_bounds,
    imu_roll_out_of_bounds,
)


# Symmetric ±30° tilt envelope. Wider than a strict stand so small drift doesn't
# die on the cliff; the slower base_link height check is the true fall terminator.
PITCH_FALL_LIMIT_DEG = 30.0
PITCH_FALL_LIMIT_GX = math.sin(math.radians(PITCH_FALL_LIMIT_DEG))
ROLL_FALL_LIMIT_DEG = 30.0
ROLL_FALL_LIMIT_GY = math.sin(math.radians(ROLL_FALL_LIMIT_DEG))

# Initial torso pitch (rad) and pitch-rate (rad/s) on reset. Symmetric ±15° /
# ±0.6 rad/s trains pitch-recovery authority (restored Jul 10 2026 after the
# no-perturbation diagnostic run). Recovery skill comes from these reset
# perturbations (and later the Push variant), NOT from extra reward terms.
PITCH_INIT_RAD = math.radians(15.0)
PITCH_RATE_INIT_RAD_S = 0.6

# Balance band (g_x) for the torso: near-upright with a slight FORWARD bias.
#
# Jul 17 2026 geometry audit (URDF link masses + foot sole STL), replacing
# the 8-12 deg BACK-lean band that preceded it:
#   * the ankle axis sits only 23 mm ahead of the heel edge of the sole
#     (the toe edge is +140 mm ahead — a toe-heavy, human-like foot);
#   * at the upright zero pose the whole-robot CoM (~14 kg, torso 6.7 kg)
#     already sits 48 mm BEHIND the ankle axis = 25 mm behind the heel
#     edge: straight-leg upright standing is statically IMPOSSIBLE. The
#     robot must flex the hips ~5 deg (slant the legs back, moving the
#     ankle under the CoM) and/or pitch the torso ~5 deg forward;
#   * the old 8-12 deg back-lean band put the CoM 131-172 mm behind the
#     ankle — 6-7x the heel margin. No posture in that band is statically
#     sustainable. Capture 20260717_213006 (run 2026-07-17_13-07-06
#     ckpt-4999) showed the consequence on hardware: the policy slid to
#     the feasible posture NEAREST the unreachable band — g_x mean -0.027
#     with hips flexed ~0.06 rad, CoM within a few mm of the heel edge —
#     where any backward wobble crosses the heel and the robot tips back
#     irrecoverably ("unstable once it adjusts pitch, has to be held").
#     Fighting the impossible target all episode also doubled its chatter:
#     24.4% slew-exceedance vs 6.3% for the Jul 15 quiet stand.
#
# The band therefore straddles the statically sustainable posture:
# near-upright (the hip anchor below is relaxed so the policy can make
# the ~5 deg geometric compensation), biased slightly FORWARD because the
# toe side carries the support margin (140 mm) while the heel side
# (23 mm) is the fall direction. g_x = -sin(pitch); g_x > 0 = forward.
BALANCE_BAND_GX_MIN = -0.02   # 1.1 deg back (heel side — stay off it)
BALANCE_BAND_GX_MAX = +0.05   # 2.9 deg forward (toe side — the safe side)

# Asymmetric Gaussian shoulders outside the band. The BACK (below) shoulder
# is wide (0.15 -> restoring slope reaches ~17 deg back) so a backward-
# tipped robot still has gradient home; the FORWARD shoulder is tighter
# (0.08) — forward excursions are self-limiting over the 140 mm toe margin.
BALANCE_EDGE_STD_BELOW = 0.15
BALANCE_EDGE_STD_ABOVE = 0.08
# Quadratic tails beyond the band, both directions (gain on
# relu(overshoot)^2, deadband at the band edge). The Gaussian shoulders'
# gradient dies ~2 std out; the tails keep a nonzero restoring gradient at
# large tilt — the learnable signal for "recover once the pitch starts to
# run away", which the pre-Jul-17 reward lacked (flat alive + termination
# cliff only). Mild gain: far-field direction, not a wall.
BALANCE_FWD_PENALTY_GAIN = 1.0
BALANCE_BWD_PENALTY_GAIN = 1.0
# Roll tolerance inside the band (g_y; ~0.15 ≈ 8.6° at 1/e).
BALANCE_ROLL_STD = 0.15

# Center of the balance band in g_x units — the balance target the movement
# penalties gate on. ~= +0.015 (~1 deg forward lean), the midpoint.
BALANCE_BAND_GX_CENTER = 0.5 * (BALANCE_BAND_GX_MIN + BALANCE_BAND_GX_MAX)
# Width of the balance gate Gaussian (in g_x units; ~0.10 ≈ 5.7° at 1/e).
# Tightened 0.15 -> 0.10 (Jul 15 2026) after capture 20260715_122500 showed
# the 0.15 gate opened too early (at moderate tilt where the policy should
# still be smooth), enabling a flailing limit cycle. At 0.10 the gate only
# relaxes the movement penalties at genuinely dangerous tilt (>6°), keeping
# the anti-chatter pressure on through the normal recovery regime.
BALANCE_GATE_STD = 0.10
# Floor for the balance gate (0..1). Prevents the movement penalties from
# vanishing entirely at tilt, which the policy otherwise exploits by
# manufacturing tilt to unlock chatter (capture 20260715_122500: vel_std
# 0.69-1.34, slew-exceedance 56.9%). At gate_floor=0.2 the -10.0 position_rate
# still contributes -2.0·rate at full tilt — enough to keep recovery smooth.
BALANCE_GATE_FLOOR = 0.20

# Bilateral joint pairs for the symmetry reward. Every L/R pair on this robot
# is sign-MIRRORED in the URDF (right-side flexion joints use a flipped -Y
# axis; hip_abduction uses the same +X axis but mirrored limits). A symmetric
# stance therefore reads q_L = -q_R for ALL four pairs, and the asymmetry
# residual is the SUM (q_L + q_R), not the difference. See the per-pair axis
# table in ``bilateral_joint_symmetry_l2``'s docstring and the verification
# against ``ros2/src/bebopv2_description/urdf/bebopv2.urdf``.
BILATERAL_SYMMETRY_PAIRS = [
    ("hip_flexion_left_joint", "hip_flexion_right_joint"),
    ("hip_abduction_left_joint", "hip_abduction_right_joint"),
    ("knee_flexion_left_joint", "knee_flexion_right_joint"),
    ("foot_left_joint", "foot_right_joint"),
]
# Weight on the bilateral-symmetry penalty (Jul 15 2026). Capture
# 20260715_214224 of run 2026-07-15_12-35-39 ckpt-4000 showed severe
# bilateral asymmetry on hardware despite the (correct, sum-convention)
# symmetry term existing in the code: hip_flexion L+R=-0.82, foot L+R=+0.49
# (both far from the symmetric 0). Root cause: the symmetry term was NOT
# in the active RewardsCfg — the only symmetry enforcement was the weak
# -0.3 all-joint joint_pos_anchor, whose gradient is diluted 8 ways and
# cannot enforce per-pair mirroring. This dedicated term is heavily
# weighted (-2.0) and balance-gated so it fires at full strength when the
# robot is balanced (the stance we want symmetric) and relaxes toward
# gate_floor when tilted (the policy is free to use an asymmetric catch).
# Why -2.0: at the capture's mean asymmetry (Σ(q_L+q_R)² ≈ 0.82²+0.49²
# ≈ 0.91 for the two worst pairs alone) this contributes ≈ -1.8/tick,
# comparable to the alive reward (+1.0) — finally enough to make an
# asymmetric stance expensive relative to the survival gradient. The
# gate_floor=0.2 keeps a -0.36/tick floor at full tilt so the policy
# can't manufacture tilt to suppress the constraint.
BILATERAL_SYMMETRY_WEIGHT = -2.0

# Fraction of each joint's soft range used when sampling reset poses. A small
# value keeps resets near the nominal pose so most episodes survive; widen
# toward 1.0 once the robot stands reliably.
RESET_JOINT_RANGE_FRACTION = 0.25

# Mid-episode push envelope for the reactive-recovery variant (m/s).
PUSH_VELOCITY_RANGE = {"x": (-0.4, 0.4), "y": (-0.4, 0.4)}

# Target standing base_link height (m). UNVERIFIED — measure the true settled
# height before enabling the optional base_height_l2 reward.
BASE_HEIGHT_TARGET = 0.65

# Torso CoM robustness range (m). Horizontal ±2 cm only; curriculum ramps from
# 25% of this range to full over 100k control steps.
TORSO_COM_RANGE = {
    "x": (-0.02, 0.02),
    "y": (-0.02, 0.02),
    "z": (0.0, 0.0),
}
TORSO_COM_START_FRACTION = 0.25
TORSO_COM_CURRICULUM_STEPS = 100_000

# Observation noise calibrated to the real hardware noise floor (capture
# 20260714_031815, DIAL_IN mode, motors armed, robot still). The previous
# values were 4-100x too high (worst on projected_gravity and joint_pos),
# which over-regularized the policy — it learned to discount the very tilt
# signal it needs on hardware. Projected_gravity is the policy's primary
# balance cue; over-noising it trained a half-blind-to-tilt policy that then
# leaned 7.7 deg sideways on hardware without correcting.
#
# Real still-state stds (rad or rad/s):
#   gyro wx/wy         0.0023-0.0028  -> sim   0.01   (4x high, mild)
#   joint_pos          0.0001-0.0007  -> sim   0.01   (15-100x high, severe)
#   joint_vel          0.036-0.072    -> sim   0.12   (1.7-3.3x high, mild)
#   proj_grav gx/gy    0.0003-0.0004  -> sim   0.02   (50-65x high, severe)
#
# Calibrated values: 2-3x the still-state std to cover the policy-active
# vibration regime (capture 20260714_030338 RUN_POLICY: gyro std ~0.2 rad/s,
# ~50x the still state, so the active floor is what the policy must weather).
# The bigger numbers reflect what the sensors actually look like when the
# robot is balancing, not the desk-still floor.
GYRO_NOISE_RAD_S = 0.006
JOINT_VEL_NOISE_RAD_S = 0.10
PROJ_GRAV_NOISE = 0.005
JOINT_POS_NOISE_RAD = 0.003

# Per-episode randomized action transport delay in 100 Hz policy ticks.
# Training across 10-40 ms forces a latency-robust (more damped) policy.
ACTION_DELAY_RANGE = (1, 4)

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

# Per-joint kp/kd clamps for the 24-dim MIT action. Must mirror the per-joint
# policy_gain_clamps in firmware/bebop-linux/config/bebop_v2.yaml.
# order: [hipflexL, hipflexR, hipabdL, hipabdR, kneeL, kneeR, footL, footR]
# Foot (ankle) kp_min raised 100 -> 212.5 (Jul 9 2026): capture of run
# 2026-07-08_03-55-34 ckpt-4500 showed foot kp parked at ~111, right at the
# old floor — the RS02 (~17 N·m, weakest joint) was being commanded the
# softest setting available and the ankle got pushed around when the robot
# tried to stand. Raising the floor to 75% of the band (100 + 0.75*(250-100)
# = 212.5) forces the policy to keep ankle authority high; the policy can
# still tune kp within [212.5, 250] but can't go soft. Mirror in firmware
# bebop_v2.yaml foot_left_joint / foot_right_joint.
POLICY_KP_MIN = [20.0, 20.0, 40.0, 40.0, 30.0, 30.0, 212.5, 212.5]
POLICY_KP_MAX = [120.0, 120.0, 150.0, 150.0, 150.0, 150.0, 250.0, 250.0]
POLICY_KD_MIN = [2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]
POLICY_KD_MAX = [5.0, 5.0, 8.0, 8.0, 8.0, 8.0, 5.0, 5.0]

# Robstride motor friction / armature (sysid-measured on right-side joints) and
# datasheet stall torque / no-load speed (re-measure with a brake fixture before
# changing the saturation_effort / velocity_limit values).
JOINT_FRICTION_HIP_FLEX = 0.567
JOINT_FRICTION_HIP_ABD = 0.373
JOINT_FRICTION_KNEE_FLEX = 0.633
JOINT_FRICTION_FOOT = 0.159

JOINT_ARMATURE_HIP_FLEX = 0.0310
JOINT_ARMATURE_HIP_ABD = 0.0114
JOINT_ARMATURE_KNEE_FLEX = 0.0312
JOINT_ARMATURE_FOOT = 0.0038

MOTOR_STALL_TORQUE_RS04 = 120.0
MOTOR_STALL_TORQUE_RS03 = 60.0
MOTOR_STALL_TORQUE_RS02 = 17.0

# RS02 continuous (rated) torque — 6 Nm @ 100 rpm per datasheet. Used as the
# sim effort limit for the ankle (Jul 10 2026): with the limit at the 17 Nm
# stall value the sim ankle could hold peak torque INDEFINITELY, so training
# converged on an ankle-supported lean (capture 20260710_033855 showed ~37 Nm
# commanded at the right foot — 2x stall). The real RS02 saturates and gets
# backdriven, hence the sim-to-real gap. Capping sim at the continuous rating
# forces the policy to balance with the hips over the foot instead of leaning
# on the ankle.
MOTOR_CONT_TORQUE_RS02 = 6.0

MOTOR_NOLOAD_VEL_RS04 = 20.9
MOTOR_NOLOAD_VEL_RS03 = 20.4
MOTOR_NOLOAD_VEL_RS02 = 42.9


def _midpoint(lo: list[float], hi: list[float]) -> list[float]:
    return [0.5 * (a + b) for a, b in zip(lo, hi)]


_Kp_MID = _midpoint(POLICY_KP_MIN, POLICY_KP_MAX)
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
            stiffness=_Kp_MID[0],
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
            stiffness=_Kp_MID[2],
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
            stiffness=_Kp_MID[4],
            damping=_KD_MID[4],
            armature=JOINT_ARMATURE_KNEE_FLEX,
            friction=JOINT_FRICTION_KNEE_FLEX,
        ),
        "foot": DCMotorCfg(
            joint_names_expr=["foot_left_joint", "foot_right_joint"],
            saturation_effort=MOTOR_STALL_TORQUE_RS02,
            # Continuous rating, NOT stall — see MOTOR_CONT_TORQUE_RS02.
            effort_limit_sim=MOTOR_CONT_TORQUE_RS02,
            velocity_limit_sim=20.0,
            velocity_limit=MOTOR_NOLOAD_VEL_RS02,
            stiffness=_Kp_MID[6],
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
        # Newton resolves joints right-before-left; firmware is left-before-right.
        # preserve_order=True keeps the action I/O in firmware order.
        preserve_order=True,
        pos_scale=0.5,
        use_default_offset=True,
        max_pos_step_per_tick=0.020,
        action_delay_steps=2,
        action_delay_range=ACTION_DELAY_RANGE,
        kp_min=POLICY_KP_MIN,
        kp_max=POLICY_KP_MAX,
        kd_min=POLICY_KD_MIN,
        kd_max=POLICY_KD_MAX,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # base_lin_vel is intentionally NOT observed: the real robot has no
        # odometry/VIO and can only feed zeros, so training against the sim's
        # privileged ground-truth base velocity created a sim-to-real gap.
        base_ang_vel = ObsTerm(
            func=mdp.imu_ang_vel,
            params={"asset_cfg": SceneEntityCfg("imu")},
            noise=Unoise(n_min=-GYRO_NOISE_RAD_S, n_max=GYRO_NOISE_RAD_S),
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-PROJ_GRAV_NOISE, n_max=PROJ_GRAV_NOISE),
        )
        # preserve_order=True keeps joint sensing in lock-step with the action
        # term and firmware obs builder (left-before-right).
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=JOINT_NAMES_ALL, preserve_order=True
                )
            },
            noise=Unoise(n_min=-JOINT_POS_NOISE_RAD, n_max=JOINT_POS_NOISE_RAD),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=JOINT_NAMES_ALL, preserve_order=True
                )
            },
            noise=Unoise(n_min=-JOINT_VEL_NOISE_RAD_S, n_max=JOINT_VEL_NOISE_RAD_S),
        )
        actions = ObsTerm(func=mdp.last_action)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )

        def __post_init__(self):
            self.enable_corruption = True

    def __post_init__(self):
        self.policy = self.PolicyCfg()


@configclass
class EventCfg:
    reset_joints = EventTerm(
        func=reset_joints_uniform_within_limits,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES_ALL),
            "velocity_range": (-0.5, 0.5),
            "range_fraction": RESET_JOINT_RANGE_FRACTION,
        },
    )
    # Only torso pitch (and pitch rate) are perturbed at reset — the real
    # failure mode is the heavy torso falling fore/aft, not a sideways drop.
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (-PITCH_INIT_RAD, PITCH_INIT_RAD),
                "yaw": (0.0, 0.0),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (-PITCH_RATE_INIT_RAD_S, PITCH_RATE_INIT_RAD_S),
                "yaw": (0.0, 0.0),
            },
        },
    )

    # Actuator dynamics randomization — sysid measured right-side joints only;
    # re-sample each episode to cover left-side / unit / thermal / wear spread.
    randomize_actuator_params = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES_ALL),
            "friction_distribution_params": (0.5, 1.6),
            "armature_distribution_params": (0.8, 1.25),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    # Small torso-CoM randomization (horizontal ±2 cm box). Curriculum ramps from
    # ±5 mm to full range; non-accumulating so it is safe in reset mode.
    randomize_torso_com = EventTerm(
        func=randomize_torso_com_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "com_range": dict(TORSO_COM_RANGE),
        },
    )


@configclass
class RewardsCfg:
    """Active-balancing standing reward (Jul 17 2026 retarget): survive, hold
    a near-upright slightly-forward posture, keep the CoM over the feet, and
    MOVE to recover.

    Jul 17 2026 posture retarget (geometry audit, see the BALANCE_BAND_GX_*
    constants block): the 8-12° back-lean band used since Jul 10 put the CoM
    131-172 mm behind an ankle axis that has only a 23 mm heel margin —
    statically impossible. On hardware the policy slid to the feasible
    posture nearest the band and balanced within millimeters of the heel
    edge, so any backward pitch wobble tipped it over irrecoverably
    (capture 20260717_213006). The band now straddles upright with a slight
    forward bias (g_x ∈ [-0.02, +0.05]), the hip anchor is relaxed so the
    policy can make the ~5° hip-flexion compensation the statics require,
    and quadratic tails keep a restoring gradient at large tilt.

    History: the Jul 10-14 2026 design was a *statue* — survive, hold
    posture, move smoothly. The smoothing worked (capture 20260715_041834:
    slew-exceedance 40-55% -> 6.3%) but the robot could not balance: on a
    gantry test it hung at g_x mean +0.065 (12° off the then-band) instead
    of correcting, because the movement penalties (position_rate -10.0,
    joint_vel -0.5, hip_flexion_anchor -0.6) made every recovery motion
    unaffordable. A 0.1 rad hip flexion cost -0.4/tick while
    ``torso_posture`` only earned +0.028/tick at that tilt — the policy
    correctly chose not to move and was caught by the gantry.

    The Jul 15 fix inverts the gate pattern already used by
    :func:`bilateral_joint_symmetry_l2` (a stillness gate that suppresses a
    penalty during motion): the movement penalties now carry a *balance gate*
    that suppresses them during tilt. The gate is a Gaussian on how far the
    torso tilt is from the balance-band center:

        gate = exp(-((g_x - center)² + g_y²) / gate_std²)

    At balance the gate is 1.0 — the anti-chatter penalties fire at full
    strength (their actual job: kill steady-state wobble). At the capture's
    failure tilt (12° off band) the gate is ~0.08 — the penalties vanish and
    the policy is free to swing the legs to catch the fall.

    * ``torso_posture`` (``torso_pitch_asymmetric_reward``) — the balance
      target: flat-top band straddling upright (Jul 17 retarget, above),
      asymmetric shoulders (wide on the heel side for recovery reach,
      tight on the toe side), quadratic tails for far-field gradient.
    * ``com_over_support`` — bounded [0,1] reward for keeping the CoM
      (base_link xy) over the foot midpoint, stillness-gated. The positive
      carrot that complements the gates: gives a gradient *toward* balance.
    * ``hip_flexion_anchor`` — balance-gated, relaxed to -0.2 (Jul 17):
      the ~5° hip flexion the statics require is now affordable; only a
      deep crouch is priced out. Frees the hips fully when tilted.
    * ``joint_vel`` — balance-gated. Kills wobble at balance, frees the
      legs to move when tilted.
    * ``position_rate`` — balance-gated. Kills chatter at balance (-10.0
      full strength), frees the setpoints to move rapidly when tilted.
    * ``joint_pos_anchor`` (``mdp.joint_deviation_l1``) over all joints —
      general posture regularizer, NOT gated (the home pose should always be
      the quiet attractor; the L1 pull is cheap and the gate machinery is
      only needed where the weight is high enough to block recovery).
    * ``bilateral_symmetry`` — Σ(q_L + q_R)² over all 4
      L/R joint pairs, stillness-gated. The HEAVY symmetry enforcer:
      capture 20260715_214224 showed the prior reward had NO active symmetry
      term (only the diluted -0.3 all-joint anchor), so the policy stood with
      hip_flexion L+R=-0.82 and foot L+R=+0.49. Heavily weighted (-2.0) so
      an asymmetric stance costs more than the alive reward earns. Uses the
      SUM (q_L + q_R) because every L/R pair on this robot is sign-mirrored
      in the URDF — see ``bilateral_joint_symmetry_l2`` for the per-pair
      axis audit. Stillness-gated (NOT balance-gated — symmetry is a posture
      constraint that should fire whenever the robot is holding any pose,
      not only when balanced; the balance gate suppressed it to -0.03/tick
      during the tilted-exploration phase of training).
    * ``feet_flat`` — Σ_feet sin²(sole tilt from
      horizontal), stillness-gated. Forces both soles parallel to the ground
      while standing, so the policy cannot ride the toe/heel edge of the
      rigid sim foot (a non-transferable contact cheat that leans on the
      weak RS02 ankle). NOT a foot-joint position anchor: in the balanced
      stance the shank slants ~5° (hip-flexion compensation), so a flat
      sole needs q_foot ≈ +5°; the term targets the sole *orientation*
      (FK from encoders) and lets the ankle pick the angle.

    The ankle-support hack stays fixed in PHYSICS, not reward: the foot
    ``effort_limit_sim`` is capped at the RS02 continuous rating (6 Nm).
    """

    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    # Torso posture: a flat-top BAND straddling upright (g_x ∈ [-0.02, +0.05],
    # 1.1° back … 2.9° forward) with a slight forward bias, asymmetric
    # Gaussian shoulders, and quadratic far-field tails. NOT the pre-Jul-17
    # 8-12° back-lean band: the geometry audit (see the BALANCE_BAND_GX_*
    # constants block) showed that band put the CoM 6-7x the heel margin
    # behind the ankle — statically impossible — and the hardware policy was
    # living at the heel edge nearest the unreachable target (capture
    # 20260717_213006). The forward bias exploits the 140 mm toe margin;
    # the back edge stays shallow because the 23 mm heel margin is the fall
    # direction. The quadratic tails (deadband at the band edges) keep a
    # restoring gradient at large tilt where the shoulders' gradient dies —
    # the learnable "recover once the pitch runs away" signal.
    # See ``torso_pitch_asymmetric_reward``.
    torso_posture = RewTerm(
        func=torso_pitch_asymmetric_reward,
        weight=0.5,
        params={
            "band_gx_min": BALANCE_BAND_GX_MIN,
            "band_gx_max": BALANCE_BAND_GX_MAX,
            "edge_std_below": BALANCE_EDGE_STD_BELOW,
            "edge_std_above": BALANCE_EDGE_STD_ABOVE,
            "roll_std": BALANCE_ROLL_STD,
            "forward_penalty_gain": BALANCE_FWD_PENALTY_GAIN,
            "backward_penalty_gain": BALANCE_BWD_PENALTY_GAIN,
        },
    )

    # Hip-flexion anchor — Σ|q| over ONLY the two hip flexion joints, gated
    # by how close the torso is to the balance band. Weight relaxed -0.6 ->
    # -0.2 (Jul 17 2026): the geometry audit showed ~0.06-0.10 rad of hip
    # flexion is the geometric compensation that moves the ankle under the
    # CoM (the ankle axis would otherwise sit 48 mm ahead of the CoM with
    # only a 23 mm heel margin) — the stance REQUIRES it, so the anchor
    # must not price it out. At -0.2 the needed flexion costs ~-0.03/tick
    # (cheap) while a deep 0.5 rad crouch still costs -0.2/tick (a wall).
    # The whole-joint ``joint_pos_anchor`` below keeps the knees near
    # straight. Restricted to hip_flexion (not knee/foot) so the knees/feet
    # can still bend for recovery.
    #
    # BALANCE-GATED (Jul 15 2026): the gate fires at full strength when
    # balanced (holds hips near straight at steady state) and vanishes when
    # tilted (frees the hips to swing a foot under the falling CoM).
    # Capture 20260715_041834 showed the ungated -0.6 anchor made recovery
    # unaffordable: the robot hung at g_x mean +0.065 instead of
    # correcting, because a 0.1 rad hip flexion cost -0.4/tick while
    # ``torso_posture`` only earned +0.028/tick at that tilt.
    hip_flexion_anchor = RewTerm(
        func=joint_deviation_l1_balance_gated,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "hip_flexion_left_joint",
                    "hip_flexion_right_joint",
                ],
                preserve_order=True,
            ),
            "gate_band_gx_center": BALANCE_BAND_GX_CENTER,
            "gate_std": BALANCE_GATE_STD,
            "gate_floor": BALANCE_GATE_FLOOR,
        },
    )

    # Direct "hold absolutely still" signal — unbounded quadratic on joint
    # velocity. This is the main anti-oscillation term: linear gradient in v
    # all the way to zero (a bounded exp kernel's gradient vanishes at v=0
    # and won't damp the residual wobble). Weight stays at -0.5 (Jul 14
    # 2026): capture 20260715_011517 of run 2026-07-14_12-51-25 ckpt-4000
    # showed joint vel_std 0.45-0.81 rad/s on hardware, but this term was
    # already contributing -1.31/tick at -0.5 (raw sum(v²)=2.62) — LARGER
    # than alive (+1.0). The chatter persists NOT because joint_vel is too
    # weak (it's already the biggest single penalty) but because the
    # posture reward (+0.21/tick) earns more than position_rate pays, so the
    # policy chatters the setpoints to chase posture. Raising joint_vel
    # further would freeze recovery without fixing the chatter; the lever
    # is position_rate, not joint_vel.
    #
    # BALANCE-GATED (Jul 15 2026): the gate fires at full strength when
    # balanced (kills residual wobble, the term's job) and vanishes when
    # tilted (frees the policy to swing the legs to catch a fall). Without
    # the gate the -0.5 weight makes recovery unaffordable at tilt — see
    # the ``hip_flexion_anchor`` comment for the capture evidence.
    joint_vel = RewTerm(
        func=joint_vel_l2,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=JOINT_NAMES_ALL, preserve_order=True
            ),
            "balance_gate": True,
            "gate_band_gx_center": BALANCE_BAND_GX_CENTER,
            "gate_std": BALANCE_GATE_STD,
            "gate_floor": BALANCE_GATE_FLOOR,
        },
    )

    # Tick-to-tick change penalty on the 16 kp/kd channels — main anti-chatter
    # term for variable impedance, and the "produce the SAME gain every tick"
    # signal for the FixedGain (learned-constant-gains) variant.
    gain_rate = RewTerm(func=action_gain_rate_l2, weight=-0.80)

    # Tick-to-tick change penalty on the 8 position channels — kills the
    # setpoint limit cycle that rides the 0.020 rad/tick slew limiter.
    # Weight bumped -1.2 → -10.0 (Jul 14 2026) after capture 20260715_011517
    # of run 2026-07-14_12-51-25 ckpt-4000 showed the policy commanding
    # 0.03-0.06 rad setpoint steps 40-55% of ticks on hips/knees. At -1.2
    # this term only contributed -0.030/tick (raw sum(Δa²)=0.025) while
    # torso_posture earned +0.21/tick — the chatter was essentially free.
    # At -10.0 the chatter cost rises to -0.25/tick, finally exceeding the
    # posture reward and forcing the policy to find a steady setpoint
    # rather than ride the slew limiter. The raw-action penalty is on the
    # pre-slew-limiter command, so this bites at the source (the network
    # output), not on the already-clamped plant target. Why not higher:
    # at -10 the steady-state cost of legitimate slow drift (raw Δa ~0.01
    # per tick) is only -0.0025/tick, so normal tracking is still cheap;
    # only the rapid chatter (raw Δa ~0.05+ per tick) gets taxed hard.
    #
    # BALANCE-GATED (Jul 15 2026): same gate as ``joint_vel`` and
    # ``hip_flexion_anchor``. The -10.0 weight is essential for killing
    # steady-state chatter (capture 20260715_041834 confirmed it worked:
    # slew-exceedance dropped 40-55% -> 6.3%), but ungated it made
    # recovery unaffordable — at 12° off the band a 0.1 rad recovery
    # across 4 joints cost -0.4/tick while ``torso_posture`` earned only
    # +0.028/tick, so the policy hung forward on the gantry instead of
    # correcting. The gate keeps -10.0 at balance (chatter still killed)
    # and drops it to ~-0.78 at the failure tilt (recovery affordable).
    position_rate = RewTerm(
        func=action_position_rate_l2,
        weight=-10.00,
        params={
            "balance_gate": True,
            "gate_band_gx_center": BALANCE_BAND_GX_CENTER,
            "gate_std": BALANCE_GATE_STD,
            "gate_floor": BALANCE_GATE_FLOOR,
        },
    )

    # CoM-over-support — bounded [0,1] reward for keeping the torso CoM
    # (approximated by base_link xy) over the midpoint between the two feet,
    # AND holding still while doing it. The stillness gate is essential:
    # without it the policy can earn the full carrot by SWINGING the CoM
    # through the target on every oscillation (the reward fires on position,
    # not velocity). Capture 20260715_122500 showed this exploit: the term
    # climbed to +0.39 while the robot thrashed (vel_std 0.69-1.34). The gate
    # makes the carrot pay out only when the CoM is over the feet AND the
    # robot is settling — turning it from a swing-enabler into a hold-enforcer.
    # The previous ``tilt_boost_gain`` is removed: it made the flailing
    # exploit MORE profitable at tilt, exactly when the movement-penalty gates
    # were most relaxed. Non-privileged (FK from joint encoders).
    com_over_support = RewTerm(
        func=com_over_support_reward,
        weight=0.4,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "foot_body_names": ("foot_left_1", "foot_right_1"),
            "max_lateral_dist": 0.12,
            "stillness_std": 2.0,
        },
    )

    # Default-pose anchor — Σ|q| over all joints (the default pose is all
    # zeros: straight-leg symmetric stand). The general posture regularizer:
    # it prevents the crouch-at-the-action-rail optimum, keeps the stance
    # bilaterally symmetric (the anchor pose is symmetric), and gives the
    # policy a home pose to return to after a recovery. Weight is deliberately
    # modest so the dedicated ``hip_flexion_anchor`` above can dominate hip
    # flexion specifically without this term diluting the gradient across all
    # 8 joints. NOT stillness-gated (unlike the old symmetry term) because an
    # L1 pull toward default is cheap during transients.
    joint_pos_anchor = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.3,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=JOINT_NAMES_ALL, preserve_order=True
            )
        },
    )

    # Bilateral symmetry — Σ(q_L + q_R)² over all 4 L/R pairs, stillness-
    # gated. This is the HEAVY symmetry enforcer (Jul 15 2026).
    #
    # Capture 20260715_214224 (run 2026-07-15_12-35-39 ckpt-4000) showed the
    # robot standing severely asymmetric on hardware despite the (correct,
    # sum-convention) symmetry term existing in the code:
    #     hip_flexion    L=-0.40  R=-0.42  L+R=-0.82  (symmetric would be 0)
    #     hip_abduction  L=+0.20  R=-0.27  L+R=-0.07  (≈ symmetric)
    #     knee_flexion   L=+0.13  R=-0.24  L+R=-0.12  (≈ symmetric)
    #     foot           L=+0.18  R=+0.31  L+R=+0.49  (symmetric would be 0)
    # Root cause: the symmetry term was NOT in the active RewardsCfg — the
    # only symmetry enforcement was this -0.3 all-joint joint_pos_anchor
    # above, whose gradient is diluted 8 ways and cannot enforce per-pair
    # mirroring. The dedicated term below is heavily weighted (-2.0) so an
    # asymmetric stance finally costs more than the alive reward earns.
    #
    # SIGN CONVENTION — the SUM, not the difference: every L/R pair on this
    # robot is sign-mirrored in the URDF. The right-side flexion joints
    # (hip_flexion, knee_flexion, foot) use a flipped -Y axis; hip_abduction
    # uses the same +X axis but mirrored limits. A symmetric stance therefore
    # reads q_L = -q_R for ALL four pairs, so the asymmetry residual is
    # (q_L + q_R)² — NOT (q_L - q_R)². The pre-Jul-2026 version used the
    # difference and actively TRAINED the twisted-hip contortion it was meant
    # to prevent (run 2026-07-09_04-16-40). See the per-pair axis table in
    # ``bilateral_joint_symmetry_l2``'s docstring for the full audit.
    #
    # STILLNESS-GATED (NOT balance-gated — Jul 15 2026 correction). The
    # first attempt wired this term with balance_gate=True, which suppressed
    # the penalty via the tilt-distance Gaussian whenever the robot was off
    # the back-lean band. Training run 2026-07-15_22-32-35 (4568 steps) showed
    # the term contributing only -0.03/tick at the -2.0 weight — the policy
    # was tilted/falling (eplen 1160/2000, 58% survival) for most of the
    # episode, so the gate clamped to its 0.2 floor and the effective weight
    # dropped to -0.4, too weak to shape the posture.
    #
    # Symmetry is a POSTURE constraint, not a motion constraint. The balance
    # gate (designed for motion penalties like joint_vel / position_rate that
    # would otherwise block recovery MOTIONS) is the wrong gate here: we want
    # the symmetry gradient active whenever the robot is holding *any* pose
    # (even a tilted lean), not only when it's in the balanced band. The
    # stillness gate (exp(-Σv²/σ²)) fires at full strength whenever the robot
    # is holding still — regardless of tilt — and decays toward 0 only during
    # active motion (when the policy legitimately needs asymmetric catch /
    # recovery motions). This keeps the posture pull alive through the full
    # learning curve, including the tilted-exploration phase where the
    # balance gate was killing it.
    bilateral_symmetry = RewTerm(
        func=bilateral_joint_symmetry_l2,
        weight=BILATERAL_SYMMETRY_WEIGHT,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pairs": BILATERAL_SYMMETRY_PAIRS,
            "balance_gate": False,
            "stillness_std": 1.5,
        },
    )

    # Feet flat — Σ_feet sin²(sole tilt from horizontal), stillness-gated
    # (Jul 16 2026, user request). The torso balance band + hip-flexion
    # anchor constrain the leg chain but leave the ankle free to hold the
    # sole at any angle — and the sim exploits that: the rigid foot's
    # contact patch lets the policy ride the toe or heel edge with a tilted
    # sole. That strategy does NOT transfer to hardware: the real foot is
    # small, the RS02 ankle is the weakest joint (~17 N·m, capped at its 6
    # N·m continuous rating in sim), and an edge contact shrinks the support
    # polygon to a line. Forcing the soles parallel to the ground maximizes
    # the contact polygon and stacks the load through the whole foot —
    # the foot-side counterpart to ``hip_flexion_anchor`` (hips straight +
    # soles flat = the ankle strategy).
    #
    # NOT a foot-joint position anchor: in the balanced stance the shank
    # slants ~5° back (the hip-flexion compensation that puts the ankle
    # under the CoM — see the BALANCE_BAND_GX_* audit), so a flat sole
    # requires q_foot ≈ +5° of ankle compensation. Anchoring q_foot at 0
    # would fight that stance. This term reads the
    # foot BODY orientation (sole normal = foot local +Z; every leg-chain
    # joint origin in the URDF has rpy=0, so the foot frame aligns with
    # base_link FLU at the zero pose) and lets the ankle find whatever angle
    # makes the sole flat. Non-privileged — FK from joint encoders + IMU.
    #
    # STILLNESS-GATED (same rationale as ``bilateral_symmetry``): flat soles
    # are a POSTURE constraint, enforced whenever the robot is holding a
    # pose, relaxing toward 0 only during active motion so recovery footwork
    # (toe-off, heel strike, lifting a foot) stays free. A balance gate
    # would clamp to its floor during the tilted-exploration phase and never
    # shape the stance — see the ``bilateral_symmetry`` comment for the
    # training-run evidence.
    #
    # Weight -2.0: the raw term is sin²(tilt) summed over 2 feet, so a 10°
    # sole tilt on both feet costs 2·sin²(10°)·2.0 ≈ -0.12/tick (a firm
    # shaping gradient, comparable to ``torso_posture``'s pull near the band
    # edge) and a 20° tilt ≈ -0.47/tick — expensive relative to alive (+1.0)
    # but still below the survival gradient. Matches the -2.0 weighting
    # precedent set by ``bilateral_symmetry`` for a stance-posture term.
    feet_flat = RewTerm(
        func=feet_flat_orientation_l2,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "foot_body_names": ("foot_left_1", "foot_right_1"),
            "balance_gate": False,
            "stillness_std": 1.5,
        },
    )


@configclass
class TerminationsCfg:
    time_out = TermTerm(func=mdp.time_out, time_out=True)
    # IMU tilt terminations DISABLED — ending at a tilt cliff let the policy
    # farm `alive` up to the boundary without learning to recover. Re-enable
    # (tighten toward 15-20°) once the robot balances off the ground-contact
    # backstop alone.
    # imu_pitch_out_of_bounds = TermTerm(
    #     func=imu_pitch_out_of_bounds,
    #     params={
    #         "pitch_forward_gx_max": PITCH_FALL_LIMIT_GX,
    #         "pitch_back_gx_min": -PITCH_FALL_LIMIT_GX,
    #     },
    # )
    # imu_roll_out_of_bounds = TermTerm(
    #     func=imu_roll_out_of_bounds,
    #     params={"roll_gy_limit": ROLL_FALL_LIMIT_GY},
    # )
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
class CurriculumCfg:
    """Curricula for robustness knobs added after the quiet stand converges."""

    torso_com = CurrTerm(
        func=torso_com_curriculum,
        params={
            "term_name": "randomize_torso_com",
            "full_com_range": dict(TORSO_COM_RANGE),
            "start_fraction": TORSO_COM_START_FRACTION,
            "num_curriculum_steps": TORSO_COM_CURRICULUM_STEPS,
        },
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
    curriculum = CurriculumCfg()

    def __post_init__(self):
        self.viewer.eye = [2.5, 2.5, 2.5]
        self.viewer.lookat = [0.0, 0.0, 0.0]

        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        # Required the moment privileged foot-contact sensing is re-added.
        self.sim.disable_contact_processing = False

        self.scene.robot = BEBOP_V2_STANDING_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.scene.terrain = terrain_gen.TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="plane",
            collision_group=-1,
            # Friction reverted to 1.0/1.0 (2026-06-20): the 0.4/0.3 "carpet-
            # matched" friction was tested and made the hardware asymmetry
            # WORSE (hip L-R went from +0.30 to +0.65), proving friction wasn't
            # the root cause. The asymmetry is the policy compensating for real
            # hardware L/R differences (motor friction, mass, alignment), not a
            # contact effect. The bilateral_symmetry + foot_deviation reward
            # terms now enforce the symmetric stance directly. Keep 1.0/1.0
            # friction so the policy can learn to balance without slipping
            # complicating the dynamics — the symmetric-stance enforcement
            # should be done by the reward, not by the contact model. Revisit
            # friction once the hardware foot is redesigned (rubber pads /
            # wider foot) to match the real deployment surface.
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


@configclass
class BebopV2StandingFixedGainCfg(BebopV2StandingCfg):
    """Quiet-stand variant with gains STRUCTURALLY pinned on both sides.

    Third design (Jul 11 2026), after two instructive failures:

    1. freeze-in-sim only (run 2026-07-10_04-12-52): physics froze the
       gains at midpoint but firmware kept decoding the 16 inert channels
       over the full clamp range → deployed kp wandered ±13-19 around
       values sim never varied.
    2. live gains + constancy rewards (run 2026-07-11_02-20-09): gain_rate
       -0.80 + gain_anchor -0.50 got the *means* near midpoint but the
       deterministic network still modulates gains with its observations
       (capture 20260711_163155: kp std 1-12 on quiet ticks vs 5-24 on
       active ticks — the variation tracks body motion). A net can only
       emit a constant by ignoring its inputs, and PPO keeps the
       modulation whenever it helps recovery more than the penalty costs.
       Reward pressure CANNOT guarantee constant outputs.

    The lock is therefore structural on BOTH decode paths:

    * sim: ``freeze_gains=True`` — physics always uses the midpoint gains,
      regardless of what the network emits;
    * firmware: ``policy_gain_clamps`` pinned to ±epsilon bands around the
      SAME midpoints (bebop_v2.yaml, Jul 11 2026 — the config loader
      rejects min==max, so e.g. kp 69.5..70.5 for hip flexion). Any raw
      value decodes to within ±0.5 Nm/rad of the trained gain.

    With both sides pinned, the 16 gain channels are irrelevant to
    dynamics everywhere, so NO gain reward terms remain (``gain_rate``
    removed, no ``gain_anchor``) — the position-only stand trains on the
    6-term simplified reward. The channels still exist in the action/obs
    layout (ONNX I/O and the 49-dim obs are unchanged) and both sim and
    firmware echo the raw action into ``last_action``, so the obs
    feedback stays consistent too.
    """

    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos.freeze_gains = True
        self.actions.joint_pos.kp_fixed = _Kp_MID
        self.actions.joint_pos.kd_fixed = _KD_MID

        # Gain channels can't reach physics (sim) or the motors (pinned
        # firmware clamps), so no gain shaping is needed at all.
        self.rewards.gain_rate = None


@configclass
class BebopV2StandingPushCfg(BebopV2StandingCfg):
    """Variable-impedance stand WITH mid-episode pushes (reactive recovery).

    Step-2 follow-up to the clean ``BebopV2StandingCfg`` baseline — validate
    that converges and transfers first, then train this. Two coupled changes:

    1. ``push_robot`` — random-interval ±0.4 m/s fore/aft + lateral root-velocity
       shoves. Non-privileged replacement for the removed ``feet_load_symmetry``
       penalty: a one-foot lean is fragile to a sideways shove, so pushes
       pressure the policy toward a centered, two-foot stance.
    2. ``push_level`` curriculum — ramps the push envelope from 40% to 100%
       over ~150k control steps.

    Pair with ``BebopPPOPushCfg`` (higher entropy_coef): pushes enlarge the
    reward landscape, so the actor needs more exploration.
    """

    def __post_init__(self):
        super().__post_init__()

        # interval_range_s picks a fresh wait between pushes per env so the
        # policy sees disturbances at varied phases of its stand.
        self.events.push_robot = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(4.0, 8.0),
            params={"velocity_range": dict(PUSH_VELOCITY_RANGE)},
        )

        self.curriculum.push_level = CurrTerm(
            func=push_magnitude_curriculum,
            params={
                "term_name": "push_robot",
                "full_velocity_range": dict(PUSH_VELOCITY_RANGE),
                "start_fraction": 0.4,
                "num_curriculum_steps": 150_000,
            },
        )
