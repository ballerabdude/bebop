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

# Resting torso posture: a slight BACK lean held as a band (not a point).
# Why a band, not a single upright target: the previous ``upright_pose``
# Gaussian was centered on g_x = g_y = 0 (perfect vertical) with its steepest
# gradient AT vertical, so the policy was pulled hard to a perfectly upright
# posture that is NOT hardware-stable — the CoM ends up over the toes / ankle
# pivot and the weak RS02 ankle motor can't hold it (run 2026-07-14_04-09-59:
# 70% of episodes ended in base_link_ground_contact while upright_pose was the
# second-largest positive shaping term). A flat-top plateau over a back-lean
# band lets the policy settle anywhere in the band rather than being collapsed
# onto one point, and the deeper back lean shifts the CoM behind the ankle
# pivot so the leg stacks load through the hip/knee instead of leaning on the
# foot. See ``torso_pitch_asymmetric_reward`` in bebop_v2_rewards.py.
#
# Sign convention (body FLU): g_x = -sin(pitch); back lean is g_x < 0.
#   BACK_LEAN_DEEP_DEG  = 12  -> g_x = -0.208 (deep edge of the plateau)
#   BACK_LEAN_SHALLOW_DEG = 8  -> g_x = -0.139 (shallow edge; still a back lean)
# 10° sits in the middle of the band (g_x ≈ -0.174), the user-requested target.
BACK_LEAN_DEEP_DEG = 12.0
BACK_LEAN_SHALLOW_DEG = 8.0
BACK_LEAN_BAND_GX = (
    -math.sin(math.radians(BACK_LEAN_DEEP_DEG)),     # band_gx_min (deep edge)
    -math.sin(math.radians(BACK_LEAN_SHALLOW_DEG)),   # band_gx_max (shallow edge)
)
# Gaussian falloff width outside the band (in g_x units; ~0.12 ≈ 7° shoulder).
BACK_LEAN_EDGE_STD = 0.12
# Roll tolerance inside the band (g_y; ~0.15 ≈ 8.6° at 1/e).
BACK_LEAN_ROLL_STD = 0.15
# Forward-pitch penalty inside the reward (g_x > 0 is the hardware-fatal
# forward lean; penalize any forward component, no deadband).
BACK_LEAN_FWD_PENALTY_GAIN = 5.0

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
    """SIMPLIFIED standing reward (Jul 10 2026): survive, hold a back-lean
    posture, keep the hips straight, move smoothly.

    History: the previous design (upright + bilateral symmetry + three
    rate/velocity penalties) grew term-by-term, each patching a hack the
    last one opened. After the symmetry sign fix, run 2026-07-09_13-07-14
    exposed the remaining hole: nothing anchored the POSTURE, so the
    optimum was a deep crouch with the hips parked on the ±0.5 rad action
    rail (raw hip action clipped 97%/74% of ticks) and the whole robot
    held up by the ankle (est. 37 Nm commanded at the right foot vs the
    RS02's 6 Nm continuous rating) — un-transferable to hardware.

    The simplification replaces indirect shaping with direct anchors:

    * ``torso_posture`` (``torso_pitch_asymmetric_reward``) — a flat-top
      back-lean band (8-12°, centered on the 10° target) replaces the old
      ``upright_pose`` Gaussian. The Gaussian was centered on perfect
      vertical (g_x = g_y = 0) with its steepest gradient AT vertical,
      which pulled the policy to a posture that is not hardware-stable
      (CoM over the ankle pivot, load on the weak RS02 foot motor). The
      flat-top band lets the torso settle anywhere in [8°, 12°] of back
      lean and shifts the CoM behind the ankle pivot so the leg stacks
      load through the hip/knee. Forward pitch is penalized inside the
      same term.
    * ``hip_flexion_anchor`` — Σ|q| over ONLY the two hip flexion joints.
      The natural compensation when the torso leans back is to
      counter-rotate by flexing the hips forward (keeping CoM over the
      feet via joint angles); we want the ankle strategy instead (entire
      body leans together, hips stay straight). Scoped to hip_flexion so
      the knees/feet can still bend for recovery.
    * ``joint_pos_anchor`` (``mdp.joint_deviation_l1``) over all joints
      — general posture regularizer: prevents the crouch-at-the-rail
      optimum, keeps the stance bilaterally symmetric, and gives the
      policy a home pose to return to after recovery. Weight is modest so
      ``hip_flexion_anchor`` can dominate hip flexion specifically.
    * The ankle-support hack is fixed in PHYSICS, not reward: the foot
      ``effort_limit_sim`` is capped at the RS02 continuous rating (6 Nm),
      so leaning on the ankle simply stops working in sim.
    * Anti-oscillation is down to two terms: ``position_rate`` (smooth
      setpoints, the cause) and ``joint_vel`` (still plant, the effect),
      plus ``gain_rate`` for the 16 live gain channels in the
      variable-impedance config. The FixedGain variant pins the gains
      structurally (sim freeze + firmware clamp pin) and drops
      ``gain_rate`` entirely (see ``BebopV2StandingFixedGainCfg``).
    """

    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    # Torso posture: a flat-top back-lean BAND (8-12°, centered on the
    # user-requested 10° back lean) — NOT a single-target Gaussian on vertical.
    # The previous ``upright_pose`` Gaussian was centered on g_x = g_y = 0
    # (perfect vertical) with its steepest gradient AT vertical, which pulled
    # the policy to a posture that is not hardware-stable: CoM over the ankle
    # pivot, load carried by the weak RS02 foot motor. Run 2026-07-14_04-09-59
    # showed the failure — 70% of episodes ended in base_link_ground_contact
    # while upright_pose was the second-largest positive shaping term,
    # steering the policy away from active balance. The flat-top band lets the
    # torso settle anywhere in [8°, 12°] of back lean (g_x ∈ [-0.208, -0.139])
    # without being collapsed onto one point, and the deeper back lean shifts
    # the CoM behind the ankle pivot so the leg stacks load through the
    # hip/knee instead of the foot. Forward pitch (g_x > 0) is penalized
    # inside the same term — the hardware-fatal direction.
    # See ``torso_pitch_asymmetric_reward`` and the BACK_LEAN_* constants.
    torso_posture = RewTerm(
        func=torso_pitch_asymmetric_reward,
        weight=0.5,
        params={
            "band_gx_min": BACK_LEAN_BAND_GX[0],
            "band_gx_max": BACK_LEAN_BAND_GX[1],
            "edge_std": BACK_LEAN_EDGE_STD,
            "roll_std": BACK_LEAN_ROLL_STD,
            "forward_penalty_gain": BACK_LEAN_FWD_PENALTY_GAIN,
            "forward_deadband": 0.0,
        },
    )

    # Hip-flexion anchor — Σ|q| over ONLY the two hip flexion joints. The
    # whole-joint ``joint_pos_anchor`` below is too weak to hold hip flexion
    # at zero once the torso is leaning back: the natural compensation for a
    # back-leaning torso is to counter-rotate by flexing the hips forward
    # (keeping CoM over the feet via joint angles). We want the ankle strategy
    # instead — the entire body leans together, hips stay straight — so the
    # hip flexion joints get their own stronger, scoped anchor. Restricted to
    # hip_flexion (not knee/foot) so the legs can still bend at the knee and
    # ankle for recovery. Not stillness-gated: an L1 pull toward zero is cheap
    # during transients and we want the hips straight at all times.
    hip_flexion_anchor = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.6,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "hip_flexion_left_joint",
                    "hip_flexion_right_joint",
                ],
                preserve_order=True,
            )
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
    joint_vel = RewTerm(
        func=joint_vel_l2,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=JOINT_NAMES_ALL, preserve_order=True
            )
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
    position_rate = RewTerm(func=action_position_rate_l2, weight=-10.00)

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
