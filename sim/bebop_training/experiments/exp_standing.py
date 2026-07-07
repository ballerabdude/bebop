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
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from ..envs.bebop_v2_actions import VariableImpedanceJointActionCfg
from ..envs.bebop_v2_events import (
    push_magnitude_curriculum,
    randomize_torso_com_uniform,
    reset_joints_uniform_within_limits,
    torso_com_curriculum,
)
from ..envs.bebop_v2_rewards import (
    action_gain_l2,
    action_gain_rate_l2,
    action_position_rate_l2,
    upright_pose_exp,
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
# ±0.6 rad/s trains pitch-recovery authority; both zeroed for a no-perturbation
# diagnostic run — restore to math.radians(15.0) / 0.6 after the experiment.
PITCH_INIT_RAD = math.radians(0.0)
PITCH_RATE_INIT_RAD_S = 0.0

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

# Observation noise matched to the hardware noise floor. Small noise forces a
# lower-gain, transfer-robust stand; last_action and cmd_vel stay clean.
GYRO_NOISE_RAD_S = 0.01
JOINT_VEL_NOISE_RAD_S = 0.12
PROJ_GRAV_NOISE = 0.02
JOINT_POS_NOISE_RAD = 0.01

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
POLICY_KP_MIN = [20.0, 20.0, 40.0, 40.0, 30.0, 30.0, 100.0, 100.0]
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
            effort_limit_sim=17.0,
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
            func=mdp.imu_projected_gravity,
            params={"asset_cfg": SceneEntityCfg("imu")},
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
    """Standing reward: action regularization only.

    All pose, balance, and symmetry shaping has been removed — the policy is
    driven purely by the stay-alive / termination-penalty survival signal plus
    action-space regularization (smoothness, gain/position rate, and gain
    magnitude centering). This is a minimal baseline; re-add task shaping
    terms individually as needed.
    """

    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    # Positive attractor toward gravity-aligned-upright. ``std = sin(10°)`` so
    # the exp reward's 1/e half-angle is ±10° — confines the policy to a ±10°
    # tilt basin (symmetric in pitch and roll) with a smooth gradient to
    # vertical. Pair with survival; tilt terminations stay disabled.
    upright_pose = RewTerm(
        func=upright_pose_exp, weight=0.75, params={"std": math.sin(math.radians(10.0))}
    )

    # Anchors the action distribution so its std can't run away (worst in
    # fixed-gain, where the 16 kp/kd channels have no other gradient).
    # Bumped -0.02 -> -0.08 (2026-07-01): capture of run 2026-07-01_03-43-29
    # ckpt-500 showed this paying only -0.0160/tick vs alive +0.8705/tick —
    # below the noise floor of the survival signal. 4x (was 10x at -0.20,
    # backed off 2026-07-01: the 10x over-constrained and dropped eplen
    # from 91% to 84% even at entropy .01 — run 2026-07-01_05-12-43).
    action_l2 = RewTerm(func=mdp.action_l2, weight=-0.08)

    # Isolates the 16 gain channels so the quiet-stand optimum is midpoint
    # gains. Bumped -0.03 -> -0.12 (2026-07-01): capture showed gains parked
    # at rails (foot kp 163/182, knee-R 104, hip-abd 108/101 — all far off
    # midpoint). 4x (was 10x at -0.30, backed off same as action_l2).
    gain_l2 = RewTerm(func=action_gain_l2, weight=-0.12)

    # Tick-to-tick change penalty on the 16 kp/kd channels — main anti-chatter
    # term for variable impedance. Bumped -0.20 -> -0.80 (2026-07-01):
    # capture showed FFT peaks at 2.5/5.6/6.7 Hz across all joints with
    # lag-1 autocorrelation > +0.87 — smooth limit cycle. 4x (was 10x at
    # -2.0, backed off same as action_l2 — 10x over-constrained and
    # degraded eplen).
    gain_rate = RewTerm(func=action_gain_rate_l2, weight=-0.80)

    # Tick-to-tick change penalty on the 8 position channels — kills the
    # setpoint limit cycle that rides the 0.020 rad/tick slew limiter.
    # Bumped -0.30 -> -1.20 (2026-07-01): capture showed 33.9% of all ticks
    # commanding |Δtarget| > 0.020 (hip_flex 44%, foot_left 42%) — the
    # policy is riding the slew rail. 4x (was 10x at -3.0, backed off
    # same as the others). Re-evaluate: if the slew-exceedance fraction
    # stays high, reformulate as mean|Δ| or a hinge-at-slew rather than
    # Σ Δ² (the squared form saturates weakly at small Δ).
    position_rate = RewTerm(func=action_position_rate_l2, weight=-1.20)


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
    """Quiet-stand variant with the variable-impedance gains FROZEN.

    The policy's 16 kp/kd action channels are ignored and physics uses fixed
    per-joint gains (the midpoint of each POLICY_KP_MIN/MAX / POLICY_KD_MIN/MAX
    band). The policy therefore only learns the 8 joint position targets.

    Recommended first hardware stand: removing 16 of 24 action dimensions
    removes the gain chattering seen in deployed logs, leaving a smoother
    position-only PD problem. The action vector stays 24-dim, so the ONNX I/O
    and the 49-dim observation match firmware unchanged. The fixed gains equal
    the firmware raw=0 decode, so no firmware change is required. (If you
    override kp_fixed / kd_fixed away from the midpoints, pin the matching
    policy_gain_clamps in firmware/bebop-linux/config/bebop_v2.yaml too.)
    """

    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos.freeze_gains = True
        self.actions.joint_pos.kp_fixed = _Kp_MID
        self.actions.joint_pos.kd_fixed = _KD_MID


@configclass
class BebopV2StandingPushCfg(BebopV2StandingCfg):
    """Variable-impedance stand WITH mid-episode pushes (reactive recovery).

    Step-2 follow-up to the clean ``BebopV2StandingCfg`` baseline — validate
    that converges and transfers first, then train this. Three coupled changes:

    1. ``push_robot`` — random-interval ±0.4 m/s fore/aft + lateral root-velocity
       shoves. Non-privileged replacement for the removed ``feet_load_symmetry``
       penalty: a one-foot lean is fragile to a sideways shove, so pushes
       pressure the policy toward a centered, two-foot stance.
    2. ``push_level`` curriculum — ramps the push envelope from 40% to 100%
       over ~150k control steps.
    3. Softened ``feet_straight`` (-3.0 -> -0.5): with lateral pushes the
       policy must be free to widen its stance to catch a sideways shove.

    Pair with ``BebopPPOPushCfg`` (higher entropy_coef): pushes enlarge the
    reward landscape, so the actor needs more exploration.
    """

    def __post_init__(self):
        super().__post_init__()

        # Free hip abduction for lateral recovery.
        self.rewards.feet_straight.weight = -0.5

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
