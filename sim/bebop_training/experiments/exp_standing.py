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
  * ``torso_pitch``         — asymmetric IMU pitch: reward ~17° back lean
                              (``proj_grav[0] ≈ -0.30``), penalize forward
                              pitch (hardware falls on any forward lean).
  * ``joint_pos_limits``    — quadratic penalty for exceeding the URDF
                              joint soft limits.
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
  * Joints sampled uniformly per-joint within their soft limits via
    ``reset_joints_uniform_within_limits`` — covers the full configuration
    box, not just a tight band around the default pose. Some sampled
    poses are unrecoverable from ``base_link.z = 0.8 m`` (e.g. knee fully
    folded with foot above ground); expect a non-trivial fraction of
    episodes to terminate early. Watch ``mean_episode_length`` and the
    ``base_link_ground_contact`` termination rate to confirm there's
    still useful signal getting through.
  * Base roll ±0.30 rad; pitch biased toward back lean ``(-0.35, +0.10)``
    rad (~−20°..+6°) — forward pitch samples are rare because hardware
    cannot recover from forward tilt. Yaw full ±π.
  * Initial angular velocity perturbed by ±0.5 rad/s on all three axes.

Deliberately off (add back one at a time after hardware validation):
  * Mid-episode pushes, observation noise, contact sensors, stepping rewards.
  * ``flat_orientation``, ``base_height``, ``joint_deviation_l1``, symmetry
    penalties beyond the asymmetric pitch term above.

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
TARGET_PITCH_BACK_DEG = 17.0
TARGET_PITCH_BACK_GX = -math.sin(math.radians(TARGET_PITCH_BACK_DEG))


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
                # cannot recover from forward pitch).
                "pitch": (-PITCH_FALL_LIMIT_GX, 0.10),
                "yaw": (-math.pi, math.pi),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
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
            "target_gx": TARGET_PITCH_BACK_GX,
            "good_std": 0.12,
            "roll_std": 0.15,
            "forward_penalty_gain": 5.0,
            "forward_deadband": 0.0,
        },
    )

    joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)

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
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.02)

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
