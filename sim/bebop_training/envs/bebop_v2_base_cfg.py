import math

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
import isaaclab.envs.mdp as mdp

from isaaclab.actuators import ImplicitActuatorCfg
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
from isaaclab.utils.noise import UniformNoiseCfg

from .bebop_v2_actions import SlewLimitedJointPositionActionCfg
from .bebop_v2_rewards import (
    femur_symmetry_penalty,
    foot_symmetry_penalty,
    hip_abduction_symmetry_penalty,
    leg_action_when_stable_penalty,
    leg_position_hold_reward,
    shin_symmetry_penalty,
    undesired_yaw_penalty,
)
from .bebop_v2_terminations import base_link_on_ground


# ---------------------------------------------------------------------------
# Sim-to-real actuator constants.
#
# These MUST be kept in lockstep with the firmware-side ground truth in
# ``firmware/bebop-linux/config/bebop_v2.yaml``. A previous training run
# diverged here (sim hip_abduction kp = 200 vs firmware 40, sim foot kp = 40
# vs firmware 150) and the resulting policy stood beautifully in sim and
# collapsed instantly on the real robot — the policy had implicitly trained
# against ~5x more hip stiffness and ~4x less ankle stiffness than the
# motors actually ship.
#
# The values below mirror ``hold_gains`` / ``hard_limits`` for each joint
# group in the YAML. If you change a number on either side, change it on
# both — and then retrain.
# ---------------------------------------------------------------------------

# Per-group PD gains. Matches ``hold_gains`` in bebop_v2.yaml.
FW_HIP_ABDUCTION_KP = 40.0
FW_HIP_ABDUCTION_KD = 3.0
FW_FEMUR_KP = 160.0
FW_FEMUR_KD = 6.0
FW_SHIN_KP = 120.0
FW_SHIN_KD = 4.0
FW_FOOT_KP = 150.0
FW_FOOT_KD = 2.5

# Per-group torque caps. Pinned to each Robstride model's mechanical
# peak torque (T-N curve at 36 V — kscalelabs/kbot-v2 metadata + datasheet
# verification). The firmware's ``hard_limits.tau_max`` must be set
# *at or above* these numbers, otherwise the real motor will trip E-STOP
# on a torque the sim trained the policy to deliver.
#
# Reasoning for going to motor peaks instead of conservative safety
# caps: at slew = 0.05 rad/tick and kp ≈ 40–150, the policy needs to
# develop ~10–20 Nm of corrective torque within 100 ms to balance the
# bipedal CoM (m ≈ 17 kg, CoM height ≈ 0.4 m -> falling timescale ≈ 150 ms).
# Capping effort below mechanical peak just starves the controller.
FW_HIP_ABDUCTION_TAU_MAX = 84.0  # RS04 peak
FW_FEMUR_TAU_MAX = 42.0          # RS03 peak
FW_SHIN_TAU_MAX = 84.0           # RS04 peak (knee shares hip motor model)
FW_FOOT_TAU_MAX = 17.0           # RS02 peak

# Per-group velocity caps. Picked at the motors' *working* top speed
# rather than the no-load peak: no-load RS04 = 26 rad/s, RS03 = 24
# rad/s, RS02 = 43 rad/s, but under load the motors comfortably reach
# only about half those numbers. Capping velocity_limit_sim at the
# working ceiling keeps the sim from training trajectories the real
# motor can't sustain when actually under the robot's weight.
FW_HIP_ABDUCTION_VEL_MAX = 12.0  # RS04 working
FW_FEMUR_VEL_MAX = 12.0          # RS03 working
FW_SHIN_VEL_MAX = 12.0           # RS04 working
FW_FOOT_VEL_MAX = 20.0           # RS02 working (datasheet rated 43 no-load)

# Slew + delay: directly from bebop_v2.yaml ``defaults.slew`` and the
# 100 Hz tokio policy_runner tick. ``ACTION_DELAY_STEPS = 1`` approximates
# one CAN round-trip (TX → RobStride PD → encoder → RX feedback) of
# observation latency.
#
# 0.05 rad/tick @ 100 Hz = 5.0 rad/s setpoint slew rate. Found
# empirically: 0.005 (0.5 rad/s) and 0.01 (1 rad/s) both starved the
# policy of authority — the bipedal CoM falls in ~150 ms from a small
# tilt, but at slew = 1 rad/s the hip can only ramp by ~0.4 Nm/tick
# (kp_hip ≈ 40), so it takes >100 ms before the PD develops a useful
# braking torque. 5 rad/s lets the policy build ~10 Nm of new torque
# in 50 ms while still living comfortably below each joint's working
# vel_max (12+ rad/s).
FW_MAX_POS_STEP_PER_TICK_RAD = 0.05
FW_ACTION_DELAY_STEPS = 1


# Explicit joint order for Bebop V2 articulation.
JOINT_NAMES_ALL = [
    "hip_abduction_left_joint",
    "hip_abduction_right_joint",
    "femur_left_joint",
    "femur_right_joint",
    "shin_left_joint",
    "shin_right_joint",
    "foot_left_joint",
    "foot_right_joint",
]


BEBOP_V2_CFG = ArticulationCfg(
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
        # USD root has a built-in translate of (0, 0, 0.6539) so feet rest on
        # ground. Match that here to avoid sinking the legs below the floor.
        pos=(0.0, 0.0, 0.6539092050794861),
        joint_pos={joint_name: 0.0 for joint_name in JOINT_NAMES_ALL},
        joint_vel={joint_name: 0.0 for joint_name in JOINT_NAMES_ALL},
    ),
    soft_joint_pos_limit_factor=0.9,
    # Per-joint Robstride actuator configs.  ALL gains, torque caps, and
    # velocity caps below are pinned to the firmware ground truth in
    # ``firmware/bebop-linux/config/bebop_v2.yaml`` — see the
    # ``FW_*`` constants above for the mapping.  The hip / shin pair share
    # an RS04 motor model but live in separate actuator groups now because
    # the firmware uses per-joint kp/kd, not per-model.
    actuators={
        # Robstride RS04 -> hip abduction (lateral leg pitch).
        "hip_abduction": ImplicitActuatorCfg(
            joint_names_expr=[
                "hip_abduction_left_joint",
                "hip_abduction_right_joint",
            ],
            effort_limit_sim=FW_HIP_ABDUCTION_TAU_MAX,
            velocity_limit_sim=FW_HIP_ABDUCTION_VEL_MAX,
            stiffness=FW_HIP_ABDUCTION_KP,
            damping=FW_HIP_ABDUCTION_KD,
            armature=0.01,
            friction=0.0,
        ),
        # Robstride RS03 -> femur (hip pitch).
        "femur": ImplicitActuatorCfg(
            joint_names_expr=["femur_left_joint", "femur_right_joint"],
            effort_limit_sim=FW_FEMUR_TAU_MAX,
            velocity_limit_sim=FW_FEMUR_VEL_MAX,
            stiffness=FW_FEMUR_KP,
            damping=FW_FEMUR_KD,
            armature=0.005,
            friction=0.0,
        ),
        # Robstride RS04 -> shin (knee).  Same motor model as the hip but
        # the firmware runs it with different gains, so it gets its own
        # actuator group.
        "shin": ImplicitActuatorCfg(
            joint_names_expr=["shin_left_joint", "shin_right_joint"],
            effort_limit_sim=FW_SHIN_TAU_MAX,
            velocity_limit_sim=FW_SHIN_VEL_MAX,
            stiffness=FW_SHIN_KP,
            damping=FW_SHIN_KD,
            armature=0.01,
            friction=0.0,
        ),
        # Robstride RS02 -> foot (ankle).
        "foot": ImplicitActuatorCfg(
            joint_names_expr=["foot_left_joint", "foot_right_joint"],
            effort_limit_sim=FW_FOOT_TAU_MAX,
            velocity_limit_sim=FW_FOOT_VEL_MAX,
            stiffness=FW_FOOT_KP,
            damping=FW_FOOT_KD,
            armature=0.003,
            friction=0.0,
        ),
    },
)


@configclass
class ActionsCfg:
    """Sim-side action term that mirrors the firmware control path.

    Replaces stock :class:`mdp.JointPositionActionCfg` with our
    :class:`SlewLimitedJointPositionActionCfg`, which adds:

    * a per-tick setpoint slew clamp matching
      ``firmware/bebop-linux/config/bebop_v2.yaml::defaults.slew``, and
    * a single-tick action-delay buffer modelling one CAN round-trip
      between the policy emitting an action and physics applying it.

    The trained policy will only behave the same on hardware if these
    numbers stay aligned with what the firmware ships — see the
    ``FW_*`` constants at the top of this file.
    """

    joints_pos = SlewLimitedJointPositionActionCfg(
        asset_name="robot",
        joint_names=JOINT_NAMES_ALL,
        scale=0.8,
        use_default_offset=True,
        max_pos_step_per_tick=FW_MAX_POS_STEP_PER_TICK_RAD,
        action_delay_steps=FW_ACTION_DELAY_STEPS,
    )


@configclass
class ObservationsCfg:
    """Policy observation vector. Layout MUST match the firmware-side
    builder in ``firmware/bebop-linux/src/observation.rs``.

    ``base_ang_vel`` and ``projected_gravity`` are sourced from the
    explicit :class:`isaaclab.sensors.ImuCfg` sensor (mounted on
    ``base_link`` with identity offset, see :class:`BebopV2BaseEnvCfg`)
    rather than from ground-truth articulation-root data. This keeps
    the sim observation pipeline byte-compatible with the real-robot
    pipeline, which reads the BNO085's body-frame angular velocity and
    derives projected gravity from the body-frame fused quaternion
    (see ``firmware/bebop-linux/src/imu.rs`` for the orientation /
    mount-rotation contract).

    **Isaac Lab 3.0 note** — the 3.0 migration guide describes a
    forthcoming split where ``Imu`` becomes a lightweight
    accelerometer + gyro sensor and a new ``Pva`` sensor inherits the
    full-state pipe (``projected_gravity_b`` etc.). That split has not
    landed in our installed Isaac Lab build yet: ``isaaclab.sensors``
    still only exports ``Imu`` / ``ImuCfg`` / ``ImuData`` and
    ``isaaclab.envs.mdp`` still exposes ``imu_ang_vel`` /
    ``imu_projected_gravity`` (verified at runtime). When we upgrade
    to a release where ``Pva`` is the full-state sensor, the symbols
    below need to flip to ``PvaCfg`` / ``mdp.pva_*`` and
    ``SceneEntityCfg("pva")``. Quaternion order is already XYZW in
    3.0 (see the OffsetCfg comment in :class:`BebopV2BaseEnvCfg`).

    ``base_lin_vel`` has no IMU equivalent on the real robot — the
    firmware sends zeros until an external velocity estimator is wired
    in (see ``policy_runner.rs::SYNTHETIC_BASE_LIN_VEL``). For the
    standing task it is naturally near zero, so we keep the
    ground-truth feed during training with wide uniform noise as a
    light proxy for "we don't fully trust this number".
    """

    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=UniformNoiseCfg(n_min=-0.2, n_max=0.2))
        base_ang_vel = ObsTerm(
            func=mdp.imu_ang_vel,
            params={"asset_cfg": SceneEntityCfg("imu")},
            noise=UniformNoiseCfg(n_min=-0.3, n_max=0.3),
        )
        projected_gravity = ObsTerm(
            func=mdp.imu_projected_gravity,
            params={"asset_cfg": SceneEntityCfg("imu")},
            noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05),
        )
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=UniformNoiseCfg(n_min=-0.02, n_max=0.02))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=UniformNoiseCfg(n_min=-0.5, n_max=0.5))
        actions = ObsTerm(func=mdp.last_action)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})

    def __post_init__(self):
        self.policy = self.PolicyCfg()


@configclass
class EventCfg:
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (-2.0, 3.0),
            "operation": "add",
        },
    )

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.4, 1.2),
            "dynamic_friction_range": (0.4, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    # Actuator-gain domain randomization. The nominal stiffness/damping
    # values now match the firmware exactly (see FW_* constants above), so
    # this scales them by ±25% per env to give the policy a robustness
    # envelope covering motor temperature drift, manufacturing variance
    # between RS04 units, and the hand-tuned hold_gains in the YAML.
    randomize_stiffness_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.75, 1.25),
            "damping_distribution_params": (0.75, 1.25),
            "operation": "scale",
        },
    )

    # Per-joint-group pose randomization at reset.
    #
    # We use ``reset_joints_by_offset`` rather than ``reset_joints_by_scale``
    # because every joint's default position is 0.0, and any scale factor
    # times 0.0 is still 0.0 — the old (0.98, 1.02) scale produced *no*
    # variation at all and the policy was always seeing the same canonical
    # standing pose at every reset. Additive offsets give us actual
    # configuration diversity (knees bent, hips turned, ankles tilted, etc.)
    # so the policy learns to recover the upright pose from any
    # mechanically-reachable initial configuration.
    #
    # Each joint family's range is pinned to the firmware ground-truth
    # ``hard_limits`` in ``firmware/bebop-linux/config/bebop_v2.yaml``,
    # pulled in by the ``soft_joint_pos_limit_factor = 0.9`` margin set on
    # ``BEBOP_V2_CFG`` above. Sampling exactly at the soft limit means we
    # stay clear of the ``joint_pos_limits`` penalty on spawn while still
    # covering the entire reachable joint state space the policy will face
    # at deployment.
    #
    # Trade-off: with wide ranges, some episodes will spawn in poses the
    # robot cannot actually balance from (e.g. both knees at ~80°). Those
    # episodes terminate quickly via ``base_link_on_ground`` and contribute
    # short failure trajectories. That is the intended signal — the policy
    # learns the boundary of its recovery envelope rather than only the
    # easy "stand still from neutral" case. The ``joint_deviation`` /
    # ``base_height`` / symmetry rewards pull the pose back toward neutral
    # once balance is recovered.
    reset_hip_abduction = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["hip_abduction_left_joint", "hip_abduction_right_joint"],
            ),
            # YAML hard_limits: pos_min=-1.0, pos_max=1.0 (rad)
            # 0.9 × hard = ±0.9 rad (~±51.6°) lateral lean.
            "position_range": (-0.9, 0.9),
            "velocity_range": (-0.5, 0.5),
        },
    )
    reset_femur = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["femur_left_joint", "femur_right_joint"],
            ),
            # YAML hard_limits: pos_min=-1.0, pos_max=1.0 (rad)
            # 0.9 × hard = ±0.9 rad (~±51.6°) hip pitch.
            "position_range": (-0.9, 0.9),
            "velocity_range": (-0.5, 0.5),
        },
    )
    reset_shin = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["shin_left_joint", "shin_right_joint"],
            ),
            # YAML hard_limits: pos_min=-π/2, pos_max=π/2 (rad)
            # 0.9 × hard ≈ ±1.413 rad (~±81°) knee bend.
            "position_range": (-1.413, 1.413),
            "velocity_range": (-0.5, 0.5),
        },
    )
    reset_foot = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["foot_left_joint", "foot_right_joint"],
            ),
            # YAML hard_limits: pos_min=-0.8, pos_max=0.8 (rad)
            # 0.9 × hard = ±0.72 rad (~±41.3°) ankle tilt.
            "position_range": (-0.72, 0.72),
            "velocity_range": (-0.5, 0.5),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                # USD origin is at ground level, so zero xy/yaw offset keeps
                # feet on ground. Small roll/pitch variation pairs naturally
                # with the bent-knee / hip-turn joint offsets above so the
                # base isn't perfectly level just because the legs moved.
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "yaw": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (-0.05, 0.05),
                "pitch": (-0.05, 0.05),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    # Pushes are opt-in per experiment (see exp_flat_locomotion_v2.py).


@configclass
class RewardsCfg:
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    alive = RewTerm(func=mdp.is_alive, weight=2.0)

    joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-0.0002)
    action_l2 = RewTerm(func=mdp.action_l2, weight=-0.005)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    joint_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-6)

    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-0.5)
    joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)

    # Posture: encourage the policy to keep joints near their default neutral
    # pose and the torso at standing height. These prevent the "crouched
    # dinosaur" gait where the policy collapses into joint limits.
    joint_deviation = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES_ALL)},
    )
    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-2.0,
        params={
            "target_height": 0.6539092050794861,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    hip_abduction_symmetry = RewTerm(
        func=hip_abduction_symmetry_penalty,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    femur_symmetry = RewTerm(
        func=femur_symmetry_penalty,
        weight=-2.5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    shin_symmetry = RewTerm(
        func=shin_symmetry_penalty,
        weight=-2.5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    foot_symmetry = RewTerm(
        func=foot_symmetry_penalty,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # Penalize yaw motion only when the policy is commanded to stand still.
    undesired_yaw = RewTerm(
        func=undesired_yaw_penalty,
        weight=-1.0,
        params={"command_name": "base_velocity"},
    )
    # Soft "hold still when stable" terms (only active when the robot is upright
    # AND not commanded to move), much smaller weights so they don't suppress
    # walking motion under non-zero commands.
    leg_action_when_stable = RewTerm(
        func=leg_action_when_stable_penalty,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    leg_hold_reward = RewTerm(
        func=leg_position_hold_reward,
        weight=0.25,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    time_out = TermTerm(func=mdp.time_out, time_out=True)
    # Reset only when base_link drops near ground (fallen robot).
    base_link_ground_contact = TermTerm(
        func=base_link_on_ground,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "ground_height_threshold": 0.30,
        },
    )


@configclass
class CommandsCfg:
    # Neutral command term. Each experiment overrides the ranges and
    # ``rel_standing_envs`` to make this a stand-only or locomotion task.
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 12.0),
        debug_vis=True,
        rel_standing_envs=1.0,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
        ),
    )


@configclass
class BebopV2BaseEnvCfg(ManagerBasedRLEnvCfg):
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

        self.scene.robot = BEBOP_V2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

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

        # Body-frame IMU sensor. Mounted on ``base_link`` with an
        # identity offset so the sensor frame coincides with the body
        # frame — the simulated equivalent of the real-robot firmware
        # applying ``mount_quat_sensor_body`` to bring every BNO085
        # reading into the body frame before publishing (see
        # ``firmware/bebop-linux/src/imu.rs`` and the ``imu.mount:``
        # block in ``firmware/bebop-linux/config/bebop_v2.yaml``).
        #
        # Both pipelines therefore expose:
        #   * ``ang_vel_b`` — body-frame angular velocity (rad/s).
        #   * ``projected_gravity_b`` — world (0, 0, -1) projected into
        #     the body frame; ``z ≈ -1`` when upright.
        # ``update_period=0.0`` means refresh every physics tick so the
        # policy sees fresh IMU data on every control step.
        #
        # The OffsetCfg's ``rot`` field is **XYZW** in Isaac Lab 3.0
        # (migration from the old WXYZ convention) — identity is
        # ``(0, 0, 0, 1)`` and matches the firmware-side XYZW order
        # used by :struct:`crate::observation::ImuState`. See the
        # 3.0 migration note on the ObservationsCfg docstring for the
        # caveat about an upcoming Imu→Pva rename.
        self.scene.imu = ImuCfg(
            prim_path="{ENV_REGEX_NS}/Robot/Geometry/base_link",
            update_period=0.0,
            debug_vis=False,
            offset=ImuCfg.OffsetCfg(
                pos=(0.0, 0.0, 0.0),
                rot=(0.0, 0.0, 0.0, 1.0),  # XYZW identity (Isaac Lab 3.0)
            ),
        )

