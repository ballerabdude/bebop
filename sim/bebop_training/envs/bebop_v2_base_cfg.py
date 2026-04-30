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
        usd_path="/workspace/bebop_bot/usd/bebopv2/bebopv2.usda",
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
    # Per-joint Robstride actuator configs (PD gains + torque/velocity limits)
    # adapted from kscalelabs/kbot-v2 metadata. Keeps sim PD behaviour close to
    # the MIT-mode controller running on the real motors.
    actuators={
        # Robstride RS04 -> hip abduction + shin (knee). Stiffer hip abduction
        # for lateral balance.
        "rs04": ImplicitActuatorCfg(
            joint_names_expr=[
                "hip_abduction_left_joint",
                "hip_abduction_right_joint",
                "shin_left_joint",
                "shin_right_joint",
            ],
            effort_limit_sim=84.0,
            velocity_limit_sim=26.0,
            stiffness={
                "hip_abduction_.*": 200.0,
                "shin_.*": 150.0,
            },
            damping={
                "hip_abduction_.*": 8.0,
                "shin_.*": 8.0,
            },
            armature=0.01,
            friction=0.0,
        ),
        # Robstride RS03 -> femur (hip pitch).
        "rs03": ImplicitActuatorCfg(
            joint_names_expr=["femur_left_joint", "femur_right_joint"],
            effort_limit_sim=42.0,
            velocity_limit_sim=24.0,
            stiffness=100.0,
            damping=5.0,
            armature=0.005,
            friction=0.0,
        ),
        # Robstride RS02 -> foot (ankle).
        "rs02": ImplicitActuatorCfg(
            joint_names_expr=["foot_left_joint", "foot_right_joint"],
            effort_limit_sim=11.9,
            velocity_limit_sim=43.0,
            stiffness=40.0,
            damping=2.0,
            armature=0.003,
            friction=0.0,
        ),
    },
)


@configclass
class ActionsCfg:
    joints_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=JOINT_NAMES_ALL,
        scale=0.8,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=UniformNoiseCfg(n_min=-0.2, n_max=0.2))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=UniformNoiseCfg(n_min=-0.3, n_max=0.3))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05))
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

    randomize_stiffness_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.7, 1.3),
            "damping_distribution_params": (0.7, 1.3),
            "operation": "scale",
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            # Standing task: reset close to default pose/velocity.
            "position_range": (0.98, 1.02),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                # USD origin is at ground level, so zero offset keeps feet on ground.
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "yaw": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
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

        # Explicit IMU sensor mounted on base_link for orientation/angular velocity sensing.
        self.scene.imu = ImuCfg(
            prim_path="{ENV_REGEX_NS}/Robot/Geometry/base_link",
            update_period=0.0,
            debug_vis=False,
        )

