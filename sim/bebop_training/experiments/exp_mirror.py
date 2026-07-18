"""Visual pose-mirror environment for Bebop V2.

Spawns a single robot with gravity disabled and passive actuators. A
companion script (`mirror_bebop.py`) teleports joint and base orientation
state from live hardware telemetry over the runtime WebSocket API.
"""

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
import isaaclab.envs.mdp as mdp

from isaaclab.actuators import DCMotorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as TermTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.configclass import configclass

from .exp_standing import (
    BEBOP_V2_STANDING_CFG,
    JOINT_ARMATURE_FOOT,
    JOINT_ARMATURE_HIP_ABD,
    JOINT_ARMATURE_HIP_FLEX,
    JOINT_ARMATURE_KNEE_FLEX,
    JOINT_FRICTION_FOOT,
    JOINT_FRICTION_HIP_ABD,
    JOINT_FRICTION_HIP_FLEX,
    JOINT_FRICTION_KNEE_FLEX,
    JOINT_NAMES_ALL,
    MOTOR_NOLOAD_VEL_RS02,
    MOTOR_NOLOAD_VEL_RS03,
    MOTOR_NOLOAD_VEL_RS04,
    MOTOR_STALL_TORQUE_RS02,
    MOTOR_STALL_TORQUE_RS03,
    MOTOR_STALL_TORQUE_RS04,
)

# Base height for a flat stance with the Jul 2026 feet (sole 0.7302 m
# below base_link + 0.035 m clearance). Keep in sync with LIFT_Z in
# sim/scripts/post_import_bebopv2.py.
MIRROR_BASE_POS = (0.0, 0.0, 0.765)


class MirrorNoOpAction(ActionTerm):
    """Action term that consumes no policy output."""

    cfg: MirrorNoOpActionCfg

    def __init__(self, cfg: MirrorNoOpActionCfg, env):
        super().__init__(cfg, env)
        self._raw_actions = torch.zeros(self.num_envs, 0, device=self.device)

    @property
    def action_dim(self) -> int:
        return 0

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._raw_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        pass

    def apply_actions(self) -> None:
        pass

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        pass


@configclass
class MirrorNoOpActionCfg(ActionTermCfg):
    class_type: type = MirrorNoOpAction


BEBOP_V2_MIRROR_CFG = BEBOP_V2_STANDING_CFG.replace(
    spawn=BEBOP_V2_STANDING_CFG.spawn.replace(
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=MIRROR_BASE_POS,
        joint_pos={joint_name: 0.0 for joint_name in JOINT_NAMES_ALL},
        joint_vel={joint_name: 0.0 for joint_name in JOINT_NAMES_ALL},
    ),
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
            stiffness=0.0,
            damping=0.0,
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
            stiffness=0.0,
            damping=0.0,
            armature=JOINT_ARMATURE_HIP_ABD,
            friction=JOINT_FRICTION_HIP_ABD,
        ),
        "knee_flexion": DCMotorCfg(
            joint_names_expr=[
                "knee_flexion_left_joint",
                "knee_flexion_right_joint",
            ],
            saturation_effort=MOTOR_STALL_TORQUE_RS03,
            effort_limit_sim=42.0,
            velocity_limit_sim=12.0,
            velocity_limit=MOTOR_NOLOAD_VEL_RS03,
            stiffness=0.0,
            damping=0.0,
            armature=JOINT_ARMATURE_KNEE_FLEX,
            friction=JOINT_FRICTION_KNEE_FLEX,
        ),
        "foot": DCMotorCfg(
            joint_names_expr=[
                "foot_left_joint",
                "foot_right_joint",
            ],
            saturation_effort=MOTOR_STALL_TORQUE_RS02,
            effort_limit_sim=6.0,
            velocity_limit_sim=20.0,
            velocity_limit=MOTOR_NOLOAD_VEL_RS02,
            stiffness=0.0,
            damping=0.0,
            armature=JOINT_ARMATURE_FOOT,
            friction=JOINT_FRICTION_FOOT,
        ),
    },
)


def _zero_reward(env) -> torch.Tensor:
    return torch.zeros(env.num_envs, device=env.device)


@configclass
class MirrorActionsCfg:
    noop = MirrorNoOpActionCfg(asset_name="robot")


@configclass
class MirrorObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=JOINT_NAMES_ALL, preserve_order=True
                )
            },
        )

        def __post_init__(self):
            self.enable_corruption = False

    def __post_init__(self):
        self.policy = self.PolicyCfg()


@configclass
class MirrorRewardsCfg:
    noop = RewTerm(func=_zero_reward, weight=0.0)


@configclass
class MirrorTerminationsCfg:
    time_out = TermTerm(func=mdp.time_out, time_out=True)


@configclass
class BebopV2MirrorCfg(ManagerBasedRLEnvCfg):
    """Single-env visual duplicate fed by hardware telemetry."""

    decimation = 1
    episode_length_s = 3600.0

    scene = InteractiveSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)
    observations = MirrorObservationsCfg()
    actions = MirrorActionsCfg()
    rewards = MirrorRewardsCfg()
    terminations = MirrorTerminationsCfg()

    def __post_init__(self):
        self.viewer.eye = [2.0, 2.0, 1.5]
        self.viewer.lookat = [0.0, 0.0, 0.6]

        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = True

        self.scene.robot = BEBOP_V2_MIRROR_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

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
