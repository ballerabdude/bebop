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

from .bebop_v2_actions import VariableImpedanceJointActionCfg
from .bebop_v2_rewards import (
    foot_flat_reward,
    leg_position_hold_reward,
    shin_symmetry_penalty,
    torso_upright_via_legs_reward,
)
from .bebop_v2_terminations import base_link_on_ground


# ---------------------------------------------------------------------------
# Explicit joint order for Bebop V2 articulation. Must match
# ``firmware/bebop-linux/src/observation.rs::JOINT_NAMES``.
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# MIT-mode variable-impedance action contract.
#
# The policy outputs 24 floats per tick:
#   action[ 0: 8] -> position commands (-> default + pos_scale * raw)
#   action[ 8:16] -> kp commands       (-> [kp_min[j], kp_max[j]] affine)
#   action[16:24] -> kd commands       (-> [kd_min[j], kd_max[j]] affine)
#
# Per-joint kp/kd clamps are anchored to each joint's Robstride encoder
# envelope and the joint's tau_max. Values are wide enough for
# "stiff during stance, soft during swing / contact" modulation but never
# let the policy walk into the motor's electrical danger zone.
#
# These MUST mirror ``defaults.policy_gain_clamps`` (and any per-joint
# overrides) in ``firmware/bebop-linux/config/bebop_v2.yaml``. If you
# change a number on either side, change it on both — and retrain.
# ---------------------------------------------------------------------------
POLICY_KP_MIN = [5.0, 5.0, 20.0, 20.0, 10.0, 10.0, 5.0, 5.0]
POLICY_KP_MAX = [100.0, 100.0, 300.0, 300.0, 250.0, 250.0, 250.0, 250.0]
POLICY_KD_MIN = [0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]
POLICY_KD_MAX = [5.0, 5.0, 8.0, 8.0, 8.0, 8.0, 4.5, 4.5]


def _midpoint(lo: list[float], hi: list[float]) -> list[float]:
    return [0.5 * (a + b) for a, b in zip(lo, hi)]


# Midpoints used as the ImplicitActuatorCfg startup values. The action
# term overwrites these every tick via write_joint_stiffness_to_sim /
# write_joint_damping_to_sim, so the only role of these numbers is to
# avoid an undefined-PD-gain window at env spawn before the first tick.
_KP_MID = _midpoint(POLICY_KP_MIN, POLICY_KP_MAX)
_KD_MID = _midpoint(POLICY_KD_MIN, POLICY_KD_MAX)


# Per-group tau_max and vel_max — peak-force / peak-velocity envelope.
# These MUST mirror ``hard_limits.tau_max`` / ``hard_limits.vel_max``
# in ``firmware/bebop-linux/config/bebop_v2.yaml``.
FW_HIP_ABDUCTION_TAU_MAX = 84.0
FW_FEMUR_TAU_MAX = 42.0
FW_SHIN_TAU_MAX = 84.0
FW_FOOT_TAU_MAX = 17.0

FW_HIP_ABDUCTION_VEL_MAX = 12.0
FW_FEMUR_VEL_MAX = 12.0
FW_SHIN_VEL_MAX = 12.0
FW_FOOT_VEL_MAX = 20.0

# Position-channel slew + delay. ``FW_MAX_POS_STEP_PER_TICK_RAD`` mirrors
# ``defaults.slew.max_pos_step_per_tick`` in bebop_v2.yaml. At the 100 Hz
# policy tick, 0.015 rad/tick = 1.5 rad/s setpoint slew. Gain channels
# are NOT slew-clamped — variable impedance demands instantaneous gain
# changes between ticks. Delay = 1 tick approximates one CAN round-trip
# (TX -> Robstride PD -> encoder -> RX feedback).
FW_MAX_POS_STEP_PER_TICK_RAD = 0.020
FW_ACTION_DELAY_STEPS = 2


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
    # Per-joint Robstride actuator configs. ``stiffness`` / ``damping``
    # are seeded at the midpoint of each joint's policy clamp range (see
    # POLICY_KP_RANGES / POLICY_KD_RANGES above) but the action term
    # overwrites them every tick via write_joint_stiffness_to_sim /
    # write_joint_damping_to_sim. ``effort_limit_sim`` and
    # ``velocity_limit_sim`` mirror the YAML hard_limits.
    actuators={
        # Robstride RS04 -> hip abduction (lateral leg pitch).
        "hip_abduction": ImplicitActuatorCfg(
            joint_names_expr=[
                "hip_abduction_left_joint",
                "hip_abduction_right_joint",
            ],
            effort_limit_sim=FW_HIP_ABDUCTION_TAU_MAX,
            velocity_limit_sim=FW_HIP_ABDUCTION_VEL_MAX,
            stiffness=_KP_MID[0],
            damping=_KD_MID[0],
            armature=0.01,
            friction=0.0,
        ),
        # Robstride RS03 -> femur (hip pitch).
        "femur": ImplicitActuatorCfg(
            joint_names_expr=["femur_left_joint", "femur_right_joint"],
            effort_limit_sim=FW_FEMUR_TAU_MAX,
            velocity_limit_sim=FW_FEMUR_VEL_MAX,
            stiffness=_KP_MID[2],
            damping=_KD_MID[2],
            armature=0.005,
            friction=0.0,
        ),
        # Robstride RS04 -> shin (knee).
        "shin": ImplicitActuatorCfg(
            joint_names_expr=["shin_left_joint", "shin_right_joint"],
            effort_limit_sim=FW_SHIN_TAU_MAX,
            velocity_limit_sim=FW_SHIN_VEL_MAX,
            stiffness=_KP_MID[4],
            damping=_KD_MID[4],
            armature=0.01,
            friction=0.0,
        ),
        # Robstride RS02 -> foot (ankle).
        "foot": ImplicitActuatorCfg(
            joint_names_expr=["foot_left_joint", "foot_right_joint"],
            effort_limit_sim=FW_FOOT_TAU_MAX,
            velocity_limit_sim=FW_FOOT_VEL_MAX,
            stiffness=_KP_MID[6],
            damping=_KD_MID[6],
            armature=0.003,
            friction=0.0,
        ),
    },
)


@configclass
class ActionsCfg:
    """Sim-side action term mirroring the firmware control path.

    Uses :class:`VariableImpedanceJointActionCfg`, the MIT-mode 24-dim
    action: 8 joint positions + 8 kp + 8 kd per tick. The position
    channel goes through the firmware-matched slew clamp + 1-tick CAN
    round-trip delay; the gain channels are instantaneous (= the point
    of variable impedance). All firmware-matched numbers live in the
    ``POLICY_*`` and ``FW_*`` constants at the top of this file.
    """

    joints_pos = VariableImpedanceJointActionCfg(
        asset_name="robot",
        joint_names=JOINT_NAMES_ALL,
        pos_scale=0.8,
        use_default_offset=True,
        max_pos_step_per_tick=FW_MAX_POS_STEP_PER_TICK_RAD,
        action_delay_steps=FW_ACTION_DELAY_STEPS,
        kp_min=POLICY_KP_MIN,
        kp_max=POLICY_KP_MAX,
        kd_min=POLICY_KD_MIN,
        kd_max=POLICY_KD_MAX,
    )


@configclass
class ObservationsCfg:
    """Policy observation vector. Layout MUST match the firmware-side
    builder in ``firmware/bebop-linux/src/observation.rs``.

    ``base_ang_vel`` and ``projected_gravity`` come from the explicit
    :class:`isaaclab.sensors.ImuCfg` sensor (mounted on ``base_link``
    with identity offset) — the simulated mirror of the real-robot
    pipeline that reads body-frame BNO085 gyro and derives projected
    gravity from the body-frame fused quaternion.

    ``last_action`` is the full 24-dim raw policy output from the
    previous tick (positions + kp + kd channels), so the total
    observation dim is 36 - 8 + 24 = 52.

    ``base_lin_vel`` has no IMU equivalent on the real robot — the
    firmware sends zeros until an external velocity estimator is wired
    in. For the standing task it is naturally near zero, so we keep
    the ground-truth feed during training with wide uniform noise as a
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
    # Scale the ``base_link`` mass at sim startup. This is the cheap
    # "what if the top half of the robot were heavier?" knob — it
    # mutates the rigid-body inertial properties in PhysX directly, so
    # we don't have to round-trip through the URDF / USD to change the
    # carried payload. Set to (2.0, 2.0) to deterministically double
    # the base mass; widen the range (e.g. (1.5, 2.5)) to randomize
    # payload across envs. Runs before ``add_base_mass`` so the ±kg
    # jitter below stacks on top of the scaled value.
    scale_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (2.0, 2.0),
            "operation": "scale",
        },
    )

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

    # Per-joint-group pose randomization at reset. Ranges are
    # deliberately wide — about 50–75% of the USD physics envelope per
    # joint — so the policy never sees the same starting configuration
    # twice and cannot memorize a single "default rest pose" output.
    # Initial joint velocities are also non-trivial (±0.5 rad/s) so the
    # policy must read joint_vel from the observation, not assume zero.
    # The ``femur_deviation`` term and the ``shin_symmetry`` knee
    # penalty pull the policy toward a symmetric near-zero pose in
    # steady state; spawn diversity only changes *which corner* of the
    # recovery basin the episode starts in.
    reset_hip_abduction = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["hip_abduction_left_joint", "hip_abduction_right_joint"],
            ),
            "position_range": (-0.40, 0.40),    # ~23°  (USD limit ±45°)
            "velocity_range": (-0.5, 0.5),
        },
    )
    # Femur is the lateral hip-abduction axis on this articulation
    # (despite the name — the joint called ``hip_abduction_*_joint`` is
    # actually fore/aft). Reset range is wide so the policy sees splayed
    # spawn poses and must learn to recover from them; the strong
    # ``femur_deviation`` reward (overridden hard in the balance
    # experiment) still pulls the steady state back to zero.
    reset_femur = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["femur_left_joint", "femur_right_joint"],
            ),
            "position_range": (-0.30, 0.30),    # ~17°
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
            "position_range": (-0.60, 0.60),    # ~34°
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
            "position_range": (-0.35, 0.35),    # ~20°
            "velocity_range": (-0.5, 0.5),
        },
    )

    # Initial base randomization. **Critical for sim-to-real.** Without
    # this the policy converges to an open-loop "emit the default-pose
    # action every tick" solution, because the env always presents the
    # same upright, motionless start state — observations carry no
    # information and gradient descent zeros out their influence. With
    # randomized tilt + initial momentum the policy is forced to read
    # the IMU and joint state to figure out *which* way to recover.
    #
    # Magnitudes are modest for the balance task (rolling pitch < 5°,
    # angular velocity < 0.5 rad/s) so the policy doesn't have to
    # learn full fall-recovery — just "I'm slightly off, lean it back".
    # Locomotion experiments can widen these further.
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x":     (-0.05, 0.05),
                "y":     (-0.05, 0.05),
                "z":     (0.0,   0.02),
                "roll":  (-0.08, 0.08),   # ~4.6°
                "pitch": (-0.08, 0.08),   # ~4.6°
                "yaw":   (-0.3,  0.3),    # ~17°
            },
            "velocity_range": {
                "x":     (-0.15, 0.15),
                "y":     (-0.15, 0.15),
                "z":     (-0.05, 0.05),
                "roll":  (-0.4,  0.4),
                "pitch": (-0.4,  0.4),
                "yaw":   (-0.4,  0.4),
            },
        },
    )

    # Periodic mid-episode pushes. With the merged balance + robust
    # task the policy must handle disturbances *during* an episode, not
    # just initial-condition variety. Magnitudes here are sized so a
    # well-balanced robot can absorb the push with an ankle/hip strategy
    # in most ticks but occasionally has to take a recovery step — the
    # lateral kick (y) is the binding case because the biped support
    # polygon is narrower in y. 4–8 s interval means a 20 s episode
    # sees 3–5 pushes, enough that the policy can't memorize a
    # post-push settling sequence.
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(4.0, 8.0),
        params={
            "velocity_range": {
                "x": (-0.4, 0.4),
                "y": (-0.3, 0.3),
            },
        },
    )


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

    # Smoothness penalties applied to the FULL 24-dim action vector
    # (positions + kp + kd channels). action_l2 implicitly regularizes
    # gain magnitudes; action_rate_l2 implicitly discourages ping-ponging
    # kp/kd between ticks.
    joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-0.0003)
    action_l2 = RewTerm(func=mdp.action_l2, weight=-0.01)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.15)
    joint_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-4.0e-6)

    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-0.5)
    joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)

    # Posture: encourage joints near default and torso at standing height.
    # joint_deviation = RewTerm(
    #     func=mdp.joint_deviation_l1,
    #     weight=-0.05,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES_ALL)},
    # )

    # Targeted anti-splay penalty on the femur joints. Femur is the
    # lateral hip-abduction axis on this articulation (yes, the joint
    # named ``hip_abduction_*_joint`` is actually fore/aft pitch; the
    # naming is historical). Without this term the policy reward-hacks
    # the standing task by spreading both femurs out for a wider base
    # of support — it earns more ``alive``/``torso_upright`` reward
    # than the global ``joint_deviation`` term can offset, and the
    # resulting pose is brittle on the real robot (low-friction floor
    # → feet slip outward → fall). Experiments override the weight:
    # the balance stage cranks it; locomotion may want it relaxed if
    # the gait needs more lateral authority.
    femur_deviation = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["femur_left_joint", "femur_right_joint"],
            ),
        },
    )
    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-1.0,
        params={
            "target_height": 0.6539092050794861,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    leg_hold_reward = RewTerm(
        func=leg_position_hold_reward,
        weight=0.25,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # Torso upright via ankle + knee compensation. Reads body-frame
    # projected gravity from the IMU sensor and rewards the pitch
    # component being offset by the average foot AND shin joint angles.
    # Roll is penalised directly since neither joint has lateral authority.
    torso_upright_via_legs = RewTerm(
        func=torso_upright_via_legs_reward,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # Soles parallel to the ground. Composes with ``torso_upright_via_legs``:
    # the torso term pins one end of the pitch chain (torso vertical),
    # this term pins the other end (foot horizontal). For locomotion
    # experiments this should be relaxed (foot legitimately tilts during
    # swing / heel-toe contact) — override ``std`` upward or drop the
    # weight there.
    foot_flat = RewTerm(
        func=foot_flat_reward,
        weight=0.5,
        params={"asset_cfg": SceneEntityCfg("robot"), "std": 0.15},
    )

    # Knee (shin) left/right symmetry penalty. The right shin's joint
    # frame is mirrored about the sagittal plane in the USD
    # (``localRot0 = (0,-1,0,0)`` vs identity on the left, limits
    # flipped from -45..+90 to -90..+45), so a physically symmetric
    # crouch is ``shin_left ≈ -shin_right``. The reward function
    # squares ``(shin_left + shin_right)`` — read its docstring before
    # changing the sign here. This stops the policy from "balancing"
    # via an asymmetric crouch (one knee bent forward, one bent
    # backward) that looks fine in sim but collapses on the real robot
    # where the two shins don't track identical PD commands the same
    # way.
    #
    # We deliberately do NOT mirror the other joint pairs:
    #   - hip_abduction / femur: ``femur_deviation`` already pulls
    #     both toward zero, which is the same value in both mirrored
    #     frames and so encodes the symmetric standing pose for free.
    #   - foot: foot pitch is dominated by ``foot_flat``, which biases
    #     both feet to the same (horizontal) orientation using
    #     world-frame foot orientation, not joint angles, and so is
    #     also mirror-invariant.
    #
    # Locomotion experiments MUST override this to zero — walking
    # requires the legs to alternate, which is asymmetry by definition.
    shin_symmetry = RewTerm(
        func=shin_symmetry_penalty,
        weight=-1.5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    time_out = TermTerm(func=mdp.time_out, time_out=True)
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

        # Body-frame IMU sensor mounted on ``base_link`` with identity
        # offset. Both pipelines (sim and on-robot firmware) expose:
        #   * ``ang_vel_b``           — body-frame angular velocity (rad/s),
        #   * ``projected_gravity_b`` — world (0,0,-1) projected into body frame.
        # OffsetCfg rotation is XYZW in Isaac Lab 3.0 (identity = (0,0,0,1)).
        self.scene.imu = ImuCfg(
            prim_path="{ENV_REGEX_NS}/Robot/Geometry/base_link",
            update_period=0.0,
            debug_vis=False,
            offset=ImuCfg.OffsetCfg(
                pos=(0.0, 0.0, 0.0),
                rot=(0.0, 0.0, 0.0, 1.0),
            ),
        )
