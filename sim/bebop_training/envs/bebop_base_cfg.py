# /workspace/bebop_bot/bebop_training/envs/bebop_base_cfg.py  
  
import math  
from dataclasses import MISSING  
  
import isaaclab.sim as sim_utils  
import isaaclab.terrains as terrain_gen  
  
from isaaclab.assets import ArticulationCfg, AssetBaseCfg  
from isaaclab.actuators import ImplicitActuatorCfg  
from isaaclab.envs import ManagerBasedRLEnvCfg  
from isaaclab.sensors import ContactSensorCfg  
from isaaclab.managers import EventTermCfg as EventTerm  
from isaaclab.managers import ObservationGroupCfg as ObsGroup  
from isaaclab.managers import ObservationTermCfg as ObsTerm  
from isaaclab.managers import RewardTermCfg as RewTerm  
from isaaclab.managers import SceneEntityCfg  
from isaaclab.managers import TerminationTermCfg as TermTerm  
from isaaclab.scene import InteractiveSceneCfg  
from isaaclab.utils import configclass  
  
from isaaclab.utils.noise import UniformNoiseCfg  
import isaaclab.envs.mdp as mdp  
import torch  
  
  
# ==============================================================================  
# EXPLICIT JOINT ORDER (must match deployment!)  
# ==============================================================================  
# This order is used consistently across training, deployment, and firmware.  
# DO NOT use regex patterns - explicit names ensure deterministic ordering.  
JOINT_NAMES_LEGS = ["left_hip_pitch", "right_hip_pitch", "left_knee_pitch", "right_knee_pitch"]  
JOINT_NAMES_WHEELS = ["left_wheel", "right_wheel"]  
JOINT_NAMES_ALL = JOINT_NAMES_LEGS + JOINT_NAMES_WHEELS  
  
  
# ==============================================================================  
# ROBOT CONFIGURATION  
# ==============================================================================  
  
BEBOP_CFG = ArticulationCfg(  
    spawn=sim_utils.UsdFileCfg(  
        usd_path="/workspace/bebop_bot/ros2_ws/src/Full-Robot-urdf-cleaned_description/urdf/bebop_robot/bebop_robot.usd",  
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
            solver_position_iteration_count=8,  # Increased from 4 to 8 for better stability 
            solver_velocity_iteration_count=4   # Increased from 0 to 4
        ),  
    ),  
    init_state=ArticulationCfg.InitialStateCfg(  
        pos=(0.0, 0.0, 0.60),   
        joint_pos={  
            # Explicit joint names for deterministic ordering  
            "left_hip_pitch": 0.0,  
            "right_hip_pitch": 0.0,  
            "left_knee_pitch": 0.0,  
            "right_knee_pitch": 0.0,  
            "left_wheel": 0.0,  
            "right_wheel": 0.0,  
        },  
        joint_vel={  
            "left_hip_pitch": 0.0,  
            "right_hip_pitch": 0.0,  
            "left_knee_pitch": 0.0,  
            "right_knee_pitch": 0.0,  
            "left_wheel": 0.0,  
            "right_wheel": 0.0,  
        },  
    ),  
    soft_joint_pos_limit_factor=0.9,  
    actuators={  
        "legs": ImplicitActuatorCfg(  
            # Explicit joint names - order matters for action mapping!  
            joint_names_expr=JOINT_NAMES_LEGS,  
            effort_limit_sim=20.0,  
            velocity_limit_sim=10.0,  
            stiffness=80.0,  
            damping=2.0,  
        ),  
        "wheels": ImplicitActuatorCfg(  
            # Explicit joint names - order matters for action mapping!  
            joint_names_expr=JOINT_NAMES_WHEELS,  
            effort_limit_sim=5.0,  
            velocity_limit_sim=20.0,  
            stiffness=0.0,  
            damping=5.0,  
        ),  
    },  
)  
  
  
def hip_symmetry_penalty(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:  
    """Penalize asymmetric hip positions (Left - Right)^2."""  
    robot = env.scene[asset_cfg.name]  
    diff = robot.data.joint_pos[:, 0] - robot.data.joint_pos[:, 1]  # L_hip - R_hip  
    return torch.square(diff)  
  
  
def knee_symmetry_penalty(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:  
    """Penalize asymmetric knee positions (Left - Right)^2."""  
    robot = env.scene[asset_cfg.name]  
    diff = robot.data.joint_pos[:, 2] - robot.data.joint_pos[:, 3]  # L_knee - R_knee  
    return torch.square(diff)  
  
  
def undesired_yaw_penalty(env, command_name: str) -> torch.Tensor:
    """Penalize yaw rotation when not commanded to turn."""
    robot = env.scene["robot"]
    yaw_vel = robot.data.root_ang_vel_b[:, 2]  # Z-axis angular velocity
    
    # Get commanded yaw velocity
    cmd = env.command_manager.get_command(command_name)
    cmd_yaw = cmd[:, 2]  # Commanded yaw velocity
    
    # Only penalize if commanded yaw is near zero
    is_standing = (cmd_yaw.abs() < 0.1).float()
    
    return (yaw_vel ** 2) * is_standing


def leg_action_when_stable_penalty(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Penalize leg actions when robot is already stable.
    
    This teaches the policy: "If you're balanced, don't move your legs!"
    Critical for sim-to-real where we want to hold current position.
    """
    robot = env.scene[asset_cfg.name]
    
    # Get projected gravity - [0, 0, -1] means perfectly upright
    proj_grav = robot.data.projected_gravity_b  # shape: (num_envs, 3)
    
    # Check if robot is "stable" (gravity mostly pointing down)
    # grav_z < -0.9 means tilted less than ~25 degrees
    is_upright = (proj_grav[:, 2] < -0.85).float()
    
    # Check if robot is not moving much (low angular velocity)
    ang_vel = robot.data.root_ang_vel_b  # shape: (num_envs, 3)
    ang_vel_magnitude = torch.norm(ang_vel, dim=1)
    is_still = (ang_vel_magnitude < 0.5).float()  # rad/s threshold
    
    # Combined stability check
    is_stable = is_upright * is_still
    
    # Get leg actions (first 4 actions: hip and knee joints)
    # actions shape: (num_envs, 6) -> [L_hip, R_hip, L_knee, R_knee, L_wheel, R_wheel]
    leg_actions = env.action_manager.action[:, :4]  # First 4 are legs
    leg_action_magnitude = torch.sum(torch.square(leg_actions), dim=1)
    
    # Only penalize leg actions when stable
    return leg_action_magnitude * is_stable


def leg_position_hold_reward(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Reward for keeping legs close to their current position when stable.
    
    This encourages the policy to "freeze" leg positions when balanced,
    rather than constantly adjusting toward a learned default stance.
    """
    robot = env.scene[asset_cfg.name]
    
    # Check stability
    proj_grav = robot.data.projected_gravity_b
    is_upright = (proj_grav[:, 2] < -0.85).float()
    
    # Get leg joint velocities (we want them near zero when stable)
    joint_vel = robot.data.joint_vel[:, :4]  # First 4 joints are legs
    leg_vel_magnitude = torch.sum(torch.square(joint_vel), dim=1)
    
    # Reward for low leg velocity when upright (inverted penalty)
    # exp(-vel^2) gives 1.0 when vel=0, decreasing as vel increases
    hold_reward = torch.exp(-0.5 * leg_vel_magnitude) * is_upright
    
    return hold_reward

# ==============================================================================  
# ENVIRONMENT MDP SETTINGS  
# ==============================================================================  
  
@configclass  
class ActionsCfg:  
    """Action configuration with explicit joint ordering and scaling documentation.  
      
    ╔══════════════════════════════════════════════════════════════════════════════╗  
    ║                       ACTION VECTOR SPECIFICATION                            ║  
    ╚══════════════════════════════════════════════════════════════════════════════╝  
      
    Action Vector Layout (6 values):  
    ┌─────────┬──────────────────┬─────────────┬─────────────────────────────────┐  
    │ Index   │ Joint            │ Control     │ How Action is Applied           │  
    ├─────────┼──────────────────┼─────────────┼─────────────────────────────────┤  
    │ [0]     │ left_hip_pitch   │ Position    │ target = default + action * 0.5 │  
    │ [1]     │ right_hip_pitch  │ Position    │ target = default + action * 0.5 │  
    │ [2]     │ left_knee_pitch  │ Position    │ target = default + action * 0.5 │  
    │ [3]     │ right_knee_pitch │ Position    │ target = default + action * 0.5 │  
    │ [4]     │ left_wheel       │ Velocity    │ target = action * 20.0 rad/s    │  
    │ [5]     │ right_wheel      │ Velocity    │ target = action * 20.0 rad/s    │  
    └─────────┴──────────────────┴─────────────┴─────────────────────────────────┘  
    """  
    legs_pos = mdp.JointPositionActionCfg(  
        asset_name="robot",  
        joint_names=JOINT_NAMES_LEGS,  # Explicit order: [L_hip, R_hip, L_knee, R_knee]  
        scale=0.8,                      # radians: action * 0.8 = target offset (±0.8 rad max)  
        use_default_offset=True,        # target = default_pos + (action * scale)  
    )  
    wheels_vel = mdp.JointVelocityActionCfg(  
        asset_name="robot",  
        joint_names=JOINT_NAMES_WHEELS,  # Explicit order: [L_wheel, R_wheel]  
        scale=20.0,                       # rad/s: action * 20.0 = velocity target  
        use_default_offset=False,         # target = action * scale (no offset)  
    )  
  
  
@configclass  
class ObservationsCfg:  
    """Observation configuration - SIM-TO-REAL OPTIMIZED.  
      
    ╔══════════════════════════════════════════════════════════════════════════════╗  
    ║                    OBSERVATION VECTOR SPECIFICATION                          ║  
    ║                                                                              ║  
    ║  IMPORTANT: empirical_normalization = False in rsl_rl_ppo_cfg.py             ║  
    ║  NOISE: Significantly increased to match real-world vibration and latency.   ║  
    ╚══════════════════════════════════════════════════════════════════════════════╝  
      
    Observation Vector Layout (30 values total):  
    ┌─────────┬────────────────────┬────────────────┬─────────────────────────────┐  
    │ Index   │ Name               │ Units          │ Typical Range               │  
    ├─────────┼────────────────────┼────────────────┼─────────────────────────────┤  
    │ [0:3]   │ base_lin_vel       │ m/s (body)     │ [-2, 2] typical             │  
    │ [3:6]   │ base_ang_vel       │ rad/s (body)   │ [-5, 5] typical             │  
    │ [6:9]   │ projected_gravity  │ normalized     │ [-1, 1] always              │  
    │ [9:15]  │ joint_pos_rel      │ radians        │ relative to default (0.0)   │  
    │ [15:21] │ joint_vel_rel      │ rad/s          │ [-10, 10] typical           │  
    │ [21:27] │ last_action        │ raw NN output  │ [-1, 1] unbounded           │  
    │ [27:30] │ velocity_commands  │ [m/s, m/s, r/s]│ x,y lin + z ang vel         │  
    └─────────┴────────────────────┴────────────────┴─────────────────────────────┘  
    """  
    @configclass  
    class PolicyCfg(ObsGroup):  
        # [0:3] Linear velocity in body frame (m/s)  
        # Increased noise: Real estimation from wheel odometry + IMU is noisy
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=UniformNoiseCfg(n_min=-0.2, n_max=0.2))  
          
        # [3:6] Angular velocity in body frame (rad/s)  
        # Increased noise: MEMS Gyros vibrate significantly on stiff robots
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=UniformNoiseCfg(n_min=-0.3, n_max=0.3))  
          
        # [6:9] Gravity vector in body frame (normalized to unit length)  
        # Gravity is usually clean if constructed from Quats, but small noise helps robustness
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05))  
          
        # [9:15] Joint positions RELATIVE to default (radians)  
        # Encoder noise
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=UniformNoiseCfg(n_min=-0.02, n_max=0.02))  
          
        # [15:21] Joint velocities (rad/s)  
        # Velocity differentiation from encoders is VERY noisy in real life
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=UniformNoiseCfg(n_min=-0.5, n_max=0.5))  
          
        # [21:27] Previous action (raw NN output)  
        actions = ObsTerm(func=mdp.last_action)  
          
        # [27:30] Commanded velocity [vx (m/s), vy (m/s), wz (rad/s)]  
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})  
  
  
    def __post_init__(self):  
        self.policy = self.PolicyCfg()  
  
  
@configclass  
class EventCfg:  
    """Domain Randomization - The Key to Sim-to-Real"""

    # 1. Mass Randomization: Simulates battery weight diffs, cable drag, etc.
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (-2.0, 3.0), # Wider range (-2kg to +3kg)
            "operation": "add",
        },
    )

    # 2. Friction Randomization: Real floors are never perfect (slippery vs sticky)
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.4, 1.2),
            "dynamic_friction_range": (0.4, 1.0),
            "restitution_range": (0.0, 0.0), # No bouncing
            "num_buckets": 64,
        },
    )

    # 3. Actuator Gains Randomization: Simulates heating, voltage drop, motor variance
    # This is critical for handling the "Latency/Backlash" feel of real motors
    randomize_stiffness_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.7, 1.3), # +/- 30% variance
            "damping_distribution_params": (0.7, 1.3),   # +/- 30% variance
            "operation": "scale",
        },
    )

    # 4. Joint Position Randomization: CRITICAL for sim-to-real!
    # Wide range teaches policy to balance from ANY starting leg configuration,
    # not just the "default" stance. This prevents the robot from immediately
    # trying to move legs to a learned pose when policy is activated.
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.0, 2.0),  # Much wider range (was 0.5-1.5)
            "velocity_range": (-1.0, 1.0), # More initial velocity noise
        },
    )

    # 5. Frequent Pushing: Teaches recovery from "lag spikes" or slips
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(3.0, 6.0), # Push every 3-6 seconds (Frequent!)
        params={"velocity_range": {"x": (-1.0, 1.0), "y": (-0.5, 0.5)}},
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.5, 0.5),   
                "y": (-0.5, 0.5),   
                "yaw": (-3.14, 3.14),  
                "z": (0.0, 0.6),  # dropin support in sim
                "roll": (-0.15, 0.15),  # Start with more tilt
                "pitch": (-0.15, 0.15), # Start with more tilt
            },  
            "velocity_range": {  
                "x": (-0.5, 0.5),  
                "y": (-0.5, 0.5),  
                "z": (-0.5, 0.5),  
                "roll": (-0.5, 0.5),  
                "pitch": (-0.5, 0.5),  
                "yaw": (-0.5, 0.5),  
            },  
        },  
    )  

  
@configclass  
class RewardsCfg:  
    """Reward terms for the environment."""  
      
    # -- Task Rewards --  
      
    # 1. Track Linear Velocity (Forward/Backward)  
    track_lin_vel_xy = RewTerm(  
        func=mdp.track_lin_vel_xy_exp,   
        weight=2.0,  # Increased weight to prioritize tracking
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)}  
    )  
      
    # 2. Track Angular Velocity (Turning)  
    track_ang_vel_z = RewTerm(  
        func=mdp.track_ang_vel_z_exp,   
        weight=1.0,   
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)}  
    )  
  
    # 3. Survival Reward  
    alive = RewTerm(func=mdp.is_alive, weight=2.0)  # Increased survival incentive
  
    # -- Efficiency Penalties (The "Laziness" Constraints) --  
      
    # 4. Torque Penalty  
    joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-0.0002)  
  
    # 5. Action Magnitude Penalty  
    action_l2 = RewTerm(func=mdp.action_l2, weight=-0.005)  
      
    # 6. Smoothness / Action Rate  
    # INCREASED PENALTY: Critical for stopping oscillation/twitching on real hardware
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.05)  
      
    # 7. Joint Acceleration  
    joint_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-6)  
  
    # -- Stability Penalties --  
  
    # 8. Vertical Velocity Penalty  
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)  
  
    # 9. Orientation Penalty  
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-0.5)  
      
    # 10. Joint Limits  
    joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)  
  
    # -- Symmetry (Forces L == R, but allows L & R to be anything) --  
    hip_symmetry = RewTerm(  
        func=hip_symmetry_penalty,  
        weight=-2.5,   
        params={"asset_cfg": SceneEntityCfg("robot")},  
    )  
  
    knee_symmetry = RewTerm(  
        func=knee_symmetry_penalty,  
        weight=-2.5,  
        params={"asset_cfg": SceneEntityCfg("robot")},  
    )  
  
    undesired_yaw = RewTerm(
        func=undesired_yaw_penalty,
        weight=-2.0,
        params={"command_name": "base_velocity"}
    )

    # -- Hold Position When Stable (Critical for Sim-to-Real!) --
    # These rewards teach: "If you're already balanced, DON'T move your legs"
    
    # Penalize leg actions when robot is already stable
    # This stops the policy from always trying to move to a "learned default stance"
    leg_action_when_stable = RewTerm(
        func=leg_action_when_stable_penalty,
        weight=-3.0,  # Strong penalty
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    
    # Reward for keeping legs still when balanced
    leg_hold_reward = RewTerm(
        func=leg_position_hold_reward,
        weight=1.0,  # Positive reward for holding position
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
  
  
@configclass  
class TerminationsCfg:  
    time_out = TermTerm(func=mdp.time_out, time_out=True)  
    
    # This term requires a sensor named "contact_forces" to exist in the scene!  
    base_contact = TermTerm(  
        func=mdp.illegal_contact,  
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_link"), "threshold": 1.0},  
    )  
  
    # Kill episode if robot gets too low  
    root_height_below_minimum = TermTerm(  
        func=mdp.root_height_below_minimum,  
        params={"minimum_height": 0.35},   
    )  
  
  
@configclass  
class CommandsCfg:  
    base_velocity = mdp.UniformVelocityCommandCfg(  
        asset_name="robot",  
        resampling_time_range=(5.0, 10.0),  
        debug_vis=True,  
          
        # --- CRITICAL FIX FOR TELEOP ---  
        # 20% of the environments will be commanded to have Velocity = [0,0,0].  
        # This teaches the policy specifically how to balance in place without drifting.  
        rel_standing_envs=0.2,   
          
        ranges=mdp.UniformVelocityCommandCfg.Ranges(  
            lin_vel_x=(-1.0, 1.0),   
            lin_vel_y=(0.0, 0.0),   
            ang_vel_z=(-1.0, 1.0)  
        ),  
    )  
  
  
@configclass  
class BebopBaseEnvCfg(ManagerBasedRLEnvCfg):  
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
          
        self.scene.robot = BEBOP_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")  
          
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
            spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))  
        )  
  
        # Contact Sensor  
        self.scene.contact_forces = ContactSensorCfg(  
            prim_path="{ENV_REGEX_NS}/Robot/.*",   
            history_length=3,   
            track_air_time=False  
        )