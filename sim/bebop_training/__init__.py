# bebop_lab/__init__.py

import gymnasium as gym

# Import the specific Experiment Config
from .experiments.exp_flat_balance import BebopFlatBalanceCfg
from .experiments.exp_flat_balance_v2 import BebopV2FlatBalanceCfg
from .experiments.exp_flat_locomotion_v2 import BebopV2FlatLocomotionCfg

# Import the Agent/PPO Config
from .agents.rsl_rl_ppo_cfg import BebopPPOBaseCfg, BebopPPOLocomotionCfg

# Register the Flat Balance Task
gym.register(
    id="Isaac-Bebop-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": BebopFlatBalanceCfg,
        "rsl_rl_cfg_entry_point": BebopPPOBaseCfg,
    },
)

# Register the Flat Balance Task for Bebop V2 articulation. This task
# now subsumes the old stand-under-push (FlatRobust) experiment: the
# base EventCfg includes both initial-condition randomisation and
# periodic mid-episode pushes, so a single training stage produces a
# policy that holds still AND recovers from disturbances.
gym.register(
    id="Isaac-BebopV2-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": BebopV2FlatBalanceCfg,
        "rsl_rl_cfg_entry_point": BebopPPOBaseCfg,
    },
)

# Register the Flat Locomotion (velocity-tracking walk) Task for Bebop V2.
# Uses the locomotion-tuned PPO config (higher entropy, fresh action std).
gym.register(
    id="Isaac-BebopV2-Locomotion-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": BebopV2FlatLocomotionCfg,
        "rsl_rl_cfg_entry_point": BebopPPOLocomotionCfg,
    },
)