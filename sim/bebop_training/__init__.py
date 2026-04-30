# bebop_lab/__init__.py

import gymnasium as gym

# Import the specific Experiment Config
from .experiments.exp_flat_balance import BebopFlatBalanceCfg
from .experiments.exp_flat_balance_v2 import BebopV2FlatBalanceCfg
from .experiments.exp_flat_balance_robust_v2 import BebopV2FlatBalanceRobustCfg
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

# Register the Flat Balance (stand-only) Task for Bebop V2 articulation.
gym.register(
    id="Isaac-BebopV2-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": BebopV2FlatBalanceCfg,
        "rsl_rl_cfg_entry_point": BebopPPOBaseCfg,
    },
)

# Register the Robust-Balance (stand-under-push) Task for Bebop V2.
# Warm-start this from a converged Isaac-BebopV2-Flat-v0 checkpoint so the
# policy only has to learn push recovery, not standing.
gym.register(
    id="Isaac-BebopV2-FlatRobust-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": BebopV2FlatBalanceRobustCfg,
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