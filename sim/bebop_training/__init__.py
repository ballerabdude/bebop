# bebop_training/__init__.py
#
# Single-task package: registers the standing-balance task and its PPO
# config with Gym so ``train_bebop.py`` / ``play_bebop.py`` can resolve
# the task by id. The flat-balance, flat-locomotion, and rough-terrain
# experiments were removed in May 2026 to focus the codebase on a
# single, validatable hardware-deployable policy. Re-add new
# experiments here as separate ``gym.register`` blocks once each one
# has its own working ``BebopV2*Cfg``.

import gymnasium as gym

from .experiments.exp_standing import BebopV2StandingCfg
from .agents.rsl_rl_ppo_cfg import BebopPPOBaseCfg

# Minimal "just stand" baseline for Bebop V2. Stripped of every
# domain-randomization, action-shaping, and reward-shaping bell and
# whistle so each subsequent v1, v2, ... experiment can re-add exactly
# one feature at a time and attribute its effect on the final policy.
gym.register(
    id="Isaac-BebopV2-Standing-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": BebopV2StandingCfg,
        "rsl_rl_cfg_entry_point": BebopPPOBaseCfg,
    },
)
