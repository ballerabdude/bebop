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

from .experiments.exp_mirror import BebopV2MirrorCfg
from .experiments.exp_standing import (
    BebopV2StandingCfg,
    BebopV2StandingFixedGainCfg,
    BebopV2StandingPushCfg,
)
from .agents.rsl_rl_ppo_cfg import BebopPPOBaseCfg, BebopPPOPushCfg

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

# Fixed-gain variant of the standing task: identical to Standing-v0 but the
# variable-impedance kp/kd action channels are frozen to fixed per-joint gains,
# so the policy only learns the 8 position targets. Recommended as the FIRST
# hardware stand to remove the kp/kd thrashing; switch back to Standing-v0 to
# re-introduce variable impedance once this stands cleanly. The action vector
# is still 24-dim, so the ONNX export and firmware decode are unchanged.
gym.register(
    id="Isaac-BebopV2-Standing-FixedGain-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": BebopV2StandingFixedGainCfg,
        "rsl_rl_cfg_entry_point": BebopPPOBaseCfg,
    },
)

# Push-recovery variant of the standing task: Standing-v0 plus mid-episode
# root-velocity shoves (with a magnitude curriculum) and a softened
# feet_straight so hip abduction is free to catch a sideways push. This is the
# NON-privileged replacement for the removed feet_load_symmetry penalty against
# the one-foot lean. Validate Standing-v0 converges + transfers FIRST, then
# train this. Uses BebopPPOPushCfg (higher entropy_coef for the harder task).
gym.register(
    id="Isaac-BebopV2-Standing-Push-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": BebopV2StandingPushCfg,
        "rsl_rl_cfg_entry_point": BebopPPOPushCfg,
    },
)

# Visual hardware duplicate: teleports sim joints from bebop-linux WebSocket
# telemetry. Not for RL training — run via ``mirror_bebop.py``.
gym.register(
    id="Isaac-BebopV2-Mirror-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": BebopV2MirrorCfg,
    },
)
