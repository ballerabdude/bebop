# /workspace/bebop_bot/play_bebop.py

"""Play a trained Bebop V2 policy.

Loads a registered Isaac Lab task and a trained rsl_rl checkpoint, then runs
the deterministic policy (no exploration noise) in a small visual env. Useful
for sanity-checking a training run before promoting it to the next curriculum
stage or exporting for deployment.

Example::

    /workspace/isaaclab/isaaclab.sh -p play_bebop.py \\
        --task Isaac-BebopV2-Flat-v0 \\
        --resume logs/rsl_rl/Isaac-BebopV2-Flat-v0/<run>/model_14000.pt

    # Pin the velocity command (e.g. test "walk forward at 0.3 m/s"):
    /workspace/isaaclab/isaaclab.sh -p play_bebop.py \\
        --task Isaac-BebopV2-Locomotion-v0 \\
        --resume logs/rsl_rl/Isaac-BebopV2-Locomotion-v0/<run> \\
        --cmd_lin_vel_x 0.3
"""

import argparse
import os
import sys

# --- STEP 1: Launch App ---
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play a trained Bebop policy.")
parser.add_argument(
    "--task",
    type=str,
    required=True,
    help="Registered task name (e.g. Isaac-BebopV2-Flat-v0).",
)
parser.add_argument(
    "--resume",
    type=str,
    required=True,
    help="Path to a checkpoint .pt file (or run directory; latest model_*.pt is used).",
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=1,
    help="Number of envs to spawn (default 1 for clean visual play).",
)
parser.add_argument("--seed", type=int, default=None, help="Random seed.")
parser.add_argument(
    "--steps",
    type=int,
    default=10_000_000,
    help="Maximum simulation steps before exiting (default effectively infinite).",
)

# Optional manual command overrides. If any --cmd_* arg is set, the velocity
# command sampler is replaced with a single fixed point so the policy is given
# that exact command for the entire play session.
parser.add_argument("--cmd_lin_vel_x", type=float, default=None, help="Forward velocity command (m/s).")
parser.add_argument("--cmd_lin_vel_y", type=float, default=None, help="Lateral velocity command (m/s).")
parser.add_argument("--cmd_ang_vel_z", type=float, default=None, help="Yaw-rate command (rad/s).")

parser.add_argument(
    "--disable_pushes",
    action="store_true",
    help="Disable random push disturbances during play (clean policy demo).",
)

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# --- STEP 2: Imports (must come AFTER AppLauncher) ---
import gymnasium as gym
import torch
import isaaclab.envs.mdp as mdp
import bebop_training  # registers Isaac-BebopV2-* tasks

from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


def _resolve_checkpoint(path: str) -> str:
    """Accept either a .pt file or a run directory containing model_*.pt files."""
    if os.path.isdir(path):
        ckpts = sorted(
            f for f in os.listdir(path) if f.startswith("model_") and f.endswith(".pt")
        )
        if not ckpts:
            raise FileNotFoundError(f"No model_*.pt found in {path}")
        return os.path.join(path, ckpts[-1])
    return path


def _maybe_override_commands(env_cfg) -> None:
    """If any ``--cmd_*`` arg is set, pin the velocity command to that exact triple."""
    cmd_overrides = (args.cmd_lin_vel_x, args.cmd_lin_vel_y, args.cmd_ang_vel_z)
    if all(v is None for v in cmd_overrides):
        return

    vx = args.cmd_lin_vel_x if args.cmd_lin_vel_x is not None else 0.0
    vy = args.cmd_lin_vel_y if args.cmd_lin_vel_y is not None else 0.0
    wz = args.cmd_ang_vel_z if args.cmd_ang_vel_z is not None else 0.0

    env_cfg.commands.base_velocity.rel_standing_envs = 0.0
    env_cfg.commands.base_velocity.ranges = mdp.UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(vx, vx),
        lin_vel_y=(vy, vy),
        ang_vel_z=(wz, wz),
    )
    print(f"[INFO] Pinned velocity command to ({vx:.2f}, {vy:.2f}, {wz:.2f})")


def main() -> int:
    # 1. Build env + agent configs from the gym registry (mirrors train_bebop.py).
    task_spec = gym.spec(args.task)

    cfg_entry_point = task_spec.kwargs.get("env_cfg_entry_point")
    if not callable(cfg_entry_point):
        raise ValueError(f"Env config entry point {cfg_entry_point} is not callable.")
    env_cfg = cfg_entry_point()

    agent_cfg_entry_point = task_spec.kwargs.get("rsl_rl_cfg_entry_point")
    if not callable(agent_cfg_entry_point):
        raise ValueError(f"Agent config entry point {agent_cfg_entry_point} is not callable.")
    agent_cfg = agent_cfg_entry_point()

    # 2. Apply play-time overrides.
    env_cfg.scene.num_envs = max(1, args.num_envs)
    if args.seed is not None:
        env_cfg.seed = args.seed

    _maybe_override_commands(env_cfg)

    if args.disable_pushes and hasattr(env_cfg.events, "push_robot"):
        env_cfg.events.push_robot = None
        print("[INFO] Disabled push disturbances.")

    # 3. Create env + wrap for rsl_rl.
    print(f"[INFO] Creating environment: {args.task} (num_envs={env_cfg.scene.num_envs})")
    env = gym.make(args.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    # 4. Build runner only to load the checkpoint and produce an inference policy.
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=env.device)

    ckpt = _resolve_checkpoint(args.resume)
    print(f"[INFO] Loading checkpoint: {ckpt}")
    runner.load(ckpt)

    # rsl_rl 5.x exposes a deterministic inference callable (mean of the
    # action distribution; no exploration noise).
    policy = runner.get_inference_policy(device=env.device)

    # 5. Play loop.
    # NOTE: rsl_rl 5.x's RslRlVecEnvWrapper returns the obs tensor directly
    # (shape: ``(num_envs, obs_dim)``), not ``(obs, extras)``. The step()
    # return arity varies across rsl_rl versions, so we index defensively.
    obs = env.get_observations()
    if isinstance(obs, tuple):  # older rsl_rl returned (obs, extras)
        obs = obs[0]

    print("[INFO] Running policy. Close the viewer or Ctrl-C to stop.")

    step = 0
    try:
        with torch.inference_mode():
            while simulation_app.is_running() and step < args.steps:
                actions = policy(obs)
                step_result = env.step(actions)
                obs = step_result[0]
                step += 1
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")

    env.close()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\n[ERROR] Play crashed: {e}\n")
        raise
    finally:
        simulation_app.close()
