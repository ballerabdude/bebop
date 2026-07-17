# pyright: reportMissingImports=false
"""Train a Bebop policy with rsl_rl PPO inside the Isaac Lab container.

Usage (from inside `bebop_isaac_lab`, with CWD = `/workspace/bebop_bot/sim`):

    /workspace/isaaclab/isaaclab.sh -p train_bebop.py \\
        --task Isaac-BebopV2-Standing-v0 \\
        --num_envs 512 --seed 42 --visualizer newton

The only registered task right now is ``Isaac-BebopV2-Standing-v0``
(see ``bebop_training/__init__.py``). The flat-balance, flat-locomotion,
and rough-terrain experiments were removed; re-add them as their own
registered tasks once each one has a working ``BebopV2*Cfg``.
"""

import argparse
import os
from datetime import datetime

# --- STEP 1: Launch App ---
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train Bebop Robot.")
parser.add_argument("--task", type=str, default="Isaac-BebopV2-Standing-v0", help="Task name.")
parser.add_argument("--num_envs", type=int, default=None, help="Override number of environments.")
parser.add_argument(
    "--num_mini_batches",
    type=int,
    default=None,
    help=(
        "Override PPO num_mini_batches. Scale together with --num_envs to "
        "keep minibatch size roughly constant (target ~32k samples/minibatch)."
    ),
)
parser.add_argument(
    "--entropy_coef",
    type=float,
    default=None,
    help=(
        "Override PPO entropy_coef (cfg default is 0.01). Lower it (e.g. 0.005) "
        "if the action std refuses to come down; raise it (e.g. 0.02) if the "
        "std collapses before the policy finds balance."
    ),
)
parser.add_argument(
    "--learning_rate",
    type=float,
    default=None,
    help=(
        "Override PPO learning_rate (cfg default is 1e-3). The adaptive "
        "schedule can only lower this from the base value, so this sets the "
        "ceiling. Try 1e-4 or 2.5e-4 if the policy converges too fast and "
        "collapses std before finding balance."
    ),
)
parser.add_argument("--seed", type=int, default=None, help="Random seed.")
parser.add_argument(
    "--max_iterations",
    type=int,
    default=None,
    help=(
        "Override total training iterations (cfg default is 10000). On a "
        "fresh run this is the absolute target; on --resume it is the number "
        "of ADDITIONAL iterations to run on top of the loaded checkpoint."
    ),
)
parser.add_argument(
    "--save_interval",
    type=int,
    default=None,
    help=(
        "Override how often (in iterations) a model_*.pt checkpoint is "
        "written (cfg default is 100). Use a larger value like 1000 for very "
        "long runs to keep fewer, more widely-spaced checkpoints."
    ),
)
parser.add_argument("--log_root", type=str, default="logs/rsl_rl", help="Root directory for logging")
parser.add_argument(
    "--resume",
    type=str,
    default=None,
    help="Path to a checkpoint .pt file (or run directory) to resume from.",
)
parser.add_argument(
    "--reset_action_std",
    type=float,
    default=None,
    help=(
        "After loading a resumed checkpoint, reset the actor's exploration "
        "std to this value (e.g. 0.8). Useful when fine-tuning a converged "
        "policy on a new task."
    ),
)
parser.add_argument(
    "--video",
    action="store_true",
    default=False,
    help=(
        "Record video clips of env 0 during training (beta2 convention). "
        "Forces --enable_cameras and rgb_array render mode. Clips land in "
        "<log_dir>/videos/train/. Pair with --video_interval and --video_length."
    ),
)
parser.add_argument(
    "--video_interval",
    type=int,
    default=2000,
    help=(
        "Interval between video recordings in ENV STEPS (not iterations). "
        "To record every N iterations, pass N * num_steps_per_env "
        "(num_steps_per_env defaults to 32, so --video_interval 1600 = "
        "every 50 iterations). Default 2000."
    ),
)
parser.add_argument(
    "--video_length",
    type=int,
    default=200,
    help="Number of env steps to record per video clip (default 200).",
)

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Video recording needs offscreen rendering (rgb_array) even in headless/no-viz
# runs. beta2's VideoRecorder resolves the capture backend from the active
# visualizer or the physics/renderer stack; PhysX selects the Kit camera path.
if args.video:
    args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# --- STEP 2: Imports ---
import gymnasium as gym
import torch

import bebop_training  # noqa: F401  (import for side-effect: registers the Gym tasks)

from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


def main():
    # 1. Setup Logging
    log_dir = os.path.join(args.log_root, args.task, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    print(f"[INFO] Logging experiments to: {log_dir}")

    # 2. Retrieve Config WITHOUT initializing the environment
    #    (avoids the double-init crash from gym.make calling AppLauncher again).
    task_spec = gym.spec(args.task)

    # Get Env Config
    cfg_entry_point = task_spec.kwargs.get("env_cfg_entry_point")
    if callable(cfg_entry_point):
        env_cfg = cfg_entry_point()
    else:
        raise ValueError(f"Env config entry point {cfg_entry_point} is not callable.")

    # Get Agent (PPO) Config
    agent_cfg_entry_point = task_spec.kwargs.get("rsl_rl_cfg_entry_point")
    if callable(agent_cfg_entry_point):
        agent_cfg = agent_cfg_entry_point()
    else:
        raise ValueError(f"Agent config entry point {agent_cfg_entry_point} is not callable.")

    # 3. Apply Overrides
    if args.num_envs:
        env_cfg.scene.num_envs = args.num_envs
    if args.seed is not None:
        env_cfg.seed = args.seed
    if args.num_mini_batches is not None:
        agent_cfg.algorithm.num_mini_batches = args.num_mini_batches
        print(f"[INFO] Override PPO num_mini_batches -> {args.num_mini_batches}")
    if args.entropy_coef is not None:
        agent_cfg.algorithm.entropy_coef = args.entropy_coef
        print(f"[INFO] Override PPO entropy_coef -> {args.entropy_coef}")
    if args.learning_rate is not None:
        agent_cfg.algorithm.learning_rate = args.learning_rate
        print(f"[INFO] Override PPO learning_rate -> {args.learning_rate}")
    if args.max_iterations is not None:
        agent_cfg.max_iterations = args.max_iterations
        print(f"[INFO] Override max_iterations -> {args.max_iterations}")
    if args.save_interval is not None:
        agent_cfg.save_interval = args.save_interval
        print(f"[INFO] Override save_interval -> {args.save_interval}")

    # 4. Create Environment (Only Once)
    print(f"[INFO] Creating environment for task: {args.task}")
    env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array" if args.video else None)

    # 4a. Optional video recording on a per-step cadence (beta2 convention).
    #     RecordVideo's step_trigger fires on env-step count; --video_interval
    #     is in steps. To land a clip at every checkpoint, set
    #     --video_interval = save_interval * num_steps_per_env.
    #
    #     Filenames use the iteration count (matching the rsl_rl checkpoint
    #     `model_<iter>.pt` convention) so a clip and its checkpoint line up:
    #       model_100.pt  <->  rl-video-iter-100.mp4
    if args.video:
        video_dir = os.path.join(log_dir, "videos", "train")
        steps_per_iter = agent_cfg.num_steps_per_env

        class _IterNamedRecordVideo(gym.wrappers.RecordVideo):
            """RecordVideo subclass naming clips by iteration, not step.

            rsl_rl checkpoints are ``model_<iter>.pt``; this names clips
            ``rl-video-iter-<iter>.mp4`` so a clip and its matching checkpoint
            share the same suffix and sort together.
            """

            def start_recording(self, video_name: str):
                # self.step_id is the env-step count at trigger time; convert
                # to the iteration count (0-indexed steps_per_iter).
                iter_num = self.step_id // steps_per_iter
                super().start_recording(f"rl-video-iter-{iter_num}")

        video_kwargs = {
            "video_folder": video_dir,
            "step_trigger": lambda step: step > 0 and step % args.video_interval == 0,
            "video_length": args.video_length,
        }
        print(f"[INFO] Recording videos: every {args.video_interval} steps "
              f"({args.video_interval // steps_per_iter} iter), "
              f"{args.video_length} steps/clip -> {video_dir}")
        env = _IterNamedRecordVideo(env, **video_kwargs)

    # 5. Wrap for RSL-RL
    env = RslRlVecEnvWrapper(env)

    # 6. Start Training
    print(f"[INFO] Starting PPO Runner on device: {env.device}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=env.device)

    # 6a. Optional resume from previous checkpoint
    if args.resume:
        resume_path = args.resume
        if os.path.isdir(resume_path):
            checkpoints = sorted(
                f for f in os.listdir(resume_path) if f.startswith("model_") and f.endswith(".pt")
            )
            if not checkpoints:
                raise FileNotFoundError(f"No model_*.pt found in {resume_path}")
            resume_path = os.path.join(resume_path, checkpoints[-1])
        print(f"[INFO] Resuming from checkpoint: {resume_path}")
        runner.load(resume_path)

        # Optionally reset the actor's exploration noise. The std collapses
        # during convergence; bumping it back up forces the policy to explore
        # again when fine-tuning on a new task.
        if args.reset_action_std is not None:
            try:
                # rsl_rl 5.x layout: PPO.actor (MLPModel) -> distribution
                # (GaussianDistribution). For std_type="scalar" the learnable
                # parameter is `std_param`; for "log" it is `log_std_param`.
                actor = runner.alg.actor  # type: ignore[attr-defined]
                dist = getattr(actor, "distribution", None)
                if dist is None:
                    raise AttributeError("actor has no `distribution` (deterministic model?)")
                target_std = float(args.reset_action_std)
                with torch.no_grad():
                    if hasattr(dist, "std_param"):
                        dist.std_param.fill_(target_std)
                    elif hasattr(dist, "log_std_param"):
                        import math
                        dist.log_std_param.fill_(math.log(target_std))
                    else:
                        raise AttributeError(
                            "distribution has neither `std_param` nor `log_std_param`"
                        )
                print(f"[INFO] Reset actor exploration std to {target_std}")
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] Failed to reset action std: {exc}")

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] Training crashed: {e}\n")
        raise
    finally:
        simulation_app.close()
