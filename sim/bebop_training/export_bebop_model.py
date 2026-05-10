#!/usr/bin/env python3
"""Export a trained Bebop V2 RSL-RL checkpoint to ONNX.

The resulting ``policy.onnx`` is the artefact consumed by:

  - ``firmware/bebop-linux`` (loads it via the ``ort`` crate in
    ``src/policy.rs``),
  - ``ros2/src/bebop_pilot/policy_runner`` (loads it via ``onnxruntime``).

Both consumers expect a plain MLP that maps an observation vector to an
action vector.

RSL-RL v4 (Isaac Lab v3 / Isaac Sim v6) split the actor into ``MLPModel``
(``mlp`` + ``obs_normalizer`` + ``distribution`` submodules) reachable via
``runner.alg.get_policy()``. The legacy ``isaaclab_rl.export_policy_as_onnx``
helper expected a ``.actor``/``.student`` attribute and crashes on the new
layout, so we drive the runner's own ``export_policy_to_onnx`` instead — it
calls the policy's ``as_onnx()`` to emit a clean ``obs -> actions`` graph
without the critic / distribution noise heads.

Usage (from inside the Isaac Lab container)::

    /workspace/isaaclab/isaaclab.sh -p \\
        /workspace/bebop_bot/sim/bebop_training/export_bebop_model.py \\
        --checkpoint logs/rsl_rl/Isaac-BebopV2-Flat-v0/<run>/model_<N>.pt

Task is auto-derived from the checkpoint path (the segment after
``logs/rsl_rl/``); pass ``--task`` to override.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# --- Isaac Lab app must be launched before any isaaclab/rsl_rl imports.
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Export a trained Bebop policy to ONNX.")
parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="Path to an RSL-RL .pt checkpoint (e.g. logs/.../model_500.pt).",
)
parser.add_argument(
    "--task",
    type=str,
    default=None,
    help="Gym task ID. Auto-detected from the checkpoint path if omitted.",
)
parser.add_argument(
    "--output",
    type=str,
    default=None,
    help="Output path for policy.onnx (default: same dir as the checkpoint).",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Force headless: the export does not need a viewer and launching the GUI
# adds ~30s of cold-start time.
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch  # noqa: F401  (imported for side-effect: registers CUDA before isaaclab)
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import bebop_training  # noqa: F401  (registers Isaac-BebopV2-* gym tasks)


def derive_task_from_checkpoint(checkpoint_path: Path) -> str:
    """Pull the task ID out of ``logs/rsl_rl/<task>/<run>/model_*.pt``.

    Raises a clear error rather than silently guessing if the layout doesn't
    match — a wrong task picks up the wrong env_cfg, which silently produces
    an ONNX with the wrong I/O shape.
    """
    parts = checkpoint_path.resolve().parts
    try:
        idx = parts.index("rsl_rl")
    except ValueError as e:
        raise SystemExit(
            f"Could not infer task from {checkpoint_path}: path does not "
            "contain a 'rsl_rl' segment. Pass --task explicitly."
        ) from e
    if idx + 1 >= len(parts):
        raise SystemExit(
            f"Could not infer task from {checkpoint_path}: no segment after "
            "'rsl_rl/'. Pass --task explicitly."
        )
    return parts[idx + 1]


def main() -> None:
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")

    task = args.task or derive_task_from_checkpoint(checkpoint_path)
    out_dir = Path(args.output).expanduser().resolve() if args.output else checkpoint_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("BEBOP POLICY EXPORT (ONNX)")
    print("=" * 70)
    print(f"  task:        {task}")
    print(f"  checkpoint:  {checkpoint_path}")
    print(f"  output dir:  {out_dir}")
    print("-" * 70)

    task_spec = gym.spec(task)
    env_cfg = task_spec.kwargs.get("env_cfg_entry_point")()
    agent_cfg = task_spec.kwargs.get("rsl_rl_cfg_entry_point")()

    # Single env is enough to instantiate the runner + load weights.
    env_cfg.scene.num_envs = 1
    env = RslRlVecEnvWrapper(gym.make(task, cfg=env_cfg, render_mode=None))

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=str(out_dir), device="cpu")
    runner.load(str(checkpoint_path))

    # rsl_rl v4 split the policy into MLPModel + Distribution and removed
    # the standalone `export_policy_as_onnx` helper's compatibility surface.
    # The runner's own exporter is the only correct path: it calls the
    # policy's `as_onnx()` which emits a clean obs -> actions graph and
    # drops the distribution noise / critic heads.
    if not hasattr(runner, "export_policy_to_onnx"):
        raise SystemExit(
            "runner.export_policy_to_onnx is missing — this script targets "
            "rsl_rl v4+. Upgrade rsl_rl or pin an older script if you need "
            "the legacy ActorCritic exporter."
        )
    print("Exporting ONNX...")
    runner.export_policy_to_onnx(str(out_dir), filename="policy.onnx", verbose=False)
    onnx_path = out_dir / "policy.onnx"
    size_kb = onnx_path.stat().st_size / 1024.0
    print(f"  wrote {onnx_path}  ({size_kb:.1f} KB)")

    # Sanity-check input/output shapes against the env so a misuse is caught
    # here rather than by the firmware's runtime assertion.
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        in_shape = sess.get_inputs()[0].shape
        out_shape = sess.get_outputs()[0].shape
        obs_dim = env.observation_space["policy"].shape[-1]
        act_dim = env.action_space.shape[-1]
        print(f"  ONNX input shape:  {in_shape}  (env obs_dim = {obs_dim})")
        print(f"  ONNX output shape: {out_shape}  (env act_dim = {act_dim})")
        print(
            "  NOTE: bebop-linux/src/config.rs expects "
            "OBS_DIM=36, ACTION_DIM=8 for Bebop V2."
        )
    except ImportError:
        print("  (install onnxruntime to print I/O shape sanity check)")

    env.close()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
    simulation_app.close()
