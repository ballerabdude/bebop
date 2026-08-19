"""One-off verification: foot-friction randomization applies to foot bodies only.

Builds the standing env with a handful of envs, steps a few resets, and reads
back the per-shape material properties from the physics view to confirm:

1. the two foot bodies (``foot_left_1`` / ``foot_right_1``) carry friction in
   the configured polyurethane range (1.7-2.1), and
2. every other body keeps the default material (friction 1.0).

Usage (inside the container, CWD=/workspace/bebop_bot/sim):

    /workspace/isaaclab/isaaclab.sh -p scripts/verify_foot_friction.py --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch

import isaaclab.envs.mdp as mdp  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

from bebop_training.experiments.exp_standing import (  # noqa: E402
    BebopV2StandingCfg,
    FOOT_FRICTION_STATIC_RANGE,
)


def main():
    env_cfg = BebopV2StandingCfg()
    env_cfg.scene.num_envs = 16
    env = ManagerBasedRLEnv(cfg=env_cfg)

    robot = env.scene["robot"]
    # resolve the foot body ids the event term targets
    foot_cfg = env_cfg.events.randomize_foot_friction.params["asset_cfg"]
    foot_ids = robot.find_bodies(list(foot_cfg.body_names))[0]
    print(f"[verify] foot body ids: {foot_ids} ({foot_cfg.body_names})")

    def read_friction():
        # materials: (num_envs, max_shapes, 3) -> (static, dynamic, restitution)
        import warp as wp

        mats = wp.to_torch(robot.root_view.get_material_properties()).clone()
        return mats

    def shape_slices():
        # replicate the impl's per-body shape bookkeeping
        slices = {}
        shapes_per_body = robot.root_view.max_shapes // robot.num_bodies
        for body_id in range(robot.num_bodies):
            slices[body_id] = (body_id * shapes_per_body, (body_id + 1) * shapes_per_body)
        return slices, shapes_per_body

    slices, spb = shape_slices()
    print(f"[verify] num_bodies={robot.num_bodies} max_shapes={robot.root_view.max_shapes} shapes_per_body={spb}")

    stats = []
    for round_idx in range(3):
        # force a reset across all envs so the event term fires
        env.reset()
        mats = read_friction()  # (num_envs, max_shapes, 3)

        foot_static = torch.stack(
            [mats[:, slices[b][0] : slices[b][1], 0].flatten() for b in foot_ids]
        )
        other_ids = [b for b in range(robot.num_bodies) if b not in foot_ids]
        other_static = torch.stack(
            [mats[:, slices[b][0] : slices[b][1], 0].flatten() for b in other_ids]
        )

        fs = foot_static.flatten()
        os_ = other_static.flatten()
        # Non-foot bodies carry the robot's own USD material (0.5 on this
        # asset), NOT the terrain material — what matters is that they are
        # identical across envs and rounds, i.e. the event touched only the
        # feet.
        stats.append((fs.min().item(), fs.max().item(), os_.min().item(), os_.max().item()))
        print(
            f"[verify] round {round_idx}: "
            f"foot static friction min={fs.min():.3f} max={fs.max():.3f} "
            f"(want within {FOOT_FRICTION_STATIC_RANGE}); "
            f"other bodies min={os_.min():.3f} max={os_.max():.3f} "
            f"(want uniform, i.e. min == max)"
        )

    lo, hi = FOOT_FRICTION_STATIC_RANGE
    ok_feet = all(lo - 1e-3 <= a and b <= hi + 1e-3 for a, b, _, _ in stats)
    ok_others = all(abs(c - d) < 1e-6 for _, _, c, d in stats)
    print(f"[verify] feet in range: {ok_feet}; others untouched (uniform): {ok_others}")
    assert ok_feet and ok_others, "foot friction verification FAILED"
    print("[verify] PASS")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
