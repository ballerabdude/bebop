"""Custom termination conditions for the Bebop V2 articulation."""

from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg

from .bebop_v2_rewards import _ensure_tensor


def base_link_on_ground(
    env,
    asset_cfg: SceneEntityCfg,
    ground_height_threshold: float = 0.30,
) -> torch.Tensor:
    """Terminate when ``base_link`` drops near ground level (fallen).

    ``base_link`` sits at the top of the robot (~0.65 m when standing). A
    threshold around 0.30 m indicates the torso has clearly fallen toward
    the ground.
    """
    robot = env.scene[asset_cfg.name]
    body_pos_w = _ensure_tensor(robot.data.body_pos_w, env_device=getattr(env, "device", None))
    base_link_height = body_pos_w[:, asset_cfg.body_ids[0], 2]
    return base_link_height <= ground_height_threshold
