"""Custom termination conditions for the Bebop V2 articulation."""

from __future__ import annotations

import torch
import warp as wp

from isaaclab.managers import SceneEntityCfg


def _ensure_tensor(value, env_device: str | None = None) -> torch.Tensor:
    """Coerce an Isaac Lab asset/sensor data field to a torch tensor.

    Isaac Lab 3.0 returns ``ProxyArray`` (a thin ``wp.array`` wrapper) from
    ``asset.data.*`` properties; pre-3.0 returned raw ``wp.array`` or
    ``torch.Tensor``. This helper accepts all three and never copies:

    * ``torch.Tensor``  -> returned as-is.
    * ``ProxyArray``    -> ``.torch`` (zero-copy view, avoids the deprecation
      warning that ``torch.as_tensor`` would emit).
    * ``wp.array``      -> ``wp.to_torch`` (zero-copy view).
    * Anything else     -> ``torch.as_tensor`` fallback (mostly for tests).
    """
    if isinstance(value, torch.Tensor):
        return value
    torch_view = getattr(value, "torch", None)
    if isinstance(torch_view, torch.Tensor):
        return torch_view
    if isinstance(value, wp.array):
        return wp.to_torch(value)
    return torch.as_tensor(
        value,
        dtype=torch.float32,
        device=env_device if env_device is not None else "cpu",
    )


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


def imu_pitch_out_of_bounds(
    env,
    imu_name: str = "imu",
    pitch_forward_gx_max: float = 0.342,
    pitch_back_gx_min: float = -0.342,
) -> torch.Tensor:
    """Terminate when torso pitch exceeds the hardware fall envelope.

    Uses IMU ``projected_gravity_b[0]`` (same convention as policy obs /
    ``firmware/bebop-linux``). For small angles, ``g_x ≈ sin(pitch)``:

    * ``g_x > pitch_forward_gx_max`` — pitched forward past the limit
      (default ``sin(20°) ≈ 0.342``).
    * ``g_x < pitch_back_gx_min`` — pitched back past the limit
      (default ``-sin(20°) ≈ -0.342``).

    Ends episodes before the slow ``base_link_on_ground`` check when the
    torso is already in a pose that falls on the real robot.
    """
    imu = env.scene[imu_name]
    proj_grav = _ensure_tensor(
        imu.data.projected_gravity_b, env_device=getattr(env, "device", None)
    )
    g_x = proj_grav[:, 0]
    return (g_x > pitch_forward_gx_max) | (g_x < pitch_back_gx_min)
