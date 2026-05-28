"""Custom reward functions for the Bebop V2 articulation."""

from __future__ import annotations

import torch
import warp as wp


def _ensure_tensor(
    value,
    ref_tensor: torch.Tensor | None = None,
    env_device: str | None = None,
) -> torch.Tensor:
    """Coerce asset/sensor data to ``torch.Tensor`` (Isaac Lab 2.x / 3.x)."""
    if isinstance(value, torch.Tensor):
        return value
    torch_view = getattr(value, "torch", None)
    if isinstance(torch_view, torch.Tensor):
        return torch_view
    if isinstance(value, wp.array):
        return wp.to_torch(value)
    if ref_tensor is not None:
        return torch.as_tensor(value, dtype=ref_tensor.dtype, device=ref_tensor.device)
    return torch.as_tensor(
        value,
        dtype=torch.float32,
        device=env_device if env_device is not None else "cpu",
    )


def torso_pitch_asymmetric_reward(
    env,
    imu_name: str = "imu",
    target_gx: float = -0.30,
    good_std: float = 0.12,
    roll_std: float = 0.15,
    forward_penalty_gain: float = 5.0,
    forward_deadband: float = 0.0,
) -> torch.Tensor:
    """Reward a slightly back-leaning torso; penalize forward pitch strongly.

    Uses the same IMU ``projected_gravity_b`` signal as
    ``mdp.imu_projected_gravity`` and the firmware observation builder.
    In body FLU:

    * ``proj_grav[:, 0] < 0`` — torso pitched **back** (stable on hardware)
    * ``proj_grav[:, 0] > 0`` — torso pitched **forward** (falls on hardware)

    Args:
        target_gx: desired ``proj_grav[0]``. Default ``-0.30`` ≈ 17° back
            (midpoint of the 15–20° hardware-stable band).
        good_std: Gaussian width around ``target_gx`` for the positive reward.
        roll_std: Gaussian width on ``proj_grav[1]`` (lateral tilt).
        forward_penalty_gain: scales ``relu(g_x - deadband)²`` — keeps the
            policy out of the forward-fall basin even when the Gaussian
            term is still non-zero.
        forward_deadband: only penalize forward tilt above this ``g_x``.
            ``0.0`` penalizes any forward component.
    """
    imu = env.scene[imu_name]
    proj_grav = _ensure_tensor(
        imu.data.projected_gravity_b, env_device=getattr(env, "device", None)
    )
    g_x = proj_grav[:, 0]
    g_y = proj_grav[:, 1]

    pitch_good = torch.exp(-torch.square(g_x - target_gx) / (good_std * good_std))
    roll_good = torch.exp(-torch.square(g_y) / (roll_std * roll_std))
    forward_overshoot = torch.relu(g_x - forward_deadband)
    forward_penalty = forward_penalty_gain * forward_overshoot * forward_overshoot

    return pitch_good * roll_good - forward_penalty
