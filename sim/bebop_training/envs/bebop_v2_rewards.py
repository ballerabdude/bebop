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
    band_gx_min: float = -0.30,
    band_gx_max: float = -0.17,
    edge_std: float = 0.12,
    roll_std: float = 0.15,
    forward_penalty_gain: float = 5.0,
    forward_deadband: float = 0.0,
) -> torch.Tensor:
    """Reward balancing anywhere in a back-lean *band*; penalize forward pitch.

    Uses the same IMU ``projected_gravity_b`` signal as
    ``mdp.imu_projected_gravity`` and the firmware observation builder.
    In body FLU:

    * ``proj_grav[:, 0] < 0`` — torso pitched **back** (stable on hardware)
    * ``proj_grav[:, 0] > 0`` — torso pitched **forward** (falls on hardware)

    Unlike a single-target Gaussian (which collapses the policy onto one
    pitch in playback), this uses a **flat-top plateau**: the pitch term
    is ``1.0`` for any ``g_x`` inside the closed band
    ``[band_gx_min, band_gx_max]`` and falls off as a Gaussian of width
    ``edge_std`` in the (signed) distance *outside* the band. So the
    policy is free to balance at any lean angle within the band rather
    than being pulled to a single point — this is what lets the torso
    settle at multiple pitches across different inits.

    Note ``g_x = -sin(pitch)``: a *more negative* ``g_x`` is a *deeper*
    back lean. Hence ``band_gx_min`` (more negative) is the deep edge and
    ``band_gx_max`` (less negative) is the shallow edge. Keep both inside
    the ``imu_pitch_out_of_bounds`` termination envelope (``|g_x| <
    sin(20°) ≈ 0.342``) or the policy will be rewarded for sitting on the
    termination cliff.

    Args:
        band_gx_min: deep-lean edge of the plateau (most negative ``g_x``).
        band_gx_max: shallow-lean edge of the plateau (least negative
            ``g_x``); should still be a back lean (``< 0``) so the plateau
            never rewards an upright/forward torso.
        edge_std: Gaussian width of the falloff outside the band.
        roll_std: Gaussian width on ``proj_grav[1]`` (lateral tilt).
        forward_penalty_gain: scales ``relu(g_x - deadband)²`` — keeps the
            policy out of the forward-fall basin even when the plateau
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

    # Signed distance to the band: 0 inside [band_gx_min, band_gx_max],
    # positive once g_x leaves either edge. Flat top, Gaussian shoulders.
    below = torch.relu(band_gx_min - g_x)  # deeper back lean than the band
    above = torch.relu(g_x - band_gx_max)  # shallower lean than the band
    dist = below + above
    pitch_good = torch.exp(-torch.square(dist) / (edge_std * edge_std))

    roll_good = torch.exp(-torch.square(g_y) / (roll_std * roll_std))
    forward_overshoot = torch.relu(g_x - forward_deadband)
    forward_penalty = forward_penalty_gain * forward_overshoot * forward_overshoot

    return pitch_good * roll_good - forward_penalty
