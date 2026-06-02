"""Custom reset / event functions for the Bebop V2 articulation."""

from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg

from .bebop_v2_terminations import _ensure_tensor


def reset_joints_uniform_within_limits(
    env,
    env_ids: torch.Tensor,
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg,
    range_fraction: float = 1.0,
) -> None:
    """Sample each joint position uniformly within its soft joint limits.

    Unlike :func:`isaaclab.envs.mdp.reset_joints_by_offset`, which adds
    one symmetric offset to every joint and clamps to the soft limits
    (piling probability mass at the limit walls and discarding the
    asymmetric knee / hip-abduction ranges), this samples each joint
    independently across its *own* ``[lower, upper]`` soft range. The
    sampled distribution is uniform over the full joint configuration
    box defined by ``soft_joint_pos_limits``.

    ``range_fraction`` (in ``(0, 1]``) shrinks each joint's sampling
    window *toward its default pose* without changing the window's
    asymmetry: the per-joint interval becomes
    ``[default + f*(lower - default), default + f*(upper - default)]``.
    So ``1.0`` reproduces the full-range behaviour, while ``0.25``
    samples within 25% of the distance from the default to each limit on
    both sides — a tight band around the nominal pose that keeps the
    asymmetric knee / hip-abduction shape intact. Use a small fraction
    early in training (most resets recoverable -> dense signal) and widen
    it later for robustness.

    Joint velocities are sampled uniformly from a single shared
    ``velocity_range`` and broadcast across joints (per-joint velocity
    ranges aren't generally meaningful — actuator limits handle that).

    Joints outside ``asset_cfg.joint_ids`` are left at their default
    pose / velocity.
    """
    asset = env.scene[asset_cfg.name]
    device = getattr(env, "device", None)

    joint_pos = _ensure_tensor(asset.data.default_joint_pos, env_device=device)[env_ids].clone()
    joint_vel = _ensure_tensor(asset.data.default_joint_vel, env_device=device)[env_ids].clone()

    soft_limits = _ensure_tensor(asset.data.soft_joint_pos_limits, env_device=device)

    joint_ids = asset_cfg.joint_ids
    if isinstance(joint_ids, slice):
        joint_ids = list(range(*joint_ids.indices(asset.num_joints)))
    elif hasattr(joint_ids, "cpu"):
        joint_ids = joint_ids.cpu().tolist()
    else:
        joint_ids = list(joint_ids)

    default_pos = joint_pos[:, joint_ids]
    lower = soft_limits[env_ids][:, joint_ids, 0]
    upper = soft_limits[env_ids][:, joint_ids, 1]

    # Shrink the per-joint window toward the default pose, preserving the
    # asymmetry of each [lower, upper] interval.
    frac = max(0.0, min(1.0, range_fraction))
    lower = default_pos + frac * (lower - default_pos)
    upper = default_pos + frac * (upper - default_pos)

    u = torch.rand_like(lower)
    joint_pos[:, joint_ids] = lower + u * (upper - lower)

    v_lo, v_hi = velocity_range
    joint_vel[:, joint_ids] = torch.empty_like(lower).uniform_(v_lo, v_hi)

    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
