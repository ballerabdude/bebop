"""Custom reset / event functions for the Bebop V2 articulation."""

from __future__ import annotations

import torch
import warp as wp

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


def randomize_torso_com_uniform(
    env,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    com_range: dict[str, tuple[float, float]],
) -> None:
    """Per-reset, *absolute* center-of-mass randomization (curriculum-safe).

    The stock :func:`isaaclab.envs.mdp.randomize_rigid_body_com` *adds* a
    sampled offset to the body's *current* CoM and explicitly does not track
    the original value, so it can only be used once at startup — running it
    every reset accumulates an unbounded CoM drift. That also makes it
    impossible to drive with a curriculum (a startup event runs before any
    training step, so a growing range never takes effect).

    This version instead caches the *nominal* CoM the first time it runs and on
    every reset SETS the selected bodies' CoM to ``nominal + sampled_offset``.
    Because it writes an absolute value rather than accumulating, it is safe in
    ``mode="reset"``: every episode draws a fresh CoG, and the sampling window
    (``com_range``) can be widened over training by :func:`torso_com_curriculum`
    so the policy graduates from a near-nominal CoG to large CoG errors.

    Mirrors the physx-view access path of the stock function
    (``asset.data.body_com_pose_b`` -> ``asset.set_coms_index``); only the
    position columns (``[..., :3]``) of the CoM pose are touched.
    """
    asset = env.scene[asset_cfg.name]
    device = asset.device

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=device)
    else:
        env_ids = env_ids.to(device)

    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device=device)
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device=device)

    coms = wp.to_torch(asset.data.body_com_pose_b).clone()

    # Cache the nominal CoM positions on first invocation (before any offset is
    # applied) so every reset can set an absolute nominal + offset value.
    nominal = getattr(asset, "_bebop_nominal_com_b", None)
    if nominal is None:
        nominal = coms[:, :, :3].clone()
        asset._bebop_nominal_com_b = nominal

    range_list = [com_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device=device, dtype=coms.dtype)
    u = torch.rand((len(env_ids), 3), device=device, dtype=coms.dtype)
    offset = (ranges[:, 0] + u * (ranges[:, 1] - ranges[:, 0])).unsqueeze(1)

    # ``set_coms_index`` expects the coms tensor sized to ``len(env_ids)`` (the
    # subset of envs being reset), not the full num_envs — so slice to the
    # reset subset before writing. (The stock startup randomizer passes the
    # full tensor only because at startup env_ids is *all* envs.)
    coms_subset = coms[env_ids]                       # (len(env_ids), num_bodies, 7)
    nominal_subset = nominal[env_ids]                 # (len(env_ids), num_bodies, 3)
    coms_subset[:, body_ids, :3] = nominal_subset[:, body_ids, :3] + offset
    asset.set_coms_index(coms=coms_subset, env_ids=env_ids)


def torso_com_curriculum(
    env,
    env_ids: torch.Tensor,
    term_name: str,
    full_com_range: dict[str, tuple[float, float]],
    start_fraction: float = 0.25,
    num_curriculum_steps: int = 100_000,
) -> float:
    """Linearly ramp the torso-CoM randomization window over training.

    Scales the named reset event's ``com_range`` from ``start_fraction * full``
    up to ``full_com_range`` over ``num_curriculum_steps`` control steps, then
    holds at full. Early episodes therefore see a near-nominal CoG (easy, dense
    standing signal) and late episodes face the full large-CoG-error envelope
    (robustness). Pairs with :func:`randomize_torso_com_uniform`.

    ``num_curriculum_steps`` is in ``env.common_step_counter`` units (one per
    control step, shared across all envs and independent of ``num_envs``). With
    ``num_steps_per_env = 32`` and ``max_iterations = 5000`` a run spans ~160k
    such steps, so the default 100k ramps the CoG envelope in over roughly the
    first ~60% of training. Returns the current fraction so the curriculum
    manager logs it.
    """
    progress = min(1.0, float(env.common_step_counter) / float(num_curriculum_steps))
    fraction = start_fraction + progress * (1.0 - start_fraction)

    term_cfg = env.event_manager.get_term_cfg(term_name)
    term_cfg.params["com_range"] = {
        axis: (lo * fraction, hi * fraction)
        for axis, (lo, hi) in full_com_range.items()
    }
    env.event_manager.set_term_cfg(term_name, term_cfg)

    return fraction


def push_magnitude_curriculum(
    env,
    env_ids: torch.Tensor,
    term_name: str,
    full_velocity_range: dict[str, tuple[float, float]],
    start_fraction: float = 0.4,
    num_curriculum_steps: int = 150_000,
) -> float:
    """Linearly ramp a push event's ``velocity_range`` over training.

    Recovery is much easier to learn if the policy first masters standing,
    then faces progressively bigger shoves. This scales the named interval
    push event's ``velocity_range`` from ``start_fraction * full`` up to the
    full range over ``num_curriculum_steps`` environment (control) steps,
    then holds at full.

    ``num_curriculum_steps`` is in ``env.common_step_counter`` units (one per
    control step, shared across all envs). With ``num_steps_per_env = 32`` and
    ``max_iterations = 10000`` the run spans ~320k such steps, so the default
    150k ramps the disturbance in over roughly the first half of training.

    Returns the current fraction so the curriculum manager logs it.
    """
    progress = min(1.0, float(env.common_step_counter) / float(num_curriculum_steps))
    fraction = start_fraction + progress * (1.0 - start_fraction)

    term_cfg = env.event_manager.get_term_cfg(term_name)
    term_cfg.params["velocity_range"] = {
        axis: (lo * fraction, hi * fraction)
        for axis, (lo, hi) in full_velocity_range.items()
    }
    env.event_manager.set_term_cfg(term_name, term_cfg)

    return fraction
