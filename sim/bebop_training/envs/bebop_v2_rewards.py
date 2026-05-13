"""Custom reward functions for the Bebop V2 articulation.

These supplement the stock ``isaaclab.envs.mdp`` reward terms with
biped-specific shaping (left/right symmetry, "hold still when stable",
yaw suppression while standing).

Keep the *definitions* here and the *weights* in experiment configs so
each experiment is just a thin set of dial overrides.

**Isaac Lab 3.0 note** — asset and sensor ``.data.*`` properties now
return :class:`ProxyArray` (a thin wrapper over the underlying
``wp.array``), not :class:`torch.Tensor`. We coerce them with
:func:`_ensure_tensor` below, which prefers the ``.torch`` zero-copy
accessor when present (Isaac Lab 3.0) and falls back to
:func:`wp.to_torch` for raw ``wp.array`` inputs (the temporary
2.x-style API). See "ProxyArray Backend for Asset and Sensor Data" in
the 3.0 migration guide.
"""

from __future__ import annotations

import torch
import warp as wp

from isaaclab.managers import SceneEntityCfg


def _ensure_tensor(
    value,
    ref_tensor: torch.Tensor | None = None,
    env_device: str | None = None,
) -> torch.Tensor:
    """Coerce ``value`` to a torch tensor regardless of Isaac Lab version.

    Accepts:

    * :class:`torch.Tensor` (Isaac Lab 2.x style, also returned by
      some 3.0 manager properties such as ``env.action_manager.action``);
    * Isaac Lab 3.0 ``ProxyArray`` objects (detected by the presence of
      a ``torch`` attribute) — uses the zero-copy ``.torch`` view rather
      than ``__torch_function__`` to avoid the one-time
      ``DeprecationWarning`` the proxy emits when wrapped in a
      ``torch.as_tensor`` call;
    * Raw :class:`warp.array` buffers — converted with
      :func:`wp.to_torch` (zero copy);
    * Anything else that :func:`torch.as_tensor` can handle (last
      resort, mostly for tests).
    """
    if isinstance(value, torch.Tensor):
        return value
    # Isaac Lab 3.0: asset / sensor data properties return ProxyArray.
    # Duck-typed so this file doesn't have to import the symbol (it
    # would otherwise need a guarded import for backward compat).
    torch_view = getattr(value, "torch", None)
    if isinstance(torch_view, torch.Tensor):
        return torch_view
    if isinstance(value, wp.array):
        # wp.to_torch returns a view sharing memory with the warp array.
        return wp.to_torch(value)
    if ref_tensor is not None:
        return torch.as_tensor(value, dtype=ref_tensor.dtype, device=ref_tensor.device)
    return torch.as_tensor(
        value,
        dtype=torch.float32,
        device=env_device if env_device is not None else "cpu",
    )


def _pair_symmetry_penalty(
    env, asset_cfg: SceneEntityCfg, left_index: int, right_index: int
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    joint_pos = _ensure_tensor(robot.data.joint_pos, env_device=getattr(env, "device", None))
    diff = joint_pos[:, left_index] - joint_pos[:, right_index]
    return torch.square(diff)


def hip_abduction_symmetry_penalty(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    return _pair_symmetry_penalty(env, asset_cfg, 0, 1)


def femur_symmetry_penalty(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    return _pair_symmetry_penalty(env, asset_cfg, 2, 3)


def shin_symmetry_penalty(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    return _pair_symmetry_penalty(env, asset_cfg, 4, 5)


def foot_symmetry_penalty(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    return _pair_symmetry_penalty(env, asset_cfg, 6, 7)


def undesired_yaw_penalty(env, command_name: str) -> torch.Tensor:
    """Penalize yaw rate when the policy is *not* commanded to turn."""
    robot = env.scene["robot"]
    root_ang_vel = _ensure_tensor(robot.data.root_ang_vel_b, env_device=getattr(env, "device", None))
    yaw_vel = root_ang_vel[:, 2]
    cmd = _ensure_tensor(env.command_manager.get_command(command_name), yaw_vel)
    cmd_yaw = cmd[:, 2]
    is_standing = (cmd_yaw.abs() < 0.1).float()
    return (yaw_vel**2) * is_standing


def leg_action_when_stable_penalty(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize action magnitude when the robot is upright AND nearly still.

    Discourages "twitching while balanced" — the policy is allowed to act
    freely whenever it's actually disturbed or trying to move.
    """
    robot = env.scene[asset_cfg.name]
    proj_grav = _ensure_tensor(robot.data.projected_gravity_b, env_device=getattr(env, "device", None))
    is_upright = (proj_grav[:, 2] < -0.85).float()
    ang_vel = _ensure_tensor(robot.data.root_ang_vel_b, proj_grav)
    is_still = (torch.norm(ang_vel, dim=1) < 0.5).float()
    is_stable = is_upright * is_still
    all_joint_actions = _ensure_tensor(env.action_manager.action, proj_grav)
    action_magnitude = torch.sum(torch.square(all_joint_actions), dim=1)
    return action_magnitude * is_stable


def leg_position_hold_reward(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward low joint velocity when the robot is upright."""
    robot = env.scene[asset_cfg.name]
    proj_grav = _ensure_tensor(robot.data.projected_gravity_b, env_device=getattr(env, "device", None))
    is_upright = (proj_grav[:, 2] < -0.85).float()
    joint_vel = _ensure_tensor(robot.data.joint_vel, proj_grav)
    joint_vel_magnitude = torch.sum(torch.square(joint_vel), dim=1)
    return torch.exp(-0.5 * joint_vel_magnitude) * is_upright
