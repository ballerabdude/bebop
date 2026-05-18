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
from isaaclab.utils.math import quat_apply_inverse


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


def leg_action_when_stable_penalty(
    env,
    asset_cfg: SceneEntityCfg,
    upright_threshold: float = -0.7,
    still_threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize action magnitude when the robot is upright AND nearly still.

    Discourages "twitching while balanced" — the policy is allowed to act
    freely whenever it's actually disturbed or trying to move.

    Args:
        upright_threshold: ``proj_grav[:, 2]`` must be **less than** this for
            the env to count as upright. ``-1.0`` is perfectly upright,
            ``0.0`` is sideways. Default ``-0.7`` ≈ within ~45° of vertical
            (was ``-0.85``, ≈25°). The looser default makes the gate fire
            during recovery, not only at perfect balance, so the policy
            learns the smoothness lesson far more often per rollout.
        still_threshold: env counts as "still" when ``|root_ang_vel_b| <``
            this in rad/s. Default ``1.0`` (was ``0.5``) for the same
            reason — the gate fires during gentle recovery, not just at
            zero velocity.
    """
    robot = env.scene[asset_cfg.name]
    proj_grav = _ensure_tensor(robot.data.projected_gravity_b, env_device=getattr(env, "device", None))
    is_upright = (proj_grav[:, 2] < upright_threshold).float()
    ang_vel = _ensure_tensor(robot.data.root_ang_vel_b, proj_grav)
    is_still = (torch.norm(ang_vel, dim=1) < still_threshold).float()
    is_stable = is_upright * is_still
    all_joint_actions = _ensure_tensor(env.action_manager.action, proj_grav)
    action_magnitude = torch.sum(torch.square(all_joint_actions), dim=1)
    return action_magnitude * is_stable


def leg_position_hold_reward(
    env,
    asset_cfg: SceneEntityCfg,
    upright_threshold: float = -0.7,
) -> torch.Tensor:
    """Reward low joint velocity when the robot is upright.

    Args:
        upright_threshold: see :func:`leg_action_when_stable_penalty`.
            Default ``-0.7`` (was ``-0.85``) so the policy is rewarded for
            slowing its joints down during recovery, not only after
            perfect balance has already been achieved.
    """
    robot = env.scene[asset_cfg.name]
    proj_grav = _ensure_tensor(robot.data.projected_gravity_b, env_device=getattr(env, "device", None))
    is_upright = (proj_grav[:, 2] < upright_threshold).float()
    joint_vel = _ensure_tensor(robot.data.joint_vel, proj_grav)
    joint_vel_magnitude = torch.sum(torch.square(joint_vel), dim=1)
    return torch.exp(-0.5 * joint_vel_magnitude) * is_upright


def foot_flat_reward(
    env,
    asset_cfg: SceneEntityCfg,
    std: float = 0.15,
    foot_body_names: tuple[str, str] = ("foot_left_1", "foot_right_1"),
) -> torch.Tensor:
    """Reward keeping the soles of both feet parallel to the ground.

    Reads each foot link's world-frame quaternion from the articulation
    and projects world gravity ``(0, 0, -1)`` into the foot's local
    frame. When the foot is flat (sole horizontal, link +z pointing
    up) the projected vector is approximately ``(0, 0, -1)`` and the
    ``x`` / ``y`` components are zero; as the foot tilts forward
    (toe-down/heel-down) or laterally, those components grow with
    ``sin(theta)``.

    The error per foot is ``g_x^2 + g_y^2``; we sum over both feet and
    pass through ``exp(-err / std^2)`` so the reward is in ``[0, 1]``
    and concentrates strongly around the flat-foot pose.

    Composes cleanly with :func:`torso_upright_via_legs_reward`: the
    torso term selects shin/ankle angles that keep the torso vertical;
    this term selects shin/ankle angles that keep the foot horizontal.
    Together they pin both ends of the pitch chain, which for a
    standing pose corresponds to a straight leg / flat foot / upright
    torso. During locomotion the foot legitimately tilts during swing
    and heel/toe contact, so locomotion experiments should either
    drop this term's weight, widen ``std`` (e.g. to ``0.4``), or
    gate it on stance contact.

    Args:
        std: shaping width. Default ``0.15`` ≈ a foot tilt of ~8.6°
            drops the reward to ``exp(-1) ≈ 0.37``. Loosen toward
            ``0.3`` if the policy needs more tolerance during recovery
            from a perturbation; tighten toward ``0.10`` for a
            stricter "feet must be visibly flat" bias.
        foot_body_names: rigid-body names for the left/right foot
            links in the articulation. Default matches the Bebop V2
            USD (``foot_left_1`` / ``foot_right_1``).
    """
    robot = env.scene[asset_cfg.name]
    device = getattr(env, "device", None)

    body_quat_w = _ensure_tensor(robot.data.body_quat_w, env_device=device)
    # body_quat_w shape: (num_envs, num_bodies, 4) in (w, x, y, z).
    body_names = robot.body_names
    foot_indices = [body_names.index(name) for name in foot_body_names]

    num_envs = body_quat_w.shape[0]
    gravity_w = torch.tensor(
        [0.0, 0.0, -1.0], device=body_quat_w.device, dtype=body_quat_w.dtype
    ).unsqueeze(0).expand(num_envs, -1)

    err = torch.zeros(num_envs, device=body_quat_w.device, dtype=body_quat_w.dtype)
    for idx in foot_indices:
        foot_quat = body_quat_w[:, idx, :]  # (num_envs, 4)
        foot_grav_b = quat_apply_inverse(foot_quat, gravity_w)
        err = err + foot_grav_b[:, 0] * foot_grav_b[:, 0]
        err = err + foot_grav_b[:, 1] * foot_grav_b[:, 1]

    return torch.exp(-err / (std * std))


def torso_upright_via_legs_reward(
    env,
    asset_cfg: SceneEntityCfg,
    std: float = 0.2,
    foot_compensation_gain: float = 1.0,
    knee_compensation_gain: float = 1.0,
    imu_name: str = "imu",
) -> torch.Tensor:
    """Reward an upright torso achieved through ankle + knee compensation.

    Reads the body-frame projected gravity from the IMU sensor — the same
    signal the policy observes via ``mdp.imu_projected_gravity`` and the
    same signal the real-robot firmware derives from the BNO085 fused
    quaternion (see ``firmware/bebop-linux/src/imu.rs``). When the robot
    is upright this vector is approximately ``(0, 0, -1)`` in the body
    frame; ``proj_grav[:, 0]`` grows positive as the torso pitches
    forward, ``proj_grav[:, 1]`` as it rolls right.

    The pitch component is **not** penalised directly. Instead, it is
    folded together with the average ankle (foot) joint angle AND the
    average knee (shin) joint angle into a *residual*: the slice of
    torso pitch that neither the ankles nor the knees are compensating
    for. The policy can therefore satisfy this reward by holding the
    torso perfectly vertical, OR by deliberately bending the knees /
    pitching the ankles to take up the slack — i.e., a full
    leg-compensation balance strategy (ankle strategy + knee strategy)
    where the lower limbs absorb the kinematic chain's tilt and keep
    the torso plate level.

    The roll component is penalised directly. Neither the foot pitch
    nor the knee pitch joints have any roll authority on this
    articulation (lateral balance is the hip-abduction group's job),
    so there is nothing to "compensate" with on that axis.

    Reward is ``exp(-(pitch_residual^2 + roll^2) / std^2)``, bounded in
    ``[0, 1]``. Multiplicatively shaped so a non-flat torso AND
    non-compensating legs is the only path to zero reward.

    Args:
        std: shaping width. Default ``0.2`` ≈ ~11° of effective tilt
            (uncompensated by the legs) before the reward drops to
            ``exp(-1) ≈ 0.37`` of its maximum. Tighten toward ``0.1``
            for a stricter upright bias, loosen toward ``0.4`` if the
            policy needs more freedom to recover from large initial
            perturbations.
        foot_compensation_gain: how strongly foot pitch is credited
            for offsetting torso pitch. ``1.0`` treats one radian of
            average foot pitch as offsetting one unit of projected-
            gravity pitch (i.e. ``sin(theta) ≈ theta`` for small
            angles). Drop to ``0.5`` if the policy starts pitching the
            ankles to "fake" being upright while the torso remains
            tilted.
        knee_compensation_gain: same idea for the shin (knee) joints.
            Default ``1.0``. The knee and ankle act in series in the
            pitch plane, so the policy can split the compensation
            between them however the reward landscape prefers; tune
            this independently if you observe knee-dominant or
            ankle-dominant gaming.
        imu_name: scene key for the IMU sensor. Defaults to the
            ``imu`` key wired up in ``BebopV2BaseEnvCfg``.
    """
    robot = env.scene[asset_cfg.name]
    imu = env.scene[imu_name]
    proj_grav = _ensure_tensor(
        imu.data.projected_gravity_b, env_device=getattr(env, "device", None)
    )
    pitch = proj_grav[:, 0]
    roll = proj_grav[:, 1]

    joint_pos = _ensure_tensor(robot.data.joint_pos, proj_grav)
    # JOINT_NAMES_ALL indices: 4/5 = shin_left/right (knee), 6/7 = foot_left/right
    # (ankle). Same convention as shin_symmetry_penalty / foot_symmetry_penalty.
    shin_avg = 0.5 * (joint_pos[:, 4] + joint_pos[:, 5])
    foot_avg = 0.5 * (joint_pos[:, 6] + joint_pos[:, 7])

    pitch_residual = (
        pitch
        - foot_compensation_gain * foot_avg
        - knee_compensation_gain * shin_avg
    )
    err_sq = pitch_residual * pitch_residual + roll * roll
    return torch.exp(-err_sq / (std * std))
