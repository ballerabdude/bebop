"""Custom reward functions for the Bebop V2 articulation."""

from __future__ import annotations

import torch
import warp as wp

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply


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


def action_gain_rate_l2(
    env,
    num_position_channels: int = 8,
) -> torch.Tensor:
    """Penalize tick-to-tick change of the variable-impedance kp/kd channels.

    The 24-dim MIT action is laid out as ``[0:8]`` position, ``[8:16]`` kp,
    ``[16:24]`` kd (see ``VariableImpedanceJointAction``). The stock
    ``mdp.action_rate_l2`` penalizes the squared change of *all* 24 channels
    together, which is too diluted to stop the gain channels from flipping —
    the exact failure seen on hardware (e.g. a foot kp snapping 250 -> 107 ->
    250 across consecutive ticks). This term isolates the 16 gain channels
    (everything at or past ``num_position_channels``) and taxes only their
    tick-to-tick change, so it can be weighted hard enough to enforce smooth,
    slowly-varying impedance without also over-damping the position targets.

    Returns ``Σ (gain_t - gain_{t-1})²`` over the gain channels; always ``>= 0``,
    use with a negative weight.
    """
    device = getattr(env, "device", None)
    action = _ensure_tensor(env.action_manager.action, env_device=device)
    prev_action = _ensure_tensor(env.action_manager.prev_action, env_device=device)
    gain = action[:, num_position_channels:]
    gain_prev = prev_action[:, num_position_channels:]
    return torch.sum(torch.square(gain - gain_prev), dim=1)


def action_position_rate_l2(
    env,
    num_position_channels: int = 8,
    balance_gate: bool = False,
    gate_band_gx_center: float = -0.17,
    gate_std: float = 0.15,
    gate_floor: float = 0.2,
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize tick-to-tick change of the 8 *position* action channels.

    The 24-dim MIT action is ``[0:8]`` position, ``[8:16]`` kp, ``[16:24]`` kd
    (see ``VariableImpedanceJointAction``). The stock ``mdp.action_rate_l2``
    averages the squared change of *all 24* channels, so the position targets'
    own smoothness signal is diluted 3:1 by the gain channels — there is no
    strong, isolated gradient telling the policy to move the joint *setpoints*
    smoothly. (The gain channels already get their own isolated rate term,
    ``action_gain_rate_l2``; this is the position-channel counterpart.)

    For a heavy-torso quiet stand the correct behaviour is slow, smooth
    counter-balancing of the position commands, NOT fast tick-to-tick setpoint
    flipping. Isolating ``[0:N]`` lets this be weighted hard enough to force
    very smooth position changes without over-damping the (separately handled)
    impedance channels.

    Returns ``Σ (pos_t - pos_{t-1})²`` over the position channels; always
    ``>= 0``, use with a negative weight.

    **Balance gate** (``balance_gate=True``): multiplies the penalty by
    ``max(gate_floor, exp(-((g_x - center)² + g_y²) / gate_std²))``. The gate
    fires at full strength (1.0) when the robot is balanced (kills steady-state
    chatter, the term's actual job) and decays toward ``gate_floor`` when the
    robot is tilted (relaxes — but does NOT eliminate — the smoothness
    constraint so the policy can move the setpoints rapidly to catch a fall).

    The floor is essential: without it (gate_floor=0) the penalty vanishes
    completely at tilt and the policy learns to *manufacture* tilt to unlock
    chatter — exactly the flailing limit cycle seen in capture
    20260715_122500 (vel_std 0.69-1.34 rad/s, slew-exceedance 56.9%, g_x
    swinging ±30°). With gate_floor=0.2 the -10.0 weight still contributes
    -2.0·rate at full tilt — enough to keep recovery motions smooth without
    making them unaffordable.

    Args:
        balance_gate: if True, apply the tilt-distance Gaussian gate.
        gate_band_gx_center: ``g_x`` center of the gate (the balance target).
        gate_std: Gaussian width; ~0.10 means the gate is ~1/e at ~5.7° off.
        gate_floor: minimum gate value (0..1). Prevents the penalty from
            vanishing entirely at tilt, which the policy otherwise exploits.
        asset_cfg: articulation for the gate's ``projected_gravity_b``.
    """
    device = getattr(env, "device", None)
    action = _ensure_tensor(env.action_manager.action, env_device=device)
    prev_action = _ensure_tensor(env.action_manager.prev_action, env_device=device)
    pos = action[:, :num_position_channels]
    pos_prev = prev_action[:, :num_position_channels]
    rate = torch.sum(torch.square(pos - pos_prev), dim=1)

    if not balance_gate:
        return rate

    asset = env.scene[asset_cfg.name if asset_cfg is not None else "robot"]
    proj_grav = _ensure_tensor(
        asset.data.projected_gravity_b, env_device=device
    )
    g_x = proj_grav[:, 0]
    g_y = proj_grav[:, 1]
    tilt_err = torch.square(g_x - gate_band_gx_center) + torch.square(g_y)
    gate = torch.exp(-tilt_err / (gate_std * gate_std))
    gate = torch.clamp(gate, min=gate_floor)
    return rate * gate


def action_gain_l2(
    env,
    num_position_channels: int = 8,
) -> torch.Tensor:
    """Penalize the magnitude of the kp/kd action channels (center on midpoint).

    Same channel split as :func:`action_gain_rate_l2`. Each gain channel is
    affine-mapped ``[-1, 1] -> [min, max]`` per joint, so ``raw = 0`` decodes
    to the per-joint midpoint gain. For a quiet stand the gains are
    under-determined (many kp/kd combinations earn the same reward), so PPO
    leaves them noisy and lets them drift toward the rails. Penalizing the
    squared raw gain magnitude gives those otherwise-flat directions a clean
    optimum at ``raw = 0`` (the sensible mid-stiffness prior), while staying
    mild enough that the policy can still move a gain off-midpoint when balance
    genuinely benefits.

    Returns ``Σ gain²`` over the gain channels; always ``>= 0``, use with a
    negative weight.
    """
    device = getattr(env, "device", None)
    action = _ensure_tensor(env.action_manager.action, env_device=device)
    gain = action[:, num_position_channels:]
    return torch.sum(torch.square(gain), dim=1)


def joint_vel_l2(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    balance_gate: bool = False,
    gate_band_gx_center: float = -0.17,
    gate_std: float = 0.15,
    gate_floor: float = 0.2,
) -> torch.Tensor:
    """Penalize any joint motion — the sharpest ``be absolutely still`` signal.

    Returns ``Σ v²`` over the selected joint velocities; always ``>= 0``, use
    with a negative weight. Unlike :func:`stationary_pose_exp` (a bounded
    ``exp(-Σv²/σ²)`` kernel whose gradient vanishes as ``v → 0``), this is an
    unbounded quadratic that keeps a strong, linear-in-``v`` gradient all the
    way down to zero velocity — which is what actually kills small limit-cycle
    oscillations. Pair with the action-rate terms; this is the plant-side
    counterpart (penalize the *result* of the chatter, not just the chatter).

    **Balance gate** (``balance_gate=True``): same gate as
    :func:`action_position_rate_l2` —
    ``max(gate_floor, exp(-((g_x - center)² + g_y²) / gate_std²))``. At full
    strength when balanced (kills residual wobble) and decays to ``gate_floor``
    when tilted (relaxes, not eliminates, the constraint so the policy can
    swing the legs to catch a fall). The floor prevents the flailing-limit-
    cycle exploit where the policy manufactures tilt to unlock chatter — see
    :func:`action_position_rate_l2` for the capture evidence.
    """
    asset = env.scene[asset_cfg.name]
    device = getattr(env, "device", None)
    joint_vel = _ensure_tensor(asset.data.joint_vel, env_device=device)
    if asset_cfg.joint_ids is not None:
        joint_vel = joint_vel[:, asset_cfg.joint_ids]
    vel_sq = torch.sum(torch.square(joint_vel), dim=1)

    if not balance_gate:
        return vel_sq

    proj_grav = _ensure_tensor(asset.data.projected_gravity_b, env_device=device)
    g_x = proj_grav[:, 0]
    g_y = proj_grav[:, 1]
    tilt_err = torch.square(g_x - gate_band_gx_center) + torch.square(g_y)
    gate = torch.exp(-tilt_err / (gate_std * gate_std))
    gate = torch.clamp(gate, min=gate_floor)
    return vel_sq * gate


def stationary_pose_exp(
    env,
    std: float = 1.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Bounded reward for *locking* into a steady pose (near-zero joint motion).

    Unlike a target-pose term, this rewards holding *whatever* configuration
    the policy settles into rather than a specific one — the carrot is on the
    joint velocity, not the joint position. It is a ``exp(-Σv²/σ²)`` kernel
    over the selected joints, so it is bounded in ``[0, 1]`` per tick (1.0
    when perfectly still, decaying as the joints move). Being bounded and
    non-negative, it can be given a large weight to strongly favour a rigid,
    locked stand without ever turning punitive during a legitimate
    balance-recovery transient (it just saturates toward 0).

    Args:
        std: velocity scale (rad/s-ish). The reward is ~1/e once the summed
            squared joint velocity reaches ``std²``. Smaller => stricter
            "be perfectly still" requirement.
        asset_cfg: which articulation / joints to score (defaults to all
            joints of ``robot``).
    """
    asset = env.scene[asset_cfg.name]
    joint_vel = _ensure_tensor(
        asset.data.joint_vel, env_device=getattr(env, "device", None)
    )
    if asset_cfg.joint_ids is not None:
        joint_vel = joint_vel[:, asset_cfg.joint_ids]
    err = torch.sum(torch.square(joint_vel), dim=1)
    return torch.exp(-err / (std * std))


def feet_slide(
    env,
    sensor_names: list[str],
    body_names: list[str],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize a foot sliding along the ground *while it is in contact*.

    Standard Isaac Lab anti-slip term: for each foot it gates the foot's
    horizontal (xy) world velocity by whether that foot is currently bearing
    load (contact normal force above ``force_threshold``), then sums over
    feet. The result is ``>= 0`` and is meant to be used with a negative
    weight.

    Crucially it only taxes velocity *while in contact*: lifting a foot off
    the ground (air time) costs nothing, so it does not discourage a recovery
    step — it just stops the policy from "balancing" by skating its contact
    point, which is a sim cheat that does not transfer to hardware.

    Because this robot is a nested kinematic tree, each foot has its own
    single-body ``ContactSensor`` (foot_left_1 / foot_right_1 are not siblings
    under a common parent, so one wildcard sensor cannot address both). This
    function therefore takes the parallel lists ``sensor_names`` and
    ``body_names``: ``sensor_names[i]`` is the contact sensor for the foot
    whose articulation body is ``body_names[i]``.

    Args:
        sensor_names: per-foot ``ContactSensor`` scene keys (each tracks one
            foot body).
        body_names: articulation body name for each foot, aligned with
            ``sensor_names`` (used to read that foot's linear velocity).
        asset_cfg: the robot articulation.
        force_threshold: contact normal-force magnitude (N) above which a
            foot counts as planted.
    """
    device = getattr(env, "device", None)
    asset = env.scene[asset_cfg.name]
    body_lin_vel = _ensure_tensor(asset.data.body_lin_vel_w, env_device=device)

    reward = None
    for sensor_name, body_name in zip(sensor_names, body_names):
        contact_sensor = env.scene.sensors[sensor_name]
        net_forces = _ensure_tensor(
            contact_sensor.data.net_forces_w_history, env_device=device
        )
        # single tracked body -> (N, history, 1, 3); peak |force| over history
        in_contact = net_forces.norm(dim=-1).max(dim=1)[0][:, 0] > force_threshold

        body_id = asset.body_names.index(body_name)
        foot_vel_xy = body_lin_vel[:, body_id, :2].norm(dim=-1)

        term = foot_vel_xy * in_contact
        reward = term if reward is None else reward + term

    return reward


def feet_load_symmetry(
    env,
    sensor_names: list[str],
    force_eps: float = 1.0,
) -> torch.Tensor:
    """Penalize uneven vertical load between the two feet (anti-one-foot-lean).

    Reads each foot's ``ContactSensor`` net force and returns the load-imbalance
    fraction::

        |F_left - F_right| / (F_left + F_right + force_eps)

    which is ``0`` when both feet carry equal force and approaches ``1`` when
    all the weight is on a single foot. Always ``>= 0`` and bounded in
    ``[0, 1)``, so it is safe to give a firm negative weight without the
    runaway risk of an unbounded penalty.

    This directly attacks the observed failure mode — the policy settling into
    an asymmetric stance that loads one leg far more than the other ("leaning
    on one foot") — by making an even left/right weight split the reward
    optimum. It is normalized by the total load so it measures the *fraction*
    of imbalance rather than absolute newtons, which keeps the signal scale-
    invariant to the robot's weight and to transient contact-force spikes.

    Because the metric is the *ratio*, lifting one foot entirely (its force
    -> 0) drives the fraction toward 1 (max penalty), so this also discourages
    standing on a single foot. Both feet briefly airborne (a fall) makes the
    numerator 0 -> no penalty, but that case is owned by the fall termination,
    not this term.

    Each foot has its own single-body ``ContactSensor`` (the feet are not
    siblings in this nested kinematic tree), so this takes the two per-foot
    sensor scene keys.

    Args:
        sensor_names: the two per-foot ``ContactSensor`` scene keys
            ``[left, right]``; each tracks one foot body.
        force_eps: small force (N) added to the denominator to keep the ratio
            finite and well-behaved when both feet are momentarily unloaded.
    """
    device = getattr(env, "device", None)
    forces = []
    for sensor_name in sensor_names:
        contact_sensor = env.scene.sensors[sensor_name]
        net_forces = _ensure_tensor(
            contact_sensor.data.net_forces_w_history, env_device=device
        )
        # single tracked body -> (N, history, 1, 3); peak |force| over history
        forces.append(net_forces.norm(dim=-1).max(dim=1)[0][:, 0])

    f_left, f_right = forces[0], forces[1]
    total = f_left + f_right + force_eps
    return torch.abs(f_left - f_right) / total


def forward_lean_penalty(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    deadband: float = 0.05,
) -> torch.Tensor:
    """Penalize parking the torso pitched *forward* (COM over the toes).

    Returns ``relu(g_x - deadband)²`` where ``g_x = projected_gravity_b[0]``
    (``g_x > 0`` is a forward/nose-down lean in body FLU, same convention as
    the policy obs / firmware). Always ``>= 0``; use with a **negative**
    weight. Backward lean (``g_x < 0``) is left entirely to the symmetric
    ``flat_orientation_l2`` term — this term only taxes the *forward* half.

    Why asymmetric, when the rest of the task is symmetric "stay upright":
    in sim the policy can hold a forward lean by riding the front edge of the
    flat foot — the rigid foot's contact patch extends to the toe, so the
    ground reaction (plus a little ankle torque) statically supports a COM
    that has crept forward over the toes. On the real robot that strategy
    falls: the foot is small and the ankle motor (RS02, ~17 N·m) is the
    weakest joint, so the torso's weight tips it over instead of being held
    back by the toes. This penalty keeps the resting COM behind the toe line
    (on the whole foot / heel, which the leg can stack load through) so the
    policy does not learn the non-transferable toe-balance.

    The ``deadband`` (in ``g_x`` units, ≈ ``sin`` of the angle: 0.05 ≈ 2.9°)
    means a small forward excursion — including the brief forward overshoot of
    a genuine recovery catch — costs nothing; only a *sustained / deep*
    forward lean is penalized, and quadratically, so the gradient grows as the
    COM approaches the toes. Raise the weight (more negative) if play mode
    still shows a forward toe-lean stand; lower it / widen the deadband if the
    robot becomes reluctant to pitch forward at all during a recovery step.

    Args:
        asset_cfg: articulation whose root ``projected_gravity_b`` to read.
            Defaults to the ``"robot"`` scene entity; the IMU sensor's
            ``projected_gravity_b`` was removed in Isaac Lab 3.0 beta2, so we
            read the articulation root (``base_link``) gravity projection
            instead — equivalent for an IMU mounted with identity orientation.
        deadband: forward-lean deadband in ``g_x`` units.
    """
    asset = env.scene[asset_cfg.name]
    proj_grav = _ensure_tensor(
        asset.data.projected_gravity_b, env_device=getattr(env, "device", None)
    )
    g_x = proj_grav[:, 0]
    overshoot = torch.relu(g_x - deadband)
    return overshoot * overshoot


def bilateral_joint_symmetry_l2(
    env,
    pairs: list[tuple[str, str]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    stillness_std: float = 1.5,
    balance_gate: bool = False,
    gate_band_gx_center: float = -0.17,
    gate_std: float = 0.15,
    gate_floor: float = 0.2,
) -> torch.Tensor:
    """Penalize asymmetry between left/right joint pairs (sagittal plane).

    Returns ``Σ (q_left + q_right)²`` over the listed joint name pairs, gated
    so the penalty only fires when the policy is holding a balanced pose — NOT
    during recovery transients when the policy needs asymmetric motions to
    catch a fall. Always ``>= 0``; use with a negative weight.

    SIGN CONVENTION — why the *sum*, not the difference: every L/R joint
    pair on this robot is sign-MIRRORED, so the same physical "both legs
    doing the same thing" pose reads ``q_left = -q_right`` and the mirrored
    stance is ``q_L + q_R = 0``. This was verified joint-by-joint against
    ``ros2/src/bebopv2_description/urdf/bebopv2.urdf`` (Jul 15 2026):

    ============= ============= ============= ===========================
    Pair           Left axis     Right axis    Symmetric residual
    ============= ============= ============= ===========================
    hip_flexion    ``(0,1,0)``   ``(0,-1,0)``  ``q_L + q_R`` (mirrored Y)
    hip_abduction  ``(1,0,0)``   ``(1,0,0)``   ``q_L + q_R`` (same axis,
                                               mirrored limits L
                                               ``[-10°,+20°]`` vs R
                                               ``[-20°,+10°]``)
    knee_flexion   ``(0,1,0)``   ``(0,-1,0)``  ``q_L + q_R`` (mirrored Y)
    foot           ``(0,1,0)``   ``(0,-1,0)``  ``q_L + q_R`` (mirrored Y)
    ============= ============= ============= ===========================

    The right-side flexion joints (hip_flexion, knee_flexion, foot) use a
    flipped ``-Y`` axis in the URDF/USD. Hip_abduction uses the same ``+X``
    axis on both sides, but its limits are mirrored (``left [-10°, +20°]``
    vs ``right [-20°, +10°]``), so "both legs splayed out" reads
    ``q_L > 0, q_R < 0`` and is symmetric at ``q_L + q_R = 0``. The same
    physical "both knees flexed the same way" pose therefore reads
    ``q_left = -q_right`` for ALL four pairs.

    Penalizing ``(q_L - q_R)²`` (the pre-Jul-2026 version of this term)
    rewarded ``q_L = q_R`` — one leg pitched forward and the other back,
    both hips rolled the same way — i.e. it actively TRAINED the
    twisted-hip contortion it was meant to prevent, and raising its weight
    only pushed harder in the wrong direction (observed on hardware run
    2026-07-09_04-16-40: hip_flexion L-R driven to ~0.01 while the robot
    stood visibly twisted). The ``analyze_capture.py`` L/R symmetry report
    prints both ``L+R`` (the correct residual) and ``L-R`` (the buggy one)
    so this can be audited per capture.

    **Gate** — two mutually exclusive options, selected by ``balance_gate``:

    * **Stillness gate** (``balance_gate=False``, default, legacy):
      ``exp(-Σv²/σ²)`` over all joint velocities. ``1.0`` when the robot is
      perfectly still (penalty fires at full strength) and decays toward ``0``
      when joints are moving fast (penalty suppressed so the policy is free to
      use asymmetric catch/recovery motions). Semantically a *pose* gate:
      symmetry is a property of the pose being held, not of the balance state.

    * **Balance gate** (``balance_gate=True``, Jul 15 2026 redesign):
      ``max(gate_floor, exp(-((g_x - center)² + g_y²) / gate_std²))`` — same
      gate used by :func:`action_position_rate_l2`, :func:`joint_vel_l2`, and
      :func:`joint_deviation_l1_balance_gated`. Fires at full strength when
      the robot is in the back-lean balance band and decays toward
      ``gate_floor`` when tilted. The floor is essential: without it the
      penalty vanishes at tilt and the policy can manufacture tilt + chatter
      to suppress the symmetry constraint (the flailing-limit-cycle exploit
      of capture 20260715_122500). Use the balance gate when the reward
      landscape already uses balance gates on the movement penalties, so the
      symmetry term relaxes in lockstep with them during recovery.

    Args:
        pairs: list of ``(left_joint_name, right_joint_name)`` tuples. Names
            are resolved to articulation joint indices, so the caller does not
            need to know the USD joint order.
        asset_cfg: the robot articulation.
        stillness_std: velocity scale for the stillness gate (ignored when
            ``balance_gate=True``). ``1.5`` means the gate is ~1/e once the
            summed squared joint velocity reaches 1.5².
        balance_gate: if True, use the tilt-distance Gaussian gate instead of
            the stillness gate (see above).
        gate_band_gx_center: ``g_x`` center of the balance gate.
        gate_std: Gaussian width; ~0.10 ≈ 5.7° at 1/e.
        gate_floor: minimum gate value (0..1). Prevents the penalty from
            vanishing entirely at tilt.
    """
    asset = env.scene[asset_cfg.name]
    device = getattr(env, "device", None)
    joint_pos = _ensure_tensor(asset.data.joint_pos, env_device=device)

    err = None
    for left_name, right_name in pairs:
        li = asset.data.joint_names.index(left_name)
        ri = asset.data.joint_names.index(right_name)
        # Mirrored sign convention: a symmetric stance is q_L == -q_R,
        # so the asymmetry residual is the SUM (see docstring).
        d = joint_pos[:, li] + joint_pos[:, ri]
        sq = d * d
        err = sq if err is None else err + sq

    if balance_gate:
        proj_grav = _ensure_tensor(asset.data.projected_gravity_b, env_device=device)
        g_x = proj_grav[:, 0]
        g_y = proj_grav[:, 1]
        tilt_err = torch.square(g_x - gate_band_gx_center) + torch.square(g_y)
        gate = torch.exp(-tilt_err / (gate_std * gate_std))
        gate = torch.clamp(gate, min=gate_floor)
        return err * gate

    # Stillness gate: 1.0 when still, -> 0 when moving. Only penalize
    # asymmetry when the policy is trying to HOLD a pose, not when it's
    # recovering from a perturbation.
    joint_vel = _ensure_tensor(asset.data.joint_vel, env_device=device)
    vel_sq = torch.sum(torch.square(joint_vel), dim=1)
    gate = torch.exp(-vel_sq / (stillness_std * stillness_std))
    return err * gate


def joint_deviation_l1_stillness_gated(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    stillness_std: float = 1.5,
) -> torch.Tensor:
    """Like ``mdp.joint_deviation_l1`` but gated by a stillness multiplier.

    Returns ``Σ |q|`` over the selected joints, multiplied by a stillness gate
    ``exp(-Σv²/σ²)`` over ALL joint velocities. The penalty only fires when the
    robot is holding a pose — NOT during recovery transients when the policy
    needs freedom to move the joints to catch a fall. Always ``>= 0``; use with
    a negative weight.

    Used for the foot (ankle) deviation penalty: the RS02 foot motor (17 N·m)
    can't hold the extreme ankle positions the policy cranks to compensate
    for hardware asymmetry, but the policy needs to be free to use the ankle
    during recovery. The gate ensures the penalty only bites when the policy
    is *trying to hold a pose*, not when it's catching a fall.

    Args:
        asset_cfg: which articulation / joints to score the deviation on.
        stillness_std: velocity scale for the gate (see
            :func:`bilateral_joint_symmetry_l2`).
    """
    asset = env.scene[asset_cfg.name]
    joint_pos = _ensure_tensor(
        asset.data.joint_pos, env_device=getattr(env, "device", None)
    )
    joint_vel = _ensure_tensor(
        asset.data.joint_vel, env_device=getattr(env, "device", None)
    )

    if asset_cfg.joint_ids is not None:
        joint_pos = joint_pos[:, asset_cfg.joint_ids]

    dev = torch.sum(torch.abs(joint_pos), dim=1)

    vel_sq = torch.sum(torch.square(joint_vel), dim=1)
    gate = torch.exp(-vel_sq / (stillness_std * stillness_std))
    return dev * gate


def joint_deviation_l1_balance_gated(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    gate_band_gx_center: float = -0.17,
    gate_std: float = 0.15,
    gate_floor: float = 0.2,
) -> torch.Tensor:
    """``Σ |q|`` over selected joints, gated by how close to balance the torso is.

    The balance-gate counterpart to :func:`joint_deviation_l1_stillness_gated`.
    Multiplies ``Σ |q|`` by
    ``max(gate_floor, exp(-((g_x - center)² + g_y²) / gate_std²))``: the
    penalty fires at full strength when the robot is balanced (holds the
    anchored joints at their home pose) and decays toward ``gate_floor`` when
    tilted (relaxes so the joints can move for active recovery, without
    vanishing entirely). Always ``>= 0``; use with a negative weight.

    Used for the hip-flexion anchor in the active-balance reward: the hips
    must stay straight when balanced (the ankle strategy — entire body leans
    together) but MUST be free to flex when the torso is falling, because
    swinging the legs to put a foot under the falling CoM is the primary
    balance-recovery motion. A non-gated hip-flexion anchor at -0.6 made that
    recovery motion unaffordable and the policy learned to hang forward instead
    of correcting (capture 20260715_041834: g_x mean +0.065, 12° off the
    back-lean band, on a gantry that caught the fall).

    The floor (gate_floor=0.2 default) prevents the flailing exploit seen in
    capture 20260715_122500, where the policy manufactured tilt to suppress the
    anchor entirely and then flailed the hips — see
    :func:`action_position_rate_l2` for the full rationale.

    Args:
        asset_cfg: which articulation / joints to score the deviation on.
        gate_band_gx_center: ``g_x`` center of the balance gate.
        gate_std: Gaussian width; ~0.10 ≈ 5.7° at 1/e.
        gate_floor: minimum gate value (0..1).
    """
    asset = env.scene[asset_cfg.name]
    device = getattr(env, "device", None)
    joint_pos = _ensure_tensor(asset.data.joint_pos, env_device=device)
    if asset_cfg.joint_ids is not None:
        joint_pos = joint_pos[:, asset_cfg.joint_ids]
    dev = torch.sum(torch.abs(joint_pos), dim=1)

    proj_grav = _ensure_tensor(asset.data.projected_gravity_b, env_device=device)
    g_x = proj_grav[:, 0]
    g_y = proj_grav[:, 1]
    tilt_err = torch.square(g_x - gate_band_gx_center) + torch.square(g_y)
    gate = torch.exp(-tilt_err / (gate_std * gate_std))
    gate = torch.clamp(gate, min=gate_floor)
    return dev * gate


def com_over_support_reward(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    foot_body_names: tuple[str, str] = ("foot_left_1", "foot_right_1"),
    max_lateral_dist: float = 0.12,
    stillness_std: float = 2.0,
) -> torch.Tensor:
    """Reward keeping the torso CoM horizontally over the foot support polygon,
    gated by stillness so it rewards HOLDING the CoM over the feet, not
    swinging it there.

    Approximates the CoM by the ``base_link`` world xy position and the support
    polygon by the midpoint between the two foot bodies' world xy positions.
    The base reward is a Gaussian in the horizontal distance, bounded in
    ``[0, 1]`` (1.0 when CoM is directly over the support midpoint). This is
    then multiplied by a stillness gate
    ``exp(-Σv² / stillness_std²)`` over all joint velocities: 1.0 when the
    robot is holding still, decaying toward 0 when the joints are moving fast.

    Why the stillness gate is essential: without it, the policy can earn the
    full carrot by *swinging* the CoM through the support midpoint on every
    oscillation — the reward fires on position, not velocity, so a flailing
    motion that passes through the target earns as much as a stable hold.
    Capture 20260715_122500 showed this exploit exactly: ``com_over_support``
    climbed to +0.39 while the robot thrashed (vel_std 0.69-1.34 rad/s, g_x
    swinging ±30°). The stillness gate makes the reward pay out only when the
    CoM is over the feet AND the robot is settling — turning the carrot from a
    swing-enabler into a hold-enforcer. The carrot still provides a positive
    gradient *toward* the balanced pose during slow recovery (the gate is ~0.5
    at moderate motion, so a controlled step still earns partial credit), it
    just no longer subsidizes chatter.

    The previous ``tilt_boost_gain`` (which scaled the reward up with tilt) is
    removed: it was intended to emphasize recovery, but in practice it made
    the flailing exploit *more* profitable at tilt — exactly when the gate on
    the movement penalties was most relaxed. The carrot should not grow when
    the robot is falling; the ``alive`` and ``torso_posture`` terms own the
    survival gradient, and this term owns the steady-state CoM target.

    Non-privileged: uses only ``body_pos_w`` (root link + foot bodies) and
    ``joint_vel``, both derivable from joint encoders + FK on hardware.

    Args:
        asset_cfg: the robot articulation.
        foot_body_names: ``(left, right)`` foot body names whose midpoint
            approximates the support polygon center.
        max_lateral_dist: distance (m) at which the reward is ~1/e. ~0.12 m
            matches half a foot length — beyond that the CoM is over the edge.
        stillness_std: velocity scale for the stillness gate. ``2.0`` means the
            gate is ~1/e once summed squared joint velocity reaches 4.0
            (rad/s)². Generous enough that a slow controlled recovery step
            still earns partial credit, strict enough that fast flailing earns
            ~0.
    """
    asset = env.scene[asset_cfg.name]
    device = getattr(env, "device", None)
    body_pos_w = _ensure_tensor(asset.data.body_pos_w, env_device=device)

    base_xy = body_pos_w[:, asset_cfg.body_ids[0], :2]

    foot_ids = [asset.body_names.index(name) for name in foot_body_names]
    foot_mid_xy = 0.5 * (
        body_pos_w[:, foot_ids[0], :2] + body_pos_w[:, foot_ids[1], :2]
    )

    lateral_dist = (base_xy - foot_mid_xy).norm(dim=-1)
    base_reward = torch.exp(
        -torch.square(lateral_dist) / (max_lateral_dist * max_lateral_dist)
    )

    joint_vel = _ensure_tensor(asset.data.joint_vel, env_device=device)
    vel_sq = torch.sum(torch.square(joint_vel), dim=1)
    stillness_gate = torch.exp(-vel_sq / (stillness_std * stillness_std))

    return base_reward * stillness_gate


def upright_pose_exp(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    std: float = 0.25,
) -> torch.Tensor:
    """Bounded reward for holding the torso upright (gravity ∥ body z).

    Returns ``exp(-(g_x² + g_y²) / σ²)`` over the horizontal components of the
    articulation root ``projected_gravity_b``. This is ``1.0`` when the torso
    is perfectly upright (gravity points straight down through the body
    z-axis) and decays smoothly as the torso tilts in any direction. Bounded
    in ``[0, 1]`` so it can carry a firm positive weight without runaway
    risk, and unlike a deviation penalty it provides a smooth positive
    gradient *all the way to vertical* — the policy earns more by being MORE
    upright, not just by being less tilted than last tick.

    Symmetric in pitch and roll (both ``g_x`` and ``g_y``), so it does not
    bias the policy toward a forward or back lean — pair with
    :func:`forward_lean_penalty` for the asymmetric anti-toe-lean term.
    Non-privileged: uses the same root ``projected_gravity_b`` the policy
    observes (via ``mdp.projected_gravity``) and the firmware ships, so it
    creates no sim-to-real gap.

    Args:
        asset_cfg: articulation whose root ``projected_gravity_b`` to read.
            Defaults to the ``"robot"`` scene entity; the IMU sensor's
            ``projected_gravity_b`` was removed in Isaac Lab 3.0 beta2, so we
            read the articulation root (``base_link``) gravity projection
            instead — equivalent for an IMU mounted with identity orientation.
        std: gravity-component scale. ``0.25`` ≈ 14° half-angle at 1/e;
            smaller demands a more precise upright hold.
    """
    asset = env.scene[asset_cfg.name]
    proj_grav = _ensure_tensor(
        asset.data.projected_gravity_b, env_device=getattr(env, "device", None)
    )
    g_xy_sq = torch.square(proj_grav[:, 0]) + torch.square(proj_grav[:, 1])
    return torch.exp(-g_xy_sq / (std * std))


def torso_pitch_asymmetric_reward(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    band_gx_min: float = -0.30,
    band_gx_max: float = -0.17,
    edge_std: float = 0.12,
    roll_std: float = 0.15,
    forward_penalty_gain: float = 5.0,
    forward_deadband: float = 0.0,
) -> torch.Tensor:
    """Reward balancing anywhere in a back-lean *band*; penalize forward pitch.

    Uses the same articulation root ``projected_gravity_b`` signal as
    ``mdp.projected_gravity`` and the firmware observation builder.
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
        asset_cfg: articulation whose root ``projected_gravity_b`` to read.
            Defaults to the ``"robot"`` scene entity; the IMU sensor's
            ``projected_gravity_b`` was removed in Isaac Lab 3.0 beta2, so we
            read the articulation root (``base_link``) gravity projection
            instead — equivalent for an IMU mounted with identity orientation.
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
    asset = env.scene[asset_cfg.name]
    proj_grav = _ensure_tensor(
        asset.data.projected_gravity_b, env_device=getattr(env, "device", None)
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


def feet_flat_orientation_l2(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    foot_body_names: tuple[str, str] = ("foot_left_1", "foot_right_1"),
    sole_normal_b: tuple[float, float, float] = (0.0, 0.0, 1.0),
    stillness_std: float = 1.5,
    balance_gate: bool = False,
    gate_band_gx_center: float = -0.17,
    gate_std: float = 0.15,
    gate_floor: float = 0.2,
) -> torch.Tensor:
    """Penalize feet whose soles are not parallel to the ground (flat-foot stand).

    For each foot body, rotates the sole normal (``sole_normal_b``, expressed
    in the foot link frame) into the world frame and returns the summed
    squared horizontal components::

        Σ_feet (u_x² + u_y²)  =  Σ_feet sin²(foot tilt from flat)

    which is ``0`` when both soles are parallel to the ground and bounded in
    ``[0, 2]`` (each foot contributes ``sin²`` of its tilt angle, quadratic
    near flat so the shaping gradient grows with the misalignment). Always
    ``>= 0``; use with a **negative** weight.

    Foot orientations come from ``asset.data.body_quat_w`` — Isaac Lab 3.0
    returns quaternions in ``(x, y, z, w)`` order (breaking change from the
    2.x ``(w, x, y, z)`` convention) — and the rotation uses
    :func:`isaaclab.utils.math.quat_apply` so the ordering convention stays
    owned by the library, not by hand-rolled math here.

    WHY a foot-orientation term, on top of the existing posture terms: the
    torso back-lean band + hip-flexion anchor only constrain the *leg chain*
    — the ankle (foot joint) is free to hold the sole at any angle, and the
    sim exploits that: the rigid foot's contact patch lets the policy ride
    the toe or heel edge with a tilted sole, a balance strategy that does
    NOT transfer to hardware (the real foot is small, the RS02 ankle is the
    weakest joint at ~17 N·m, and an edge contact shrinks the support
    polygon to a line). Forcing the sole parallel to the ground maximizes
    the contact polygon and stacks the load through the whole foot instead
    of leaning on the ankle motor. This is the foot-side counterpart to
    ``hip_flexion_anchor``: hips straight + soles flat = the ankle strategy,
    the entire body leaning together over a full contact patch.

    This is deliberately NOT a joint-position anchor on the foot joints:
    under the 8-12° back lean the shank tilts back with the torso, so a flat
    sole requires the ankle to hold a *nonzero* compensation angle
    (q_foot ≈ ±10° in the mirrored sign convention — cf. capture
    20260715_214224: foot L=+0.18, R=+0.31 rad). Anchoring q_foot at 0 would
    fight the back lean; this orientation term targets the *result* (sole ∥
    ground) and lets the ankle find whatever angle achieves it.

    SOLE NORMAL — why the default ``(0, 0, 1)``: every leg-chain joint
    origin in ``ros2/src/bebopv2_description/urdf/bebopv2.urdf`` has
    ``rpy="0 0 0"`` (verified Jul 16 2026: hip_flexion, hip_abduction,
    knee_flexion, foot joints, both sides), so at the all-zeros default pose
    — the documented straight-leg symmetric stand with both feet flat — the
    foot link frame is aligned with the base_link FLU frame and the sole
    normal is the foot's local ``+Z``. If a future foot redesign rotates the
    foot frame in the USD, pass the corrected axis via ``sole_normal_b``
    instead of editing this function.

    **Gate** — two mutually exclusive options, selected by ``balance_gate``
    (same machinery as :func:`bilateral_joint_symmetry_l2`):

    * **Stillness gate** (``balance_gate=False``, default):
      ``exp(-Σv²/σ²)`` over all joint velocities. Fires at full strength
      whenever the robot is holding a pose — "when standing" — regardless of
      torso tilt, and decays toward ``0`` during active motion, so recovery
      footwork (toe-off, heel strike, lifting a foot) stays free. Flat feet
      are a POSTURE constraint like bilateral symmetry, not a motion
      penalty, so the stillness gate is the right default (see the
      ``bilateral_symmetry`` term comment in ``exp_standing.py`` for why
      posture constraints use stillness, not balance, gating — a
      balance-gated posture term drops to its floor during the
      tilted-exploration phase of training and never shapes the stance).
    * **Balance gate** (``balance_gate=True``):
      ``max(gate_floor, exp(-((g_x - center)² + g_y²) / gate_std²))`` —
      the same tilt-distance Gaussian used by ``joint_vel_l2`` and
      ``action_position_rate_l2``. Full strength in the back-lean band,
      relaxing toward ``gate_floor`` when tilted.

    Non-privileged: foot orientation is forward kinematics from the joint
    encoders + root IMU, both observed on hardware — same sim-to-real
    justification as :func:`com_over_support_reward`.

    Args:
        asset_cfg: the robot articulation.
        foot_body_names: the foot articulation body names to score.
        sole_normal_b: unit vector in the foot link frame normal to the sole
            (points world ``+Z`` when the foot is flat). Default ``(0,0,1)``
            — see the frame audit above.
        stillness_std: velocity scale for the stillness gate (ignored when
            ``balance_gate=True``). ``1.5`` matches ``bilateral_symmetry``.
        balance_gate: if True, use the tilt-distance Gaussian gate instead
            of the stillness gate.
        gate_band_gx_center: ``g_x`` center of the balance gate.
        gate_std: Gaussian width; ~0.10 ≈ 5.7° at 1/e.
        gate_floor: minimum gate value (0..1).
    """
    asset = env.scene[asset_cfg.name]
    device = getattr(env, "device", None)
    body_quat_w = _ensure_tensor(asset.data.body_quat_w, env_device=device)

    err = None
    for body_name in foot_body_names:
        body_id = asset.body_names.index(body_name)
        # (N, 4) — Isaac Lab 3.0 quaternion order is (x, y, z, w); quat_apply
        # owns the convention (the pre-3.0 WXYZ unpack made X-axis roll —
        # the lateral edge-riding this term exists to prevent — invisible).
        quat = body_quat_w[:, body_id]
        normal = torch.tensor(
            sole_normal_b, dtype=quat.dtype, device=quat.device
        ).expand(quat.shape[0], 3)
        # Rotate the sole normal into the world frame: u = R(q) @ n.
        sole_normal_w = quat_apply(quat, normal)
        # Horizontal components of the world-frame sole normal: 0 when the
        # sole is parallel to the ground, sin²(tilt) otherwise.
        misalign = torch.sum(torch.square(sole_normal_w[:, :2]), dim=1)
        err = misalign if err is None else err + misalign

    if balance_gate:
        proj_grav = _ensure_tensor(asset.data.projected_gravity_b, env_device=device)
        g_x = proj_grav[:, 0]
        g_y = proj_grav[:, 1]
        tilt_err = torch.square(g_x - gate_band_gx_center) + torch.square(g_y)
        gate = torch.exp(-tilt_err / (gate_std * gate_std))
        gate = torch.clamp(gate, min=gate_floor)
        return err * gate

    # Stillness gate: 1.0 when still, -> 0 when moving. Only enforce flat
    # soles when the policy is trying to HOLD a stance, not when it's
    # articulating the feet for a recovery step.
    joint_vel = _ensure_tensor(asset.data.joint_vel, env_device=device)
    vel_sq = torch.sum(torch.square(joint_vel), dim=1)
    gate = torch.exp(-vel_sq / (stillness_std * stillness_std))
    return err * gate
