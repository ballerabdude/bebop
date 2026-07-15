"""Custom reward functions for the Bebop V2 articulation."""

from __future__ import annotations

import torch
import warp as wp

from isaaclab.managers import SceneEntityCfg


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
    ``exp(-((g_x - center)² + g_y²) / gate_std²)``, a Gaussian on how far the
    torso tilt is from the balance target. The penalty fires at FULL strength
    when the robot is balanced (kills steady-state chatter, the term's actual
    job) and vanishes when the robot is tilted (frees the policy to move the
    setpoints rapidly to catch a fall). Without this gate a high weight (-10)
    makes recovery unaffordable: at 12° off the band, a 0.1 rad recovery across
    4 joints costs -0.4/tick while ``torso_posture`` only earns +0.028/tick —
    the policy correctly chooses not to move and hangs forward (capture
    20260715_041834, gantry-supported: g_x mean +0.065 vs band [-0.208,-0.139]).
    The gate makes the same recovery cost -0.031/tick at tilt, restoring the
    incentive to actively correct. At balance the gate is 1.0, so the anti-
    chatter job is unchanged.

    Args:
        balance_gate: if True, apply the tilt-distance Gaussian gate.
        gate_band_gx_center: ``g_x`` center of the gate (the balance target).
        gate_std: Gaussian width; ~0.15 means the gate is ~1/e at ~8.6° off.
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
) -> torch.Tensor:
    """Penalize any joint motion — the sharpest ``be absolutely still`` signal.

    Returns ``Σ v²`` over the selected joint velocities; always ``>= 0``, use
    with a negative weight. Unlike :func:`stationary_pose_exp` (a bounded
    ``exp(-Σv²/σ²)`` kernel whose gradient vanishes as ``v → 0``), this is an
    unbounded quadratic that keeps a strong, linear-in-``v`` gradient all the
    way down to zero velocity — which is what actually kills small limit-cycle
    oscillations. Pair with the action-rate terms; this is the plant-side
    counterpart (penalize the *result* of the chatter, not just the chatter).

    **Balance gate** (``balance_gate=True``): multiplies the penalty by
    ``exp(-((g_x - center)² + g_y²) / gate_std²)`` (same gate as
    :func:`action_position_rate_l2`). At full strength when balanced (kills
    residual wobble, the term's job) and vanishes when tilted (frees the policy
    to swing the legs to catch a fall). Without the gate the -0.5 weight makes
    recovery unaffordable at tilt — see :func:`action_position_rate_l2` for the
    full rationale.
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
) -> torch.Tensor:
    """Penalize asymmetry between left/right joint pairs (sagittal plane).

    Returns ``Σ (q_left + q_right)²`` over the listed joint name pairs, gated
    by a *stillness* multiplier so the penalty only fires when the robot is
    holding a pose — NOT during recovery transients when the policy needs
    asymmetric motions to catch a fall. Always ``>= 0``; use with a negative
    weight.

    SIGN CONVENTION — why the *sum*, not the difference: every L/R joint
    pair on this robot is sign-MIRRORED. The right-side pitch joints
    (hip_flexion, knee_flexion, foot) use a flipped ``-Y`` axis in the
    URDF/USD, and the abduction rolls physically mirror across the sagittal
    plane (URDF limits: left ``[-10°, +20°]`` vs right ``[-20°, +10°]``;
    knee: left ``[-45°, +90°]`` vs right ``[-90°, +45°]``). The same
    physical "both knees flexed the same way" pose therefore reads
    ``q_left = -q_right``, so the mirrored stance is ``q_L + q_R = 0``.
    Penalizing ``(q_L - q_R)²`` (the pre-Jul-2026 version of this term)
    rewarded ``q_L = q_R`` — one leg pitched forward and the other back,
    both hips rolled the same way — i.e. it actively TRAINED the
    twisted-hip contortion it was meant to prevent, and raising its weight
    only pushed harder in the wrong direction (observed on hardware run
    2026-07-09_04-16-40: hip_flexion L-R driven to ~0.01 while the robot
    stood visibly twisted).

    The stillness gate is ``exp(-Σv²/σ²)`` over all joint velocities, where
    ``σ = stillness_std``. This is ``1.0`` when the robot is perfectly still
    (the penalty fires at full strength) and decays toward ``0`` when joints
    are moving fast (the penalty is suppressed so the policy is free to use
    whatever asymmetric catch/recovery motion it needs). This directly
    addresses the failure mode where the symmetry penalty was fighting the
    policy during recovery: every time the robot tipped and the policy tried
    an asymmetric catch, it got penalized, pushing it back toward falling.

    Args:
        pairs: list of ``(left_joint_name, right_joint_name)`` tuples. Names
            are resolved to articulation joint indices, so the caller does not
            need to know the USD joint order.
        asset_cfg: the robot articulation.
        stillness_std: velocity scale for the stillness gate. ``1.5`` means the
            gate is ~1/e once the summed squared joint velocity reaches 1.5².
            Larger = more lenient (penalty fires even with some motion);
            smaller = stricter (penalty only fires when nearly motionless).
    """
    asset = env.scene[asset_cfg.name]
    joint_pos = _ensure_tensor(
        asset.data.joint_pos, env_device=getattr(env, "device", None)
    )
    joint_vel = _ensure_tensor(
        asset.data.joint_vel, env_device=getattr(env, "device", None)
    )

    err = None
    for left_name, right_name in pairs:
        li = asset.data.joint_names.index(left_name)
        ri = asset.data.joint_names.index(right_name)
        # Mirrored sign convention: a symmetric stance is q_L == -q_R,
        # so the asymmetry residual is the SUM (see docstring).
        d = joint_pos[:, li] + joint_pos[:, ri]
        sq = d * d
        err = sq if err is None else err + sq

    # Stillness gate: 1.0 when still, -> 0 when moving. Only penalize
    # asymmetry when the policy is trying to HOLD a pose, not when it's
    # recovering from a perturbation.
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
) -> torch.Tensor:
    """``Σ |q|`` over selected joints, gated by how close to balance the torso is.

    The balance-gate counterpart to :func:`joint_deviation_l1_stillness_gated`.
    Multiplies ``Σ |q|`` by ``exp(-((g_x - center)² + g_y²) / gate_std²)``: the
    penalty fires at full strength when the robot is balanced (holds the
    anchored joints at their home pose) and vanishes when tilted (frees those
    joints to move for active recovery). Always ``>= 0``; use with a negative
    weight.

    Used for the hip-flexion anchor in the active-balance reward: the hips
    must stay straight when balanced (the ankle strategy — entire body leans
    together) but MUST be free to flex when the torso is falling, because
    swinging the legs to put a foot under the falling CoM is the primary
    balance-recovery motion. A non-gated hip-flexion anchor at -0.6 made that
    recovery motion unaffordable and the policy learned to hang forward instead
    of correcting (capture 20260715_041834: g_x mean +0.065, 12° off the
    back-lean band, on a gantry that caught the fall).

    Args:
        asset_cfg: which articulation / joints to score the deviation on.
        gate_band_gx_center: ``g_x`` center of the balance gate.
        gate_std: Gaussian width; ~0.15 ≈ 8.6° at 1/e.
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
    return dev * gate


def com_over_support_reward(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    foot_body_names: tuple[str, str] = ("foot_left_1", "foot_right_1"),
    max_lateral_dist: float = 0.12,
    tilt_boost_gain: float = 4.0,
) -> torch.Tensor:
    """Reward keeping the torso CoM horizontally over the foot support polygon.

    Approximates the CoM by the ``base_link`` world xy position and the support
    polygon by the midpoint between the two foot bodies' world xy positions.
    Returns a bounded reward in ``[0, 1]`` that is 1.0 when the CoM is directly
    over the support midpoint and falls off as a Gaussian in the horizontal
    distance. This is the positive carrot that complements the balance gates:
    the gates (on ``position_rate`` / ``joint_vel`` / ``hip_flexion_anchor``)
    remove the penalty that blocked recovery, and this term adds a positive
    gradient *toward* the balanced pose, growing stronger the more tilted the
    robot is (so a falling robot has the largest incentive to step/catch).

    Sign convention / why lateral distance: a falling robot is one whose CoM
    has moved outside the support polygon (over the edge of the foot). The
    most direct non-privileged signal for "get the CoM back over the feet" is
    the horizontal distance from the torso to the foot midpoint. Penalizing
    that distance (or rewarding its negation) gives the policy a clean target
    that is independent of the ankle motor's weakness: even when the RS02
    can't hold a torque, the policy can shift the CoM by flexing the hips and
    stepping. This term is what makes active leg-lifting the rewarded
    recovery strategy rather than just a tolerated one.

    The reward is also scaled by ``1 + tilt_boost_gain * (g_x² + g_y²)`` so it
    is small when balanced (letting ``torso_posture`` own the steady-state
    shape) and grows when tilted (where the recovery incentive must dominate).
    At ``tilt_boost_gain = 4.0`` and 12° off-band (g_x ≈ 0.21), the boost is
    ~1.18x — a mild carrot at the capture's failure tilt, growing for deeper
    falls.

    Non-privileged: uses only ``body_pos_w`` (root link + foot bodies), which
    the firmware can reconstruct from joint encoders + forward kinematics.
    No contact sensing required — the midpoint is taken unconditionally, which
    is correct for a two-foot stand and degrades gracefully during a step
    (the midpoint just tracks the lifted foot's horizontal position).

    Args:
        asset_cfg: the robot articulation.
        foot_body_names: ``(left, right)`` foot body names whose midpoint
            approximates the support polygon center.
        max_lateral_dist: distance (m) at which the reward is ~1/e. ~0.12 m
            matches half a foot length — beyond that the CoM is over the edge.
        tilt_boost_gain: scales how much the reward grows with tilt. 0 makes
            the term tilt-invariant (pure CoM tracking); 4.0 gives a mild
            recovery emphasis.
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

    proj_grav = _ensure_tensor(asset.data.projected_gravity_b, env_device=device)
    tilt_sq = torch.square(proj_grav[:, 0]) + torch.square(proj_grav[:, 1])
    boost = 1.0 + tilt_boost_gain * tilt_sq

    return base_reward * boost


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
