"""Custom reward functions for the Bebop V2 articulation."""

from __future__ import annotations

import math

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


def _com_off_support_dist(
    env,
    asset,
    foot_body_names: tuple[str, str] = ("foot_left_1", "foot_right_1"),
) -> torch.Tensor:
    """Horizontal distance of the torso CoM proxy from the foot midpoint.

    Approximates the whole-body CoM by the ``base_link`` world xy position
    (the torso is by far the heaviest link, ~6.7 kg of ~14 kg, so it is a
    good CoM proxy) and the support polygon center by the midpoint between
    the two foot bodies' world xy positions. Returns ``‖base_xy −
    foot_mid_xy‖`` (metres), ``>= 0``.

    Non-privileged: uses only ``body_pos_w`` (root link + foot bodies),
    derivable from joint encoders + FK on hardware — same sim-to-real
    justification as :func:`com_over_support_reward`.

    Shared by :func:`com_over_support_reward` (the CoM-over-feet carrot) and
    the ``com_gate`` path of :func:`action_position_rate_l2` /
    :func:`joint_vel_l2` (the balance gate keyed on CoM excursion instead of
    torso tilt — see those functions for why the knee+ankle strategy needs
    it).

    Args:
        env: the environment (for the device).
        asset: the robot articulation.
        foot_body_names: ``(left, right)`` foot body names whose midpoint
            approximates the support polygon center.
    """
    device = getattr(env, "device", None)
    body_pos_w = _ensure_tensor(asset.data.body_pos_w, env_device=device)

    base_id = asset.body_names.index("base_link")
    base_xy = body_pos_w[:, base_id, :2]

    foot_ids = [asset.body_names.index(name) for name in foot_body_names]
    foot_mid_xy = 0.5 * (
        body_pos_w[:, foot_ids[0], :2] + body_pos_w[:, foot_ids[1], :2]
    )

    return (base_xy - foot_mid_xy).norm(dim=-1)


def _balance_gate_from_dist(
    dist: torch.Tensor,
    gate_std: float,
    gate_floor: float,
) -> torch.Tensor:
    """CoM-excursion balance gate: 1.0 when centered, decaying to a floor.

    Returns ``max(gate_floor, exp(-(dist / gate_std)^2))``. The movement
    penalties multiply by this gate so they fire at full strength when the
    CoM is over the support (kills steady-state chatter, their actual job)
    and relax toward ``gate_floor`` when the CoM is off the support (frees
    the legs to move for recovery). The floor keeps a minimum anti-chatter
    pressure at full excursion so the policy can't manufacture CoM swings to
    unlock chatter (the flailing-limit-cycle exploit — see
    :func:`action_position_rate_l2`).

    Args:
        dist: per-env CoM-off-support distance (metres), e.g. from
            :func:`_com_off_support_dist`.
        gate_std: Gaussian width (metres); the gate is ~1/e once the CoM is
            one ``gate_std`` off the support center.
        gate_floor: minimum gate value (0..1).
    """
    gate = torch.exp(-torch.square(dist) / (gate_std * gate_std))
    return torch.clamp(gate, min=gate_floor)


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

    A CoM-excursion balance gate (``balance_gate``) was TRIED here Jul 21 2026
    (round 3) and REVERTED the same day: the gate can't tell "legitimate
    recovery" from "destructive flailing" — both have the CoM off-support —
    so it relaxes the penalty to its 0.2 floor *exactly during the flailing
    phase*, when the policy most needs the penalty to brake exploration. With
    the brake gone, PPO's entropy bonus pushed ``mean_std`` to 5.97 (vs 1.30
    in the ungated round-2 run), the policy flailed, and eplen cratered to
    ~65 (run 2026-07-22_00-00-48). Keeping ``gain_rate`` UNGATED (full -2.0
    always) preserves the anti-flail governor that keeps ``mean_std`` bounded
    so the policy can learn; it is the one movement penalty that must NOT be
    CoM-gated. The net-negative-while-alive budget that motivated the gate is
    instead fixed by raising ``alive`` (see the ``alive`` term in
    ``RewardsCfg``).

    UNWIRED Jul 22 2026 (round 4) — then REWIRED the same day (round 5):
    the round-4 push runs (10-55-08, 12-29-28) showed the tax-vs-survival
    conflict this penalty created — at high push force the policy's ONLY
    survival strategy was high-bandwidth kp/kd modulation, so the -2.0
    quadratic tax grew with competence (reward collapsed +27.7 -> -21
    while eplen still rose). Smoothness is now enforced STRUCTURALLY in
    the action space (``gain_ema_tau_s=0.15`` EMA on decoded kp/kd), so
    this tax no longer punishes survival — but deleting it entirely let
    the shared log-std explode to 4.7e15 (run 2026-07-22_20-49-06): with
    the gain channels filtered, no task pressure keeps their exploration
    noise small. Re-added at -2.0 as the anti-noise std governor — the
    role it was silently load-bearing for in every prior ungated run
    (std bounded ~1.2).
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
    gate_band_gx_center: float = 0.015,
    gate_std: float = 0.15,
    gate_floor: float = 0.2,
    com_gate: bool = False,
    foot_body_names: tuple[str, str] = ("foot_left_1", "foot_right_1"),
    gate_dist_std: float = 0.06,
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

    **Balance gate** — two mutually exclusive gate signals, selected by
    ``com_gate`` (both relax the penalty during recovery, both keep a
    ``gate_floor`` so the penalty never vanishes entirely):

    * **Tilt gate** (``com_gate=False``, default): multiplies the penalty by
      ``max(gate_floor, exp(-((g_x - center)² + g_y²) / gate_std²))``. Fires
      at full strength when the torso is at the balance-band pitch, decays
      toward ``gate_floor`` when the torso is tilted. This was the Jul 15
      2026 design, built for the torso-pitch balancing strategy: recovery
      pitched the torso, which opened the gate and made recovery affordable.

    * **CoM gate** (``com_gate=True``, Jul 20 2026): multiplies the penalty
      by ``max(gate_floor, exp(-(dist / gate_dist_std)²))`` where ``dist``
      is the CoM-off-support distance from :func:`_com_off_support_dist`.
      Fires at full strength when the CoM is over the feet, decays toward
      ``gate_floor`` as the CoM heads off the support polygon. Required for
      the knee+ankle CoG-control strategy: that strategy pins the torso at
      0° and recovers from pushes by articulating the knees/ankles WHILE
      KEEPING THE TORSO LEVEL, so a torso-tilt gate stays closed (gate≈1)
      during exactly the recovery motion it should be relaxing for — the
      policy was being charged full anti-chatter price to recover
      (push-task run 2026-07-21_03-21-08 plateaued at ~25% eplen with
      gain_rate/position_rate the dominant penalties). Keying on CoM
      excursion — the actual balance criterion — opens the gate during a
      level-torso recovery, making the desired knee/ankle recovery
      affordable while still killing chatter at balance.

    The floor is essential in both modes: without it (gate_floor=0) the
    penalty vanishes completely at tilt/excursion and the policy learns to
    *manufacture* tilt/CoM-swing to unlock chatter — exactly the flailing
    limit cycle seen in capture 20260715_122500 (vel_std 0.69-1.34 rad/s,
    slew-exceedance 56.9%, g_x swinging ±30°). With gate_floor=0.2 the
    -10.0 weight still contributes -2.0·rate at full excursion — enough to
    keep recovery motions smooth without making them unaffordable.

    Args:
        balance_gate: if True, apply a balance gate (see ``com_gate`` for
            which signal).
        gate_band_gx_center: ``g_x`` center of the tilt gate (ignored when
            ``com_gate=True``).
        gate_std: tilt-gate Gaussian width; ~0.10 means the gate is ~1/e at
            ~5.7° off (ignored when ``com_gate=True``).
        gate_floor: minimum gate value (0..1). Prevents the penalty from
            vanishing entirely at tilt/excursion, which the policy otherwise
            exploits.
        com_gate: if True, key the balance gate on CoM-off-support distance
            instead of torso tilt (the knee+ankle-strategy gate).
        foot_body_names: ``(left, right)`` foot body names for the CoM gate
            (ignored when ``com_gate=False``).
        gate_dist_std: CoM-gate Gaussian width (metres); the gate is ~1/e
            once the CoM is one ``gate_dist_std`` off the support center.
            ~0.06 m ≈ the old 5.7° tilt-gate width (0.65 m torso height ×
            tan(5.7°)). Ignored when ``com_gate=False``.
        asset_cfg: articulation for the gate's ``projected_gravity_b`` (tilt
            gate) and ``body_pos_w`` (CoM gate).
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

    if com_gate:
        dist = _com_off_support_dist(env, asset, foot_body_names)
        gate = _balance_gate_from_dist(dist, gate_dist_std, gate_floor)
        return rate * gate

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
    gate_band_gx_center: float = 0.015,
    gate_std: float = 0.15,
    gate_floor: float = 0.2,
    com_gate: bool = False,
    foot_body_names: tuple[str, str] = ("foot_left_1", "foot_right_1"),
    gate_dist_std: float = 0.06,
) -> torch.Tensor:
    """Penalize any joint motion — the sharpest ``be absolutely still`` signal.

    Returns ``Σ v²`` over the selected joint velocities; always ``>= 0``, use
    with a negative weight. Unlike :func:`stationary_pose_exp` (a bounded
    ``exp(-Σv²/σ²)`` kernel whose gradient vanishes as ``v → 0``), this is an
    unbounded quadratic that keeps a strong, linear-in-``v`` gradient all the
    way down to zero velocity — which is what actually kills small limit-cycle
    oscillations. Pair with the action-rate terms; this is the plant-side
    counterpart (penalize the *result* of the chatter, not just the chatter).

    **Balance gate** — two mutually exclusive gate signals, selected by
    ``com_gate`` (both relax the penalty during recovery, both keep a
    ``gate_floor``):

    * **Tilt gate** (``com_gate=False``, default): same gate as the
      pre-Jul-20 :func:`action_position_rate_l2` —
      ``max(gate_floor, exp(-((g_x - center)² + g_y²) / gate_std²))``. At
      full strength when the torso is at the balance-band pitch (kills
      residual wobble) and decays to ``gate_floor`` when tilted.

    * **CoM gate** (``com_gate=True``, Jul 20 2026):
      ``max(gate_floor, exp(-(dist / gate_dist_std)²))`` where ``dist`` is
      the CoM-off-support distance from :func:`_com_off_support_dist`. Fires
      at full strength when the CoM is over the feet, decays toward
      ``gate_floor`` as the CoM heads off the support. Required for the
      knee+ankle CoG-control strategy (which keeps the torso level during
      recovery, so a torso-tilt gate never opens for it) — see
      :func:`action_position_rate_l2` for the full rationale.

    The floor prevents the flailing-limit-cycle exploit where the policy
    manufactures tilt/CoM-swing to unlock chatter — see
    :func:`action_position_rate_l2` for the capture evidence.

    Args:
        asset_cfg: which articulation / joints to score.
        balance_gate: if True, apply a balance gate (see ``com_gate``).
        gate_band_gx_center: ``g_x`` center of the tilt gate (ignored when
            ``com_gate=True``).
        gate_std: tilt-gate Gaussian width (ignored when ``com_gate=True``).
        gate_floor: minimum gate value (0..1).
        com_gate: if True, key the balance gate on CoM-off-support distance
            instead of torso tilt (the knee+ankle-strategy gate).
        foot_body_names: ``(left, right)`` foot body names for the CoM gate
            (ignored when ``com_gate=False``).
        gate_dist_std: CoM-gate Gaussian width (metres); ~0.06 m ≈ the old
            5.7° tilt-gate width. Ignored when ``com_gate=False``.
    """
    asset = env.scene[asset_cfg.name]
    device = getattr(env, "device", None)
    joint_vel = _ensure_tensor(asset.data.joint_vel, env_device=device)
    if asset_cfg.joint_ids is not None:
        joint_vel = joint_vel[:, asset_cfg.joint_ids]
    vel_sq = torch.sum(torch.square(joint_vel), dim=1)

    if not balance_gate:
        return vel_sq

    if com_gate:
        dist = _com_off_support_dist(env, asset, foot_body_names)
        gate = _balance_gate_from_dist(dist, gate_dist_std, gate_floor)
        return vel_sq * gate

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
    upright_gate: bool = False,
    tilt_std: float = 0.12,
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

    **Upright gate** (``upright_gate=True``, Jul 20 2026): multiplies the
    stillness carrot by ``exp(-(g_x² + g_y²) / tilt_std²)`` over the torso's
    horizontal projected gravity, so the reward only pays full when the robot
    is still AND near-upright. Without this, the carrot rewards holding
    *whatever* pose the policy settles into — INCLUDING A TILTED ONE. The
    push-task run 2026-07-21_03-21-08 showed the exploit: the policy settled
    into a still ~6° torso lean and collected alive (+0.23) + stillness
    (+0.18) while paying the bounded torso_pitch_penalty, instead of
    actively correcting to 0° — "freeze and accept the lean" beat "move to
    correct" because correcting costs joint motion (stillness) plus the
    action-rate penalties. Conditioning stillness on uprightness removes the
    subsidy for a quiet tilted statue: at 0° the gate is 1.0 (full carrot),
    at ~6.9° (``tilt_std = 0.12``) it is 1/e ≈ 0.37, and past ~14° it is
    ≈ 0.05 — so a tilted statue earns almost nothing and only an UPRIGHT
    still stand pays.

    A CoM-on-support balance gate was TRIED here Jul 21 2026 (round 3) and
    REVERTED the same day: it was both ineffective (a *frozen balanced*
    statue has the CoM over the feet, so the gate is 1.0 — it does nothing
    to the freeze it was meant to kill) and harmful (it zeroed the carrot
    during early learning when the robot is always falling, making the reward
    too sparse for the policy to bootstrap — run 2026-07-21_22-27-00
    regressed to eplen 71 with std exploding to 4.19). The freeze exploit is
    instead addressed by raising ``alive`` so the net per-step reward while
    alive is positive (see the ``alive`` term in ``RewardsCfg``): with a
    positive net, surviving long (which only active balance can do against
    pushes) always beats freezing (which topples at the first shove).

    Non-privileged: reads the articulation root ``projected_gravity_b`` (the
    same signal the policy observes) and joint velocities (encoders).

    Args:
        std: velocity scale (rad/s-ish). The reward is ~1/e once the summed
            squared joint velocity reaches ``std²``. Smaller => stricter
            "be perfectly still" requirement.
        asset_cfg: which articulation / joints to score (defaults to all
            joints of ``robot``).
        upright_gate: if True, multiply by an uprightness Gaussian so the
            carrot only pays when still AND near-upright.
        tilt_std: torso-tilt Gaussian width (in g_xy units) for the upright
            gate; ~0.12 means the gate is ~1/e at ~6.9° tilt. Ignored when
            ``upright_gate=False``.
    """
    asset = env.scene[asset_cfg.name]
    device = getattr(env, "device", None)
    joint_vel = _ensure_tensor(asset.data.joint_vel, env_device=device)
    if asset_cfg.joint_ids is not None:
        joint_vel = joint_vel[:, asset_cfg.joint_ids]
    err = torch.sum(torch.square(joint_vel), dim=1)
    reward = torch.exp(-err / (std * std))

    if upright_gate:
        proj_grav = _ensure_tensor(
            asset.data.projected_gravity_b, env_device=device
        )
        g_xy_sq = torch.square(proj_grav[:, 0]) + torch.square(proj_grav[:, 1])
        reward = reward * torch.exp(-g_xy_sq / (tilt_std * tilt_std))

    return reward


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
    gate_band_gx_center: float = 0.015,
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
    gate_band_gx_center: float = 0.015,
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

    lateral_dist = _com_off_support_dist(env, asset, foot_body_names)
    base_reward = torch.exp(
        -torch.square(lateral_dist) / (max_lateral_dist * max_lateral_dist)
    )

    joint_vel = _ensure_tensor(asset.data.joint_vel, env_device=device)
    vel_sq = torch.sum(torch.square(joint_vel), dim=1)
    stillness_gate = torch.exp(-vel_sq / (stillness_std * stillness_std))

    return base_reward * stillness_gate


def crouch_height_reward(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_height: float = 0.55,
    height_std: float = 0.06,
    upright_gate: bool = False,
    tilt_std: float = 0.12,
) -> torch.Tensor:
    """Bounded [0, 1] reward for holding the torso at a crouch height.

    Returns ``exp(-((z_base - target_height) / height_std)²)`` over the
    ``base_link`` world z. ``1.0`` when the base is exactly at
    ``target_height``, decaying smoothly as the base rises (too stiff) or
    drops (too deep). Upright-gated so it does not reward a crouch the robot
    is falling into — only a deliberate, balanced crouch earns the carrot.

    **Why a crouch, not a joint anchor:** the knee+ankle CoG strategy needs
    the robot to adopt an athletic stance — knees bent, CoM lowered — so the
    legs can articulate to steer the CoM. Without this term the policy has no
    gradient toward crouching: ``standing_stillness`` rewards holding
    *whatever* pose it settles into (including a stiff straight stand), and
    the deviation anchors (``joint_deviation_l1_*``) pull toward zero
    (straight legs) — so the robot freezes stiff and topples at the first
    shove. A height target captures the actual benefit (lower CoM = more
    robust balance) without prescribing *which* joints bend, so the policy
    can find any knee/hip/ankle combo that hits the height. Pair with
    :func:`com_over_support_reward` (CoM over feet AND low) for the full
    "athletic balanced stance" carrot.

    ``target_height = 0.55 m``: the base_link sits at ~0.65 m when standing
    straight (see ``base_link_on_ground`` termination), so 0.55 m is a ~10 cm
    drop — moderate knee flexion, the stance a human adopts when expecting a
    shove. ``height_std = 0.06 m``: at standing height (0.65 m) the reward is
    ``exp(-2.78) ≈ 0.06`` (small but nonzero gradient pulling down), at 0.60 m
    (slight crouch) it is ``exp(-0.69) ≈ 0.50`` (half credit), at 0.55 m it is
    1.0 (full credit), and at 0.50 m (deep crouch) it is ``exp(-2.78) ≈ 0.06``
    again — so the gradient extends from standing through the target to deep.

    Non-privileged: reads ``body_pos_w`` (FK from encoders) and
    ``projected_gravity_b`` (IMU) — same signals the policy observes.

    Args:
        asset_cfg: the articulation whose ``base_link`` z to score.
        target_height: base_link world z (metres) at the target crouch.
        height_std: Gaussian half-width (metres); ``1/e`` at one ``height_std``
            off the target. ``0.06`` gives a smooth gradient from standing
            (0.65 m) to the target (0.55 m).
        upright_gate: if True, multiply by an uprightness Gaussian so the
            carrot only pays when the torso is near-upright (not falling).
        tilt_std: torso-tilt Gaussian width for the upright gate; ~0.12 means
            the gate is ``1/e`` at ~6.9° tilt. Ignored when
            ``upright_gate=False``.

    UNWIRED Jul 22 2026 (round 4): logged +0.0000 over its only run
    (2026-07-22_12-29-28) — the upright gate zeroes it during the tippy
    short episodes that dominate training, and the 0.06 m Gaussian pays
    only ~0.06 at the 0.65 m standing height, so the carrot never
    actuated. Removed from ``RewardsCfg``; kept here for a future
    single-variable retry (likely needs a wider Gaussian and a solid
    pre-trained stand).
    """
    asset = env.scene[asset_cfg.name]
    device = getattr(env, "device", None)
    body_pos_w = _ensure_tensor(asset.data.body_pos_w, env_device=device)

    base_id = asset.body_names.index("base_link")
    base_z = body_pos_w[:, base_id, 2]
    reward = torch.exp(
        -torch.square(base_z - target_height) / (height_std * height_std)
    )

    if upright_gate:
        proj_grav = _ensure_tensor(
            asset.data.projected_gravity_b, env_device=device
        )
        g_xy_sq = torch.square(proj_grav[:, 0]) + torch.square(proj_grav[:, 1])
        reward = reward * torch.exp(-g_xy_sq / (tilt_std * tilt_std))

    return reward


def com_recovery_shaping(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    foot_body_names: tuple[str, str] = ("foot_left_1", "foot_right_1"),
    gamma: float = 0.99,
    reset_jump_threshold: float = 0.2,
) -> torch.Tensor:
    """Potential-based reward shaping for CoM recovery toward the support.

    Returns ``Φ_prev - γ·Φ`` where ``Φ = ‖CoM_xy − foot_midpoint_xy‖`` is the
    CoM-off-support distance (from :func:`_com_off_support_dist`). Positive
    when the CoM is getting closer to the support center (recovering),
    negative when it is heading away (falling), and ~0 when balanced and
    still. This rewards the *recovery motion itself* — including stepping
    back to put a foot under a falling CoM — which the existing
    :func:`com_over_support_reward` cannot do (it only pays when the CoM is
    *already* over the feet AND still).

    **Potential-based shaping** (Ng et al. 1999): because the reward is the
    discounted change in a potential function ``Φ``, it provably preserves
    the optimal policy — it can only accelerate learning, not change what the
    policy converges to. This makes it safe to add a strong weight without
    distorting the objective.

    **Why not a velocity projection:** ``dot(CoM_vel, direction_to_support)``
    only captures CoM movement toward the feet, not *foot movement toward the
    CoM* (stepping). The potential ``Φ = ‖CoM − foot_mid‖`` captures BOTH:
    if the robot steps to put a foot under the falling CoM, the foot
    midpoint moves and ``Φ`` decreases — the shaping rewards it. This is the
    term that teaches "a step back is a valid recovery."

    **Reset handling:** when an env resets, the distance jumps discontinuously
    (the robot teleports). Without handling, this would produce a spurious
    large reward/penalty. Resets are detected by ``|dist − prev_dist| >
    reset_jump_threshold`` (0.2 m — normal per-step change is ~0.01 m at
    100 Hz, so this is never triggered by motion). On detected resets the
    reward is zeroed and the previous-distance buffer is re-seeded.

    **Not hackable:** oscillating the CoM back and forth around the support
    earns +reward on the inward half-cycle and −reward on the outward
    half-cycle, netting ~0 (slightly negative due to ``γ < 1``). Combined
    with the movement penalties (which tax the oscillation), oscillation is
    strictly unprofitable.

    Non-privileged: uses only ``body_pos_w`` (FK from encoders), same as
    :func:`com_over_support_reward` / :func:`_com_off_support_dist`.

    Args:
        asset_cfg: the articulation.
        foot_body_names: ``(left, right)`` foot body names for the support
            midpoint.
        gamma: discount factor (match the PPO config). The shaping is
            ``Φ_prev − γ·Φ``; with ``γ=0.99`` a stationary distance gives
            ``Φ·(1−γ) = 0.01·Φ`` — negligible, so only actual motion earns.
        reset_jump_threshold: distance change (metres) above which a reset is
            assumed and the reward is zeroed. 0.2 m is well above any
            per-step motion (~0.01 m) and well below a reset teleport.

    UNWIRED Jul 22 2026 (round 4): logged +0.0006 over its only run
    (2026-07-22_12-29-28) — the episode MEAN of a potential-based shaping
    term is ~0 by construction (the sum telescopes), so at weight +5.0 it
    contributed nothing against the -2.9/step gain_rate tax it was meant
    to offset, and the run collapsed on the same arc as the run without
    it. Removed from ``RewardsCfg``; kept here for future experiments.
    """
    asset = env.scene[asset_cfg.name]
    dist = _com_off_support_dist(env, asset, foot_body_names)

    if (
        not hasattr(env, "_com_recovery_prev_dist")
        or env._com_recovery_prev_dist.shape != dist.shape
    ):
        env._com_recovery_prev_dist = dist.clone()
        return torch.zeros_like(dist)

    prev = env._com_recovery_prev_dist
    reset_mask = (dist - prev).abs() > reset_jump_threshold
    reward = prev - gamma * dist
    reward[reset_mask] = 0.0
    env._com_recovery_prev_dist = dist.clone()
    return reward


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
    band_gx_min: float = -0.02,
    band_gx_max: float = 0.05,
    edge_std: float = 0.12,
    roll_std: float = 0.15,
    forward_penalty_gain: float = 1.0,
    forward_deadband: float | None = None,
    edge_std_below: float | None = None,
    edge_std_above: float | None = None,
    backward_penalty_gain: float = 1.0,
    backward_deadband: float | None = None,
) -> torch.Tensor:
    """Reward balancing anywhere in a pitch *band*; quadratic tails beyond it.

    Uses the same articulation root ``projected_gravity_b`` signal as
    ``mdp.projected_gravity`` and the firmware observation builder.
    In body FLU (``g_x = -sin(pitch)``):

    * ``proj_grav[:, 0] < 0`` — torso pitched **back** (toward the heel edge)
    * ``proj_grav[:, 0] > 0`` — torso pitched **forward** (toward the toe edge)

    Structure: a flat-top plateau (``1.0`` for any ``g_x`` inside
    ``[band_gx_min, band_gx_max]``), Gaussian shoulders just outside the
    band (asymmetric widths ``edge_std_below`` / ``edge_std_above``), and
    *quadratic tails* (``backward_penalty`` / ``forward_penalty``) that
    keep a nonzero restoring gradient at large tilt where the shoulders'
    gradient has died. The tails are what make "recover once the pitch
    starts to run away" learnable: without them the only far-field signals
    are the flat ``alive`` reward and the termination cliff.

    Where the band sits is a GEOMETRY question, answered by the Jul 17 2026
    audit (URDF masses + foot sole STL, see ``exp_standing.py``): the ankle
    axis is only 23 mm ahead of the heel edge (toe edge +140 mm), and at
    the upright zero pose the CoM already sits 48 mm BEHIND the ankle axis
    — so straight-leg upright standing is statically impossible and any
    back lean deepens the deficit (~60 mm of extra CoM travel per 6 deg).
    The sustainable posture is near-upright with ~5 deg of hip flexion
    (legs slanted back, ankle under the CoM) and/or a slight forward torso
    pitch. The band should therefore straddle upright with a slight forward
    bias — NOT a back lean: the pre-Jul-17 8-12 deg back-lean band put the
    CoM 131-172 mm behind the ankle (6-7x the heel margin) and the hardware
    policy slid to the feasible posture nearest the unreachable band, i.e.
    balancing within a few mm of the heel edge (capture 20260717_213006).

    Args:
        asset_cfg: articulation whose root ``projected_gravity_b`` to read.
            Defaults to the ``"robot"`` scene entity; the IMU sensor's
            ``projected_gravity_b`` was removed in Isaac Lab 3.0 beta2, so we
            read the articulation root (``base_link``) gravity projection
            instead — equivalent for an IMU mounted with identity orientation.
        band_gx_min: back edge of the plateau (most negative ``g_x``; the
            heel side — keep it shallow, the heel margin is tiny).
        band_gx_max: forward edge of the plateau (the toe side, where the
            support margin lives; may be positive).
        edge_std: shared Gaussian shoulder width, used for a side whose
            specific width below is ``None``.
        roll_std: Gaussian width on ``proj_grav[1]`` (lateral tilt).
        forward_penalty_gain: scales ``relu(g_x - forward_deadband)²``.
        forward_deadband: only penalize forward tilt above this ``g_x``;
            ``None`` (default) starts the tail at ``band_gx_max``.
        edge_std_below: Gaussian width on the back (``g_x < band_gx_min``)
            shoulder. Wider = restoring reach further into a backward tip.
        edge_std_above: Gaussian width on the forward (``g_x > band_gx_max``)
            shoulder.
        backward_penalty_gain: scales ``relu(backward_deadband - g_x)²`` —
            the heel-side far-field restoring gradient.
        backward_deadband: only penalize back tilt below this ``g_x``;
            ``None`` (default) starts the tail at ``band_gx_min``.
    """
    asset = env.scene[asset_cfg.name]
    proj_grav = _ensure_tensor(
        asset.data.projected_gravity_b, env_device=getattr(env, "device", None)
    )
    g_x = proj_grav[:, 0]
    g_y = proj_grav[:, 1]

    # Signed overshoot past each band edge: 0 inside
    # [band_gx_min, band_gx_max], positive outside. Only one side is ever
    # nonzero. Flat top, asymmetric Gaussian shoulders.
    below = torch.relu(band_gx_min - g_x)  # back past the heel-side edge
    above = torch.relu(g_x - band_gx_max)  # forward past the toe-side edge
    std_below = edge_std if edge_std_below is None else edge_std_below
    std_above = edge_std if edge_std_above is None else edge_std_above
    pitch_good = torch.exp(
        -torch.square(below) / (std_below * std_below)
        - torch.square(above) / (std_above * std_above)
    )

    roll_good = torch.exp(-torch.square(g_y) / (roll_std * roll_std))

    fwd_db = band_gx_max if forward_deadband is None else forward_deadband
    forward_overshoot = torch.relu(g_x - fwd_db)
    forward_penalty = forward_penalty_gain * forward_overshoot * forward_overshoot

    bwd_db = band_gx_min if backward_deadband is None else backward_deadband
    backward_overshoot = torch.relu(bwd_db - g_x)
    backward_penalty = backward_penalty_gain * backward_overshoot * backward_overshoot

    return pitch_good * roll_good - forward_penalty - backward_penalty


def torso_settle_in_band_l2(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    band_gx_min: float = -0.02,
    band_gx_max: float = 0.05,
    stillness_std: float = 1.5,
) -> torch.Tensor:
    """Penalize SETTLING outside the torso pitch band while holding still.

    Returns ``(dist(g_x, band)² + g_y²) × stillness_gate`` where
    ``dist`` is the signed overshoot outside ``[band_gx_min, band_gx_max]``
    (0 inside) and the gate is ``exp(-Σv² / stillness_std²)`` over all
    joint velocities. Always ``>= 0``; use with a **negative** weight.

    Why this exists on top of ``torso_pitch_asymmetric_reward`` (Jul 18
    2026): the carrot only REWARDS being in the band — nothing punishes
    settling OFF it, and the balance-gated movement penalties
    (joint_vel / position_rate / hip_flexion_anchor) sit at their
    ``gate_floor`` when tilted, so an off-band stance is nearly free of
    movement cost. Capture 20260718_040142 (run 2026-07-18_02-16-59
    ckpt-2000, hardware) showed the exploit: the policy settled steadily
    (g_x std 0.012!) at g_x = -0.24 — 14° back, far off the
    [-0.02, +0.05] band — because the robot's real heel (~81 mm behind
    the ankle, vs 23 mm in the sim STL at the time) makes that stance
    statically survivable, and quiet + symmetric + flat-footed satisfied
    every stillness-gated posture term. With the foot model corrected to
    the real geometry the same settle-off-band optimum exists in sim too
    — this term closes it in both worlds: hold still outside the band and
    you pay, per tick, growing quadratically with the distance.

    STILLNESS-GATED (the ``bilateral_symmetry`` / ``feet_flat`` pattern):
    settling is a POSTURE failure, so the penalty fires at full strength
    whenever the robot is holding any pose — including a tilted one — and
    decays toward 0 only during active motion, so recovery stays free. A
    balance gate would be wrong here: it would clamp to its floor exactly
    in the tilted-settle state this term exists to prevent.

    Roll is included as a plain ``g_y²`` (no band): settling rolled is as
    much a posture failure as settling pitched.

    Non-privileged: reads the articulation root ``projected_gravity_b``
    (the same signal the policy observes via ``mdp.projected_gravity``)
    and joint velocities (encoders).

    Args:
        asset_cfg: articulation whose root ``projected_gravity_b`` to read.
        band_gx_min: back edge of the pitch plateau (most negative ``g_x``).
        band_gx_max: forward edge of the pitch plateau.
        stillness_std: velocity scale for the stillness gate; matches the
            sibling posture terms (``1.5``).
    """
    asset = env.scene[asset_cfg.name]
    device = getattr(env, "device", None)
    proj_grav = _ensure_tensor(asset.data.projected_gravity_b, env_device=device)
    g_x = proj_grav[:, 0]
    g_y = proj_grav[:, 1]

    below = torch.relu(band_gx_min - g_x)
    above = torch.relu(g_x - band_gx_max)
    dist_sq = torch.square(below + above)

    joint_vel = _ensure_tensor(asset.data.joint_vel, env_device=device)
    vel_sq = torch.sum(torch.square(joint_vel), dim=1)
    stillness_gate = torch.exp(-vel_sq / (stillness_std * stillness_std))
    return (dist_sq + torch.square(g_y)) * stillness_gate


def torso_pitch_zero_penalty(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    saturation_pitch_deg: float = 10.0,
) -> torch.Tensor:
    """Penalize ANY torso tilt off vertical (pitch AND roll) — inverted
    Gaussian with a hard cap.

    Returns ``1 - exp(-min(g_x^2 + g_y^2, g_sat^2) / std^2)`` where
    ``g_x = projected_gravity_b[:, 0]`` (``g_x = -sin(pitch)`` in body FLU),
    ``g_y = projected_gravity_b[:, 1]`` (``g_y = sin(roll)``),
    ``g_sat = sin(saturation_pitch_deg)``, and ``std = g_sat / sqrt(3)`` so
    the penalty reaches ``1 - e^-3 ~= 0.95`` at the saturation tilt
    magnitude and the ``min`` clamp holds it EXACTLY there for every larger
    tilt. Always in ``[0, 0.95]``; use with a **negative** weight.

    Shape (default ``saturation_pitch_deg = 10.0``, ``std ~= 0.1003``), where
    the |tilt| column is the magnitude of the combined pitch+roll vector
    (``|g_xy| = sqrt(g_x^2 + g_y^2) = sin(|tilt|)``):

    ============ ====== ====== ====== ====== ======= ========
    |tilt| deg     0      1      2      5      10      >10
    ------------ ------ ------ ------ ------ ------- --------
    penalty        0.00   0.03   0.11   0.53   0.95    0.95
    ============ ====== ====== ====== ====== ======= ========

    0 at exactly upright, quadratic-soft just off zero (the Gaussian's flat
    top: near-zero tilt is nearly free), rising smoothly through the knee,
    and hard-flat at the maximum from ``saturation_pitch_deg`` outward — a
    tipped robot feels constant max pressure with no gradient cliff, while
    the un-clamped Gaussian alone would keep creeping 0.95 -> 1.0 and waste
    dynamic range differentiating 10 deg from 30 deg (both are equally bad:
    recover).

    Symmetric in BOTH pitch and roll (``g_x^2 + g_y^2``): the push task
    shoves laterally too (``PUSH_VELOCITY_RANGE`` x AND y), so a roll lean
    can be settled into just like a pitch lean — pricing the full tilt
    magnitude closes that lateral settle-lean exploit. NOT gated (no
    stillness/balance gate, Jul 19 2026): the penalty is bounded, so it
    prices being tilted without ever blocking a recovery motion — the max
    cost at full saturation is set by the configured weight, not by an
    unbounded quadratic. This distinguishes it from the superseded
    ``torso_settle_in_band_l2`` (stillness-gated, unbounded dist^2 outside
    an asymmetric band): this term always fires, centered exactly at 0 deg
    of tilt in both axes.

    Jul 20 2026 extension: the pre-Jul-20 version read only ``g_x`` (pitch).
    With the push task's lateral shoves (±0.7 m/s in y) the policy could
    settle quietly into a rolled lean that no pitch-only term priced — the
    same settle-off-target exploit this codebase has hit repeatedly
    (capture 20260718_040142 settled at g_x=-0.24). The function name keeps
    the historical ``pitch`` label; the behavior is tilt (pitch + roll).

    Non-privileged: reads the articulation root ``projected_gravity_b`` —
    the same signal the policy observes via ``mdp.projected_gravity`` and
    the firmware ships, so it creates no sim-to-real gap.

    Args:
        asset_cfg: articulation whose root ``projected_gravity_b`` to read.
            Defaults to the ``"robot"`` scene entity (same convention as
            ``torso_pitch_asymmetric_reward``).
        saturation_pitch_deg: tilt magnitude (deg) — combined pitch+roll
            vector — at which the penalty reaches (and clamps at) its max
            of ``1 - e^-3 ~= 0.95``. The param name retains the historical
            ``pitch`` label; it applies to the full tilt magnitude.
    """
    asset = env.scene[asset_cfg.name]
    proj_grav = _ensure_tensor(
        asset.data.projected_gravity_b, env_device=getattr(env, "device", None)
    )
    g_x = proj_grav[:, 0]
    g_y = proj_grav[:, 1]

    g_sat = math.sin(math.radians(saturation_pitch_deg))
    std = g_sat / math.sqrt(3.0)
    tilt_sq = torch.square(g_x) + torch.square(g_y)
    err = torch.minimum(tilt_sq, torch.full_like(tilt_sq, g_sat * g_sat))
    return 1.0 - torch.exp(-err / (std * std))


def feet_flat_orientation_l2(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    foot_body_names: tuple[str, str] = ("foot_left_1", "foot_right_1"),
    sole_normal_b: tuple[float, float, float] = (0.0, 0.0, 1.0),
    stillness_std: float = 1.5,
    balance_gate: bool = False,
    gate_band_gx_center: float = 0.015,
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
