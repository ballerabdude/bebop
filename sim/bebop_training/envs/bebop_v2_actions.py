"""Custom action terms for the Bebop V2 articulation.

This module exists to close three known sim-to-real gaps that the stock
``isaaclab.envs.mdp.JointPositionAction`` ignores:

1. **MIT-mode variable impedance.** The Robstride motors driving Bebop V2
   accept a per-tick 5-tuple ``(position, velocity, torque, kp, kd)`` over
   the CAN bus, and the on-robot policy_runner already forwards whatever
   gains the policy emits straight into ``safe_send_ctrl``. With fixed-gain
   training the policy can only adapt the *target* per tick, not the
   compliance — which is exactly the wrong way around for a legged robot
   (stance leg wants high kp, swing leg wants low kp). We expand the
   action vector to 3 channels per joint:

   - 8 raw position commands  (clipped to [-1, 1], scaled to target offset),
   - 8 raw kp commands        (clipped to [-1, 1], affine-mapped per joint),
   - 8 raw kd commands        (clipped to [-1, 1], affine-mapped per joint).

   The per-joint kp/kd ranges come from
   ``bebop_v2_base_cfg.py::POLICY_KP_RANGES`` / ``POLICY_KD_RANGES`` and
   MUST mirror the ``policy_gain_clamps`` block in
   ``firmware/bebop-linux/config/bebop_v2.yaml``.

2. **Setpoint slew clamp.** The on-robot supervisor caps every PD target
   write at ``max_pos_step_per_tick`` rad per 100 Hz tick (see
   ``firmware/bebop-linux/config/bebop_v2.yaml::defaults.slew`` and the
   clamp in ``firmware/bebop-linux/src/safety/supervisor.rs``
   ``safe_send_ctrl``). The slew clamp lives on the **position channel
   only** — gain channels are instantaneous, which is the whole point of
   variable impedance.

3. **Action / actuation latency.** On the real robot the policy's action
   travels tokio task -> CAN frame -> motor PD loop -> encoder -> CAN
   reply -> next observation, about one control tick of round-trip. We
   model that with a 1-tick delay buffer on the full 24-vec (position +
   gains land on physics together).

Tune the per-joint clamp ranges and the slew cap to match whatever the
firmware ships with. If you change a number on either side, change it on
both — the policy bakes in the achievable control bandwidth.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING, field
from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class VariableImpedanceJointAction(JointPositionAction):
    """MIT-mode action: 8 joint positions + 8 kp + 8 kd per tick.

    Inherits all of :class:`JointPositionAction`'s joint resolution +
    default-offset machinery, but overrides ``action_dim``,
    ``process_actions``, and ``apply_actions`` to handle the 24-dim
    layout and to push kp/kd into PhysX via ``write_joint_stiffness_to_sim``
    / ``write_joint_damping_to_sim`` each tick.

    Action layout (per env, last axis):

    - ``raw[:, 0:N]``  -> position targets (N = num joints, here 8)
    - ``raw[:, N:2N]`` -> kp commands
    - ``raw[:, 2N:3N]``-> kd commands

    All three channels are clipped to ``[-1, 1]`` before scaling. The
    position channel is then ``default + pos_scale * raw_pos`` (so
    ``raw=0`` keeps the joint at its default pose). The gain channels are
    affine-mapped from ``[-1, 1]`` to ``[kp_min, kp_max]`` /
    ``[kd_min, kd_max]`` per joint, so ``raw=0`` lands at the midpoint of
    each range and the policy explores symmetrically around it on day 1.
    """

    cfg: VariableImpedanceJointActionCfg

    def __init__(self, cfg: VariableImpedanceJointActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        if cfg.max_pos_step_per_tick <= 0.0:
            raise ValueError(
                "VariableImpedanceJointActionCfg.max_pos_step_per_tick must be > 0; "
                f"got {cfg.max_pos_step_per_tick}."
            )
        if cfg.action_delay_steps < 0:
            raise ValueError(
                "VariableImpedanceJointActionCfg.action_delay_steps must be >= 0; "
                f"got {cfg.action_delay_steps}."
            )

        n_joints = self._num_joints
        for name, vec in (
            ("kp_min", cfg.kp_min),
            ("kp_max", cfg.kp_max),
            ("kd_min", cfg.kd_min),
            ("kd_max", cfg.kd_max),
        ):
            if len(vec) != n_joints:
                raise ValueError(
                    f"VariableImpedanceJointActionCfg.{name} must have len {n_joints} "
                    f"(one entry per joint in joint_names order); got {len(vec)}."
                )

        device = env.device
        self._kp_min_t = torch.tensor(cfg.kp_min, device=device, dtype=torch.float32)
        self._kp_max_t = torch.tensor(cfg.kp_max, device=device, dtype=torch.float32)
        self._kd_min_t = torch.tensor(cfg.kd_min, device=device, dtype=torch.float32)
        self._kd_max_t = torch.tensor(cfg.kd_max, device=device, dtype=torch.float32)

        if torch.any(self._kp_min_t >= self._kp_max_t):
            raise ValueError(
                "VariableImpedanceJointActionCfg: every kp_min must be < kp_max "
                f"(got kp_min={cfg.kp_min}, kp_max={cfg.kp_max})."
            )
        if torch.any(self._kd_min_t >= self._kd_max_t):
            raise ValueError(
                "VariableImpedanceJointActionCfg: every kd_min must be < kd_max "
                f"(got kd_min={cfg.kd_min}, kd_max={cfg.kd_max})."
            )

        # Slew tracker holds the *last applied* (slewed) position target
        # per env per joint. Lazy-init on first process_actions so the
        # default joint positions are populated.
        self._last_pos_target: torch.Tensor | None = None

        # Action-delay ring buffer: list of length (delay_steps + 1) of
        # tensors shaped (num_envs, 3 * num_joints). Index 0 is the
        # oldest entry. We delay the *decoded* (post-affine, post-slew)
        # full 24-vec so position and gains land on physics together.
        self._delay_len = cfg.action_delay_steps + 1
        self._delay_buffer: list[torch.Tensor] | None = None

    # ------------------------------------------------------------------
    # ActionTerm overrides
    # ------------------------------------------------------------------

    @property
    def action_dim(self) -> int:
        # 3 channels per joint: position, kp, kd.
        return 3 * self._num_joints

    def _default_joint_pos_for_action(self, env_ids: Sequence[int] | slice) -> torch.Tensor:
        d = self._asset.data.default_joint_pos
        if not isinstance(d, torch.Tensor):
            d = wp.to_torch(d)
        return d[env_ids][:, self._joint_ids].clone()

    def _ensure_state(self, num_envs: int, device: torch.device) -> None:
        if self._last_pos_target is None:
            seed_pos = self._default_joint_pos_for_action(slice(None))
            self._last_pos_target = seed_pos

            kp_mid = 0.5 * (self._kp_min_t + self._kp_max_t)
            kd_mid = 0.5 * (self._kd_min_t + self._kd_max_t)
            seed_kp = kp_mid.unsqueeze(0).expand(num_envs, -1).clone()
            seed_kd = kd_mid.unsqueeze(0).expand(num_envs, -1).clone()
            seed_vec = torch.cat([seed_pos, seed_kp, seed_kd], dim=-1)
            self._delay_buffer = [seed_vec.clone() for _ in range(self._delay_len)]

    def process_actions(self, actions: torch.Tensor) -> None:
        # Store the raw 24-dim action for `last_action` observation /
        # action_rate_l2 / action_l2 reward computation. We do this
        # ourselves rather than calling super().process_actions() because
        # the base class assumes a 1-channel (positions-only) layout and
        # would mis-scale our kp / kd channels.
        self._raw_actions[:] = actions

        n = self._num_joints
        num_envs = actions.shape[0]
        self._ensure_state(num_envs, actions.device)
        assert self._last_pos_target is not None
        assert self._delay_buffer is not None

        # Clip every channel to [-1, 1]. Defense-in-depth: rsl_rl's
        # Gaussian head can emit outliers above 1.
        raw = actions.clamp(min=-1.0, max=1.0)
        raw_pos = raw[:, 0:n]
        raw_kp = raw[:, n : 2 * n]
        raw_kd = raw[:, 2 * n : 3 * n]

        # Position: default + pos_scale * raw  (matches the firmware
        # mirror in `observation.rs::decode_policy_action`).
        defaults = self._default_joint_pos_for_action(slice(None))
        pos_target = defaults + self.cfg.pos_scale * raw_pos

        # Affine map raw_{kp,kd} from [-1, 1] to [min, max] per joint.
        kp = self._kp_min_t + 0.5 * (raw_kp + 1.0) * (self._kp_max_t - self._kp_min_t)
        kd = self._kd_min_t + 0.5 * (raw_kd + 1.0) * (self._kd_max_t - self._kd_min_t)

        # Slew clamp on position channel only.
        max_step = self.cfg.max_pos_step_per_tick
        pos_delta = (pos_target - self._last_pos_target).clamp(min=-max_step, max=max_step)
        pos_slewed = self._last_pos_target + pos_delta
        self._last_pos_target = pos_slewed.clone()

        # 1-tick action delay on the full decoded 24-vec.
        full = torch.cat([pos_slewed, kp, kd], dim=-1)
        self._delay_buffer.append(full)
        applied = self._delay_buffer.pop(0)

        # Stash decoded outputs for apply_actions.
        self._processed_actions = applied

        # Variable impedance: write per-env stiffness / damping into the
        # articulation now, before the physics steps inside this tick
        # consume them. We do this in process_actions (called once per
        # tick) rather than apply_actions (called `decimation` times per
        # tick) because gain writes don't change between sub-steps.
        applied_kp = applied[:, n : 2 * n]
        applied_kd = applied[:, 2 * n : 3 * n]
        self._asset.write_joint_stiffness_to_sim(applied_kp, joint_ids=self._joint_ids)
        self._asset.write_joint_damping_to_sim(applied_kd, joint_ids=self._joint_ids)

    def apply_actions(self) -> None:
        # Per-physics-substep position target write. The base class would
        # send the whole self._processed_actions (24-dim) into
        # set_joint_position_target on 8 joints and crash; we slice down
        # to the position channel here.
        n = self._num_joints
        pos = self._processed_actions[:, 0:n]
        self._asset.set_joint_position_target(pos, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)

        if self._last_pos_target is None:
            return
        assert self._delay_buffer is not None

        n = self._num_joints
        kp_mid = 0.5 * (self._kp_min_t + self._kp_max_t)
        kd_mid = 0.5 * (self._kd_min_t + self._kd_max_t)

        if env_ids is None:
            seed_pos = self._default_joint_pos_for_action(slice(None))
            seed_kp = kp_mid.unsqueeze(0).expand(seed_pos.shape[0], -1)
            seed_kd = kd_mid.unsqueeze(0).expand(seed_pos.shape[0], -1)
            self._last_pos_target.copy_(seed_pos)
            for buf in self._delay_buffer:
                buf[:, 0:n] = seed_pos
                buf[:, n : 2 * n] = seed_kp
                buf[:, 2 * n : 3 * n] = seed_kd
        else:
            seed_pos = self._default_joint_pos_for_action(env_ids)
            seed_kp = kp_mid.unsqueeze(0).expand(seed_pos.shape[0], -1)
            seed_kd = kd_mid.unsqueeze(0).expand(seed_pos.shape[0], -1)
            self._last_pos_target[env_ids] = seed_pos
            for buf in self._delay_buffer:
                buf[env_ids, 0:n] = seed_pos
                buf[env_ids, n : 2 * n] = seed_kp
                buf[env_ids, 2 * n : 3 * n] = seed_kd


@configclass
class VariableImpedanceJointActionCfg(JointPositionActionCfg):
    """Cfg for :class:`VariableImpedanceJointAction`.

    Inherits ``joint_names``, ``scale``, ``offset``, ``use_default_offset``,
    ``clip``, ``preserve_order`` from :class:`JointPositionActionCfg` for
    joint resolution. The ``scale`` field is unused (we read ``pos_scale``
    below instead, to keep the position scale and gain scales semantically
    separate).
    """

    class_type: type = VariableImpedanceJointAction

    pos_scale: float = 0.8
    """Position-channel scale: ``target = default + pos_scale * raw_pos``.

    Mirrors what the firmware does in
    ``observation.rs::decode_policy_action`` (``scales::SCALE_ACTION``).
    """

    max_pos_step_per_tick: float = MISSING
    """Maximum |pos_target_now - pos_target_prev| per policy tick, in
    radians. Mirrors ``defaults.slew.max_pos_step_per_tick`` in
    ``firmware/bebop-linux/config/bebop_v2.yaml``."""

    action_delay_steps: int = 0
    """Number of policy ticks of pure transport delay applied to the
    full decoded 24-vec before physics sees it. ``1`` ≈ one CAN
    round-trip @ 100 Hz."""

    kp_min: list[float] = field(default_factory=list)
    """Per-joint lower bound on the decoded kp value, in JOINT_NAMES
    order. Mirrors ``policy_gain_clamps.kp_min`` in the firmware YAML."""

    kp_max: list[float] = field(default_factory=list)
    """Per-joint upper bound on the decoded kp value, in JOINT_NAMES order."""

    kd_min: list[float] = field(default_factory=list)
    """Per-joint lower bound on the decoded kd value, in JOINT_NAMES order."""

    kd_max: list[float] = field(default_factory=list)
    """Per-joint upper bound on the decoded kd value, in JOINT_NAMES order."""
