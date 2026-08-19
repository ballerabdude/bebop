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
   only**; the gain channels instead pass through a first-order low-pass
   (``gain_ema_tau_s``, added Jul 22 2026 — see the cfg field docs).
   Gains were originally left instantaneous ("the whole point of variable
   impedance"), but the round-4 push runs showed the policy's ONLY
   survival strategy at high push force is high-bandwidth kp/kd
   modulation, which the ``gain_rate`` reward penalty then taxed into a
   reward collapse. Filtering the gains in the ACTION SPACE (like the
   position slew clamp does) makes gain-snapping physically impossible,
   so ``gain_rate`` no longer taxes the survival strategy — it survives
   only as an anti-noise governor on the shared log-std (round 5) — and
   fast recovery is pushed onto the (slew-limited) position channels.

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

import math
from collections.abc import Sequence
from dataclasses import MISSING, field
from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def gain_low_pass_ema(prev: torch.Tensor, cmd: torch.Tensor, alpha: float) -> torch.Tensor:
    """First-order low-pass (exponential moving average) on gain commands.

    ``out = prev + alpha * (cmd - prev)`` with ``alpha = 1 - exp(-dt / tau)``,
    i.e. a unity-DC-gain IIR filter with time constant ``tau``. A step
    command reaches ~63% in ``tau`` seconds; a tick-rate (100 Hz)
    square-wave command is attenuated to a steady-state ripple of
    ``alpha / (2 - alpha)`` of its amplitude (~3% for tau=0.15 s at 100 Hz),
    so tick-to-tick gain snapping is physically impossible no matter what
    the policy emits.

    Kept as a module-level pure function so the filter math is unit-testable
    without instantiating the full action term (needs a live env).
    """
    return prev + alpha * (cmd - prev)


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

        # Sim-to-real joint-order guard. The deployed firmware
        # (firmware/bebop-linux/src/observation.rs::JOINT_NAMES) builds the
        # observation and decodes the 24-dim action in EXACTLY
        # ``cfg.joint_names`` order (left-before-right per pair). The Newton
        # articulation, however, resolves joints in RIGHT-before-LEFT pair order,
        # so with ``preserve_order=False`` the policy trains in articulation
        # order and would deploy mirror-swapped (legs L<->R) against firmware.
        # The fix (see ActionsCfg in exp_standing.py) is ``preserve_order=True``
        # on BOTH this action term AND the joint_pos / joint_vel observation
        # terms, which forces all policy I/O into firmware order. This assertion
        # verifies the action side resolved correctly and fails fast if a future
        # USD/URDF re-import or a dropped ``preserve_order`` silently reshuffles
        # the layout (which would balance in sim but ring / fall on hardware).
        resolved_names = list(self._joint_names)
        requested_names = list(cfg.joint_names)
        if resolved_names != requested_names:
            raise ValueError(
                "VariableImpedanceJointAction: the resolved joint order does not "
                "match the requested joint_names order, so the observation/action "
                "layout would be PERMUTED relative to the firmware contract "
                "(observation.rs::JOINT_NAMES).\n"
                f"  requested (firmware order): {requested_names}\n"
                f"  resolved  (actual order):   {resolved_names}\n"
                "Set preserve_order=True on this action term AND on the "
                "joint_pos / joint_vel observation terms (asset_cfg with "
                "joint_names=JOINT_NAMES_ALL, preserve_order=True) so sim obs + "
                "action both match firmware order; or re-export the USD with the "
                "joints in the requested order."
            )

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
        if cfg.gain_ema_tau_s < 0.0:
            raise ValueError(
                "VariableImpedanceJointActionCfg.gain_ema_tau_s must be >= 0 "
                f"(0 disables the filter); got {cfg.gain_ema_tau_s}."
            )

        # Gain low-pass state. ``_gain_ema_alpha`` is derived lazily in
        # ``_ensure_state`` (needs ``env.step_dt``); ``_kp_ema`` / ``_kd_ema``
        # hold the running filtered gains per env and are seeded at the
        # midpoint gains alongside the delay buffer (and re-seeded on
        # episode reset). alpha == 1.0 means the filter is disabled
        # (tau == 0) and the EMA step is skipped entirely.
        self._gain_ema_alpha: float = 1.0
        self._kp_ema: torch.Tensor | None = None
        self._kd_ema: torch.Tensor | None = None

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

        # Fixed-gain mode. When enabled, the policy's 16 kp/kd action channels
        # are IGNORED for physics and a fixed per-joint gain is applied instead
        # (``kp_fixed`` / ``kd_fixed``, defaulting to the per-joint midpoint of
        # [kp_min, kp_max] / [kd_min, kd_max]). The action vector stays 24-dim
        # so the ONNX I/O and the 49-dim observation (last_action is 24) are
        # unchanged from variable-impedance training and firmware; the gain
        # channels are simply inert and get regularized toward 0 by action_l2.
        #
        # The midpoint default is the value firmware decodes at raw_kp = raw_kd
        # = 0 (decode_policy_action affine-maps [-1, 1] -> [min, max]), so a
        # policy whose gain channels are driven to ~0 deploys with the SAME
        # gains trained here without any firmware change.
        self._freeze_gains = bool(cfg.freeze_gains)
        if self._freeze_gains:
            kp_fixed = list(cfg.kp_fixed) if cfg.kp_fixed else [
                0.5 * (lo + hi) for lo, hi in zip(cfg.kp_min, cfg.kp_max)
            ]
            kd_fixed = list(cfg.kd_fixed) if cfg.kd_fixed else [
                0.5 * (lo + hi) for lo, hi in zip(cfg.kd_min, cfg.kd_max)
            ]
            for name, vec in (("kp_fixed", kp_fixed), ("kd_fixed", kd_fixed)):
                if len(vec) != n_joints:
                    raise ValueError(
                        f"VariableImpedanceJointActionCfg.{name} must have len "
                        f"{n_joints} (one entry per joint in joint_names order); "
                        f"got {len(vec)}."
                    )
            self._kp_fixed_t = torch.tensor(kp_fixed, device=device, dtype=torch.float32)
            self._kd_fixed_t = torch.tensor(kd_fixed, device=device, dtype=torch.float32)
        else:
            self._kp_fixed_t = None
            self._kd_fixed_t = None

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

        # Action-delay buffer. Supports a FIXED delay (``action_delay_steps``)
        # or a per-episode RANDOMIZED delay (``action_delay_range = (min, max)``).
        # Randomizing the transport delay the policy trains against is the key
        # robustness knob against latency-induced limit cycles on hardware: a
        # high-gain stand that is stable at exactly one delay can ring when the
        # real CAN/motor round-trip differs. The buffer is always sized to the
        # MAX delay; each tick we gather the per-env-delayed frame. We delay the
        # *decoded* (post-affine, post-slew) full 24-vec so position and gains
        # land on physics together.
        if cfg.action_delay_range is not None:
            d_min, d_max = int(cfg.action_delay_range[0]), int(cfg.action_delay_range[1])
            if d_min < 0 or d_max < d_min:
                raise ValueError(
                    "VariableImpedanceJointActionCfg.action_delay_range must be "
                    f"(min, max) with 0 <= min <= max; got {cfg.action_delay_range}."
                )
            self._delay_min, self._delay_max = d_min, d_max
        else:
            self._delay_min = self._delay_max = cfg.action_delay_steps
        self._randomize_delay = self._delay_min != self._delay_max
        self._delay_len = self._delay_max + 1
        self._delay_steps_per_env: torch.Tensor | None = None
        self._env_arange: torch.Tensor | None = None
        self._delay_buffer: list[torch.Tensor] | None = None

        # Map each Articulation actuator to the column indices in our
        # action's 8-joint vector. Built once at init.
        #
        # Background: ``write_joint_stiffness_to_sim`` / ``..._damping_to_sim``
        # update PhysX joint drives (which ``ImplicitActuator`` reads) and
        # ``articulation.data._joint_stiffness`` / ``_joint_damping`` —
        # but NOT the per-actuator ``actuator.stiffness`` /
        # ``actuator.damping`` tensors that the EXPLICIT actuators
        # (``IdealPDActuator``, ``DCMotor``, ``DelayedPDActuator``) read
        # in their ``compute()`` PD calculation. See Isaac Lab issue
        # #128 (the same caveat is noted inline in the physx backend
        # at ``write_joint_damping_to_sim_index``).
        #
        # For ``ImplicitActuator`` this loop is harmless — the implicit
        # actuator's ``compute()`` is a no-op and never reads
        # ``self.stiffness`` for control purposes. For any explicit
        # actuator (which is what you want for sim-to-real fidelity)
        # this loop is the difference between "policy controls the
        # gains" and "policy commands are silently ignored, actuator
        # forever uses the seeded midpoints".
        our_joint_id_list = self._joint_ids
        if isinstance(our_joint_id_list, slice):
            our_joint_id_list = list(
                range(*our_joint_id_list.indices(self._asset.num_joints))
            )
        elif hasattr(our_joint_id_list, "cpu"):
            our_joint_id_list = our_joint_id_list.cpu().tolist()
        else:
            our_joint_id_list = list(our_joint_id_list)

        # List of (actuator_obj, indices_into_action_vec_in_actuator_joint_order)
        self._actuator_gain_routes: list[tuple] = []
        for _name, actuator in self._asset.actuators.items():
            actuator_joint_ids = actuator.joint_indices
            if isinstance(actuator_joint_ids, slice):
                actuator_joint_ids = list(
                    range(*actuator_joint_ids.indices(self._asset.num_joints))
                )
            elif hasattr(actuator_joint_ids, "cpu"):
                actuator_joint_ids = actuator_joint_ids.cpu().tolist()
            else:
                actuator_joint_ids = list(actuator_joint_ids)
            try:
                indices_in_our_vec = [our_joint_id_list.index(j) for j in actuator_joint_ids]
            except ValueError:
                # Actuator owns joints outside this action term's set —
                # skip, gains for those joints aren't ours to manage.
                continue
            self._actuator_gain_routes.append((actuator, indices_in_our_vec))

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

    def _current_joint_pos_for_action(self, env_ids: Sequence[int] | slice) -> torch.Tensor:
        """Current joint positions for the joints this action term controls.

        Used to seed the slew tracker / delay buffer at reset, so the action
        term starts from where the joints actually are (i.e. the random
        spawn pose written by ``mdp.reset_joints_by_offset``) instead of
        snapping the target back to the default pose on tick 1. Without
        this, the high-stiffness actuator yanks every joint back to zero
        within ~50 ms after reset and the policy never experiences the
        randomized initial conditions.
        """
        p = self._asset.data.joint_pos
        if not isinstance(p, torch.Tensor):
            p = wp.to_torch(p)
        return p[env_ids][:, self._joint_ids].clone()

    def _ensure_state(self, num_envs: int, device: torch.device) -> None:
        if self._last_pos_target is None:
            # Match the per-reset seeding logic: start the slew tracker
            # and delay buffer at the *current* joint pose. By the first
            # process_actions call Isaac Lab has already fired the
            # bootstrap reset (including our random-offset reset events),
            # so reading default_joint_pos would silently snap targets
            # back to 0 on tick 1.
            seed_pos = self._current_joint_pos_for_action(slice(None))
            self._last_pos_target = seed_pos

            kp_mid = 0.5 * (self._kp_min_t + self._kp_max_t)
            kd_mid = 0.5 * (self._kd_min_t + self._kd_max_t)
            seed_kp = kp_mid.unsqueeze(0).expand(num_envs, -1).clone()
            seed_kd = kd_mid.unsqueeze(0).expand(num_envs, -1).clone()
            seed_vec = torch.cat([seed_pos, seed_kp, seed_kd], dim=-1)
            self._delay_buffer = [seed_vec.clone() for _ in range(self._delay_len)]
            self._env_arange = torch.arange(num_envs, device=device)
            self._delay_steps_per_env = self._sample_delays(num_envs, device)

            # Gain low-pass: derive the per-tick EMA coefficient from the
            # control dt and seed the filter state at the same midpoint
            # gains the delay buffer starts with, so the applied gains
            # begin exactly at mid with no filter spin-up transient.
            tau = float(self.cfg.gain_ema_tau_s)
            if tau > 0.0:
                dt = float(self._env.step_dt)
                self._gain_ema_alpha = 1.0 - math.exp(-dt / tau)
                self._kp_ema = seed_kp.clone()
                self._kd_ema = seed_kd.clone()
            else:
                self._gain_ema_alpha = 1.0
                self._kp_ema = None
                self._kd_ema = None

    def _sample_delays(self, num: int, device: torch.device) -> torch.Tensor:
        """Per-env action-delay (in ticks). Uniform in [min, max] when
        randomizing, else a constant ``max`` (the fixed-delay behavior)."""
        if self._randomize_delay:
            return torch.randint(
                self._delay_min, self._delay_max + 1, (num,), device=device, dtype=torch.long
            )
        return torch.full((num,), self._delay_max, dtype=torch.long, device=device)

    def process_actions(self, actions: torch.Tensor) -> None:
        n = self._num_joints
        num_envs = actions.shape[0]
        self._ensure_state(num_envs, actions.device)
        assert self._last_pos_target is not None
        assert self._delay_buffer is not None

        # Clip every channel to [-1, 1] BEFORE storing as the "raw"
        # action. The deployed firmware clamps the policy output to
        # [-1, 1] in `POLICY_*` clamps inside `bebop_v2.yaml`, so this
        # is the action the real robot would see anyway.
        #
        # Also a critical training-stability fix: the stored
        # ``_raw_actions`` feeds three downstream consumers that all
        # need bounded inputs to avoid divergence:
        #   (1) ``last_action`` observation — gets fed back into the
        #       actor / critic next tick. With unclipped values, a
        #       single env where the Gaussian head sampled a tail
        #       outlier (mean=50, std=1) plants 50 into the next obs,
        #       which matmuls into ever-larger network outputs, until
        #       within ~10 iters the critic's value head overflows
        #       float32 (``Mean value loss: inf`` → NaN gradients →
        #       NaN in ``log_std`` → ``Normal(mean, NaN)`` crashes
        #       with "normal expects all elements of std >= 0.0").
        #   (2) ``mdp.action_l2`` / ``mdp.action_rate_l2`` rewards —
        #       quadratic in the raw action; an unclipped 1e6 outlier
        #       becomes a −10¹¹ reward, becomes a value target,
        #       trains the critic into ruin.
        #   (3) action-delay ring buffer (below) — propagates the same
        #       raw scale into the slew tracker on the next tick.
        # All three are bounded once ``_raw_actions`` is clipped.
        raw = actions.clamp(min=-1.0, max=1.0)
        self._raw_actions[:] = raw
        raw_pos = raw[:, 0:n]
        raw_kp = raw[:, n : 2 * n]
        raw_kd = raw[:, 2 * n : 3 * n]

        # Position: default + pos_scale * raw  (matches the firmware
        # mirror in `observation.rs::decode_policy_action`).
        defaults = self._default_joint_pos_for_action(slice(None))
        pos_target = defaults + self.cfg.pos_scale * raw_pos

        # Gains. In variable-impedance mode (default) affine-map raw_{kp,kd}
        # from [-1, 1] to [min, max] per joint. In fixed-gain mode the raw
        # kp/kd channels are ignored (left only as inert, action_l2-regularized
        # outputs) and a constant per-joint gain is broadcast to every env.
        if self._freeze_gains:
            kp = self._kp_fixed_t.unsqueeze(0).expand(num_envs, -1)
            kd = self._kd_fixed_t.unsqueeze(0).expand(num_envs, -1)
        else:
            kp = self._kp_min_t + 0.5 * (raw_kp + 1.0) * (self._kp_max_t - self._kp_min_t)
            kd = self._kd_min_t + 0.5 * (raw_kd + 1.0) * (self._kd_max_t - self._kd_min_t)

        # Slew clamp on position channel only.
        max_step = self.cfg.max_pos_step_per_tick
        pos_delta = (pos_target - self._last_pos_target).clamp(min=-max_step, max=max_step)
        pos_slewed = self._last_pos_target + pos_delta
        self._last_pos_target = pos_slewed.clone()

        # Action delay on the full decoded 24-vec. Keep the buffer at a fixed
        # length (max_delay + 1): index 0 is the oldest (= max delay), the last
        # entry is the freshest (delay 0). With a randomized delay we gather the
        # per-env-delayed frame; with a fixed delay we just take the oldest.
        full = torch.cat([pos_slewed, kp, kd], dim=-1)
        self._delay_buffer.append(full)
        if len(self._delay_buffer) > self._delay_len:
            self._delay_buffer.pop(0)
        if self._randomize_delay:
            stack = torch.stack(self._delay_buffer, dim=0)  # (L, num_envs, 3n)
            sel = (stack.shape[0] - 1 - self._delay_steps_per_env).clamp_(min=0)
            applied = stack[sel, self._env_arange]
        else:
            applied = self._delay_buffer[0]

        # Gain low-pass on the DELAYED gains, so the filter smooths the
        # exact sequence physics sees (including per-env delay jumps).
        # This is the hard anti-snap guarantee that freed the ``gain_rate``
        # reward penalty from taxing the survival strategy (round 4-5):
        # whatever the policy emits, the applied kp/kd can move no faster
        # than the cfg.gain_ema_tau_s EMA allows. ``applied`` is never
        # mutated in place — with the fixed-delay path it ALIASES
        # ``self._delay_buffer[0]`` — the torch.cat below builds a fresh
        # tensor instead.
        if self._gain_ema_alpha < 1.0:
            assert self._kp_ema is not None and self._kd_ema is not None
            kp_filt = gain_low_pass_ema(self._kp_ema, applied[:, n : 2 * n], self._gain_ema_alpha)
            kd_filt = gain_low_pass_ema(self._kd_ema, applied[:, 2 * n : 3 * n], self._gain_ema_alpha)
            self._kp_ema = kp_filt
            self._kd_ema = kd_filt
            applied = torch.cat([applied[:, 0:n], kp_filt, kd_filt], dim=-1)

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

        # Also propagate the live gains into each actuator's per-instance
        # stiffness / damping tensors. This is required for explicit
        # actuators (DCMotor / IdealPDActuator / DelayedPDActuator) —
        # their PD compute() reads from these tensors, not from the
        # articulation's joint_stiffness buffer. See the issue-#128
        # explanation in __init__.
        for actuator, indices in self._actuator_gain_routes:
            actuator.stiffness[:] = applied_kp[:, indices]
            actuator.damping[:] = applied_kd[:, indices]

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

        # Seed the slew tracker and delay buffer from the *current* joint
        # positions, not the defaults. By the time the action manager
        # calls our reset(), Isaac Lab has already fired the reset events
        # (mode="reset") and the random offsets from
        # ``mdp.reset_joints_by_offset`` are visible in
        # ``self._asset.data.joint_pos``. Seeding from defaults instead
        # would put ``_last_pos_target`` at 0 while the joints are at
        # e.g. knee_flexion=+0.5 rad, so the very first tick would yank
        # everything back to 0 with full stiffness and erase the spawn
        # pose before render frame 2.
        if env_ids is None:
            seed_pos = self._current_joint_pos_for_action(slice(None))
            seed_kp = kp_mid.unsqueeze(0).expand(seed_pos.shape[0], -1)
            seed_kd = kd_mid.unsqueeze(0).expand(seed_pos.shape[0], -1)
            self._last_pos_target.copy_(seed_pos)
            for buf in self._delay_buffer:
                buf[:, 0:n] = seed_pos
                buf[:, n : 2 * n] = seed_kp
                buf[:, 2 * n : 3 * n] = seed_kd
        else:
            seed_pos = self._current_joint_pos_for_action(env_ids)
            seed_kp = kp_mid.unsqueeze(0).expand(seed_pos.shape[0], -1)
            seed_kd = kd_mid.unsqueeze(0).expand(seed_pos.shape[0], -1)
            self._last_pos_target[env_ids] = seed_pos
            for buf in self._delay_buffer:
                buf[env_ids, 0:n] = seed_pos
                buf[env_ids, n : 2 * n] = seed_kp
                buf[env_ids, 2 * n : 3 * n] = seed_kd

        # Re-sample the per-episode action delay for the reset envs so each
        # episode trains against a fresh latency draw.
        if self._randomize_delay and self._delay_steps_per_env is not None:
            dev = self._delay_steps_per_env.device
            if env_ids is None:
                self._delay_steps_per_env.copy_(
                    self._sample_delays(self._delay_steps_per_env.shape[0], dev)
                )
            else:
                ids = (
                    env_ids
                    if isinstance(env_ids, torch.Tensor)
                    else torch.as_tensor(env_ids, device=dev, dtype=torch.long)
                )
                self._delay_steps_per_env[ids] = self._sample_delays(int(ids.numel()), dev)

        # Re-seed the gain low-pass state at the midpoint gains for the
        # reset envs, matching the delay-buffer re-seed above: the applied
        # gains right after reset ARE mid, so the filter must start there
        # too (otherwise it would falsely ramp from a stale episode-end
        # value and yank the gains on the first ticks of the new episode).
        if self._kp_ema is not None:
            assert self._kd_ema is not None
            if env_ids is None:
                self._kp_ema[:] = kp_mid
                self._kd_ema[:] = kd_mid
            else:
                self._kp_ema[env_ids] = kp_mid
                self._kd_ema[env_ids] = kd_mid


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
    round-trip @ 100 Hz. Ignored when ``action_delay_range`` is set."""

    action_delay_range: tuple[int, int] | None = None
    """Optional ``(min, max)`` per-episode randomized transport delay, in
    policy ticks. When set it OVERRIDES the fixed ``action_delay_steps``: each
    env samples a delay uniformly in ``[min, max]`` on reset and the decoded
    24-vec is delayed by that many ticks before physics. Randomizing latency is
    the main robustness knob against latency-induced limit cycles on hardware (a
    stand tuned for one exact delay can ring when the real CAN/motor round-trip
    differs). ``None`` keeps the fixed ``action_delay_steps`` behavior."""

    kp_min: list[float] = field(default_factory=list)
    """Per-joint lower bound on the decoded kp value, in JOINT_NAMES
    order. Mirrors ``policy_gain_clamps.kp_min`` in the firmware YAML."""

    kp_max: list[float] = field(default_factory=list)
    """Per-joint upper bound on the decoded kp value, in JOINT_NAMES order."""

    kd_min: list[float] = field(default_factory=list)
    """Per-joint lower bound on the decoded kd value, in JOINT_NAMES order."""

    kd_max: list[float] = field(default_factory=list)
    """Per-joint upper bound on the decoded kd value, in JOINT_NAMES order."""

    freeze_gains: bool = False
    """If True, ignore the policy's 16 kp/kd action channels for physics and
    apply the fixed per-joint gains ``kp_fixed`` / ``kd_fixed`` instead.

    The action vector stays 24-dim (so the ONNX I/O and the 49-dim observation
    that includes the 24-wide ``last_action`` are unchanged from the
    variable-impedance config and from firmware); the gain channels are simply
    inert. Use this to learn a clean position-only quiet stand first, then turn
    it off to re-introduce variable impedance once the robot stands on hardware.
    """

    kp_fixed: list[float] = field(default_factory=list)
    """Per-joint kp applied when ``freeze_gains`` is True, in JOINT_NAMES order.
    Empty (default) -> midpoint of [kp_min, kp_max] per joint, which is what the
    firmware decodes at raw_kp = 0, so a policy whose gain channels are
    regularized toward 0 deploys with these exact gains with no firmware change.
    """

    kd_fixed: list[float] = field(default_factory=list)
    """Per-joint kd applied when ``freeze_gains`` is True, in JOINT_NAMES order.
    Empty (default) -> midpoint of [kd_min, kd_max] per joint (firmware raw_kd =
    0 decode)."""

    gain_ema_tau_s: float = 0.15
    """First-order low-pass time constant (seconds) on the decoded kp/kd
    channels, applied each policy tick AFTER the action-delay selection.
    0 disables the filter (gains instantaneous, the pre-Jul-22-2026
    behavior).

    Why this exists (post-mortem of push runs 2026-07-22_10-55-08 and
    2026-07-22_12-29-28): with instantaneous gains, the policy's ONLY
    survival strategy once the push curriculum passed ~±0.45 m/s was
    high-bandwidth kp/kd modulation — and the ``gain_rate`` reward
    penalty taxed that strategy quadratically, harder the better the
    policy balanced (reward peaked +27.7 at 30% of training, then
    collapsed to -21 while eplen was still rising; both runs degraded to
    eplen ~600-900). Enforcing smoothness as a REWARD TAX makes survival
    unaffordable; enforcing it in the ACTION SPACE (exactly like the
    position slew clamp) makes it free. With tau=0.15 s at the 100 Hz
    control rate, tick-to-tick gain snapping is attenuated ~30x
    (steady-state ripple ~3% of command amplitude) while impedance
    shifts slower than ~1-2 Hz pass through — the policy keeps slow
    stiffness scheduling (soften at rest, stiffen for recovery) and fast
    recovery moves are pushed onto the slew-limited position channels,
    i.e. the stepping/crouching behavior we actually want. Note: the
    ``gain_rate`` reward term was briefly deleted with this change
    (round 4) and RE-ADDED the same day (round 5) in a new role — with
    the filter decoupling gain noise from physics, an ungoverned shared
    log-std exploded to 4.7e15 (run 2026-07-22_20-49-06); the -2.0 tax
    on the RAW gain-channel rate now serves purely as an anti-noise
    std governor (slow legitimate ramps cost ~-0.06/step).

    Round 6 (Jul 22 2026): tau relaxed 0.15 -> 0.08 in the standing
    ActionsCfg after four configs hit the same push wall at ~±0.4-0.5
    m/s — recovery needs impedance bandwidth the 0.15 s filter (1 Hz)
    couldn't pass. At 0.08 the tick-rate ripple rises to ~6.5% (still
    ~15x snap attenuation, far inside the hardware envelope) and the
    -3 dB bandwidth doubles to ~2 Hz (stiffen in ~0.08 s).

    DEPLOYMENT CONTRACT: the firmware MUST apply the same EMA to the
    decoded kp/kd in ``observation.rs::decode_policy_action`` (alpha =
    1 - exp(-dt / 0.15)) before writing gains to the motors, or the
    deployed gains will move far faster than anything the policy
    trained against.
    """
