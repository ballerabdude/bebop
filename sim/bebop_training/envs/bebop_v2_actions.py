"""Custom action terms for the Bebop V2 articulation.

This module exists to **close two known sim-to-real gaps** that the stock
``isaaclab.envs.mdp.JointPositionAction`` ignores:

1. **Setpoint slew clamp.** The on-robot supervisor caps every PD target
   write at ``max_pos_step_per_tick`` rad per 100 Hz tick (see
   ``firmware/bebop-linux/config/bebop_v2.yaml::defaults.slew`` and the
   clamp in ``firmware/bebop-linux/src/safety/supervisor.rs``
   ``safe_send_ctrl``). A policy trained against the stock
   ``JointPositionAction`` can produce step targets that are *physically
   unrealizable* on the real robot — the firmware quietly slews them at
   ~0.5 rad/s, which makes the resulting torque <1% of what the sim
   developed in the same control window. The robot then "feels weightless"
   in sim and "feels limp" on hardware.
2. **Action / actuation latency.** On the real robot the policy's action
   travels: tokio task → CAN frame → motor PD loop → encoder → CAN reply
   → next observation. That round-trip is ~1 control tick (10 ms @
   100 Hz). Without it, the sim policy learns a feedback loop tighter
   than the real motors can close.

We model both by subclassing :class:`isaaclab.envs.mdp.JointPositionAction`
and processing the action *once per env step* (i.e. per policy tick), not
per physics sub-step:

- ``apply_actions`` is called ``decimation`` times per policy tick. We do
  the slew + delay work inside ``process_actions`` (called once per tick)
  and then let ``apply_actions`` keep re-asserting the same setpoint each
  physics step, mirroring the firmware's behaviour where the same target
  is held between CAN sends.
- The slew tracker (``_last_target``) is per-env / per-joint and resets
  to the current joint position on env reset, exactly like the firmware's
  ``LockedMotorState::last_target_pos`` is locked to the current pose
  when a joint is armed.
- The delay buffer holds the last ``action_delay_steps + 1`` *slewed*
  targets; the oldest entry is what physics actually applies. A delay
  of ``1`` step ≈ one CAN frame (~10 ms) of latency.

Tune ``max_pos_step_per_tick`` and ``action_delay_steps`` to match
whatever the firmware ships with at deploy time. If you change the
firmware values you must retrain — the policy bakes in the achievable
control bandwidth.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class SlewLimitedJointPositionAction(JointPositionAction):
    """Joint position action with a per-tick slew cap and N-tick action delay.

    See module docstring for the rationale. Drop-in replacement for
    :class:`isaaclab.envs.mdp.JointPositionAction` — it still accepts the
    same scale / offset / use_default_offset / clip parameters, and adds
    two more on the cfg:

    * ``max_pos_step_per_tick`` — rad per policy tick. Mirrors
      ``firmware/bebop-linux/config/bebop_v2.yaml::defaults.slew.max_pos_step_per_tick``.
    * ``action_delay_steps`` — integer number of policy ticks of pure
      transport delay between the policy emitting an action and physics
      applying it. ``0`` disables the delay (still keeps the slew clamp).
    """

    cfg: SlewLimitedJointPositionActionCfg

    def __init__(self, cfg: SlewLimitedJointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        if cfg.max_pos_step_per_tick <= 0.0:
            raise ValueError(
                "SlewLimitedJointPositionActionCfg.max_pos_step_per_tick must be > 0; "
                f"got {cfg.max_pos_step_per_tick}. Use the stock JointPositionActionCfg "
                "if you want unbounded slew."
            )
        if cfg.action_delay_steps < 0:
            raise ValueError(
                "SlewLimitedJointPositionActionCfg.action_delay_steps must be >= 0; "
                f"got {cfg.action_delay_steps}."
            )

        # Slew tracker: holds the *last applied* (slewed) target per env per
        # action joint. Lazy-init on first process_actions call so we can
        # seed it from the actual joint positions after the very first
        # reset (default_joint_pos may not be populated at __init__).
        self._last_target: torch.Tensor | None = None

        # Action-delay ring buffer: list of length (delay_steps + 1) of
        # tensors shaped (num_envs, action_dim). Index 0 is the oldest
        # entry — that's what gets sent to physics. New entries are
        # appended at the end after popping index 0.
        self._delay_len = cfg.action_delay_steps + 1
        self._delay_buffer: list[torch.Tensor] | None = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _default_joint_pos_for_action(self, env_ids: Sequence[int] | slice) -> torch.Tensor:
        """Return ``default_joint_pos[env_ids, joint_ids]`` as a torch tensor.

        Handles both ``Articulation.data.default_joint_pos`` flavours:
        modern Isaac Lab returns a warp array, older / patched builds
        return a torch tensor directly.
        """
        d = self._asset.data.default_joint_pos
        if not isinstance(d, torch.Tensor):
            d = wp.to_torch(d)
        return d[env_ids][:, self._joint_ids].clone()

    def _ensure_state(self) -> None:
        """Lazy-init slew tracker + delay buffer on first use."""
        if self._last_target is None:
            seed = self._default_joint_pos_for_action(slice(None))
            self._last_target = seed
            self._delay_buffer = [seed.clone() for _ in range(self._delay_len)]

    # ------------------------------------------------------------------
    # ActionTerm overrides
    # ------------------------------------------------------------------

    def process_actions(self, actions: torch.Tensor) -> None:
        # Run the base class (handles raw store, scale + offset, optional clip).
        # After this call self._processed_actions = scale * raw + offset
        # and is the *intended* per-tick target the policy is asking for.
        super().process_actions(actions)

        self._ensure_state()
        assert self._last_target is not None  # for type checker
        assert self._delay_buffer is not None

        # 1) Slew clamp: cap |target - last_target| at max_pos_step_per_tick.
        #    Matches `firmware/bebop-linux/src/safety/supervisor.rs`
        #    `safe_send_ctrl`'s clamp around `g.last_target_pos`.
        max_step = self.cfg.max_pos_step_per_tick
        delta = self._processed_actions - self._last_target
        delta = torch.clamp(delta, min=-max_step, max=max_step)
        slewed = self._last_target + delta

        # Update tracker with the *new* committed slewed target. The
        # firmware updates `last_target_pos` regardless of how much later
        # the motor actually reaches it; we do the same.
        self._last_target = slewed.clone()

        # 2) Action delay: push the slewed target onto the back of the
        #    ring buffer, pop the oldest as the value physics will apply
        #    this tick. With delay_steps=0 the buffer is length 1 and
        #    this is a no-op (the value we just pushed is what we pop).
        self._delay_buffer.append(slewed)
        applied = self._delay_buffer.pop(0)

        # `processed_actions` is what `JointPositionAction.apply_actions`
        # writes to the articulation. Re-point it at the delayed value.
        self._processed_actions = applied

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        # Clear raw_actions for the reset envs (base class behaviour).
        super().reset(env_ids)

        if self._last_target is None:
            return

        assert self._delay_buffer is not None

        if env_ids is None:
            seed = self._default_joint_pos_for_action(slice(None))
            self._last_target.copy_(seed)
            for buf in self._delay_buffer:
                buf.copy_(seed)
        else:
            seed = self._default_joint_pos_for_action(env_ids)
            self._last_target[env_ids] = seed
            for buf in self._delay_buffer:
                buf[env_ids] = seed


@configclass
class SlewLimitedJointPositionActionCfg(JointPositionActionCfg):
    """Cfg for :class:`SlewLimitedJointPositionAction`.

    Inherits all of :class:`JointPositionActionCfg` (``joint_names``,
    ``scale``, ``offset``, ``use_default_offset``, ``clip``,
    ``preserve_order``) and adds the firmware-matched slew + delay
    parameters.
    """

    class_type: type = SlewLimitedJointPositionAction

    max_pos_step_per_tick: float = MISSING
    """Maximum |target_now - target_prev| per policy tick, in radians.

    Mirrors ``defaults.slew.max_pos_step_per_tick`` in
    ``firmware/bebop-linux/config/bebop_v2.yaml``. At 100 Hz a value of
    ``0.005`` corresponds to a 0.5 rad/s setpoint slew rate."""

    action_delay_steps: int = 0
    """Number of policy ticks of pure transport delay applied to the
    *slewed* target before physics sees it.

    A value of ``1`` approximates one CAN round-trip (~10 ms @ 100 Hz);
    set to ``0`` to disable delay while keeping the slew clamp."""
