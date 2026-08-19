"""Learned torque-response actuator for the Robstride groups (td-b05f58).

Hybrid actuator-net (paper-derived, arXiv:2607.18135, adapted for Bebop's
variable-impedance MIT action). The paper's ``(pos_err, vel) -> torque``
actuator-net assumes FIXED kp/kd; Bebop's policy modulates gains per tick and
the existing sysid logs are all torque-mode (kp=kd=0), so the gain dimension
cannot be learned from data. We decompose instead:

1. **Analytic PD** — ``desired = kp*(q* - q) + kd*(0 - qd)`` with the LIVE
   policy gains (``bebop_v2_actions.VariableImpedanceJointAction`` routes its
   decoded, slew-limited, EMA-filtered gains into ``self.stiffness`` /
   ``self.damping`` before every physics sub-step; see its issue-#128 note).
2. **Learned response net** — a 3x64 softsign MLP (trained by
   ``tools/actuator_net_fit.py`` on ``sim/bebop-sysid-logs``) maps a short
   history of (desired torque, measured velocity) to the REALIZED
   electromagnetic torque. This captures the torque constant, current
   limiting / saturation, and driver lag that the datasheet-stall
   ``saturation_effort`` badly overestimates (sysid: RS04 cmd 36 Nm ->
   fb ~14 Nm; RS02 cmd 5.1 -> fb ~1.8).
3. **Analytic rail** — the realized torque still passes through
   ``DCMotor._clip_effort`` (speed-torque envelope), so a net output can
   never exceed the physical envelope no matter what it learned.

Friction and armature stay joint-level properties (PhysX applies them after
the actuator effort), so the existing per-episode
``randomize_actuator_params`` DR event is unaffected.

History cadence: ``compute()`` runs every physics step (200 Hz), and the fit
tool resamples the ~125 Hz logs to 200 Hz, so the 5-sample window is 25 ms
on both sides.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING

import torch

from isaaclab.actuators import DCMotor, DCMotorCfg
from isaaclab.utils.assets import read_file
from isaaclab.utils.configclass import configclass
from isaaclab.utils.types import ArticulationActions


class RobstrideResponseActuator(DCMotor):
    """DCMotor whose computed effort comes from a learned response net.

    The PD part (stiffness/damping, live policy gains) is analytic; the
    command->realized-torque mapping is the TorchScript net trained by
    ``tools/actuator_net_fit.py``. Clipping stays analytic (``_clip_effort``).
    """

    cfg: "RobstrideResponseActuatorCfg"

    def __init__(self, cfg: "RobstrideResponseActuatorCfg", *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)

        file_bytes = read_file(self.cfg.network_file)
        self._response_net = torch.jit.load(file_bytes, map_location=self._device).eval()

        # History buffers, index 0 = newest (mirrors ActuatorNetMLP's roll(1)).
        h = self.cfg.history_length
        self._cmd_hist = torch.zeros(self._num_envs, h, self.num_joints, device=self._device)
        self._vel_hist = torch.zeros(self._num_envs, h, self.num_joints, device=self._device)

    """
    Operations.
    """

    def reset(self, env_ids: Sequence[int]):
        # zero the response-net history for the reset environments
        self._cmd_hist[env_ids] = 0.0
        self._vel_hist[env_ids] = 0.0

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        # save current joint vel for the DC-motor envelope clip
        self._joint_vel[:] = joint_vel

        # 1. Analytic MIT PD with live policy gains. Velocity targets are
        # never written by our action term (buffer stays zero), and no
        # feedforward effort is sent, matching real MIT-mode use.
        desired = (
            self.stiffness * (control_action.joint_positions - joint_pos)
            + self.damping * (control_action.joint_velocities - joint_vel)
            + control_action.joint_efforts
        )

        # 2. Push history (newest at index 0) and run the response net.
        self._cmd_hist = self._cmd_hist.roll(1, 1)
        self._cmd_hist[:, 0] = desired
        self._vel_hist = self._vel_hist.roll(1, 1)
        self._vel_hist[:, 0] = joint_vel

        # Per-(env, joint) samples, window dim last: (E*J, H) each, then
        # concat -> (E*J, 2H), matching the fit tool's input layout.
        cmd_in = self._cmd_hist.permute(0, 2, 1).reshape(-1, self.cfg.history_length) / self.cfg.cmd_scale
        vel_in = self._vel_hist.permute(0, 2, 1).reshape(-1, self.cfg.history_length) / self.cfg.vel_scale
        net_in = torch.cat([cmd_in, vel_in], dim=1)
        with torch.inference_mode():
            realized = self._response_net(net_in).view(self._num_envs, self.num_joints)
        self.computed_effort = realized * self.cfg.cmd_scale

        # 3. Analytic rail: DC speed-torque envelope clip.
        self.applied_effort = self._clip_effort(self.computed_effort)

        control_action.joint_efforts = self.applied_effort
        control_action.joint_positions = None
        control_action.joint_velocities = None
        return control_action


@configclass
class RobstrideResponseActuatorCfg(DCMotorCfg):
    """Configuration for :class:`RobstrideResponseActuator`.

    Inherits every physical parameter of ``DCMotorCfg`` (stall/no-load/
    effort rails, sysid friction + armature, seed gains); only the
    command->realized-torque mapping changes.
    """

    class_type: type = RobstrideResponseActuator

    network_file: str = MISSING
    """Path to the TorchScript response net (``tools/actuator_net_fit.py`` output)."""

    history_length: int = 5
    """Samples of (desired torque, velocity) history. 5 @ 200 Hz = 25 ms."""

    cmd_scale: float = MISSING
    """Torque normalization used during the fit (= datasheet stall torque)."""

    vel_scale: float = MISSING
    """Velocity normalization used during the fit (= datasheet no-load speed)."""
