"""Robstride RS01/RS02/RS03/RS04 CAN protocol — pure Python.

Ported from the Rust ``firmware/bebop-linux/src/{robstride,can_interface}.rs``.

No ROS dependencies; only ``struct`` and ``math`` from the standard library.
The CAN transport (python-can / socketcan) is injected as plain (id, data)
tuples so this module is unit-testable without hardware.

Terminology:
    motor_id          — 8-bit Robstride node ID, set in the Robstride
                        debugger per motor (e.g. 31, 41, 32, ...).
    frame_id          — full 29-bit extended CAN identifier sent on the
                        wire, built from cmd_type + data_area2 + motor_id.

Wire format (Robstride MIT-mode, CAN 2.0 Extended, 1 Mbps):

    29-bit frame_id (TX, host -> motor)
        bits 28..24 : cmd_type        (5 bits)
        bits 23.. 8 : data area 2     (16 bits, e.g. torque feedforward)
        bits  7.. 0 : motor_id        (8 bits)

    8-byte payload for cmd_type = MOTOR_CTRL (0x01), big-endian:
        [0..2)  position_raw   (uint16, -4pi..+4pi)
        [2..4)  velocity_raw   (uint16, model-specific)
        [4..6)  kp_raw         (uint16, 0..5000)
        [6..8)  kd_raw         (uint16, 0..100)

    29-bit frame_id (RX, motor -> host, cmd_type = 0x02)
        bits 28..24 : cmd_type = 0x02
        bits 23..22 : mode_status
        bits 21..16 : fault_bits
        bits 15.. 8 : motor_id  (which motor responded)
        bits  7.. 0 : host_id   (0xFD)

    8-byte payload, big-endian:
        [0..2)  position_raw   (uint16, -4pi..+4pi)
        [2..4)  velocity_raw   (uint16, model-specific)
        [4..6)  torque_raw     (uint16, model-specific)
        [6..8)  temperature_raw (uint16, °C * 10)
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, Optional, Tuple


# ----------------------------------------------------------------- protocol
HOST_ID: int = 0xFD

P_MIN: float = -4.0 * math.pi  # -12.566...
P_MAX: float = +4.0 * math.pi
KP_MIN: float = 0.0
KP_MAX: float = 5000.0
KD_MIN: float = 0.0
KD_MAX: float = 100.0


class CmdType(IntEnum):
    GET_ID = 0x00
    MOTOR_CTRL = 0x01
    FEEDBACK = 0x02
    ENABLE = 0x03
    STOP = 0x04
    SET_ZERO = 0x06
    PARAM_READ = 0x11
    PARAM_WRITE = 0x12
    FAULT_FEEDBACK = 0x15
    ACTIVE_REPORT = 0x18


# ----------------------------------------------------------------- model specs
@dataclass(frozen=True)
class RobstrideSpecs:
    """Per-model torque/velocity ranges used for uint16 <-> float scaling."""
    name: str
    torque_min: float
    torque_max: float
    velocity_min: float
    velocity_max: float


SPECS: Dict[str, RobstrideSpecs] = {
    "RS01": RobstrideSpecs("RS01", -12.0, 12.0, -45.0, 45.0),
    "RS02": RobstrideSpecs("RS02", -25.0, 25.0, -30.0, 30.0),
    "RS03": RobstrideSpecs("RS03", -60.0, 60.0, -20.0, 20.0),
    "RS04": RobstrideSpecs("RS04", -120.0, 120.0, -15.0, 15.0),
}


def get_specs(model: str) -> RobstrideSpecs:
    try:
        return SPECS[model.upper()]
    except KeyError as exc:
        raise ValueError(
            f"Unknown Robstride model {model!r}. Expected one of {list(SPECS)}."
        ) from exc


# ----------------------------------------------------------------- helpers
def _f_to_u16(value: float, lo: float, hi: float) -> int:
    if hi <= lo:
        raise ValueError(f"Invalid range [{lo}, {hi}]")
    clamped = min(max(value, lo), hi)
    return int(round((clamped - lo) / (hi - lo) * 65535.0))


def _u16_to_f(raw: int, lo: float, hi: float) -> float:
    return lo + (raw / 65535.0) * (hi - lo)


# ----------------------------------------------------------------- frame builders
def _make_frame_id(cmd_type: int, data_area2: int, motor_id: int) -> int:
    if not (0 <= cmd_type <= 0x1F):
        raise ValueError(f"cmd_type out of range: {cmd_type:#x}")
    if not (0 <= data_area2 <= 0xFFFF):
        raise ValueError(f"data_area2 out of range: {data_area2:#x}")
    if not (0 <= motor_id <= 0xFF):
        raise ValueError(f"motor_id out of range: {motor_id:#x}")
    return ((cmd_type & 0x1F) << 24) | ((data_area2 & 0xFFFF) << 8) | (motor_id & 0xFF)


def build_enable(motor_id: int) -> Tuple[int, bytes]:
    """Frame to enable a motor (transition to operation mode)."""
    return _make_frame_id(CmdType.ENABLE, HOST_ID, motor_id), bytes(8)


def build_disable(motor_id: int) -> Tuple[int, bytes]:
    """Frame to stop a motor (transition out of operation mode)."""
    return _make_frame_id(CmdType.STOP, HOST_ID, motor_id), bytes(8)


def build_set_zero(motor_id: int) -> Tuple[int, bytes]:
    """Frame to set the current position as the new mechanical zero."""
    payload = bytes([1, 0, 0, 0, 0, 0, 0, 0])
    return _make_frame_id(CmdType.SET_ZERO, HOST_ID, motor_id), payload


def build_active_report(motor_id: int, interval_ms: int = 5) -> Tuple[int, bytes]:
    """Ask the motor to stream feedback every ``interval_ms`` ms.

    Default 5 ms = 200 Hz, well above the 100 Hz training control loop.
    """
    if not (1 <= interval_ms <= 255):
        raise ValueError("interval_ms must be in [1, 255]")
    payload = bytes([1, interval_ms, 0, 0, 0, 0, 0, 0])
    return _make_frame_id(CmdType.ACTIVE_REPORT, HOST_ID, motor_id), payload


def build_motor_ctrl(
    motor_id: int,
    model: str,
    position: float,
    velocity: float = 0.0,
    kp: float = 0.0,
    kd: float = 0.0,
    torque_ff: float = 0.0,
) -> Tuple[int, bytes]:
    """Build a MIT-mode control frame for the motor with ``motor_id``.

    ``torque_ff`` is the feed-forward torque (Nm) packed into the ID's data
    area 2 — it's *not* in the payload. The payload carries the PD setpoint.

    Output: (frame_id, 8-byte payload), where ``frame_id`` is the full
    29-bit extended CAN identifier. Big-endian throughout.
    """
    specs = get_specs(model)

    pos_raw = _f_to_u16(position, P_MIN, P_MAX)
    vel_raw = _f_to_u16(velocity, specs.velocity_min, specs.velocity_max)
    kp_raw = _f_to_u16(kp, KP_MIN, KP_MAX)
    kd_raw = _f_to_u16(kd, KD_MIN, KD_MAX)
    tau_raw = _f_to_u16(torque_ff, specs.torque_min, specs.torque_max)

    frame_id = _make_frame_id(CmdType.MOTOR_CTRL, tau_raw, motor_id)
    payload = struct.pack(">HHHH", pos_raw, vel_raw, kp_raw, kd_raw)
    return frame_id, payload


# ----------------------------------------------------------------- feedback parser
@dataclass
class Feedback:
    """Parsed Robstride feedback frame."""
    motor_id: int
    host_id: int
    fault_bits: int
    mode_status: int  # 0=reset, 1=cali, 2=motor (enabled), ...
    position: float   # rad
    velocity: float   # rad/s
    torque: float     # Nm
    temperature: float  # °C

    @property
    def is_enabled(self) -> bool:
        return self.mode_status == 2

    @property
    def has_fault(self) -> bool:
        return self.fault_bits != 0

    def fault_description(self) -> str:
        if self.fault_bits == 0:
            return ""
        flags = []
        if self.fault_bits & 0x01:
            flags.append("undervoltage")
        if self.fault_bits & 0x02:
            flags.append("overcurrent")
        if self.fault_bits & 0x04:
            flags.append("overtemperature")
        if self.fault_bits & 0x08:
            flags.append("encoder_fault")
        if self.fault_bits & 0x10:
            flags.append("gridlock_overload")
        if self.fault_bits & 0x20:
            flags.append("uncalibrated")
        return ",".join(flags) if flags else f"unknown(0x{self.fault_bits:02x})"


def parse_feedback(frame_id: int, data: bytes, model: str) -> Optional[Feedback]:
    """Parse an extended-frame feedback message. Returns None if not feedback.

    ``frame_id`` is the full 29-bit extended CAN identifier as received from
    the bus. The model is needed for the velocity / torque uint16 ranges
    since the Robstride frame does not encode the model.
    """
    if len(data) < 8:
        return None

    cmd_type = (frame_id >> 24) & 0x1F
    if cmd_type != CmdType.FEEDBACK:
        return None

    mode_status = (frame_id >> 22) & 0x03
    fault_bits = (frame_id >> 16) & 0x3F
    motor_id = (frame_id >> 8) & 0xFF
    host_id = frame_id & 0xFF

    pos_raw, vel_raw, tau_raw, temp_raw = struct.unpack(">HHHH", data[:8])

    specs = get_specs(model)
    position = _u16_to_f(pos_raw, P_MIN, P_MAX)
    velocity = _u16_to_f(vel_raw, specs.velocity_min, specs.velocity_max)
    torque = _u16_to_f(tau_raw, specs.torque_min, specs.torque_max)
    temperature = temp_raw / 10.0

    return Feedback(
        motor_id=motor_id,
        host_id=host_id,
        fault_bits=fault_bits,
        mode_status=mode_status,
        position=position,
        velocity=velocity,
        torque=torque,
        temperature=temperature,
    )
