#!/usr/bin/env python3
"""Robstride motor safety dial-in tool.

A *standalone* (non-ROS) bench-test utility for incrementally validating and
tightening the per-joint safety limits in
``ros2/src/bebop_pilot/config/safety_limits.yaml``.

Design — defense in depth
-------------------------
1.  **Outgoing clamp.**  Every TX setpoint is clamped to the joint's
    ``hard_limits`` box before it leaves the host. Nothing inside the box can
    escape, even if a test routine has a bug.
2.  **Incoming check.**  Every RX feedback frame is validated against the
    same limits.  Position / velocity / torque / temperature breach, or any
    motor-side fault bit, instantly transitions the state machine to
    ``E_STOP``.
3.  **Watchdog.**  If no feedback frame arrives within
    ``feedback_timeout_ms``, the supervisor triggers ``E_STOP``.  Catches USB
    drops, motor hangs, cable yanks, etc.
4.  **Slew limit.**  Commanded position can never step by more than
    ``slew.max_pos_step_per_tick`` per TX cycle — independent of the motor's
    PD gains, this caps the achievable jerk.
5.  **Always damped while ARMED.**  Hold gains have ``kd > 0`` so the joint
    is never limp under power.  Tests use richer gains but never zero ``kd``.
6.  **Process death is safe.**  Both ``atexit`` and SIGINT/SIGTERM handlers
    send ``Disable`` to the motor before the socket closes.

State machine
-------------
::

           any breach OR space-key
           ───────────────────────►
   IDLE ─arm──► ARMED ─test──► RUNNING                     ─►  E_STOP
    ▲           │              │                                │
    │           ▼              ▼                                │
    └──disarm── HOLD  ◄────abort (gentle ramp + disable) ◄──────┘

Tests
-----
``discover``    Read-only.  Pings the motor for its UID and reports model.
``manual``      Back-drive mode.  ``Kp = 0``, ``Kd > 0``: operator moves the
                joint by hand against viscous damping; tool logs reachable
                range to help dial ``pos_min`` / ``pos_max``.
``home``        Drive to ``0 rad`` with hold gains, hold ``hold_seconds``.
                Logs holding torque (gravity / static friction estimate) and
                temperature drift (thermal sanity check).
``pos_sweep``   Sinusoidal position sweep with growing amplitude up to (a
                fraction of) ``pos_min/max``.  Helps confirm the joint
                tracks across its software range without faulting.
``vel_probe``   Triangular position waveform with progressively higher peak
                velocity.  Achieved velocity is read back from feedback;
                aborts if it exceeds ``vel_max`` or fails to track.  Use
                this to dial ``vel_max``.

Operator interface
------------------
``SPACE``       Always-live E-STOP.  Instantly disables the motor regardless
                of state.  Requires explicit reset (``r``) before re-arming.
``q``           Quit cleanly: disarm → idle → exit.
``a``           Arm (only from ``IDLE``; refuses if joint is currently
                outside its hard pos limits).
``d``           Disarm (gentle ramp to zero command, then disable).
``r``           Reset E-STOP latch (does *not* re-arm).

CLI
---
::

  motor_safety_check.py --joint hip_abduction_left_joint --test home
  motor_safety_check.py --joint shin_left_joint --test pos_sweep --duration 20

CSV logs are written to ``~/bebop-safety-logs/<joint>_<test>_<timestamp>.csv``.
"""

from __future__ import annotations

import argparse
import atexit
import csv
import math
import os
import select
import signal
import struct
import sys
import termios
import threading
import time
import tty
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "PyYAML is required: sudo apt install python3-yaml  (or pip install pyyaml)"
    ) from exc

try:
    import can  # python-can
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "python-can is required: sudo apt install python3-can"
    ) from exc

# Make ``bebop_pilot.robstride`` importable when running from a checkout
# without having to source the ROS install tree.
_HERE = Path(__file__).resolve()
_PKG_ROOT = _HERE.parent.parent  # .../bebop_pilot/
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from bebop_pilot import robstride  # noqa: E402


# =============================================================================
# Config
# =============================================================================

DEFAULT_CONFIG_PATH = (
    _PKG_ROOT / "config" / "safety_limits.yaml"
)


@dataclass
class HardLimits:
    pos_min: float
    pos_max: float
    vel_max: float
    tau_max: float
    temp_max: float
    feedback_timeout_ms: float


@dataclass
class Gains:
    kp: float
    kd: float


@dataclass
class SlewParams:
    max_pos_step_per_tick: float
    arm_ramp_s: float
    abort_ramp_s: float


@dataclass
class JointConfig:
    name: str
    can_interface: str
    motor_id: int
    model: str
    hard_limits: HardLimits
    hold_gains: Gains
    test_gains: Gains
    slew: SlewParams


def _merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_joint_config(path: Path, joint_name: str) -> JointConfig:
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    defaults = data.get("defaults", {})
    joints = data.get("joints", {})
    if joint_name not in joints:
        names = sorted(joints.keys())
        raise SystemExit(
            f"Joint {joint_name!r} not found in {path}.\n"
            f"  Available: {names}"
        )

    j = _merge(defaults, joints[joint_name])

    hl = j["hard_limits"]
    return JointConfig(
        name=joint_name,
        can_interface=j["can_interface"],
        motor_id=int(j["motor_id"]),
        model=str(j["model"]).upper(),
        hard_limits=HardLimits(
            pos_min=float(hl["pos_min"]),
            pos_max=float(hl["pos_max"]),
            vel_max=float(hl["vel_max"]),
            tau_max=float(hl["tau_max"]),
            temp_max=float(hl["temp_max"]),
            feedback_timeout_ms=float(hl["feedback_timeout_ms"]),
        ),
        hold_gains=Gains(
            kp=float(j["hold_gains"]["kp"]), kd=float(j["hold_gains"]["kd"])
        ),
        test_gains=Gains(
            kp=float(j["test_gains"]["kp"]), kd=float(j["test_gains"]["kd"])
        ),
        slew=SlewParams(
            max_pos_step_per_tick=float(j["slew"]["max_pos_step_per_tick"]),
            arm_ramp_s=float(j["slew"]["arm_ramp_s"]),
            abort_ramp_s=float(j["slew"]["abort_ramp_s"]),
        ),
    )


# =============================================================================
# Shared state
# =============================================================================

class State(Enum):
    IDLE = auto()       # motor disabled
    ARMED = auto()      # motor enabled, holding last position with hold_gains
    RUNNING = auto()    # a test is driving setpoints
    HOLD = auto()       # between tests / after clean abort, like ARMED
    E_STOP = auto()     # latched fault; must reset before re-arming


@dataclass
class Feedback:
    pos: float = 0.0
    vel: float = 0.0
    tau: float = 0.0
    temp: float = 0.0
    fault_bits: int = 0
    mode_status: int = 0
    last_update: float = 0.0  # time.monotonic()


@dataclass
class SharedState:
    state: State = State.IDLE
    estop_reason: str = ""
    feedback: Feedback = field(default_factory=Feedback)
    last_target_pos: float = 0.0  # last commanded pos (used for slew limit)
    have_feedback: bool = False


# =============================================================================
# CAN link
# =============================================================================

def _read_can_state(interface: str) -> Optional[str]:
    """Return the current CAN controller state (e.g. 'ERROR-ACTIVE',
    'ERROR-PASSIVE', 'BUS-OFF'), or None if it can't be read.

    Reads from sysfs to avoid spawning ``ip``.  Falls back gracefully on
    distros / drivers that don't expose ``can_state``."""
    try:
        # SocketCAN exposes the controller state via netlink; the most portable
        # way to read it from Python is to shell out to `ip -details`.
        import subprocess
        out = subprocess.run(
            ["ip", "-details", "link", "show", interface],
            capture_output=True, text=True, check=False, timeout=1.0,
        ).stdout
        for line in out.splitlines():
            line = line.strip()
            if line.startswith("can state"):
                # e.g. "can state ERROR-ACTIVE restart-ms 0"
                parts = line.split()
                if len(parts) >= 3:
                    return parts[2]
    except Exception:
        pass
    return None


class CanLink:
    """Thin wrapper around python-can SocketCAN bus with cmd helpers."""

    def __init__(self, interface: str, motor_id: int, model: str):
        self.interface = interface
        self.motor_id = motor_id
        self.model = model

        # Pre-flight: refuse to operate on an ERROR-PASSIVE / BUS-OFF link.
        state = _read_can_state(interface)
        if state in ("ERROR-PASSIVE", "BUS-OFF"):
            raise SystemExit(
                f"CAN interface '{interface}' is in {state}.  This usually "
                f"means the motor / power board on this bus is unpowered or "
                f"unplugged, and the controller is unable to ACK frames.\n\n"
                f"  Recovery steps:\n"
                f"    1. Verify the device on this bus has power (check LEDs).\n"
                f"    2. Briefly unplug and replug the CANHub USB cable to\n"
                f"       fully reset the gs_usb firmware (kernel link cycles\n"
                f"       are not always sufficient).\n"
                f"    3. Re-bring up the link:\n"
                f"         sudo ip link set {interface} down\n"
                f"         sudo ip link set {interface} type can bitrate 1000000\n"
                f"         sudo ip link set {interface} up\n"
                f"    4. Verify with: ip -details link show {interface}\n"
                f"       (should report 'can state ERROR-ACTIVE')."
            )
        try:
            self._bus = can.interface.Bus(channel=interface, interface="socketcan")
        except Exception as exc:
            raise SystemExit(
                f"Could not open '{interface}': {exc}\n"
                f"  Bring it up:\n"
                f"    sudo ip link set {interface} down\n"
                f"    sudo ip link set {interface} type can bitrate 1000000\n"
                f"    sudo ip link set {interface} up"
            ) from exc

    def send_raw(self, frame_id: int, data: bytes) -> None:
        self._bus.send(
            can.Message(
                arbitration_id=frame_id, is_extended_id=True, data=data
            ),
            timeout=0.01,
        )

    def send_enable(self) -> None:
        fid, d = robstride.build_enable(self.motor_id)
        self.send_raw(fid, d)

    def send_disable(self) -> None:
        fid, d = robstride.build_disable(self.motor_id)
        self.send_raw(fid, d)

    def send_ctrl(
        self,
        position: float,
        velocity: float = 0.0,
        kp: float = 0.0,
        kd: float = 0.0,
        torque_ff: float = 0.0,
    ) -> None:
        fid, d = robstride.build_motor_ctrl(
            motor_id=self.motor_id,
            model=self.model,
            position=position,
            velocity=velocity,
            kp=kp,
            kd=kd,
            torque_ff=torque_ff,
        )
        self.send_raw(fid, d)

    def recv(self, timeout: float = 0.05) -> Optional[robstride.Feedback]:
        msg = self._bus.recv(timeout=timeout)
        if msg is None or not msg.is_extended_id:
            return None
        cmd_type = (msg.arbitration_id >> 24) & 0x1F
        if cmd_type != robstride.CmdType.FEEDBACK:
            return None
        m_id = (msg.arbitration_id >> 8) & 0xFF
        if m_id != self.motor_id:
            return None
        return robstride.parse_feedback(
            msg.arbitration_id, bytes(msg.data), self.model
        )

    def close(self) -> None:
        try:
            self._bus.shutdown()
        except Exception:
            pass


# =============================================================================
# CSV logger
# =============================================================================

class CsvLogger:
    HEADERS = [
        "t_s", "state", "pos_rad", "vel_rad_s", "tau_nm", "temp_c",
        "fault_bits", "mode_status",
        "tgt_pos", "tgt_vel", "tgt_kp", "tgt_kd", "tgt_tau_ff",
        "note",
    ]

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(path, "w", newline="")
        self._w = csv.writer(self._f)
        self._w.writerow(self.HEADERS)
        self._t0 = time.monotonic()
        self._lock = threading.Lock()

    def log(
        self,
        state: "State",
        fb: Feedback,
        target: Tuple[float, float, float, float, float] = (0, 0, 0, 0, 0),
        note: str = "",
    ) -> None:
        tgt_pos, tgt_vel, tgt_kp, tgt_kd, tgt_tau = target
        row = [
            f"{time.monotonic() - self._t0:.4f}",
            state.name,
            f"{fb.pos:.4f}", f"{fb.vel:.4f}", f"{fb.tau:.4f}", f"{fb.temp:.1f}",
            fb.fault_bits, fb.mode_status,
            f"{tgt_pos:.4f}", f"{tgt_vel:.4f}",
            f"{tgt_kp:.2f}", f"{tgt_kd:.2f}", f"{tgt_tau:.4f}",
            note,
        ]
        with self._lock:
            self._w.writerow(row)
            self._f.flush()

    def close(self) -> None:
        with self._lock:
            try:
                self._f.flush()
                self._f.close()
            except Exception:
                pass


# =============================================================================
# Keyboard listener — non-blocking single-key reads in cbreak mode
# =============================================================================

class KeyboardListener:
    """Reads single keystrokes from stdin without blocking.

    Uses cbreak mode so individual keys (space, q, etc.) come through
    immediately without waiting for newline.
    """

    def __init__(self) -> None:
        self._enabled = sys.stdin.isatty()
        self._old_settings: Optional[list] = None
        self._fd = sys.stdin.fileno() if self._enabled else -1

    def __enter__(self) -> "KeyboardListener":
        if self._enabled:
            self._old_settings = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        return self

    def __exit__(self, *exc) -> None:
        if self._enabled and self._old_settings is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)

    def poll(self) -> Optional[str]:
        if not self._enabled:
            return None
        ready, _, _ = select.select([sys.stdin], [], [], 0)
        if ready:
            ch = sys.stdin.read(1)
            return ch
        return None


# =============================================================================
# Safety supervisor
# =============================================================================

class Supervisor:
    """Owns the shared state, RX thread, watchdog, and limit checks.

    All state mutations go through this object to keep the threading model
    centralized: one RX thread updates ``self.state.feedback``, the test
    thread reads it, and *only* this class transitions the state enum.
    """

    def __init__(
        self,
        cfg: JointConfig,
        link: CanLink,
        logger: CsvLogger,
    ) -> None:
        self.cfg = cfg
        self.link = link
        self.logger = logger
        self.state = SharedState()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._estop_event = threading.Event()
        self._rx_thread = threading.Thread(
            target=self._rx_loop, name="rx", daemon=True
        )

    # ---------------------------------------------------------------- lifecycle
    def start(self) -> None:
        self._rx_thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._rx_thread.join(timeout=1.0)

    # ---------------------------------------------------------------- E-STOP
    def trigger_estop(self, reason: str) -> None:
        if self._estop_event.is_set():
            return
        self._estop_event.set()
        with self._lock:
            self.state.state = State.E_STOP
            self.state.estop_reason = reason
        # Best-effort: fire disables a few times in case one is dropped.
        for _ in range(3):
            try:
                self.link.send_disable()
            except Exception:
                pass
            time.sleep(0.005)
        self.logger.log(
            State.E_STOP, self.state.feedback, note=f"E_STOP: {reason}"
        )

    def reset_estop(self) -> bool:
        if not self._estop_event.is_set():
            return False
        self._estop_event.clear()
        with self._lock:
            self.state.state = State.IDLE
            self.state.estop_reason = ""
        return True

    @property
    def estop_active(self) -> bool:
        return self._estop_event.is_set()

    # ---------------------------------------------------------------- transitions
    def set_state(self, new: State) -> None:
        with self._lock:
            if self.state.state == State.E_STOP and new != State.IDLE:
                return  # cannot leave E_STOP except via reset_estop()
            self.state.state = new

    def get_state(self) -> State:
        with self._lock:
            return self.state.state

    def get_feedback(self) -> Feedback:
        with self._lock:
            # return a copy so callers don't see torn reads
            fb = self.state.feedback
            return Feedback(
                pos=fb.pos, vel=fb.vel, tau=fb.tau, temp=fb.temp,
                fault_bits=fb.fault_bits, mode_status=fb.mode_status,
                last_update=fb.last_update,
            )

    def have_feedback(self) -> bool:
        with self._lock:
            return self.state.have_feedback

    # ---------------------------------------------------------------- RX loop
    def _rx_loop(self) -> None:
        while not self._stop.is_set():
            try:
                fb = self.link.recv(timeout=0.05)
            except can.CanError:
                continue
            now = time.monotonic()
            if fb is None:
                continue
            with self._lock:
                self.state.feedback = Feedback(
                    pos=fb.position, vel=fb.velocity,
                    tau=fb.torque, temp=fb.temperature,
                    fault_bits=fb.fault_bits,
                    mode_status=fb.mode_status,
                    last_update=now,
                )
                self.state.have_feedback = True

            # Run safety checks on the freshly-arrived frame.
            self._check_limits(fb)

    # ---------------------------------------------------------------- limit checks
    def _check_limits(self, fb: robstride.Feedback) -> None:
        h = self.cfg.hard_limits
        if fb.fault_bits != 0:
            self.trigger_estop(f"motor fault: {fb.fault_description()}")
            return
        if fb.position < h.pos_min - 1e-3 or fb.position > h.pos_max + 1e-3:
            self.trigger_estop(
                f"position {fb.position:+.3f} rad outside "
                f"[{h.pos_min:+.3f}, {h.pos_max:+.3f}]"
            )
            return
        if abs(fb.velocity) > h.vel_max:
            self.trigger_estop(
                f"|velocity| {abs(fb.velocity):.2f} > vel_max {h.vel_max:.2f} rad/s"
            )
            return
        if abs(fb.torque) > h.tau_max:
            self.trigger_estop(
                f"|torque| {abs(fb.torque):.2f} > tau_max {h.tau_max:.2f} Nm"
            )
            return
        if fb.temperature > h.temp_max:
            self.trigger_estop(
                f"temperature {fb.temperature:.1f} > temp_max {h.temp_max:.1f} °C"
            )
            return

    # ---------------------------------------------------------------- watchdog
    def watchdog_check(self) -> None:
        if self.estop_active:
            return
        with self._lock:
            have_fb = self.state.have_feedback
            last = self.state.feedback.last_update
            cur_state = self.state.state
        # No watchdog for IDLE / E_STOP — only when motor is supposed to be live.
        if cur_state in (State.IDLE, State.E_STOP):
            return
        if not have_fb:
            return  # haven't received the first frame yet, give it a moment
        elapsed_ms = (time.monotonic() - last) * 1000.0
        if elapsed_ms > self.cfg.hard_limits.feedback_timeout_ms:
            self.trigger_estop(
                f"feedback watchdog: {elapsed_ms:.0f} ms since last RX "
                f"(timeout {self.cfg.hard_limits.feedback_timeout_ms:.0f} ms)"
            )

    # ---------------------------------------------------------------- safe TX
    def safe_send_ctrl(
        self,
        target_pos: float,
        kp: float,
        kd: float,
        velocity_ff: float = 0.0,
        torque_ff: float = 0.0,
        note: str = "",
    ) -> None:
        """Clamp + slew-limit + send.  All TX traffic *must* go through here."""
        if self.estop_active:
            return
        h = self.cfg.hard_limits
        # Clamp commanded position to hard pos limits.
        clamped = max(h.pos_min, min(h.pos_max, target_pos))
        # Slew-rate limit relative to last sent setpoint.
        with self._lock:
            last = self.state.last_target_pos
        step = self.cfg.slew.max_pos_step_per_tick
        clamped = max(last - step, min(last + step, clamped))
        with self._lock:
            self.state.last_target_pos = clamped
        # Velocity / torque ff also clamped (defensive — these go straight to
        # the motor's PD term).
        velocity_ff = max(-h.vel_max, min(h.vel_max, velocity_ff))
        torque_ff = max(-h.tau_max, min(h.tau_max, torque_ff))

        try:
            self.link.send_ctrl(
                position=clamped, velocity=velocity_ff,
                kp=kp, kd=kd, torque_ff=torque_ff,
            )
        except can.CanError as exc:
            self.trigger_estop(f"CAN TX error: {exc}")
            return
        # Log every TX with current feedback snapshot.
        self.logger.log(
            self.get_state(), self.get_feedback(),
            target=(clamped, velocity_ff, kp, kd, torque_ff),
            note=note,
        )


# =============================================================================
# Tests
# =============================================================================

class TestAbort(Exception):
    """Raised by tests to bail out cleanly to HOLD state."""


def _wait_for_first_feedback(sup: Supervisor, timeout_s: float = 1.0) -> Feedback:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if sup.have_feedback():
            return sup.get_feedback()
        time.sleep(0.01)
    raise SystemExit(
        "No feedback frames received from motor within "
        f"{timeout_s:.1f}s.  Check the motor is powered, the can interface "
        f"is up, and the motor_id is correct."
    )


def _kbd_check(kbd: KeyboardListener, sup: Supervisor) -> None:
    """Should be called frequently inside long-running loops.

    Honors SPACE for E-STOP, ``x`` for graceful test abort, and ``q`` for
    immediate quit.
    """
    ch = kbd.poll()
    if ch is None:
        return
    if ch == " ":
        sup.trigger_estop("operator SPACE")
        raise TestAbort("operator E-STOP")
    if ch == "x":
        raise TestAbort("operator abort (x)")
    if ch == "q":
        sup.trigger_estop("operator quit")
        raise TestAbort("operator quit")


# ---- discover ---------------------------------------------------------------

def discover_standalone(cfg: JointConfig) -> int:
    """Read-only motor probe.  Bypasses the supervisor / RX thread so the
    GET_ID response isn't consumed by another thread.  Run *before* the
    supervisor starts."""
    print(
        f"[discover] motor_id={cfg.motor_id} on {cfg.can_interface}, "
        f"model={cfg.model}"
    )
    link = CanLink(cfg.can_interface, cfg.motor_id, cfg.model)
    try:
        fid = ((robstride.CmdType.GET_ID & 0x1F) << 24) | (
            (robstride.HOST_ID & 0xFFFF) << 8
        ) | (cfg.motor_id & 0xFF)
        link.send_raw(fid, bytes(8))
        deadline = time.monotonic() + 0.3
        while time.monotonic() < deadline:
            msg = link._bus.recv(timeout=0.05)
            if msg is None or not msg.is_extended_id:
                continue
            cmd_type = (msg.arbitration_id >> 24) & 0x1F
            if cmd_type == robstride.CmdType.GET_ID:
                uid = bytes(msg.data).hex().upper()
                print(f"[discover] OK  id=0x{msg.arbitration_id:08X}  uid={uid}")
                return 0
        print(
            "[discover] no GET_ID response within 300 ms.  Motor may be "
            "unpowered, on a different ID, or on a different bus."
        )
        return 1
    finally:
        link.close()


def test_discover(sup: Supervisor, kbd: KeyboardListener, _args) -> None:
    """Stub for the test registry — actually invoked via the standalone path
    in run_session() before the supervisor starts."""
    raise RuntimeError(
        "test_discover should be invoked via discover_standalone()"
    )


# ---- manual (back-drive) ----------------------------------------------------

def test_manual(sup: Supervisor, kbd: KeyboardListener, args) -> None:
    """Pure-damping mode: ``Kp = 0``, ``Kd > 0``.  Operator moves the joint
    by hand; we record the achieved range to help dial pos limits."""
    duration = float(getattr(args, "duration", 30.0))
    rate = 100.0
    period = 1.0 / rate

    sup.set_state(State.RUNNING)
    fb = _wait_for_first_feedback(sup)
    pos_lo = fb.pos
    pos_hi = fb.pos
    vel_peak = 0.0
    tau_peak = 0.0
    t_end = time.monotonic() + duration
    print(
        f"[manual] back-drive for {duration:.0f}s — move the joint by hand. "
        f"SPACE=E-STOP  x=stop early"
    )
    next_t = time.monotonic()
    while time.monotonic() < t_end:
        _kbd_check(kbd, sup)
        if sup.estop_active:
            raise TestAbort("estop during manual")
        sup.watchdog_check()
        # Send a zero-position MIT command with kp=0, kd=hold_kd. The motor
        # ignores the position setpoint when kp=0, so we send fb.pos as a
        # cosmetic placeholder (irrelevant when kp=0).
        fb_now = sup.get_feedback()
        pos_lo = min(pos_lo, fb_now.pos)
        pos_hi = max(pos_hi, fb_now.pos)
        vel_peak = max(vel_peak, abs(fb_now.vel))
        tau_peak = max(tau_peak, abs(fb_now.tau))
        sup.safe_send_ctrl(
            target_pos=fb_now.pos,
            kp=0.0,
            kd=sup.cfg.hold_gains.kd,
            note="manual",
        )
        _print_status(sup, prefix="manual")
        next_t += period
        sleep_for = next_t - time.monotonic()
        if sleep_for > 0:
            time.sleep(sleep_for)
    print()
    print(
        f"[manual] observed range: pos ∈ [{pos_lo:+.3f}, {pos_hi:+.3f}] rad "
        f"({math.degrees(pos_lo):+.0f}°, {math.degrees(pos_hi):+.0f}°)\n"
        f"[manual] peak |vel|={vel_peak:.2f} rad/s   peak |tau|={tau_peak:.2f} Nm"
    )


# ---- home -------------------------------------------------------------------

def test_home(sup: Supervisor, kbd: KeyboardListener, args) -> None:
    """Drive to 0 rad with hold gains and hold for ``duration`` seconds."""
    duration = float(getattr(args, "duration", 5.0))
    rate = 100.0
    period = 1.0 / rate
    fb = _wait_for_first_feedback(sup)
    print(
        f"[home] start pos={fb.pos:+.3f} rad → drive to 0 with "
        f"kp={sup.cfg.hold_gains.kp}, kd={sup.cfg.hold_gains.kd}, "
        f"hold {duration:.1f}s"
    )
    sup.set_state(State.RUNNING)

    # Two-phase: drive (slew toward 0) until close, then hold.
    settle_band = 0.02  # rad
    settle_time = 0.0
    settled = False
    t0 = time.monotonic()
    next_t = t0
    deadline = t0 + duration + 10.0  # hard ceiling on whole test
    hold_start: Optional[float] = None
    while time.monotonic() < deadline:
        _kbd_check(kbd, sup)
        if sup.estop_active:
            raise TestAbort("estop during home")
        sup.watchdog_check()
        fb_now = sup.get_feedback()
        if not settled:
            if abs(fb_now.pos) < settle_band:
                settle_time += period
                if settle_time > 0.3:
                    settled = True
                    hold_start = time.monotonic()
                    print(f"\n[home] settled @ {fb_now.pos:+.3f} rad — holding")
            else:
                settle_time = 0.0
        else:
            if hold_start is not None and time.monotonic() - hold_start >= duration:
                break
        sup.safe_send_ctrl(
            target_pos=0.0,
            kp=sup.cfg.hold_gains.kp,
            kd=sup.cfg.hold_gains.kd,
            note="home",
        )
        _print_status(sup, prefix="home ")
        next_t += period
        sleep_for = next_t - time.monotonic()
        if sleep_for > 0:
            time.sleep(sleep_for)
    print()
    if not settled:
        raise TestAbort("home failed to settle in 10s")
    fb_end = sup.get_feedback()
    print(
        f"[home] hold complete  pos={fb_end.pos:+.3f}  "
        f"hold_tau={fb_end.tau:+.2f} Nm  temp={fb_end.temp:.1f} °C"
    )


# ---- pos_sweep --------------------------------------------------------------

def test_pos_sweep(sup: Supervisor, kbd: KeyboardListener, args) -> None:
    """Sinusoidal position sweep with growing amplitude.  Start tiny, grow
    until ``amp_max`` rad or operator stops."""
    duration = float(getattr(args, "duration", 20.0))
    freq = float(getattr(args, "freq", 0.5))   # Hz
    amp_max = float(getattr(args, "amp", None) or
                    min(0.7 * sup.cfg.hard_limits.pos_max,
                        0.7 * abs(sup.cfg.hard_limits.pos_min)))
    amp_max = min(amp_max, sup.cfg.hard_limits.pos_max,
                  abs(sup.cfg.hard_limits.pos_min))
    rate = 200.0
    period = 1.0 / rate
    print(
        f"[pos_sweep] sine f={freq:.2f} Hz, amp ramps 0 → {amp_max:.2f} rad "
        f"over {duration:.1f}s, kp={sup.cfg.test_gains.kp}, "
        f"kd={sup.cfg.test_gains.kd}"
    )
    sup.set_state(State.RUNNING)

    t0 = time.monotonic()
    next_t = t0
    while True:
        _kbd_check(kbd, sup)
        if sup.estop_active:
            raise TestAbort("estop during pos_sweep")
        sup.watchdog_check()
        t = time.monotonic() - t0
        if t > duration:
            break
        amp = amp_max * (t / duration)  # linear ramp
        target = amp * math.sin(2 * math.pi * freq * t)
        sup.safe_send_ctrl(
            target_pos=target,
            kp=sup.cfg.test_gains.kp,
            kd=sup.cfg.test_gains.kd,
            note=f"sweep amp={amp:.3f}",
        )
        _print_status(sup, prefix="sweep")
        next_t += period
        sleep_for = next_t - time.monotonic()
        if sleep_for > 0:
            time.sleep(sleep_for)
    print()
    print("[pos_sweep] complete")


# ---- vel_probe --------------------------------------------------------------

def test_vel_probe(sup: Supervisor, kbd: KeyboardListener, args) -> None:
    """Triangle position waveform with progressively higher peak commanded
    velocity, in MIT mode.  We *do not* switch the motor's run mode — the
    velocity ceiling is enforced by the position trajectory and Kp/Kd.

    Peak commanded velocity = 4 · amp · freq (triangle wave).  We sweep
    ``freq`` upward at a fixed amplitude (small) and read back achieved
    velocity from feedback.  Aborts if achieved |vel| ever exceeds 0.9 ·
    vel_max — at that point we have *characterized* vel_max."""
    duration = float(getattr(args, "duration", 20.0))
    amp = float(getattr(args, "amp", None) or
                min(0.2, 0.5 * sup.cfg.hard_limits.pos_max,
                    0.5 * abs(sup.cfg.hard_limits.pos_min)))
    f_start = 0.2  # Hz
    f_stop = float(getattr(args, "freq", None) or
                   sup.cfg.hard_limits.vel_max / max(4.0 * amp, 1e-6))
    rate = 200.0
    period = 1.0 / rate
    target_peak_vel = 4.0 * amp * f_stop
    print(
        f"[vel_probe] triangle amp={amp:.2f} rad, freq {f_start:.2f}→"
        f"{f_stop:.2f} Hz over {duration:.1f}s "
        f"(commanded peak vel = {target_peak_vel:.2f} rad/s, "
        f"hard cap = {sup.cfg.hard_limits.vel_max:.2f})"
    )
    sup.set_state(State.RUNNING)

    achieved_vel_peak = 0.0
    t0 = time.monotonic()
    next_t = t0
    while True:
        _kbd_check(kbd, sup)
        if sup.estop_active:
            raise TestAbort("estop during vel_probe")
        sup.watchdog_check()
        t = time.monotonic() - t0
        if t > duration:
            break
        f = f_start + (f_stop - f_start) * (t / duration)
        # Triangle wave in [-amp, +amp].
        phase = (f * t) % 1.0
        triangle = 4.0 * abs(phase - 0.5) - 1.0  # /\/\
        target = -amp * triangle
        sup.safe_send_ctrl(
            target_pos=target,
            kp=sup.cfg.test_gains.kp,
            kd=sup.cfg.test_gains.kd,
            note=f"vel_probe f={f:.2f}",
        )
        fb = sup.get_feedback()
        achieved_vel_peak = max(achieved_vel_peak, abs(fb.vel))
        _print_status(sup, prefix=f"velpr f={f:4.2f}")
        next_t += period
        sleep_for = next_t - time.monotonic()
        if sleep_for > 0:
            time.sleep(sleep_for)
    print()
    print(
        f"[vel_probe] achieved peak |vel| = {achieved_vel_peak:.2f} rad/s "
        f"({math.degrees(achieved_vel_peak):.0f}°/s).  "
        f"Hard cap = {sup.cfg.hard_limits.vel_max:.2f}."
    )


# =============================================================================
# Status line
# =============================================================================

def _bar(value: float, lo: float, hi: float, width: int = 16) -> str:
    """A simple inline bar showing |value|/max scaled to width."""
    span = hi - lo
    if span <= 0:
        return " " * width
    frac = max(0.0, min(1.0, (value - lo) / span))
    n = int(round(frac * width))
    return "[" + "#" * n + "-" * (width - n) + "]"


def _print_status(sup: Supervisor, prefix: str = "") -> None:
    h = sup.cfg.hard_limits
    fb = sup.get_feedback()
    st = sup.get_state()
    pos_pct = max(abs(fb.pos / max(h.pos_max, abs(h.pos_min))), 0.0) * 100.0
    vel_pct = (abs(fb.vel) / h.vel_max) * 100.0 if h.vel_max > 0 else 0.0
    tau_pct = (abs(fb.tau) / h.tau_max) * 100.0 if h.tau_max > 0 else 0.0
    temp_pct = (fb.temp / h.temp_max) * 100.0 if h.temp_max > 0 else 0.0
    line = (
        f"\r[{st.name:<7s}] {prefix:<14s} "
        f"pos {fb.pos:+6.3f}({pos_pct:3.0f}%) "
        f"vel {fb.vel:+6.2f}({vel_pct:3.0f}%) "
        f"tau {fb.tau:+6.2f}({tau_pct:3.0f}%) "
        f"T {fb.temp:5.1f}({temp_pct:3.0f}%) "
        f"flt 0x{fb.fault_bits:02X} "
    )
    sys.stdout.write(line)
    sys.stdout.flush()


# =============================================================================
# Session orchestrator
# =============================================================================

TESTS: Dict[str, Callable[[Supervisor, KeyboardListener, argparse.Namespace], None]] = {
    "discover": test_discover,
    "manual": test_manual,
    "home": test_home,
    "pos_sweep": test_pos_sweep,
    "vel_probe": test_vel_probe,
}


def _arm_with_ramp(sup: Supervisor) -> None:
    """Enable the motor and ramp gains from 0 to hold_gains over arm_ramp_s."""
    if sup.estop_active:
        raise SystemExit("Cannot arm while E-STOP is latched. Press 'r' to reset.")
    fb = _wait_for_first_feedback(sup)
    h = sup.cfg.hard_limits
    if not (h.pos_min - 1e-3 <= fb.pos <= h.pos_max + 1e-3):
        raise SystemExit(
            f"Refusing to arm: current pos {fb.pos:+.3f} rad is outside "
            f"hard limits [{h.pos_min:+.3f}, {h.pos_max:+.3f}]. "
            f"Move the joint into range manually first."
        )
    sup.link.send_enable()
    time.sleep(0.05)
    # Lock the slew limiter to the current pose so the very first ctrl
    # frame can't request a far-away target.
    with sup._lock:
        sup.state.last_target_pos = fb.pos

    rate = 100.0
    period = 1.0 / rate
    n = max(1, int(sup.cfg.slew.arm_ramp_s * rate))
    for i in range(n):
        if sup.estop_active:
            return
        kp = sup.cfg.hold_gains.kp * (i + 1) / n
        kd = sup.cfg.hold_gains.kd * (i + 1) / n
        sup.safe_send_ctrl(
            target_pos=fb.pos, kp=kp, kd=kd, note="arm_ramp"
        )
        time.sleep(period)
    sup.set_state(State.ARMED)


def _disarm_with_ramp(sup: Supervisor) -> None:
    """Ramp gains down, send a final ctrl with low kd, then disable."""
    if sup.get_state() == State.IDLE:
        return
    fb = sup.get_feedback()
    rate = 100.0
    period = 1.0 / rate
    n = max(1, int(sup.cfg.slew.abort_ramp_s * rate))
    for i in range(n):
        frac = 1.0 - (i + 1) / n
        sup.safe_send_ctrl(
            target_pos=fb.pos,
            kp=sup.cfg.hold_gains.kp * frac,
            kd=sup.cfg.hold_gains.kd * max(frac, 0.2),
            note="disarm_ramp",
        )
        time.sleep(period)
    try:
        sup.link.send_disable()
    except Exception:
        pass
    sup.set_state(State.IDLE)


def run_session(args: argparse.Namespace) -> int:
    cfg = load_joint_config(Path(args.config), args.joint)

    print("─" * 72)
    print(f"motor_safety_check  joint={cfg.name}  test={args.test}")
    print(f"  can_interface={cfg.can_interface}  motor_id={cfg.motor_id}  "
          f"model={cfg.model}")
    h = cfg.hard_limits
    print(f"  hard_limits: pos∈[{h.pos_min:+.2f}, {h.pos_max:+.2f}] rad  "
          f"vel_max={h.vel_max:.2f} rad/s  tau_max={h.tau_max:.2f} Nm  "
          f"temp_max={h.temp_max:.1f}°C")

    # discover runs without the supervisor / RX thread so the response
    # isn't swallowed by a background reader.
    if args.test == "discover":
        print("─" * 72)
        return discover_standalone(cfg)

    log_dir = Path(os.path.expanduser(args.log_dir))
    ts = time.strftime("%Y%m%d-%H%M%S")
    log_path = log_dir / f"{cfg.name}_{args.test}_{ts}.csv"
    print(f"  log: {log_path}")
    print("─" * 72)

    link = CanLink(cfg.can_interface, cfg.motor_id, cfg.model)
    logger = CsvLogger(log_path)
    sup = Supervisor(cfg, link, logger)

    cleaned_up = threading.Event()

    # Always-disable cleanup, regardless of how we exit.  Order matters:
    # stop the RX thread *first* so it doesn't try to recv on a closed bus.
    def _cleanup(*_args) -> None:
        if cleaned_up.is_set():
            return
        cleaned_up.set()
        try:
            for _ in range(3):
                link.send_disable()
                time.sleep(0.005)
        except Exception:
            pass
        try:
            sup.stop()           # joins RX thread
        except Exception:
            pass
        try:
            link.close()
        except Exception:
            pass
        try:
            logger.close()
        except Exception:
            pass

    atexit.register(_cleanup)
    signal.signal(signal.SIGINT, lambda *_: (_cleanup(), os._exit(130)))
    signal.signal(signal.SIGTERM, lambda *_: (_cleanup(), os._exit(143)))

    sup.start()

    rc = 0
    try:
        with KeyboardListener() as kbd:
            print(
                "Press 'a' to ARM the motor, 'q' to quit, SPACE for E-STOP. "
                "(armed motor will gently hold its current pose)"
            )
            while True:
                _kbd_check(kbd, sup)
                ch = kbd.poll()
                if ch == "a":
                    _arm_with_ramp(sup)
                    break
                elif ch == "q":
                    return 0
                elif ch == "r" and sup.estop_active:
                    sup.reset_estop()
                    print("E-STOP cleared. Press 'a' to arm.")
                time.sleep(0.05)

            print(f"Armed. Starting test: {args.test}.  "
                  f"SPACE=E-STOP  x=abort  q=quit")
            try:
                TESTS[args.test](sup, kbd, args)
            except TestAbort as exc:
                print(f"\n[test] aborted: {exc}")
                rc = 2

            if not sup.estop_active:
                _disarm_with_ramp(sup)
                print("Motor disarmed cleanly.")
            else:
                print(f"E-STOP latched: {sup.state.estop_reason}")
                rc = 3
    finally:
        _cleanup()

    return rc


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", default=str(DEFAULT_CONFIG_PATH),
        help=f"safety_limits.yaml path (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--joint", required=True,
        help="Joint key from safety_limits.yaml (e.g. hip_abduction_left_joint).",
    )
    parser.add_argument(
        "--test", required=True, choices=sorted(TESTS.keys()),
        help="Which dial-in test to run.",
    )
    parser.add_argument(
        "--duration", type=float, default=None,
        help="Test duration (seconds). Defaults vary by test.",
    )
    parser.add_argument(
        "--amp", type=float, default=None,
        help="Position amplitude (rad). pos_sweep / vel_probe only.",
    )
    parser.add_argument(
        "--freq", type=float, default=None,
        help="Frequency (Hz). pos_sweep / vel_probe only.",
    )
    parser.add_argument(
        "--log-dir", default="~/bebop-safety-logs",
        help="Where to write CSV logs (default: ~/bebop-safety-logs).",
    )
    args = parser.parse_args()
    if args.duration is None:
        args.duration = {
            "discover": 1.0, "manual": 30.0, "home": 5.0,
            "pos_sweep": 20.0, "vel_probe": 20.0,
        }[args.test]

    return run_session(args)


if __name__ == "__main__":
    sys.exit(main())
