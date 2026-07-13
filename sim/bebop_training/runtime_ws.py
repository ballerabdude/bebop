"""WebSocket client for bebop-linux runtime telemetry.

Connects to ``ws://<host>:<port>/ws``, subscribes to periodic
``TelemetryFrame`` protobuf messages, and exposes the latest frame to the
Isaac Lab pose-mirror loop.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Sequence

import websocket
from websocket import ABNF

from bebop_training.proto import bebop_runtime_pb2 as pb

DEFAULT_PORT = 9090
ACK_TIMEOUT_S = 5.0
RECONNECT_DELAY_S = 5.0


@dataclass(frozen=True)
class MotorSample:
    joint_name: str
    position_rad: float
    velocity_rad_s: float
    position_received: bool = False
    feedback_stale: bool = False
    armed: bool = False


def motor_position_live(motor: MotorSample) -> bool:
    """True when telemetry carries a usable joint angle for mirror mode."""
    if motor.position_received:
        return True
    # Back-compat: firmware builds before ``position_received`` still stream
    # live ``position_rad`` once feedback is flowing — protobuf decodes the
    # missing bool as False, so gate on ``feedback_stale`` instead.
    if not motor.feedback_stale:
        return True
    return False


@dataclass(frozen=True)
class ImuSample:
    present: bool
    received: bool
    stale: bool
    quaternion_xyzw: tuple[float, float, float, float]


@dataclass
class TelemetrySnapshot:
    host_unix_ms: int
    motors: dict[str, MotorSample] = field(default_factory=dict)
    imu: ImuSample = field(
        default_factory=lambda: ImuSample(
            present=False,
            received=False,
            stale=True,
            quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
        )
    )


def motors_in_joint_order(
    snapshot: TelemetrySnapshot,
    joint_names: Sequence[str],
    *,
    last_positions: dict[str, float] | None = None,
    last_velocities: dict[str, float] | None = None,
    default_pos: float = 0.0,
    default_vel: float = 0.0,
) -> tuple[list[float], list[float], list[str]]:
    """Map a telemetry frame into firmware joint order.

    When ``position_received`` is false for a motor, the last known value
    is held (if provided) so mirror consumers do not snap back to zero.

    Returns ``(positions, velocities, missing_joint_names)``.
    """
    positions: list[float] = []
    velocities: list[float] = []
    missing: list[str] = []
    for name in joint_names:
        motor = snapshot.motors.get(name)
        if motor is None:
            missing.append(name)
            positions.append(
                last_positions.get(name, default_pos) if last_positions else default_pos
            )
            velocities.append(
                last_velocities.get(name, default_vel) if last_velocities else default_vel
            )
        elif motor_position_live(motor):
            positions.append(motor.position_rad)
            velocities.append(motor.velocity_rad_s)
        else:
            positions.append(
                last_positions.get(name, default_pos) if last_positions else default_pos
            )
            velocities.append(
                last_velocities.get(name, default_vel) if last_velocities else default_vel
            )
    return positions, velocities, missing


def _snapshot_from_frame(frame: pb.TelemetryFrame) -> TelemetrySnapshot:
    motors = {
        motor.joint_name: MotorSample(
            joint_name=motor.joint_name,
            position_rad=motor.position_rad,
            velocity_rad_s=motor.velocity_rad_s,
            position_received=motor.position_received,
            feedback_stale=motor.feedback_stale,
            armed=motor.armed,
        )
        for motor in frame.motors
        if motor.joint_name
    }
    imu = frame.imu
    imu_sample = ImuSample(
        present=imu.present,
        received=imu.received,
        stale=imu.stale,
        quaternion_xyzw=(
            imu.quaternion_x,
            imu.quaternion_y,
            imu.quaternion_z,
            imu.quaternion_w,
        ),
    )
    return TelemetrySnapshot(
        host_unix_ms=int(frame.host_unix_ms),
        motors=motors,
        imu=imu_sample,
    )


class RuntimeTelemetryClient:
    """Background WebSocket subscriber for ``TelemetryFrame`` messages."""

    def __init__(
        self,
        host: str,
        port: int = DEFAULT_PORT,
        *,
        rate_hz: int = 30,
    ) -> None:
        self._host = host
        self._port = port
        self._rate_hz = rate_hz
        self._url = f"ws://{host}:{port}/ws"
        self._next_request_id = 1
        self._lock = threading.Lock()
        self._latest: TelemetrySnapshot | None = None
        self._connected = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._ws: websocket.WebSocket | None = None
        self._last_error: str | None = None

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="bebop-runtime-telemetry",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        ws = self._ws
        self._ws = None
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._connected = False

    def latest(self) -> TelemetrySnapshot | None:
        with self._lock:
            return self._latest

    def wait_for_frame(self, timeout_s: float = ACK_TIMEOUT_S) -> TelemetrySnapshot:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            frame = self.latest()
            if frame is not None:
                return frame
            if self._stop.is_set():
                break
            time.sleep(0.01)
        raise TimeoutError(
            f"timed out waiting for telemetry from {self._url} after {timeout_s:.1f}s"
            + (f" ({self._last_error})" if self._last_error else "")
        )

    @staticmethod
    def _send(ws: websocket.WebSocket, msg: pb.ClientRuntimeMessage) -> None:
        # bebop-linux ignores text frames; protobuf must be sent as binary.
        ws.send(msg.SerializeToString(), opcode=ABNF.OPCODE_BINARY)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._connect_once()
            except Exception as exc:
                self._last_error = f"{type(exc).__name__}: {exc}"
                self._connected = False
                if self._stop.wait(RECONNECT_DELAY_S):
                    break

    def _connect_once(self) -> None:
        ws = websocket.create_connection(self._url, timeout=ACK_TIMEOUT_S)
        ws.settimeout(ACK_TIMEOUT_S)
        self._ws = ws
        self._subscribe(ws)
        self._connected = True
        self._last_error = None
        while not self._stop.is_set():
            try:
                data = ws.recv()
            except websocket.WebSocketTimeoutException:
                continue
            except websocket.WebSocketConnectionClosedException:
                break
            if not isinstance(data, (bytes, bytearray)):
                continue
            msg = pb.ServerRuntimeMessage()
            try:
                msg.ParseFromString(data)
            except Exception:
                continue
            payload = msg.WhichOneof("payload")
            if payload == "telemetry":
                snap = _snapshot_from_frame(msg.telemetry)
                with self._lock:
                    self._latest = snap
            elif payload == "error":
                if msg.request_id == 0 or msg.error.message:
                    raise RuntimeError(msg.error.message or "runtime error")

    def _subscribe(self, ws: websocket.WebSocket) -> None:
        request_id = self._next_request_id
        self._next_request_id += 1
        req = pb.ClientRuntimeMessage(
            request_id=request_id,
            subscribe_telemetry=pb.SubscribeTelemetry(rate_hz=self._rate_hz),
        )
        self._send(ws, req)
        deadline = time.monotonic() + ACK_TIMEOUT_S
        while time.monotonic() < deadline:
            data = ws.recv()
            if not isinstance(data, (bytes, bytearray)):
                continue
            msg = pb.ServerRuntimeMessage()
            msg.ParseFromString(data)
            payload = msg.WhichOneof("payload")
            if payload == "error" and msg.request_id in (0, request_id):
                raise RuntimeError(msg.error.message or "subscribe failed")
            if msg.request_id != request_id:
                if payload == "telemetry":
                    snap = _snapshot_from_frame(msg.telemetry)
                    with self._lock:
                        self._latest = snap
                continue
            if payload == "ack":
                if not msg.ack.ok:
                    raise RuntimeError(msg.ack.message or "subscribe failed")
                return
            if payload == "error":
                raise RuntimeError(msg.error.message or "subscribe failed")
        raise TimeoutError(f"timed out waiting for subscribe ack on {self._url}")
