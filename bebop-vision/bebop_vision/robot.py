"""Robot client: protobuf-over-WebSocket bridge to the bebop-linux runtime API.

Speaks the same wire protocol as the operator app: one ClientRuntimeMessage /
ServerRuntimeMessage per binary WS frame on GET /ws.
"""

import asyncio
import threading
import time

from .proto.bebop.runtime.v1 import bebop_runtime_pb2 as pb

try:
    import websockets
except ImportError as exc:  # pragma: no cover
    raise ImportError("pip install websockets") from exc

DEFAULT_URL = "ws://bebop.local:9090/ws"


class RobotState:
    """Latest robot state mirrored from telemetry. Attr reads are GIL-atomic."""

    def __init__(self):
        self.connected = False
        self.mode = pb.MODE_IDLE
        self.estop_latched = False
        self.estop_reason = ""
        self.cmd = (0.0, 0.0)
        self.odom = (0.0, 0.0, 0.0)
        self.wheel_armed = {}
        self.wheel_vel = {}
        self.wheel_feedback_stale = {}
        self.battery_v = None
        self.camera_present = False
        self.camera_pan_deg = 0.0
        self.camera_tilt_deg = 0.0
        self.camera_moving = False
        self.telemetry_hz = 0.0
        self.last_telemetry_ts = 0.0


class RobotClient:
    """Async WS client on a background thread; thread-safe command methods."""

    def __init__(self, url=DEFAULT_URL, telemetry_hz=30, reconnect_s=1.0, name="bebop-vision"):
        self.url = url
        self.telemetry_hz = telemetry_hz
        self.reconnect_s = reconnect_s
        self.name = name
        self.state = RobotState()
        self.on_mode_changed = []
        self.on_estop = []
        self._loop = None
        self._thread = None
        self._stop_evt = threading.Event()
        self._outbox = None
        self._req_id = 0
        self._tel_frames = 0
        self._tel_window = time.monotonic()

    def start(self):
        if self._thread is not None:
            return self
        self._thread = threading.Thread(target=self._run, daemon=True, name="robot-ws")
        self._thread.start()
        return self

    def stop(self, timeout=3.0):
        self._stop_evt.set()
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._outbox_put, self._twist_msg(0.0, 0.0))
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def _run(self):
        asyncio.run(self._main())

    async def _main(self):
        self._loop = asyncio.get_running_loop()
        while not self._stop_evt.is_set():
            try:
                async with websockets.connect(self.url, max_size=2**24) as ws:
                    self.state.connected = True
                    print(f"[robot] connected to {self.url}")
                    outbox = asyncio.Queue(maxsize=64)
                    self._outbox = outbox
                    outbox.put_nowait(self._subscribe_msg().SerializeToString())
                    sender = asyncio.create_task(self._sender(ws, outbox))
                    try:
                        await self._pump(ws)
                    finally:
                        sender.cancel()
                        self._outbox = None
            except asyncio.CancelledError:
                return
            except Exception as exc:
                if not self._stop_evt.is_set():
                    print(f"[robot] connection error: {type(exc).__name__}: {exc}")
            self.state.connected = False
            if not self._stop_evt.is_set():
                await asyncio.sleep(self.reconnect_s)

    async def _sender(self, ws, outbox):
        while True:
            data = await outbox.get()
            await ws.send(data)

    async def _pump(self, ws):
        async for raw in ws:
            msg = pb.ServerRuntimeMessage()
            msg.ParseFromString(raw)
            self._handle(msg)
            if self._stop_evt.is_set():
                return

    def _handle(self, msg):
        which = msg.WhichOneof("payload")
        st = self.state
        if which in ("telemetry", "snapshot"):
            self._update_from_fields(msg.telemetry if which == "telemetry" else msg.snapshot)
        elif which == "mode_changed":
            st.mode = msg.mode_changed.mode
            print(f"[robot] mode -> {pb.Mode.Name(st.mode)}")
            for cb in self.on_mode_changed:
                cb(st.mode)
        elif which == "estop_latched":
            st.estop_latched = True
            st.estop_reason = msg.estop_latched.reason
            print(f"[robot] E-STOP latched: {st.estop_reason!r}")
            for cb in self.on_estop:
                cb(st.estop_reason)
        elif which == "error":
            print(f"[robot] server error (req {msg.request_id}): {msg.error.message}")

    def _update_from_fields(self, t):
        st = self.state
        st.mode = t.mode
        st.estop_latched = t.estop_latched
        if t.estop_reason:
            st.estop_reason = t.estop_reason
        if t.drive.present:
            st.cmd = (t.drive.cmd_linear_x, t.drive.cmd_angular_z)
            st.odom = (t.drive.odom_x, t.drive.odom_y, t.drive.odom_theta)
        st.camera_present = t.camera.present
        st.camera_pan_deg = t.camera.pan_deg
        st.camera_tilt_deg = t.camera.tilt_deg
        st.camera_moving = t.camera.moving
        st.wheel_armed = {w.name: w.armed for w in t.wheels}
        st.wheel_vel = {w.name: w.velocity_rad_s for w in t.wheels}
        st.wheel_feedback_stale = {w.name: w.feedback_stale for w in t.wheels}
        if t.power.present:
            st.battery_v = t.power.battery_voltage_v
        st.last_telemetry_ts = time.monotonic()
        self._tel_frames += 1
        now = time.monotonic()
        if now - self._tel_window >= 1.0:
            st.telemetry_hz = self._tel_frames / (now - self._tel_window)
            self._tel_frames = 0
            self._tel_window = now

    # --- outbound helpers ---------------------------------------------------

    def _next_req_id(self):
        self._req_id = (self._req_id + 1) % (2**16)
        return self._req_id

    def _subscribe_msg(self):
        m = pb.ClientRuntimeMessage()
        m.subscribe_telemetry.rate_hz = self.telemetry_hz
        return m

    def _twist_msg(self, vx, wz):
        m = pb.ClientRuntimeMessage()
        m.set_velocity_command.linear_x = float(vx)
        m.set_velocity_command.angular_z = float(wz)
        return m

    def _outbox_put(self, msg):
        if self._outbox is not None:
            self._outbox.put_nowait(msg.SerializeToString())

    def _submit(self, msg):
        loop = self._loop
        if loop is None or self._outbox is None:
            return False
        loop.call_soon_threadsafe(self._outbox_put, msg)
        return True

    # --- thread-safe public commands ---------------------------------------

    def send_twist(self, vx, wz):
        """Command a body-frame twist (m/s forward, rad/s yaw, + left)."""
        return self._submit(self._twist_msg(vx, wz))

    def set_camera_pose(self, pan_deg, tilt_deg):
        """Command the camera gimbal to an absolute pose (degrees).

        Pan + = right, tilt + = up, 0/0 = power-on center. The firmware
        clamps to the camera's limits; not mode-gated, and the settled
        pose arrives back via telemetry (`camera_pan_deg` / `camera_*`
        state fields).
        """
        m = pb.ClientRuntimeMessage()
        m.set_camera_pose.pan_deg = float(pan_deg)
        m.set_camera_pose.tilt_deg = float(tilt_deg)
        return self._submit(m)

    def estop(self, reason="bebop-vision"):
        m = pb.ClientRuntimeMessage()
        m.emergency_stop.reason = reason
        return self._submit(m)

    def reset_estop(self):
        return self._submit(pb.ClientRuntimeMessage(reset_estop=pb.ResetEStop()))

    def set_mode(self, mode):
        m = pb.ClientRuntimeMessage()
        m.set_mode.mode = mode
        return self._submit(m)

    def arm_wheels(self, enabled):
        m = pb.ClientRuntimeMessage()
        m.set_all_wheels_enabled.enabled = bool(enabled)
        return self._submit(m)

    def reset_odometry(self):
        return self._submit(pb.ClientRuntimeMessage(reset_odometry=pb.ResetOdometry()))

    def snapshot(self):
        return self._submit(pb.ClientRuntimeMessage(get_snapshot=pb.GetSnapshot()))

    def await_connection(self, timeout=5.0):
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            if self.state.connected and self.state.last_telemetry_ts > 0:
                return True
            time.sleep(0.05)
        return False

    def describe(self):
        st = self.state
        mode = pb.Mode.Name(st.mode) if st.mode else "?"
        battery = f"{st.battery_v:.1f}V" if st.battery_v else "n/a"
        odom = "({:.2f}, {:.2f}, {:.0f}deg)".format(st.odom[0], st.odom[1],
                                                    __import__("math").degrees(st.odom[2]))
        return (f"mode={mode} estop={st.estop_latched} battery={battery} "
                f"telemetry={st.telemetry_hz:.0f}Hz wheels={st.wheel_armed} odom={odom}")