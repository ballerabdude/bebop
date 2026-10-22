"""NavDrive loop + HTTP tests with a fake robot and fake rig."""

import json
import math
import threading
import time
import urllib.request
from types import SimpleNamespace

import pytest

from bebop_vision.nav import (BearingController, ConeConfig, ControllerConfig,
                              DepthCone, NavDrive, NavServer)
from bebop_vision.nav.synth import render_depth, small_geometry
from bebop_vision.proto.bebop.runtime.v1 import bebop_runtime_pb2 as pb
from bebop_vision.robot import RobotState


class FakeRobot:
    """Mirrors RobotClient's surface; integrates twists into odometry."""

    def __init__(self):
        self.state = RobotState()
        self.state.connected = True
        self.state.mode = pb.MODE_IDLE
        self.state.wheel_armed = {"left": False, "right": False}
        self.state.last_telemetry_ts = time.monotonic()
        self.twists = []
        self.calls = []
        self._lock = threading.Lock()
        self._last = time.monotonic()
        self.arm_responds = True

    def _tick(self):
        now = time.monotonic()
        self.state.last_telemetry_ts = now
        if self.twists:
            vx, wz = self.twists[-1]
            dt = now - self._last
            x, y, th = self.state.odom
            if all(self.state.wheel_armed.values()):
                self.state.odom = (x + vx * math.cos(th) * dt,
                                   y + vx * math.sin(th) * dt, th + wz * dt)
        self._last = now

    def send_twist(self, vx, wz):
        with self._lock:
            self._tick()
            self.twists.append((vx, wz))
        return True

    def set_mode(self, mode):
        self.calls.append(("mode", mode))
        self.state.mode = mode
        return True

    def arm_wheels(self, enabled):
        self.calls.append(("arm", enabled))
        if self.arm_responds or not enabled:
            self.state.wheel_armed = {"left": enabled, "right": enabled}
        return True

    def stop(self):
        pass


class FakeCam:
    def __init__(self, depth):
        self.depth = depth

    def read(self):
        return SimpleNamespace(depth=self.depth, depth_scale_mm=1.0,
                               stamp_us=1, recv_ts=time.monotonic())


class FakeRig:
    def __init__(self, cams):
        self.cameras = cams


def make_drive(robot=None, depth=None, **ctrl):
    g = small_geometry()
    depth = render_depth(g) if depth is None else depth
    cone = DepthCone([g], ConeConfig(footprint_m=(-0.13, 0.32, -0.34, 0.19),
                                     stride=2))
    robot = robot or FakeRobot()
    ctl = BearingController(ControllerConfig(**ctrl))
    drive = NavDrive(FakeRig({"near": FakeCam(depth)}), robot, cone, ctl,
                     hz=30.0, log=lambda *a: None)
    return drive, robot


def wait_for(pred, timeout=5.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.02)
    return False


def test_goal_arms_drives_reaches_and_disarms():
    drive, robot = make_drive()
    drive.start()
    try:
        assert wait_for(lambda: drive.clearance() is not None)
        drive.set_goal(bearing_deg=0.0, distance_m=0.6)
        assert wait_for(lambda: drive.status()["state"] == "reached", 8.0)
    finally:
        drive.stop()
    st = drive.status()
    assert st["active"] is False
    assert st["armed_by_nav"] is False
    assert robot.twists[-1] == (0.0, 0.0)
    assert any(vx > 0 for vx, _ in robot.twists)
    assert ("mode", pb.MODE_DIAL_IN) in robot.calls
    assert ("arm", True) in robot.calls and ("arm", False) in robot.calls
    # Mode restored to what it was (IDLE) and wheels left disarmed.
    assert robot.state.mode == pb.MODE_IDLE
    assert not any(robot.state.wheel_armed.values())
    assert st["history"][-1]["state"] == "reached"
    assert st["history"][-1]["goal"]["x"] == pytest.approx(0.6, abs=1e-6)
    assert st["history"][-1]["travelled_m"] > 0.4


def test_estop_mid_drive_aborts_and_stops():
    drive, robot = make_drive()
    drive.start()
    try:
        assert wait_for(lambda: drive.clearance() is not None)
        drive.set_goal(bearing_deg=0.0, distance_m=1.5)
        assert wait_for(lambda: any(vx > 0 for vx, _ in robot.twists))
        robot.state.estop_latched = True
        robot.state.estop_reason = "test"
        assert wait_for(lambda: drive.status()["state"] == "aborted_estop")
    finally:
        drive.stop()
    assert robot.twists[-1] == (0.0, 0.0)
    assert drive.status()["history"][-1]["state"] == "aborted_estop"
    # A new goal is refused while the E-STOP is latched.
    with pytest.raises(RuntimeError):
        drive.set_goal(bearing_deg=0.0, distance_m=0.5)


def test_cancel_stops_and_releases():
    drive, robot = make_drive()
    drive.start()
    try:
        assert wait_for(lambda: drive.clearance() is not None)
        drive.set_goal(bearing_deg=0.0, distance_m=3.0)
        assert wait_for(lambda: drive.status()["armed_by_nav"])
        drive.cancel()
        assert wait_for(lambda: drive.status()["state"] == "stopped")
    finally:
        drive.stop()
    assert robot.twists[-1] == (0.0, 0.0)
    assert not any(robot.state.wheel_armed.values())


def test_arm_failure_aborts_without_moving():
    robot = FakeRobot()
    robot.arm_responds = False
    drive, _ = make_drive(robot=robot)
    import bebop_vision.nav.drive as drive_mod
    old = drive_mod.ARM_WAIT_S
    drive_mod.ARM_WAIT_S = 0.3
    drive.start()
    try:
        assert wait_for(lambda: drive.clearance() is not None)
        drive.set_goal(bearing_deg=0.0, distance_m=1.0)
        assert wait_for(lambda: drive.status()["state"] == "aborted_arm")
    finally:
        drive_mod.ARM_WAIT_S = old
        drive.stop()
    assert all(vx == 0.0 for vx, _ in robot.twists)
    assert drive.status()["arm_error"] == "wheels did not arm"
    # The mode we switched is restored even though arming failed.
    assert robot.state.mode == pb.MODE_IDLE


def test_estop_during_arming_is_reported_and_mode_restored():
    robot = FakeRobot()
    robot.arm_responds = False

    def latch_on_arm(enabled):
        robot.calls.append(("arm", enabled))
        if enabled:
            robot.state.estop_latched = True
            robot.state.estop_reason = "CAN bus can4: failed"
    robot.arm_wheels = latch_on_arm
    drive, _ = make_drive(robot=robot)
    drive.start()
    try:
        assert wait_for(lambda: drive.clearance() is not None)
        drive.set_goal(bearing_deg=0.0, distance_m=1.0)
        assert wait_for(lambda: drive.status()["state"] == "aborted_arm")
    finally:
        drive.stop()
    assert "CAN bus" in drive.status()["arm_error"]
    assert robot.state.mode == pb.MODE_IDLE
    assert all(vx == 0.0 for vx, _ in robot.twists)


def test_blocked_wall_never_moves_forward():
    g = small_geometry()
    depth = render_depth(g, walls=[(0.5, -2, 2, 0.8)])
    drive, robot = make_drive(depth=depth, blocked_timeout_s=0.5)
    drive.start()
    try:
        assert wait_for(lambda: drive.clearance() is not None)
        drive.set_goal(bearing_deg=0.0, distance_m=1.5)
        assert wait_for(lambda: drive.status()["state"] == "aborted_blocked",
                        6.0)
    finally:
        drive.stop()
    assert all(vx == 0.0 for vx, _ in robot.twists)


def test_stale_telemetry_aborts():
    drive, robot = make_drive()
    drive.max_telemetry_age_s = 0.2
    drive.start()
    try:
        assert wait_for(lambda: drive.clearance() is not None)
        drive.set_goal(bearing_deg=0.0, distance_m=2.0)
        assert wait_for(lambda: drive.status()["armed_by_nav"])
        # Freeze telemetry: FakeRobot only stamps on send_twist, so stop
        # the loop from refreshing it by making the robot look silent.
        robot._tick = lambda: None
        robot.state.last_telemetry_ts = time.monotonic() - 1.0
        assert wait_for(lambda: drive.status()["state"] == "aborted_link")
    finally:
        drive.stop()


def test_http_goal_status_clearance_and_stop():
    drive, robot = make_drive()
    drive.start()
    server = NavServer(drive, host="127.0.0.1", port=0).start()
    base = f"http://127.0.0.1:{server.bound_port}"

    def call(path, body=None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(
            base + path, data=data,
            headers={"Content-Type": "application/json"} if data else {},
            method="POST" if data is not None else "GET")
        try:
            with urllib.request.urlopen(req, timeout=3) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    try:
        assert wait_for(lambda: drive.clearance() is not None)
        code, body = call("/healthz")
        assert code == 200 and body["service"] == "nav"
        code, body = call("/clearance")
        assert code == 200 and len(body["clearance_m"]) == 15
        code, body = call("/goal", {"bearing_deg": "x"})
        assert code == 400
        code, body = call("/goal", {"bearing_deg": 0, "distance_m": 3.0,
                                    "label": "test"})
        assert code == 202 and body["goal"]["label"] == "test"
        assert wait_for(lambda: call("/status")[1]["active"])
        code, body = call("/stop", {})
        assert code == 200
        assert wait_for(lambda: call("/status")[1]["state"] == "stopped")
        code, _ = call("/nope")
        assert code == 404
    finally:
        server.stop()
        drive.stop()
    assert robot.twists[-1] == (0.0, 0.0)


def test_reverse_goal_over_http():
    drive, robot = make_drive()
    drive.start()
    try:
        assert wait_for(lambda: drive.clearance() is not None)
        g = drive.set_goal(reverse_m=0.4, label="back")
        assert g["reverse_m"] == pytest.approx(0.4)
        assert wait_for(lambda: drive.status()["state"] == "reached", 8.0)
    finally:
        drive.stop()
    assert any(vx < 0 for vx, _ in robot.twists)
    assert all(vx <= 0 for vx, _ in robot.twists)
    assert robot.state.odom[0] < -0.25
    with pytest.raises(ValueError):
        drive.set_goal(reverse_m=-1)


def test_turn_goal_over_the_drive():
    drive, robot = make_drive()
    drive.start()
    try:
        assert wait_for(lambda: drive.clearance() is not None)
        g = drive.set_goal(turn_deg=60, label="turn")
        assert g["rotate_to_deg"] == pytest.approx(60, abs=0.1)
        assert wait_for(lambda: drive.status()["state"] == "reached", 8.0)
    finally:
        drive.stop()
    assert all(vx == 0.0 for vx, _ in robot.twists)
    assert any(wz > 0 for _, wz in robot.twists)
    assert math.degrees(robot.state.odom[2]) == pytest.approx(60, abs=5)
    with pytest.raises(ValueError):
        drive.set_goal(turn_deg=0.5)
