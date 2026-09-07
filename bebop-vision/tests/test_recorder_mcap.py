"""Recorder v2 unit tests: MCAP round-trip with fake cameras/robot."""

import base64
import io
import json
import time
from types import SimpleNamespace

from bebop_vision.proto.bebop.runtime.v1 import bebop_runtime_pb2 as pb

import numpy as np
import pytest

from bebop_vision.orbbec import StampedFrame
from bebop_vision.recorder_mcap import NavdRecorder


class FakeCamera:
    """near exercises the RGB re-encode path, far the MJPEG passthrough."""

    def __init__(self, serial, role):
        import collections
        self.serial, self.role = serial, role
        self.mask_rects = []
        self._n = 0
        self._hist = collections.deque(maxlen=6)

    def _make(self):
        self._n += 1
        depth = np.full((480, 848), 1500, np.uint16)
        stamp_us = self._n
        if self.role == "far":
            import cv2
            ok, jpg = cv2.imencode(
                ".jpg", np.full((800, 1280, 3), 128, np.uint8),
                [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            return StampedFrame(depth=depth, stamp_us=stamp_us,
                                recv_ts=time.monotonic(),
                                width=848, height=480, fps=30.0,
                                color=None, color_jpeg=jpg.tobytes(),
                                serial=self.serial, role=self.role)
        color = np.full((800, 1280, 3), 128, np.uint8)
        return StampedFrame(depth=depth, stamp_us=stamp_us,
                            recv_ts=time.monotonic(),
                            width=848, height=480, fps=30.0,
                            color=color, serial=self.serial, role=self.role)

    def read(self):
        f = self._make()
        self._hist.append(f)
        return f

    def recent(self):
        return list(self._hist)


class ScriptedCamera:
    """Camera with preset history: recv_ts/stamp list, freshest last."""

    def __init__(self, serial, role, frames):
        self.serial, self.role = serial, role
        self.mask_rects = []
        self._frames = list(frames)

    def read(self):
        return self._frames[-1]

    def recent(self):
        return list(self._frames)


def _frame(role, serial, stamp_us, recv_ts, color_jpeg=None):
    depth = np.full((480, 848), 1500, np.uint16)
    return StampedFrame(depth=depth, stamp_us=stamp_us, recv_ts=recv_ts,
                        width=848, height=480, fps=15.0,
                        color=None, color_jpeg=color_jpeg,
                        serial=serial, role=role)


class FakeRig:
    def __init__(self):
        self.cameras = {"near": FakeCamera("S-NEAR", "near"),
                        "far": FakeCamera("S-FAR", "far")}


class FakeRobot:
    def __init__(self):
        self.state = SimpleNamespace(cmd=(0.2, -0.1), odom=(0.5, 0.1, 0.02),
                                     connected=True, estop_latched=False,
                                     mode=pb.MODE_RUN_POLICY,
                                     wheel_armed={"left": True, "right": True})


@pytest.fixture
def builder():
    cfg = {
        "cameras": {"S-NEAR": {"role": "near"}, "S-FAR": {"role": "far"}},
        "bev": {"range_m": 3.0, "width_m": 3.0, "cell_m": 0.05,
                "near_authority_m": 1.5, "min_range_m": 0.0},
        "robot": {},
        "safety": {},
    }
    intr = {s: {"fx": 424.0, "fy": 424.0, "cx": 424.0, "cy": 240.0,
                "width": 848, "height": 480} for s in ("S-NEAR", "S-FAR")}
    from bebop_vision.bev import BevBuilder, Mount
    mounts = {s: Mount(1.0, -30.0, 0.0) for s in ("S-NEAR", "S-FAR")}
    return BevBuilder(rig_cfg=cfg, mounts=mounts, intrinsics=intr,
                      inflate_radius_m=0.0, seed=1)


def test_mcap_roundtrip(tmp_path, builder):
    from bebop_vision.goal_planner import GoalSlot, GoalHeading
    rig, robot, slot = FakeRig(), FakeRobot(), GoalSlot()
    slot.set(GoalHeading(0.3))
    path = tmp_path / "session.mcap"
    rec = NavdRecorder(rig, robot, slot, path, builder=builder,
                       rate_hz=20.0, jpeg_quality=80)
    rec.start()
    time.sleep(0.8)
    rec.stop()
    print(f"\n[dbg] frames={rec.frames} bytes={rec.bytes_written}")

    assert path.exists() and path.stat().st_size > 10_000

    from mcap.reader import make_reader
    with open(path, "rb") as f:
        msgs = {}
        for schema, channel, message in make_reader(f).iter_messages():
            payload = message.data
            if channel.message_encoding == "json":
                payload = json.loads(payload)
            msgs.setdefault(channel.topic, []).append(payload)

    # every topic present with a sane number of ticks
    for topic in ("/cmd_vel", "/odom", "/goal", "/bev_teacher",
                  "/color_near", "/color_far",
                  "/depth_near", "/depth_far",
                  "/depth_near_preview", "/depth_far_preview", "/bev_map"):
        assert topic in msgs, f"missing {topic}"
        assert len(msgs[topic]) >= 3, f"{topic}: too few messages"
    assert len(msgs["/calib"]) == 1  # written once at session start
    # images decode
    import cv2
    color_msg = msgs["/color_near"][0]
    assert color_msg["format"] == "jpeg"
    assert color_msg["timestamp"]["sec"] > 1_600_000_000
    jpg = np.frombuffer(base64.b64decode(color_msg["data"]), np.uint8)
    color = cv2.imdecode(jpg, cv2.IMREAD_COLOR)
    assert color is not None and color.shape == (800, 1280, 3)
    color_far_msg = msgs["/color_far"][0]
    assert color_far_msg["format"] == "jpeg"
    assert color_far_msg["frame_id"] == "far_color"
    # far rides the passthrough path: MCAP payload == camera JPEG bytes
    assert base64.b64decode(color_far_msg["data"]) == \
        rig.cameras["far"].read().color_jpeg
    jpg_far = np.frombuffer(base64.b64decode(color_far_msg["data"]), np.uint8)
    color_far = cv2.imdecode(jpg_far, cv2.IMREAD_COLOR)
    assert color_far is not None and color_far.shape == (800, 1280, 3)
    png_msg = msgs["/depth_near"][0]
    assert png_msg["format"] == "png"
    png = np.frombuffer(base64.b64decode(png_msg["data"]), np.uint8)
    depth = cv2.imdecode(png, cv2.IMREAD_UNCHANGED)
    assert depth.dtype == np.uint16 and depth.shape == (480, 848)
    assert (depth == 1500).all()
    prev = msgs["/depth_near_preview"][0]
    assert prev["encoding"] == "16UC1" and prev["width"] == 106
    prev_arr = np.frombuffer(base64.b64decode(prev["data"]), np.uint16)
    assert prev_arr.size == prev["width"] * prev["height"]
    bev_map = msgs["/bev_map"][0]
    assert bev_map["encoding"] == "rgb8"
    map_arr = np.frombuffer(base64.b64decode(bev_map["data"]), np.uint8)
    assert map_arr.size == bev_map["width"] * bev_map["height"] * 3
    # state payloads decode and carry the fake robot's values
    cmd = msgs["/cmd_vel"][0]
    assert cmd["vx"] == pytest.approx(0.2)
    assert cmd["wz"] == pytest.approx(-0.1)
    goal = msgs["/goal"][0]
    assert goal["type"] == "heading"
    assert goal["heading_rad"] == pytest.approx(0.3)
    bev = msgs["/bev_teacher"][0]
    grid = np.frombuffer(__import__("base64").b64decode(bev["raw"]), np.uint8)
    assert grid.shape == (60 * 60,)
    assert set(bev["plane_ok"]) == {"near", "far"}
    calib = msgs["/calib"][0]
    assert calib["intrinsics"]["S-NEAR"]["fx"] == pytest.approx(424.0)
    # ticks share log_time across channels (extractor alignment contract)
    with open(path, "rb") as f:
        times = {}
        for schema, channel, message in make_reader(f).iter_messages():
            times.setdefault(message.log_time, set()).add(channel.topic)
    aligned = [t for t, topics in times.items()
               if {"/depth_near", "/cmd_vel"} <= topics]
    assert aligned, "no tick aligned across topics"


def test_extractor_layout(tmp_path, builder):
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
    from tools.mcap_extract import extract
    from bebop_vision.goal_planner import GoalSlot
    rig, robot, slot = FakeRig(), FakeRobot(), GoalSlot()
    path = tmp_path / "session.mcap"
    rec = NavdRecorder(rig, robot, slot, path, builder=builder, rate_hz=20.0)
    rec.start()
    time.sleep(0.6)
    rec.stop()

    out = tmp_path / "navd-v0" / "s01"
    rows = extract(str(path), str(out))
    assert len(rows) >= 5
    assert (out / "manifest.jsonl").exists()
    row = rows[0]
    d = out / "depth" / f"{row['stamp_ns']:020d}.npz"
    assert d.exists()
    data = np.load(d)
    assert data["near"].shape == (480, 848)
    lab = np.load(out / "labels" / f"{row['stamp_ns']:020d}.npz")
    assert lab["teacher"].shape == (60, 60)
    assert row["cmd_vel"]["vx"] == pytest.approx(0.2)
    assert row["has_color"] is True and row["has_color_far"] is True
    import cv2
    for sub in ("color", "color_far"):
        img = cv2.imread(str(out / sub / f"{row['stamp_ns']:020d}.jpg"))
        assert img is not None and img.shape == (800, 1280, 3), sub


def test_on_grid_callback(tmp_path, builder):
    """The parent's live-BEV hook fires per tick with (grid, goal)."""
    from bebop_vision.goal_planner import GoalSlot, GoalHeading
    rig, robot, slot = FakeRig(), FakeRobot(), GoalSlot()
    slot.set(GoalHeading(0.3))
    calls = []
    path = tmp_path / "session.mcap"
    rec = NavdRecorder(rig, robot, slot, path, builder=builder,
                       rate_hz=20.0,
                       on_grid=lambda grid, goal: calls.append((grid, goal)))
    rec.start()
    time.sleep(0.6)
    rec.stop()
    assert len(calls) >= 3
    grid, goal = calls[-1]
    assert grid is not None and grid is rec.grid
    assert goal is slot.get()


def test_model_grid_recorded(tmp_path, builder):
    """--navd-model: the grid the planner drove on lands in /bev_model with
    its provider, alongside the geometric /bev_teacher (§7.3 A/B)."""
    import base64
    from bebop_vision.bev import BevGrid
    from bebop_vision.goal_planner import GoalSlot, GoalHeading
    rig, robot, slot = FakeRig(), FakeRobot(), GoalSlot()
    slot.set(GoalHeading(0.3))
    mgrid = BevGrid(occ=np.full((60, 60), 1, np.uint8),
                    raw=np.zeros((60, 60), np.uint8), stamp_us=42,
                    per_camera_age_s={}, plane_ok={}, roles=["navd"],
                    cell_m=0.05, recv_ts=time.monotonic())
    path = tmp_path / "session.mcap"
    rec = NavdRecorder(rig, robot, slot, path, builder=builder,
                       rate_hz=20.0,
                       model_grid_fn=lambda: (mgrid, "navd"))
    rec.start()
    time.sleep(0.6)
    rec.stop()

    from mcap.reader import make_reader
    topics = {}
    with open(path, "rb") as f:
        for schema, channel, message in make_reader(f).iter_messages():
            if channel.message_encoding == "json":
                topics.setdefault(channel.topic, []).append(
                    json.loads(message.data))
    assert "/bev_model" in topics and len(topics["/bev_model"]) >= 3
    for m in topics["/bev_model"]:
        assert m["provider"] == "navd"
        raw = np.frombuffer(base64.b64decode(m["raw"]), np.uint8)
        assert raw.shape == (60 * 60,)


def test_prune_sessions(tmp_path):
    from main import _prune_sessions
    for i, size in enumerate((300, 200, 100)):
        p = tmp_path / f"navd_session_s{i}.mcap"
        p.write_bytes(b"x" * size)
        import os
        os.utime(p, (time.time() + i, time.time() + i))  # s0 oldest
    # the firmware's policy captures in the same dir must never be touched
    policy = tmp_path / "policy_capture_x.mcap"
    policy.write_bytes(b"y" * 9999)
    _prune_sessions(tmp_path, budget_bytes=350)
    remaining = sorted(p.name for p in tmp_path.glob("*.mcap"))
    assert remaining == ["navd_session_s1.mcap", "navd_session_s2.mcap",
                         "policy_capture_x.mcap"]  # oldest navd pruned first


def test_pair_frames_matches_capture_instants(tmp_path, builder):
    """Freshest same-instant pair wins over two independent latest reads.

    Scenario (the measured 76 ms case): near slot is fresh, far slot is
    ~66 ms stale. Pairing must fall back to near's previous frame to match
    far's freshest instant instead of storing a one-period-skewed pair.
    """
    from bebop_vision.goal_planner import GoalSlot
    now = time.monotonic()
    rig = SimpleNamespace(cameras={
        "near": ScriptedCamera("S-NEAR", "near", [
            _frame("near", "S-NEAR", 100, now - 0.069),
            _frame("near", "S-NEAR", 101, now - 0.002)]),
        "far": ScriptedCamera("S-FAR", "far", [
            _frame("far", "S-FAR", 201, now - 0.133),
            _frame("far", "S-FAR", 202, now - 0.066)]),
    })
    rec = NavdRecorder(rig, FakeRobot(), GoalSlot(), tmp_path / "s.mcap",
                       builder=builder, rate_hz=20.0)
    frames, pair_ms = rec._pair_frames(max_age_s=0.3)
    assert frames["near"].stamp_us == 100  # not the freshest near frame
    assert frames["far"].stamp_us == 202
    assert pair_ms["far"] < 5.0  # was 64 ms apart via naive latest-reads
    rec.stop()


def test_pair_frames_single_camera(tmp_path, builder):
    """Near-only rig (roles=("near",)) records without pairing metadata."""
    from bebop_vision.goal_planner import GoalSlot
    now = time.monotonic()
    rig = SimpleNamespace(cameras={
        "near": ScriptedCamera("S-NEAR", "near", [
            _frame("near", "S-NEAR", 101, now - 0.002)]),
    })
    rec = NavdRecorder(rig, FakeRobot(), GoalSlot(), tmp_path / "s.mcap",
                       builder=builder, rate_hz=20.0)
    frames, pair_ms = rec._pair_frames(max_age_s=0.3)
    assert frames["near"].stamp_us == 101 and pair_ms == {}
    rec.stop()


def test_drive_active():
    from main import _drive_active
    from bebop_vision.proto.bebop.runtime.v1 import bebop_runtime_pb2 as pb
    robot = FakeRobot()
    robot.state.wheel_armed = {"left": True, "right": True}
    robot.state.mode = pb.MODE_RUN_POLICY
    assert _drive_active(robot)
    # a single armed wheel is a failed enable, not a drivable state
    robot.state.wheel_armed = {"left": True, "right": False}
    assert not _drive_active(robot)
    robot.state.wheel_armed = {"left": False, "right": True}
    assert not _drive_active(robot)
    robot.state.wheel_armed = {"left": True, "right": True}
    robot.state.estop_latched = True
    assert not _drive_active(robot)
    robot.state.estop_latched = False
    robot.state.mode = pb.MODE_IDLE
    assert not _drive_active(robot)
    robot.state.mode = pb.MODE_RUN_POLICY
    robot.state.connected = False
    assert not _drive_active(robot)
