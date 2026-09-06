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
        self.serial, self.role = serial, role
        self.mask_rects = []
        self._n = 0

    def read(self):
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
