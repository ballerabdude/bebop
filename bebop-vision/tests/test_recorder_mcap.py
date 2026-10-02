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


def test_mcap_roundtrip(tmp_path):
    rig, robot = FakeRig(), FakeRobot()
    path = tmp_path / "session.mcap"
    rec = NavdRecorder(rig, robot, path, rate_hz=20.0,
                       jpeg_quality=80, color_codec=None)
    rec.start()
    time.sleep(0.8)
    rec.stop()
    print(f"\n[dbg] frames={rec.frames} bytes={rec.bytes_written}")

    assert path.exists() and path.stat().st_size > 10_000

    from mcap.reader import make_reader
    from bebop_vision import foxglove_proto as fp
    from bebop_vision import navd_proto
    _PROTO = {m.DESCRIPTOR.full_name: m for m in
              (fp.CompressedImage, fp.CompressedVideo, fp.RawImage,
               navd_proto.Twist, navd_proto.Odom)}
    with open(path, "rb") as f:
        msgs = {}
        for schema, channel, message in make_reader(f).iter_messages():
            data = message.data
            if channel.message_encoding == "protobuf":
                data = _PROTO[schema.name].FromString(data)
            elif channel.message_encoding == "json":
                data = json.loads(data)
            msgs.setdefault(channel.topic, []).append(data)

    # every topic present with a sane number of ticks
    for topic in ("/cmd_vel", "/odom",
                  "/color_near", "/color_far",
                  "/depth_near", "/depth_far"):
        assert topic in msgs, f"missing {topic}"
        assert len(msgs[topic]) >= 3, f"{topic}: too few messages"
    assert "/depth_near_preview" not in msgs
    assert len(msgs["/calib"]) == 1  # written once at session start
    # images decode (protobuf-encoded foxglove messages)
    import cv2
    color_msg = msgs["/color_near"][0]
    assert color_msg.format == "jpeg"
    assert color_msg.timestamp.seconds > 1_600_000_000
    jpg = np.frombuffer(color_msg.data, np.uint8)
    color = cv2.imdecode(jpg, cv2.IMREAD_COLOR)
    assert color is not None and color.shape == (800, 1280, 3)
    color_far_msg = msgs["/color_far"][0]
    assert color_far_msg.format == "jpeg"
    assert color_far_msg.frame_id == "far_color"
    # far rides the passthrough path: MCAP payload == camera JPEG bytes
    assert color_far_msg.data == rig.cameras["far"].read().color_jpeg
    jpg_far = np.frombuffer(color_far_msg.data, np.uint8)
    color_far = cv2.imdecode(jpg_far, cv2.IMREAD_COLOR)
    assert color_far is not None and color_far.shape == (800, 1280, 3)
    png_msg = msgs["/depth_near"][0]
    assert png_msg.format == "png"
    png = np.frombuffer(png_msg.data, np.uint8)
    depth = cv2.imdecode(png, cv2.IMREAD_UNCHANGED)
    assert depth.dtype == np.uint16 and depth.shape == (480, 848)
    assert (depth == 1500).all()
    # telemetry decodes and carries the fake robot's values
    cmd = msgs["/cmd_vel"][0]
    assert cmd.vx == pytest.approx(0.2)
    assert cmd.wz == pytest.approx(-0.1)
    assert "/bev_teacher" not in msgs and "/bev_map" not in msgs \
        and "/goal" not in msgs
    calib = msgs["/calib"][0]
    from bebop_vision.orbbec import load_rig_config
    assert set(calib["mounts"]) == set(load_rig_config()["robots"]["default"]["cameras"])
    # ticks share log_time across channels (extractor alignment contract)
    with open(path, "rb") as f:
        times = {}
        for schema, channel, message in make_reader(f).iter_messages():
            times.setdefault(message.log_time, set()).add(channel.topic)
    aligned = [t for t, topics in times.items()
               if {"/depth_near", "/cmd_vel"} <= topics]
    assert aligned, "no tick aligned across topics"


def test_extractor_layout(tmp_path):
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
    from tools.mcap_extract import extract
    rig, robot = FakeRig(), FakeRobot()
    path = tmp_path / "session.mcap"
    rec = NavdRecorder(rig, robot, path, rate_hz=20.0)
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
    assert "teacher" not in lab.files   # no BEV builder -> no recorded teacher
    assert row["cmd_vel"]["vx"] == pytest.approx(0.2)
    assert row["has_color"] is True and row["has_color_far"] is True
    import cv2
    for sub in ("color", "color_far"):
        img = cv2.imread(str(out / sub / f"{row['stamp_ns']:020d}.jpg"))
        assert img is not None and img.shape == (800, 1280, 3), sub


def test_h265_parameter_set_injector():
    """Every keyframe AU must carry VPS/SPS/PPS so a decode starting at it is
    self-contained (Foxglove seek / GOP replay); NVENC only emits them once."""
    from bebop_vision.hw_video import ParameterSetInjector

    def nalu(nal_type):
        # Annex-B start code + 2-byte H.265 header (type<<1, tid_plus1=1)
        return b"\x00\x00\x00\x01" + bytes([(nal_type << 1) & 0xFE, 0x01]) + b"\xaa"

    vps, sps, pps = nalu(32), nalu(33), nalu(34)
    idr, cra, trail = nalu(19), nalu(21), nalu(1)
    params = vps + sps + pps
    inj = ParameterSetInjector("h265")

    assert inj.process(params + idr) == params + idr  # first IDR cached as-is
    assert inj.process(trail) == trail                # delta untouched
    fixed = inj.process(cra)                          # later CRA made whole
    assert fixed == params + cra
    assert inj.process(params + cra) == params + cra  # no duplication
    # an op that starts mid-GOP matters only at keyframes; P-frames pass through
    assert inj.process(nalu(1)) == nalu(1)


def test_mcap_h265_hardware(tmp_path):
    """Hardware NVENC color path: MCAP carries CompressedVideo H.265 and the
    extractor decodes it back to per-tick training JPEGs. Skipped off-Thor."""
    pytest.importorskip("bebop_vision.hw_video")
    from bebop_vision.hw_video import encoder_available
    if not encoder_available("h265"):
        pytest.skip("no nvv4l2h265enc on this host")

    rig, robot = FakeRig(), FakeRobot()
    path = tmp_path / "session.mcap"
    rec = NavdRecorder(rig, robot, path, rate_hz=20.0, color_codec="h265")
    assert rec._video_codec == "h265"
    rec.start()
    time.sleep(0.8)
    rec.stop()

    from mcap.reader import make_reader
    from bebop_vision import foxglove_proto as fp
    with open(path, "rb") as f:
        color = [fp.CompressedVideo.FromString(m.data)
                 for s, c, m in make_reader(f).iter_messages()
                 if c.topic == "/color_near"]
    assert color and color[0].format == "h265"
    # H.265 access units are start-code prefixed (00 00 00 01 / 00 00 01)
    au = color[0].data
    assert au[:4] == b"\x00\x00\x00\x01" or au[:3] == b"\x00\x00\x01"

    # Every keyframe AU is self-contained (VPS/SPS/PPS) for Foxglove seeking.
    from bebop_vision.hw_video import (_IRAP_TYPES, _PARAM_SET_TYPES,
                                       _iter_annexb_nalus, _nalu_type)
    irap, param = _IRAP_TYPES["h265"], _PARAM_SET_TYPES["h265"]
    keyframes = 0
    for msg in color:
        raw = msg.data
        types = {_nalu_type(raw, hdr, "h265") for _, _, hdr in _iter_annexb_nalus(raw)}
        if types & irap:
            keyframes += 1
            assert (types & param) == param, "keyframe AU missing VPS/SPS/PPS"
    assert keyframes >= 1  # at least the opening IDR

    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
    from tools.mcap_extract import extract
    out = tmp_path / "navd-v0" / "s01"
    rows = extract(str(path), str(out))
    assert len(rows) >= 5
    import cv2
    for row in rows[:5]:
        img = cv2.imread(str(out / "color" / f"{row['stamp_ns']:020d}.jpg"))
        assert img is not None and img.shape == (800, 1280, 3)


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


def test_pair_frames_matches_capture_instants(tmp_path):
    """Freshest same-instant pair wins over two independent latest reads.

    Scenario (the measured 76 ms case): near slot is fresh, far slot is
    ~66 ms stale. Pairing must fall back to near's previous frame to match
    far's freshest instant instead of storing a one-period-skewed pair.
    """
    now = time.monotonic()
    rig = SimpleNamespace(cameras={
        "near": ScriptedCamera("S-NEAR", "near", [
            _frame("near", "S-NEAR", 100, now - 0.069),
            _frame("near", "S-NEAR", 101, now - 0.002)]),
        "far": ScriptedCamera("S-FAR", "far", [
            _frame("far", "S-FAR", 201, now - 0.133),
            _frame("far", "S-FAR", 202, now - 0.066)]),
    })
    rec = NavdRecorder(rig, FakeRobot(), tmp_path / "s.mcap",
                       rate_hz=20.0)
    frames, pair_ms = rec._pair_frames(max_age_s=0.3)
    assert frames["near"].stamp_us == 100  # not the freshest near frame
    assert frames["far"].stamp_us == 202
    assert pair_ms["far"] < 5.0  # was 64 ms apart via naive latest-reads
    rec.stop()


def test_pair_frames_single_camera(tmp_path):
    """Near-only rig (roles=("near",)) records without pairing metadata."""
    now = time.monotonic()
    rig = SimpleNamespace(cameras={
        "near": ScriptedCamera("S-NEAR", "near", [
            _frame("near", "S-NEAR", 101, now - 0.002)]),
    })
    rec = NavdRecorder(rig, FakeRobot(), tmp_path / "s.mcap",
                       rate_hz=20.0)
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
