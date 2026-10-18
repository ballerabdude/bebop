"""Hardware-free tests for the live visual-localization POC.

No camera, no cuVSLAM wheel, no robot runtime, and no wheel odometry.  The
tracker is a deterministic fake; frames are synthetic RGB-D with device
stamps; calibration is a tiny synthetic rig in tmp_path.

The tests encode the *intended* live contract, not whatever a partially
implemented build happens to do:

* a pose is valid only for the current frame/tick (frozen duplicates and a
  blocked worker age out, and both pose and map_pose become null);
* stale, missing, skewed, or serial-mismatched frames never reach the
  tracker;
* tracking loss/error invalidates the pose and recovery starts a new segment;
* corrected/map pose stays separate from continuous odometry;
* poses remain in the camera optical frame (no hidden body conversion);
* the trajectory is bounded and the package never imports wheel odometry.
"""

from __future__ import annotations

import base64
import json
import math
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from bebop_vision.slam.backend import Pose, TrackResult
from bebop_vision.slam.calibration import (CalibrationError, CameraModel,
                                           RigCalibration, SlamConfigError,
                                           load_calibration)
from bebop_vision.slam.depth import DepthRegistration
from bebop_vision.slam.fake import FakeTracker
from bebop_vision.slam.live import (IMU_REJECT_REASON, LiveSlam,
                                    LiveSlamConfig)

ROLE = "far"
SERIAL = "TESTSERIAL"
STAMP = 10_000_000  # microseconds
COLOR_W, COLOR_H = 16, 12
DEPTH_W, DEPTH_H = 8, 6


def _write_rig(tmp_path: Path, *, serial: str = SERIAL, role: str = ROLE,
               rotation=None, translation=(0.0, 0.0, 0.0),
               color_fx: float = 8.0) -> Path:
    cfg = tmp_path / "config"
    cfg.mkdir(exist_ok=True)
    (cfg / "orbbec_rig.yaml").write_text(
        "robots:\n"
        "  default:\n"
        "    cameras:\n"
        f"      {serial}:\n"
        f"        role: {role}\n"
        "        height_m: 1.32\n"
        "        pitch_deg: -17.4\n"
        "        yaw_deg: 0.0\n")
    rot = list(rotation) if rotation is not None else [
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    ]
    (cfg / f"orbbec_intrinsics_{serial}.json").write_text(json.dumps({
        "serial": serial,
        "width": DEPTH_W, "height": DEPTH_H,
        "fx": 4.0, "fy": 4.0, "cx": 4.0, "cy": 3.0,
        "distortion": {"k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0,
                       "k3": 0.0, "k4": 0.0, "k5": 0.0, "k6": 0.0},
        "color_width": COLOR_W, "color_height": COLOR_H,
        "color_fx": color_fx, "color_fy": color_fx,
        "color_cx": COLOR_W / 2.0, "color_cy": COLOR_H / 2.0,
        "color_rgb_distortion": {"k1": 0.0, "k2": 0.0, "p1": 0.0,
                                 "p2": 0.0, "k3": 0.0, "k4": 0.0,
                                 "k5": 0.0, "k6": 0.0},
        "color_to_depth_transform": {
            "rotation": rot,
            "translation": list(translation),
        },
    }))
    return cfg


@pytest.fixture
def calibration(tmp_path) -> RigCalibration:
    return load_calibration(ROLE, _write_rig(tmp_path))


class FakeFrame:
    """The subset of StampedFrame the live path consumes."""

    def __init__(self, stamp_us: int = STAMP, color_stamp_us=None,
                 recv_ts: float | None = None, age_s: float | None = None,
                 serial: str = SERIAL, depth_scale_mm: float = 1.0,
                 depth=None, color=None):
        self.stamp_us = int(stamp_us)
        self.color_stamp_us = (int(stamp_us) if color_stamp_us is None
                               else int(color_stamp_us))
        self.recv_ts = time.monotonic() if recv_ts is None else recv_ts
        self._age_s = age_s
        self.serial = serial
        self.depth_scale_mm = depth_scale_mm
        self.depth = (np.full((DEPTH_H, DEPTH_W), 1000, np.uint16)
                      if depth is None else depth)
        self.color = (np.zeros((COLOR_H, COLOR_W, 3), np.uint8)
                      if color is None else color)
        self.color_jpeg = None

    def age_s(self):
        if self._age_s is not None:
            return self._age_s
        return time.monotonic() - self.recv_ts


class FakeCamera:
    def __init__(self, frame, history=None):
        self.frame = frame
        self.history = list(history) if history is not None else [frame]

    def read(self):
        return self.history[-1] if self.history else self.frame

    def recent(self):
        return list(self.history)


class FakeRig:
    def __init__(self, frame):
        self.cameras = {ROLE: FakeCamera(frame)}


def _pose(t_ns: int = STAMP * 1000, translation=(1.0, 2.0, 3.0),
          rotation=(0.0, 0.0, 0.0, 1.0)) -> Pose:
    return Pose(t_ns, tuple(translation), tuple(rotation))


def _tracked(pose, corrected=None, state="tracking", error=None):
    return TrackResult(pose, corrected, state, error)


def _make(tmp_path, frame, results, *, enable_slam=False,
          capacity=900, max_age=0.3, max_pair_delta_us=5_000,
          enable_point_cloud=True, enable_structure=False):
    cfg = _write_rig(tmp_path)
    cal = load_calibration(ROLE, cfg)
    backend = FakeTracker(results=results)
    slam = LiveSlam(
        FakeRig(frame),
        LiveSlamConfig(role=ROLE, config_dir=cfg, enable_slam=enable_slam,
                       max_frame_age_s=max_age,
                       max_pair_delta_us=max_pair_delta_us,
                       trajectory_capacity=capacity,
                       status_log_interval_s=0,
                       enable_point_cloud=enable_point_cloud,
                       enable_structure=enable_structure),
        backend=backend, calibration=cal)
    return slam, backend, cal


# ---------------------------------------------------------------------------
# calibration / registration
# ---------------------------------------------------------------------------

def test_load_calibration_shapes_and_optical_frame(calibration):
    assert calibration.role == ROLE
    assert calibration.serial == SERIAL
    assert calibration.depth_shape == (DEPTH_H, DEPTH_W)
    assert calibration.color_shape == (COLOR_H, COLOR_W)
    assert calibration.frame_id == "camera_optical"
    assert calibration.T_depth_from_color.shape == (4, 4)
    # Inverse is a proper rigid transform.
    R = calibration.T_color_from_depth[:3, :3]
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-6)
    assert np.linalg.det(R) > 0.9


def test_calibration_rejects_non_rotation(tmp_path):
    cfg = _write_rig(tmp_path, rotation=[2.0, 0.0, 0.0,
                                         0.0, 1.0, 0.0,
                                         0.0, 0.0, 1.0])
    with pytest.raises(CalibrationError):
        load_calibration(ROLE, cfg)


def test_calibration_rejects_missing_role(tmp_path):
    cfg = _write_rig(tmp_path)
    with pytest.raises(CalibrationError):
        load_calibration("near", cfg)


def test_camera_model_validation():
    CameraModel(16, 12, 8.0, 8.0, 8.0, 6.0).validate("ok")
    with pytest.raises(CalibrationError):
        CameraModel(0, 12, 8.0, 8.0, 8.0, 6.0).validate("bad-size")
    with pytest.raises(CalibrationError):
        CameraModel(16, 12, 0.0, 8.0, 8.0, 6.0).validate("bad-fx")
    with pytest.raises(CalibrationError):
        CameraModel(16, 12, 8.0, 8.0, 8.0, 6.0,
                    (0.0,) * 7).validate("bad-dist")


def test_depth_scale_is_applied_once_before_extrinsic(tmp_path):
    cfg = _write_rig(tmp_path, translation=(100.0, 0.0, 0.0))  # 100 mm
    cal = load_calibration(ROLE, cfg)
    reg = DepthRegistration(cal)
    raw = np.full((DEPTH_H, DEPTH_W), 1000, np.uint16)
    # Raw 1000 at scale 1 mm/unit -> 1 m. At 2 mm/unit -> 2 m. The
    # extrinsic translation stays 0.1 m in both cases; it is not scaled.
    one = reg.align_meters(raw, 1.0)
    two = reg.align_meters(raw, 2.0)
    assert np.isfinite(one).any() and np.isfinite(two).any()
    assert reg._metric_depth(raw, 1.0)[0, 0] == pytest.approx(1.0)
    assert reg._metric_depth(raw, 2.0)[0, 0] == pytest.approx(2.0)
    # The aligned output is uint16 millimetres, 0 = invalid.
    mm = reg.align(raw, 1.0)
    assert mm.dtype == np.uint16
    assert mm.shape == (COLOR_H, COLOR_W)


def test_depth_registration_rejects_bad_scale_and_shape(calibration):
    reg = DepthRegistration(calibration)
    raw = np.full((DEPTH_H, DEPTH_W), 1000, np.uint16)
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(CalibrationError):
            reg.align(raw, bad)
    with pytest.raises(CalibrationError):
        reg.align(np.zeros((DEPTH_H + 1, DEPTH_W), np.uint16), 1.0)


# ---------------------------------------------------------------------------
# frame pairing / staleness / dedup
# ---------------------------------------------------------------------------

def test_duplicate_frozen_frame_ages_out_pose(tmp_path):
    frame = FakeFrame()
    slam, backend, _ = _make(tmp_path, frame, [_tracked(
        _pose(), _pose(translation=(9.0, 9.0, 9.0)))],
        enable_slam=True)
    slam._tick()
    st = slam.status()
    assert st["pose_valid"] and st["pose"] is not None
    assert st["map_pose"] is not None

    # Same device stamp: the tracker must not be fed it twice, and once the
    # frozen frame is older than the limit the published pose/map must age
    # out instead of looking current forever.
    frame.recv_ts = time.monotonic() - 0.5
    slam._tick()
    st = slam.status()
    assert len(backend.calls) == 1
    assert st["pose_valid"] is False
    assert st["pose"] is None
    assert st["map_pose"] is None
    assert st["tracking_state"] in ("stale", "invalid", "lost")


def test_status_ages_out_pose_while_worker_blocked(tmp_path):
    frame = FakeFrame()
    slam, _, _ = _make(tmp_path, frame, [_tracked(
        _pose(), _pose(translation=(9.0, 9.0, 9.0)))],
        enable_slam=True)
    slam._tick()
    assert slam.status()["pose_valid"]

    # Simulate the worker being blocked: no new tick, but the status call
    # happens later.  The reader must independently age the pose out.
    slam._pose_recv_ts = time.monotonic() - 1.0
    st = slam.status()
    assert st["pose_valid"] is False
    assert st["pose"] is None
    assert st["map_pose"] is None
    assert st["tracking_state"] in ("stale", "invalid", "lost")


def test_missing_color_stamp_never_tracks(tmp_path):
    frame = FakeFrame(color_stamp_us=0)
    slam, backend, _ = _make(tmp_path, frame, [_tracked(_pose())])
    slam._tick()
    st = slam.status()
    assert backend.calls == []
    assert st["pose_valid"] is False
    assert st["pose"] is None
    assert "color_stamp_us" in (st["last_error"] or "")


def test_color_depth_skew_over_default_is_rejected(tmp_path):
    frame = FakeFrame(color_stamp_us=STAMP + 6_000)  # 6 ms > 5 ms
    slam, backend, _ = _make(tmp_path, frame, [_tracked(_pose())])
    slam._tick()
    st = slam.status()
    assert backend.calls == []
    assert st["pose_valid"] is False
    assert "skew" in (st["last_error"] or "").lower()


def test_default_pair_skew_matches_sdk_phase(tmp_path):
    # The SDK-synchronized color/depth phase reaches 16.6 ms at 30 Hz.
    # The 17 ms default accepts that (half period + timestamp rounding) while
    # still rejecting a full 33 ms adjacent-frame mismatch.
    cfg = LiveSlamConfig()
    assert 16_600 <= cfg.max_pair_delta_us <= 17_000
    assert cfg.max_pair_delta_us < 33_000


def test_default_skew_accepts_16_6ms_and_rejects_33ms(tmp_path):
    cfg_dir = _write_rig(tmp_path)
    cal = load_calibration(ROLE, cfg_dir)

    accepted = FakeFrame(stamp_us=STAMP, color_stamp_us=STAMP + 16_600)
    backend = FakeTracker(results=[_tracked(_pose())])
    slam = LiveSlam(FakeRig(accepted),
                    LiveSlamConfig(role=ROLE, config_dir=cfg_dir,
                                   status_log_interval_s=0),
                    backend=backend, calibration=cal)
    slam._tick()
    assert len(backend.calls) == 1
    assert slam.status()["pose_valid"] is True

    rejected = FakeFrame(stamp_us=STAMP + 100_000,
                         color_stamp_us=STAMP + 100_000 + 33_000)
    backend2 = FakeTracker(results=[_tracked(_pose())])
    slam2 = LiveSlam(FakeRig(rejected),
                     LiveSlamConfig(role=ROLE, config_dir=cfg_dir,
                                    status_log_interval_s=0),
                     backend=backend2, calibration=cal)
    slam2._tick()
    assert backend2.calls == []
    assert slam2.status()["pose_valid"] is False
    assert "skew" in (slam2.status()["last_error"] or "").lower()


def test_stale_frame_is_rejected_without_tracking(tmp_path):
    frame = FakeFrame(age_s=0.5)
    slam, backend, _ = _make(tmp_path, frame, [_tracked(_pose())],
                             max_age=0.3)
    slam._tick()
    st = slam.status()
    assert backend.calls == []
    assert st["pose_valid"] is False
    assert st["tracking_state"] == "stale"


def test_driver_history_skips_newer_depth_only_frame(tmp_path):
    good = FakeFrame(stamp_us=STAMP)
    depth_only = FakeFrame(stamp_us=STAMP + 33_333, color_stamp_us=0,
                           depth_scale_mm=1.0)
    depth_only.color = None
    depth_only.color_jpeg = None
    rig = FakeRig(good)
    rig.cameras[ROLE] = FakeCamera(depth_only, history=[good, depth_only])
    cfg = _write_rig(tmp_path)
    cal = load_calibration(ROLE, cfg)
    backend = FakeTracker(results=[_tracked(_pose())])
    slam = LiveSlam(rig, LiveSlamConfig(role=ROLE, config_dir=cfg,
                                        status_log_interval_s=0),
                    backend=backend, calibration=cal)

    # camera.read() would return the newer depth-only sample; the live path
    # must scan history and pick the older complete RGB-D pair with its
    # ORIGINAL age/stamps.
    selected = slam._read_frame()
    assert selected is good
    slam._tick()
    assert len(backend.calls) == 1
    assert slam.status()["pose_valid"] is True

    # A new complete pair is processed exactly once; a duplicate stamp is not
    # fed to the tracker again.
    newer = FakeFrame(stamp_us=STAMP + 66_666)
    rig.cameras[ROLE].history.append(newer)
    slam._tick()
    slam._tick()
    assert len(backend.calls) == 2


def test_driver_history_no_complete_returns_latest(tmp_path):
    depth_only = FakeFrame(stamp_us=STAMP, color_stamp_us=0)
    depth_only.color = None
    depth_only.color_jpeg = None
    rig = FakeRig(depth_only)
    rig.cameras[ROLE] = FakeCamera(depth_only, history=[depth_only])
    cfg = _write_rig(tmp_path)
    cal = load_calibration(ROLE, cfg)
    slam = LiveSlam(rig, LiveSlamConfig(role=ROLE, config_dir=cfg,
                                        status_log_interval_s=0),
                    backend=FakeTracker(), calibration=cal)
    assert slam._read_frame() is depth_only


def test_serial_mismatch_is_rejected(tmp_path):
    frame = FakeFrame(serial="NOT-THE-CALIBRATED-CAMERA")
    slam, backend, _ = _make(tmp_path, frame, [_tracked(_pose())])
    slam._tick()
    st = slam.status()
    assert backend.calls == []
    assert st["pose_valid"] is False
    assert "serial" in (st["last_error"] or "").lower()


# ---------------------------------------------------------------------------
# tracking validity / segments
# ---------------------------------------------------------------------------

def test_tracking_loss_invalidates_pose_and_recovery_starts_segment(tmp_path):
    frame = FakeFrame()
    slam, backend, _ = _make(tmp_path, frame, [
        TrackResult(None, None, "lost"),
        _tracked(_pose(t_ns=(STAMP + 33_333) * 1000)),
    ])
    slam._tick()
    st = slam.status()
    assert st["pose_valid"] is False
    assert st["pose"] is None
    assert st["tracking_state"] == "lost"
    assert st["segment"] == 0

    frame.stamp_us += 33_333
    frame.color_stamp_us = frame.stamp_us
    slam._tick()
    st = slam.status()
    assert st["pose_valid"] is True
    assert st["pose"] is not None
    assert st["segment"] == 1
    assert len(backend.calls) == 2


def test_backwards_device_clock_errors_and_never_reaches_tracker(tmp_path):
    frame = FakeFrame(stamp_us=STAMP)
    slam, backend, _ = _make(tmp_path, frame, [
        _tracked(_pose()),
        _tracked(_pose(t_ns=1)),
    ])
    slam._tick()
    assert len(backend.calls) == 1

    # A camera restart can move the device clock backwards.  cuVSLAM's
    # odometry frame assumes monotonic stamps, so this must not be fed into
    # the same tracker; it errors and requires a fresh service/tracker.
    frame.stamp_us = STAMP - 2_000_000
    frame.color_stamp_us = frame.stamp_us
    frame.recv_ts = time.monotonic()
    slam._tick()
    st = slam.status()
    assert len(backend.calls) == 1
    assert st["pose_valid"] is False
    assert st["pose"] is None
    assert st["tracking_state"] == "error"
    assert "backwards" in (st["last_error"] or "")

    # Even a later forward stamp cannot silently revive the corrupt tracker.
    frame.stamp_us = STAMP + 33_333
    frame.color_stamp_us = frame.stamp_us
    slam._tick()
    assert len(backend.calls) == 1
    assert slam.status()["tracking_state"] == "error"


def test_backend_error_invalidates_pose(tmp_path):
    frame = FakeFrame()
    slam, _, _ = _make(tmp_path, frame, [
        TrackResult(None, None, "error", "cuvslam exploded")])
    slam._tick()
    st = slam.status()
    assert st["pose_valid"] is False
    assert st["pose"] is None
    assert st["tracking_state"] == "error"
    assert "cuvslam exploded" in (st["last_error"] or "")


def test_non_finite_pose_is_rejected(tmp_path):
    frame = FakeFrame()
    slam, _, _ = _make(tmp_path, frame, [_tracked(
        Pose(STAMP * 1000, (float("nan"), 0.0, 0.0),
             (0.0, 0.0, 0.0, 1.0)))])
    slam._tick()
    st = slam.status()
    assert st["pose_valid"] is False
    assert st["pose"] is None


# ---------------------------------------------------------------------------
# pose frame / odom vs map
# ---------------------------------------------------------------------------

def test_pose_stays_in_camera_optical_frame(tmp_path):
    frame = FakeFrame()
    pose = _pose(translation=(1.0, 2.0, 3.0),
                 rotation=(0.0, 0.0, math.sin(0.2), math.cos(0.2)))
    slam, _, cal = _make(tmp_path, frame, [_tracked(pose)])
    slam._tick()
    st = slam.status()
    assert st["frame_id"] == "camera_optical"
    assert st["pose"]["frame_id"] == "camera_optical"
    # The synthetic rig has a nonzero mount pitch; the package must not
    # silently rotate the pose into a body frame.
    assert st["pose"]["translation"] == pytest.approx([1.0, 2.0, 3.0])
    assert st["calibration"]["frame_id"] == "camera_optical"
    assert cal.frame_id == "camera_optical"


def test_corrected_pose_is_separate_from_odometry(tmp_path):
    frame = FakeFrame()
    odom = _pose(translation=(1.0, 2.0, 3.0))
    corrected = _pose(translation=(9.0, 8.0, 7.0))
    slam, _, _ = _make(tmp_path, frame,
                       [_tracked(odom, corrected)],
                       enable_slam=True)
    slam._tick()
    st = slam.status()
    assert st["pose"]["translation"] == pytest.approx([1.0, 2.0, 3.0])
    assert st["map_pose"]["translation"] == pytest.approx([9.0, 8.0, 7.0])
    assert st["map_pose"]["frame_id"] == "camera_optical"
    traj = slam.trajectory()
    assert traj["points"][-1]["map_pose"]["translation"] == pytest.approx(
        [9.0, 8.0, 7.0])


def test_corrected_pose_ignored_when_slam_disabled(tmp_path):
    frame = FakeFrame()
    slam, _, _ = _make(tmp_path, frame,
                       [_tracked(_pose(), _pose(translation=(9.0, 9.0, 9.0)))],
                       enable_slam=False)
    slam._tick()
    st = slam.status()
    assert st["pose_valid"] is True
    assert st["map_pose"] is None


# ---------------------------------------------------------------------------
# bounded state / no wheel dependency / IMU
# ---------------------------------------------------------------------------

def test_trajectory_is_bounded(tmp_path):
    frame = FakeFrame()
    slam, backend, _ = _make(tmp_path, frame, None, capacity=3)
    for i in range(5):
        frame.stamp_us = STAMP + i * 33_333
        frame.color_stamp_us = frame.stamp_us
        frame.recv_ts = time.monotonic()
        slam._tick()
    st = slam.status()
    assert st["trajectory_points"] == 3
    assert st["trajectory_capacity"] == 3
    traj = slam.trajectory()
    assert traj["count"] == 3
    assert traj["returned"] == 3
    assert len(traj["points"]) == 3
    assert len(backend.calls) == 5


def test_slam_package_has_no_wheel_odometry_dependency():
    pkg = Path(__file__).resolve().parents[1] / "bebop_vision" / "slam"
    for path in pkg.glob("*.py"):
        text = path.read_text()
        assert "from ..robot" not in text, path
        assert "bebop_vision.robot" not in text, path
        assert ".state.odom" not in text, path
        assert "wheel_odom" not in text, path


def test_imu_is_rejected_not_faked():
    with pytest.raises(SlamConfigError) as exc:
        LiveSlamConfig(use_imu=True).validate()
    assert "IMU" in str(exc.value) or "imu" in str(exc.value)
    assert IMU_REJECT_REASON


def test_unknown_role_is_rejected(tmp_path):
    cfg = _write_rig(tmp_path)
    cal = load_calibration(ROLE, cfg)
    with pytest.raises(SlamConfigError):
        LiveSlam(FakeRig(FakeFrame()),
                 LiveSlamConfig(role="near", config_dir=cfg),
                 backend=FakeTracker(), calibration=cal)


# ---------------------------------------------------------------------------
# HTTP status service (optional; the server module may land after live.py)
# ---------------------------------------------------------------------------

def test_status_server_lifecycle_and_read_only(tmp_path):
    from bebop_vision.slam.server import SlamStatusServer
    frame = FakeFrame()
    slam, _, _ = _make(tmp_path, frame, None)
    server = SlamStatusServer(slam, host="127.0.0.1", port=0)
    server.start()
    base = f"http://127.0.0.1:{server.bound_port}"
    try:
        import urllib.error
        import urllib.request

        def get(path):
            with urllib.request.urlopen(base + path, timeout=2.0) as resp:
                return resp.status, json.loads(resp.read().decode())

        code, body = get("/status")
        assert code == 200
        assert body["service"] == "live-slam"
        assert body["frame_id"] == "camera_optical"
        assert body["pose_frame"] == "world_from_camera_optical"
        assert "pose" in body and "map_pose" in body
        assert body["imu"]["enabled"] is False

        code, body = get("/healthz")
        assert code == 200 and body["service"] == "live-slam"
        code, body = get("/pose")
        assert code == 200 and body["valid"] is False
        code, body = get("/trajectory?limit=2")
        assert code == 200 and body["capacity"] == 900

        with pytest.raises(urllib.error.HTTPError) as exc:
            get("/nope")
        assert exc.value.code == 404

        req = urllib.request.Request(base + "/status", data=b"",
                                     method="POST")
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(req, timeout=2.0)
        assert exc.value.code == 405
    finally:
        server.stop()

    # Idempotent stop, and the port is released.
    server.stop()
    with pytest.raises(OSError):
        urllib.request.urlopen(base + "/status", timeout=0.5)


# ---------------------------------------------------------------------------
# manual turn diagnostic helpers (no hardware, no robot connection)
# ---------------------------------------------------------------------------

def _turn_tool():
    import importlib.util
    path = Path(__file__).resolve().parents[1] / "tools" / "slam_turn_test.py"
    spec = importlib.util.spec_from_file_location("slam_turn_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_turn_targets_are_cumulative_45_0_minus45():
    tool = _turn_tool()
    assert tool.parse_targets("45,0,-45") == [45.0, 0.0, -45.0]
    with pytest.raises(ValueError):
        tool.parse_targets("181")
    with pytest.raises(ValueError):
        tool.parse_targets("")


def test_turn_yaw_sign_from_optical_rotation():
    tool = _turn_tool()
    initial = tool.PoseSample(
        {"translation": [0, 0, 0], "rotation": [0, 0, 0, 1]},
        "camera_optical", 0, "tracking", 0.0)
    # A body +yaw (left) is a negative rotation about the optical y axis.
    for body_yaw in (math.radians(45), math.radians(-45)):
        q = [0.0, math.sin(-body_yaw / 2.0), 0.0,
             math.cos(-body_yaw / 2.0)]
        sample = tool.PoseSample(
            {"translation": [0, 0, 0], "rotation": q},
            "camera_optical", 0, "tracking", 0.0)
        got = tool.relative_yaw(sample, initial, (0.0, 0.0), "far")
        assert got == pytest.approx(body_yaw, abs=1e-9)
        assert tool.camera_yaw(sample, (0.0, 0.0), "far") == pytest.approx(
            body_yaw, abs=1e-9)


def test_turn_pose_rejects_stale_or_malformed_status():
    tool = _turn_tool()
    good = {
        "tracking_state": "tracking",
        "pose_valid": True,
        "role": "far",
        "frame_id": "camera_optical",
        "segment": 2,
        "pose": {"t_ns": 1, "translation": [0, 0, 0],
                 "rotation": [0, 0, 0, 1], "frame_id": "camera_optical",
                 "segment": 2, "age_s": 0.01},
    }
    assert tool.parse_pose(good, 0.35, "far") is not None

    missing_age = json.loads(json.dumps(good))
    del missing_age["pose"]["age_s"]
    assert tool.parse_pose(missing_age, 0.35, "far") is None

    nan_age = json.loads(json.dumps(good))
    nan_age["pose"]["age_s"] = float("nan")
    assert tool.parse_pose(nan_age, 0.35, "far") is None

    stale = json.loads(json.dumps(good))
    stale["pose"]["age_s"] = 1.0
    assert tool.parse_pose(stale, 0.35, "far") is None

    bad_q = json.loads(json.dumps(good))
    bad_q["pose"]["rotation"] = [0, 0, 0, 0]
    assert tool.parse_pose(bad_q, 0.35, "far") is None

    bad_t = json.loads(json.dumps(good))
    bad_t["pose"]["translation"] = [0, float("inf"), 0]
    assert tool.parse_pose(bad_t, 0.35, "far") is None

    lost = json.loads(json.dumps(good))
    lost["tracking_state"] = "lost"
    lost["pose_valid"] = False
    assert tool.parse_pose(lost, 0.35, "far") is None

    wrong_role = json.loads(json.dumps(good))
    wrong_role["role"] = "near"
    assert tool.parse_pose(wrong_role, 0.35, "far") is None

    wrong_segment = json.loads(json.dumps(good))
    wrong_segment["pose"]["segment"] = 3
    sample = tool.parse_pose(wrong_segment, 0.35, "far")
    assert sample is not None and sample.segment == 3

    missing_segment = json.loads(json.dumps(good))
    del missing_segment["pose"]["segment"]
    assert tool.parse_pose(missing_segment, 0.35, "far") is None

    zero_stamp = json.loads(json.dumps(good))
    zero_stamp["pose"]["t_ns"] = 0
    assert tool.parse_pose(zero_stamp, 0.35, "far") is None

    missing_frame = json.loads(json.dumps(good))
    missing_frame["pose"].pop("frame_id")
    missing_frame.pop("frame_id")
    assert tool.parse_pose(missing_frame, 0.35, "far") is None


class _FakeRobotState:
    def __init__(self):
        self.connected = True
        self.last_telemetry_ts = time.monotonic()
        self.estop_latched = False
        self.mode = 2
        self.wheel_armed = {"left": True, "right": True}
        self.cmd = (0.1, 0.1)


class _FakeRobot:
    def __init__(self, *, arm_works=True, telemetry=True):
        self.state = _FakeRobotState()
        self.arm_works = arm_works
        self.telemetry = telemetry
        self.calls = []

    def _fresh(self):
        if self.telemetry:
            self.state.last_telemetry_ts = time.monotonic()

    def send_twist(self, vx, wz):
        self.calls.append(("twist", vx, wz))
        self.state.cmd = (vx, wz)
        self._fresh()

    def arm_wheels(self, enabled):
        self.calls.append(("arm", enabled))
        if self.arm_works:
            self.state.wheel_armed = {k: bool(enabled)
                                      for k in self.state.wheel_armed}
        self._fresh()

    def set_mode(self, mode):
        self.calls.append(("mode", mode))
        self.state.mode = mode
        self._fresh()


def test_turn_cleanup_orders_zero_disarm_mode_and_confirms():
    tool = _turn_tool()
    robot = _FakeRobot()
    failures = tool.cleanup(robot, initial_mode=2, timeout_s=0.2)
    assert failures == []
    assert robot.calls == [("twist", 0.0, 0.0), ("arm", False),
                           ("mode", 2)]
    assert robot.state.cmd == (0.0, 0.0)
    assert not any(robot.state.wheel_armed.values())
    assert robot.state.mode == 2


def test_turn_cleanup_surfaces_unconfirmed_disarm():
    tool = _turn_tool()
    robot = _FakeRobot(arm_works=False)
    failures = tool.cleanup(robot, initial_mode=2, timeout_s=0.1)
    assert any("disarm" in f for f in failures)
    # Zero and mode were still attempted, but the failure is not hidden.
    assert robot.calls[0] == ("twist", 0.0, 0.0)
    assert robot.calls[-1] == ("mode", 2)


def test_turn_cleanup_rejects_stale_telemetry():
    tool = _turn_tool()
    robot = _FakeRobot(telemetry=False)
    failures = tool.cleanup(robot, initial_mode=2, timeout_s=0.1)
    assert len(failures) == 3
    assert all("fresh telemetry" in f for f in failures)


# ---------------------------------------------------------------------------
# bounded depth-geometry map feed
# ---------------------------------------------------------------------------

from bebop_vision.slam.map import (DepthGeometryConfig, DepthGeometryMap,
                                   undistort_pixels)  # noqa: E402
from bebop_vision.slam.server import SlamStatusServer  # noqa: E402
import urllib.error  # noqa: E402
import urllib.request  # noqa: E402


def test_undistort_pixels_roundtrips_rational_model():
    model = CameraModel(16, 12, 10.0, 10.0, 8.0, 6.0,
                        (-0.12, 0.03, 0.001, -0.002, 0.0, 0.0, 0.0, 0.0))
    xn = np.array([-0.6, -0.2, 0.4])
    yn = np.array([0.3, -0.5, 0.2])
    k1, k2, p1, p2, k3, k4, k5, k6 = model.distortion
    r2 = xn * xn + yn * yn
    radial = ((1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3)
              / (1 + k4 * r2 + k5 * r2 ** 2 + k6 * r2 ** 3))
    tx = 2 * p1 * xn * yn + p2 * (r2 + 2 * xn * xn)
    ty = p1 * (r2 + 2 * yn * yn) + 2 * p2 * xn * yn
    u = model.fx * (xn * radial + tx) + model.cx
    v = model.fy * (yn * radial + ty) + model.cy
    got_x, got_y = undistort_pixels(u, v, model)
    assert np.allclose(got_x, xn, atol=1e-7)
    assert np.allclose(got_y, yn, atol=1e-7)


def _geometry(cal, **kwargs):
    cfg = DepthGeometryConfig(pixel_stride=1, max_points=2000,
                              voxel_size_m=0.01, max_range_m=6.0,
                              min_range_m=0.15, sample_hz=2.0)
    for key, value in kwargs.items():
        setattr(cfg, key, value)
    return DepthGeometryMap(cal, cfg, segment=0)


def test_depth_geometry_map_transform_scale_bounds_and_color(calibration):
    rgb = np.zeros((COLOR_H, COLOR_W, 3), np.uint8)
    rgb[:, :] = (10, 20, 30)
    depth = np.full((COLOR_H, COLOR_W), 1000, np.uint16)

    geom = _geometry(calibration)
    assert geom.add(depth, rgb, _pose(translation=(0.0, 0.0, 0.0)), 0)
    snap = geom.snapshot()
    assert 0 < snap["count"] <= snap["capacity"]
    # Center pixel (8, 6) is the optical axis: (0, 0, z).
    center = min(snap["points"], key=lambda p: abs(p[0]) + abs(p[1]))
    assert abs(center[2] - 1.0) < 0.02
    assert center[3:] == [10, 20, 30]
    json.dumps(snap, allow_nan=False)

    # Translation moves the world_from_camera cloud, not the camera rays.
    moved = _geometry(calibration)
    assert moved.add(depth, rgb, _pose(translation=(1.0, 2.0, 3.0)), 0)
    pts = moved.snapshot()["points"]
    near = min(pts, key=lambda p: abs(p[0] - 1.0) + abs(p[1] - 2.0))
    assert abs(near[2] - 4.0) < 0.02

    # Aligned depth is millimetres: 2000 -> 2 m.
    scaled = _geometry(calibration)
    assert scaled.add(np.full((COLOR_H, COLOR_W), 2000, np.uint16), rgb,
                      _pose(translation=(0.0, 0.0, 0.0)), 0)
    assert all(abs(p[2] - 2.0) < 0.05
               for p in scaled.snapshot()["points"])

    # Out-of-range depth is dropped.
    empty = _geometry(calibration)
    assert not empty.add(np.full((COLOR_H, COLOR_W), 7000, np.uint16), rgb,
                         _pose(translation=(0.0, 0.0, 0.0)), 0)
    assert empty.snapshot()["count"] == 0
    assert not empty.add(np.full((COLOR_H, COLOR_W), 50, np.uint16), rgb,
                         _pose(translation=(0.0, 0.0, 0.0)), 0)


def test_depth_geometry_map_bounded_and_resets_on_segment(calibration):
    geom = _geometry(calibration, max_points=40, voxel_size_m=0.02)
    depth = np.full((COLOR_H, COLOR_W), 1000, np.uint16)
    assert geom.add(depth, None, _pose(translation=(0.0, 0.0, 0.0)), 0)
    assert geom.add(depth, None, _pose(translation=(10.0, 0.0, 0.0)), 0)
    snap = geom.snapshot()
    assert snap["count"] <= 40
    assert snap["revision"] >= 1

    # Segment change drops all old geometry before adding the new segment.
    assert geom.add(depth, None, _pose(translation=(0.0, 0.0, 0.0)), 1)
    snap = geom.snapshot()
    assert snap["segment"] == 1
    assert snap["count"] <= 40
    assert all(abs(p[0]) < 2.0 for p in snap["points"])
    assert snap["corrected"] is False
    assert snap["geometry_source"] == "depth_odometry"
    assert snap["frame_id"] == "camera_optical"
    assert snap["pose_frame"] == "world_from_camera_optical"


def test_live_map_uses_uncorrected_pose_and_resets_on_recovery(tmp_path):
    frame = FakeFrame()
    results = [
        _tracked(_pose(translation=(0.0, 0.0, 0.0)),
                 corrected=_pose(translation=(9.0, 9.0, 9.0))),
        _tracked(None, state="lost"),
        _tracked(_pose(translation=(5.0, 5.0, 5.0))),
    ]
    slam, backend, _ = _make(tmp_path, frame, results, enable_slam=True)
    slam._tick()
    snap = slam.map_snapshot()
    assert snap["corrected"] is False
    assert snap["geometry_source"] == "depth_odometry"
    assert snap["count"] > 0
    # Corrected pose is (9,9,9); geometry must stay on the odometry pose.
    assert all(abs(p[0]) < 3.0 and abs(p[1]) < 3.0 for p in snap["points"])
    assert snap["session_id"] == snap["status"]["session_id"]
    assert snap["session_id"] == snap["trajectory"]["session_id"]
    json.dumps(snap, allow_nan=False)

    frame.stamp_us += 1
    frame.color_stamp_us = frame.stamp_us
    slam._tick()
    assert slam.status()["tracking_state"] == "lost"
    assert slam.map_snapshot()["segment"] == 0

    frame.stamp_us += 1
    frame.color_stamp_us = frame.stamp_us
    slam._tick()
    st = slam.status()
    assert st["tracking_state"] == "tracking"
    assert st["segment"] == 1
    snap = slam.map_snapshot()
    assert snap["segment"] == 1
    assert snap["count"] > 0
    # Old origin-segment geometry is gone; new cloud is near (5,5,5).
    assert all(p[0] > 3.0 and p[1] > 3.0 for p in snap["points"])


def test_live_map_sampling_is_rate_limited(tmp_path):
    frame = FakeFrame()
    results = [_tracked(_pose(translation=(0.0, 0.0, 0.0))) for _ in range(3)]
    slam, _, _ = _make(tmp_path, frame, results)
    slam._tick()
    first = slam.map_snapshot()
    assert first["count"] > 0

    frame.stamp_us += 1
    frame.color_stamp_us = frame.stamp_us
    slam._tick()
    second = slam.map_snapshot()
    assert second["revision"] == first["revision"]

    slam._last_geometry_mono -= 1.0
    frame.stamp_us += 1
    frame.color_stamp_us = frame.stamp_us
    slam._tick()
    assert slam.map_snapshot()["revision"] > first["revision"]


def test_map_http_snapshot_schema_and_headers(tmp_path):
    frame = FakeFrame()
    slam, _, _ = _make(tmp_path, frame,
                       [_tracked(_pose(translation=(0.0, 0.0, 0.0)))])
    slam._tick()
    server = SlamStatusServer(slam, port=0).start()
    try:
        with urllib.request.urlopen(
                f"http://127.0.0.1:{server.bound_port}/map") as response:
            body = json.loads(response.read())
            assert response.status == 200
            assert response.headers["Access-Control-Allow-Origin"] == "*"
            assert response.headers["Cache-Control"] == "no-store"
    finally:
        server.stop()

    required = {"session_id", "revision", "frame_id", "pose_frame",
                "geometry_source", "corrected", "segment", "points", "count",
                "capacity", "voxel_size_m", "updated_age_s", "status",
                "trajectory"}
    assert required <= set(body)
    assert body["session_id"] == slam.status()["session_id"]
    assert body["geometry_source"] == "depth_odometry"
    assert body["corrected"] is False
    assert body["count"] == len(body["points"]) <= body["capacity"]
    assert body["updated_age_s"] is not None
    for point in body["points"]:
        assert len(point) == 6
        assert all(math.isfinite(v) for v in point[:3])
        assert all(0 <= v <= 255 for v in point[3:])
    json.dumps(body, allow_nan=False)


def test_geometry_default_capacity_is_100k():
    cfg = DepthGeometryConfig()
    assert cfg.max_points == 100_000
    assert cfg.max_transmit_points == 30_000
    assert cfg.sample_hz == 1.0


def test_geometry_cache_is_lazy_and_immutable(calibration):
    geom = _geometry(calibration)
    depth = np.full((COLOR_H, COLOR_W), 1000, np.uint16)
    assert geom.add(depth, None, _pose(translation=(0.0, 0.0, 0.0)), 0)
    # add() only bumps the revision; it must not rebuild/serialize geometry.
    assert geom._cached_revision != geom.revision
    first = geom.snapshot()
    assert geom._cached_revision == geom.revision
    assert first["count"] > 0
    second = geom.snapshot()
    # Same revision: the point list is shared and not reconstructed.
    assert second["points"] is first["points"]
    assert second["revision"] == first["revision"]


def test_map_bytes_schema_order_and_finite(tmp_path):
    frame = FakeFrame()
    slam, _, _ = _make(tmp_path, frame,
                       [_tracked(_pose(translation=(0.0, 0.0, 0.0)))])
    slam._tick()
    body = json.loads(slam.map_snapshot_bytes())
    assert list(body) == [
        "session_id", "revision", "frame_id", "pose_frame",
        "geometry_source", "corrected", "segment", "points", "count",
        "capacity", "voxel_size_m", "map_points", "transmit_stride",
        "updated_age_s", "status", "trajectory",
    ]
    assert body["capacity"] == 100_000
    assert body["count"] == len(body["points"]) <= 30_000
    assert body["map_points"] == body["count"]
    assert body["transmit_stride"] == 1
    assert body["updated_age_s"] is not None
    assert body["status"]["session_id"] == body["session_id"]
    assert body["trajectory"]["session_id"] == body["session_id"]
    json.dumps(body, allow_nan=False)


def test_geometry_serialization_does_not_hold_map_lock(calibration):
    geom = _geometry(calibration)
    depth = np.full((COLOR_H, COLOR_W), 1000, np.uint16)
    assert geom.add(depth, None, _pose(translation=(0.0, 0.0, 0.0)), 0)

    entered = threading.Event()
    release = threading.Event()
    original = geom._serialize_points

    def slow_serialize(points, chunk=4000):
        entered.set()
        release.wait(5.0)
        return original(points, chunk)

    geom._serialize_points = slow_serialize
    thread = threading.Thread(target=geom.snapshot_bundle)
    try:
        thread.start()
        assert entered.wait(2.0)
        # add() must not wait for the (blocked) serializer.
        started = time.monotonic()
        assert geom.add(depth, None,
                        _pose(translation=(1.0, 0.0, 0.0)), 0)
        assert time.monotonic() - started < 1.0
    finally:
        release.set()
        thread.join(2.0)
        geom._serialize_points = original


def test_geometry_transmit_decimation_is_deterministic_and_bounded(calibration):
    cfg = dict(max_points=500, max_transmit_points=7, voxel_size_m=0.005,
               pixel_stride=1)
    depth = np.full((COLOR_H, COLOR_W), 1000, np.uint16)
    poses = [_pose(translation=(float(i), 0.0, 0.0)) for i in range(3)]
    geom = _geometry(calibration, **cfg)
    for pose in poses:
        assert geom.add(depth, None, pose, 0)
    snap = geom.snapshot()
    internal = len(geom._voxels)
    assert internal > 7
    assert snap["capacity"] == 500
    assert snap["map_points"] == internal
    assert snap["count"] == len(snap["points"]) <= 7
    assert snap["transmit_stride"] == math.ceil(internal / 7)

    # Same revision reuses the same cached list.
    again = geom.snapshot()
    assert again["points"] is snap["points"]
    assert again["map_points"] == snap["map_points"]

    # A second identical map decimates identically.
    other = _geometry(calibration, **cfg)
    for pose in poses:
        assert other.add(depth, None, pose, 0)
    assert other.snapshot()["points"] == snap["points"]


# ---------------------------------------------------------------------------
# Phase 1 structural map
# ---------------------------------------------------------------------------

from bebop_vision.slam.structure import (STRUCTURE_FRAME_ID, StructureConfig,
                                         StructureMap)  # noqa: E402


def _floor_cloud():
    xs = np.arange(-1.0, 1.001, 0.05)
    zs = np.arange(1.0, 4.001, 0.05)
    return np.array([[x, 1.3, z] for x in xs for z in zs], dtype=float)


def _wall_cloud():
    # Vertical structure above the floor (camera height 1.3 m): heights
    # 0.40-1.85 m, so each cell spans enough extent and clears the modest
    # 0.8 m wall floor while remaining a thin, elongated component.
    xs = np.arange(-1.0, 1.001, 0.05)
    ys = np.arange(-0.55, 0.901, 0.05)
    # Put the wall at the far edge of _floor_cloud so synthetic rays do not
    # carve free cells *behind* it (real depth is occluded; point lists are
    # not).  Otherwise it looks like a wall seen from both sides.
    return np.array([[x, y, 4.0] for x in xs for y in ys], dtype=float)


def _decode_rle(encoded, n):
    if not encoded:
        return np.zeros(n, dtype=bool)
    runs = np.frombuffer(base64.b64decode(encoded), dtype="<u2")
    out = np.zeros(n, dtype=bool)
    idx = 0
    value = 0
    for run in runs:
        run = int(run)
        if value:
            out[idx:idx + run] = True
        idx += run
        value ^= 1
    return out


def _structure_config(**kwargs):
    cfg = StructureConfig(floor_min_inliers=300, floor_min_area_m2=0.5,
                          max_cells=20_000)
    for key, value in kwargs.items():
        setattr(cfg, key, value)
    return cfg


def test_structure_wall_defaults_are_obstacle_safe():
    cfg = StructureConfig()
    assert cfg.wall_min_top_m == pytest.approx(0.8)
    assert cfg.wall_min_extent_m == pytest.approx(0.40)
    assert cfg.min_wall_segment_m == pytest.approx(0.8)
    assert cfg.wall_min_aspect_ratio == pytest.approx(3.0)
    assert cfg.wall_max_thickness_m == pytest.approx(0.5)
    assert cfg.wall_boundary_check is True
    # The height floor is a modest low-clutter filter (~desk height).  The
    # component shape + free-boundary tests are the real wall/obstacle
    # discriminator; the old 1.4 m floor silently killed real walls from a
    # chest-height camera (live: 1621 occupied cells -> 3 candidates).
    assert 0.75 < cfg.wall_min_top_m <= 0.9
    assert cfg.min_wall_segment_m > 0.25


def _floor_rect(x0, x1, z0, z1, *, step=0.1, height=1.3):
    xs = np.arange(x0, x1 + 1e-9, step)
    zs = np.arange(z0, z1 + 1e-9, step)
    return np.array([[x, height, z] for x in xs for z in zs], dtype=float)


def _wall_rect(x0, z0, x1, z1, *, bottom_m=0.05, top_m=2.0, step=0.05):
    """A dense vertical wall sheet between two floor-plan points."""
    pts = []
    n = max(2, int(round(max(abs(x1 - x0), abs(z1 - z0)) / step)) + 1)
    heights = np.arange(bottom_m, top_m + 1e-9, step)
    for t in np.linspace(0.0, 1.0, n):
        x = x0 + (x1 - x0) * t
        z = z0 + (z1 - z0) * t
        for h in heights:
            pts.append([x, 1.3 - h, z])
    return np.array(pts, dtype=float)


def _slab(x0, x1, z0, z1, *, bottom_m=0.05, top_m=0.75, step=0.05):
    """A compact rectangular solid (desk/box): vertical sides plus top."""
    pts = []
    heights = np.arange(bottom_m, top_m + 1e-9, step)
    ys = 1.3 - heights
    for x in np.arange(x0, x1 + 1e-9, step):
        for y in ys:
            pts.append([x, y, z0])
            pts.append([x, y, z1])
    for z in np.arange(z0, z1 + 1e-9, step):
        for y in ys:
            pts.append([x0, y, z])
            pts.append([x1, y, z])
    for x in np.arange(x0, x1 + 1e-9, step):
        for z in np.arange(z0, z1 + 1e-9, step):
            pts.append([x, 1.3 - top_m, z])
    return np.array(pts, dtype=float)


def test_structure_no_estimate_before_floor_support(calibration):
    st = StructureMap(calibration, _structure_config(floor_min_inliers=500))
    pts = _floor_cloud()[:100]
    assert st.add_points(pts, _pose(translation=(0.0, 0.0, 0.0)), 0)
    snap = st.snapshot()
    assert snap["frame_id"] is None
    assert snap["pose_frame"] is None
    assert snap["gravity"] == {"valid": False, "up": None, "roll_deg": None,
                               "pitch_deg": None, "inlier_ratio": None}
    assert snap["floor"] == {"valid": False, "normal": None, "offset_m": None,
                             "camera_height_m": None, "inliers": None}
    assert snap["pose"]["valid"] is False
    assert snap["grid"]["occupied_count"] == 0
    assert snap["grid"]["free_count"] == 0
    assert snap["walls"] == []


def test_structure_estimates_gravity_floor_and_wall_lines(calibration):
    st = StructureMap(calibration, _structure_config())
    floor = _floor_cloud()
    assert st.add_points(floor, _pose(translation=(0.0, 0.0, 0.0)), 0)
    first = st.snapshot()
    assert first["frame_id"] == "map"
    assert first["pose_frame"] == "map_from_camera_optical"
    assert first["gravity"]["valid"] is True
    assert first["floor"]["valid"] is True
    assert first["gravity"]["up"] == pytest.approx([0.0, -1.0, 0.0],
                                                   abs=1e-5)
    assert first["floor"]["camera_height_m"] == pytest.approx(1.3, abs=0.01)
    assert first["pose"]["valid"] is True
    assert first["pose"]["yaw_deg"] == pytest.approx(0.0, abs=0.1)
    assert first["grid"]["occupied_count"] == 0  # floor is not a wall

    wall = _wall_cloud()
    assert st.add_points(wall, _pose(translation=(0.0, 0.0, 0.0)), 0)
    snap = st.snapshot()
    assert snap["grid"]["occupied_count"] > 0
    assert snap["grid"]["free_count"] > 0
    n = snap["grid"]["width"] * snap["grid"]["height"]
    occ = _decode_rle(snap["grid"]["occupied_rle"], n)
    free = _decode_rle(snap["grid"]["free_rle"], n)
    assert int(occ.sum()) == snap["grid"]["occupied_count"]
    assert int(free.sum()) == snap["grid"]["free_count"]
    assert snap["walls"], "expected fitted wall segments"
    for x0, y0, x1, y1 in snap["walls"]:
        assert math.hypot(x1 - x0, y1 - y0) >= 0.8 - 1e-9
    assert snap["wall_cells"]
    assert all(0 <= ix < snap["grid"]["width"]
               and 0 <= iy < snap["grid"]["height"]
               for ix, iy in snap["wall_cells"])
    assert snap["path"]
    assert len(snap["path"][0]) == 2
    assert snap["revision"] > 0
    assert snap["updated_age_s"] is not None
    assert len(json.dumps(snap, allow_nan=False)) < 200_000


def _cell_of(snap, x, y):
    origin = snap["grid"]["origin"]
    res = snap["grid"]["resolution_m"]
    return (int(math.floor((x - origin[0]) / res)),
            int(math.floor((y - origin[1]) / res)))


def _masks(snap):
    n = snap["grid"]["width"] * snap["grid"]["height"]
    shape = (snap["grid"]["height"], snap["grid"]["width"])
    return (_decode_rle(snap["grid"]["occupied_rle"], n).reshape(shape),
            _decode_rle(snap["grid"]["free_rle"], n).reshape(shape))


def test_structure_occupancy_is_above_floor_and_ray_carving(calibration):
    st = StructureMap(calibration, _structure_config(max_cells=20_000))
    assert st.add_points(_floor_cloud(), _pose(translation=(0.0, 0.0, 0.0)), 0)
    # Camera height is 1.3 m.  A hit 0.05 m above the floor is floor; a hit
    # 0.4 m above it is raw occupancy but a single height has no vertical
    # extent, so it is not wall-like.  Rays to both hits carve free space.
    low = np.array([[0.0, 1.25, 2.0]])
    high = np.array([[0.5, 0.9, 2.0]])
    st.add_points(np.vstack([low, high]),
                  _pose(translation=(0.0, 0.0, 0.0)), 0)
    snap = st.snapshot()
    occ, free = _masks(snap)
    # Map x = camera forward (world z), map y = -world x.
    lx, ly = _cell_of(snap, 2.0, 0.0)
    hx, hy = _cell_of(snap, 2.0, -0.5)
    assert occ[ly, lx] == 0, "a point below occupied_min_height is floor"
    assert occ[hy, hx] == 1, "above-floor hits stay in the obstacle layer"
    assert snap["diagnostics"]["occupied_cells"] >= 1
    assert snap["diagnostics"]["wall_cells"] == 0
    assert snap["wall_cells"] == [] and snap["walls"] == []
    assert all(snap["grid"]["height"] > y >= 0
               and snap["grid"]["width"] > x >= 0 for x, y in [(lx, ly),
                                                               (hx, hy)])
    # Free space lies between the camera and the high hit (Bresenham ray),
    # and the camera cell itself is carved.
    cx, cy = _cell_of(snap, 0.0, 0.0)
    mx, my = _cell_of(snap, 1.0, -0.25)
    assert free[cy, cx] == 1
    assert free[my, mx] == 1
    assert snap["grid"]["free_count"] > 0


def test_structure_floor_only_scene_carves_free_but_has_no_walls(calibration):
    # Regression: rays must be cast to floor hits, not only above-floor hits.
    st = StructureMap(calibration, _structure_config(max_cells=20_000))
    assert st.add_points(_floor_cloud(), _pose(translation=(0.0, 0.0, 0.0)), 0)
    snap = st.snapshot()
    assert snap["grid"]["free_count"] > 0
    assert snap["grid"]["occupied_count"] == 0
    assert snap["diagnostics"]["wall_cells"] == 0
    assert snap["walls"] == []
    free = _masks(snap)[1]
    fx, fy = _cell_of(snap, 1.0, 0.0)
    assert free[fy, fx] == 1
    fx2, fy2 = _cell_of(snap, 3.5, 0.0)
    assert free[fy2, fx2] == 1


def _tilted_floor_cloud(deg=5.0, height=1.3):
    """Floor sloping away from the camera (simulates plane-estimate error)."""
    slope = math.tan(math.radians(deg))
    xs = np.arange(-1.0, 1.001, 0.05)
    zs = np.arange(1.0, 5.001, 0.05)
    # Above the locked floor plane at distance, i.e. what a small plane
    # estimation error looks like in the gravity-aligned frame.
    return np.array([[x, height - slope * z, z] for x in xs for z in zs],
                    dtype=float)


def test_structure_tilted_floor_has_zero_wall_cells(calibration):
    # Regression from the live robot: a 1-3 deg floor-plane error makes far
    # floor points look 0.1-0.3 m high.  Those cells are still in the
    # obstacle layer (grey) but must never feed the wall fitter, which used
    # to produce a solid blob and diagonal segments.
    st = StructureMap(calibration, _structure_config(max_cells=20_000))
    assert st.add_points(_floor_cloud(), _pose(translation=(0.0, 0.0, 0.0)), 0)
    assert st.add_points(_tilted_floor_cloud(5.0),
                         _pose(translation=(0.0, 0.0, 0.0)), 0)
    snap = st.snapshot()
    assert snap["diagnostics"]["occupied_cells"] > 0, \
        "the synthetic tilt must exceed occupied_min_height somewhere"
    assert snap["diagnostics"]["wall_cells"] == 0
    assert snap["grid"]["occupied_count"] == \
        snap["diagnostics"]["occupied_cells"]
    assert snap["wall_cells"] == []
    assert snap["walls"] == []


def test_structure_vertical_wall_produces_wall_cells_and_segment(
        calibration):
    st = StructureMap(calibration, _structure_config(max_cells=20_000))
    assert st.add_points(_floor_cloud(), _pose(translation=(0.0, 0.0, 0.0)), 0)
    assert st.add_points(_wall_cloud(), _pose(translation=(0.0, 0.0, 0.0)), 0)
    snap = st.snapshot()
    assert snap["diagnostics"]["wall_cells"] > 0
    assert len(snap["wall_cells"]) == snap["diagnostics"]["wall_cells"]
    assert snap["grid"]["occupied_count"] == snap["diagnostics"]["wall_cells"]
    assert snap["walls"], "a vertical wall must yield a fitted segment"
    for x0, y0, x1, y1 in snap["walls"]:
        assert math.hypot(x1 - x0, y1 - y0) >= 0.8 - 1e-9


def _room_with_desk_scene():
    pts = [_floor_rect(-2.0, 2.0, -2.0, 3.0)]
    pts += [
        _wall_rect(-2.0, -2.0, 2.0, -2.0),
        _wall_rect(-2.0, 3.0, 2.0, 3.0),
        _wall_rect(-2.0, -2.0, -2.0, 3.0),
        _wall_rect(2.0, -2.0, 2.0, 3.0),
    ]
    # A standing desk in the middle: 1.2 x 1.0 m, 0.95 m tall.  It clears the
    # 0.8 m cell-stage height floor, so this scene proves the COMPONENT SHAPE
    # test (not the height gate) keeps desk-height clutter out of the walls;
    # the live robot saw chairs/monitors at this height as walls.
    pts.append(_slab(-0.6, 0.6, 0.5, 1.5, top_m=0.95))
    return np.vstack(pts)


def _near_room_boundary(x, y):
    # Map x = forward (world z), map y = left (-world x) in this scene.
    return min(abs(x + 2.0), abs(x - 3.0), abs(y + 2.0), abs(y - 2.0)) < 0.3


def test_structure_room_walls_extract_and_desk_is_not_a_wall(calibration):
    st = StructureMap(calibration, _structure_config())
    assert st.add_points(_room_with_desk_scene(),
                         _pose(translation=(0.0, 0.0, 0.0)), 0)
    snap = st.snapshot()
    assert snap["walls"], "the four room walls must yield segments"
    for x0, y0, x1, y1 in snap["walls"]:
        assert math.hypot(x1 - x0, y1 - y0) >= 0.8 - 1e-9
        assert _near_room_boundary((x0 + x1) / 2.0, (y0 + y1) / 2.0), \
            f"wall segment {(x0, y0, x1, y1)} is not on a room boundary"
    # The desk is in the obstacle layer ...
    occ, _ = _masks(snap)
    dx, dy = _cell_of(snap, 1.0, 0.0)  # world (x=0, z=1) -> map (1, 0)
    assert occ[dy, dx] == 1, "the desk must remain occupied"
    assert snap["diagnostics"]["occupied_cells"] > \
        snap["diagnostics"]["wall_cells"]
    # The 0.95 m desk clears the 0.8 m cell-stage height floor, so it is the
    # COMPONENT SHAPE test (not the height gate) that rejects it.
    assert snap["diagnostics"]["wall_cells_rejected_height"] == 0
    assert snap["diagnostics"]["wall_candidate_cells"] > 0
    assert snap["diagnostics"]["wall_components_rejected_shape"] >= 1
    # ... and contributes neither wall cells nor a wall segment.
    origin = snap["grid"]["origin"]
    res = snap["grid"]["resolution_m"]
    for ix, iy in snap["wall_cells"]:
        x = origin[0] + (ix + 0.5) * res
        y = origin[1] + (iy + 0.5) * res
        assert not (0.4 < x < 1.6 and -0.7 < y < 0.7), \
            f"desk cell {(x, y)} was classified as wall"
    for x0, y0, x1, y1 in snap["walls"]:
        mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        assert not (0.4 < mx < 1.6 and -0.7 < my < 0.7), \
            "a wall segment crosses the desk"


def test_structure_free_standing_box_is_occupied_but_not_a_wall(calibration):
    pts = np.vstack([
        _floor_rect(-2.0, 2.0, 0.0, 4.0),
        # 0.6 x 0.6 m, 1.8 m tall: it clears the wall height floor and is
        # compact, so only the component-shape test can reject it.
        _slab(-0.3, 0.3, 1.5, 2.1, top_m=1.8),
    ])
    st = StructureMap(calibration, _structure_config())
    assert st.add_points(pts, _pose(translation=(0.0, 0.0, 0.0)), 0)
    snap = st.snapshot()
    assert snap["grid"]["occupied_count"] > 0
    occ, _ = _masks(snap)
    bx, by = _cell_of(snap, 1.8, 0.0)  # world (x=0, z=1.8) -> map (1.8, 0)
    assert occ[by, bx] == 1, "the box must remain occupied"
    assert snap["wall_cells"] == []
    assert snap["walls"] == []
    assert snap["diagnostics"]["wall_components_rejected_shape"] >= 1
    assert snap["diagnostics"]["wall_components_elongated"] == 0


def test_structure_long_thin_wall_still_extracts(calibration):
    # The floor stops at the wall: no synthetic rays pass through it, so it
    # is observed from one side only (as real depth would be).
    pts = np.vstack([
        _floor_rect(-2.0, 2.0, 0.0, 4.0),
        _wall_rect(-2.0, 4.0, 2.0, 4.0),
    ])
    st = StructureMap(calibration, _structure_config())
    assert st.add_points(pts, _pose(translation=(0.0, 0.0, 0.0)), 0)
    snap = st.snapshot()
    assert snap["diagnostics"]["wall_cells"] > 0
    assert snap["walls"], "a 4 m thin wall must still produce a segment"
    lengths = [math.hypot(s[2] - s[0], s[3] - s[1])
               for s in snap["walls"]]
    assert max(lengths) >= 3.5
    assert all(length >= 0.8 - 1e-9 for length in lengths)


def test_structure_low_wall_below_old_height_floor_still_extracts(calibration):
    """A 0.9 m wall must survive the (now modest) height gate.

    This is the live regression: the old 1.4 m floor required every wall cell
    to top out above 1.4 m, which a chest-height camera pitched down rarely
    sees.  0.9 m is still a wall; shape + free boundary decide, not height.
    """
    pts = np.vstack([
        _floor_rect(-2.0, 2.0, 0.0, 4.0),
        _wall_rect(-2.0, 4.0, 2.0, 4.0, top_m=0.9),
    ])
    st = StructureMap(calibration, _structure_config())
    assert st.add_points(pts, _pose(translation=(0.0, 0.0, 0.0)), 0)
    snap = st.snapshot()
    assert snap["diagnostics"]["wall_candidate_cells"] > 0
    assert snap["diagnostics"]["wall_cells"] > 0
    assert snap["walls"], "a 0.9 m thin wall must still produce a segment"
    lengths = [math.hypot(s[2] - s[0], s[3] - s[1])
               for s in snap["walls"]]
    assert max(lengths) >= 3.5


def test_structure_desk_rejected_while_tall_wall_run_accepted(calibration):
    """0.75 m compact desk -> height-rejected; 1.2 m elongated run -> wall."""
    pts = np.vstack([
        _floor_rect(-2.0, 2.0, 0.0, 4.0),
        _wall_rect(-2.0, 4.0, 2.0, 4.0, top_m=1.2),
        _slab(-0.6, 0.6, 1.5, 2.5, top_m=0.75),
    ])
    st = StructureMap(calibration, _structure_config())
    assert st.add_points(pts, _pose(translation=(0.0, 0.0, 0.0)), 0)
    snap = st.snapshot()
    diag = snap["diagnostics"]
    # The desk never reaches the 0.8 m candidate floor ...
    assert diag["wall_cells_rejected_height"] >= 1
    # ... while the 1.2 m run does, and becomes a segment.
    assert diag["wall_candidate_cells"] > 0
    assert snap["walls"], "the 1.2 m wall run must yield a segment"
    # No accepted wall cell or segment touches the desk footprint
    # (world x in [-0.6, 0.6], z in [1.5, 2.5] -> map x, y).
    origin = snap["grid"]["origin"]
    res = snap["grid"]["resolution_m"]
    for ix, iy in snap["wall_cells"]:
        x = origin[0] + (ix + 0.5) * res
        y = origin[1] + (iy + 0.5) * res
        assert not (1.4 < x < 2.6 and -0.7 < y < 0.7), \
            f"desk cell {(x, y)} was classified as wall"
    for x0, y0, x1, y1 in snap["walls"]:
        mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        assert mx > 3.5, f"wall segment {(x0, y0, x1, y1)} left the wall run"


def test_structure_wall_boundary_needs_free_on_one_side(calibration):
    st = StructureMap(calibration, _structure_config(
        wall_boundary_probe_cells=2, wall_boundary_min_fraction=0.5))
    component = [(20, y) for y in range(20, 40)]
    one_side = {(18, y) for y in range(18, 42)}
    both_sides = one_side | {(22, y) for y in range(18, 42)}
    assert st._component_bounds_free_space(component, one_side) is True
    # Free space on both sides is an obstacle in open floor, not a wall.
    assert st._component_bounds_free_space(component, both_sides) is False


def test_structure_floor_refit_reduces_residual(calibration):
    rng = np.random.default_rng(7)
    pts = _floor_cloud().copy()
    pts += rng.normal(0.0, 0.01, pts.shape)
    # Gross outliers must not survive the inlier gate.
    pts[: len(pts) // 10, 1] += 0.40
    pose = _pose(translation=(0.0, 0.0, 0.0))
    raw = StructureMap(calibration, _structure_config(
        floor_refine_iterations=0))
    refit = StructureMap(calibration, _structure_config(
        floor_refine_iterations=3))
    assert raw.add_points(pts, pose, 0)
    assert refit.add_points(pts, pose, 0)
    r0 = raw.snapshot()["floor"]["rmse_m"]
    r3 = refit.snapshot()["floor"]["rmse_m"]
    assert r0 is not None and r3 is not None
    assert r3 <= r0
    assert r3 < 0.02
    assert refit.snapshot()["floor"]["refinements"] >= 2
    assert refit.snapshot()["diagnostics"]["floor_rmse_m"] == r3


def test_structure_aligned_frame_and_pose_math(calibration):
    st = StructureMap(calibration, _structure_config())
    assert st.add_points(_floor_cloud(), _pose(translation=(0.0, 0.0, 0.0)), 0)
    # Move the camera +1 m forward (world z) and +0.5 m left (world x) with
    # the same orientation.  Map x is forward, map y is left.
    pose = _pose(translation=(0.5, 0.0, 1.0))
    # A vertical column 2 m ahead of the camera; world = camera + translation.
    ys = np.arange(0.5, 1.001, 0.05)
    camera_points = np.array([[0.0, y, 2.0] for y in ys])
    st.add_points(camera_points, pose, 0)
    snap = st.snapshot()
    # Camera ground position is map (1.0, -0.5): forward 1 m, left 0.5 m
    # (world +x is camera-left because optical y points down).
    assert snap["pose"]["x"] == pytest.approx(1.0, abs=0.01)
    assert snap["pose"]["y"] == pytest.approx(-0.5, abs=0.01)
    assert snap["pose"]["yaw_deg"] == pytest.approx(0.0, abs=0.1)
    occ, _ = _masks(snap)
    ix, iy = _cell_of(snap, 3.0, -0.5)
    assert occ[iy, ix] == 1, "column must land at aligned (3.0, -0.5)"


def test_structure_rotated_camera_yaw_sign(calibration):
    st = StructureMap(calibration, _structure_config())
    assert st.add_points(_floor_cloud(), _pose(translation=(0.0, 0.0, 0.0)), 0)
    # Optical +y points down, so a body left turn is a -45 deg rotation
    # about optical y.  It must report positive map yaw (CCW from above).
    q = (0.0, math.sin(math.radians(-45) / 2.0), 0.0,
         math.cos(math.radians(-45) / 2.0))
    st.add_points(_floor_cloud()[:20], _pose(rotation=q), 0)
    yaw = st.snapshot()["pose"]["yaw_deg"]
    assert yaw == pytest.approx(45.0, abs=0.5)


def test_structure_segment_reset_drops_the_floor_plan(calibration):
    st = StructureMap(calibration, _structure_config())
    assert st.add_points(_floor_cloud(), _pose(translation=(0.0, 0.0, 0.0)), 0)
    assert st.add_points(_wall_cloud(), _pose(translation=(0.0, 0.0, 0.0)), 0)
    before = st.snapshot()
    assert before["gravity"]["valid"] and before["grid"]["occupied_count"] > 0
    st.reset(1)
    snap = st.snapshot()
    assert snap["frame_id"] is None
    assert snap["gravity"]["valid"] is False
    assert snap["floor"]["valid"] is False
    assert snap["grid"]["occupied_count"] == 0
    assert snap["grid"]["free_count"] == 0
    assert snap["walls"] == []
    assert snap["path"] == []
    assert snap["pose"]["valid"] is False
    assert snap["pose"]["segment"] == 1
    assert snap["updated_age_s"] is None
    assert snap["revision"] > before["revision"]


def test_structure_payload_and_rebuild_cost(calibration):
    # Boundary checking is off here: the synthetic rays pass through the
    # thin test walls, so every wall is legitimately "seen from both sides"
    # and the boundary rule would (correctly) reject it.  Shape/serialization
    # cost is what this test measures.
    st = StructureMap(calibration, _structure_config(
        max_cells=20_000, wall_boundary_check=False))
    assert st.add_points(_floor_cloud(), _pose(translation=(0.0, 0.0, 0.0)), 0)
    # Twenty-four parallel 6 m x 2 m wall sheets, 0.3 m apart (>2 cells, so
    # the closing step cannot merge them): >2,000 wall cells in the payload.
    big = np.vstack([
        _wall_rect(-3.0, 1.0 + 0.3 * i, 3.0, 1.0 + 0.3 * i,
                   bottom_m=0.5, top_m=2.0)
        for i in range(24)
    ])
    started = time.monotonic()
    assert st.add_points(big, _pose(translation=(0.0, 0.0, 0.0)), 0)
    add_s = time.monotonic() - started
    started = time.monotonic()
    raw = st.body_bytes()
    build_s = time.monotonic() - started
    snap = st.snapshot()
    assert snap["grid"]["occupied_count"] > 2000
    assert snap["walls"]
    assert len(raw) < 200_000, f"structure payload {len(raw)} bytes"
    assert len(json.dumps(snap, allow_nan=False)) < 200_000
    # Structural mapping is cheap enough to run at 1 Hz on the worker.
    assert add_s < 2.0 and build_s < 2.0


def test_structure_http_schema_cors_and_payload(tmp_path):
    frame = FakeFrame()
    slam, _, cal = _make(tmp_path, frame,
                         [_tracked(_pose(translation=(0.0, 0.0, 0.0)))],
                         enable_point_cloud=False, enable_structure=False)
    st = StructureMap(cal, _structure_config())
    assert st.add_points(_floor_cloud(), _pose(translation=(0, 0, 0)), 0)
    assert st.add_points(_wall_cloud(), _pose(translation=(0, 0, 0)), 0)
    slam.structure = st
    server = SlamStatusServer(slam, port=0).start()
    try:
        with urllib.request.urlopen(
                f"http://127.0.0.1:{server.bound_port}/structure") as r:
            raw = r.read()
            body = json.loads(raw)
            assert r.status == 200
            assert r.headers["Access-Control-Allow-Origin"] == "*"
            assert r.headers["Cache-Control"] == "no-store"
        req = urllib.request.Request(
            f"http://127.0.0.1:{server.bound_port}/structure", data=b"",
            method="POST")
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(req, timeout=2.0)
        assert exc.value.code == 405
    finally:
        server.stop()
    assert list(body) == [
        "ok", "service", "session_id", "frame_id", "pose_frame",
        "structural", "persistent", "loop_corrected", "gravity_source",
        "gravity", "floor", "grid", "diagnostics", "walls", "wall_cells",
        "path", "pose", "trajectory", "status", "revision", "updated_age_s",
    ]
    assert body["frame_id"] == "map"
    assert body["diagnostics"]["wall_cells"] == len(body["wall_cells"])
    assert body["diagnostics"]["floor_rmse_m"] is not None
    assert body["floor"]["rmse_m"] is not None
    assert body["structural"] is True
    assert body["persistent"] is False
    assert body["loop_corrected"] is False
    assert body["gravity_source"] == "estimated_from_floor"
    assert body["session_id"] == slam.session_id
    assert body["status"]["session_id"] == slam.session_id
    assert len(raw) < 200_000
    json.dumps(body, allow_nan=False)


def test_point_cloud_is_off_by_default_and_map_stays_compatible(tmp_path):
    cfg = LiveSlamConfig()
    assert cfg.enable_point_cloud is False
    assert cfg.enable_structure is True
    frame = FakeFrame()
    slam, _, _ = _make(
        tmp_path, frame,
        [_tracked(_pose(translation=(0.0, 0.0, 0.0)))],
        enable_point_cloud=False, enable_structure=False)
    slam._tick()
    snap = slam.map_snapshot()
    assert snap["count"] == 0
    assert snap["map_points"] == 0
    assert snap["transmit_stride"] == 1
    assert snap["points"] == []
    assert len(slam.geometry._voxels) == 0


def test_structure_payload_bounded_at_cell_cap(calibration):
    st = StructureMap(calibration, _structure_config(
        max_cells=20_000, max_wall_cells_transmit=100))
    assert st.add_points(_floor_cloud(), _pose(translation=(0, 0, 0)), 0)
    st._cells.clear()
    # One 200-cell-long, 2 m tall wall-like run with free space below it.
    for iy in range(100):
        for ix in range(200):
            if iy == 0:
                st._cells[(ix, iy)] = [0, 1, 30, 200]
            else:
                st._cells[(ix, iy)] = [1, 0, 255, 0]
    st.revision += 1
    st._cached_revision = -1
    snap = st.snapshot()
    raw = json.dumps(snap, allow_nan=False)
    assert snap["grid"]["width"] * snap["grid"]["height"] == 20_000
    assert snap["grid"]["occupied_count"] == 200
    assert snap["grid"]["free_count"] == 99 * 200
    assert snap["diagnostics"]["occupied_cells"] == 200
    assert snap["diagnostics"]["wall_cells"] == 200
    assert len(snap["wall_cells"]) == 100  # transmit cap
    assert len(snap["walls"]) <= 500
    assert len(raw) < 200_000
