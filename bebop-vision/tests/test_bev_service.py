"""Model-only BevService wiring tests (fake runtime, no hardware/TRT)."""

import time

import numpy as np

from bebop_vision.bev_service import BevService
from bebop_vision.master_bev import BevGrid
from bebop_vision.orbbec import StampedFrame


class FakeCam:
    def __init__(self, frame):
        self.frame = frame

    def read(self):
        return self.frame


class FakeRig:
    def __init__(self, frames):
        self.cameras = {role: FakeCam(f) for role, f in frames.items()}


class FakeVideo:
    def __init__(self):
        self.published = []

    def publish_bev(self, grid, jpeg=None):
        self.published.append((grid, jpeg))


class FakeRuntime:
    engine_path = "fake.engine"
    in_shape = [2, 4, 240, 424]
    range_m = 6.0
    width_m = 3.0
    cell_m = 0.05

    def __init__(self):
        self.calls = 0

    def infer(self, frames, geo_grid=None, ages_s=None):
        self.calls += 1
        assert geo_grid is None
        shape = (120, 60)
        return BevGrid(
            occ=np.zeros(shape, np.uint8),
            raw=np.zeros(shape, np.uint8),
            coverage=np.ones(shape, bool),
            source=np.full(shape, 3, np.uint8),
            confidence=np.full(shape, 200, np.uint8),
            cell_m=0.05, range_m=6.0, width_m=3.0,
            stamp_us=1234, recv_ts=time.monotonic(),
            per_camera_age_s=ages_s or {}, plane_ok={}, plane={},
            contributions={"model": {"engine": "fake.engine",
                                     "hazard_source": "model",
                                     "infer_ms": 1.0}},
            roles=sorted(frames),
            semantic=np.zeros(shape, np.uint8),
            wall=np.zeros(shape, bool),
            ego=np.zeros(shape, bool))

    def status(self):
        return {"engine": "fake.engine", "infer_ms": 1.0,
                "hazard_source": "model"}


def _frame(role, stamp_us=10):
    return StampedFrame(
        depth=np.zeros((480, 848), np.uint16), stamp_us=stamp_us,
        recv_ts=time.monotonic(), width=848, height=480, fps=30.0,
        color=None, color_jpeg=b"jpeg", serial="S-" + role, role=role)


def test_tick_once_publishes_a_model_grid_and_jpeg():
    rig = FakeRig({"near": _frame("near"), "far": _frame("far")})
    video = FakeVideo()
    svc = BevService(rig, runtime=FakeRuntime(), hz=20, video_server=video)
    grid = svc.tick_once()
    assert grid is not None
    assert grid.shape == (120, 60)
    assert "model" in grid.contributions
    assert grid.semantic is not None
    assert svc.latest() is grid
    assert svc.latest_jpeg() is not None
    assert len(video.published) == 1
    assert video.published[0][0] is grid
    assert video.published[0][1] is not None

    status = svc.status()
    assert status["grid"]["rows"] == 120
    assert status["grid"]["cols"] == 60
    assert status["has_jpeg"] is True
    assert status["running"] is False
    assert status["model"]["engine"] == "fake.engine"


def test_stale_camera_does_not_fall_back():
    frames = {"near": _frame("near"), "far": _frame("far")}
    frames["far"].recv_ts -= 10.0
    svc = BevService(FakeRig(frames), runtime=FakeRuntime(), hz=10)
    assert svc.tick_once() is None
    assert "missing" in svc._last_error
    assert svc.latest() is None
