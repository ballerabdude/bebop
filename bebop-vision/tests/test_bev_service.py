"""BevService wiring tests (fake rig, no hardware)."""

from test_master_bev import make_builder, scene

from bebop_vision.bev_service import BevService


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


def test_tick_once_publishes_a_grid_and_jpeg():
    rig = FakeRig(scene())
    video = FakeVideo()
    svc = BevService(rig, builder=make_builder(), hz=20,
                     video_server=video)
    grid = svc.tick_once()
    assert grid is not None
    assert grid.shape == (120, 60)
    assert grid.contributions["far"]["points"] > 0
    assert svc.latest() is grid
    assert svc.latest_jpeg() is not None
    assert len(video.published) == 1
    assert video.published[0][0] is grid
    assert video.published[0][1] is not None

    status = svc.status()
    assert status["grid"]["rows"] == 120
    assert status["grid"]["cols"] == 60
    assert status["has_jpeg"] is True
    assert status["running"] is False  # not started; tick_once is manual


def test_tick_once_with_one_stale_camera_still_fuses():
    frames = scene()
    # The near frame is fresh; the far one is old enough to be dropped.
    frames["far"].recv_ts -= 10.0
    rig = FakeRig(frames)
    svc = BevService(rig, builder=make_builder(), hz=10)
    grid = svc.tick_once()
    assert grid is not None
    assert grid.roles == ["near"]
    assert "far" not in grid.contributions
