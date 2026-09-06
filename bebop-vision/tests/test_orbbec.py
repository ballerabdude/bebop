"""Orbbec camera service tests (no hardware: config parsing, profile
negotiation, intrinsics round-trip)."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from bebop_vision.orbbec import (_negotiate_depth_profile, _dump_intrinsics,
                                 intrinsics_path, load_intrinsics,
                                 load_rig_config)


def test_default_rig_config():
    cfg = load_rig_config()
    cams = cfg["robots"]["default"]["cameras"]
    assert set(cams) == {"CPBLC53000PE", "CPBLC53000ED"}
    assert cams["CPBLC53000PE"]["role"] == "near"
    assert cams["CPBLC53000ED"]["role"] == "far"
    assert cams["CPBLC53000PE"]["pitch_deg"] < cams["CPBLC53000ED"]["pitch_deg"] < 0
    bev = cfg["robots"]["default"]["bev"]
    assert (bev["range_m"], bev["width_m"], bev["cell_m"]) == (3.0, 3.0, 0.05)


class FakeProfile:
    def __init__(self, w, h, fps, fmt):
        self._w, self._h, self._fps, self._fmt = w, h, fps, fmt

    def as_video_stream_profile(self):
        return self

    def get_format(self):
        return self._fmt

    def get_width(self):
        return self._w

    def get_height(self):
        return self._h

    def get_fps(self):
        return self._fps


class FakeProfileList:
    def __init__(self, profiles):
        self._p = profiles

    def get_count(self):
        return len(self._p)

    def get_stream_profile_by_index(self, i):
        return self._p[i]


class FakeSensor:
    def __init__(self, profiles):
        self._p = profiles

    def get_stream_profile_list(self):
        return FakeProfileList(self._p)


@pytest.fixture
def fake_ob():
    return SimpleNamespace(OBFormat=SimpleNamespace(Y16="Y16", MJPG="MJPG",
                                                    RGB="RGB"))


PROFILES = [
    FakeProfile(1280, 800, 30, "MJPG"),
    FakeProfile(848, 480, 30, "Y16"),
    FakeProfile(848, 480, 15, "Y16"),
    FakeProfile(848, 480, 10, "Y16"),
    FakeProfile(640, 480, 30, "Y16"),
    FakeProfile(640, 360, 10, "Y16"),
]


def test_negotiate_exact_profile(fake_ob):
    sensor = FakeSensor(PROFILES)
    assert _negotiate_depth_profile(sensor, fake_ob, (848, 480, 30)) == (848, 480, 30)


def test_negotiate_drops_fps_on_usb2(fake_ob):
    # The far camera on the old cable advertises 848x480@10 only at our
    # preferred resolution — negotiation must land on 10 fps.
    sensor = FakeSensor([p for p in PROFILES if p.get_fps() <= 10])
    assert _negotiate_depth_profile(sensor, fake_ob, (848, 480, 30)) == (848, 480, 10)


def test_negotiate_falls_back_to_smaller_resolution(fake_ob):
    sensor = FakeSensor([FakeProfile(640, 480, 15, "Y16")])
    assert _negotiate_depth_profile(sensor, fake_ob, (848, 480, 30)) == (640, 480, 15)


def test_negotiate_ignores_non_y16(fake_ob):
    sensor = FakeSensor([FakeProfile(1280, 800, 30, "MJPG")])
    with pytest.raises(RuntimeError):
        _negotiate_depth_profile(sensor, fake_ob, (848, 480, 30))


def test_intrinsics_roundtrip(tmp_path):
    param = SimpleNamespace(
        depth_intrinsic=SimpleNamespace(width=848, height=480, fx=430.5,
                                        fy=430.1, cx=423.7, cy=238.9),
        depth_distortion=SimpleNamespace(model=0, k1=0.0, k2=0.0, p1=0.0,
                                         p2=0.0, k3=0.0, k4=0.0, k5=0.0,
                                         k6=0.0))
    pipeline = SimpleNamespace(get_camera_param=lambda: param)
    data = _dump_intrinsics(pipeline, "CPBLC53000PE", config_dir=tmp_path)
    path = intrinsics_path("CPBLC53000PE", tmp_path)
    assert path.exists()
    loaded = load_intrinsics("CPBLC53000PE", tmp_path)
    assert loaded == data
    assert loaded["fx"] == pytest.approx(430.5)
    assert loaded["width"] == 848
    with open(path) as f:
        on_disk = json.load(f)
    assert on_disk["serial"] == "CPBLC53000PE"
