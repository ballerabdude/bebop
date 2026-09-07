"""Orbbec camera service tests (no hardware: config parsing, profile
negotiation, intrinsics round-trip, self-view mask shapes)."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from bebop_vision.orbbec import (_negotiate_depth_profile, _dump_intrinsics,
                                 build_self_view_mask, intrinsics_path,
                                 load_intrinsics, load_rig_config,
                                 parse_self_mask_entries)


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


def test_parse_self_mask_entries():
    rects, polys = parse_self_mask_entries(
        [[336, 228, 560, 480],
         [[540, 375], [870, 377], [868, 748], [538, 746]],
         [0, 0, 100, 100]])
    assert rects == [(336, 228, 560, 480), (0, 0, 100, 100)]
    assert polys == [[[540, 375], [870, 377], [868, 748], [538, 746]]]


def test_parse_self_mask_entries_float_coords_are_scalars():
    # A rect written with float coords (YAML 0.0 style) is still a rect.
    rects, polys = parse_self_mask_entries([[1.0, 2.0, 3.0, 4.0]])
    assert rects == [(1, 2, 3, 4)] and polys == []


def test_parse_self_mask_entries_rejects_junk():
    with pytest.raises(ValueError):
        parse_self_mask_entries([[1, 2, "x", 4]])


def test_parse_self_mask_entries_empty():
    assert parse_self_mask_entries(None) == ([], [])
    assert parse_self_mask_entries([]) == ([], [])


def test_normalize_mask_shapes():
    from bebop_vision.orbbec import normalize_mask_shapes
    rects, polys = normalize_mask_shapes(
        [[1.0, 2.0, 3.0, 4.0]],
        [[[1.5, 2.5], [3.5, 4.5], [5.5, 6.5]]])
    assert rects == [(1, 2, 3, 4)]
    assert polys == [[[1, 2], [3, 4], [5, 6]]]
    assert normalize_mask_shapes(None, None) == ([], [])


def test_orbbec_camera_init_builds_polygon_mask(monkeypatch):
    # Regression: the constructor's polygon coercion crashed on the rig
    # polygon (map(int, poly) over vertex pairs). Exercises the full
    # constructor path with the hardware open + reader thread stubbed.
    from bebop_vision import orbbec as orb

    def fake_open(self):
        self.width, self.height, self.fps = 848, 480, 15
        self._filters = []

    monkeypatch.setattr(orb.OrbbecCamera, "_open", fake_open)
    monkeypatch.setattr(
        orb.threading, "Thread",
        lambda *a, **k: SimpleNamespace(start=lambda: None, join=lambda *a: None))
    rig_poly = [[632, 479], [530, 479], [500, 250], [350, 252],
                [334, 479], [280, 479]]
    cam = orb.OrbbecCamera("SER", "near", mask_polys=[rig_poly])
    assert cam.mask_polys == [rig_poly] and cam.mask_rects == []
    assert cam._pix_mask is not None
    assert int(cam._pix_mask.sum()) == 40009      # matches the rig polygon
    assert cam._pix_mask[479, 631] and cam._pix_mask[251, 500]
    # float coords from hand-edited YAML coerce to ints
    cam2 = orb.OrbbecCamera("SER", "near",
                            mask_rects=[[1.0, 2.0, 3.0, 4.0]],
                            mask_polys=[[[1.5, 2.5], [3.5, 4.5], [5.5, 6.5]]])
    assert cam2.mask_rects == [(1, 2, 3, 4)]
    assert cam2._pix_mask[2, 1] and cam2._pix_mask[3, 2]


def test_build_self_view_mask_rect():
    m = build_self_view_mask([(10, 20, 30, 40)], [], 100, 80)
    assert m is not None and m.shape == (80, 100)
    assert m[20:40, 10:30].all()
    assert not m[0:20, :].any() and not m[:, 30:].any()


def test_build_self_view_mask_trapezoid_polygon():
    # A trapezoid that widens toward the bottom: the rect hull would mask
    # the upper side floor, the polygon must not.
    poly = [[20, 20], [60, 20], [80, 79], [0, 79]]
    m = build_self_view_mask([], [poly], 100, 80)
    assert m[40, 50]                      # inside
    assert not m[25, 12]                  # outside at the narrow top,
    assert m[75, 5]                       # but inside at the wide bottom
    # ...and it must cover strictly less than the bounding rect.
    r = build_self_view_mask([(0, 20, 80, 80)], [], 100, 80)
    assert (r & ~m).any()


def test_build_self_view_mask_clamps():
    m = build_self_view_mask([(-10, -10, 2000, 2000)], [], 100, 80)
    assert m.all()
    m = build_self_view_mask([], [[[50, 50], [500, 50], [500, 500],
                                   [50, 500]]], 100, 80)
    assert m[60, 60] and m[79, 99] and not m[40, 60]


def test_build_self_view_mask_empty():
    assert build_self_view_mask([], [], 848, 480) is None


def test_build_self_view_mask_union():
    m = build_self_view_mask([(0, 0, 10, 10)],
                             [[[50, 50], [60, 50], [60, 60], [50, 60]]],
                             100, 80)
    assert m[5, 5] and m[55, 55]
    assert not m[30, 30]
