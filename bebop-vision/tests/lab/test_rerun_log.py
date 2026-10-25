"""bebop_vision.lab.rerun_log: the lab's Rerun view.

No viewer is spawned: ``show(..., spawn=False, save=...)`` writes an .rrd
which is read back with ``rerun.recording.load_recording`` to check that
the MCAP and the lab entities landed on the same timeline. The session,
run and label objects are minimal fakes of the CONTRACT interfaces (the
real modules are written in parallel and are not imported here).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

import synth
from bebop_vision.lab import rerun_log
from bebop_vision.lab.types import (Calib, Classes, Embedding, Intrinsics, Mask, Path2D,
                                    Scalars)

rr = pytest.importorskip("rerun")

# A colour camera the size of the synthetic colour frames (the MCAP carries
# no colour intrinsics; a real session gets them from
# config/orbbec_intrinsics_<serial>.json when the file exists).
COLOR_K = Intrinsics(fx=100.0, fy=100.0, cx=64.0, cy=40.0, width=128, height=80)


def synthetic_calib(with_color: bool = True) -> Calib:
    dw, dh = synth.DEPTH_WH
    depth = {r: Intrinsics(51.7, 51.7, dw / 2, dh / 2, dw, dh) for r in synth.SERIALS}
    return Calib(roles=dict(synth.SERIALS), depth=depth,
                 color={"near": COLOR_K} if with_color else {},
                 mounts={r: dict(m) for r, m in synth.MOUNTS.items()})


class FakeFrames:
    def __init__(self, t_ns):
        self.t_ns = np.asarray(t_ns)

    def at(self, t_ns):
        i = int(np.argmin(np.abs(self.t_ns - int(t_ns))))
        return synth.colour_frame(i)


class FakeTicks:
    def __init__(self, t_ns):
        self.t_ns = np.asarray(t_ns, dtype=np.int64)

    def __len__(self):
        return len(self.t_ns)


class FakeSession:
    def __init__(self, path: Path, n: int = 80, hz: float = 20.0, with_color: bool = True):
        self.path = Path(path)
        self.name = self.path.stem
        self.calib = synthetic_calib(with_color)
        self._t = np.array([synth.T0_NS + i * int(round(1e9 / hz)) for i in range(n)], np.int64)

    def ticks(self, hz=None, start_ns=None, end_ns=None):
        return FakeTicks(self._t)

    def frames(self, role):
        return FakeFrames(self._t)


class FakeRun:
    def __init__(self, name, t_ns, outputs):
        self.t_ns = np.asarray(t_ns, dtype=np.int64)
        self.outputs = outputs
        self.meta = {"name": name}


class FakeLabels:
    variant = "hindsight"

    def __init__(self, t_ns, paths, valid):
        self.t_ns = np.asarray(t_ns, dtype=np.int64)
        self._d = {int(t): {"path": p, "valid": bool(v)} for t, p, v in zip(t_ns, paths, valid)}

    def get(self, t_ns):
        return self._d.get(int(t_ns))


def arc_path(v=0.3, w=0.4, horizon_s=3.0, dt_s=0.2, k=1) -> Path2D:
    """Constant-twist arcs; hypothesis j turns a little more than j-1."""
    rel_t = np.arange(0.0, horizon_s + 1e-9, dt_s)
    pts = []
    for j in range(k):
        wj = w * (1.0 + 0.3 * j)
        th = wj * rel_t
        if abs(wj) < 1e-9:
            pts.append(np.stack([v * rel_t, np.zeros_like(rel_t)], -1))
        else:
            pts.append(np.stack([v / wj * np.sin(th), v / wj * (1 - np.cos(th))], -1))
    weights = np.linspace(1.0, 0.5, k)
    return Path2D(rel_t=rel_t, pts=np.stack(pts), weights=weights / weights.sum())


def fake_run(t_ns, name="cv"):
    n = len(t_ns)
    rng = np.random.default_rng(0)
    return FakeRun(name, t_ns, {
        "path": [arc_path(k=2) for _ in range(n)],
        # a 10x16 mask at stride 8 covering the full 80x128 frame; class 1
        # (obstacle) in the top-left corner so the canvas has both ids
        "mask": [Mask("near", np.pad(np.ones((3, 4), np.uint8), ((0, 7), (0, 12))),
                      classes=("floor", "obstacle"), stride=8) for _ in range(n)],
        "embedding": [Embedding(rng.normal(size=16).astype(np.float32), "fake") for _ in range(n)],
        "scalars": [Scalars({"speed": 0.3, "turn": 0.4}) for _ in range(n)],
        "classes": [Classes(np.array([0.2, 0.8]), ("stop", "go")) for _ in range(n)],
    })


def _entities(rrd_path):
    from rerun.recording import load_recording
    rec = load_recording(str(rrd_path))
    schema = rec.schema()
    paths = {str(p) for p in schema.entity_paths()}
    timelines = {str(c) for c in schema.index_columns()}
    comps = {(str(c.entity_path), str(c.component)) for c in schema.component_columns()}
    return paths, timelines, comps


# --------------------------------------------------------------------------

def test_show_writes_rrd_with_mcap_and_lab_entities(h265_session, tmp_path, monkeypatch):
    def no_spawn(*a, **k):
        raise AssertionError("show() must not spawn a viewer in tests")
    monkeypatch.setattr(rr.RecordingStream, "spawn", no_spawn)

    session = FakeSession(h265_session)
    t = session.ticks().t_ns
    run = fake_run(t)
    valid = np.ones(len(t), bool)
    valid[-5:] = False                           # the horizon runs past the end
    labels = FakeLabels(t, [arc_path() for _ in t], valid)

    out = tmp_path / "x.rrd"
    rerun_log.show(session, runs=[run], labels=labels, spawn=False, save=out)
    assert out.is_file() and out.stat().st_size > 10_000

    paths, timelines, comps = _entities(out)
    # the MCAP itself was loaded, under the lab's own timeline
    assert "/odom" in paths and "/color_near" in paths
    assert f"Index(timeline:{rerun_log.TIMELINE})" in timelines
    # plan view: footprint (static) + run + label strips
    assert ("/lab/plan/footprint", "LineStrips2D:strips") in comps
    assert ("/lab/plan/cv", "LineStrips2D:strips") in comps
    assert any(p.startswith("/lab/plan/label") and p.endswith("hindsight") for p in paths)
    # projection into the near image (the fake calib has colour intrinsics)
    assert ("/color_near/lab/cv", "LineStrips2D:strips") in comps
    # mask overlay under the image entity with its class names
    assert ("/color_near/lab/cv/mask", "SegmentationImage:buffer") in comps
    assert ("/color_near/lab/cv/mask", "AnnotationContext:context") in comps
    # overlays sit in the video's image-plane frame (else 0.31 draws nothing)
    assert ("/color_near/lab/cv", "CoordinateFrame:frame") in comps
    assert ("/color_near/lab/cv/mask", "CoordinateFrame:frame") in comps
    # embedding PCA, scalars and classes as series under /lab/<run>
    for leaf in ("embedding/pc0", "embedding/pc1", "embedding/pc2", "scalars/speed",
                 "scalars/turn", "classes/go"):
        assert ("/lab/cv/" + leaf, "Scalars:scalars") in comps, leaf


def test_show_without_colour_intrinsics_skips_projection(h265_session, tmp_path):
    session = FakeSession(h265_session, n=10, with_color=False)
    run = FakeRun("cv", session.ticks().t_ns, {"path": [arc_path() for _ in range(10)]})
    out = tmp_path / "nocolor.rrd"
    rerun_log.show(session, runs=[run], spawn=False, save=out)
    paths, _, comps = _entities(out)
    assert ("/lab/plan/cv", "LineStrips2D:strips") in comps
    assert "/color_near/lab/cv" not in paths


def test_blueprint_builds_with_lab_tab():
    import rerun.blueprint as rrb
    bp = rerun_log.build_blueprint(["cv", "label:hindsight"])
    assert isinstance(bp, rrb.Blueprint)
    left, tabs = bp.root_container.contents
    assert isinstance(tabs, rrb.Tabs)
    navd = rerun_log._dashboard_module("navd_blueprint")
    assert len(tabs.contents) == len(navd.TABS) + 1
    assert tabs.active_tab == len(navd.TABS)
    lab = tabs.contents[-1]
    cams, right = lab.contents                   # camera column | plan + series
    origins = [v.origin for v in cams.contents]
    assert [str(o) for o in origins] == ["/color_near", "/color_far"]
    views = list(right.contents)
    assert isinstance(views[0], rrb.Spatial2DView)
    assert len(views) == 3                       # plan + one series view per run
    # it serialises (what send_blueprint does)
    bp.to_blueprint()
    # the plain dashboard is still what extra_views=False gets
    assert len(navd.build().root_container.contents[1].contents) == len(navd.TABS)


def test_image_frame_id_reads_the_recorded_frame(h265_session):
    assert rerun_log.image_frame_id(FakeSession(h265_session), "near") == "camera_near"
    class NoPath:
        path = None
    assert rerun_log.image_frame_id(NoPath(), "far") == "far_color"


def test_plan_xy_puts_forward_up_and_left_left():
    uv = rerun_log.plan_xy(np.array([[1.0, 0.0], [0.0, 1.0]]))
    assert np.allclose(uv[0], [0.0, -1.0])      # x forward -> up (negative v)
    assert np.allclose(uv[1], [-1.0, 0.0])      # y left -> left (negative u)
    box = rerun_log.footprint_strip()
    assert box.shape == (5, 2) and np.allclose(box[0], box[-1])
    x0, x1, y0, y1 = rerun_log.footprint_rect()
    assert np.ptp(box[:, 0]) == pytest.approx(y1 - y0) and np.ptp(box[:, 1]) == pytest.approx(x1 - x0)
    # the front of the body (max x) is the top of the box (min v)
    assert box[:, 1].min() == pytest.approx(-x1) and box[:, 0].min() == pytest.approx(-y1)


def test_footprint_rect_reads_the_rig_ego_rectangle(tmp_path):
    """The plan view draws the measured chassis (config/orbbec_rig.yaml
    bev_student.ego), not a box centred on the mast."""
    import yaml
    from bebop_vision.orbbec import DEFAULT_RIG_YAML
    rig = yaml.safe_load(Path(DEFAULT_RIG_YAML).read_text(encoding="utf-8"))
    ego = rig["robots"]["default"]["bev_student"]["ego"]
    rect = rerun_log.footprint_rect()
    assert rect == (ego["x_range_m"][0], ego["x_range_m"][1], ego["y_range_m"][0], ego["y_range_m"][1])
    assert rect[1] > -rect[0]                     # longer ahead of the mast than behind
    # an explicit config wins; unordered ranges are sorted; missing keys fall back
    assert rerun_log.footprint_rect({"robots": {"default": {"bev_student": {"ego": {
        "x_range_m": [0.3, -0.1], "y_range_m": [-0.2, 0.2]}}}}}) == (-0.1, 0.3, -0.2, 0.2)
    assert rerun_log.footprint_rect({"robots": {}}) == rerun_log.FOOTPRINT_XY


def test_project_floor_point_ahead_lands_in_image():
    cam = synthetic_calib().camera("near")
    uv, keep = rerun_log.project_floor(cam, np.array([[1.0, 0.0]]))
    assert keep.tolist() == [True]
    u, v = uv[0]
    # the near camera looks 60 deg down from 1.27 m; 1 m ahead is 51.8 deg
    # below horizontal, i.e. 8.2 deg above the optical axis -> above centre
    expect_v = COLOR_K.cy - COLOR_K.fy * math.tan(math.radians(60.0) - math.atan2(1.27, 1.0))
    assert u == pytest.approx(COLOR_K.cx, abs=1e-6)
    assert v == pytest.approx(expect_v, abs=1e-3)
    assert 0 <= u < COLOR_K.width and 0 <= v < COLOR_K.height
    # behind the camera and far to the side are dropped
    uv2, keep2 = rerun_log.project_floor(cam, np.array([[-2.0, 0.0], [1.0, 0.0], [1.0, 50.0]]))
    assert keep2.tolist() == [False, True, False] and len(uv2) == 1


def test_mask_canvas_upsamples_shifts_and_pads():
    m = Mask("near", np.array([[0, 1], [2, 0]], np.uint8), classes=("a", "b", "c"),
             stride=4, crop=(8, 16, 4, 12))
    c = rerun_log.mask_canvas(m, (32, 32))
    assert c.shape == (32, 32) and c.dtype == np.uint8
    assert c[:8].max() == 0 and c[16:].max() == 0 and c[:, :4].max() == 0   # uncovered -> 0
    assert c[8, 4] == 1 and c[8, 8] == 2 and c[12, 4] == 3 and c[12, 8] == 1
    # float [C, H, W] takes the argmax; no crop fills the mask's own extent
    probs = np.zeros((2, 3, 3), np.float32)
    probs[1, 1, 1] = 1.0
    c2 = rerun_log.mask_canvas(Mask("far", probs))
    assert c2.shape == (3, 3) and c2[1, 1] == 2 and c2[0, 0] == 1


def test_mask_class_count_uses_channels_for_probability_masks():
    probs = np.zeros((3, 4, 4), np.float32)
    probs[0] += 0.9                               # max value < 1: the old max+1 rule said 1 class
    assert rerun_log._mask_class_count([Mask("near", probs)]) == 3
    assert rerun_log._mask_class_count([None, Mask("near", np.zeros((2, 2), np.uint8)),
                                        Mask("near", np.full((2, 2), 4, np.uint8))]) == 5
    assert rerun_log._mask_class_count([None]) == 0
    assert rerun_log._masks_extent([Mask("near", np.zeros((2, 3), np.uint8), stride=4),
                                    Mask("near", np.zeros((1, 1)), crop=(20, 30, 0, 5)), None]) == (30, 12)


def _mask_series(rrd_path, entity):
    """``(t_ns [T], canvases [T, H, W])`` of a SegmentationImage entity."""
    from rerun.recording import load_recording
    rec = load_recording(str(rrd_path))
    ts, out = [], []
    # the SDK batcher may merge several send_columns into one physical chunk
    # or keep them apart; either way every temporal chunk of the entity counts
    for chunk in rec.chunks():
        if str(chunk.entity_path) != entity or chunk.is_static:
            continue
        rb = chunk.to_record_batch()
        names = rb.schema.names
        if "SegmentationImage:buffer" not in names:
            continue
        for row in rb.to_pylist():
            f = row["SegmentationImage:format"][0]
            h, w = int(f["height"]), int(f["width"])
            ts.append(np.datetime64(row[rerun_log.TIMELINE], "ns").astype(np.int64))
            out.append(np.asarray(row["SegmentationImage:buffer"][0], np.uint8).reshape(h, w))
    order = np.argsort(ts, kind="stable")
    return np.asarray(ts, np.int64)[order], np.stack(out)[order]


def test_masks_are_sent_in_bounded_chunks(h265_session, tmp_path, monkeypatch):
    """A long session's masks must not be stacked in RAM at once; chunks
    land on the same entity and still cover every tick, Nones cleared."""
    session = FakeSession(h265_session, n=20)
    t = session.ticks().t_ns
    masks = [Mask("near", np.full((10, 16), i % 2, np.uint8), classes=("a", "b"), stride=8)
             for i in range(20)]
    masks[7] = None
    run = FakeRun("seg", t, {"mask": masks})
    # 80x128 canvases: 10 KB each -> 3 rows per chunk -> 7 send_columns calls
    monkeypatch.setattr(rerun_log, "MASK_CHUNK_BYTES", 3 * 80 * 128)
    sends = []
    orig = rr.RecordingStream.send_columns

    def spy(self, path, *a, **k):
        if path.endswith("/mask"):
            sends.append(len(k["indexes"][0].as_arrow_array()))
        return orig(self, path, *a, **k)
    monkeypatch.setattr(rr.RecordingStream, "send_columns", spy)
    out = tmp_path / "chunks.rrd"
    rerun_log.show(session, runs=[run], spawn=False, save=out)
    assert sends == [3] * 6 + [2]
    tt, canv = _mask_series(out, "/color_near/lab/seg/mask")
    assert tt.tolist() == t.tolist() and canv.shape == (20, 80, 128)
    ids = canv.reshape(20, -1).max(axis=1)
    assert ids[7] == 0 and ids[6] == 1 and ids[8] == 1     # ids shifted by one; None -> cleared


def test_masks_with_mixed_crops_share_one_canvas_without_intrinsics(h265_session, tmp_path, monkeypatch):
    """No colour intrinsics and no decodable frame: the canvas is the union
    of the masks' extents rather than each mask's own crop."""
    session = FakeSession(h265_session, n=4, with_color=False)
    monkeypatch.setattr(FakeFrames, "at", lambda self, t: None)
    t = session.ticks().t_ns
    masks = [Mask("far", np.ones((10, 10), np.uint8), crop=(0, 10, 0, 10)),
             Mask("far", np.ones((10, 10), np.uint8), crop=(20, 30, 20, 30)),
             None,
             Mask("far", np.ones((2, 2), np.uint8), stride=4)]
    out = tmp_path / "crops.rrd"
    rerun_log.show(session, runs=[FakeRun("seg", t, {"mask": masks})], spawn=False, save=out)
    tt, canv = _mask_series(out, "/color_far/lab/seg/mask")
    assert canv.shape == (4, 30, 30)
    # class 1 is logged as id 2 (ids shift by one so 0 stays transparent)
    assert canv[0, 5, 5] == 2 and canv[0, 25, 25] == 0
    assert canv[1, 25, 25] == 2 and canv[1, 5, 5] == 0
    assert canv[2].max() == 0 and canv[3, 7, 7] == 2 and canv[3, 8, 8] == 0


def test_image_hw_probes_the_frame_series_first_stamp(h265_session, monkeypatch):
    """The far camera drops frames: a missing frame at the run's first tick
    must not cache None when the series has frames elsewhere."""
    session = FakeSession(h265_session, n=4, with_color=False)
    first = session.ticks().t_ns[0]

    def at(self, t_ns):
        return None if int(t_ns) == int(first) + 50_000_000 else synth.colour_frame(0)
    monkeypatch.setattr(FakeFrames, "at", at)
    logger = rerun_log._Logger(None, rr, session)
    hw = logger.image_hw("far", [int(first) + 50_000_000])
    assert hw == synth.colour_frame(0).shape[:2]


def test_show_cleans_up_when_prepare_or_logging_fails(h265_session, tmp_path, monkeypatch):
    session = FakeSession(h265_session, n=4)
    t = session.ticks().t_ns
    mcap_prep = rerun_log._dashboard_module("mcap_prep")
    tmp_dir = tmp_path / "tmp"
    tmp_dir.mkdir()
    monkeypatch.setenv("TMPDIR", str(tmp_dir))
    import tempfile
    tempfile.tempdir = None                       # re-read TMPDIR
    try:
        # a failing prepare leaves no temp .mcap behind
        def boom(path, tmp):
            raise OSError("corrupt")
        monkeypatch.setattr(mcap_prep, "prepare", boom)
        with pytest.raises(OSError, match="corrupt"):
            rerun_log.show(session, runs=[], spawn=False, save=tmp_path / "a.rrd")
        assert list(tmp_dir.iterdir()) == []
        monkeypatch.undo()
        monkeypatch.setenv("TMPDIR", str(tmp_dir))
        tempfile.tempdir = None

        # a run that fails while logging still flushes and disconnects the
        # stream, so the .rrd is a complete recording of what was logged
        calls = []
        for name in ("flush", "disconnect"):
            orig = getattr(rr.RecordingStream, name)
            monkeypatch.setattr(rr.RecordingStream, name,
                                (lambda o, n: lambda self, *a, **k: (calls.append(n), o(self, *a, **k))[1])(orig, name))
        bad = FakeRun("bad", t, {"classes": [Classes(np.array([0.5]), ("a", "b"))] * 4})
        with pytest.raises(IndexError):
            rerun_log.show(session, runs=[bad], spawn=False, save=tmp_path / "b.rrd")
        assert calls == ["flush", "disconnect"]
        assert list(tmp_dir.iterdir()) == []
        paths, _, _ = _entities(tmp_path / "b.rrd")
        assert "/odom" in paths
    finally:
        tempfile.tempdir = None


def test_pca3_orders_variance_and_pads():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(50, 8)) * np.array([5, 3, 1, .1, .1, .1, .1, .1])
    s = rerun_log.pca3(X)
    assert s.shape == (50, 3)
    var = s.var(axis=0)
    assert var[0] > var[1] > var[2] > 0
    assert rerun_log.pca3(np.ones((4, 2))).shape == (4, 3)   # rank 0 -> zeros
    assert rerun_log.pca3(np.zeros((0, 5))).shape == (0, 3)


def test_labels_as_run_and_unique_names():
    t = np.arange(3) * 10 + 100
    labels = FakeLabels(t, [arc_path(), None, arc_path()], [True, True, False])
    view = rerun_log.labels_as_run(labels)
    assert view.name == "label:hindsight" and view.kinds() == {"path": Path2D}
    assert [p is not None for p in view.outputs["path"]] == [True, False, False]
    a, b = (rerun_log.run_view(FakeRun("cv", t, {"path": [None] * 3})) for _ in range(2))
    rerun_log._unique_names([a, b])
    assert (a.name, b.name) == ("cv", "cv#2")
    assert a.leaf("path") == "cv" and a.leaf("alt") == "cv.alt"
