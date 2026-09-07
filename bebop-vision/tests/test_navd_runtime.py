"""NavdGridSource runtime unit tests (plan §7.3).

Synthetic only: a stub model stands in for the ONNX session, so grid
production, class mapping and the no-grid failure paths are tested
without torch/onnxruntime. The real-ONNX parity test is skipped unless
torch + onnxruntime are importable (they are declared deps, but the
Jetson's torch comes from a separate index).

Note on the frac_navigable guard: implausible uniform outputs (everything
navigable or everything blocked) yield no grid, so every "model wins"
fixture paints a plausible scene — a blocked far wall plus a small
obstacle band.
"""

import importlib.util
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from bebop_vision.bev import FREE, OCCUPIED, HAZARD, INFLATED
from bebop_vision.goal_planner import GoalHeading, GoalPlanner, GoalPoint
from bebop_vision.navd_pre import (IMG_H, IMG_W, build_goal_raster,
                                   prep_color, prep_depth)
from bebop_vision.navd_runtime import (BLOCKED, NAVIGABLE, CAUTION,
                                       NavdGridSource, gather_frames,
                                       goal_raster_from_slot)
from bebop_vision.orbbec import StampedFrame

ROWS = COLS = 60
CELL = 0.05


def plausible_scene():
    """(60, 60) class map the frac_navigable guard accepts: far wall
    (rows 0..9) + everything else navigable."""
    cls = np.full((ROWS, COLS), NAVIGABLE, np.uint8)
    cls[0:10, :] = BLOCKED
    return cls


def make_frame(depth_mm, stamp_us, age_s=0.01, color_jpeg=None, color=None):
    return StampedFrame(depth=depth_mm, stamp_us=stamp_us,
                        recv_ts=time.monotonic() - age_s,
                        width=depth_mm.shape[1], height=depth_mm.shape[0],
                        fps=15.0, color=color, color_jpeg=color_jpeg,
                        serial="S", role="x")


def logits_for(cls_map):
    """(60, 60) int class map -> (3, 60, 60) one-hot-ish logits."""
    logits = np.zeros((3, ROWS, COLS), np.float32)
    for c in (BLOCKED, NAVIGABLE, CAUTION):
        logits[c][cls_map == c] = 5.0
    return logits


class StubModel:
    """Stand-in for NavdModel: returns a preset logits array (or raises)."""

    def __init__(self, logits=None, error=None):
        self.logits = logits
        self.error = error

    def predict(self, depth_near_mm, depth_far_mm, color_rgb, goal_raster):
        if self.error is not None:
            raise self.error
        return self.logits


def make_source(logits=None, error=None, **kw):
    src = NavdGridSource.__new__(NavdGridSource)   # skip the ONNX session
    src.model = StubModel(logits, error)
    src.max_frame_age_s = kw.get("max_frame_age_s", 0.3)
    src.frac_lo, src.frac_hi = kw.get("frac_range", (0.05, 0.95))
    src._log = kw.get("log", print)
    src.last_reason = None
    src._last_logged = None
    src._color_stamp = None
    src._color_rgb = None
    src._lock = threading.Lock()
    return src


def sample_frames(color="array"):
    dn = np.full((480, 848), 2000, np.uint16)
    df = np.full((480, 848), 2500, np.uint16)
    c = np.zeros((480, 848, 3), np.uint8) if color == "array" else None
    return {"near": make_frame(dn, 1000, color=c),
            "far": make_frame(df, 1000)}


# --- grid production ---------------------------------------------------------

def test_model_grid_maps_classes():
    cls = plausible_scene()
    cls[40:44, 20:24] = CAUTION     # an unconfirmed band mid-floor
    src = make_source(logits_for(cls))
    grid = src.update(sample_frames(), None, (0.0, 0.0, 0.0))
    assert grid.shape == (ROWS, COLS) and grid.cell_m == CELL
    assert (grid.occ[0:10, :] == OCCUPIED).all()
    assert (grid.occ[40:44, 20:24] == INFLATED).all()
    rest = grid.occ.copy()
    rest[0:10, :] = FREE
    rest[40:44, 20:24] = FREE
    assert (rest == FREE).all()
    # raw telemetry: caution reads as HAZARD, blocked as OCCUPIED
    assert (grid.raw[40:44, 20:24] == HAZARD).all()
    assert (grid.raw[0:10, :] == OCCUPIED).all()
    # model-only: no geometric plane fit rides along
    assert grid.plane_ok == {}
    assert grid.stamp_us == 1000
    assert set(grid.per_camera_age_s) == {"near", "far"}
    assert src.last_reason is None


def test_planner_blocks_on_model_caution():
    """The grid is the seam: a caution cell must stop the polar march."""
    cls = plausible_scene()
    cls[50:52, 28:32] = CAUTION     # ~0.5 m ahead, dead ahead
    src = make_source(logits_for(cls))
    grid = src.update(sample_frames(), None, (0.0, 0.0, 0.0))
    vx, wz, info = GoalPlanner().compute(grid, GoalHeading(0.0))
    assert info["state"] == "hard_stop"
    assert vx == 0.0 and wz == 0.0


# --- no-grid failures (no fallback — the drive node waits) --------------------

def test_missing_far_yields_no_grid():
    src = make_source(logits_for(plausible_scene()))
    frames = sample_frames()
    frames.pop("far")
    assert src.update(frames, None, (0.0, 0.0, 0.0)) is None
    assert "near+far" in src.last_reason


def test_stale_frame_yields_no_grid():
    src = make_source(logits_for(plausible_scene()))
    frames = sample_frames()
    frames["far"] = make_frame(np.full((480, 848), 2500, np.uint16), 1000,
                               age_s=1.0)
    assert src.update(frames, None, (0.0, 0.0, 0.0)) is None
    assert src.last_reason == "camera frame stale"


def test_nan_logits_yield_no_grid():
    bad = logits_for(plausible_scene())
    bad[0, 0, 0] = np.nan
    src = make_source(bad)
    assert src.update(sample_frames(), None, (0.0, 0.0, 0.0)) is None
    assert src.last_reason == "non-finite logits"


@pytest.mark.parametrize("cls_fill", [BLOCKED, NAVIGABLE])
def test_implausible_frac_yields_no_grid(cls_fill):
    """Fully uniform predictions are implausible either way (§7.3)."""
    cls = np.full((ROWS, COLS), cls_fill, np.uint8)
    src = make_source(logits_for(cls))
    assert src.update(sample_frames(), None, (0.0, 0.0, 0.0)) is None
    assert "frac_navigable" in src.last_reason


def test_no_color_yields_no_grid():
    src = make_source(logits_for(plausible_scene()))
    frames = sample_frames(color="none")   # no color array, no jpeg
    assert src.update(frames, None, (0.0, 0.0, 0.0)) is None
    assert src.last_reason == "no decodable near color"


def test_exception_yields_no_grid_then_recovers():
    src = make_source(error=RuntimeError("CUDA exploded"))
    assert src.update(sample_frames(), None, (0.0, 0.0, 0.0)) is None
    assert src.last_reason == "RuntimeError: CUDA exploded"
    # next good tick serves a grid again
    src.model.logits, src.model.error = logits_for(plausible_scene()), None
    grid = src.update(sample_frames(), None, (0.0, 0.0, 0.0))
    assert grid is not None
    assert (grid.occ[0:10, :] == OCCUPIED).all()
    assert src.last_reason is None


def test_failure_logged_once_per_transition():
    logs = []
    src = make_source(error=RuntimeError("x"), log=logs.append)
    src.update(sample_frames(), None, (0.0, 0.0, 0.0))
    src.update(sample_frames(), None, (0.0, 0.0, 0.0))   # same failure
    src.model.logits, src.model.error = logits_for(plausible_scene()), None
    src.update(sample_frames(), None, (0.0, 0.0, 0.0))
    src.update(sample_frames(), None, (0.0, 0.0, 0.0))   # still good
    assert len([m for m in logs if "no model grid" in m]) == 1
    assert len([m for m in logs if "recovered" in m]) == 1


# --- inputs ------------------------------------------------------------------

def test_color_from_camera_jpeg_is_decoded_and_cached():
    img = np.full((240, 424, 3), 90, np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    src = make_source(logits_for(plausible_scene()))
    frames = sample_frames(color="none")
    frames["near"] = make_frame(np.full((480, 848), 2000, np.uint16), 1000,
                                color_jpeg=buf.tobytes())
    grid = src.update(frames, None, (0.0, 0.0, 0.0))
    assert grid is not None
    # same device stamp -> cached decode, no second imdecode
    first = src._color_rgb
    src.update(frames, None, (0.0, 0.0, 0.0))
    assert src._color_rgb is first


def test_gather_frames_filters_stale():
    fresh = make_frame(np.zeros((4, 4), np.uint16), 1, age_s=0.01)
    stale = make_frame(np.zeros((4, 4), np.uint16), 2, age_s=1.0)
    rig = SimpleNamespace(cameras={
        "near": SimpleNamespace(read=lambda: fresh),
        "far": SimpleNamespace(read=lambda: stale)})
    frames, ages = gather_frames(rig, 0.3)
    assert frames["near"] is fresh and frames["far"] is None
    assert ages["near"] <= 0.3 and "far" not in ages


# --- goal raster -------------------------------------------------------------

def test_goal_raster_from_slot_matches_builder():
    odom = (0.4, -0.2, 0.7)
    want = build_goal_raster({"type": "heading", "heading_rad": 0.35},
                             {"x": odom[0], "y": odom[1], "theta": odom[2]})
    assert np.array_equal(goal_raster_from_slot(GoalHeading(0.35), odom), want)
    want = build_goal_raster(
        {"type": "point", "x": 1.5, "y": 0.5},
        {"x": odom[0], "y": odom[1], "theta": odom[2]})
    assert np.array_equal(goal_raster_from_slot(GoalPoint(1.5, 0.5), odom),
                          want)
    assert not goal_raster_from_slot(None, odom).any()


# --- preprocessing (shared with training) ------------------------------------

def test_prep_depth_clips_and_keeps_invalid():
    # cv2 INTER_NEAREST at exact 2x decimation samples even input columns:
    # output col j reads input col 2j.
    d = np.zeros((480, 848), np.uint16)
    d[:, 0] = 100      # 0.10 m -> clipped to 0.3
    d[:, 2] = 2000     # 2.0 m
    d[:, 4] = 9000     # 9.0 m -> clipped to 6.0
    out, mask = prep_depth(d)
    assert out.shape == (1, IMG_H, IMG_W)
    assert (mask[0, :, 0] == 1.0).all()
    assert np.allclose(out[0, :, 0], 0.3)
    assert np.allclose(out[0, :, 1], 2.0)
    assert np.allclose(out[0, :, 2], 6.0)
    # untouched pixels are invalid and stay 0
    assert (mask[0, :, 6] == 0.0).all() and (out[0, :, 6] == 0.0).all()


def test_prep_color_normalizes():
    rgb = np.full((480, 848, 3), 127, np.uint8)
    c = prep_color(rgb)
    assert c.shape == (3, IMG_H, IMG_W)
    want = (127 / 255.0 - np.array([0.485, 0.456, 0.406])) \
        / np.array([0.229, 0.224, 0.225])
    assert np.allclose(c[:, 0, 0], want, atol=1e-5)


# --- real-ONNX parity (skipped without torch) --------------------------------

def _load_export_tool():
    p = Path(__file__).resolve().parents[1] / "tools" / "export_navd_onnx.py"
    spec = importlib.util.spec_from_file_location("export_navd_onnx", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("export_navd_onnx", mod)
    spec.loader.exec_module(mod)
    return mod


def test_onnx_parity_vs_torch(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnxruntime")
    from bebop_vision.navd import NavdUNet
    mod = _load_export_tool()
    model = NavdUNet()
    model.eval()   # random weights — parity is about the graph, not accuracy
    out = tmp_path / "navd_test.onnx"
    mod.export(model, out)
    src = NavdGridSource(out, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    for _ in range(2):
        dn = (rng.random((480, 848)) * 4000).astype(np.uint16)
        df = (rng.random((480, 848)) * 4000).astype(np.uint16)
        rgb = rng.integers(0, 255, (480, 848, 3), dtype=np.uint8)
        goal = np.abs(rng.normal(0, 0.5, (ROWS, COLS))).astype(np.float32)
        with torch.inference_mode():
            dn_t, _ = prep_depth(dn)
            df_t, _ = prep_depth(df)
            c_t = prep_color(rgb)
            t = model(torch.from_numpy(dn_t[None]),
                      torch.from_numpy(df_t[None]),
                      torch.from_numpy(c_t[None]),
                      torch.from_numpy(goal[None, None])).numpy()
        o = src.model.predict(dn, df, rgb, goal)
        assert (t[0].argmax(0) == o.argmax(0)).mean() >= 0.99
