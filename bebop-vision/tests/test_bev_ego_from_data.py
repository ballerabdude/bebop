"""Self-supervised chassis footprint: derivation, rasterisation, wiring."""

import json
from pathlib import Path

import numpy as np
import pytest

from bebop_vision.bev_student_labels import GridSpec, LabelConfig, ego_mask
from bebop_vision.master_bev import load_ego_raster, rasterize_ego
from tools.bev_ego_from_data import derive_ego_mask, extent_m


CELL = 0.01
X0, Y0 = -0.6, -0.6


def _maps(nx=150, ny=120):
    """Synthetic persistence maps: a 0.40 x 0.30 m frame top seen in every
    frame, the floor observed everywhere except two wheel shadows beside it
    and a strip the frame hides directly ahead, plus a transient box."""
    p_obs = np.ones((nx, ny), np.float32)
    p_body = np.zeros((nx, ny), np.float32)

    def idx(x, y):
        return int((x - X0) / CELL), int((y - Y0) / CELL)

    (fx0, fy0), (fx1, fy1) = idx(-0.10, -0.20), idx(0.30, 0.10)
    p_body[fx0:fx1, fy0:fy1] = 0.97          # chassis top, always there
    # wheels: never observed, 6 cm wide, 20 cm long, beside the frame
    for y_lo, y_hi in ((-0.28, -0.22), (0.12, 0.18)):
        (wx0, wy0), (wx1, wy1) = idx(0.0, y_lo), idx(0.20, y_hi)
        p_obs[wx0:wx1, wy0:wy1] = 0.0
    # frame shadow ahead of the bumper: unobserved floor, not chassis
    (sx0, sy0), (sx1, sy1) = idx(0.30, -0.20), idx(0.50, 0.10)
    p_obs[sx0:sx1, sy0:sy1] = 0.0
    # nothing observed behind the robot
    (bx0, _), (bx1, _) = idx(-0.60, 0), idx(-0.25, 0)
    p_obs[bx0:bx1, :] = 0.0
    # a box that stood in front for a third of the drive: not chassis
    (ox0, oy0), (ox1, oy1) = idx(0.60, -0.10), idx(0.75, 0.10)
    p_body[ox0:ox1, oy0:oy1] = 0.33
    p_obs[ox0:ox1, oy0:oy1] = 0.9
    # a speck of persistent noise
    p_body[idx(0.55, 0.40)] = 0.95
    return p_obs, p_body


def test_derive_ego_mask_core_wheels_not_shadow_ahead():
    p_obs, p_body = _maps()
    mask, core, shadow = derive_ego_mask(p_obs, p_body, CELL)
    cx0, cx1, cy0, cy1 = extent_m(core, X0, Y0, CELL)
    assert (cx0, cx1, cy0, cy1) == pytest.approx((-0.10, 0.30, -0.20, 0.10), abs=0.011)
    x0, x1, y0, y1 = extent_m(mask, X0, Y0, CELL)
    # wheels folded in sideways, nothing behind the rear edge
    assert y0 == pytest.approx(-0.28, abs=0.011) and y1 == pytest.approx(0.18, abs=0.011)
    assert x0 == pytest.approx(-0.10, abs=0.02)
    # the unobserved strip ahead of the bumper is floor, not body
    assert x1 == pytest.approx(0.30, abs=0.02)
    # the transient box and the speck are out
    bx = int((0.65 - X0) / CELL)
    assert not mask[bx, int((0.0 - Y0) / CELL)]
    assert not mask[int((0.55 - X0) / CELL), int((0.40 - Y0) / CELL)]
    assert not (shadow & core).any()


def test_derive_ego_mask_needs_a_core():
    p_obs, p_body = _maps()
    with pytest.raises(ValueError):
        derive_ego_mask(p_obs, p_body * 0.5, CELL)


def _write_raster(path, mask):
    np.savez_compressed(path, mask=mask, x0=X0, y0=Y0, cell=CELL)


def test_rasterize_ego_on_rectangular_grid(tmp_path):
    p_obs, p_body = _maps()
    mask, _, _ = derive_ego_mask(p_obs, p_body, CELL)
    f = tmp_path / "ego.npz"
    _write_raster(f, mask)
    raster = load_ego_raster(f)
    grid = GridSpec(range_m=6.0, width_m=4.0, rows_override=360, cols_override=640)
    xc, yc = grid.centers()
    ego = rasterize_ego(raster, xc, yc)
    assert ego.shape == (360, 640)
    rows = np.flatnonzero(ego.any(axis=1))
    cols = np.flatnonzero(ego.any(axis=0))
    # only the part ahead of the mast (x >= 0) is inside the grid: 0.30 m / 1.67 cm
    assert rows.max() == 359 and 17 <= len(rows) <= 19
    # 0.46 m wide / 0.625 cm
    assert 72 <= len(cols) <= 75
    # a centre far outside the raster is never ego
    assert not rasterize_ego(raster, np.array([5.0]), np.array([0.0]))[0, 0]
    # square legacy grid works too
    g2 = GridSpec()
    assert rasterize_ego(raster, *g2.centers()).shape == (120, 60)


def test_label_config_mask_file_replaces_rectangle(tmp_path, monkeypatch):
    import bebop_vision.bev_student_labels as labels
    p_obs, p_body = _maps()
    mask, _, _ = derive_ego_mask(p_obs, p_body, CELL)
    f = tmp_path / "ego_mask.npz"
    _write_raster(f, mask)
    monkeypatch.setattr(labels, "CONFIG_DIR", tmp_path)
    cfg = LabelConfig.from_config({"bev_student": {"ego": {
        "x_range_m": [0.0, 0.55], "y_range_m": [-0.22, 0.22],
        "mask_file": "ego_mask.npz"}}})
    assert cfg.ego_mask_file == "ego_mask.npz"
    grid = GridSpec()
    learned = ego_mask(cfg, grid)
    rect = ego_mask(LabelConfig(), grid)
    assert learned.any() and not np.array_equal(learned, rect)
    # the 55 cm rectangle reaches further forward than the learned 30 cm body
    assert rect.any(axis=1).sum() > learned.any(axis=1).sum()


def test_apply_to_labels_rewrites_ego_layer(tmp_path):
    from tools.bev_ego_from_data import apply_to_labels
    p_obs, p_body = _maps()
    mask, _, _ = derive_ego_mask(p_obs, p_body, CELL)
    f = tmp_path / "ego_mask.npz"
    _write_raster(f, mask)
    raster = load_ego_raster(f)
    root = tmp_path
    sess = root / "sessions" / "navd_session_20260101_000000"
    sub = sess / "labels_gridtest"
    sub.mkdir(parents=True)
    (sess / "index.jsonl").write_text("")
    grid = GridSpec(range_m=6.0, width_m=4.0, rows_override=36, cols_override=64)
    (sub / "grid.json").write_text(json.dumps({"grid": grid.metadata(),
                                               "label_config": {}}))
    sem = np.zeros((36, 64), np.uint8)
    cov = np.ones((36, 64), bool)
    np.savez_compressed(sub / "00000000000000000001.npz", semantic=sem,
                        ego=np.zeros((36, 64), bool), coverage=cov)
    n = apply_to_labels(root, "labels_gridtest", raster)
    assert n == 1
    with np.load(sub / "00000000000000000001.npz") as d:
        ego = d["ego"]
        assert ego.any()
        assert np.all(d["semantic"][ego] == 255)
        assert not d["coverage"][ego].any()
        assert np.all(d["coverage"][~ego])
    meta = json.loads((sub / "grid.json").read_text())
    assert meta["label_config"]["ego_mask_file"] == "ego_mask.npz"
    assert meta["ego_mask"]["cells"] == int(ego.sum())
