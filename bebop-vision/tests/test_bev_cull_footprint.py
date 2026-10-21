"""The near-body depth cull follows the learned chassis footprint."""

import dataclasses

import numpy as np

import bebop_vision.bev_student_labels as labels
from bebop_vision.bev_student_labels import (
    LabelConfig, ego_cull_mask, points_in_raster, prepare_observation)
from test_bev_student import GRID, LABEL_CFG, make_spec, render_scene


def _raster(tmp_path, x0=-0.6, y0=-0.6, cell=0.01):
    """A 0.40 x 0.30 m body x -0.10..0.30, y -0.20..0.10."""
    mask = np.zeros((150, 120), bool)
    mask[int((-0.10 - x0) / cell):int((0.30 - x0) / cell),
         int((-0.20 - y0) / cell):int((0.10 - y0) / cell)] = True
    f = tmp_path / "ego_mask.npz"
    np.savez_compressed(f, mask=mask, x0=x0, y0=y0, cell=cell)
    return f


def test_cull_raster_is_footprint_plus_margin(tmp_path, monkeypatch):
    _raster(tmp_path)
    monkeypatch.setattr(labels, "CONFIG_DIR", tmp_path)
    labels._CULL_CACHE.clear()
    cfg = dataclasses.replace(LABEL_CFG, ego_mask_file="ego_mask.npz",
                              cull_ego_margin_m=0.02)
    r = ego_cull_mask(cfg)
    assert r["mask"].sum() > 0.40 * 0.30 / 1e-4          # bigger than the body
    assert r["mask"].sum() < 0.46 * 0.36 / 1e-4          # but by at most ~3 cm
    x = np.array([0.10, 0.31, 0.34, -0.05, 0.10, 2.0])
    y = np.array([0.00, 0.00, 0.00, -0.21, 0.13, 0.0])
    inside = points_in_raster(r, x, y)
    # body centre, 1 cm past the front edge (within margin), 4 cm past (out),
    # 1 cm past the right edge (in), 3 cm past the left edge (out), far away
    assert inside.tolist() == [True, True, False, True, False, False]


def test_prepare_observation_keeps_floor_beside_body(tmp_path, monkeypatch):
    _raster(tmp_path)
    monkeypatch.setattr(labels, "CONFIG_DIR", tmp_path)
    labels._CULL_CACHE.clear()
    spec = make_spec(role="near", pitch=-66.0, height=1.27)
    depth, hit = render_scene(spec=spec)        # flat floor everywhere
    floor_conf = (hit == 1).astype(np.float32)
    foot = dataclasses.replace(LABEL_CFG, min_range_m=0.55,
                               ego_mask_file="ego_mask.npz",
                               cull_ego_margin_m=0.02)
    rect = dataclasses.replace(LABEL_CFG, min_range_m=0.55,
                               cull_rect_m=(-0.30, 0.37, -0.35, 0.25))
    p_foot = prepare_observation(spec, depth, floor_conf=floor_conf, cfg=foot, grid=GRID)
    p_rect = prepare_observation(spec, depth, floor_conf=floor_conf, cfg=rect, grid=GRID)
    px, py = p_foot.points[:, 0], p_foot.points[:, 1]
    # nothing survives inside the dilated footprint
    assert not np.any((px > -0.08) & (px < 0.28) & (py > -0.18) & (py < 0.08))
    # floor in the band the old rectangle threw away is kept now
    band = (px > 0.33) & (px < 0.37) & (np.abs(py) < 0.1)
    assert band.sum() > 0
    rx = p_rect.points[:, 0]
    assert not np.any((rx > 0.33) & (rx < 0.37) & (np.abs(p_rect.points[:, 1]) < 0.1))
    assert len(p_foot.points) > len(p_rect.points)


def test_margin_rule_needs_a_raster_else_rect_or_disc():
    cfg = LabelConfig.from_config({"bev_student": {"label": {
        "cull_ego_margin_m": 0.02, "cull_rect_m": [-0.3, 0.37, -0.35, 0.25]}}})
    assert cfg.cull_ego_margin_m == 0.02 and cfg.ego_mask_file is None
    spec = make_spec(role="near", pitch=-66.0, height=1.27)
    depth, hit = render_scene(spec=spec)
    p = prepare_observation(spec, depth, floor_conf=(hit == 1).astype(np.float32),
                            cfg=cfg, grid=GRID)
    x, y = p.points[:, 0], p.points[:, 1]
    # without a raster the rectangle applies
    assert not np.any((x > -0.3) & (x < 0.37) & (y > -0.35) & (y < 0.25))


# ---------------------------------------------------------------------------
# Hazard-free margin: rail-edge flying pixels beside the chassis are not drops
# ---------------------------------------------------------------------------

from bebop_vision.bev_student_labels import (camera_evidence, ego_mask, fit_plane,
                                             fuse_camera_evidence)


def _label_near(depth, hit, cfg):
    spec = make_spec(role="near", pitch=-66.0, height=1.27)
    prep = prepare_observation(spec, depth, floor_conf=(hit == 1).astype(np.float32),
                               cfg=cfg, grid=GRID)
    plane = fit_plane(prep, cfg)
    assert plane["ok"]
    ev = camera_evidence(prep, plane, cfg, GRID)
    return fuse_camera_evidence({"near": ev}, cfg, GRID, ego=ego_mask(cfg, GRID)), ev


def test_hazard_margin_drops_below_plane_only_beside_the_body(tmp_path, monkeypatch):
    _raster(tmp_path)                      # body x -0.10..0.30, y -0.20..0.10
    monkeypatch.setattr(labels, "CONFIG_DIR", tmp_path)
    labels._CULL_CACHE.clear()
    spec = make_spec(role="near", pitch=-66.0, height=1.27)
    # a 12 cm sunken band 3-12 cm ahead of the body's front edge, full width
    depth, hit = render_scene(spec=spec, floors=((0.0, -20.0, 0.33), (-0.12, 0.33, 0.42),
                                                 (0.0, 0.42, 20.0)))
    base = dataclasses.replace(LABEL_CFG, min_range_m=0.55, ego_mask_file="ego_mask.npz",
                               cull_ego_margin_m=0.02)
    ruled = dataclasses.replace(base, hazard_ego_margin_m=0.15)
    out0, _ = _label_near(depth, hit, base)
    out1, ev1 = _label_near(depth, hit, ruled)
    xc, yc = GRID.centers()
    band = ((xc >= 0.33) & (xc <= 0.42))[:, None]
    inside = band & (np.abs(yc) < 0.08)[None, :]        # within 15 cm of the body
    outside = band & (np.abs(yc) > 0.45)[None, :]       # well beyond the margin
    assert (out0.semantic[inside] == 3).sum() > 0        # old rule: a drop strip
    assert (out1.semantic[inside] == 3).sum() == 0       # new rule: gone
    assert (out0.semantic[outside] == 3).sum() > 0
    assert (out1.semantic[outside] == 3).sum() > 0       # real drops elsewhere stay
    assert ev1["edge_hazard"] > 0


def test_hazard_margin_config_and_validation():
    cfg = LabelConfig.from_config({"bev_student": {"label": {
        "hazard_ego_margin_m": 0.15}, "ego": {"mask_file": "ego_mask.npz"}}})
    assert cfg.hazard_ego_margin_m == 0.15 and cfg.ego_mask_file == "ego_mask.npz"
    assert LabelConfig.from_config({}).hazard_ego_margin_m is None
    import pytest
    with pytest.raises(ValueError):
        LabelConfig.from_config({"bev_student": {"label": {"hazard_ego_margin_m": -1}}})
