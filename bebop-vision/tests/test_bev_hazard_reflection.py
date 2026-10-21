"""Glossy-floor reflections must not become drop hazards (or floor)."""

import dataclasses

import numpy as np

from bebop_vision.bev_student_labels import (
    FLOOR, HAZARD, camera_evidence, ego_mask, fit_plane,
    fuse_camera_evidence, prepare_observation)
from test_bev_student import GRID, LABEL_CFG, make_spec, rc, render_scene


def _band(out, x0, x1):
    """Semantic rows whose forward distance lies in [x0, x1]."""
    xc, _ = GRID.centers()
    return out.semantic[(xc >= x0) & (xc <= x1)]


def _label(depth, hit, *, stairs_conf=None, cfg=LABEL_CFG):
    spec = make_spec()
    prep = prepare_observation(
        spec, depth, floor_conf=(hit == 1).astype(np.float32),
        hazard_conf=stairs_conf, cfg=cfg, grid=GRID)
    plane = fit_plane(prep, cfg)
    assert plane["ok"]
    ev = camera_evidence(prep, plane, cfg, GRID)
    out = fuse_camera_evidence({"far": ev}, cfg, GRID, ego=ego_mask(cfg, GRID))
    return out, ev


def _pit_scene(depth_m):
    """Floor to 2 m, then a 'floor' ``depth_m`` below it (no riser, like a
    reflection that reads the far floor too deep)."""
    floors = ((0.0, -20.0, 2.0), (-depth_m, 2.0, 20.0))
    return render_scene(floors=floors)


def test_deep_drop_without_stairs_prompt_is_rejected_not_hazard():
    depth, hit = _pit_scene(0.50)
    out, ev = _label(depth, hit)
    assert ev["reflections"] > 0
    band = _band(out, 2.5, 4.0)
    assert not np.any(band == HAZARD)
    # the reflected samples are not floor observations either
    assert not np.any(band == FLOOR)
    xc, _ = GRID.centers()
    assert not out.coverage[(xc >= 3.2) & (xc <= 3.9)].any()
    # the real floor in front is untouched
    assert np.any(_band(out, 1.0, 1.9) == FLOOR)


def test_deep_drop_with_stairs_prompt_stays_hazard():
    depth, hit = _pit_scene(0.50)
    stairs = (hit == 1).astype(np.float32)   # SAM says stairs/hole everywhere
    out, ev = _label(depth, hit, stairs_conf=stairs)
    assert ev["reflections"] == 0
    assert np.any(_band(out, 2.5, 4.0) == HAZARD)


def test_single_step_down_is_still_a_depth_only_hazard():
    depth, hit = _pit_scene(0.15)
    out, ev = _label(depth, hit)
    assert ev["reflections"] == 0
    assert np.any(_band(out, 2.3, 4.0) == HAZARD)


def test_rule_can_be_disabled():
    depth, hit = _pit_scene(0.50)
    cfg = dataclasses.replace(LABEL_CFG, hazard_deep_drop_m=0.0)
    out, ev = _label(depth, hit, cfg=cfg)
    assert ev["reflections"] == 0
    assert np.any(_band(out, 2.5, 4.0) == HAZARD)
