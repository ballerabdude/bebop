"""Post-processing / layer contract tests for the TensorRT BEV runtime.

No TensorRT engine is needed: the pure postprocess is exercised with fake
logits and a fake geometric grid.
"""

import numpy as np

from bebop_vision.bev_student_runtime import BevStudentRuntime
from bebop_vision.master_bev import (HAZARD, INFLATED, OCCUPIED, SRC_BOTH,
                                     _rle)


class _Frame:
    stamp_us = 1234


def _runtime(hazard_source="geometric", inflate=False):
    rt = BevStudentRuntime.__new__(BevStudentRuntime)
    rt.rows, rt.cols = 120, 60
    rt.cell_m, rt.range_m, rt.width_m = 0.05, 6.0, 3.0
    rt.ego_x = (0.0, 0.55)
    rt.ego_y = (-0.22, 0.22)
    x = rt.range_m - (np.arange(rt.rows) + 0.5) * rt.cell_m
    y = (np.arange(rt.cols) + 0.5) * rt.cell_m - rt.width_m / 2.0
    rt.ego_static = ((x[:, None] >= rt.ego_x[0])
                     & (x[:, None] <= rt.ego_x[1])
                     & (y[None, :] >= rt.ego_y[0])
                     & (y[None, :] <= rt.ego_y[1]))
    rt.inflate_kernel = None
    rt.hazard_source = hazard_source
    rt.engine_path = "fake.engine"
    rt.device = "cpu"
    return rt


def _logits(semantic):
    logits = np.full((4, 120, 60), -8.0, np.float32)
    for c in range(4):
        logits[c][semantic == c] = 8.0
    return logits


class _Geo:
    def __init__(self):
        self.raw = np.zeros((120, 60), np.uint8)
        self.raw[20:24, 10:14] = HAZARD


def test_postprocess_maps_classes_and_geometric_hazard():
    rt = _runtime()
    semantic = np.zeros((120, 60), np.uint8)
    semantic[60:70, 20:30] = 1   # wall
    semantic[70:80, 30:40] = 2   # obstacle
    semantic[10:15, 40:45] = 3   # model-only hazard (should be replaced)
    logits = _logits(semantic)
    ego = np.full((1, 120, 60), -8.0, np.float32)
    cov = np.full((1, 120, 60), 8.0, np.float32)
    grid, stats = rt.postprocess(logits, ego, cov, {"near": _Frame(),
                                                    "far": _Frame()},
                                 geo_grid=_Geo(), ages_s={})
    # geometric hazard wins, model-only hazard is removed
    assert np.all(grid.semantic[20:24, 10:14] == 3)
    assert np.all(grid.semantic[10:15, 40:45] != 3)
    assert np.all(grid.raw[60:70, 20:30] == OCCUPIED)
    assert np.all(grid.raw[70:80, 30:40] == OCCUPIED)
    assert np.all(grid.raw[20:24, 10:14] == HAZARD)
    assert np.all(grid.wall == (grid.semantic == 1))
    assert int(grid.ego.sum()) == 88
    assert np.all(grid.coverage[~grid.ego])
    assert np.all(grid.source[grid.coverage] == SRC_BOTH)
    d = grid.to_dict()
    total = 120 * 60
    for key in ("semantic_rle", "wall_rle", "ego_rle", "occ_rle",
                "coverage_rle"):
        assert sum(c for _, c in d[key]) == total
    assert d["semantic_counts"]["hazard"] == 4 * 4
    assert d["contributions"]["model"]["hazard_source"] == "geometric"


def test_postprocess_model_hazard_mode_keeps_student():
    rt = _runtime(hazard_source="model")
    semantic = np.zeros((120, 60), np.uint8)
    semantic[10:15, 40:45] = 3
    logits = _logits(semantic)
    ego = np.full((1, 120, 60), -8.0, np.float32)
    cov = np.full((1, 120, 60), 8.0, np.float32)
    grid, _ = rt.postprocess(logits, ego, cov, {"near": _Frame(),
                                                "far": _Frame()},
                             geo_grid=_Geo(), ages_s={})
    assert np.all(grid.semantic[10:15, 40:45] == 3)


def test_inflation_layer():
    rt = _runtime()
    rt.inflate_kernel = np.ones((3, 3), np.uint8)
    semantic = np.zeros((120, 60), np.uint8)
    semantic[60, 30] = 2
    logits = _logits(semantic)
    ego = np.full((1, 120, 60), -8.0, np.float32)
    cov = np.full((1, 120, 60), 8.0, np.float32)
    grid, _ = rt.postprocess(logits, ego, cov, {"near": _Frame(),
                                                "far": _Frame()},
                             ages_s={})
    assert np.sum(grid.occ == INFLATED) > 0
    assert np.all(grid.raw[60, 30] == OCCUPIED)
