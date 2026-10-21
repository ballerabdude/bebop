"""Post-processing / layer contract tests for the TensorRT BEV runtime.

No TensorRT engine is needed: the pure postprocess is exercised with fake
logits and a fake geometric grid.
"""

import numpy as np

from bebop_vision.bev_student_runtime import BevStudentRuntime
from bebop_vision.master_bev import (FREE, HAZARD, INFLATED, OCCUPIED, SRC_BOTH,
                                     _rle)


class _Frame:
    stamp_us = 1234


def _runtime(hazard_source="geometric", inflate=False):
    rt = BevStudentRuntime.__new__(BevStudentRuntime)
    rt.rows, rt.cols = 120, 60
    rt.cell_m, rt.range_m, rt.width_m = 0.05, 6.0, 3.0
    rt.cell_x_m = rt.cell_y_m = 0.05
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


def test_rectangular_grid_geometry_from_metadata():
    """Matched-grid engines (null cell_m) drive ego/inflation per axis."""
    from bebop_vision.master_bev import render_bev
    rt = BevStudentRuntime.__new__(BevStudentRuntime)
    rt.hazard_source = "model"
    rt.engine_path = "fake.engine"
    rt.configure_grid(
        {"rows": 360, "cols": 640, "range_m": 6.0, "width_m": 4.0,
         "cell_m": None, "cell_x_m": 6.0 / 360, "cell_y_m": 4.0 / 640},
        {"bev_student": {"ego": {"x_range_m": [-0.10, 0.29],
                                 "y_range_m": [-0.27, 0.17]}},
         "robot": {"radius_m": 0.20}})
    assert rt.cell_m is None
    assert (rt.cell_x_m, rt.cell_y_m) == (6.0 / 360, 4.0 / 640)
    # The grid starts at the mast (x = 0), so only the 0.29 m of chassis
    # ahead of it is inside: 0.29 / 1.667 cm = 17 rows ending at the last
    # row; sideways 0.44 m / 0.625 cm = 70-71 columns.
    ego_rows = np.flatnonzero(rt.ego_static.any(axis=1))
    ego_cols = np.flatnonzero(rt.ego_static.any(axis=0))
    assert 17 <= len(ego_rows) <= 18 and ego_rows.max() == 359
    assert 69 <= len(ego_cols) <= 71
    # 20 cm radius: 12 rows forward, 32 columns sideways
    assert rt.inflate_kernel.shape == (25, 65)
    semantic = np.zeros((360, 640), np.uint8)
    semantic[100, 320] = 2
    logits = np.full((4, 360, 640), -8.0, np.float32)
    for c in range(4):
        logits[c][semantic == c] = 8.0
    ego = np.full((1, 360, 640), -8.0, np.float32)
    cov = np.full((1, 360, 640), 8.0, np.float32)
    grid, _ = rt.postprocess(logits, ego, cov, {"near": _Frame(),
                                                "far": _Frame()}, ages_s={})
    infl = np.argwhere(grid.occ == INFLATED)
    assert infl[:, 0].min() == 100 - 12 and infl[:, 1].min() == 320 - 32
    d = grid.to_dict()
    assert d["cell_m"] is None
    assert (d["cell_x_m"], d["cell_y_m"]) == (6.0 / 360, 4.0 / 640)
    assert sum(c for _, c in d["ego_rle"]) == 360 * 640
    assert render_bev(grid, scale=1).shape == (360, 640, 3)


def test_hazard_threshold_override():
    # Hazard probability ~0.475 (softmax(1,0,0,0)); below 0.5 but above 0.4.
    logits = np.zeros((4, 120, 60), np.float32)
    logits[3] = 1.0
    ego = np.full((1, 120, 60), -8.0, np.float32)
    cov = np.full((1, 120, 60), 8.0, np.float32)
    rt = _runtime(hazard_source="model")
    rt.hazard_threshold = 0.4
    grid, _ = rt.postprocess(logits, ego, cov, {"near": _Frame(),
                                                "far": _Frame()}, ages_s={})
    assert np.all(grid.semantic[grid.coverage] == 3)
    rt.hazard_threshold = 0.6
    grid, _ = rt.postprocess(logits, ego, cov, {"near": _Frame(),
                                                "far": _Frame()}, ages_s={})
    assert not np.any(grid.semantic == 3)


def test_hazard_pad_ring_clears_drops_beside_the_chassis():
    """``ego.hazard_pad_m``: a predicted drop in the ring around the static
    chassis becomes the model's best other class; drops further out stay."""
    rt = BevStudentRuntime.__new__(BevStudentRuntime)
    rt.hazard_source = "model"
    rt.hazard_threshold = 0.5
    rt.engine_path = "fake.engine"
    rt.single_head = False
    rt.configure_grid(
        {"rows": 160, "cols": 320, "range_m": 4.0, "width_m": 4.0,
         "cell_m": None, "cell_x_m": 4.0 / 160, "cell_y_m": 4.0 / 320},
        {"bev_student": {"ego": {"x_range_m": [-0.10, 0.29],
                                 "y_range_m": [-0.27, 0.17],
                                 "hazard_pad_m": 0.15}},
         "robot": {"radius_m": 0.0}})
    assert rt.hazard_pad_m == 0.15 and rt.hazard_pad_ring is not None
    assert not (rt.hazard_pad_ring & rt.ego_static).any()
    # ring thickness ~ 0.15 m: 6 rows (2.5 cm) / 12 cols (1.25 cm)
    ring_cols = np.flatnonzero(rt.hazard_pad_ring.any(axis=0))
    ego_cols = np.flatnonzero(rt.ego_static.any(axis=0))
    assert 11 <= ego_cols[0] - ring_cols[0] <= 13
    semantic = np.zeros((160, 320), np.uint8)
    semantic[rt.hazard_pad_ring] = 3                     # drop all round the body
    semantic[40:44, 100:110] = 3                         # a real drop 2.9 m ahead
    logits = np.full((4, 160, 320), -8.0, np.float32)
    for c in range(4):
        logits[c][semantic == c] = 8.0
    logits[0][rt.hazard_pad_ring] = 4.0                  # floor is the runner-up there
    ego = np.full((1, 160, 320), -8.0, np.float32)
    cov = np.full((1, 160, 320), 8.0, np.float32)

    class _F:
        stamp_us = 1
    grid, stats = rt.postprocess(logits, ego, cov, {"near": _F(), "far": _F()},
                                 ages_s={})
    assert stats["edge_hazard_cells"] == int(rt.hazard_pad_ring.sum())
    assert np.all(grid.semantic[rt.hazard_pad_ring] == 0)
    assert np.all(grid.raw[rt.hazard_pad_ring] == FREE)
    assert np.all(grid.semantic[40:44, 100:110] == 3)
    # pad off -> ring untouched
    rt.hazard_pad_ring_backup, rt.hazard_pad_ring = rt.hazard_pad_ring, None
    grid, stats = rt.postprocess(logits, ego, cov, {"near": _F(), "far": _F()},
                                 ages_s={})
    assert stats["edge_hazard_cells"] == 0
    assert np.all(grid.semantic[rt.hazard_pad_ring_backup] == 3)
