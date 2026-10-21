"""Cropping a label set to a shorter forward range keeps the near rows."""

import json

import numpy as np
import pytest

from bebop_vision.bev_student_labels import GridSpec
from tools.bev_crop_labels import crop_rows, crop_session


def test_crop_rows_exact_only():
    grid = GridSpec(range_m=6.0, width_m=4.0, rows_override=360, cols_override=640)
    assert crop_rows(grid, 4.0) == 120
    assert crop_rows(grid, 6.0) == 0
    with pytest.raises(ValueError):
        crop_rows(grid, 4.01)
    with pytest.raises(ValueError):
        crop_rows(grid, 7.0)


def test_crop_session_keeps_near_rows_and_rewrites_metadata(tmp_path):
    sess = tmp_path / "sessions" / "navd_session_20260101_000000"
    src = sess / "labels_gridsrc"
    src.mkdir(parents=True)
    grid = GridSpec(range_m=6.0, width_m=4.0, rows_override=36, cols_override=64)
    (src / "grid.json").write_text(json.dumps(
        {"grid": grid.metadata(), "frame_stride": 3, "label_config": {"k": 1}}))
    sem = np.full((36, 64), 255, np.uint8)
    sem[30:36, :] = 0            # floor near the robot
    sem[0:6, :] = 1              # wall at the far edge (dropped)
    ego = np.zeros((36, 64), bool)
    ego[34:36, 30:34] = True
    sem[ego] = 255
    cov = sem != 255
    np.savez_compressed(src / "00000000000000000001.npz", semantic=sem, ego=ego,
                        coverage=cov)
    (src / "stats.json").write_text("{}")
    n = crop_session(sess, "labels_gridsrc", "labels_gridout", 4.0)
    assert n == 1
    out = sess / "labels_gridout"
    meta = json.loads((out / "grid.json").read_text())
    assert meta["grid"]["rows"] == 24 and meta["grid"]["range_m"] == 4.0
    assert meta["grid"]["cell_x_m"] == pytest.approx(6.0 / 36)
    assert meta["cropped_from"] == {"labels": "labels_gridsrc", "range_m": 6.0,
                                    "rows_dropped_far": 12}
    assert meta["frame_stride"] == 3 and meta["label_config"] == {"k": 1}
    with np.load(out / "00000000000000000001.npz") as d:
        assert d["semantic"].shape == (24, 64)
        assert np.all(d["semantic"][18:24][~d["ego"][18:24]] == 0)
        assert not np.any(d["semantic"] == 1)
        assert d["ego"][22:24, 30:34].all() and d["ego"].sum() == 8
    assert (out / "stats.json").exists()
