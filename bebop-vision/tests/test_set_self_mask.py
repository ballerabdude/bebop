"""set_self_mask projection tests (no GUI, no server).

Synthetic intrinsics cover the coordinate conventions (scale vs extrinsic
translation, mm units) and the real cached rig intrinsics validate the
depth<->color round trip the tool's overlays rely on.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from set_self_mask import (MODES, color_to_depth_point, depth_poly_to_color_poly,
                           depth_rect_to_poly, depth_to_color_point,
                           poly_to_depth_poly, roi_to_depth_rect,
                           shape_kind, yml_snippet)  # noqa: E402

# Synthetic rig: matched FOVs (color half-ray = depth half-ray), aligned
# axes. Identity extrinsic (t=0) -> 'extrinsic' == 'scale', and the full
# color frame projects exactly onto the full depth frame.
INTR_ID = {
    "fx": 400.0, "fy": 400.0, "cx": 424.0, "cy": 240.0,
    "color_fx": 400.0 * 640 / 424, "color_fy": 400.0 * 400 / 240,
    "color_cx": 640.0, "color_cy": 400.0,
    "color_to_depth_transform": {
        "rotation": [1, 0, 0, 0, 1, 0, 0, 0, 1],
        "translation": [0.0, 0.0, 0.0],
    },
}

# Same rig with a -20 mm x baseline (SDK units: millimetres; the stored
# transform is depth->color per the SDK, so color->depth shifts +x).
INTR_BASE = dict(
    INTR_ID,
    color_to_depth_transform={
        "rotation": [1, 0, 0, 0, 1, 0, 0, 0, 1],
        "translation": [-20.0, 0.0, 0.0],
    })


def test_scale_equals_extrinsic_with_zero_translation():
    # Center of the color frame -> center of the depth frame.
    u, v = color_to_depth_point(640, 400, 1.0, INTR_ID, "extrinsic")
    assert (u, v) == pytest.approx((424, 240))
    u, v = color_to_depth_point(640, 400, 1.0, INTR_ID, "scale")
    assert (u, v) == pytest.approx((424, 240))


def test_full_frame_roi_maps_to_full_depth_frame():
    r = roi_to_depth_rect(0, 0, 1280, 800, 1.0, INTR_ID, "extrinsic")
    assert r == [0, 0, 848, 480]


def test_translation_shifts_projection_in_mm_units():
    # Stored transform is depth->color with t_x=-20 mm; color->depth is
    # its inverse, so at z=1: x_d = x_c + 0.02 -> depth u = 424 + 8.
    u, v = color_to_depth_point(640, 400, 1.0, INTR_BASE, "extrinsic")
    assert u == pytest.approx(424 + 8.0)
    assert v == pytest.approx(240)
    # 'scale' mode must ignore the extrinsic entirely.
    u, v = color_to_depth_point(640, 400, 1.0, INTR_BASE, "scale")
    assert (u, v) == pytest.approx((424, 240))


def test_projection_round_trip():
    intr = json.load(open(Path(__file__).resolve().parents[1] / "config"
                          / "orbbec_intrinsics_CPBLC53000PE.json"))
    for mode in MODES:
        for (u, v) in [(300, 200), (640, 400), (1000, 700)]:
            ud, vd = color_to_depth_point(u, v, 0.7, intr, mode)
            uc, vc = depth_to_color_point(ud, vd, 0.7, intr, mode)
            # two matrix projections through the real (slightly rotated)
            # extrinsic: 0.01 px is float noise
            assert (uc, vc) == pytest.approx((u, v), abs=1e-2), mode


def test_roi_clamps_to_depth_frame():
    r = roi_to_depth_rect(-500, -500, 3000, 3000, 1.0, INTR_ID, "extrinsic")
    assert r == [0, 0, 848, 480]


def test_depth_rect_to_poly_round_trip():
    intr = json.load(open(Path(__file__).resolve().parents[1] / "config"
                          / "orbbec_intrinsics_CPBLC53000PE.json"))
    rect = [336, 228, 560, 480]
    poly = depth_rect_to_poly(rect, 0.7, intr, "extrinsic")
    assert poly is not None and len(poly) == 4
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    back = roi_to_depth_rect(min(xs), min(ys), max(xs) - min(xs),
                             max(ys) - min(ys), 0.7, intr, "extrinsic")
    # The ~0.2 deg extrinsic rotation makes the projected rect a hair
    # non-axis-aligned; taking its axis-aligned bbox costs a couple px.
    assert all(abs(b - w) <= 3 for b, w in zip(back, rect))


def test_yml_snippet_shape():
    s = yml_snippet(("rect", [1, 2, 3, 4]), 0.7, "extrinsic", "SER")
    assert "SER" in s and "- [1, 2, 3, 4]" in s
    assert "self_mask_pixels:" in s


def test_shape_kind():
    assert shape_kind([336, 228, 560, 480]) == "rect"
    assert shape_kind([[540, 375], [870, 377], [868, 748], [538, 746]]) == "poly"
    assert shape_kind([[1, 2], [3, 4], [5, 6]]) == "poly"
    # float rect coords are still a rect
    assert shape_kind([1.0, 2.0, 3.0, 4.0]) == "rect"


def test_poly_to_depth_poly_identity():
    # Matched-FOV synthetic rig: color -> depth scales by 848/1280 (x)
    # and 480/800 (y) around the principal points.
    pts = [[320, 400], [640, 400], [640, 800], [320, 800]]
    out = poly_to_depth_poly(pts, 1.0, INTR_ID, "extrinsic")
    assert out == [[212, 240], [424, 240], [424, 479], [212, 479]]


def test_poly_to_depth_poly_clamps():
    pts = [[-100, -100], [2000, -100], [2000, 1500], [-100, 1500]]
    out = poly_to_depth_poly(pts, 1.0, INTR_ID, "extrinsic")
    assert all(0 <= x <= 847 and 0 <= y <= 479 for x, y in out)


def test_poly_round_trip_real_intrinsics():
    intr = json.load(open(Path(__file__).resolve().parents[1] / "config"
                          / "orbbec_intrinsics_CPBLC53000PE.json"))
    poly = [[540, 375], [870, 377], [868, 748], [538, 746]]
    d = poly_to_depth_poly(poly, 0.7, intr, "extrinsic")
    c = depth_poly_to_color_poly(d, 0.7, intr, "extrinsic")
    # 2 px projection noise; a vertex clamped to the depth-frame edge
    # (here y=748 projects past the 480-row depth frame -> 479) rounds
    # out by up to ~4 px when mapped back.
    assert all(abs(a[0] - b[0]) <= 2 and abs(a[1] - b[1]) <= 4
               for a, b in zip(c, poly))


def test_yml_snippet_polygon():
    s = yml_snippet(("poly", [[540, 375], [870, 377], [868, 748]]),
                    0.7, "extrinsic", "SER")
    assert "- [[540, 375], [870, 377], [868, 748]]" in s
    assert "polygon vertices" in s
