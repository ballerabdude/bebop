"""Synthetic-depth tests for the master BEV (no hardware needed).

The scene is ray-cast through the exact mount + intrinsics model the builder
uses, so deprojection, the joint floor plane, per-camera classification and
the weighted fusion are all exercised end to end.

Camera geometry mirrors the robot: near 1.27 m / -66 deg, far 1.32 m /
-17.4 deg.  The near camera covers the floor from ~0.55 m to ~1.75 m; the
far camera covers ~1.2 m to the horizon.  That overlap is what makes a
near-only or far-only regression visible in the tests.
"""

import time

import numpy as np

from bebop_vision.master_bev import (FREE, HAZARD, INFLATED, OCCUPIED,
                                     SRC_BOTH, SRC_FAR, SRC_NEAR,
                                     MasterBevBuilder, Mount, mount_rotation,
                                     render_bev, render_source)
from bebop_vision.orbbec import StampedFrame

W, H = 424, 240
FX = FY = 206.7
CX = {"near": 208.36, "far": 209.85}
CY = 121.33

NEAR = "CPBLC53000PE"
FAR = "CPBLC53000ED"
SERIAL = {"near": NEAR, "far": FAR}
ROLE = {NEAR: "near", FAR: "far"}


def make_builder(radius_m=0.0, seed=0):
    cfg = {
        "cameras": {
            NEAR: {"role": "near", "height_m": 1.27, "pitch_deg": -66.0,
                   "yaw_deg": 0.0},
            FAR: {"role": "far", "height_m": 1.32, "pitch_deg": -17.4,
                  "yaw_deg": 0.0},
        },
        "master_bev": {
            "range_m": 6.0, "width_m": 3.0, "cell_m": 0.05,
            "min_range_m": 0.55, "max_range_m": 8.0,
            "stride_x": 2, "stride_y_near": 1, "stride_y_far": 1,
            "near_range_m": [0.55, 1.20, 2.00],
            "far_range_m": [1.00, 2.00, 4.50, 7.00],
            "occ_min_w": 0.75,
            "occ_dom_ratio": 0.35, "haz_min_w": 1.0,
        },
        "robot": {
            "radius_m": radius_m,
            "self_mask": {
                "x_range_m": [-0.30, 1.00],
                "y_range_m": [-0.30, 0.60],
                "min_height_m": 0.05,
            },
        },
        "safety": {
            "plane_residual_m": 0.015,
            "plane_min_inliers": 50,
            "plane_min_frac": 0.15,
            "hazard_drop_m": 0.05,
            "max_frame_age_s": 0.3,
        },
    }
    mounts = {
        NEAR: Mount(1.27, -66.0, 0.0),
        FAR: Mount(1.32, -17.4, 0.0),
    }
    intrinsics = {
        s: {"fx": FX, "fy": FY, "cx": CX[ROLE[s]], "cy": CY,
            "width": W, "height": H}
        for s in (NEAR, FAR)
    }
    return MasterBevBuilder(rig_cfg=cfg, mounts=mounts,
                            intrinsics=intrinsics, seed=seed)


def render_depth(mount, role, *, boxes=(), floors=((0.0, -10.0, 10.0),),
                 w=W, h=H, max_range=8.0):
    """Ray-cast a Y16 depth image (mm, 0 = invalid).

    ``floors``: (plane_z, x_min, x_max) pieces.  A ray takes the nearest
    plane it reaches inside the x window.  ``boxes``: (x0, x1, y0, y1, z0, z1)
    slabs.
    """
    R = mount_rotation(mount.pitch_deg, mount.yaw_deg)
    t = np.array([mount.x_m, mount.y_m, mount.height_m], dtype=np.float64)
    u = (np.arange(w) - CX[role]).astype(np.float64)
    v = (np.arange(h) - CY).astype(np.float64)
    uu, vv = np.meshgrid(u, v)
    dir_cam = np.stack([uu / FX, vv / FY, np.ones_like(uu)], axis=-1)
    D = dir_cam @ R.T
    depth = np.full((h, w), np.inf)

    for z0, x0, x1 in floors:
        with np.errstate(divide="ignore", invalid="ignore"):
            tz = (z0 - t[2]) / D[..., 2]
        px = t[0] + tz * D[..., 0]
        py = t[1] + tz * D[..., 1]
        hit = (np.isfinite(tz) & (tz > 0) & (tz < max_range)
               & (px >= x0) & (px < x1) & (np.abs(py) < 20.0))
        depth = np.where(hit & (tz < depth), tz, depth)

    for x0, x1, y0, y1, z0, z1 in boxes:
        tmin = np.full((h, w), -np.inf)
        tmax = np.full((h, w), np.inf)
        for lo, hi, o, A in ((x0, x1, t[0], D[..., 0]),
                             (y0, y1, t[1], D[..., 1]),
                             (z0, z1, t[2], D[..., 2])):
            with np.errstate(divide="ignore", invalid="ignore"):
                ta = (lo - o) / A
                tb = (hi - o) / A
            tmin = np.maximum(tmin, np.minimum(ta, tb))
            tmax = np.minimum(tmax, np.maximum(ta, tb))
        hit = ((tmin < tmax) & (tmax > 0) & (tmin > 0)
               & (tmin < max_range))
        depth = np.where(hit & (tmin < depth), tmin, depth)

    return np.where(np.isfinite(depth), depth * 1000.0, 0.0).astype(np.uint16)


def frame(depth, role, stamp_us=1234):
    serial = SERIAL[role]
    return StampedFrame(depth=depth, stamp_us=stamp_us,
                        recv_ts=time.monotonic(),
                        width=depth.shape[1], height=depth.shape[0],
                        fps=30.0, serial=serial, role=role,
                        depth_scale_mm=1.0)


def scene(*, boxes=(), floors=((0.0, -10.0, 10.0),)):
    return {
        role: frame(render_depth(Mount(1.27 if role == "near" else 1.32,
                                       -66.0 if role == "near" else -17.4),
                                 role, boxes=boxes, floors=floors), role)
        for role in ("near", "far")
    }


def prepared(builder, frames):
    return {role: builder.prepare(f) for role, f in frames.items()}


def fuse(builder, frames):
    return builder.fuse(prepared(builder, frames),
                        {"near": 0.01, "far": 0.01})


def cell_at(builder, x, y):
    row = int((builder.range_m - x) / builder.cell_m)
    col = int((y + builder.width_m / 2.0) / builder.cell_m)
    return row, col


# --- geometry ---------------------------------------------------------------

def test_grid_is_nonsquare():
    b = make_builder()
    assert (b.range_m, b.width_m, b.cell_m) == (6.0, 3.0, 0.05)
    assert b.rows == 120
    assert b.cols == 60


# --- the dual-camera map ----------------------------------------------------

def test_flat_floor_uses_both_cameras():
    b = make_builder()
    g = fuse(b, scene())
    assert g is not None
    assert g.shape == (120, 60)
    assert g.plane["ok"] is True
    assert g.plane_ok["near"] is True
    assert g.plane_ok["far"] is True

    for x in (1.0, 3.0, 5.0):
        r, c = cell_at(b, x, 0.0)
        assert g.coverage[r, c], f"x={x} should be observed"
        assert g.occ[r, c] == FREE

    # Close cells are near-only, the overlap is both, far-only appears out
    # past the near camera's floor horizon.
    assert SRC_NEAR in set(np.unique(g.source))
    assert SRC_FAR in set(np.unique(g.source))
    assert SRC_BOTH in set(np.unique(g.source))

    assert g.contributions["near"]["points"] > 0
    assert g.contributions["far"]["points"] > 0
    assert g.contributions["near"]["cells"] > 0
    assert g.contributions["far"]["cells"] > 0


def test_far_camera_reaches_past_the_near_camera():
    b = make_builder()
    frames = scene()
    g_far = b.fuse({"far": b.prepare(frames["far"])}, {"far": 0.01})
    r_near, c = cell_at(b, 1.0, 0.0)
    r_far, c = cell_at(b, 4.0, 0.0)
    # The far camera cannot see the floor inside ~1.2 m...
    assert not g_far.coverage[r_near, c]
    # ...but it owns the distance.
    assert g_far.coverage[r_far, c]
    assert g_far.occ[r_far, c] == FREE
    assert g_far.source[r_far, c] & SRC_FAR


def test_near_camera_covers_the_near_field_only():
    b = make_builder()
    frames = scene()
    g_near = b.fuse({"near": b.prepare(frames["near"])}, {"near": 0.01})
    r_near, c = cell_at(b, 1.0, 0.0)
    r_far, c = cell_at(b, 4.0, 0.0)
    assert g_near.coverage[r_near, c]
    assert g_near.occ[r_near, c] == FREE
    assert g_near.source[r_near, c] & SRC_NEAR
    assert not g_near.coverage[r_far, c]


def test_near_and_far_boxes_are_both_reported():
    b = make_builder()
    boxes = (
        (1.15, 1.35, -0.20, 0.20, 0.00, 0.40),   # near
        (2.40, 2.60, -0.20, 0.20, 0.00, 0.40),   # far
    )
    g = fuse(b, scene(boxes=boxes))
    # The visible face of each box is its near side: x=1.15 and x=2.40.
    r_near, c = cell_at(b, 1.12, 0.0)
    r_far, c = cell_at(b, 2.38, 0.0)
    assert g.occ[r_near, c] == OCCUPIED
    assert g.occ[r_far, c] == OCCUPIED
    assert g.source[r_far, c] & SRC_FAR
    # A couple of boxes must not flood the map with occupied cells.
    assert int(np.sum(g.raw == OCCUPIED)) < 200


def test_unknown_is_not_painted_as_free():
    b = make_builder()
    g = fuse(b, scene())
    # Inside the dead disc / behind the near camera's min range.
    r, c = cell_at(b, 0.20, 0.0)
    assert not g.coverage[r, c]
    # The far camera still observes the distance.
    r, c = cell_at(b, 5.0, 0.0)
    assert g.coverage[r, c]


def test_joint_floor_plane_is_consistent():
    b = make_builder()
    g = fuse(b, scene())
    n = np.asarray(g.plane["n"])
    assert abs(g.plane["d"]) < 0.05
    assert n[2] > 0.95
    assert g.plane["inliers"] > 100
    assert g.plane["residual_m"] is not None
    assert g.plane["residual_m"] < 0.03


def test_hazard_detects_a_step_down():
    b = make_builder()
    # Floor at z=0 in front of x=2.0; floor 30 cm lower beyond it, with the
    # vertical riser at the edge.
    boxes = ((1.99, 2.01, -2.0, 2.0, -0.30, 0.0),)
    floors = ((0.0, -10.0, 2.0), (-0.30, 2.0, 10.0))
    g = fuse(b, scene(boxes=boxes, floors=floors))
    assert g is not None
    # Look for hazard cells in the horizontal band around the drop.
    band = g.raw[:, :]
    rows = [cell_at(b, x, 0.0)[0] for x in (2.1, 2.3, 2.5)]
    cols = [cell_at(b, x, 0.0)[1] for x in (2.1, 2.3, 2.5)]
    assert any(band[r, c] == HAZARD for r, c in zip(rows, cols)), \
        "the step-down should produce hazard cells"


def test_inflation_marks_a_ring_around_the_box():
    b = make_builder(radius_m=0.20)
    boxes = ((1.15, 1.35, -0.20, 0.20, 0.00, 0.40),)
    g = fuse(b, scene(boxes=boxes))
    assert np.any(g.occ == INFLATED)
    assert not np.any(g.raw == INFLATED)


# --- serialization / rendering ----------------------------------------------

def test_to_dict_round_trips_the_full_grid():
    b = make_builder()
    g = fuse(b, scene())
    d = g.to_dict()
    assert d["rows"] == 120 and d["cols"] == 60
    assert d["encoding"] == "rle_pairs"
    total = d["rows"] * d["cols"]
    for key in ("occ_rle", "raw_rle", "coverage_rle", "source_rle"):
        assert sum(count for _, count in d[key]) == total
    c = d["counts"]
    assert (c["free"] + c["occupied"] + c["hazard"] + c["inflated"]
            + c["unknown"]) == total
    assert d["plane"]["ok"] is True
    assert d["contributions"]["far"]["points"] > 0


def test_render_shape_and_source_view():
    b = make_builder()
    g = fuse(b, scene())
    img = render_bev(g, scale=4)
    assert img.shape == (480, 240, 3)
    assert img.dtype == np.uint8
    src = render_source(g, scale=4)
    assert src.shape == (480, 240, 3)

    # Unknown must not be painted as observed free.
    from bebop_vision.master_bev import BEV_COLORS, UNKNOWN_COLOR
    # Pick cells away from the HUD, the range rings and the robot footprint:
    # a side cell inside the dead disc is unknown, a nearby floor cell free.
    r_unknown, c = cell_at(b, 0.12, 1.28)
    r_free, c = cell_at(b, 0.80, 0.25)
    assert not g.coverage[r_unknown, c]
    assert g.coverage[r_free, c]
    # The render mirrors horizontally; sample the cell's pixel centre.
    def pixel(row, col, scale=4):
        return img[row * scale + scale // 2,
                   (g.cols - 1 - col) * scale + scale // 2]
    assert tuple(pixel(r_unknown, c)) == UNKNOWN_COLOR
    assert tuple(pixel(r_free, c)) == BEV_COLORS[FREE]


def test_stale_camera_degrades_gracefully():
    b = make_builder()
    frames = scene()
    g = b.fuse({"near": b.prepare(frames["near"]), "far": None},
               {"near": 0.01})
    assert g is not None
    assert g.roles == ["near"]
    assert "far" not in g.contributions
    r, c = cell_at(b, 1.0, 0.0)
    assert g.coverage[r, c]
    r, c = cell_at(b, 4.0, 0.0)
    assert not g.coverage[r, c]
