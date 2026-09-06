"""Synthetic-depth unit tests for the BEV pipeline (plan Section 6.7).

Renders depth images by ray-casting a flat-floor scene (+ boxes / stair
segments) through the exact mount + intrinsics model the BEV code uses,
so deprojection, plane fitting, classification and fusion are all
exercised end-to-end without hardware.
"""

import math

import numpy as np
import pytest

from bebop_vision.bev import (FREE, HAZARD, INFLATED, OCCUPIED, BevBuilder,
                              Mount, deproject, mount_rotation)
from bebop_vision.orbbec import StampedFrame

FX = FY = 424.0
CX, CY = 424.0, 240.0
W, H = 848, 480

NEAR = "CPBLC53000PE"
FAR = "CPBLC53000ED"


def make_builder(inflate_radius_m=0.0, seed=0):
    cfg = {
        "cameras": {NEAR: {"role": "near"}, FAR: {"role": "far"}},
        "bev": {"range_m": 3.0, "width_m": 3.0, "cell_m": 0.05,
                "near_authority_m": 1.5},
        "robot": {"radius_m": inflate_radius_m},
        "safety": {"plane_residual_m": 0.015, "plane_min_inliers": 50,
                   "plane_min_frac": 0.2, "hazard_drop_m": 0.05,
                   "max_frame_age_s": 0.3},
    }
    intr = {s: {"fx": FX, "fy": FY, "cx": CX, "cy": CY,
                "width": W, "height": H} for s in (NEAR, FAR)}
    mounts = {NEAR: Mount(0.55, -35.0, 0.0), FAR: Mount(0.75, -12.0, 0.0)}
    return BevBuilder(rig_cfg=cfg, mounts=mounts, intrinsics=intr,
                      inflate_radius_m=inflate_radius_m, seed=seed)


def render_depth(mount, boxes=(), floor_segments=((0.0, 100.0),), w=W, h=H,
                 max_range=6.0):
    """Ray-cast a Y16 depth image (mm, 0 = invalid).

    floor_segments: (plane_z, x_max) pieces — the first segment whose plane
    the ray reaches with hit-x < x_max wins (min range).
    """
    R = mount_rotation(mount.pitch_deg, mount.yaw_deg)
    t = np.array([0.0, 0.0, mount.height_m])
    u = np.arange(w) - CX
    v = np.arange(h) - CY
    uu, vv = np.meshgrid(u, v)
    dir_cam = np.stack([uu / FX, vv / FY, np.ones_like(uu, float)], axis=-1)
    D = dir_cam @ R.T
    z = np.full((h, w), np.inf)
    for z0, xmax in floor_segments:
        with np.errstate(divide="ignore", invalid="ignore"):
            tz = (z0 - t[2]) / D[..., 2]
        px = t[0] + tz * D[..., 0]
        py = t[1] + tz * D[..., 1]
        hit = np.isfinite(tz) & (tz > 0) & (tz < max_range) & (px < xmax) \
            & (np.abs(py) < 50.0)
        z = np.where(hit & (tz < z), tz, z)
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
        hit = (tmin < tmax) & (tmax > 0) & (tmin > 0) & (tmin < max_range)
        z = np.where(hit & (tmin < z), tmin, z)
    return np.where(np.isfinite(z), z * 1000.0, 0.0).astype(np.uint16)


def frame(depth, serial=NEAR, role="near"):
    return StampedFrame(depth=depth, stamp_us=1234, recv_ts=0.0,
                        width=depth.shape[1], height=depth.shape[0],
                        fps=30.0, serial=serial, role=role)


def cell_at(builder, x, y):
    row = int((builder.range_m - x) / builder.cell_m)
    col = int((y + builder.width_m / 2.0) / builder.cell_m)
    return min(max(row, 0), builder.rows - 1), min(max(col, 0), builder.cols - 1)


def test_mount_rotation_sign():
    # pitch_deg negative = camera center points below horizon: the camera's
    # forward axis must map to a body direction with negative z.
    d = mount_rotation(-35.0, 0.0) @ np.array([0.0, 0.0, 1.0])
    assert d[2] < 0
    assert d[0] > 0  # still forward
    # A level camera looks straight ahead.
    d0 = mount_rotation(0.0, 0.0) @ np.array([0.0, 0.0, 1.0])
    assert np.allclose(d0, [1.0, 0.0, 0.0])


def test_center_ray_floor_distance():
    # Camera at 0.55 m pitched down 35 deg: the optical center hits the
    # floor at 0.55 / tan(35 deg) ahead.
    b = make_builder()
    depth = render_depth(b.mounts[NEAR])
    pts = b.to_body(deproject(depth, FX, FY, CX, CY), NEAR)
    m = b.mounts[NEAR]
    expected = m.height_m / math.tan(math.radians(-m.pitch_deg))
    center = pts[(np.abs(pts[:, 1]) < 0.01) & (np.abs(pts[:, 0] - expected) < 0.05)]
    assert len(center) > 0


def test_flat_floor_free():
    b = make_builder()
    res = b.process(frame(render_depth(b.mounts[NEAR])))
    assert res["plane_ok"]
    assert res["occ"].max() == FREE
    assert res["coverage"].sum() > 300  # most of the 3x3 m grid observed


def test_box_occupied_at_right_cells():
    b = make_builder()
    box = (0.95, 1.15, -0.25, 0.25, 0.0, 0.6)
    res = b.process(frame(render_depth(b.mounts[NEAR], boxes=[box])))
    assert res["plane_ok"]
    occ = res["occ"]
    r0, c0 = cell_at(b, 1.05, 0.0)
    block = occ[r0 - 2:r0 + 3, c0 - 5:c0 + 6]
    assert (block == OCCUPIED).any()
    # Clear floor well before and after the box.
    for x, y in ((0.5, 0.0), (2.0, 0.0), (1.05, 0.8)):
        r, c = cell_at(b, x, y)
        assert occ[r, c] == FREE, f"cell at ({x},{y}) not free"
    # Box top (0.6 m) is overhang and must be ignored.


def test_overhang_ignored():
    b = make_builder()
    floating = (0.95, 1.15, -0.25, 0.25, 0.5, 0.8)
    res = b.process(frame(render_depth(b.mounts[NEAR], boxes=[floating])))
    assert res["occ"].max() == FREE


def test_stair_drop_hazard():
    b = make_builder()
    depth = render_depth(b.mounts[NEAR],
                         floor_segments=((0.0, 1.5), (-0.3, 100.0)))
    res = b.process(frame(depth))
    occ = res["occ"]
    # The lower floor is occluded by the step edge until ~2.4 m from this
    # mount (0.55 m height, -35 deg); the first visible lower-floor cells
    # must read as hazard, the upper floor as free.
    r, c = cell_at(b, 2.6, 0.0)
    assert occ[r, c] == HAZARD
    r, c = cell_at(b, 1.0, 0.0)
    assert occ[r, c] == FREE


def test_near_authority_fusion():
    b = make_builder()
    shape = (b.rows, b.cols)
    near = dict(occ=np.zeros(shape, np.uint8),
                coverage=np.ones(shape, bool), plane_ok=True, stamp_us=1)
    far = dict(occ=np.zeros(shape, np.uint8),
               coverage=np.ones(shape, bool), plane_ok=True, stamp_us=2)
    ra, ca = cell_at(b, 1.0, 0.0)    # inside near authority (< 1.5 m)
    rb, cb = cell_at(b, 2.5, 0.0)    # beyond near authority
    ra2, ca2 = cell_at(b, 1.0, 0.5)  # near zone, laterally distinct cell
    near["occ"][ra, ca] = OCCUPIED   # near sees it, far does not
    far["occ"][ra2, ca2] = OCCUPIED  # far-only claim in the near zone
    far["occ"][rb, ca] = OCCUPIED    # far-only claim far out
    g = b.fuse({"near": near, "far": far}, {"near": 0.01, "far": 0.02})
    assert g.occ[ra, ca] == OCCUPIED    # near contribution kept
    assert g.occ[ra2, ca2] == FREE      # near authority overrides far
    assert g.occ[rb, ca] == OCCUPIED    # union beyond 1.5 m
    assert set(g.roles) == {"near", "far"}


def test_inflation_marks_ring():
    b = make_builder(inflate_radius_m=0.20)
    box = (0.95, 1.15, -0.25, 0.25, 0.0, 0.6)
    res = b.process(frame(render_depth(b.mounts[NEAR], boxes=[box])))
    g = b.fuse({"near": res}, {"near": 0.01})
    r, c = cell_at(b, 0.85, 0.0)  # 10 cm in front of the box: inside the radius
    assert g.occ[r, c] == INFLATED
    assert g.occ[cell_at(b, 0.5, 0.0)] == FREE
    assert g.raw[cell_at(b, 0.85, 0.0)] == FREE  # raw grid stays clean


def test_stale_camera_degrades():
    b = make_builder()
    shape = (b.rows, b.cols)
    near = dict(occ=np.zeros(shape, np.uint8),
                coverage=np.ones(shape, bool), plane_ok=True, stamp_us=1)
    far = dict(occ=np.zeros(shape, np.uint8),
               coverage=np.ones(shape, bool), plane_ok=True, stamp_us=2)
    far["occ"][cell_at(b, 2.5, 0.0)] = OCCUPIED
    g = b.fuse({"near": near, "far": None}, {"near": 0.01, "far": 5.0})
    assert g.roles == ["near"]
    assert g.occ[cell_at(b, 2.5, 0.0)] == FREE  # far's region is gone
    assert g.per_camera_age_s["far"] == 5.0
    g = b.fuse({"near": None, "far": far}, {"near": 5.0, "far": 0.01})
    assert g.roles == ["far"]
    assert g.occ[cell_at(b, 2.5, 0.0)] == OCCUPIED  # far fills alone
    assert b.fuse({"near": None, "far": None}, {}) is None


def test_plane_fit_failure_falls_back_to_prior():
    b = make_builder()
    wall = (0.5, 0.6, -5.0, 5.0, -1.0, 3.0)
    res = b.process(frame(render_depth(b.mounts[NEAR], boxes=[wall],
                                       floor_segments=())))
    assert not res["plane_ok"]  # prior z=0 plane used that tick


def test_deproject_drops_invalid_and_far():
    depth = np.zeros((H, W), np.uint16)
    depth[240, 424] = 1500   # 1.5 m straight ahead
    depth[100, 200] = 9000   # beyond max range -> dropped
    pts = deproject(depth, FX, FY, CX, CY, stride=1)
    assert len(pts) == 1
    assert pts[0, 2] == pytest.approx(1.5)
