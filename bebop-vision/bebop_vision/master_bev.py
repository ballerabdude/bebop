"""Shared depth-geometry helpers kept from the retired BEV stack.

The BEV itself is gone (the geometric builder in 2026-10, the TensorRT
student, its service and the operator ``bev`` stream on 2026-10-23); this
module keeps only what the offline SAM x depth label pipeline and the
chassis-footprint tool still use:

* ``mount_rotation`` (camera-optical -> body for a mounted camera) and
  ``fit_ground_plane`` (RANSAC floor plane);
* ``load_ego_raster`` / ``rasterize_ego`` for the learned chassis raster
  written by ``tools/bev_ego_from_data.py`` (``config/ego_mask.npz``).

The module name is historical.  The removed ``BevGrid`` contract and the
renderers are at ``git show 1f22b61:bebop-vision/bebop_vision/master_bev.py``.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

# Keep OpenBLAS/OpenMP from leaving a spinning pool behind; see main.py.
for _thread_var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_thread_var, "1")

import numpy as np


def mount_rotation(pitch_deg, yaw_deg):
    """Camera-optical -> body rotation for a mounted camera.

    Optical: x right, y down, z forward.  Body: x forward, y left, z up.
    ``pitch_deg`` negative means the camera centre points below the horizon.
    """
    p = math.radians(pitch_deg)
    y = math.radians(yaw_deg)
    cp, sp = math.cos(-p), math.sin(-p)
    cy, sy = math.cos(y), math.sin(y)
    r0 = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
                  dtype=np.float32)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]],
                  dtype=np.float32)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]],
                  dtype=np.float32)
    return rz @ ry @ r0


def fit_ground_plane(pts, *, residual_m=0.015, min_inliers=50, min_frac=0.15,
                     iters=96, rng=None, d_range=(-0.20, 0.20),
                     range_scale=0.01, ranges=None, normal_deg=20.0):
    """Batched RANSAC ground plane in the body frame.

    Returns ``(n, d, inlier_count)`` with ``n·p = d`` and ``n_z > 0``, or
    ``None`` when the fit is not trustworthy.  The floor prior gates candidate
    planes to an almost-vertical normal and a plane height at the origin
    inside ``d_range`` (the body origin is the floor, so a well-calibrated rig
    lands near d=0).  Residuals grow with range: ``max(residual_m,
    range_scale * range)``.
    """
    n_pts = len(pts)
    if n_pts < max(3, min_inliers):
        return None
    rng = rng or np.random.default_rng()
    # Sampling without replacement needs iters * 3 distinct points; a sparse
    # near-floor band (e.g. facing a wall) can have fewer, so fall back to
    # sampling with replacement there.  Degenerate triplets with repeated
    # points give a zero normal and are rejected below.
    idx = rng.choice(n_pts, (iters, 3), replace=n_pts < iters * 3)
    p = pts[idx]
    n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    norm = np.linalg.norm(n, axis=1)
    valid = norm > 1e-6
    n[valid] /= norm[valid][:, None]
    flip = n[:, 2] < 0
    n[flip] = -n[flip]
    valid &= n[:, 2] >= math.cos(math.radians(normal_deg))
    d = np.einsum("ij,ij->i", n, p[:, 0])
    valid &= (d >= d_range[0]) & (d <= d_range[1])

    n_eval = min(n_pts, 6000)
    if n_eval == n_pts:
        pts_e, r_e = pts, ranges
    else:
        ev = rng.choice(n_pts, n_eval, replace=False)
        pts_e = pts[ev]
        r_e = None if ranges is None else ranges[ev]
    dist = np.abs(pts_e @ n.T - d[None, :])
    if ranges is not None:
        dist = dist < np.maximum(residual_m, range_scale * r_e)[:, None]
        counts = dist.sum(axis=0)
    else:
        counts = (dist < residual_m).sum(axis=0)
    counts[~valid] = 0
    best = int(counts.argmax())
    if not valid[best] or counts[best] == 0:
        return None

    full = np.abs(pts @ n[best] - d[best])
    if ranges is not None:
        full = full < np.maximum(residual_m, range_scale * ranges)
    else:
        full = full < residual_m
    cnt = int(full.sum())
    if cnt >= min_inliers and cnt >= min_frac * n_pts:
        return n[best], float(d[best]), cnt
    return None


def load_ego_raster(path):
    """Body-frame chassis raster written by tools/bev_ego_from_data.py.

    Returns {mask [nx, ny] bool, x0, y0, cell, name, path}: index 0 sits at
    (x0, y0) metres, index i covers [x0 + i*cell, x0 + (i+1)*cell).
    """
    path = Path(path)
    with np.load(path) as d:
        return {"mask": np.asarray(d["mask"], bool),
                "x0": float(d["x0"]), "y0": float(d["y0"]),
                "cell": float(d["cell"]), "name": path.name,
                "path": str(path)}


def rasterize_ego(raster, xc, yc):
    """Chassis raster -> bool [len(xc), len(yc)] on a grid of cell centres.

    A grid cell is ego when its centre falls in a raster cell of the mask;
    centres outside the raster are not ego.
    """
    mask = raster["mask"]
    xc = np.asarray(xc, np.float64)
    yc = np.asarray(yc, np.float64)
    ix = np.floor((xc - raster["x0"]) / raster["cell"]).astype(np.int64)
    iy = np.floor((yc - raster["y0"]) / raster["cell"]).astype(np.int64)
    ok_x = (ix >= 0) & (ix < mask.shape[0])
    ok_y = (iy >= 0) & (iy < mask.shape[1])
    out = np.zeros((len(xc), len(yc)), bool)
    sub = mask[np.clip(ix, 0, mask.shape[0] - 1)[:, None],
               np.clip(iy, 0, mask.shape[1] - 1)[None, :]]
    out[ok_x[:, None] & ok_y[None, :]] = sub[ok_x[:, None] & ok_y[None, :]]
    return out
