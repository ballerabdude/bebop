"""Synthetic depth rendering for tests and tools (no hardware).

Renders what a `CameraGeometry` would see of a flat floor plus a few
simple primitives, as a uint16 millimetre depth image.  Used by the nav
tests to build frames with known geometry.
"""
from __future__ import annotations

import numpy as np

from .depth_cone import CameraGeometry


def render_depth(geom: CameraGeometry, walls=(), holes=(), boxes=(),
                 floor: bool = True, max_depth_m: float = 8.0) -> np.ndarray:
    """Depth image (H, W) uint16 mm for a scene in the body frame.

    walls: (x, y0, y1, height) — a vertical plane at forward distance x
           spanning y0..y1 and 0..height.
    boxes: (x0, x1, y0, y1, height) — an axis-aligned block on the floor.
           Rendered as its top face plus its front (x0) face.
    holes: (x0, x1, y0, y1, depth) — floor region lowered by `depth`.
    """
    dirs, u, v = geom.sample_rays(1)
    dirs = dirs.astype(np.float64)
    cam = np.array([0.0, 0.0, geom.height_m])
    t_best = np.full(dirs.shape[0], np.inf)

    def consider(t, ok):
        nonlocal t_best
        ok = ok & (t > 0.05) & (t < t_best)
        t_best = np.where(ok, t, t_best)

    with np.errstate(divide="ignore", invalid="ignore"):
        if floor:
            t = -cam[2] / dirs[:, 2]
            ok = dirs[:, 2] < 0
            p = cam + t[:, None] * dirs
            for x0, x1, y0, y1, depth in holes:
                inside = ((p[:, 0] >= x0) & (p[:, 0] <= x1)
                          & (p[:, 1] >= y0) & (p[:, 1] <= y1))
                th = -(cam[2] + depth) / dirs[:, 2]
                ph = cam + th[:, None] * dirs
                still = ((ph[:, 0] >= x0) & (ph[:, 0] <= x1)
                         & (ph[:, 1] >= y0) & (ph[:, 1] <= y1))
                # Rays that enter the hole: hit the lowered floor instead
                # (or the hole's far wall, approximated by the lowered floor).
                t = np.where(inside & still, th, t)
                ok = ok & ~(inside & ~still)
            consider(t, ok)
        for x, y0, y1, h in walls:
            t = x / dirs[:, 0]
            p = cam + t[:, None] * dirs
            ok = ((dirs[:, 0] > 0) & (p[:, 1] >= y0) & (p[:, 1] <= y1)
                  & (p[:, 2] >= 0) & (p[:, 2] <= h))
            consider(t, ok)
        for x0, x1, y0, y1, h in boxes:
            # front face at x0
            t = x0 / dirs[:, 0]
            p = cam + t[:, None] * dirs
            ok = ((dirs[:, 0] > 0) & (p[:, 1] >= y0) & (p[:, 1] <= y1)
                  & (p[:, 2] >= 0) & (p[:, 2] <= h))
            consider(t, ok)
            # top face at z = h
            t = (h - cam[2]) / dirs[:, 2]
            p = cam + t[:, None] * dirs
            ok = ((dirs[:, 2] < 0) & (p[:, 0] >= x0) & (p[:, 0] <= x1)
                  & (p[:, 1] >= y0) & (p[:, 1] <= y1))
            consider(t, ok)
    depth = np.where(np.isfinite(t_best) & (t_best <= max_depth_m),
                     t_best * 1000.0, 0.0)
    img = np.zeros((geom.height, geom.width), dtype=np.uint16)
    img[v, u] = np.clip(depth, 0, 65535).astype(np.uint16)
    return img


def small_geometry(role: str = "near", height_m: float = 1.27,
                   pitch_deg: float = -66.0, width: int = 212,
                   height: int = 120, self_polys=()) -> CameraGeometry:
    """A quarter-resolution Gemini-like camera for fast tests."""
    scale = width / 848.0
    return CameraGeometry(
        role=role, fx=413.42 * scale, fy=413.42 * scale,
        cx=416.71 * scale, cy=242.32 * scale, width=width, height=height,
        height_m=height_m, pitch_deg=pitch_deg, yaw_deg=0.0,
        self_polys=tuple(self_polys))
