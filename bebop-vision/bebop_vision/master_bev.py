"""BEV grid contract, rendering and shared depth geometry helpers.

The **live** BEV is now the learned TensorRT student
(``bebop_vision.bev_student_runtime``); the old geometric occupancy builder
was removed.  This module keeps only what is still used:

* ``BevGrid`` and its JSON/RLE/semantic rendering contract (app/planner);
* ``mount_rotation`` / ``fit_ground_plane`` for the offline SAM/depth label
  and training pipeline.

The module name is historical; there is no geometric runtime builder here.
"""

from __future__ import annotations

import dataclasses
import math
import os
import time

# Keep OpenBLAS/OpenMP from leaving a spinning pool behind; see main.py.
for _thread_var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_thread_var, "1")

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise ImportError("pip install opencv-python") from exc

# against that grid can consume this one unchanged.
FREE, OCCUPIED, HAZARD, INFLATED = 0, 1, 2, 3

# Per-cell source attribution.
SRC_NONE, SRC_NEAR, SRC_FAR, SRC_BOTH = 0, 1, 2, 3


@dataclasses.dataclass
class BevGrid:
    """One fused master-BEV tick.

    ``occ`` and ``raw`` keep the old navd contract; the extra layers make the
    fusion inspectable and let the operator UI tell unknown from free.
    """

    occ: np.ndarray             # uint8 (R, C), INFLATED applied
    raw: np.ndarray             # uint8 (R, C), pre-inflation
    coverage: np.ndarray        # bool (R, C): observed this tick
    source: np.ndarray          # uint8 (R, C): SRC_*
    confidence: np.ndarray      # uint8 (R, C): 0..255 evidence strength
    cell_m: float
    range_m: float
    width_m: float
    stamp_us: int
    recv_ts: float
    per_camera_age_s: dict
    plane_ok: dict
    plane: dict
    contributions: dict
    roles: list
    # Optional learned-student layers.  The geometric builder leaves these
    # None; the model-backed producer fills them.  ``occ``/``raw`` remain
    # the planner-compatible layers.
    semantic: np.ndarray = None   # uint8 (R,C): 0 floor,1 wall,2 obstacle,3 hazard,255 ignore
    wall: np.ndarray = None       # bool (R,C)
    ego: np.ndarray = None        # bool (R,C)

    @property
    def shape(self):
        return self.occ.shape

    @property
    def rows(self):
        return int(self.occ.shape[0])

    @property
    def cols(self):
        return int(self.occ.shape[1])

    def counts(self) -> dict:
        known = self.coverage
        return {
            "free": int(np.sum(known & (self.occ == FREE))),
            "occupied": int(np.sum(self.occ == OCCUPIED)),
            "hazard": int(np.sum(self.occ == HAZARD)),
            "inflated": int(np.sum(self.occ == INFLATED)),
            "unknown": int(np.sum(~known)),
        }

    def to_dict(self) -> dict:
        """JSON-ready payload (compact run-length layers)."""
        payload = {
            "stamp_us": int(self.stamp_us),
            "age_s": round(time.monotonic() - self.recv_ts, 3),
            "cell_m": self.cell_m,
            "range_m": self.range_m,
            "width_m": self.width_m,
            "rows": self.rows,
            "cols": self.cols,
            "encoding": "rle_pairs",
            "occ_rle": _rle(self.occ),
            "raw_rle": _rle(self.raw),
            "coverage_rle": _rle(self.coverage.astype(np.uint8)),
            "source_rle": _rle(self.source),
            "counts": self.counts(),
            "plane_ok": self.plane_ok,
            "plane": self.plane,
            "ages_s": self.per_camera_age_s,
            "roles": self.roles,
            "contributions": self.contributions,
        }
        if self.semantic is not None:
            sem = np.asarray(self.semantic)
            payload["semantic_rle"] = _rle(sem)
            payload["semantic_counts"] = {
                "floor": int(np.sum(sem == 0)),
                "wall": int(np.sum(sem == 1)),
                "obstacle": int(np.sum(sem == 2)),
                "hazard": int(np.sum(sem == 3)),
                "unknown": int(np.sum(sem == 255)),
            }
        if self.wall is not None:
            payload["wall_rle"] = _rle(np.asarray(self.wall, np.uint8))
        if self.ego is not None:
            payload["ego_rle"] = _rle(np.asarray(self.ego, np.uint8))
        return payload


def _rle(values) -> list[list[int]]:
    """Row-major run-length encoding: [[value, count], ...]."""
    flat = np.asarray(values).reshape(-1)
    if flat.size == 0:
        return []
    change = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [flat.size]))
    return [[int(flat[s]), int(e - s)] for s, e in zip(starts, ends)]



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
    idx = rng.choice(n_pts, (iters, 3), replace=False)
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



# Rendering (debug / operator view)
# ---------------------------------------------------------------------------

BEV_COLORS = {
    FREE: (70, 70, 70),          # BGR dark grey
    OCCUPIED: (0, 0, 220),       # red
    HAZARD: (0, 140, 255),       # orange
    INFLATED: (60, 60, 120),     # dark red / planning margin
}
UNKNOWN_COLOR = (28, 28, 28)
# Learned semantic palette (BGR): floor grey, wall blue/purple, obstacle
# red, hazard orange.  INFLATED is overlaid dark red; ego is outlined cyan.
SEMANTIC_COLORS = {
    0: (90, 90, 90),
    1: (255, 120, 40),
    2: (40, 40, 220),
    3: (0, 160, 255),
    255: UNKNOWN_COLOR,
}
EGO_FILL = (55, 55, 0)
EGO_COLOR = (220, 220, 40)
SOURCE_COLORS = {
    SRC_NONE: (28, 28, 28),
    SRC_NEAR: (255, 120, 0),     # blue-ish (BGR)
    SRC_FAR: (0, 200, 0),        # green
    SRC_BOTH: (235, 235, 235),   # white
}


def render_bev(grid: BevGrid, *, scale=4, source_view=False,
               show_hud=True):
    """Occupancy grid -> colorized top-down BGR image.

    The grid is drawn in its native convention (row 0 far, col 0 right) and
    mirrored horizontally at the end, so body +y (left) renders left.
    """
    rows, cols = grid.shape
    small = np.zeros((rows, cols, 3), np.uint8)
    if source_view:
        for code, color in SOURCE_COLORS.items():
            small[grid.source == code] = color
        small[~grid.coverage] = UNKNOWN_COLOR
    elif grid.semantic is not None:
        sem = np.asarray(grid.semantic)
        for cls, color in SEMANTIC_COLORS.items():
            small[sem == cls] = color
        small[~grid.coverage] = UNKNOWN_COLOR
        # Inflation is a planning overlay on top of the semantic classes.
        if grid.occ is not None:
            infl = (grid.occ == INFLATED) & grid.coverage
            small[infl] = BEV_COLORS[INFLATED]
        if grid.ego is not None:
            ego_u8 = np.asarray(grid.ego, np.uint8)
            small[ego_u8 > 0] = EGO_FILL
            contours, _ = cv2.findContours(
                ego_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(small, contours, -1, EGO_COLOR, 1)
    else:
        for cls, color in BEV_COLORS.items():
            small[(grid.occ == cls) & grid.coverage] = color
        # Unknown cells are class FREE for planner compatibility; paint them
        # last so they are not mistaken for observed free space.
        small[~grid.coverage] = UNKNOWN_COLOR
    img = np.repeat(np.repeat(small, scale, axis=0), scale, axis=1)
    img = np.ascontiguousarray(img[:, ::-1])
    h, w = img.shape[:2]

    # Origin at the bottom centre; range rings every metre.
    cx = (cols // 2) * scale
    cy = rows * scale
    for metre in range(1, int(grid.range_m) + 1):
        r = int(round(metre / grid.cell_m * scale))
        cv2.circle(img, (cx, cy), r, (95, 95, 95), 1, cv2.LINE_AA)

    # Robot footprint (body 0.4 m wide, 0.5 m long for the wheeled base).
    fw = int(round(0.40 / grid.cell_m * scale))
    fh = int(round(0.50 / grid.cell_m * scale))
    cv2.ellipse(img, (cx, cy), (fw // 2, fh // 2), 0, 180, 360,
                (210, 210, 210), 2, cv2.LINE_AA)

    if show_hud:
        lines = [
            f"BEV {grid.range_m:.1f}x{grid.width_m:.1f}m  "
            f"{grid.cell_m*100:.0f}cm",
        ]
        model = grid.contributions.get("model")
        if model:
            lines.append(
                f"model {model.get('engine', '?')} "
                f"{model.get('infer_ms', '?')}ms "
                f"hazard={model.get('hazard_source', '?')}")
        else:
            for role in grid.roles:
                c = grid.contributions.get(role, {})
                age = c.get("age_s")
                age_s = f"{age:.2f}s" if age is not None else "?"
                lines.append(
                    f"{role}: {c.get('points', 0)}pts "
                    f"{c.get('cells', 0)}cells "
                    f"plane={'ok' if c.get('plane_ok') else 'no'} {age_s}")
        lines.append(
            "occ={occupied} haz={hazard} infl={inflated} unk={unknown}".format(
                **grid.counts()))
        box_h = 8 + 14 * len(lines)
        cv2.rectangle(img, (0, 0), (w, box_h), (0, 0, 0), -1)
        for i, line in enumerate(lines):
            cv2.putText(img, line, (5, 14 + i * 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        (220, 220, 220), 1, cv2.LINE_AA)
    return img


def render_source(grid: BevGrid, *, scale=4):
    """Debug view: cells tinted by the camera(s) that observed them."""
    return render_bev(grid, scale=scale, source_view=True)
