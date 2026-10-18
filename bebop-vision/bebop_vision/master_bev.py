"""Master BEV — one fused body-frame occupancy grid from both depth cameras.

The grid is deliberately **not square**: the near camera sees the floor
close to the robot, the far camera sees much further ahead than the robot
is wide.  The default is 6.0 m forward x 3.0 m wide at 5 cm (120 x 60
cells), configurable under ``master_bev`` in ``config/orbbec_rig.yaml``.

Conventions (same family as the old navd BEV):
  * body frame: x forward, y left, z up; origin on the floor under the robot;
  * row 0 = far edge (x = range_m), rows grow toward the robot;
  * col 0 = right edge (y = -width_m/2), cols grow leftward;
  * class ids: 0 FREE, 1 OCCUPIED, 2 HAZARD, 3 INFLATED (planning layer);
  * ``coverage`` is separate from the class ids: an unobserved cell is
    FREE by id (for planner compatibility) but ``coverage`` is False.

Per tick:
  1. ``prepare(frame)`` per camera: depth -> body-frame points.  No floor
     decision is made per camera; that is what makes the fusion stable.
  2. ``fuse(per_camera, ages)``: fit **one joint floor plane** over both
     cameras' points, classify each camera against it, accumulate weighted
     occupied/free/hazard evidence, and fuse into a single grid.

Why the weighted evidence instead of the old hard "near wins below 1.5 m"
switch:
  * the near camera is dense and accurate close, but only reaches ~1.75 m;
  * the far camera reaches 6 m, but is grazing/noisy below ~1.5 m and beyond
    ~5 m;
  * each point carries a range weight, and a cell needs enough weighted
    evidence before it becomes occupied/free/hazard.  One stray point cannot
    create an obstacle, and the near/far handover is a blend, not a seam.
"""

from __future__ import annotations

import dataclasses
import math
import time

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise ImportError("pip install opencv-python") from exc

from .orbbec import load_intrinsics, load_rig_config

# Cell classes.  Kept identical to the old navd BEV so a planner written
# against that grid can consume this one unchanged.
FREE, OCCUPIED, HAZARD, INFLATED = 0, 1, 2, 3

# Per-cell source attribution.
SRC_NONE, SRC_NEAR, SRC_FAR, SRC_BOTH = 0, 1, 2, 3


@dataclasses.dataclass
class Mount:
    """Measured camera mount in the robot body frame."""

    height_m: float
    pitch_deg: float
    yaw_deg: float = 0.0
    x_m: float = 0.0
    y_m: float = 0.0


@dataclasses.dataclass
class CameraPoints:
    """One camera's depth frame, deprojected into the body frame."""

    role: str
    serial: str
    points: np.ndarray          # (N, 3) float32, body frame
    ranges: np.ndarray          # (N,) float32, range from the camera
    stamp_us: int
    n_raw: int                  # valid depth samples before filtering
    n_used: int                 # samples kept after range/mask/grid filters


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
        return {
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


def deproject(depth_m, fx, fy, cx, cy, stride_x=4, stride_y=4,
              max_range_m=8.0):
    """Pinhole-deproject a float depth image (metres) to optical-frame points.

    The strides are separate on purpose: the far camera's floor grazing angle
    needs vertical samples much closer together than the near camera's.
    """
    z = depth_m[::stride_y, ::stride_x]
    h, w = z.shape
    u = (np.arange(w, dtype=np.float32) * stride_x) - cx
    v = (np.arange(h, dtype=np.float32) * stride_y) - cy
    valid = (z > 0.0) & (z < max_range_m)
    x = (u[None, :] / fx) * z
    y = (v[:, None] / fy) * z
    return np.stack([x[valid], y[valid], z[valid]], axis=1).astype(np.float32)


def fit_ground_plane(pts, *, residual_m=0.015, min_inliers=50, min_frac=0.15,
                     iters=96, rng=None, d_range=(-0.20, 0.20),
                     range_scale=0.01, ranges=None):
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
    valid &= n[:, 2] >= math.cos(math.radians(20.0))
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


def _height(points, n, d):
    """Height above the plane n·p = d (n is near-vertical)."""
    nz = float(n[2])
    if abs(nz) < 1e-6:
        return points[:, 2].copy()
    return points[:, 2] - (d - n[0] * points[:, 0] - n[1] * points[:, 1]) / nz


class MasterBevBuilder:
    """Dual-camera depth -> one non-square body-frame occupancy grid."""

    def __init__(self, rig_cfg=None, config_dir=None, mounts=None,
                 intrinsics=None, seed=None):
        cfg = rig_cfg
        if cfg is None:
            cfg = load_rig_config()["robots"]["default"]
        elif "robots" in cfg:
            cfg = cfg["robots"]["default"]
        self.cfg = cfg
        cam_cfg = cfg["cameras"]
        mb = cfg.get("master_bev") or cfg.get("bev") or {}

        self.range_m = float(mb.get("range_m", 6.0))
        self.width_m = float(mb.get("width_m", 3.0))
        self.cell_m = float(mb.get("cell_m", 0.05))
        self.min_range_m = float(mb.get("min_range_m", 0.55))
        self.max_range_m = float(mb.get("max_range_m", 8.0))
        # Horizontal subsampling can be aggressive; vertical subsampling
        # cannot, because the far camera sees the floor at a grazing angle.
        legacy_stride = int(mb.get("stride", 4))
        self.stride_x = int(mb.get("stride_x", legacy_stride))
        self.stride_y_near = int(mb.get("stride_y_near", 2))
        self.stride_y_far = int(mb.get("stride_y_far", 1))
        self.stride_y_other = int(mb.get("stride_y_other", 2))

        # Range-weight schedules (metres).  Near: full below `full1`, fade to
        # zero at `zero`.  Far: ramp up from `ramp0`, full through `full1`,
        # fade to zero at `zero`.
        near_w = mb.get("near_range_m", [0.55, 1.20, 2.00])
        far_w = mb.get("far_range_m", [1.00, 2.00, 4.50, 7.00])
        self.near_range_m = tuple(float(v) for v in near_w)
        self.far_range_m = tuple(float(v) for v in far_w)

        safety = cfg.get("safety", {})
        self.max_frame_age_s = float(safety.get("max_frame_age_s", 0.3))
        self.plane_residual_m = float(safety.get("plane_residual_m", 0.015))
        self.plane_min_inliers = int(safety.get("plane_min_inliers", 100))
        self.plane_min_frac = float(safety.get("plane_min_frac", 0.15))
        self.noise_scale = float(safety.get("noise_scale", 0.01))

        self.occ_min_m = float(mb.get("occ_min_m", 0.03))
        self.occ_max_m = float(mb.get("occ_max_m", 0.30))
        self.hazard_drop_m = float(mb.get("hazard_drop_m", 0.05))

        # Evidence thresholds (weighted points per cell).
        self.occ_min_w = float(mb.get("occ_min_w", 0.75))
        self.occ_dom_ratio = float(mb.get("occ_dom_ratio", 0.35))
        self.haz_min_w = float(mb.get("haz_min_w", 1.0))
        # Sparse far returns can leave 1-2 empty cells between observed floor
        # cells.  Fill those interior gaps (never the outer boundary, never
        # across an obstacle) so the map is not speckled.
        self.gap_fill_iters = int(mb.get("gap_fill_iters", 2))

        robot = cfg.get("robot", {})
        self.robot_radius_m = float(robot.get("radius_m", 0.20))
        sm = robot.get("self_mask", {})
        self.self_mask = None
        if sm:
            x0, x1 = sm.get("x_range_m", [-0.30, 1.00])
            y0, y1 = sm.get("y_range_m", [-0.30, 0.60])
            self.self_mask = {
                "x0": float(x0), "x1": float(x1),
                "y0": float(y0), "y1": float(y1),
                "min_h": float(sm.get("min_height_m", 0.05)),
            }

        if mounts is None:
            mounts = {
                serial: Mount(
                    height_m=float(c.get("height_m", 0.0)),
                    pitch_deg=float(c.get("pitch_deg", 0.0)),
                    yaw_deg=float(c.get("yaw_deg", 0.0)),
                    x_m=float(c.get("x_m", 0.0)),
                    y_m=float(c.get("y_m", 0.0)),
                )
                for serial, c in cam_cfg.items()
            }
        self.mounts = mounts
        self.serial_role = {
            serial: c["role"] for serial, c in cam_cfg.items()
            if serial in mounts
        }
        self.role_serial = {role: s for s, role in self.serial_role.items()}

        if intrinsics is None:
            intrinsics = {serial: load_intrinsics(serial, config_dir)
                          for serial in mounts}
        self.intrinsics = {}
        for serial, intr in intrinsics.items():
            self.intrinsics[serial] = {
                "fx": float(intr["fx"]), "fy": float(intr["fy"]),
                "cx": float(intr["cx"]), "cy": float(intr["cy"]),
                "width": int(intr.get("width", 0)),
                "height": int(intr.get("height", 0)),
            }

        self._rot = {serial: mount_rotation(m.pitch_deg, m.yaw_deg)
                     for serial, m in mounts.items()}
        self._t = {serial: np.array(
            [m.x_m, m.y_m, m.height_m], dtype=np.float32)
            for serial, m in mounts.items()}

        self.rows = int(round(self.range_m / self.cell_m))
        self.cols = int(round(self.width_m / self.cell_m))
        self.x_centers = (self.range_m
                          - (np.arange(self.rows) + 0.5) * self.cell_m)
        self.y_centers = ((np.arange(self.cols) + 0.5) * self.cell_m
                          - self.width_m / 2.0)

        inflate_cells = int(round(self.robot_radius_m / self.cell_m))
        self._inflate_kernel = (
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (2 * inflate_cells + 1, 2 * inflate_cells + 1))
            if inflate_cells > 0 else None)

        self._seed = seed
        self._rng = np.random.default_rng(seed)

    # -- per-camera ---------------------------------------------------------

    def prepare(self, frame) -> CameraPoints:
        """StampedFrame -> body-frame points (no floor decision)."""
        intr = self.intrinsics.get(frame.serial)
        if intr is None:
            raise RuntimeError(
                f"no intrinsics for mounted serial {frame.serial!r}")
        fx, fy = intr["fx"], intr["fy"]
        cx, cy = intr["cx"], intr["cy"]
        iw, ih = intr["width"], intr["height"]
        fw = int(frame.width or frame.depth.shape[1])
        fh = int(frame.height or frame.depth.shape[0])
        if iw and ih and (fw, fh) != (iw, ih):
            fx *= fw / iw
            fy *= fh / ih
            cx *= fw / iw
            cy *= fh / ih

        scale = frame.depth_scale_mm
        if scale is None:
            scale = 1.0
        depth_m = frame.depth.astype(np.float32) * (float(scale) * 1e-3)
        stride_y = {"near": self.stride_y_near,
                    "far": self.stride_y_far}.get(frame.role,
                                                  self.stride_y_other)
        pts_cam = deproject(depth_m, fx, fy, cx, cy,
                            stride_x=self.stride_x, stride_y=stride_y,
                            max_range_m=self.max_range_m)
        n_raw = int(len(pts_cam))
        if n_raw == 0:
            return CameraPoints(frame.role, frame.serial,
                                np.zeros((0, 3), np.float32),
                                np.zeros((0,), np.float32),
                                int(frame.stamp_us), 0, 0)

        ranges = np.linalg.norm(pts_cam, axis=1).astype(np.float32)
        rot = self._rot[frame.serial]
        trans = self._t[frame.serial]
        pts = (rot @ pts_cam.T).T + trans

        # The min-range is a BODY-frame dead disc (the chassis silhouette),
        # not a camera range: a floor point 0.2 m in front is ~1.3 m from
        # the near camera and must still be dropped.
        body_r2 = pts[:, 0] ** 2 + pts[:, 1] ** 2
        keep = body_r2 >= self.min_range_m ** 2

        # Keep a sensible working volume: the grid plus a margin for the
        # joint floor fit.  Distant walls cannot add floor evidence and
        # would only bias the RANSAC sample.
        keep &= ((pts[:, 0] > -0.50)
                 & (pts[:, 0] < max(self.range_m + 2.0, self.max_range_m))
                 & (np.abs(pts[:, 1]) < self.width_m / 2.0 + 1.5)
                 & (pts[:, 2] > -1.0)
                 & (pts[:, 2] < 2.5))

        if self.self_mask is not None:
            sm = self.self_mask
            on_body = ((pts[:, 0] > sm["x0"]) & (pts[:, 0] < sm["x1"])
                       & (pts[:, 1] > sm["y0"]) & (pts[:, 1] < sm["y1"])
                       & (pts[:, 2] > sm["min_h"]))
            keep &= ~on_body
        pts = pts[keep]
        ranges = ranges[keep]
        return CameraPoints(frame.role, frame.serial,
                            pts.astype(np.float32),
                            ranges.astype(np.float32),
                            int(frame.stamp_us), n_raw, int(len(pts)))

    # -- joint floor --------------------------------------------------------

    def _fit_joint_plane(self, per_camera):
        """One floor plane over every camera's points, with per-camera checks."""
        clouds, ranges = [], []
        for cp in per_camera.values():
            if cp is None or len(cp.points) == 0:
                continue
            n = len(cp.points)
            take = min(n, 6000)
            idx = (np.linspace(0, n - 1, take).astype(np.int64)
                   if n > take else np.arange(n))
            clouds.append(cp.points[idx])
            ranges.append(cp.ranges[idx])
        if not clouds:
            return {"ok": False, "n": [0.0, 0.0, 1.0], "d": 0.0,
                    "inliers": 0, "total": 0, "residual_m": None,
                    "per_camera": {}}
        pts = np.concatenate(clouds)
        rngs = np.concatenate(ranges)
        fit = fit_ground_plane(
            pts, residual_m=self.plane_residual_m,
            min_inliers=self.plane_min_inliers,
            min_frac=self.plane_min_frac, rng=self._rng,
            d_range=(-0.20, 0.20), range_scale=self.noise_scale,
            ranges=rngs)
        if fit is None:
            plane = {"ok": False, "n": [0.0, 0.0, 1.0], "d": 0.0,
                     "inliers": 0, "total": int(len(pts)),
                     "residual_m": None, "per_camera": {}}
        else:
            n, d, cnt = fit
            plane = {"ok": True,
                     "n": [float(v) for v in n],
                     "d": float(d), "inliers": int(cnt),
                     "total": int(len(pts)),
                     "residual_m": None, "per_camera": {}}
        nvec = np.asarray(plane["n"], dtype=np.float64)
        for role, cp in per_camera.items():
            if cp is None or len(cp.points) == 0:
                plane["per_camera"][role] = {
                    "ok": False, "inlier_frac": 0.0, "residual_m": None}
                continue
            h = _height(cp.points, nvec, plane["d"])
            res = np.abs(h)
            gate = np.maximum(self.plane_residual_m,
                              self.noise_scale * cp.ranges)
            inl = res < gate
            frac = float(inl.mean()) if res.size else 0.0
            plane["per_camera"][role] = {
                "ok": bool(frac >= self.plane_min_frac),
                "inlier_frac": round(frac, 3),
                "residual_m": round(
                    float(res[inl].mean()) if inl.any() else float(res.mean()),
                    4),
            }
        if plane["ok"]:
            h = _height(pts, nvec, plane["d"])
            gate = np.maximum(self.plane_residual_m,
                              self.noise_scale * rngs)
            inl = np.abs(h) < gate
            plane["residual_m"] = round(
                float(np.abs(h[inl]).mean()) if inl.any() else 0.0, 4)
        return plane

    # -- per-camera evidence ------------------------------------------------

    def _range_weight(self, role, forward_m):
        """Per-point confidence from the **forward** distance.

        Forward distance, not Euclidean camera range: the near camera is
        still accurate a metre ahead even though its mount height makes the
        Euclidean range larger.  The schedules are matched to where each
        camera actually sees the floor well.
        """
        forward_m = np.asarray(forward_m, dtype=np.float32)
        if role == "near" and len(self.near_range_m) == 3:
            _full0, full1, zero = self.near_range_m
            w = np.ones_like(forward_m)
            fade = forward_m > full1
            w[fade] = np.clip(
                (zero - forward_m[fade]) / max(zero - full1, 1e-6),
                0.0, 1.0)
            return w
        if role == "far" and len(self.far_range_m) == 4:
            ramp0, full0, full1, zero = self.far_range_m
            w = np.clip((forward_m - ramp0) / max(full0 - ramp0, 1e-6),
                        0.0, 1.0)
            w *= np.clip((zero - forward_m) / max(zero - full1, 1e-6),
                         0.0, 1.0)
            return w.astype(np.float32)
        return np.ones_like(forward_m)

    def _evidence(self, cp: CameraPoints, plane):
        """Weighted occupied/free/hazard/coverage evidence for one camera."""
        n_cells = self.rows * self.cols
        out = {
            "occ_w": np.zeros(n_cells, np.float32),
            "free_w": np.zeros(n_cells, np.float32),
            "haz_w": np.zeros(n_cells, np.float32),
            "cov_w": np.zeros(n_cells, np.float32),
            "occ_n": np.zeros(n_cells, np.int32),
            "haz_n": np.zeros(n_cells, np.int32),
            "cov_n": np.zeros(n_cells, np.int32),
            "cells": 0,
        }
        if cp is None or len(cp.points) == 0:
            return out
        nvec = np.asarray(plane["n"], dtype=np.float64)
        h = _height(cp.points, nvec, plane["d"])
        x = cp.points[:, 0]
        y = cp.points[:, 1]
        rows = ((self.range_m - x) / self.cell_m).astype(np.int32)
        cols = ((y + self.width_m / 2.0) / self.cell_m).astype(np.int32)
        inb = ((rows >= 0) & (rows < self.rows)
               & (cols >= 0) & (cols < self.cols))
        if not inb.any():
            return out
        rows = rows[inb]
        cols = cols[inb]
        h = h[inb]
        rng = cp.ranges[inb]
        forward = x[inb]
        flat = rows * self.cols + cols
        w = self._range_weight(cp.role, forward)

        min_h = np.maximum(self.occ_min_m, self.noise_scale * rng)
        drop_gate = np.maximum(self.hazard_drop_m, self.noise_scale * rng)
        occupied = (h >= min_h) & (h <= self.occ_max_m)
        floor = (h >= -0.02) & (h < min_h)
        below = h < -drop_gate

        occ_w = np.bincount(flat[occupied], weights=w[occupied],
                            minlength=n_cells).astype(np.float32)
        free_w = np.bincount(flat[floor], weights=w[floor],
                             minlength=n_cells).astype(np.float32)
        haz_w = np.bincount(flat[below], weights=w[below],
                            minlength=n_cells).astype(np.float32)
        known = floor | occupied | below
        cov_w = np.bincount(flat[known], weights=w[known],
                            minlength=n_cells).astype(np.float32)
        cov_n = np.bincount(flat[known], minlength=n_cells).astype(np.int32)
        occ_n = np.bincount(flat[occupied], minlength=n_cells).astype(np.int32)
        haz_n = np.bincount(flat[below], minlength=n_cells).astype(np.int32)

        # Negative obstacles from a local ground drop: a cell whose floor is
        # more than hazard_drop_m below an adjacent observed floor cell.
        ground_n = np.bincount(flat[floor], minlength=n_cells).astype(np.int32)
        ground_sum = np.bincount(flat[floor], weights=h[floor],
                                 minlength=n_cells).astype(np.float32)
        ground = np.where(ground_n > 0,
                          ground_sum / np.maximum(ground_n, 1),
                          np.nan).reshape(self.rows, self.cols)
        ground_n2 = ground_n.reshape(self.rows, self.cols)
        drop_vote = np.zeros((self.rows, self.cols), np.float32)
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            shifted = np.full_like(ground, np.nan)
            shifted_n = np.zeros_like(ground_n2)
            ys = slice(max(dy, 0), self.rows + min(dy, 0))
            xs = slice(max(dx, 0), self.cols + min(dx, 0))
            ys2 = slice(max(-dy, 0), self.rows + min(-dy, 0))
            xs2 = slice(max(-dx, 0), self.cols + min(-dx, 0))
            shifted[ys2, xs2] = ground[ys, xs]
            shifted_n[ys2, xs2] = ground_n2[ys, xs]
            with np.errstate(invalid="ignore"):
                drop = ground - shifted
            local = ((ground_n2 > 0) & (shifted_n > 0)
                     & (drop > self.hazard_drop_m))
            vote = np.where(local,
                            np.minimum(ground_n2, 2).astype(np.float32) * 0.5,
                            0.0)
            drop_vote = np.maximum(drop_vote, vote.astype(np.float32))
        haz_w += drop_vote.reshape(-1)
        haz_n += (drop_vote.reshape(-1) > 0).astype(np.int32)

        out.update({
            "occ_w": occ_w,
            "free_w": free_w,
            "haz_w": haz_w,
            "cov_w": cov_w,
            "occ_n": occ_n,
            "haz_n": haz_n,
            "cov_n": cov_n,
            "cells": int(np.sum(cov_n > 0)),
        })
        return out

    # -- fusion -------------------------------------------------------------

    def fuse(self, per_camera, ages_s=None):
        """Fuse prepared camera point clouds into one BevGrid."""
        available = {role: cp for role, cp in per_camera.items()
                     if cp is not None and len(cp.points) > 0}
        if not available:
            return None
        ages_s = dict(ages_s or {})
        plane = self._fit_joint_plane(available)
        n_cells = self.rows * self.cols

        occ_w = np.zeros(n_cells, np.float32)
        free_w = np.zeros(n_cells, np.float32)
        haz_w = np.zeros(n_cells, np.float32)
        cov_w = np.zeros(n_cells, np.float32)
        cov_n = np.zeros(n_cells, np.int32)
        contributions = {}
        camera_cov = {}
        for role, cp in available.items():
            ev = self._evidence(cp, plane)
            occ_w += ev["occ_w"]
            free_w += ev["free_w"]
            haz_w += ev["haz_w"]
            cov_w += ev["cov_w"]
            cov_n += ev["cov_n"]
            camera_cov[role] = ev["cov_w"]
            contributions[role] = {
                "raw_points": cp.n_raw,
                "points": cp.n_used,
                "cells": ev["cells"],
                "occupied_cells": int(np.sum(ev["occ_w"] > 0)),
                "free_cells": int(np.sum(ev["free_w"] > 0)),
                "hazard_cells": int(np.sum(ev["haz_w"] > 0)),
                "coverage_weight": round(float(ev["cov_w"].sum()), 1),
                "age_s": (round(float(ages_s[role]), 3)
                          if ages_s.get(role) is not None else None),
                "plane_ok": bool(plane["per_camera"].get(role, {}).get("ok")),
            }

        # Coverage is "at least one valid return landed here".  The weighted
        # evidence decides the class, but a single far point is still an
        # observation; confidence (cov_w) says how strong it was.
        coverage = cov_n >= 1
        occupied = ((occ_w >= self.occ_min_w)
                    & (occ_w >= self.occ_dom_ratio * free_w))
        hazard = ((haz_w >= self.haz_min_w)
                  & (haz_w >= occ_w))
        free = coverage & ~occupied & ~hazard

        # Fill only interior free gaps: a cell needs observed free space on
        # both sides and must not be occupied/hazard.  This keeps sparse far
        # returns from dappling the floor while never growing the outer
        # boundary or bridging an obstacle.
        free2 = free.reshape(self.rows, self.cols)
        blocked2 = (occupied | hazard).reshape(self.rows, self.cols)
        for _ in range(max(0, self.gap_fill_iters)):
            fill = np.zeros_like(free2)
            fill[1:-1, :] = (free2[:-2, :] & free2[2:, :]
                             & ~blocked2[1:-1, :])
            fill[:, 1:-1] |= (free2[:, :-2] & free2[:, 2:]
                              & ~blocked2[:, 1:-1])
            fill &= ~free2 & ~blocked2
            if not fill.any():
                break
            free2 |= fill
        free = free2.reshape(-1)
        coverage = coverage | free

        raw = np.zeros(n_cells, np.uint8)
        raw[free] = FREE
        raw[occupied] = OCCUPIED
        raw[hazard] = HAZARD
        coverage = coverage | occupied | hazard

        # Per-cell source attribution: which camera carried the evidence.
        near_cov = camera_cov.get("near", np.zeros(n_cells, np.float32))
        far_cov = camera_cov.get("far", np.zeros(n_cells, np.float32))
        source = np.zeros(n_cells, np.uint8)
        both = (near_cov >= 0.5) & (far_cov >= 0.5)
        near_only = (near_cov >= 0.5) & (far_cov < 0.5)
        far_only = (far_cov >= 0.5) & (near_cov < 0.5)
        source[near_only] = SRC_NEAR
        source[far_only] = SRC_FAR
        source[both] = SRC_BOTH

        confidence = np.clip(cov_w / 2.0, 0.0, 1.0) * 255.0
        confidence = confidence.astype(np.uint8)

        raw = raw.reshape(self.rows, self.cols)
        occ = raw.copy()
        if self._inflate_kernel is not None:
            blocked = ((occ == OCCUPIED) | (occ == HAZARD)).astype(np.uint8)
            dil = cv2.dilate(blocked, self._inflate_kernel)
            occ[(dil > 0) & (occ == FREE)] = INFLATED
        coverage = coverage.reshape(self.rows, self.cols)
        source = source.reshape(self.rows, self.cols)
        confidence = confidence.reshape(self.rows, self.cols)

        stamp = max(int(cp.stamp_us) for cp in available.values())
        plane_ok = {role: bool(plane["per_camera"].get(role, {}).get("ok"))
                    for role in available}
        return BevGrid(
            occ=occ, raw=raw, coverage=coverage, source=source,
            confidence=confidence, cell_m=self.cell_m,
            range_m=self.range_m, width_m=self.width_m,
            stamp_us=stamp, recv_ts=time.monotonic(),
            per_camera_age_s=ages_s, plane_ok=plane_ok, plane=plane,
            contributions=contributions, roles=sorted(available))


# ---------------------------------------------------------------------------
# Rendering (debug / operator view)
# ---------------------------------------------------------------------------

BEV_COLORS = {
    FREE: (70, 70, 70),          # BGR dark grey
    OCCUPIED: (0, 0, 220),       # red
    HAZARD: (0, 140, 255),       # orange
    INFLATED: (60, 60, 120),     # dark red / planning margin
}
UNKNOWN_COLOR = (28, 28, 28)
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
        for role in grid.roles:
            c = grid.contributions.get(role, {})
            age = c.get("age_s")
            age_s = f"{age:.2f}s" if age is not None else "?"
            lines.append(
                f"{role}: {c.get('points', 0)}pts {c.get('cells', 0)}cells "
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
