"""Bounded depth-geometry reconstruction for the app's Map page.

This is deliberately **not** an occupancy grid and **not** a loop-corrected
map.  Each accepted live RGB-D tick contributes a sparse cloud, unprojected
from the already color-aligned depth with calibrated color rays, transformed
by the current *continuous odometry* pose, and fused into a small voxel set.
The result is an experimental odometry reconstruction that drifts with the
tracker.

Invariants:
  * camera optical frame only: world_from_camera_optical;
  * at most ``max_points`` voxels (LRU eviction), range <= ``max_range_m``;
  * sampling is rate-limited by LiveSlam (<= ``sample_hz``);
  * geometry is cleared whenever the tracker starts a new segment;
  * corrected/map poses are never used here.
"""

from __future__ import annotations

import collections
import json
import math
import threading
import time
from dataclasses import dataclass

import numpy as np

from .calibration import FRAME_ID, POSE_FRAME, CameraModel, RigCalibration


@dataclass
class DepthGeometryConfig:
    max_points: int = 100_000
    # Internal storage can be much larger than what the app can draw.  The
    # transmitted cloud is deterministically decimated at rebuild time.
    max_transmit_points: int = 30_000
    voxel_size_m: float = 0.05
    max_range_m: float = 6.0
    min_range_m: float = 0.15
    sample_hz: float = 1.0
    pixel_stride: int = 6

    def validate(self) -> None:
        if int(self.max_points) < 1:
            raise ValueError("max_points must be >= 1")
        if int(self.max_transmit_points) < 1:
            raise ValueError("max_transmit_points must be >= 1")
        for name in ("voxel_size_m", "max_range_m", "min_range_m",
                     "sample_hz"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if self.min_range_m >= self.max_range_m:
            raise ValueError("min_range_m must be < max_range_m")
        if int(self.pixel_stride) < 1:
            raise ValueError("pixel_stride must be >= 1")


def undistort_pixels(xs, ys, model: CameraModel,
                     iterations: int = 20, tol: float = 1e-9):
    """Invert the rational color distortion: pixel -> normalized ray.

    Returns ``(xn, yn)`` such that projecting them with ``model`` yields
    ``(xs, ys)``.  Fixed-point iteration is sufficient for the calibrated
    coefficients (all small); the map applies this once per sampled pixel.
    """
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    xd = (xs - model.cx) / model.fx
    yd = (ys - model.cy) / model.fy
    x = xd.copy()
    y = yd.copy()
    k1, k2, p1, p2, k3, k4, k5, k6 = model.distortion
    for _ in range(max(1, int(iterations))):
        r2 = x * x + y * y
        num = 1.0 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
        den = 1.0 + k4 * r2 + k5 * r2 ** 2 + k6 * r2 ** 3
        radial = num / den
        tx = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        ty = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
        x_new = (xd - tx) / radial
        y_new = (yd - ty) / radial
        if np.all(np.abs(x_new - x) < tol) and \
                np.all(np.abs(y_new - y) < tol):
            x, y = x_new, y_new
            break
        x, y = x_new, y_new
    bad = ~(np.isfinite(x) & np.isfinite(y))
    if np.any(bad):
        raise ValueError(
            "undistort_pixels produced non-finite rays; refusing a pinhole "
            "fallback that would ignore the calibrated distortion")
    return x, y


def _quat_matrix(rotation) -> np.ndarray | None:
    try:
        x, y, z, w = (float(v) for v in rotation)
    except (TypeError, ValueError):
        return None
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if not math.isfinite(n) or n <= 1e-9:
        return None
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _pose_arrays(pose):
    if isinstance(pose, dict):
        translation = pose.get("translation")
        rotation = pose.get("rotation")
    else:
        translation = getattr(pose, "translation", None)
        rotation = getattr(pose, "rotation", None)
    try:
        t = np.asarray(translation, dtype=np.float64).reshape(3)
    except (TypeError, ValueError):
        return None
    R = _quat_matrix(rotation)
    if R is None or not np.all(np.isfinite(t)):
        return None
    return R, t


class DepthGeometryMap:
    """Thread-safe bounded voxel map of uncorrected odometry geometry."""

    def __init__(self, calibration: RigCalibration,
                 config: DepthGeometryConfig | None = None,
                 segment: int = 0):
        self.calibration = calibration
        self.config = config or DepthGeometryConfig()
        self.config.validate()
        self._lock = threading.RLock()
        self._voxels: collections.OrderedDict = collections.OrderedDict()
        self.segment = int(segment)
        self.revision = 0
        self.updated_mono: float | None = None
        self._cached_revision = -1
        self._cached_base: dict | None = None
        self._cached_head_bytes = b""
        self._cached_points_bytes = b""
        self._cached_points_bytes_revision = -1
        self._last_rebuild_mono: float | None = None

        stride = int(self.config.pixel_stride)
        self._xs = np.arange(0, calibration.color.width, stride,
                             dtype=np.int64)
        self._ys = np.arange(0, calibration.color.height, stride,
                             dtype=np.int64)
        grid_x, grid_y = np.meshgrid(self._xs, self._ys)
        xn, yn = undistort_pixels(grid_x, grid_y, calibration.color)
        self._ray_x = xn.astype(np.float32)
        self._ray_y = yn.astype(np.float32)

    # -- updates ----------------------------------------------------------

    def reset(self, segment: int | None = None) -> None:
        """Drop all geometry (new tracker segment)."""
        with self._lock:
            self._reset_locked(self.segment if segment is None else segment)

    def _reset_locked(self, segment) -> None:
        self._voxels.clear()
        self.segment = int(segment)
        self.revision += 1
        self.updated_mono = None
        # Lazy: geometry is rebuilt only when a /map reader asks for it.
        self._cached_revision = -1

    @staticmethod
    def _serialize_points(points, chunk: int = 4000) -> bytes:
        """JSON-encode the points array in bounded GIL slices.

        A 100k-point array is ~3.3 MB; one json.dumps holds the GIL long
        enough to disturb capture on Thor.  Encode fixed-size chunks and
        yield between them so the capture thread gets scheduled.
        """
        if not points:
            return b"[]"
        parts = [b"["]
        for start in range(0, len(points), chunk):
            if start:
                parts.append(b",")
            parts.append(json.dumps(
                points[start:start + chunk], separators=(",", ":"),
                allow_nan=False).encode()[1:-1])
            time.sleep(0)
        parts.append(b"]")
        return b"".join(parts)

    def _rebuild_cache_locked(self) -> None:
        """Materialize points once per revision with numpy (lazy).

        Runs only from a /map read (never from add/reset), outside LiveSlam's
        status lock.  The point list and its JSON bytes are immutable and
        shared by reference until the next revision.
        """
        rows = list(self._voxels.values())
        points = []
        map_points = 0
        transmit_stride = 1
        if rows:
            values = np.asarray(rows, dtype=np.float64)
            counts = values[:, 6]
            valid = counts > 0
            values = values[valid]
            counts = counts[valid]
            if len(values):
                means = values[:, :6] / counts[:, None]
                finite = np.isfinite(means).all(axis=1)
                means = means[finite]
                if len(means):
                    map_points = int(len(means))
                    max_tx = int(self.config.max_transmit_points)
                    transmit_stride = max(
                        1, math.ceil(map_points / max_tx))
                    keep = np.arange(0, map_points, transmit_stride)
                    selected = means[keep]
                    xyz = np.round(selected[:, :3], 4).tolist()
                    rgb = np.clip(np.rint(selected[:, 3:6]), 0,
                                  255).astype(np.int64).tolist()
                    # Chunk/yield the Python materialization; do not hold
                    # the GIL for the whole 30k-point list.
                    chunk = 4096
                    for start in range(0, len(xyz), chunk):
                        end = min(start + chunk, len(xyz))
                        points.extend(xyz[i] + rgb[i]
                                      for i in range(start, end))
                        time.sleep(0)
        self._cached_base = {
            "revision": int(self.revision),
            "frame_id": FRAME_ID,
            "pose_frame": POSE_FRAME,
            "geometry_source": "depth_odometry",
            "corrected": False,
            "segment": int(self.segment),
            "points": points,
            "count": len(points),
            "capacity": int(self.config.max_points),
            "voxel_size_m": float(self.config.voxel_size_m),
            "map_points": map_points,
            "transmit_stride": int(transmit_stride),
        }
        head = {
            "revision": int(self.revision),
            "frame_id": FRAME_ID,
            "pose_frame": POSE_FRAME,
            "geometry_source": "depth_odometry",
            "corrected": False,
            "segment": int(self.segment),
        }
        self._cached_head_bytes = json.dumps(
            head, separators=(",", ":"), allow_nan=False).encode()[1:-1]
        # Points bytes are serialized outside this lock (see snapshot_bundle).
        self._cached_points_bytes = b""
        self._cached_points_bytes_revision = -1
        self._cached_revision = int(self.revision)
        self._last_rebuild_mono = time.monotonic()

    def _ensure_cache_locked(self, min_interval_s: float = 0.0) -> None:
        if self._cached_base is not None \
                and self._cached_revision == self.revision:
            return
        # Coalesce same-segment revisions that arrive faster than the small
        # minimum interval; a segment change always rebuilds immediately.
        if (self._cached_base is not None and min_interval_s > 0
                and self._cached_base["segment"] == self.segment
                and self._last_rebuild_mono is not None
                and time.monotonic() - self._last_rebuild_mono
                < min_interval_s):
            return
        self._rebuild_cache_locked()

    def snapshot_bundle(self, min_interval_s: float = 0.0):
        """(snapshot dict, head bytes, points bytes).

        The geometry lock is held only to materialize/copy references.  Point
        serialization runs outside it; ``add()`` is never blocked by it.  The
        bytes are published only if the revision is still current.
        """
        with self._lock:
            self._ensure_cache_locked(min_interval_s)
            out = dict(self._cached_base)
            out["updated_age_s"] = (
                time.monotonic() - self.updated_mono
                if self.updated_mono is not None else None)
            head = self._cached_head_bytes
            revision = self._cached_revision
            if self._cached_points_bytes_revision == revision:
                return out, head, self._cached_points_bytes
            points = self._cached_base["points"]
        points_bytes = self._serialize_points(points)
        with self._lock:
            if (self.revision == revision
                    and self._cached_revision == revision):
                self._cached_points_bytes = points_bytes
                self._cached_points_bytes_revision = revision
        return out, head, points_bytes

    def points_bytes(self, min_interval_s: float = 0.0) -> bytes:
        """Pre-serialized immutable points array for this revision."""
        return self.snapshot_bundle(min_interval_s)[2]

    def geometry_head_bytes(self, min_interval_s: float = 0.0) -> bytes:
        """Pre-serialized scalar geometry fields (no braces) for this revision."""
        return self.snapshot_bundle(min_interval_s)[1]

    def add(self, depth_mm: np.ndarray, rgb: np.ndarray | None,
            pose, segment: int) -> bool:
        """Fuse one aligned depth frame with its continuous odometry pose.

        ``depth_mm`` is the color-aligned uint16 millimetre depth; ``rgb`` is
        optional HxWx3 RGB.  Returns True when the map changed.
        """
        with self._lock:
            segment = int(segment)
            if segment != self.segment:
                self._reset_locked(segment)

            depth = np.asarray(depth_mm)
            expected = self.calibration.color_shape
            if depth.ndim != 2 or tuple(depth.shape) != expected:
                raise ValueError(
                    f"map depth shape {tuple(depth.shape)} != {expected}")
            if rgb is not None:
                rgb_arr = np.asarray(rgb)
                if rgb_arr.shape != (expected[0], expected[1], 3):
                    raise ValueError(
                        f"map rgb shape {tuple(rgb_arr.shape)} != "
                        f"{(expected[0], expected[1], 3)}")
            else:
                rgb_arr = None

            z = depth[np.ix_(self._ys, self._xs)].astype(np.float32) * 0.001
            valid = (np.isfinite(z)
                     & (z >= float(self.config.min_range_m))
                     & (z <= float(self.config.max_range_m)))
            if not valid.any():
                return False
            rx = self._ray_x[valid] * z[valid]
            ry = self._ray_y[valid] * z[valid]
            rz = z[valid]
            cam = np.stack([rx, ry, rz], axis=1)
            in_range = (np.sum(cam * cam, axis=1)
                        <= float(self.config.max_range_m) ** 2)
            if not in_range.any():
                return False
            cam = cam[in_range]

            pose_rt = _pose_arrays(pose)
            if pose_rt is None:
                return False
            R, t = pose_rt
            world = cam @ R.T + t

            if rgb_arr is None:
                colors = np.zeros((len(world), 3), np.float64)
            else:
                sampled = rgb_arr[np.ix_(self._ys, self._xs)][valid][in_range]
                colors = sampled.astype(np.float64)

            keys = np.floor(world / float(self.config.voxel_size_m)).astype(
                np.int64)
            uniq, inv = np.unique(keys, axis=0, return_inverse=True)
            if len(uniq) == 0:
                return False
            obs = np.concatenate([world, colors], axis=1)
            sums = np.zeros((len(uniq), 6), np.float64)
            np.add.at(sums, inv, obs)
            counts = np.bincount(inv, minlength=len(uniq))

            cap = int(self.config.max_points)
            for i, key in enumerate(uniq):
                k = (int(key[0]), int(key[1]), int(key[2]))
                cur = self._voxels.get(k)
                if cur is None:
                    while len(self._voxels) >= cap:
                        self._voxels.popitem(last=False)
                    self._voxels[k] = [*sums[i].tolist(), int(counts[i])]
                else:
                    cur[0] += sums[i, 0]
                    cur[1] += sums[i, 1]
                    cur[2] += sums[i, 2]
                    cur[3] += sums[i, 3]
                    cur[4] += sums[i, 4]
                    cur[5] += sums[i, 5]
                    cur[6] += int(counts[i])
                    self._voxels.move_to_end(k)
            self.revision += 1
            self.updated_mono = time.monotonic()
            return True

    # -- snapshot ---------------------------------------------------------

    def snapshot(self, min_interval_s: float = 0.0) -> dict:
        with self._lock:
            self._ensure_cache_locked(min_interval_s)
            out = dict(self._cached_base)
            out["updated_age_s"] = (
                time.monotonic() - self.updated_mono
                if self.updated_mono is not None else None)
            return out
