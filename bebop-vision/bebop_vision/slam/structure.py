"""Phase 1 structural map: estimated gravity, floor plane, carved 2D grid.

This is deliberately **per-session and odometry-based**, not a persistent map:

* gravity is **estimated from the floor**, never measured by an IMU: RANSAC
  seeds the dominant depth floor plane in the odometry world frame, then up
  to ``floor_refine_iterations`` least-squares passes tighten it; the inlier
  RMSE is published;
* the floor normal is constrained to the camera -Y (up) prior within 35°;
* the 2D grid is locked to the first valid gravity estimate and uses 5 cm
  cells; free space is carved from the camera to **every** depth hit,
  including floor hits, so walkable floor is marked free;
* occupancy is the **obstacle** layer: every cell with a hit above
  ``occupied_min_height_m`` is occupied and is rendered, including compact
  furniture.  A cell is a *wall candidate* only if its accumulated height
  span is at least ``wall_min_extent_m`` and its top clears a modest
  ``wall_min_top_m`` (~0.8 m, just above desk height); that height gate only
  removes low clutter (a tilted floor band has almost no extent) while the
  component shape and free-boundary tests below decide wall vs furniture;
* wall candidates are grouped into connected components and a component is a
  **wall** only when it is elongated and thin (aspect ratio and a robust
  thickness estimate).  Compact components (desk, chair, box, person,
  monitor) are obstacles and never produce wall cells or segments.  A wall
  must also bound observed space: carved free cells on one side and
  not-free on the other, unless ``wall_boundary_check`` is disabled.  A wall
  seen from both sides legitimately has free space on both sides and is
  rejected by this test -- a known limitation until a real thinness/medial
  axis model replaces it;
* wall line segments are greedily grown **only from accepted wall cells**,
  collinear neighbours are merged, and segments shorter than
  ``min_wall_segment_m`` are dropped;
* there is no loop closure, relocalization, or persistence across sessions, so
  the map drifts with odometry and is discarded when the process exits or the
  tracker starts a new segment.

Frames and conventions
----------------------
``frame_id`` is ``"map"`` only once gravity/floor are valid.  That frame is
gravity-aligned with **z = floor plane** (up is +z); its x axis is the camera's
horizontal forward direction at the moment the first floor estimate locked,
and y completes a right-handed frame.  ``yaw_deg`` is the heading of the
camera-forward projection about +z: 0° when facing the initial x axis and
**positive counter-clockwise seen from above** (a left turn is positive).

``gravity.up`` and ``floor.normal`` are the estimated up/floor normal expressed
in the **camera optical frame** (that is where roll/pitch are meaningful); when
no estimate is available they are ``null`` and ``valid`` is ``false``.
``floor.offset_m`` and ``floor.camera_height_m`` are the perpendicular distance
from the camera origin to the floor along that normal.

The raw display point cloud is a separate, opt-in path.  This module keeps only
a bounded floor-point buffer, a bounded LRU cell set, and a bounded path.
"""

from __future__ import annotations

import base64
import collections
import json
import math
import threading
import time
from dataclasses import dataclass

import numpy as np

from .calibration import FRAME_ID, POSE_FRAME, RigCalibration
from .map import _pose_arrays, undistort_pixels

STRUCTURE_FRAME_ID = "map"


@dataclass
class StructureConfig:
    resolution_m: float = 0.05
    # Free carving now uses every depth hit (including the floor), so a room
    # needs far more free cells than the old occupied-only carving.
    max_cells: int = 60_000
    max_range_m: float = 6.0
    min_range_m: float = 0.2
    pixel_stride: int = 12
    floor_buffer_points: int = 20_000
    floor_min_inliers: int = 300
    floor_min_area_m2: float = 0.8
    floor_inlier_dist_m: float = 0.02
    floor_prior_max_deg: float = 35.0
    ransac_iterations: int = 60
    # Least-squares refinement passes over the RANSAC inliers.  A tilted or
    # slightly-misestimated floor otherwise leaks into distant occupancy.
    floor_refine_iterations: int = 3
    # Hits below this height above the floor are never occupancy.  The old
    # 0.12 m let a 1-3 deg floor-plane error mark the floor at 5 m as a wall.
    occupied_min_height_m: float = 0.28
    occupied_max_height_m: float = 2.5
    # A cell is a wall *candidate* only if it saw real vertical structure, not
    # a thin (possibly tilted) floor band, and reaches above furniture height.
    # The height floor is deliberately modest (~0.8 m, just above a desk): it
    # only removes low clutter, while the component SHAPE and FREE-SPACE
    # BOUNDARY tests do the real wall-vs-furniture discrimination.  The old
    # 1.4 m floor rejected most real walls from a chest-height camera (live:
    # 1621 occupied cells -> 3 candidates -> 0 segments).
    wall_min_extent_m: float = 0.40
    wall_min_top_m: float = 0.8
    # Candidates are connected-componented; only elongated, thin components
    # are walls.  A desk/chair/box/person is compact and stays an obstacle.
    wall_min_aspect_ratio: float = 3.0
    wall_max_thickness_m: float = 0.5
    # A thin boundary can be roughly square (a room, or an L/T wall network),
    # so PCA extent alone would reject it.  Accept a thin component whose two
    # principal extents both reach this size; a small compact loop (box, bin)
    # stays an obstacle.
    wall_min_loop_extent_m: float = 1.5
    # A wall bounds observed space: free carved cells on one side, not-free on
    # the other.  An obstacle in open floor has free space on both sides.
    # (A wall observed from both sides also has free on both sides and is
    # rejected; disable this check only if that trade-off is unacceptable.)
    wall_boundary_check: bool = True
    wall_boundary_probe_cells: int = 3
    wall_boundary_min_fraction: float = 0.25
    sample_hz: float = 1.0
    path_capacity: int = 300
    max_rays_per_update: int = 1200
    # Short clutter runs must disappear; real wall runs are metres long.
    min_wall_segment_m: float = 0.8
    max_wall_segments: int = 500
    max_wall_cells_transmit: int = 2000
    cache_min_interval_s: float = 0.25
    seed: int = 0

    def validate(self) -> None:
        for name in ("resolution_m", "max_range_m", "min_range_m",
                     "floor_inlier_dist_m", "floor_prior_max_deg",
                     "occupied_min_height_m", "occupied_max_height_m",
                     "wall_min_extent_m", "wall_min_top_m",
                     "wall_min_aspect_ratio", "wall_max_thickness_m",
                     "wall_min_loop_extent_m", "sample_hz"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        boundary_fraction = float(self.wall_boundary_min_fraction)
        if not math.isfinite(boundary_fraction) \
                or boundary_fraction < 0.0 or boundary_fraction > 1.0:
            raise ValueError(
                "wall_boundary_min_fraction must be finite and in [0, 1]")
        if int(self.wall_boundary_probe_cells) < 1:
            raise ValueError("wall_boundary_probe_cells must be >= 1")
        if self.min_range_m >= self.max_range_m:
            raise ValueError("min_range_m must be < max_range_m")
        if int(self.max_cells) < 1:
            raise ValueError("max_cells must be >= 1")
        # RLE runs are uint16, and every run is a subset of the bounded cell
        # set, so keeping max_cells below the uint16 limit prevents overflow.
        if int(self.max_cells) > 65_535:
            raise ValueError("max_cells must be <= 65535 (uint16 RLE)")
        if int(self.pixel_stride) < 1:
            raise ValueError("pixel_stride must be >= 1")
        if int(self.floor_buffer_points) < 3:
            raise ValueError("floor_buffer_points must be >= 3")
        if int(self.floor_min_inliers) < 3:
            raise ValueError("floor_min_inliers must be >= 3")
        if int(self.path_capacity) < 1:
            raise ValueError("path_capacity must be >= 1")
        if float(self.min_wall_segment_m) <= 0:
            raise ValueError("min_wall_segment_m must be positive")
        if int(self.max_wall_segments) < 0:
            raise ValueError("max_wall_segments must be >= 0")
        if int(self.max_wall_cells_transmit) < 0:
            raise ValueError("max_wall_cells_transmit must be >= 0")
        if int(self.max_rays_per_update) < 0:
            raise ValueError("max_rays_per_update must be >= 0")
        if int(self.floor_refine_iterations) < 0:
            raise ValueError("floor_refine_iterations must be >= 0")
        if float(self.floor_min_area_m2) < 0:
            raise ValueError("floor_min_area_m2 must be >= 0")
        cache_interval = float(self.cache_min_interval_s)
        if not math.isfinite(cache_interval) or cache_interval < 0:
            raise ValueError(
                "cache_min_interval_s must be finite and >= 0")


def _unit(v) -> np.ndarray | None:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(v))
    if not math.isfinite(n) or n <= 1e-9:
        return None
    return v / n


def _bresenham(x0: int, y0: int, x1: int, y1: int):
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    while (x0, y0) != (x1, y1):
        yield x0, y0
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy


class StructureMap:
    """Bounded gravity-aligned occupancy grid from live depth + odometry."""

    def __init__(self, calibration: RigCalibration,
                 config: StructureConfig | None = None, segment: int = 0):
        self.calibration = calibration
        self.config = config or StructureConfig()
        self.config.validate()
        self._lock = threading.RLock()
        self.segment = int(segment)
        self.revision = 0
        self._floor_points: collections.deque = collections.deque()
        self._floor: dict | None = None
        self._up_world: np.ndarray | None = None
        self._origin_world: np.ndarray | None = None
        self._R_align: np.ndarray | None = None
        self._cells: collections.OrderedDict = collections.OrderedDict()
        self._path: collections.deque = collections.deque(
            maxlen=int(self.config.path_capacity))
        self._last_pose: dict | None = None
        self._rng = np.random.default_rng(int(self.config.seed))
        self._last_update_mono: float | None = None
        self._cached_revision = -1
        self._cached_body: dict | None = None
        self._cached_body_bytes = b""
        self._cached_fields_bytes = b""
        self._last_cache_mono: float | None = None

        stride = int(self.config.pixel_stride)
        self._xs = np.arange(0, calibration.color.width, stride,
                             dtype=np.int64)
        self._ys = np.arange(0, calibration.color.height, stride,
                             dtype=np.int64)
        grid_x, grid_y = np.meshgrid(self._xs, self._ys)
        xn, yn = undistort_pixels(grid_x, grid_y, calibration.color)
        self._ray_x = xn.astype(np.float32)
        self._ray_y = yn.astype(np.float32)

    # -- public updates ---------------------------------------------------

    def reset(self, segment: int | None = None) -> None:
        with self._lock:
            self._reset_locked(self.segment if segment is None else segment)

    def _reset_locked(self, segment) -> None:
        self.segment = int(segment)
        self._floor_points.clear()
        self._floor = None
        self._up_world = None
        self._origin_world = None
        self._R_align = None
        self._cells.clear()
        self._path.clear()
        self._last_pose = None
        self._last_update_mono = None
        self._rng = np.random.default_rng(int(self.config.seed))
        self.revision += 1
        self._cached_revision = -1

    def add(self, depth_mm: np.ndarray, pose, segment: int) -> bool:
        """Unproject one aligned depth frame and update floor/grid."""
        depth = np.asarray(depth_mm)
        expected = self.calibration.color_shape
        if depth.ndim != 2 or tuple(depth.shape) != expected:
            raise ValueError(
                f"structure depth shape {tuple(depth.shape)} != {expected}")
        z = depth[np.ix_(self._ys, self._xs)].astype(np.float32) * 0.001
        valid = (np.isfinite(z)
                 & (z >= float(self.config.min_range_m))
                 & (z <= float(self.config.max_range_m)))
        if not valid.any():
            return False
        cam = np.stack([self._ray_x[valid] * z[valid],
                        self._ray_y[valid] * z[valid],
                        z[valid]], axis=1)
        return self.add_points(cam, pose, segment)

    def add_points(self, points_camera: np.ndarray, pose,
                   segment: int) -> bool:
        """Synthetic/test entry point: camera-frame points + odometry pose."""
        pts = np.asarray(points_camera, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError("points_camera must be Nx3")
        pts = pts[np.isfinite(pts).all(axis=1)]
        if len(pts) == 0:
            return False
        pose_rt = _pose_arrays(pose)
        if pose_rt is None:
            return False
        R, t = pose_rt
        with self._lock:
            segment = int(segment)
            if segment != self.segment:
                self._reset_locked(segment)
            world = pts @ R.T + t
            if self._floor is not None:
                # Once a floor is locked, keep only near-floor points so walls
                # and furniture cannot outvote it in later refits.  The
                # threshold is loose enough to follow slow odometry drift.
                normal = self._floor["normal_world"]
                d = float(self._floor["d_world"])
                near = np.abs(world @ normal + d) <= max(
                    0.10, 3.0 * float(self.config.floor_inlier_dist_m))
                world = world[near]
            if len(world):
                self._floor_points.append(world)
            max_floor_points = int(self.config.floor_buffer_points)
            total_points = sum(len(a) for a in self._floor_points)
            while total_points > max_floor_points \
                    and len(self._floor_points) > 1:
                total_points -= len(self._floor_points.popleft())
            if total_points > max_floor_points and self._floor_points:
                self._floor_points[-1] = \
                    self._floor_points[-1][-max_floor_points:]
            est = self._fit_floor_locked(R, t)
            if est is not None:
                self._floor = est
                if self._R_align is None:
                    self._init_aligned_locked(est, R, t)
            if self._R_align is not None and self._floor is not None:
                self._update_grid_locked(pts, R, t, pose, segment)
            self._last_update_mono = time.monotonic()
            self.revision += 1
            self._cached_revision = -1
            return True

    # -- floor estimation -------------------------------------------------

    def _fit_floor_locked(self, R: np.ndarray, t: np.ndarray):
        if not self._floor_points:
            return None
        pts = np.concatenate(list(self._floor_points), axis=0)
        if len(pts) < int(self.config.floor_min_inliers):
            return None
        if len(pts) < 3:
            return None
        prior = _unit(R @ np.array([0.0, -1.0, 0.0]))
        if prior is None:
            return None
        cos_prior_limit = math.cos(math.radians(
            float(self.config.floor_prior_max_deg)))
        threshold = float(self.config.floor_inlier_dist_m)
        best = None
        n = len(pts)
        for _ in range(max(1, int(self.config.ransac_iterations))):
            idx = self._rng.choice(n, 3, replace=False)
            p0, p1, p2 = pts[idx]
            normal = _unit(np.cross(p1 - p0, p2 - p0))
            if normal is None:
                continue
            if float(np.dot(normal, prior)) < 0:
                normal = -normal
            if float(np.dot(normal, prior)) < cos_prior_limit:
                continue
            d = -float(np.dot(normal, p0))
            if not math.isfinite(d) or d <= 0:
                continue
            mask = np.abs(pts @ normal + d) <= threshold
            count = int(mask.sum())
            if count < int(self.config.floor_min_inliers):
                continue
            area = self._support_area(pts[mask], normal)
            if area < float(self.config.floor_min_area_m2):
                continue
            if best is None or count > best["inlier_count"]:
                best = {"normal": normal, "d": d, "mask": mask,
                        "inlier_count": count, "area": area}
        if best is None:
            return None
        # Least-squares refinement over the RANSAC inliers, repeated so the
        # plane tightens around the true floor instead of staying at the
        # 3-sample RANSAC estimate.  The residual is published so plane
        # accuracy is measurable (a few degrees at 5 m is 0.1-0.3 m of
        # apparent height and must not become a wall).
        normal = best["normal"]
        d = float(best["d"])
        mask = np.asarray(best["mask"], dtype=bool)
        refinements = 0
        for _ in range(int(self.config.floor_refine_iterations)):
            inliers = pts[mask]
            if len(inliers) < 3:
                break
            centroid = inliers.mean(axis=0)
            _, _, vt = np.linalg.svd(inliers - centroid, full_matrices=False)
            refined = _unit(vt[-1])
            if refined is None:
                break
            if float(np.dot(refined, prior)) < 0:
                refined = -refined
            refined_d = -float(np.dot(refined, centroid))
            if not math.isfinite(refined_d) or refined_d <= 0:
                break
            refined_mask = np.abs(pts @ refined + refined_d) <= threshold
            if int(refined_mask.sum()) < int(self.config.floor_min_inliers):
                break
            normal, d, mask = refined, refined_d, refined_mask
            refinements += 1
        count = int(mask.sum())
        if count < int(self.config.floor_min_inliers):
            return None
        area = self._support_area(pts[mask], normal)
        if area < float(self.config.floor_min_area_m2):
            return None
        residuals = np.abs(pts[mask] @ normal + d)
        rmse = float(np.sqrt(np.mean(residuals * residuals))) \
            if len(residuals) else 0.0
        height = float(np.dot(normal, t) + d)
        if not math.isfinite(height) or height <= 0:
            return None
        up_camera = _unit(R.T @ normal)
        if up_camera is None:
            return None
        return {
            "normal_world": normal,
            "d_world": d,
            "up_world": normal,
            "up_camera": up_camera,
            "camera_height_m": height,
            "inlier_count": count,
            "area_m2": area,
            "inlier_ratio": count / max(1, len(pts)),
            "rmse_m": rmse,
            "refinements": refinements,
        }

    @staticmethod
    def _support_area(points: np.ndarray, normal: np.ndarray) -> float:
        if len(points) < 3:
            return 0.0
        helper = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(helper, normal))) > 0.9:
            helper = np.array([0.0, 1.0, 0.0])
        u = _unit(np.cross(normal, helper))
        if u is None:
            return 0.0
        v = np.cross(normal, u)
        uv = points @ np.stack([u, v], axis=1)
        cells = np.unique(np.floor(uv / 0.10).astype(np.int64), axis=0)
        return float(len(cells)) * 0.01

    def _init_aligned_locked(self, est, R: np.ndarray, t: np.ndarray) -> None:
        up = _unit(est["up_world"])
        forward = _unit(R @ np.array([0.0, 0.0, 1.0]))
        if up is None:
            return
        if forward is not None:
            horizontal = forward - np.dot(forward, up) * up
            x_axis = _unit(horizontal)
        else:
            x_axis = None
        if x_axis is None:
            x_axis = _unit(np.cross(up, np.array([0.0, 0.0, 1.0])))
        if x_axis is None:
            return
        y_axis = _unit(np.cross(up, x_axis))
        if y_axis is None:
            return
        self._up_world = up
        self._R_align = np.stack([x_axis, y_axis, up])
        self._origin_world = t - up * float(est["camera_height_m"])
        cam = self._aligned_points(t.reshape(1, 3))[0]
        self._path.append([round(float(cam[0]), 3),
                           round(float(cam[1]), 3)])

    def _aligned_points(self, points_world: np.ndarray) -> np.ndarray:
        return (points_world - self._origin_world) @ self._R_align.T

    # -- occupancy / carving ----------------------------------------------

    def _evict_one_locked(self) -> None:
        """Evict the oldest free cell; protect cells with occupancy evidence.

        Floor carving to every depth hit creates far more free cells than
        walls.  Plain LRU eviction would push recently-observed wall cells out
        of a bounded map, so free-only cells are evicted first.
        """
        for key, cur in self._cells.items():
            if cur[1] == 0:
                del self._cells[key]
                return
        self._cells.popitem(last=False)

    def _cell(self, ix: int, iy: int):
        key = (int(ix), int(iy))
        cur = self._cells.get(key)
        if cur is None:
            while len(self._cells) >= int(self.config.max_cells):
                self._evict_one_locked()
            cur = [0, 0, 255, 0]  # free_count, occupied_count, h_min, h_max
            self._cells[key] = cur
        else:
            self._cells.move_to_end(key)
        return cur

    def _mark_free(self, ix: int, iy: int) -> None:
        self._cell(ix, iy)[0] += 1

    def _mark_occupied(self, ix: int, iy: int, h_min_cm: int,
                       h_max_cm: int) -> None:
        """Record one cell's vertical span for this frame.

        A cell can receive several depth samples in one frame (a wall column);
        its vertical extent is what separates a wall from a tilted floor
        band, so both the minimum and maximum height are kept.
        """
        cur = self._cell(ix, iy)
        cur[1] += 1
        cur[2] = min(cur[2], int(h_min_cm))
        cur[3] = max(cur[3], int(h_max_cm))

    def _update_grid_locked(self, pts_camera: np.ndarray,
                            R: np.ndarray, t: np.ndarray,
                            pose, segment: int) -> None:
        if self._R_align is None or self._origin_world is None:
            return
        world = pts_camera @ R.T + t
        aligned = self._aligned_points(world)
        cam = self._aligned_points(t.reshape(1, 3))[0]
        self._path.append([round(float(cam[0]), 3),
                           round(float(cam[1]), 3)])
        forward = self._R_align @ (R @ np.array([0.0, 0.0, 1.0]))
        yaw = math.degrees(math.atan2(float(forward[1]),
                                      float(forward[0])))
        t_ns = 0
        if isinstance(pose, dict):
            t_ns = int(pose.get("t_ns") or 0)
        else:
            t_ns = int(getattr(pose, "t_ns", 0) or 0)
        self._last_pose = {
            "valid": True,
            "x": round(float(cam[0]), 4),
            "y": round(float(cam[1]), 4),
            "yaw_deg": round(float(yaw), 3),
            "t_ns": t_ns,
            "segment": int(segment),
        }

        z = aligned[:, 2]
        res = float(self.config.resolution_m)
        # Occupancy: only hits well above the floor are candidates.  Wall-like
        # filtering additionally needs vertical extent per cell (snapshot).
        valid = (np.isfinite(z)
                 & (z >= float(self.config.occupied_min_height_m))
                 & (z <= float(self.config.occupied_max_height_m)))
        cam_ix = int(math.floor(float(cam[0]) / res))
        cam_iy = int(math.floor(float(cam[1]) / res))
        if valid.any():
            hit = aligned[valid]
            ix = np.floor(hit[:, 0] / res).astype(np.int64)
            iy = np.floor(hit[:, 1] / res).astype(np.int64)
            h_cm = np.clip(np.rint(hit[:, 2] * 100.0), 0, 255).astype(
                np.int64)
            # One observation per grid cell per frame, carrying that cell's
            # minimum AND maximum height.  Keeping only the max threw away
            # the vertical extent a wall-like cell needs.
            order = np.lexsort((h_cm, iy, ix))
            ix, iy, h_cm = ix[order], iy[order], h_cm[order]
            new_cell = np.ones(len(ix), dtype=bool)
            new_cell[1:] = (ix[1:] != ix[:-1]) | (iy[1:] != iy[:-1])
            starts = np.flatnonzero(new_cell)
            ends = np.concatenate((starts[1:], [len(ix)])) - 1
            for start, end in zip(starts, ends):
                self._mark_occupied(int(ix[start]), int(iy[start]),
                                    int(h_cm[start]), int(h_cm[end]))
        # Carve free space along rays to EVERY depth hit, including the floor.
        # Carving only to above-floor hits left the walkable floor unmarked.
        finite = np.isfinite(aligned).all(axis=1)
        ray_ok = (finite
                  & (z >= -0.5)
                  & (z <= float(self.config.occupied_max_height_m)))
        ray_cells = np.floor(
            aligned[ray_ok, :2] / res).astype(np.int64)
        if len(ray_cells) == 0:
            return
        ray_cells = np.unique(ray_cells, axis=0)
        limit = int(self.config.max_rays_per_update)
        if len(ray_cells) > limit:
            sel = np.linspace(0, len(ray_cells) - 1, limit).astype(np.int64)
            ray_cells = ray_cells[sel]
        for x, y in ray_cells.tolist():
            if x == cam_ix and y == cam_iy:
                continue
            for fx, fy in _bresenham(cam_ix, cam_iy, int(x), int(y)):
                self._mark_free(fx, fy)

    # -- snapshot ---------------------------------------------------------

    @staticmethod
    def _rle_base64(mask: np.ndarray) -> str:
        """Row-major run lengths (uint16), starting with a 0-run."""
        bits = np.asarray(mask, dtype=np.uint8).reshape(-1)
        if bits.size == 0 or not bits.any():
            return ""
        change = np.flatnonzero(bits[1:] != bits[:-1]) + 1
        starts = np.concatenate(([0], change))
        ends = np.concatenate((change, [bits.size]))
        # uint16 is enough for this bounded grid (<= 60k cells) and halves
        # the payload versus uint32; the app decodes the same dtype.
        runs = (ends - starts).astype("<u2")
        if bits[0] == 1:
            runs = np.concatenate((np.array([0], dtype="<u2"), runs))
        return base64.b64encode(runs.tobytes()).decode("ascii")

    @staticmethod
    def _merge_wall_segments(segments, angle_tol_deg: float = 8.0,
                             line_tol_m: float = 0.12,
                             gap_tol_m: float = 0.15):
        """Merge collinear, nearby segments (bounded passes, deterministic)."""
        if len(segments) < 2:
            return [list(s) for s in segments]
        cos_tol = math.cos(math.radians(angle_tol_deg))
        merged = [list(map(float, s)) for s in segments]

        def unit(seg):
            dx, dy = seg[2] - seg[0], seg[3] - seg[1]
            n = math.hypot(dx, dy)
            if n <= 1e-9:
                return None
            return dx / n, dy / n

        def can_merge(a, b) -> bool:
            ua, ub = unit(a), unit(b)
            if ua is None or ub is None:
                return False
            if abs(ua[0] * ub[0] + ua[1] * ub[1]) < cos_tol:
                return False
            # Perpendicular offset of b's endpoints from a's infinite line.
            nx, ny = -ua[1], ua[0]
            off_b0 = abs((b[0] - a[0]) * nx + (b[1] - a[1]) * ny)
            off_b1 = abs((b[2] - a[0]) * nx + (b[3] - a[1]) * ny)
            if max(off_b0, off_b1) > line_tol_m:
                return False
            # Interval overlap/gap along a's direction.
            t = [((p[0] - a[0]) * ua[0] + (p[1] - a[1]) * ua[1])
                 for p in (a[:2], a[2:], b[:2], b[2:])]
            a0, a1 = sorted((t[0], t[1]))
            b0, b1 = sorted((t[2], t[3]))
            if b1 < a0 - gap_tol_m or b0 > a1 + gap_tol_m:
                return False
            return True

        def merge(a, b):
            ua = unit(a)
            pts = [a[:2], a[2:], b[:2], b[2:]]
            t = [(p[0] - a[0]) * ua[0] + (p[1] - a[1]) * ua[1] for p in pts]
            lo = t.index(min(t))
            hi = t.index(max(t))
            return [pts[lo][0], pts[lo][1], pts[hi][0], pts[hi][1]]

        for _ in range(4):
            changed = False
            out = []
            used = [False] * len(merged)
            for i, a in enumerate(merged):
                if used[i]:
                    continue
                used[i] = True
                for j in range(i + 1, len(merged)):
                    if used[j] or not can_merge(a, merged[j]):
                        continue
                    a = merge(a, merged[j])
                    used[j] = True
                    changed = True
                out.append(a)
            merged = out
            if not changed:
                break
        return merged

    def _wall_segments(self, rel_occ, origin):
        """Greedy 2D line segments from occupied cells, then collinear merge.

        A segment starts at a deterministic seed cell, picks the dominant
        direction among its nearby neighbours (support within a short window),
        grows contiguously in both directions with a small gap tolerance, and
        consumes the cells it covered.  Contiguous runs can be at any angle, so
        diagonal/rounded walls are not forced onto the grid axes.
        """
        res = float(self.config.resolution_m)
        min_len = float(self.config.min_wall_segment_m)
        max_segments = int(self.config.max_wall_segments)
        if not rel_occ or max_segments <= 0:
            return []
        cells = np.asarray(sorted(rel_occ), dtype=np.float64)
        centers = cells + 0.5  # cell centres in cell units
        unused = np.ones(len(cells), dtype=bool)
        tol = max(0.75, 0.02 / res)  # perpendicular tolerance, cell units
        gap = 2.5  # allowed gap along a run, cell units
        window = 12.0  # direction-evidence window, cell units
        segments = []
        while unused.any() and len(segments) < max_segments:
            seed_i = int(np.flatnonzero(unused)[0])
            seed = centers[seed_i]
            rel = centers - seed
            dist = np.hypot(rel[:, 0], rel[:, 1])
            local = unused & (dist > 0) & (dist <= 2.9)
            if not local.any():
                unused[seed_i] = False
                continue
            # At most the 8 nearest neighbours provide direction candidates.
            idx = np.flatnonzero(local)
            idx = idx[np.argsort(dist[idx], kind="stable")][:8]
            dirs = rel[idx] / dist[idx, None]
            proj_all = rel @ dirs.T
            perp_all = np.abs(rel[:, 0][:, None] * dirs[:, 1][None, :]
                             - rel[:, 1][:, None] * dirs[:, 0][None, :])
            support = (((np.abs(proj_all) <= window)
                        & (perp_all <= tol) & unused[:, None]).sum(axis=0))
            best = int(np.argmax(support))
            if int(support[best]) < 2:
                unused[seed_i] = False
                continue
            d = dirs[best]
            proj = rel @ d
            perp = np.abs(rel[:, 0] * d[1] - rel[:, 1] * d[0])
            line = unused & (perp <= tol)
            if not line.any():
                unused[seed_i] = False
                continue
            ps = np.sort(proj[line])
            lo = max(int(np.searchsorted(ps, 0.0, side="right")) - 1, 0)
            hi = lo + 1
            while lo > 0 and ps[lo] - ps[lo - 1] <= gap:
                lo -= 1
            while hi < len(ps) and ps[hi] - ps[hi - 1] <= gap:
                hi += 1
            t0, t1 = float(ps[lo]), float(ps[hi - 1])
            if (t1 - t0) * res < min_len:
                unused[seed_i] = False
                continue
            mask = line & (proj >= t0 - 1e-9) & (proj <= t1 + 1e-9)
            unused &= ~mask
            p0 = seed + d * t0
            p1 = seed + d * t1
            segments.append([
                round(origin[0] + p0[0] * res, 4),
                round(origin[1] + p0[1] * res, 4),
                round(origin[0] + p1[0] * res, 4),
                round(origin[1] + p1[1] * res, 4),
            ])
        segments = self._merge_wall_segments(
            segments, line_tol_m=max(res * 1.5, 0.05),
            gap_tol_m=max(res * 3.0, 0.10))
        segments.sort(key=lambda s: (s[1], s[0], s[3], s[2]))
        return segments[:max_segments]

    @staticmethod
    def _connected_components(cells):
        """8-connected components of a set of (ix, iy) cells, deterministic."""
        remaining = set(cells)
        components = []
        for seed in sorted(cells):
            if seed not in remaining:
                continue
            remaining.remove(seed)
            stack = [seed]
            component = [seed]
            while stack:
                x, y = stack.pop()
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        nxt = (x + dx, y + dy)
                        if nxt in remaining:
                            remaining.remove(nxt)
                            stack.append(nxt)
                            component.append(nxt)
            components.append(component)
        return components

    @staticmethod
    def _component_thickness_cells(component) -> float:
        """Robust thickness of a cell component, in cells.

        A multi-source BFS from the component's background gives a
        chessboard distance for every component cell; twice the median is a
        thickness estimate that a single noisy spur or a corner cannot
        inflate.  A one-cell-wide line measures 2 cells; a filled 0.6 m blob
        measures its short side.
        """
        xs = [p[0] for p in component]
        ys = [p[1] for p in component]
        min_x, min_y = min(xs), min(ys)
        width = max(xs) - min_x + 1
        height = max(ys) - min_y + 1
        mask = np.zeros((height + 2, width + 2), dtype=np.bool_)
        for x, y in component:
            mask[y - min_y + 1, x - min_x + 1] = True
        dist = np.full(mask.shape, -1, dtype=np.int32)
        queue = collections.deque()
        for yy, xx in np.argwhere(~mask).tolist():
            dist[yy, xx] = 0
            queue.append((int(xx), int(yy)))
        h, w = mask.shape
        neighbours = ((-1, -1), (-1, 0), (-1, 1), (0, -1),
                      (0, 1), (1, -1), (1, 0), (1, 1))
        while queue:
            x, y = queue.popleft()
            nd = int(dist[y, x]) + 1
            for dx, dy in neighbours:
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    if dist[ny, nx] < 0 or int(dist[ny, nx]) > nd:
                        dist[ny, nx] = nd
                        queue.append((nx, ny))
        values = dist[mask]
        return 2.0 * float(np.median(values))

    def _close_cells(self, cells):
        """Morphological closing (3x3 dilate then erode) of a cell set.

        Depth samples can land on cell boundaries, leaving one- or two-cell
        raster gaps in an otherwise continuous wall.  Connected components
        and the shape estimate run on the closed set so those gaps do not
        fragment a real wall into single cells; the wall cells that are
        finally transmitted are still only the cells that carried depth
        evidence (``cells``).
        """
        if not cells:
            return []
        xs = [p[0] for p in cells]
        ys = [p[1] for p in cells]
        min_x, min_y = min(xs), min(ys)
        width = max(xs) - min_x + 1
        height = max(ys) - min_y + 1
        pad = 2
        mask = np.zeros((height + 2 * pad, width + 2 * pad), dtype=np.bool_)
        for x, y in cells:
            mask[y - min_y + pad, x - min_x + pad] = True

        def shifted(source, dx, dy):
            out = np.zeros_like(source)
            h, w = source.shape
            sx0, sx1 = max(0, -dx), min(w, w - dx)
            sy0, sy1 = max(0, -dy), min(h, h - dy)
            out[sy0 + dy:sy1 + dy, sx0 + dx:sx1 + dx] = \
                source[sy0:sy1, sx0:sx1]
            return out

        dilated = mask.copy()
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx or dy:
                    dilated |= shifted(mask, dx, dy)
        closed = dilated.copy()
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx or dy:
                    closed &= shifted(dilated, dx, dy)
        rows, cols = np.nonzero(closed)
        return [(int(c) - pad + min_x, int(r) - pad + min_y)
                for r, c in zip(rows.tolist(), cols.tolist())]

    @staticmethod
    def _component_extents(component, res: float):
        """Principal-axis (major, minor) extents of a component, in metres."""
        points = np.asarray(component, dtype=np.float64) * res
        centred = points - points.mean(axis=0)
        # eigh on the 2x2 covariance; the last eigenvector is the major axis.
        _, vectors = np.linalg.eigh(centred.T @ centred)
        major = vectors[:, -1]
        minor = vectors[:, 0]
        proj_major = centred @ major
        proj_minor = centred @ minor
        # Add one cell so a single row of cells is not zero-width.
        major_m = float(proj_major.max() - proj_major.min()) + res
        minor_m = float(proj_minor.max() - proj_minor.min()) + res
        return major_m, minor_m

    @staticmethod
    def _local_normal(x, y, cells):
        """Local minor-axis direction of a component around (x, y), or None.

        A line-like neighbourhood has a dominant along-wall axis; the minor
        eigenvector is the wall normal.  An isotropic neighbourhood (blob) has
        no meaningful normal and is skipped.
        """
        offsets = []
        for dx in (-2, -1, 0, 1, 2):
            for dy in (-2, -1, 0, 1, 2):
                if dx == 0 and dy == 0:
                    continue
                if (x + dx, y + dy) in cells:
                    offsets.append((dx, dy))
        if len(offsets) < 3:
            return None
        points = np.asarray(offsets, dtype=np.float64)
        centred = points - points.mean(axis=0)
        values, vectors = np.linalg.eigh(centred.T @ centred)
        if values[1] <= 1e-9 or values[0] > 0.5 * values[1]:
            return None
        return vectors[:, 0]

    def _component_bounds_free_space(self, component, free_cells) -> bool:
        """True when carved free cells lie on one side but not the other.

        Each sampled cell is probed along its **local wall normal**, so grid
        gaps along the run cannot masquerade as the far side.  Both sides free
        means the candidate sits in open observed space (an obstacle); both
        sides not-free means no free boundary was observed.  A wall seen from
        both sides is rejected by this test -- documented limitation.
        """
        probe = int(self.config.wall_boundary_probe_cells)
        cells = set(component)
        if len(component) > 200:
            idx = np.linspace(0, len(component) - 1, 200).astype(np.int64)
            sample = [component[int(i)] for i in idx]
        else:
            sample = component
        good = valid = 0
        for x, y in sample:
            normal = self._local_normal(x, y, cells)
            if normal is None:
                continue
            valid += 1
            dx = int(round(float(normal[0]) * probe))
            dy = int(round(float(normal[1]) * probe))
            if dx == 0 and dy == 0:
                continue
            x_pos = (x + dx, y + dy) in free_cells
            x_neg = (x - dx, y - dy) in free_cells
            if x_pos != x_neg:
                good += 1
        if valid == 0:
            # Too little shape to judge; leave it to the segment fitter.
            return True
        need = max(1, int(math.ceil(
            float(self.config.wall_boundary_min_fraction) * valid)))
        return good >= need

    def _wall_components(self, candidates, free_cells):
        """Split candidates into wall vs obstacle components.

        Only elongated, thin components that bound observed free space are
        walls.  Everything else stays in the occupied/obstacle layer.
        """
        res = float(self.config.resolution_m)
        candidate_set = set(candidates)
        closed = self._close_cells(candidate_set)
        components = self._connected_components(closed)
        accepted = set()
        stats = {"components": len(components), "elongated": 0,
                 "rejected_shape": 0, "rejected_boundary": 0}
        max_thickness = float(self.config.wall_max_thickness_m)
        min_aspect = float(self.config.wall_min_aspect_ratio)
        min_loop = float(self.config.wall_min_loop_extent_m)
        for component in components:
            # Shape is measured on the closed component (gap-tolerant);
            # accepted wall cells are only cells that carried evidence.
            evidence = [c for c in component if c in candidate_set]
            if not evidence:
                continue
            thickness_m = self._component_thickness_cells(component) * res
            major_m, minor_m = self._component_extents(component, res)
            aspect = major_m / max(minor_m, res)
            # Elongated and thin is a wall.  A large thin boundary (room ring,
            # L/T network, diagonal wall) is also a wall even when its PCA
            # extents are similar; a small compact loop is an object.
            large = (major_m >= min_loop and minor_m >= min_loop)
            if thickness_m > max_thickness or (aspect < min_aspect
                                               and not large):
                stats["rejected_shape"] += 1
                continue
            stats["elongated"] += 1
            if (self.config.wall_boundary_check
                    and not self._component_bounds_free_space(
                        evidence, free_cells)):
                stats["rejected_boundary"] += 1
                continue
            accepted.update(evidence)
        return accepted, stats

    def _build_body_locked(self) -> None:
        res = float(self.config.resolution_m)
        min_extent_cm = float(self.config.wall_min_extent_m) * 100.0
        min_top_cm = float(self.config.wall_min_top_m) * 100.0
        entries = []
        raw_occupied = free = 0
        candidate_cells: set[tuple[int, int]] = set()
        rejected_extent_cells = 0
        rejected_height_cells = 0
        free_cells: set[tuple[int, int]] = set()
        for (ix, iy), (free_count, occ_count, h_min, h_max) in \
                self._cells.items():
            if occ_count > 0:
                raw_occupied += 1
                # A wall *candidate* must span real vertical structure and
                # clear a modest height floor.  The height gate is only a
                # low-clutter filter (desk/chair tops); the component shape
                # and free-boundary tests below are the real discriminator.
                span_cm = h_max - h_min
                if span_cm < min_extent_cm:
                    rejected_extent_cells += 1
                elif h_max < min_top_cm:
                    rejected_height_cells += 1
                else:
                    candidate_cells.add((int(ix), int(iy)))
                entries.append((ix, iy, 2))
            elif free_count > 0:
                free += 1
                free_cells.add((int(ix), int(iy)))
                entries.append((ix, iy, 1))
        wall_stats = {"components": 0, "elongated": 0,
                      "rejected_shape": 0, "rejected_boundary": 0}
        accepted_cells: set[tuple[int, int]] = set()
        if entries:
            min_ix = min(e[0] for e in entries)
            min_iy = min(e[1] for e in entries)
            max_ix = max(e[0] for e in entries)
            max_iy = max(e[1] for e in entries)
            origin = [round(min_ix * res, 4), round(min_iy * res, 4)]
            width = max_ix - min_ix + 1
            height = max_iy - min_iy + 1
            occ_mask = np.zeros((height, width), dtype=np.uint8)
            free_mask = np.zeros((height, width), dtype=np.uint8)
            if candidate_cells:
                accepted_cells, wall_stats = self._wall_components(
                    candidate_cells, free_cells)
            rel_wall = []
            for ix, iy, state in entries:
                rx, ry = ix - min_ix, iy - min_iy
                if state == 2:
                    # The occupied layer is the obstacle layer: every
                    # above-floor cell is rendered, wall or furniture.
                    occ_mask[ry, rx] = 1
                    if (ix, iy) in accepted_cells:
                        rel_wall.append((rx, ry))
                else:
                    free_mask[ry, rx] = 1
            occupied_rle = self._rle_base64(occ_mask)
            free_rle = self._rle_base64(free_mask)
            wall_cells = [[int(x), int(y)] for x, y in
                          sorted(rel_wall)[:int(
                              self.config.max_wall_cells_transmit)]]
            # Segment fitting sees ONLY accepted wall cells.  Feeding every
            # above-floor hit (or compact furniture) produced the live
            # "obstruction becomes a wall" noise.
            walls = self._wall_segments(rel_wall, origin)
        else:
            origin = [0.0, 0.0]
            width = height = 0
            occupied_rle = free_rle = ""
            wall_cells = []
            walls = []

        valid = self._floor is not None
        gravity = {"valid": False, "up": None, "roll_deg": None,
                   "pitch_deg": None, "inlier_ratio": None}
        floor = {"valid": False, "normal": None, "offset_m": None,
                 "camera_height_m": None, "inliers": None}
        if valid:
            est = self._floor
            up = [round(float(v), 6) for v in est["up_camera"]]
            up_cam = np.asarray(up, dtype=np.float64)
            pitch = math.asin(max(-1.0, min(1.0, -float(up_cam[2]))))
            roll = math.atan2(float(up_cam[0]), -float(up_cam[1]))
            gravity = {
                "valid": True,
                "up": up,
                "roll_deg": round(math.degrees(roll), 3),
                "pitch_deg": round(math.degrees(pitch), 3),
                "inlier_ratio": round(float(est["inlier_ratio"]), 4),
            }
            floor = {
                "valid": True,
                "normal": up,
                "offset_m": round(float(est["camera_height_m"]), 4),
                "camera_height_m": round(float(est["camera_height_m"]), 4),
                "inliers": int(est["inlier_count"]),
                "rmse_m": round(float(est.get("rmse_m", 0.0)), 4),
                "refinements": int(est.get("refinements", 0)),
            }

        occupied = int(raw_occupied)
        diagnostics = {
            # Obstacle layer: every cell with an above-floor hit.  This is
            # what occupied_rle encodes and what the app renders.
            "occupied_cells": int(raw_occupied),
            # Cell-stage wall gate: occupied cells that survived the vertical
            # extent test, plus why the rest were dropped (this is where a
            # too-high wall_min_top_m used to kill real walls before the
            # component stage ever saw them).
            "wall_candidate_cells": int(len(candidate_cells)),
            "wall_cells_rejected_extent": int(rejected_extent_cells),
            "wall_cells_rejected_height": int(rejected_height_cells),
            # Candidate cells that survived the elongated/thin component and
            # free-boundary tests; these feed wall_cells and walls only.
            "wall_cells": int(len(accepted_cells)),
            "wall_components": int(wall_stats["components"]),
            "wall_components_elongated": int(wall_stats["elongated"]),
            "wall_components_rejected_shape":
                int(wall_stats["rejected_shape"]),
            "wall_components_rejected_boundary":
                int(wall_stats["rejected_boundary"]),
            "wall_segments": int(len(walls)),
            "free_cells": int(free),
            "floor_rmse_m": (round(float(self._floor.get("rmse_m", 0.0)), 4)
                             if valid else None),
            "floor_refinements": (int(self._floor.get("refinements", 0))
                                  if valid else None),
        }
        pose = self._last_pose
        if not valid or pose is None:
            pose = {"valid": False, "x": None, "y": None,
                    "yaw_deg": None, "t_ns": 0,
                    "segment": int(self.segment)}
        body = {
            "frame_id": STRUCTURE_FRAME_ID if valid else None,
            "pose_frame": "map_from_camera_optical" if valid else None,
            "structural": True,
            "persistent": False,
            "loop_corrected": False,
            "gravity_source": "estimated_from_floor",
            "gravity": gravity,
            "floor": floor,
            "grid": {
                "origin": origin,
                "resolution_m": res,
                "width": int(width),
                "height": int(height),
                "occupied_count": int(occupied),
                "free_count": int(free),
                "encoding": "rle_base64",
                "occupied_rle": occupied_rle,
                "free_rle": free_rle,
            },
            "diagnostics": diagnostics,
            "walls": walls,
            "wall_cells": wall_cells,
            "path": [[round(float(v), 3) for v in p] for p in self._path],
            "pose": pose,
            "revision": int(self.revision),
        }
        self._cached_body = body
        fields = {k: v for k, v in body.items() if k != "revision"}
        self._cached_fields_bytes = json.dumps(
            fields, separators=(",", ":"), allow_nan=False).encode()[1:-1]
        self._cached_body_bytes = (
            b"{" + self._cached_fields_bytes
            + b',"revision":' + str(int(self.revision)).encode() + b"}")
        self._cached_revision = int(self.revision)
        self._last_cache_mono = time.monotonic()

    def _ensure_body_locked(self, min_interval_s: float = 0.0) -> None:
        if self._cached_revision == self.revision and self._cached_body:
            return
        if (self._cached_body is not None and min_interval_s > 0
                and self._last_cache_mono is not None
                and time.monotonic() - self._last_cache_mono < min_interval_s):
            return
        self._build_body_locked()

    def snapshot(self, min_interval_s: float = 0.0) -> dict:
        with self._lock:
            self._ensure_body_locked(min_interval_s)
            out = dict(self._cached_body)
            out["updated_age_s"] = (
                time.monotonic() - self._last_update_mono
                if self._last_update_mono is not None else None)
            return out

    def body_bytes(self, min_interval_s: float = 0.0) -> bytes:
        """Complete serialized structure body (including ``revision``)."""
        with self._lock:
            self._ensure_body_locked(min_interval_s)
            return self._cached_body_bytes

    def body_bundle(self, min_interval_s: float = 0.0):
        """(body fields bytes without braces/revision, revision, age seconds).

        The scalar ``revision`` is split out so the HTTP assembly can publish
        it last, after trajectory/status, without re-serializing the
        (potentially large) grid strings.
        """
        with self._lock:
            self._ensure_body_locked(min_interval_s)
            age = (time.monotonic() - self._last_update_mono
                   if self._last_update_mono is not None else None)
            return (self._cached_fields_bytes, int(self._cached_revision), age)

    def updated_age_s(self) -> float | None:
        with self._lock:
            return (time.monotonic() - self._last_update_mono
                    if self._last_update_mono is not None else None)

    def cached_revision(self, min_interval_s: float = 0.0) -> int:
        with self._lock:
            self._ensure_body_locked(min_interval_s)
            return int(self._cached_revision)
