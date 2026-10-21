"""Depth stop-cone: a body-frame clearance map from the mast depth cameras.

Every valid depth pixel (on a stride) is unprojected with the camera
intrinsics and placed in the robot body frame (x forward, y left, z up,
origin at the mast on the floor) using the measured rig extrinsics
(`config/orbbec_rig.yaml`: height + pitch per camera).  Points are then
classified by height alone:

    floor      |z| <= floor_tol_m            -> observed free space
    obstacle   obstacle_min_z <= z <= max    -> something to avoid
    drop       z <= drop_z_m                 -> a step / hole (also a stop)

Chassis returns are removed twice: by the rig's depth-pixel self-mask
polygon and by a body-frame footprint box (the learned chassis extent
from `config/ego_mask.npz`, dilated by a margin).

The output is a polar clearance map over `n_sectors` bearings: per
sector, the nearest obstacle/drop range and the farthest observed floor
range.  Clearance is the minimum of the two, so the robot never treats
unobserved space as free.  A forward corridor (chassis width + margin)
is summarised separately for the stop decision.

Nothing here sends commands; see `bearing.py` / `drive.py`.
"""
from __future__ import annotations

import dataclasses
import math
import time
from pathlib import Path

import numpy as np

from ..orbbec import (CONFIG_DIR, load_intrinsics, load_rig_config,
                      parse_self_mask_entries)


@dataclasses.dataclass(frozen=True)
class CameraGeometry:
    """Depth intrinsics + body-frame extrinsics for one rig camera."""

    role: str
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    height_m: float
    pitch_deg: float
    yaw_deg: float = 0.0
    # Depth-pixel polygons covering the robot's own chassis (rig YAML
    # `self_mask_pixels`), each a list of [x, y] vertices.
    self_polys: tuple = ()

    @classmethod
    def from_rig(cls, role: str, rig_path=None, config_dir=None,
                 rig_cfg=None) -> "CameraGeometry":
        cfg = rig_cfg or load_rig_config(rig_path)
        cams = cfg["robots"]["default"]["cameras"]
        for serial, c in cams.items():
            if c.get("role") != role:
                continue
            intr = load_intrinsics(serial, config_dir)
            _rects, polys = parse_self_mask_entries(c.get("self_mask_pixels"))
            polys = tuple(tuple((float(x), float(y)) for x, y in p)
                          for p in (polys or []))
            # Rects become 4-vertex polygons so one mask path serves both.
            for x0, y0, x1, y1 in (_rects or []):
                polys += (((x0, y0), (x1, y0), (x1, y1), (x0, y1)),)
            return cls(
                role=role,
                fx=float(intr["fx"]), fy=float(intr["fy"]),
                cx=float(intr["cx"]), cy=float(intr["cy"]),
                width=int(intr["width"]), height=int(intr["height"]),
                height_m=float(c["height_m"]),
                pitch_deg=float(c.get("pitch_deg", 0.0)),
                yaw_deg=float(c.get("yaw_deg", 0.0)),
                self_polys=polys)
        raise KeyError(f"no camera with role {role!r} in rig config")

    def rotation(self) -> np.ndarray:
        """3x3 matrix taking camera-optical vectors to body-frame vectors.

        Optical: x right, y down, z forward.  Body: x forward, y left,
        z up.  Pitch is negative when the camera looks below the horizon
        (rig YAML convention); yaw is positive to the left.
        """
        # Optical -> body with zero pitch/yaw: (x, y, z)_c -> (z, -x, -y)_b.
        base = np.array([[0.0, 0.0, 1.0],
                         [-1.0, 0.0, 0.0],
                         [0.0, -1.0, 0.0]])
        th = math.radians(self.pitch_deg)
        # Rotate about body +y so that forward (1,0,0) -> (cos th, 0, sin th).
        pitch = np.array([[math.cos(th), 0.0, -math.sin(th)],
                          [0.0, 1.0, 0.0],
                          [math.sin(th), 0.0, math.cos(th)]])
        ps = math.radians(self.yaw_deg)
        yaw = np.array([[math.cos(ps), -math.sin(ps), 0.0],
                        [math.sin(ps), math.cos(ps), 0.0],
                        [0.0, 0.0, 1.0]])
        return yaw @ pitch @ base

    def sample_rays(self, stride: int):
        """Body-frame ray directions for a strided pixel grid.

        Returns (dirs[N,3] float32, u[N], v[N]) where a depth value d at
        pixel (u, v) places the point at d * dirs + (0, 0, height_m).
        `dirs` is scaled so that d is the optical-axis depth (what the
        sensor reports), not the ray length.
        """
        us = np.arange(0, self.width, stride)
        vs = np.arange(0, self.height, stride)
        uu, vv = np.meshgrid(us, vs)
        uu = uu.reshape(-1)
        vv = vv.reshape(-1)
        d_cam = np.stack([(uu - self.cx) / self.fx,
                          (vv - self.cy) / self.fy,
                          np.ones_like(uu, dtype=np.float64)], axis=1)
        dirs = d_cam @ self.rotation().T
        return dirs.astype(np.float32), uu, vv

    def self_mask(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """True where a sampled pixel lies inside a chassis polygon."""
        inside = np.zeros(u.shape[0], dtype=bool)
        for poly in self.self_polys:
            inside |= _points_in_polygon(u.astype(np.float64),
                                         v.astype(np.float64), poly)
        return inside


def _points_in_polygon(x, y, poly) -> np.ndarray:
    """Even-odd rule, vectorised over points."""
    n = len(poly)
    inside = np.zeros(x.shape[0], dtype=bool)
    if n < 3:
        return inside
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        cross = (yi > y) != (yj > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            xint = (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
        inside ^= cross & (x < xint)
        j = i
    return inside


def footprint_from_rig(rig_path=None, config_dir=None, rig_cfg=None):
    """Chassis footprint box [x0, x1, y0, y1] in the body frame (metres).

    Prefers the learned extent in `config/ego_mask.npz`; falls back to the
    rig YAML `bev_student.ego.x_range_m/y_range_m`, then to a generic box.
    """
    cdir = Path(config_dir or CONFIG_DIR)
    cfg = rig_cfg or load_rig_config(rig_path)
    robot = cfg["robots"]["default"]
    ego = (robot.get("bev_student") or {}).get("ego") or {}
    mask_file = ego.get("mask_file")
    if mask_file:
        p = cdir / mask_file
        if p.exists():
            try:
                z = np.load(p)
                ext = [float(v) for v in z["extent_m"]]
                if len(ext) == 4 and ext[0] < ext[1] and ext[2] < ext[3]:
                    return ext
            except Exception:  # noqa: BLE001 - fall through to YAML
                pass
    xr = ego.get("x_range_m")
    yr = ego.get("y_range_m")
    if xr and yr:
        return [float(xr[0]), float(xr[1]), float(yr[0]), float(yr[1])]
    return [-0.15, 0.35, -0.30, 0.20]


@dataclasses.dataclass
class ConeConfig:
    stride: int = 4
    min_depth_m: float = 0.15
    max_depth_m: float = 5.0
    floor_tol_m: float = 0.06
    obstacle_min_z_m: float = 0.10
    obstacle_max_z_m: float = 1.80
    drop_z_m: float = -0.10
    # Chassis footprint [x0, x1, y0, y1] in the body frame + dilation.
    footprint_m: tuple = (-0.13, 0.32, -0.34, 0.19)
    footprint_margin_m: float = 0.06
    # A drop this close to the body is rail-silhouette noise, not a hole
    # (the near camera cannot see a real drop that close anyway).
    drop_ego_margin_m: float = 0.15
    # Forward corridor: chassis width + margin each side, from the front
    # of the footprint out to `lookahead_m`.
    corridor_margin_m: float = 0.08
    lookahead_m: float = 2.5
    # Polar sectors over +-half_fov_deg (0 = straight ahead, + = left).
    n_sectors: int = 15
    half_fov_deg: float = 60.0
    # Observed floor beyond the last floor point is unknown; allow this
    # much slack so quantisation does not make a clean floor look short.
    observed_slack_m: float = 0.10
    # Observed floor is the first contiguous run outward: a radial gap of
    # one empty bin (no floor return) ends it, so floor seen on the far
    # side of a hole, a step or an occluder never counts as reachable.
    gap_bin_m: float = 0.25
    # A frame with fewer valid sampled pixels than this fraction is not
    # trusted (lens covered, sensor fault).
    min_valid_frac: float = 0.25
    # Per-frame floor-plane refinement.  The nominal rig pitch/height are
    # a prior; a RANSAC plane within `floor_fit_max_tilt_deg` of it is
    # fitted on every frame and heights are measured from that plane.
    # (Measured 2026-10-21: both mast cameras sit ~6-7 deg off the YAML
    # pitch, enough to turn flat carpet into "drops" 1 m ahead.)
    floor_fit: bool = True
    floor_fit_max_tilt_deg: float = 15.0
    floor_fit_max_height_m: float = 0.15
    floor_fit_inlier_m: float = 0.025
    floor_fit_min_inliers: int = 250
    floor_fit_min_inlier_frac: float = 0.45
    floor_fit_samples: int = 48
    floor_fit_subsample: int = 3
    floor_fit_alpha: float = 0.4
    # The reference camera (the one that sees the most floor) fits against
    # the nominal prior; every other camera's fit must agree with the
    # reference plane to within these bounds, otherwise it inherits it.
    # (Live 2026-10-21: facing a desk, the far camera's RANSAC locked onto
    # a plane 31 cm below the floor and turned the carpet into obstacles.)
    reference_role: str = "near"
    floor_fit_ref_tilt_deg: float = 3.0
    floor_fit_ref_height_m: float = 0.06
    # Depth noise grows with range: widen the drop threshold by this much
    # per metre, and only flag a corridor drop on a few agreeing points.
    drop_slope_m_per_m: float = 0.03
    drop_min_points: int = 4
    # Only these cameras may report drops.  The far camera's residual plane
    # error (~1 deg vs the near camera) is several cm at 2 m and produced
    # phantom drops live; the near camera sees any real drop in time at
    # 0.3 m/s.  Empty = every camera.
    drop_roles: tuple = ("near",)
    # Slowly learned per-camera offset from the reference plane (normal
    # and offset deltas), applied when a camera cannot fit its own floor.
    plane_delta_alpha: float = 0.1

    def validate(self) -> None:
        if self.stride < 1:
            raise ValueError("stride must be >= 1")
        if not (0 < self.min_depth_m < self.max_depth_m):
            raise ValueError("min_depth_m must be < max_depth_m")
        if self.obstacle_min_z_m <= self.floor_tol_m:
            raise ValueError("obstacle_min_z_m must exceed floor_tol_m")
        if self.drop_z_m >= -self.floor_tol_m:
            raise ValueError("drop_z_m must be below -floor_tol_m")
        if self.n_sectors < 3 or self.n_sectors % 2 == 0:
            raise ValueError("n_sectors must be odd and >= 3")
        x0, x1, y0, y1 = self.footprint_m
        if not (x0 < x1 and y0 < y1):
            raise ValueError("footprint_m must be [x0<x1, y0<y1]")


@dataclasses.dataclass
class ClearanceMap:
    """Polar clearance (metres from the body origin) per bearing sector."""

    bearings_rad: np.ndarray        # (n,) sector centres, + = left
    clearance_m: np.ndarray         # (n,) min(obstacle, drop, observed)
    obstacle_m: np.ndarray          # (n,) nearest obstacle/drop, inf if none
    observed_m: np.ndarray          # (n,) farthest observed floor, 0 if none
    corridor_min_m: float           # nearest obstacle/drop in the corridor
    corridor_observed_m: float      # farthest floor seen in the corridor
    drop: bool                      # any drop point in the corridor
    n_floor: int
    n_obstacle: int
    n_drop: int
    valid_frac: float
    trusted: bool                   # enough valid pixels to act on
    stamp_us: int
    recv_ts: float
    roles: tuple
    floor_fit: dict = dataclasses.field(default_factory=dict)

    def age_s(self, now=None) -> float:
        return (now or time.monotonic()) - self.recv_ts

    def clearance_at(self, bearing_rad: float) -> float:
        i = int(np.argmin(np.abs(self.bearings_rad - bearing_rad)))
        return float(self.clearance_m[i])

    def to_dict(self) -> dict:
        return {
            "bearings_deg": [round(math.degrees(b), 1)
                             for b in self.bearings_rad.tolist()],
            "clearance_m": [round(float(c), 3) for c in self.clearance_m],
            "obstacle_m": [None if not math.isfinite(o) else round(float(o), 3)
                           for o in self.obstacle_m],
            "observed_m": [round(float(o), 3) for o in self.observed_m],
            "corridor_min_m": (None if not math.isfinite(self.corridor_min_m)
                               else round(self.corridor_min_m, 3)),
            "corridor_observed_m": round(self.corridor_observed_m, 3),
            "drop": self.drop,
            "n_floor": self.n_floor,
            "n_obstacle": self.n_obstacle,
            "n_drop": self.n_drop,
            "valid_frac": round(self.valid_frac, 3),
            "trusted": self.trusted,
            "stamp_us": self.stamp_us,
            "age_s": round(self.age_s(), 3),
            "roles": list(self.roles),
            "floor_fit": self.floor_fit,
        }


class DepthCone:
    """Turns one or more depth frames into a `ClearanceMap`."""

    def __init__(self, geometries, config: ConeConfig | None = None):
        self.config = config or ConeConfig()
        self.config.validate()
        self.geometries = {g.role: g for g in geometries}
        if not self.geometries:
            raise ValueError("DepthCone needs at least one camera geometry")
        self._rays = {}
        for role, g in self.geometries.items():
            dirs, u, v = g.sample_rays(self.config.stride)
            keep = ~g.self_mask(u, v)
            self._rays[role] = (dirs[keep], u[keep], v[keep], int(u.shape[0]))
        c = self.config
        half = math.radians(c.half_fov_deg)
        self.bearings = np.linspace(half, -half, c.n_sectors)  # left -> right
        self._sector_w = 2 * half / (c.n_sectors - 1)
        x0, x1, y0, y1 = c.footprint_m
        m = c.footprint_margin_m
        self._fp = (x0 - m, x1 + m, y0 - m, y1 + m)
        self._fp_drop = (x0 - c.drop_ego_margin_m, x1 + c.drop_ego_margin_m,
                         y0 - c.drop_ego_margin_m, y1 + c.drop_ego_margin_m)
        self._corr_y = (y0 - c.corridor_margin_m, y1 + c.corridor_margin_m)
        self._corr_x0 = x1 + m
        # Per-role floor plane (unit normal n, offset d: n.p + d = 0) and
        # the last fit report.  Starts at the nominal floor z = 0.
        self._floor = {role: (np.array([0.0, 0.0, 1.0]), 0.0)
                       for role in self.geometries}
        self._floor_report = {role: {"source": "prior"}
                              for role in self.geometries}
        # Learned (normal delta, offset delta) of each camera's floor plane
        # relative to the reference camera's, from frames where both fit.
        self._plane_delta = {role: None for role in self.geometries}
        self._rng = np.random.default_rng(0)

    # -- floor plane ----------------------------------------------------------

    def fit_floor(self, role: str, pts: np.ndarray) -> np.ndarray:
        """Refine the floor plane for `role` from body-frame points and
        return per-point signed heights above that plane.

        Falls back to the previous plane (initially the nominal floor)
        when the fit has too few inliers or tilts beyond the prior bound.
        """
        c = self.config
        n_prev, d_prev = self._floor[role]
        # Bounds: the reference camera is checked against the nominal floor
        # (z = 0); other cameras against the reference camera's live plane.
        ref = self._reference_plane(role)
        if ref is None:
            n_ref, d_ref = np.array([0.0, 0.0, 1.0]), 0.0
            tilt_bound = math.radians(c.floor_fit_max_tilt_deg)
            height_bound = c.floor_fit_max_height_m
        else:
            n_ref, d_ref = ref
            tilt_bound = math.radians(c.floor_fit_ref_tilt_deg)
            height_bound = c.floor_fit_ref_height_m
            if not self._plane_within(n_prev, d_prev, n_ref, d_ref,
                                      tilt_bound, height_bound):
                # Our own plane drifted away from the reference: adopt the
                # reference (plus this camera's learned offset from it).
                n_prev, d_prev = self._reference_with_delta(role, n_ref,
                                                            d_ref)
                self._floor[role] = (n_prev, d_prev)
        if ref is not None:
            # A camera that cannot fit this frame follows the reference
            # plane (with its learned offset) rather than a stale fit.
            n_follow, d_follow = self._reference_with_delta(role, n_ref,
                                                            d_ref)
            self._floor[role] = (n_follow, d_follow)
            n_prev, d_prev = n_follow, d_follow
        if not c.floor_fit or pts.shape[0] < c.floor_fit_min_inliers:
            self._floor_report[role] = {"source": "prior",
                                        "reason": "too_few_points"}
            return pts @ n_prev + d_prev
        fx0, fx1, fy0, fy1 = self._fp
        in_fp = ((pts[:, 0] >= fx0) & (pts[:, 0] <= fx1)
                 & (pts[:, 1] >= fy0) & (pts[:, 1] <= fy1))
        # Only near-floor points (relative to the current plane) feed the
        # fit; furniture at 0.5 m must not drag the plane up.
        h_prev = pts @ n_prev + d_prev
        cand = pts[~in_fp & (np.abs(h_prev) < 0.25)][::c.floor_fit_subsample]
        if cand.shape[0] < c.floor_fit_min_inliers:
            self._floor_report[role] = {"source": "prior",
                                        "reason": "too_few_candidates",
                                        "candidates": int(cand.shape[0])}
            return h_prev
        k = c.floor_fit_samples
        idx = self._rng.integers(0, cand.shape[0], size=(k, 3))
        a, b, cc = cand[idx[:, 0]], cand[idx[:, 1]], cand[idx[:, 2]]
        nrm = np.cross(b - a, cc - a)
        ln = np.linalg.norm(nrm, axis=1)
        ok = ln > 1e-6
        nrm[ok] /= ln[ok, None]
        nrm[nrm[:, 2] < 0] *= -1
        max_tilt = math.cos(math.radians(c.floor_fit_max_tilt_deg))
        ok &= nrm[:, 2] >= max_tilt
        # Candidate planes must sit within the bounds of the reference.
        a_d = -np.einsum("ij,ij->i", a, nrm)
        ok &= np.array([self._plane_within(nrm[i], a_d[i], n_ref, d_ref,
                                           tilt_bound, height_bound)
                        for i in range(nrm.shape[0])], dtype=bool)
        if not np.any(ok):
            self._floor_report[role] = {"source": "prior",
                                        "reason": "no_valid_sample"}
            return h_prev
        nrm, a = nrm[ok], a[ok]
        dist = np.abs(cand @ nrm.T - np.einsum("ij,ij->i", a, nrm)[None, :])
        counts = np.count_nonzero(dist < c.floor_fit_inlier_m, axis=0)
        best = int(np.argmax(counts))
        if (counts[best] < c.floor_fit_min_inliers
                or counts[best] < c.floor_fit_min_inlier_frac * cand.shape[0]):
            self._floor_report[role] = {"source": "prior",
                                        "reason": "too_few_inliers",
                                        "inliers": int(counts[best]),
                                        "candidates": int(cand.shape[0])}
            return h_prev
        inl = cand[dist[:, best] < c.floor_fit_inlier_m]
        ctr = inl.mean(axis=0)
        _u, _s, vt = np.linalg.svd(inl - ctr, full_matrices=False)
        n = vt[2]
        if n[2] < 0:
            n = -n
        if n[2] < max_tilt:
            self._floor_report[role] = {"source": "prior",
                                        "reason": "refined_tilt_exceeded"}
            return h_prev
        d = -float(ctr @ n)
        if not self._plane_within(n, d, n_ref, d_ref, tilt_bound,
                                  height_bound):
            self._floor_report[role] = {"source": "prior",
                                        "reason": "refined_out_of_bounds"}
            return h_prev
        # Low-pass the plane so a single noisy frame cannot flip the
        # classification; renormalise after blending.
        al = c.floor_fit_alpha
        n_new = (1 - al) * n_prev + al * n
        n_new /= np.linalg.norm(n_new)
        d_new = (1 - al) * d_prev + al * d
        self._floor[role] = (n_new, d_new)
        if ref is not None:
            dn, dd = n_new - n_ref, d_new - d_ref
            prev = self._plane_delta.get(role)
            al = c.plane_delta_alpha
            self._plane_delta[role] = ((dn, dd) if prev is None else
                                       ((1 - al) * prev[0] + al * dn,
                                        (1 - al) * prev[1] + al * dd))
        resid = (inl - ctr) @ n
        self._floor_report[role] = {
            "source": "fit",
            "inliers": int(inl.shape[0]),
            "candidates": int(cand.shape[0]),
            "tilt_deg": round(math.degrees(math.acos(min(1.0, n_new[2]))), 2),
            "pitch_deg": round(math.degrees(math.atan2(n_new[0], n_new[2])), 2),
            "roll_deg": round(math.degrees(math.atan2(n_new[1], n_new[2])), 2),
            "height_m": round(float(-d_new / n_new[2]), 3),
            "rmse_mm": round(float(np.sqrt(np.mean(resid ** 2))) * 1000, 1),
        }
        return pts @ n_new + d_new

    def floor_report(self) -> dict:
        return dict(self._floor_report)

    def plane(self, role: str) -> dict:
        """Current floor plane for `role` as {height_m, pitch_deg, roll_deg}."""
        n, d = self._floor[role]
        return {"height_m": round(float(-d / n[2]), 3),
                "pitch_deg": round(math.degrees(math.atan2(n[0], n[2])), 2),
                "roll_deg": round(math.degrees(math.atan2(n[1], n[2])), 2)}

    def _reference_with_delta(self, role: str, n_ref, d_ref):
        delta = self._plane_delta.get(role)
        if delta is None:
            return n_ref.copy(), float(d_ref)
        n = n_ref + delta[0]
        n = n / np.linalg.norm(n)
        return n, float(d_ref + delta[1])

    def _reference_plane(self, role: str):
        ref = self.config.reference_role
        if role == ref or ref not in self._floor:
            return None
        if self._floor_report.get(ref, {}).get("source") != "fit":
            return None
        return self._floor[ref]

    @staticmethod
    def _plane_within(n, d, n_ref, d_ref, tilt_bound, height_bound) -> bool:
        cosang = float(np.clip(np.dot(n, n_ref), -1.0, 1.0))
        if math.acos(cosang) > tilt_bound:
            return False
        # Height of each plane at the body origin: z = -d / n_z.
        h = -d / max(1e-6, float(n[2]))
        h_ref = -d_ref / max(1e-6, float(n_ref[2]))
        return abs(h - h_ref) <= height_bound

    # -- geometry -------------------------------------------------------------

    def points(self, role: str, depth: np.ndarray, depth_scale_mm=None):
        """Body-frame points (N,3) for one depth frame + valid fraction."""
        g = self.geometries[role]
        dirs, u, v, n_total = self._rays[role]
        if depth.shape != (g.height, g.width):
            raise ValueError(
                f"{role}: depth shape {depth.shape} != "
                f"calibrated {(g.height, g.width)}")
        scale = float(depth_scale_mm) if depth_scale_mm else 1.0
        d = depth[v, u].astype(np.float32) * (scale / 1000.0)
        ok = (d >= self.config.min_depth_m) & (d <= self.config.max_depth_m)
        valid_frac = float(np.count_nonzero(depth[v, u] > 0)) / max(1, n_total)
        pts = dirs[ok] * d[ok, None]
        pts[:, 2] += g.height_m
        return pts, valid_frac

    # -- evaluation -----------------------------------------------------------

    def evaluate(self, frames: dict) -> ClearanceMap | None:
        """`frames` maps role -> StampedFrame (or None).  Returns None when
        no configured camera supplied a frame."""
        c = self.config
        chunks = []
        drop_masks = []
        fracs = []
        stamp = 0
        recv = 0.0
        roles = []
        order = sorted(self.geometries,
                       key=lambda r: r != self.config.reference_role)
        for role in order:
            fr = frames.get(role)
            if fr is None or fr.depth is None:
                continue
            pts, frac = self.points(role, fr.depth,
                                    getattr(fr, "depth_scale_mm", None))
            h = self.fit_floor(role, pts)
            pts = pts.copy()
            pts[:, 2] = h
            chunks.append(pts)
            drop_ok = (not self.config.drop_roles
                       or role in self.config.drop_roles)
            drop_masks.append(np.full(pts.shape[0], drop_ok, dtype=bool))
            fracs.append(frac)
            roles.append(role)
            stamp = max(stamp, int(getattr(fr, "stamp_us", 0) or 0))
            recv = max(recv, float(getattr(fr, "recv_ts", 0.0) or 0.0))
        if not chunks:
            return None
        pts = np.concatenate(chunks, axis=0)
        return self._classify(pts, min(fracs), stamp, recv or time.monotonic(),
                              tuple(roles),
                              drop_allowed=np.concatenate(drop_masks))

    def evaluate_points(self, pts: np.ndarray, roles=("synthetic",),
                        valid_frac: float = 1.0) -> ClearanceMap:
        """Classify already body-frame points (z = height above floor)."""
        return self._classify(np.asarray(pts, dtype=np.float32), valid_frac,
                              0, time.monotonic(), tuple(roles))

    def _contiguous_floor(self, sector_idx, rng) -> np.ndarray:
        """Per sector: range where the first contiguous run of observed
        floor ends (the start of the first empty radial bin after floor
        was first seen), or the farthest floor return if there is no gap."""
        c = self.config
        n = c.n_sectors
        nb = int(math.ceil(c.lookahead_m / c.gap_bin_m)) + 1
        b = np.minimum((rng / c.gap_bin_m).astype(int), nb - 1)
        counts = np.zeros((n, nb), dtype=np.int32)
        np.add.at(counts, (sector_idx, b), 1)
        occupied = counts > 0
        out = np.zeros(n, dtype=np.float32)
        far = np.zeros(n, dtype=np.float32)
        np.maximum.at(far, sector_idx, rng.astype(np.float32))
        for s in range(n):
            row = occupied[s]
            if not row.any():
                continue
            first = int(np.argmax(row))
            empties = np.flatnonzero(~row[first:])
            if empties.size == 0:
                out[s] = far[s]
            else:
                gap_bin = first + int(empties[0])
                out[s] = min(far[s], gap_bin * c.gap_bin_m)
        return out

    def _classify(self, pts, valid_frac, stamp, recv, roles,
                  drop_allowed=None) -> ClearanceMap:
        c = self.config
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        fx0, fx1, fy0, fy1 = self._fp
        in_fp = (x >= fx0) & (x <= fx1) & (y >= fy0) & (y <= fy1)
        dx0, dx1, dy0, dy1 = self._fp_drop
        in_fp_drop = (x >= dx0) & (x <= dx1) & (y >= dy0) & (y <= dy1)
        rng = np.hypot(x, y)
        floor = (np.abs(z) <= c.floor_tol_m) & ~in_fp
        obstacle = ((z >= c.obstacle_min_z_m) & (z <= c.obstacle_max_z_m)
                    & ~in_fp)
        drop = ((z <= c.drop_z_m - c.drop_slope_m_per_m * rng)
                & ~in_fp_drop & (rng <= c.lookahead_m))
        if drop_allowed is not None:
            drop &= drop_allowed
        block = obstacle | drop
        bearing = np.arctan2(y, x)
        half = math.radians(c.half_fov_deg) + self._sector_w / 2
        in_fov = np.abs(bearing) <= half
        n = c.n_sectors
        # Sector index: 0 = leftmost (+half_fov) ... n-1 = rightmost.
        idx = np.rint((self.bearings[0] - bearing) / self._sector_w).astype(int)
        idx = np.clip(idx, 0, n - 1)

        obstacle_m = np.full(n, np.inf, dtype=np.float32)
        observed_m = np.zeros(n, dtype=np.float32)
        sel = block & in_fov
        if np.any(sel):
            np.minimum.at(obstacle_m, idx[sel], rng[sel].astype(np.float32))
        sel = floor & in_fov
        if np.any(sel):
            observed_m = self._contiguous_floor(idx[sel], rng[sel])
        clearance = np.minimum(obstacle_m,
                               observed_m + c.observed_slack_m)
        clearance = np.minimum(clearance, c.lookahead_m).astype(np.float32)

        cy0, cy1 = self._corr_y
        in_corr = ((y >= cy0) & (y <= cy1) & (x >= self._corr_x0)
                   & (x <= c.lookahead_m))
        corr_block = block & in_corr
        corridor_min = float(np.min(x[corr_block])) if np.any(corr_block) \
            else math.inf
        corr_floor = floor & in_corr
        corridor_obs = float(np.max(x[corr_floor])) if np.any(corr_floor) \
            else 0.0
        trusted = valid_frac >= c.min_valid_frac
        return ClearanceMap(
            bearings_rad=self.bearings.astype(np.float32),
            clearance_m=clearance,
            obstacle_m=obstacle_m,
            observed_m=observed_m,
            corridor_min_m=corridor_min,
            corridor_observed_m=min(corridor_obs, c.lookahead_m),
            drop=bool(np.count_nonzero(drop & in_corr) >= c.drop_min_points),
            n_floor=int(np.count_nonzero(floor)),
            n_obstacle=int(np.count_nonzero(obstacle)),
            n_drop=int(np.count_nonzero(drop)),
            valid_frac=float(valid_frac),
            trusted=bool(trusted),
            stamp_us=int(stamp),
            recv_ts=float(recv),
            roles=tuple(roles),
            floor_fit={r: dict(self._floor_report.get(r, {}),
                               plane=self.plane(r)) for r in roles},
        )
