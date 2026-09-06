"""Depth -> bird's-eye-view occupancy grid (navd Phase A, plan Section 6.3).

Per camera, per tick: stride-subsample the depth frame, deproject to the
camera optical frame, transform to the robot body frame (x forward, y left,
z up; floor at z=0), fit/refine a ground plane with RANSAC, classify cells
(occupied / hazard / floor / overhang), fuse the two cameras (near camera is
authoritative below `near_authority_m`), and inflate by the robot radius for
planning. Deterministic numpy only — ~10 ms per frame-pair at 10 Hz.

Grid convention (60x60 for the 3x3 m @ 5 cm default):
  occ[row, col], row 0 = far edge (+range_m ahead), row grows toward the
  robot; col 0 = right edge (y = -width_m/2), col grows leftward. Classes:
  0 free, 1 occupied, 2 hazard, 3 inflated (planning only; the raw
  pre-inflation grid rides along on BevGrid for telemetry/dataset labels).

Sign convention for the mount rotation: `pitch_deg` negative = camera
center points below the horizon (per the rig YAML). Camera optical frame
is x right, y down, z forward; body = Rz(yaw) @ Ry(-pitch) @ R0 @ cam + t
with R0 mapping optical axes onto body axes.
"""

import dataclasses
import math
import time

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise ImportError("pip install opencv-python") from exc

from .orbbec import load_intrinsics, load_rig_config, CONFIG_DIR

FREE, OCCUPIED, HAZARD, INFLATED = 0, 1, 2, 3

# Height band above local ground for an obstacle the robot must avoid.
OCC_MIN_M = 0.03
OCC_MAX_M = 0.30
# Points above this height are overhang; the robot clears under them.
STRIDE = 4
MAX_RANGE_M = 6.0


@dataclasses.dataclass
class Mount:
    height_m: float
    pitch_deg: float
    yaw_deg: float = 0.0


@dataclasses.dataclass
class BevGrid:
    occ: np.ndarray            # uint8 (R, C) with INFLATED applied
    raw: np.ndarray            # uint8 (R, C) pre-inflation (telemetry/labels)
    stamp_us: int
    per_camera_age_s: dict
    plane_ok: dict
    roles: list
    cell_m: float = 0.05
    recv_ts: float = 0.0       # time.monotonic() at fuse (deadman clock)

    @property
    def shape(self):
        return self.occ.shape


def mount_rotation(pitch_deg, yaw_deg):
    """Camera-optical -> body rotation for a mounted camera (see module doc)."""
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


def deproject(depth_mm, fx, fy, cx, cy, stride=STRIDE, max_range_m=MAX_RANGE_M):
    """Pinhole deproject a Y16 depth image to camera-frame points (m).

    Returns (N, 3) float32 in the optical frame (x right, y down, z forward),
    invalid (0) and out-of-range pixels dropped.
    """
    z = depth_mm[::stride, ::stride].astype(np.float32) * 1e-3
    h, w = z.shape
    u = (np.arange(w, dtype=np.float32) * stride) - cx
    v = (np.arange(h, dtype=np.float32) * stride) - cy
    valid = (z > 0.0) & (z < max_range_m)
    x = (u[None, :] / fx) * z
    y = (v[:, None] / fy) * z
    return np.stack([x[valid], y[valid], z[valid]], axis=1)


def fit_ground_plane(pts, residual_m=0.015, min_inliers=50, min_frac=0.2,
                     iters=96, rng=None, d_range=None, range_scale=0.01,
                     ranges=None):
    """RANSAC a ground plane in the body frame. Returns (n, d, inlier_count)
    with n·p = d and n_z > 0, or None if the fit is untrustworthy.

    The floor prior gates candidate planes: normal within 20° of vertical,
    and — when `d_range` (min, max) is given — plane height at the origin
    inside that window. `ranges` (per-point distance from the camera) enables
    a range-scaled residual — max(residual_m, range_scale * range) — because
    fixed millimetre gates are tighter than stereo noise beyond ~2.5 m.
    Batching all iterations as one matrix product keeps this inside the
    10 Hz budget.
    """
    n_pts = len(pts)
    if n_pts < min_inliers:
        return None
    rng = rng or np.random.default_rng()
    idx = rng.choice(n_pts, (iters, 3), replace=False)
    p = pts[idx]                                   # (iters, 3, 3)
    n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    norm = np.linalg.norm(n, axis=1)
    valid = norm > 1e-6
    n[valid] /= norm[valid][:, None]
    flip = n[:, 2] < 0
    n[flip] = -n[flip]
    valid &= n[:, 2] >= math.cos(math.radians(20.0))
    d = np.einsum("ij,ij->i", n, p[:, 0])
    if d_range is not None:
        valid &= (d >= d_range[0]) & (d <= d_range[1])
    # Inlier counting on a subsample (cache-friendly; the winner is then
    # recounted on the full set). (n_eval, iters) distance matrix.
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
    # Recount the winning plane over the full point set.
    full = np.abs(pts @ n[best] - d[best])
    if ranges is not None:
        full = full < np.maximum(residual_m, range_scale * ranges)
    else:
        full = full < residual_m
    cnt = int(full.sum())
    if cnt >= min_inliers and cnt >= min_frac * n_pts:
        return n[best], float(d[best]), cnt
    return None


class BevBuilder:
    """Per-tick BEV builder for the mounted rig (plan Section 6.3)."""

    def __init__(self, rig_cfg=None, config_dir=None, mounts=None,
                 intrinsics=None, inflate_radius_m=None, seed=None):
        cfg = rig_cfg or load_rig_config()["robots"]["default"]
        self.cam_cfg = cfg["cameras"]
        bev = cfg["bev"]
        self.range_m = float(bev["range_m"])
        self.width_m = float(bev["width_m"])
        self.cell_m = float(bev["cell_m"])
        self.near_authority_m = float(bev["near_authority_m"])
        safety = cfg.get("safety", {})
        self.max_frame_age_s = float(safety.get("max_frame_age_s", 0.3))
        self.plane_residual_m = float(safety.get("plane_residual_m", 0.015))
        self.plane_min_inliers = int(safety.get("plane_min_inliers", 50))
        self.plane_min_frac = float(safety.get("plane_min_frac", 0.2))
        self.hazard_drop_m = float(safety.get("hazard_drop_m", 0.05))
        self.robot_radius_m = float(
            inflate_radius_m
            if inflate_radius_m is not None
            else cfg.get("robot", {}).get("radius_m", 0.20))
        # Self-view mask: the mast-mounted cameras see parts of the robot's
        # own chassis at fixed body-frame positions; those points must never
        # enter classification (the robot would block itself). The box is
        # asymmetric and measured from the mounted rig (see rig YAML).
        sm = cfg.get("robot", {}).get("self_mask", {})
        self.self_mask = None
        if sm:
            x0, x1 = sm.get("x_range_m", [-0.30, 1.00])
            y0, y1 = sm.get("y_range_m", [-0.30, 0.60])
            self.self_mask = dict(
                x0=float(x0), x1=float(x1), y0=float(y0), y1=float(y1),
                min_h=float(sm.get("min_height_m", 0.05)))
        # Returns inside this body-frame radius are dropped before plane
        # fitting and classification: the chassis silhouette generates
        # below-floor stereo artifacts just outside the self-mask, and no
        # cell this close is actionable anyway (ray march starts at
        # r_min = 0.35 m, inside this disc).
        self.min_range_m = float(bev.get("min_range_m", 0.55))
        self.rows = int(round(self.range_m / self.cell_m))
        self.cols = int(round(self.width_m / self.cell_m))
        # One RNG per camera: process() may run on parallel threads.
        self._seed = seed
        self._rngs = {}

        mounts = mounts if mounts is not None else {
            serial: Mount(float(c["height_m"]), float(c["pitch_deg"]),
                          float(c.get("yaw_deg", 0.0)))
            for serial, c in self.cam_cfg.items()}
        self.mounts = mounts
        intrinsics = intrinsics if intrinsics is not None else {
            serial: load_intrinsics(serial, config_dir) for serial in mounts}
        self.intrinsics = {}
        for serial, intr in intrinsics.items():
            self.intrinsics[serial] = {k: float(intr[k]) for k in ("fx", "fy", "cx", "cy")}
            self.intrinsics[serial]["width"] = int(intr.get("width", 0))
            self.intrinsics[serial]["height"] = int(intr.get("height", 0))
        # Role -> serial lookup (Stamps carry the serial).
        self._role_serial = {c["role"]: serial for serial, c in self.cam_cfg.items()}
        self._rot = {serial: mount_rotation(m.pitch_deg, m.yaw_deg)
                     for serial, m in mounts.items()}
        self._t = {serial: np.array([0.0, 0.0, mounts[serial].height_m], dtype=np.float32)
                   for serial in mounts}
        inflate_cells = int(round(self.robot_radius_m / self.cell_m))
        self._inflate_kernel = (
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * inflate_cells + 1, 2 * inflate_cells + 1))
            if inflate_cells > 0 else None)
        # Cell-center coordinates for planning/telemetry.
        self.x_centers = self.range_m - (np.arange(self.rows) + 0.5) * self.cell_m
        self.y_centers = (np.arange(self.cols) + 0.5) * self.cell_m - self.width_m / 2.0

    # --- per-camera ---------------------------------------------------------

    def to_body(self, pts_cam, serial):
        return (self._rot[serial] @ pts_cam.T).T + self._t[serial]

    def process(self, frame):
        """StampedFrame -> dict(occ, coverage, plane_ok, stamp_us) or None."""
        intr = self.intrinsics.get(frame.serial)
        if intr is None:
            raise RuntimeError(f"no intrinsics for mounted serial {frame.serial}")
        fx, fy = intr["fx"], intr["fy"]
        cx, cy = intr["cx"], intr["cy"]
        iw, ih = intr["width"], intr["height"]
        if iw and ih and (frame.width, frame.height) != (iw, ih):
            fx *= frame.width / iw
            fy *= frame.height / ih
            cx *= frame.width / iw
            cy *= frame.height / ih
        pts_cam = deproject(frame.depth, fx, fy, cx, cy)
        if len(pts_cam) == 0:
            return dict(occ=np.zeros((self.rows, self.cols), np.uint8),
                        coverage=np.zeros((self.rows, self.cols), bool),
                        plane_ok=False, stamp_us=frame.stamp_us)
        pts = self.to_body(pts_cam, frame.serial)
        if self.min_range_m > 0.0:
            close = np.hypot(pts[:, 0], pts[:, 1]) < self.min_range_m
            pts = pts[~close]
        if self.self_mask is not None:
            sm = self.self_mask
            on_body = (pts[:, 0] > sm["x0"]) & (pts[:, 0] < sm["x1"]) \
                & (pts[:, 1] > sm["y0"]) & (pts[:, 1] < sm["y1"]) \
                & (pts[:, 2] > sm["min_h"])
            pts = pts[~on_body]
        return self._classify(pts, frame.stamp_us, frame.serial)

    def _classify(self, pts, stamp_us, serial):
        # Floor prior window: the ground cannot sit above the assumed origin
        # by more than a few cm, nor below the assumed mount height + margin
        # (guards against fitting a tabletop as ground while tolerating
        # unmeasured mount heights).
        mount = self.mounts[serial]
        d_range = (-(mount.height_m + 0.5), 0.05)
        ranges = np.linalg.norm(pts - self._t[serial], axis=1)
        rng = self._rngs.get(serial)
        if rng is None:
            rng = self._rngs[serial] = np.random.default_rng(self._seed)
        fit = fit_ground_plane(
            pts, residual_m=self.plane_residual_m,
            min_inliers=self.plane_min_inliers, min_frac=self.plane_min_frac,
            rng=rng, d_range=d_range, ranges=ranges)
        if fit is not None:
            n, d, _ = fit
            plane_ok = True
        else:
            n, d = np.array([0.0, 0.0, 1.0], dtype=np.float32), 0.0
            plane_ok = False
        # Height above local ground (the fitted plane z = (d - n_xy·p_xy)/n_z).
        height = pts[:, 2] - (d - n[0] * pts[:, 0] - n[1] * pts[:, 1]) / n[2]

        x, y = pts[:, 0], pts[:, 1]
        rows = ((self.range_m - x) / self.cell_m).astype(np.int32)
        cols = ((y + self.width_m / 2.0) / self.cell_m).astype(np.int32)
        inb = (rows >= 0) & (rows < self.rows) & (cols >= 0) & (cols < self.cols)
        rows, cols, height = rows[inb], cols[inb], height[inb]
        flat = rows * self.cols + cols

        occ = np.zeros(self.rows * self.cols, np.uint8)
        n_cells = self.rows * self.cols

        occupied = (height >= OCC_MIN_M) & (height <= OCC_MAX_M)
        floor = (height >= 0.0) & (height < OCC_MIN_M)
        below = height < -self.hazard_drop_m
        flat_occ = flat[occupied]
        occ_cnt = np.bincount(flat_occ, minlength=n_cells)
        occ[occ_cnt > 0] = OCCUPIED
        ground_cnt = np.bincount(flat[floor], minlength=n_cells)
        ground_sum = np.bincount(flat[floor], weights=height[floor],
                                 minlength=n_cells)
        drop_cnt = np.bincount(flat[below], minlength=n_cells)
        drop_sum = np.bincount(flat[below], weights=height[below],
                               minlength=n_cells)

        occ = occ.reshape(self.rows, self.cols)
        coverage = ((ground_cnt + drop_cnt + occ_cnt) > 0).reshape(self.rows, self.cols)
        # Negative obstacles: (a) cells whose points sit predominantly well
        # below the plane (stair treads, holes), (b) cells whose local ground
        # drops > hazard_drop_m vs a 4-neighbour (plan: "ground-plane drop
        # > 0.05 m between adjacent cells"). The majority test keeps single
        # stray below-plane pixels (depth noise at range) from tripping it.
        total_cnt = ground_cnt + drop_cnt + occ_cnt
        hazard = (drop_cnt >= 2) & (2 * drop_cnt > total_cnt)
        with np.errstate(invalid="ignore"):
            ground = np.where(ground_cnt > 0, ground_sum / np.maximum(ground_cnt, 1),
                              np.nan).reshape(self.rows, self.cols)
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            shifted = np.full_like(ground, np.nan)
            ys = slice(max(dy, 0), self.rows + min(dy, 0))
            xs = slice(max(dx, 0), self.cols + min(dx, 0))
            ys2 = slice(max(-dy, 0), self.rows + min(-dy, 0))
            xs2 = slice(max(-dx, 0), self.cols + min(-dx, 0))
            shifted[ys2, xs2] = ground[ys, xs]
            with np.errstate(invalid="ignore"):
                drop = ground - shifted  # neighbour lower than cell by > drop
            hazard |= (drop.reshape(-1) > self.hazard_drop_m)
        # Bridge the sparse far field (stride-4 cells at 3 m can hold 1-2
        # points) so a drop reads as a region, not scattered speckle.
        hazard = cv2.morphologyEx(
            hazard.reshape(self.rows, self.cols).astype(np.uint8),
            cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)).astype(bool).reshape(-1)
        occ = occ.reshape(-1)
        occ[hazard & (occ == FREE)] = HAZARD
        occ = occ.reshape(self.rows, self.cols)
        return dict(occ=occ, coverage=coverage, plane_ok=plane_ok, stamp_us=stamp_us)

    # --- fusion -------------------------------------------------------------

    def fuse(self, per_camera, ages_s):
        """Fuse per-camera dicts from process() into a BevGrid.

        per_camera: {role: dict or None} — None/missing = stale or absent
        camera; the grid degrades to the other camera's region only.
        Semantics: union of all contributions, then the near camera's cells
        override within its authority radius where it actually has coverage
        (closer = better depth accuracy — plan Section 6.3).
        """
        available = {r: g for r, g in per_camera.items() if g is not None}
        if not available:
            return None
        raw = np.zeros((self.rows, self.cols), np.uint8)
        for g in available.values():
            raw = np.maximum(raw, g["occ"])
        near = available.get("near")
        if near is not None:
            authority = np.tile((self.x_centers < self.near_authority_m)[:, None],
                                (1, self.cols))
            raw[authority & near["coverage"]] = near["occ"][authority & near["coverage"]]
        raw_fused = raw.copy()
        occ = raw.copy()
        if self._inflate_kernel is not None:
            blocked = ((occ == OCCUPIED) | (occ == HAZARD)).astype(np.uint8)
            dil = cv2.dilate(blocked, self._inflate_kernel)
            occ[(dil > 0) & (occ == FREE)] = INFLATED
        stamp = max(g["stamp_us"] for g in available.values())
        return BevGrid(occ=occ, raw=raw_fused, stamp_us=stamp,
                       per_camera_age_s=dict(ages_s),
                       plane_ok={r: g["plane_ok"] for r, g in available.items()},
                       roles=list(available), cell_m=self.cell_m,
                       recv_ts=time.monotonic())
