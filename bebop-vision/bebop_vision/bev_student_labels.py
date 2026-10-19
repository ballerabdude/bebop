"""Semantic BEV labels for the learned master-BEV student.

The teacher is SAM 3.1 for semantics and the rig's depth streams for
geometry.  Per labeled keyframe and camera:

  1. every sampled depth pixel is deprojected into the body frame using the
     same mount / intrinsics model as :mod:`bebop_vision.master_bev`;
  2. the point is projected into the **color** frame through the camera's
     depth->color extrinsic, so the SAM mask is sampled at the exact pixel
     that saw that depth sample (no zero-baseline approximation);
  3. the point is classified against that camera's fitted floor plane and
     accumulated as weighted evidence per 120x60 cell;
  4. the two cameras are fused and resolved with the priority
     ``hazard > wall > obstacle > floor``.

Contract (``labels/<stamp>.npz``)::

    semantic  uint8 [120, 60]   # 0 floor, 1 wall, 2 obstacle, 3 hazard,
                                # 255 ignore
    ego       bool  [120, 60]   # robot chassis
    coverage  bool  [120, 60]   # valid depth observation

Unobserved/invalid-depth cells are ``semantic=255, coverage=False``.
Chassis cells are ``semantic=255, ego=True``.  Never label an unobserved
cell as free.

The module is deliberately free of SAM/torch imports so the geometry can be
unit-tested on synthetic depth without a GPU.  The SAM runner lives in
``tools/bev_student_labels.py``.
"""

from __future__ import annotations

import dataclasses
import os
import time
from pathlib import Path

# Keep the Thor's OpenBLAS pool from spin-waiting; main.py does the same
# before numpy is imported.
for _thread_var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_thread_var, "1")

import cv2
import numpy as np

from .master_bev import fit_ground_plane, mount_rotation
from .orbbec import (build_self_view_mask, load_intrinsics, load_rig_config,
                     parse_self_mask_entries)

# Label classes.  These are *not* the planner's FREE/OCCUPIED/HAZARD ids.
FLOOR, WALL, OBSTACLE, HAZARD, IGNORE = 0, 1, 2, 3, 255

CLASS_NAMES = {
    FLOOR: "floor",
    WALL: "wall",
    OBSTACLE: "obstacle",
    HAZARD: "hazard",
    IGNORE: "ignore",
}

# Curated prompt sets.  A prompt is matched case-insensitively; SAM returns
# the original (lower-cased) concept as ``InstanceSegment.label``.
FLOOR_PROMPTS = ("floor", "carpet", "rug", "ground")
WALL_PROMPTS = ("wall", "door", "door frame", "pillar")
OBSTACLE_PROMPTS = (
    "box", "bag", "chair", "table", "person", "cable", "trash can",
    "shelf", "cardboard box",
    # The 2026-10-18 drives are home tours; beds/sofas/cabinets are the
    # dominant obstacles and the brief's box/chair set alone misses them.
    "bed", "sofa", "couch", "cabinet", "dresser", "wardrobe", "desk",
    "counter", "appliance",
)
# Hazard is depth-primary, but SAM can flag stairs/holes on the lower
# surface.  These prompts only support below-plane points; they never
# create a hazard by themselves.
HAZARD_PROMPTS = ("stairs", "staircase", "step", "hole", "ledge", "drop")
ALL_PROMPTS = (FLOOR_PROMPTS + WALL_PROMPTS + OBSTACLE_PROMPTS
               + HAZARD_PROMPTS)

PROMPT_CLASS = {
    **{p: FLOOR for p in FLOOR_PROMPTS},
    **{p: WALL for p in WALL_PROMPTS},
    **{p: OBSTACLE for p in OBSTACLE_PROMPTS},
    **{p: HAZARD for p in HAZARD_PROMPTS},
}

# BGR palette for review images.  Floor grey, wall blue, obstacle red,
# hazard orange, ignore near-black.  The operator BEV keeps its own palette.
LABEL_COLORS = {
    FLOOR: (120, 120, 120),
    WALL: (255, 120, 40),
    OBSTACLE: (40, 40, 230),
    HAZARD: (0, 160, 255),
    IGNORE: (18, 18, 18),
}
EGO_COLOR = (220, 220, 40)
EGO_FILL = (55, 55, 0)


def prompt_class(prompt):
    """Prompt string -> label class (or None for an unknown concept)."""
    key = str(prompt).strip().lower()
    return PROMPT_CLASS.get(key)


@dataclasses.dataclass(frozen=True)
class GridSpec:
    """Body-frame BEV grid, identical to ``master_bev.MasterBevBuilder``."""

    range_m: float = 6.0
    width_m: float = 3.0
    cell_m: float = 0.05

    @property
    def rows(self):
        return int(round(self.range_m / self.cell_m))

    @property
    def cols(self):
        return int(round(self.width_m / self.cell_m))

    @property
    def shape(self):
        return (self.rows, self.cols)

    @classmethod
    def from_config(cls, cfg):
        mb = (cfg.get("bev_student") or {}).get("grid")
        if not mb:
            mb = cfg.get("master_bev") or cfg.get("bev") or {}
        return cls(
            range_m=float(mb.get("range_m", 6.0)),
            width_m=float(mb.get("width_m", 3.0)),
            cell_m=float(mb.get("cell_m", 0.05)),
        )

    def cell_of(self, x, y):
        """Body (x, y) -> (row, col), or (-1, -1) outside the grid."""
        row = int(np.floor((self.range_m - np.asarray(x)) / self.cell_m))
        col = int(np.floor((np.asarray(y) + self.width_m / 2.0)
                           / self.cell_m))
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            return -1, -1
        return row, col

    def centers(self):
        x = self.range_m - (np.arange(self.rows) + 0.5) * self.cell_m
        y = (np.arange(self.cols) + 0.5) * self.cell_m - self.width_m / 2.0
        return x, y

    def contains(self, x, y):
        return ((-0.5 * self.width_m <= y)
                & (y < 0.5 * self.width_m)
                & (0.0 <= x) & (x < self.range_m))


@dataclasses.dataclass
class LabelConfig:
    """Label-fusion knobs.  Values come from ``bev_student.label`` in the rig
    YAML when present, otherwise the defaults below."""

    floor_band_m: float = 0.40
    plane_residual_m: float = 0.015
    plane_min_inliers: int = 50
    plane_min_frac: float = 0.10
    plane_normal_deg: float = 10.0
    noise_scale: float = 0.01
    near_range_m: tuple = (0.55, 1.20, 2.00)
    far_range_m: tuple = (1.00, 2.00, 4.50, 7.00)
    occ_min_m: float = 0.03
    occ_max_m: float = 1.50
    wall_min_h_m: float = 0.80
    wall_max_h_m: float = 2.50
    floor_tol_m: float = 0.04
    hazard_drop_m: float = 0.05
    # Per-cell weighted-evidence thresholds.
    floor_min_w: float = 0.40
    wall_min_w: float = 0.35
    obstacle_min_w: float = 0.35
    hazard_min_w: float = 0.50
    # Depth-only floor fallback in cells with no SAM mask at all.  This is
    # deliberately a weak vote: SAM remains the semantic source.  Set to 0
    # to require a SAM floor mask.
    depth_floor_weight: float = 0.25
    # A compact, short wall component is a solid object (bookshelf edge),
    # not a wall.  Tall pillars stay walls.
    wall_compact_m: float = 0.60
    wall_pillar_min_h_m: float = 2.00
    # Local floor-height drops (rug/step edges) are noisy indoors; the
    # master BEV leaves them off by default too.  Below-plane points are the
    # primary negative-obstacle signal unless this is enabled.
    local_drop_hazard: bool = False
    # Below-plane depth points on an unlabeled dark surface (sofa undersides,
    # shadows, rug fringes) are usually multipath noise, not holes.  Require
    # the point itself to be SAM floor/ground or stairs/hole.
    hazard_require_floor: bool = True
    min_range_m: float = 0.55
    max_range_m: float = 8.0
    ego_x_range_m: tuple = (0.0, 0.55)
    ego_y_range_m: tuple = (-0.22, 0.22)
    # Body-frame self mask from ``robot.self_mask`` (mast/rails), normalised
    # to x0/x1/y0/y1/min_h.  Floor points stay because z < min_h.
    self_mask: dict = None

    @classmethod
    def from_config(cls, cfg):
        raw = (cfg.get("bev_student") or {}).get("label") or {}
        mb = cfg.get("master_bev") or {}
        kw = {}
        for f in dataclasses.fields(cls):
            if f.name in raw:
                kw[f.name] = raw[f.name]
        for name in ("floor_band_m", "noise_scale", "near_range_m",
                     "far_range_m", "min_range_m", "max_range_m"):
            if name in mb and name not in kw:
                kw[name] = mb[name]
        safety = cfg.get("safety") or {}
        for name in ("plane_residual_m", "plane_min_inliers",
                     "plane_min_frac", "hazard_drop_m", "noise_scale"):
            if name in safety and name not in kw:
                kw[name] = safety[name]
        if "near_range_m" in kw:
            kw["near_range_m"] = tuple(float(v) for v in kw["near_range_m"])
        if "far_range_m" in kw:
            kw["far_range_m"] = tuple(float(v) for v in kw["far_range_m"])
        ego = (cfg.get("bev_student") or {}).get("ego") or {}
        if "x_range_m" in ego:
            kw["ego_x_range_m"] = tuple(float(v) for v in ego["x_range_m"])
        if "y_range_m" in ego:
            kw["ego_y_range_m"] = tuple(float(v) for v in ego["y_range_m"])
        sm = (cfg.get("robot") or {}).get("self_mask") or {}
        if sm:
            xr = sm.get("x_range_m", [-0.30, 1.00])
            yr = sm.get("y_range_m", [-0.30, 0.60])
            kw["self_mask"] = {
                "x0": float(xr[0]), "x1": float(xr[1]),
                "y0": float(yr[0]), "y1": float(yr[1]),
                "min_h": float(sm.get("min_height_m", 0.05)),
            }
        return cls(**kw)


@dataclasses.dataclass
class CameraSpec:
    """Static geometry for one camera: mount, intrinsics, depth sampling."""

    role: str
    serial: str
    width_d: int
    height_d: int
    fx_d: float
    fy_d: float
    cx_d: float
    cy_d: float
    width_c: int
    height_c: int
    fx_c: float
    fy_c: float
    cx_c: float
    cy_c: float
    stride_x: int
    stride_y: int
    R_body: np.ndarray
    t_body: np.ndarray
    R_dc: np.ndarray
    t_dc: np.ndarray
    # (N,) sampled depth pixel coordinates and (N,3) optical ray directions
    u_d: np.ndarray = None
    v_d: np.ndarray = None
    dir_d: np.ndarray = None
    dir_body: np.ndarray = None
    dir_color: np.ndarray = None
    self_pixels: np.ndarray = None

    def __post_init__(self):
        us = np.arange(0, self.width_d, self.stride_x, dtype=np.int32)
        vs = np.arange(0, self.height_d, self.stride_y, dtype=np.int32)
        uu, vv = np.meshgrid(us, vs)
        self.u_d = uu.reshape(-1).astype(np.int32)
        self.v_d = vv.reshape(-1).astype(np.int32)
        dirs = np.stack([
            (self.u_d - self.cx_d) / self.fx_d,
            (self.v_d - self.cy_d) / self.fy_d,
            np.ones_like(self.u_d, np.float32),
        ], axis=1).astype(np.float32)
        self.dir_d = dirs
        self.dir_body = (self.R_body @ dirs.T).T.astype(np.float32)
        self.dir_color = (self.R_dc @ dirs.T).T.astype(np.float32)
        if self.self_pixels is None:
            self.self_pixels = np.zeros(len(self.u_d), bool)

    @property
    def n_pixels(self):
        return int(len(self.u_d))


@dataclasses.dataclass
class PreparedObservation:
    """One camera's valid depth samples, ready for floor/evidence work."""

    role: str
    points: np.ndarray            # (N,3) body frame
    ranges: np.ndarray            # (N,) optical range
    forward: np.ndarray           # (N,) body x
    rows: np.ndarray              # (N,) int grid rows
    cols: np.ndarray              # (N,) int grid cols
    flat: np.ndarray              # (N,) row*cols + col
    in_grid: np.ndarray           # (N,) bool
    u_c: np.ndarray               # (N,) color pixel (clipped)
    v_c: np.ndarray
    in_color: np.ndarray          # (N,) bool
    floor_conf: np.ndarray        # (N,) 0..1 sampled SAM confidence
    wall_conf: np.ndarray
    obstacle_conf: np.ndarray
    hazard_conf: np.ndarray
    stamp_us: int = 0


@dataclasses.dataclass
class LabelArrays:
    semantic: np.ndarray
    ego: np.ndarray
    coverage: np.ndarray
    evidence: dict = None
    planes: dict = None
    stats: dict = None


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def height_above_plane(points, normal, offset):
    """Signed height above the plane ``n . p = d`` (n near-vertical)."""
    n = np.asarray(normal, np.float64)
    p = np.asarray(points, np.float64)
    nz = float(n[2])
    if abs(nz) < 1e-6:
        return p[:, 2].copy()
    return p[:, 2] - (float(offset) - n[0] * p[:, 0]
                      - n[1] * p[:, 1]) / nz


def range_weight(role, forward_m, cfg=None):
    """Confidence from the **forward** distance, matching master_bev.

    Near: full to 1.20 m, fade to zero at 2.00 m.  Far: ramp 1.00->2.00,
    full 2.00-4.50, fade to zero at 7.00 m.
    """
    cfg = cfg or LabelConfig()
    f = np.asarray(forward_m, np.float32)
    if role == "near" and len(cfg.near_range_m) == 3:
        _unused, full1, zero = cfg.near_range_m
        w = np.ones_like(f)
        fade = f > full1
        w[fade] = np.clip((zero - f[fade]) / max(zero - full1, 1e-6),
                          0.0, 1.0)
        return w.astype(np.float32)
    if role == "far" and len(cfg.far_range_m) == 4:
        ramp0, full0, full1, zero = cfg.far_range_m
        w = np.clip((f - ramp0) / max(full0 - ramp0, 1e-6), 0.0, 1.0)
        w *= np.clip((zero - f) / max(zero - full1, 1e-6), 0.0, 1.0)
        return w.astype(np.float32)
    return np.ones_like(f)


def build_camera_spec(role, serial, cam_cfg, intr, cfg, grid):
    """CameraSpec from a rig-YAML camera block + cached intrinsics JSON."""
    width_d = int(intr.get("width") or 0)
    height_d = int(intr.get("height") or 0)
    if width_d <= 0 or height_d <= 0:
        raise ValueError(f"{serial}: depth intrinsics missing width/height")
    fx_d = float(intr["fx"])
    fy_d = float(intr["fy"])
    cx_d = float(intr["cx"])
    cy_d = float(intr["cy"])

    width_c = int(intr.get("color_width") or width_d)
    height_c = int(intr.get("color_height") or height_d)
    fx_c = float(intr.get("color_fx") or fx_d)
    fy_c = float(intr.get("color_fy") or fy_d)
    cx_c = float(intr.get("color_cx") or cx_d)
    cy_c = float(intr.get("color_cy") or cy_d)

    R_body = mount_rotation(float(cam_cfg.get("pitch_deg", 0.0)),
                            float(cam_cfg.get("yaw_deg", 0.0)))
    t_body = np.array([float(cam_cfg.get("x_m", 0.0)),
                       float(cam_cfg.get("y_m", 0.0)),
                       float(cam_cfg.get("height_m", 0.0))],
                      np.float32)

    # The cached key is named color_to_depth_transform, but pyorbbecsdk's
    # param.transform is the DEPTH->COLOR rigid transform (see
    # tools/set_self_mask.py).  Translation is millimetres.
    ext = intr.get("color_to_depth_transform") or {}
    if ext.get("rotation") is not None and len(ext["rotation"]) == 9:
        R_dc = np.asarray(ext["rotation"], np.float64).reshape(3, 3)
        t_dc = np.asarray(ext.get("translation", [0, 0, 0]),
                          np.float64) / 1000.0
    else:
        R_dc = np.eye(3, dtype=np.float64)
        t_dc = np.zeros(3, np.float64)

    stride_x = int(cam_cfg.get("label_stride_x", cfg.get("label_stride_x", 4)))
    stride_y = int(cam_cfg.get(
        "label_stride_y",
        cfg.get(f"label_stride_y_{role}", 2 if role == "near" else 1)))

    # Self-view pixels: the rig config keeps the polygon even though the live
    # recorder no longer zeroes the published depth.  Labels must exclude
    # those pixels explicitly or the chassis becomes an obstacle.
    rects, polys = parse_self_mask_entries(
        cam_cfg.get("self_mask_pixels"))
    pix_mask = build_self_view_mask(rects, polys, width_d, height_d)
    self_pixels = None
    if pix_mask is not None:
        us = np.arange(0, width_d, stride_x, dtype=np.int32)
        vs = np.arange(0, height_d, stride_y, dtype=np.int32)
        uu, vv = np.meshgrid(us, vs)
        self_pixels = pix_mask[vv.reshape(-1), uu.reshape(-1)]

    return CameraSpec(
        role=role, serial=serial,
        width_d=width_d, height_d=height_d,
        fx_d=fx_d, fy_d=fy_d, cx_d=cx_d, cy_d=cy_d,
        width_c=width_c, height_c=height_c,
        fx_c=fx_c, fy_c=fy_c, cx_c=cx_c, cy_c=cy_c,
        stride_x=stride_x, stride_y=stride_y,
        R_body=R_body.astype(np.float32), t_body=t_body,
        R_dc=R_dc.astype(np.float32), t_dc=t_dc.astype(np.float32),
        self_pixels=self_pixels,
    )


def specs_from_config(rig_cfg=None, config_dir=None, roles=None):
    """Build CameraSpecs for the rig's cameras (roles optionally filtered)."""
    if rig_cfg is None:
        rig_cfg = load_rig_config()
    if "robots" in rig_cfg:
        cfg = rig_cfg["robots"]["default"]
    else:
        cfg = rig_cfg
    grid = GridSpec.from_config(cfg)
    label_cfg = LabelConfig.from_config(cfg)
    cams = cfg.get("cameras") or {}
    specs = {}
    for serial, cam_cfg in cams.items():
        role = cam_cfg.get("role")
        if roles and role not in roles:
            continue
        intr = load_intrinsics(serial, config_dir)
        specs[role] = build_camera_spec(role, serial, cam_cfg, intr,
                                        cfg, grid)
    return specs, cfg, grid, label_cfg


# ---------------------------------------------------------------------------
# Per-frame depth preparation
# ---------------------------------------------------------------------------


def prepare_observation(spec, depth_mm, floor_conf=None, wall_conf=None,
                        obstacle_conf=None, hazard_conf=None, cfg=None,
                        grid=None):
    """Valid depth pixels -> body points, grid cells and color samples."""
    cfg = cfg or LabelConfig()
    grid = grid or GridSpec()
    depth = np.asarray(depth_mm)
    if depth.ndim != 2:
        raise ValueError(f"{spec.role}: depth must be 2-D, got {depth.shape}")
    if depth.shape != (spec.height_d, spec.width_d):
        raise ValueError(
            f"{spec.role}: depth {depth.shape} != intrinsics "
            f"{(spec.height_d, spec.width_d)}")
    z = depth[spec.v_d, spec.u_d].astype(np.float32) * 1e-3  # uint16 mm
    self_pixel = (np.asarray(spec.self_pixels, bool).copy()
                  if spec.self_pixels is not None
                  else np.zeros(len(z), bool))
    valid = np.isfinite(z) & (z > 0.0) & ~self_pixel

    pts = spec.dir_body * z[:, None] + spec.t_body
    x, y, zz = pts[:, 0], pts[:, 1], pts[:, 2]
    body_r = np.hypot(x, y)
    # Same working volume as master_bev.prepare, plus the dead disc.  Points
    # inside the disc are the robot/chassis or stereo artifacts; the ego
    # footprint is a separate, explicit layer.
    valid &= (body_r >= cfg.min_range_m)
    valid &= ((x > -0.50)
              & (x < max(grid.range_m + 2.0, cfg.max_range_m))
              & (np.abs(y) < grid.width_m / 2.0 + 1.5)
              & (zz > -1.0) & (zz < 2.5))

    # Body-frame self mask (mast/rails).  Floor points stay because their
    # z is below the mask's min_height.
    cfg_self = (cfg.__dict__.get("self_mask") or {})
    on_body = np.zeros(len(z), bool)
    if cfg_self:
        sm = cfg_self
        on_body = ((x > sm["x0"]) & (x < sm["x1"])
                   & (y > sm["y0"]) & (y < sm["y1"])
                   & (zz > sm["min_h"]))
        valid &= ~on_body

    if not valid.any():
        n = 0
        return PreparedObservation(
            role=spec.role, points=np.zeros((0, 3), np.float32),
            ranges=np.zeros(0, np.float32), forward=np.zeros(0, np.float32),
            rows=np.zeros(0, np.int32), cols=np.zeros(0, np.int32),
            flat=np.zeros(0, np.int64), in_grid=np.zeros(0, bool),
            u_c=np.zeros(0, np.int32), v_c=np.zeros(0, np.int32),
            in_color=np.zeros(0, bool),
            floor_conf=np.zeros(0, np.float32),
            wall_conf=np.zeros(0, np.float32),
            obstacle_conf=np.zeros(0, np.float32),
            hazard_conf=np.zeros(0, np.float32))

    pts = pts[valid].astype(np.float32)
    z = z[valid]
    ranges = np.linalg.norm(spec.dir_d[valid] * z[:, None],
                            axis=1).astype(np.float32)
    x, y = pts[:, 0], pts[:, 1]

    # Color projection through the depth->color extrinsic.
    p_c = spec.dir_color[valid] * z[:, None] + spec.t_dc
    zc = p_c[:, 2]
    good = zc > 1e-6
    u_cf = np.zeros_like(zc)
    v_cf = np.zeros_like(zc)
    u_cf[good] = spec.fx_c * p_c[good, 0] / zc[good] + spec.cx_c
    v_cf[good] = spec.fy_c * p_c[good, 1] / zc[good] + spec.cy_c
    in_color = (good & (u_cf >= 0) & (u_cf < spec.width_c)
                & (v_cf >= 0) & (v_cf < spec.height_c))
    u_i = np.clip(np.rint(u_cf).astype(np.int32), 0, spec.width_c - 1)
    v_i = np.clip(np.rint(v_cf).astype(np.int32), 0, spec.height_c - 1)
    u_i[~in_color] = 0
    v_i[~in_color] = 0

    def sample(conf):
        out = np.zeros(len(pts), np.float32)
        if conf is None:
            return out
        conf = np.asarray(conf, np.float32)
        if conf.shape != (spec.height_c, spec.width_c):
            conf = cv2.resize(conf, (spec.width_c, spec.height_c),
                              interpolation=cv2.INTER_NEAREST)
        out[in_color] = conf[v_i[in_color], u_i[in_color]]
        return np.clip(out, 0.0, 1.0)

    rows = np.floor((grid.range_m - x) / grid.cell_m).astype(np.int32)
    cols = np.floor((y + grid.width_m / 2.0) / grid.cell_m).astype(np.int32)
    in_grid = ((rows >= 0) & (rows < grid.rows)
               & (cols >= 0) & (cols < grid.cols))
    flat = np.where(in_grid, rows.astype(np.int64) * grid.cols + cols, -1)

    return PreparedObservation(
        role=spec.role, points=pts, ranges=ranges, forward=x,
        rows=rows, cols=cols, flat=flat, in_grid=in_grid,
        u_c=u_i, v_c=v_i, in_color=in_color,
        floor_conf=sample(floor_conf), wall_conf=sample(wall_conf),
        obstacle_conf=sample(obstacle_conf),
        hazard_conf=sample(hazard_conf))


# ---------------------------------------------------------------------------
# Floor planes
# ---------------------------------------------------------------------------


def fit_plane(prep, cfg=None, rng=None):
    """Fit one camera's floor on a near-floor band, then refine."""
    cfg = cfg or LabelConfig()
    if prep is None or len(prep.points) == 0:
        return _fallback_plane(False)
    pts, ranges = prep.points, prep.ranges
    band = np.abs(pts[:, 2]) <= cfg.floor_band_m
    band_n = int(band.sum())
    if band_n < max(3, cfg.plane_min_inliers):
        return _fallback_plane(False)
    bp, br = pts[band], ranges[band]
    take = min(band_n, 6000)
    idx = (np.linspace(0, band_n - 1, take).astype(np.int64)
           if band_n > take else np.arange(band_n))
    fit = fit_ground_plane(
        bp[idx], residual_m=cfg.plane_residual_m,
        min_inliers=cfg.plane_min_inliers, min_frac=cfg.plane_min_frac,
        rng=rng, d_range=(-0.20, 0.20), range_scale=cfg.noise_scale,
        ranges=br[idx], normal_deg=cfg.plane_normal_deg)
    if fit is None:
        return _fallback_plane(False)
    nvec, d, cnt = fit

    h = height_above_plane(pts, nvec, d)
    gate = np.maximum(cfg.plane_residual_m, cfg.noise_scale * ranges)
    inl = np.abs(h) < gate
    if int(inl.sum()) >= max(3, cfg.plane_min_inliers):
        ip, ir = pts[inl], ranges[inl]
        n2 = len(ip)
        take2 = min(n2, 6000)
        idx2 = (np.linspace(0, n2 - 1, take2).astype(np.int64)
                if n2 > take2 else np.arange(n2))
        fit2 = fit_ground_plane(
            ip[idx2], residual_m=cfg.plane_residual_m,
            min_inliers=cfg.plane_min_inliers, min_frac=cfg.plane_min_frac,
            rng=rng, d_range=(-0.20, 0.20),
            range_scale=cfg.noise_scale, ranges=ir[idx2],
            normal_deg=cfg.plane_normal_deg)
        if fit2 is not None:
            nvec, d, cnt = fit2

    h = height_above_plane(pts, nvec, d)
    gate = np.maximum(cfg.plane_residual_m, cfg.noise_scale * ranges)
    inl = np.abs(h) < gate
    return {
        "ok": True,
        "n": [float(v) for v in nvec],
        "d": float(d),
        "inliers": int(cnt),
        "band_points": band_n,
        "inlier_frac": round(float(inl.mean()) if h.size else 0.0, 3),
        "residual_m": round(
            float(np.abs(h[inl]).mean()) if inl.any() else 0.0, 4),
        "fallback": False,
    }


def _fallback_plane(ok=False):
    return {"ok": bool(ok), "n": [0.0, 0.0, 1.0], "d": 0.0,
            "inliers": 0, "band_points": 0, "inlier_frac": 0.0,
            "residual_m": None, "fallback": True}


def fit_joint_plane(preps, cfg=None, rng=None):
    """One floor plane over every camera's points, used as a fallback."""
    cfg = cfg or LabelConfig()
    clouds, ranges = [], []
    for prep in preps.values():
        if prep is None or len(prep.points) == 0:
            continue
        n = len(prep.points)
        take = min(n, 6000)
        idx = (np.linspace(0, n - 1, take).astype(np.int64)
               if n > take else np.arange(n))
        clouds.append(prep.points[idx])
        ranges.append(prep.ranges[idx])
    if not clouds:
        return _fallback_plane(False)
    pts = np.concatenate(clouds)
    rngs = np.concatenate(ranges)
    fit = fit_ground_plane(
        pts, residual_m=cfg.plane_residual_m,
        min_inliers=cfg.plane_min_inliers, min_frac=cfg.plane_min_frac,
        rng=rng, d_range=(-0.20, 0.20), range_scale=cfg.noise_scale,
        ranges=rngs, normal_deg=cfg.plane_normal_deg)
    if fit is None:
        return _fallback_plane(False)
    nvec, d, cnt = fit
    h = height_above_plane(pts, nvec, d)
    gate = np.maximum(cfg.plane_residual_m, cfg.noise_scale * rngs)
    inl = np.abs(h) < gate
    return {
        "ok": True, "n": [float(v) for v in nvec], "d": float(d),
        "inliers": int(cnt), "band_points": int(len(pts)),
        "inlier_frac": round(float(inl.mean()) if h.size else 0.0, 3),
        "residual_m": round(
            float(np.abs(h[inl]).mean()) if inl.any() else 0.0, 4),
        "fallback": True,
    }


# ---------------------------------------------------------------------------
# Per-camera evidence
# ---------------------------------------------------------------------------


def camera_evidence(prep, plane, cfg, grid, fallback=False):
    """Weighted per-cell evidence for one camera."""
    n_cells = grid.rows * grid.cols
    out = {
        "occ_w": np.zeros(n_cells, np.float32),
        "floor_w": np.zeros(n_cells, np.float32),
        "wall_w": np.zeros(n_cells, np.float32),
        "haz_w": np.zeros(n_cells, np.float32),
        "cov_n": np.zeros(n_cells, np.int32),
        "cov_w": np.zeros(n_cells, np.float32),
        "floor_n": np.zeros(n_cells, np.int32),
        "floor_h_sum": np.zeros(n_cells, np.float32),
        "wall_n": np.zeros(n_cells, np.int32),
        "wall_h_max": np.full(n_cells, -np.inf, np.float32),
        "wall_h_min": np.full(n_cells, np.inf, np.float32),
        "cells": 0,
    }
    if prep is None or len(prep.points) == 0 or not prep.in_grid.any():
        return out
    keep = prep.in_grid
    flat = prep.flat[keep]
    pts = prep.points[keep]
    rng = prep.ranges[keep]
    x = prep.forward[keep]
    h = height_above_plane(pts, plane["n"], plane["d"])
    fconf = prep.floor_conf[keep]
    wconf = prep.wall_conf[keep]
    oconf = prep.obstacle_conf[keep]
    hconf = prep.hazard_conf[keep]

    w = range_weight(prep.role, x, cfg)
    if fallback:
        w = w * 0.5

    floor_gate = np.maximum(cfg.floor_tol_m, cfg.noise_scale * rng)
    # A stair/hole prompt wins over a floor prompt: never let a staircase
    # surface become drivable floor just because SAM also called it floor.
    floor_sam = ((fconf > 0) & (hconf <= 0) & (np.abs(h) <= floor_gate))
    floor_depth = ((fconf <= 0) & (wconf <= 0) & (oconf <= 0)
                   & (hconf <= 0) & (np.abs(h) <= floor_gate)
                   & (cfg.depth_floor_weight > 0))
    wall = ((wconf > 0) & (h >= cfg.wall_min_h_m)
            & (h <= cfg.wall_max_h_m))
    # Ascending stairs / steps are obstacles; descending ones are hazards.
    # Once the stairs/step/hole prompt fires, the surface is never floor:
    # at/above the floor plane it is an obstacle (even a flat landing), and
    # below it, hazard.  This survives small floor-plane fit variation.
    drop_gate = np.maximum(cfg.hazard_drop_m, cfg.noise_scale * rng)
    obj = ((oconf > 0) & (h >= cfg.occ_min_m) & (h <= cfg.occ_max_m))
    stair = ((hconf > 0) & (h >= -drop_gate) & (h <= cfg.occ_max_m))
    obstacle = obj | stair
    obs_conf = np.where(obj, oconf, hconf)
    hazard = h < -drop_gate
    if cfg.hazard_require_floor:
        hazard &= ((fconf > 0) | (hconf > 0))

    def add(target, mask, weights):
        if mask.any():
            target += np.bincount(flat[mask], weights=weights[mask],
                                  minlength=n_cells).astype(np.float32)

    add(out["floor_w"], floor_sam, w * np.clip(fconf, 0.05, 1.0))
    add(out["floor_w"], floor_depth, w * cfg.depth_floor_weight)
    add(out["wall_w"], wall, w * np.clip(wconf, 0.05, 1.0))
    add(out["occ_w"], obstacle, w * np.clip(obs_conf, 0.05, 1.0))
    add(out["haz_w"], hazard, w)
    out["cov_n"] = np.bincount(flat, minlength=n_cells).astype(np.int32)
    out["cov_w"] = np.bincount(flat, weights=w,
                               minlength=n_cells).astype(np.float32)
    floor = floor_sam | floor_depth
    out["floor_n"] = np.bincount(flat[floor],
                                 minlength=n_cells).astype(np.int32)
    out["floor_h_sum"] = np.bincount(
        flat[floor], weights=h[floor], minlength=n_cells).astype(np.float32)
    out["wall_n"] = np.bincount(flat[wall],
                                minlength=n_cells).astype(np.int32)
    if wall.any():
        np.maximum.at(out["wall_h_max"], flat[wall], h[wall])
        np.minimum.at(out["wall_h_min"], flat[wall], h[wall])
    out["cells"] = int(np.sum(out["cov_n"] > 0))
    return out


# ---------------------------------------------------------------------------
# Fusion / conflict resolution
# ---------------------------------------------------------------------------


def _compact_wall_fix(wall_w, obs_w, wall_h_max, wall_candidate, cfg, grid):
    """Reclassify compact, short wall components as obstacles."""
    if not wall_candidate.any():
        return wall_w, obs_w
    group = cv2.dilate(wall_candidate.astype(np.uint8),
                       np.ones((3, 3), np.uint8))
    n, labels, _stats, _ = cv2.connectedComponentsWithStats(
        group, connectivity=8)
    xc, yc = grid.centers()
    for i in range(1, n):
        comp = (labels == i) & wall_candidate
        if not comp.any():
            continue
        rr, cc = np.where(comp)
        extent_x = float(xc[rr].max() - xc[rr].min() + grid.cell_m)
        extent_y = float(yc[cc].max() - yc[cc].min() + grid.cell_m)
        hmax = float(np.max(wall_h_max[comp]))
        if (extent_x <= cfg.wall_compact_m
                and extent_y <= cfg.wall_compact_m
                and hmax < cfg.wall_pillar_min_h_m):
            moved = wall_w[comp].copy()
            wall_w[comp] = 0.0
            obs_w[comp] = np.maximum(obs_w[comp], moved)
    return wall_w, obs_w


def _neighbor_count(mask):
    """Count true 4-neighbours of each cell (no scipy dependency)."""
    m = np.asarray(mask, bool)
    out = np.zeros_like(m, np.int32)
    out[1:, :] += m[:-1, :]
    out[:-1, :] += m[1:, :]
    out[:, 1:] += m[:, :-1]
    out[:, :-1] += m[:, 1:]
    return out


def _local_drop_hazard(floor_w, floor_n, floor_h_sum, haz_w, cfg, grid):
    """Mark cells whose observed floor is >5 cm below an adjacent floor."""
    ground_n = floor_n.reshape(grid.rows, grid.cols)
    ground_sum = floor_h_sum.reshape(grid.rows, grid.cols)
    with np.errstate(invalid="ignore", divide="ignore"):
        ground = np.where(ground_n > 0, ground_sum / ground_n, np.nan)
    vote = np.zeros((grid.rows, grid.cols), np.float32)
    threshold = max(float(cfg.hazard_drop_m), 0.02)
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        shifted = np.full_like(ground, np.nan)
        shifted_n = np.zeros_like(ground_n)
        ys = slice(max(dy, 0), grid.rows + min(dy, 0))
        xs = slice(max(dx, 0), grid.cols + min(dx, 0))
        ys2 = slice(max(-dy, 0), grid.rows + min(-dy, 0))
        xs2 = slice(max(-dx, 0), grid.cols + min(-dx, 0))
        shifted[ys2, xs2] = ground[ys, xs]
        shifted_n[ys2, xs2] = ground_n[ys, xs]
        with np.errstate(invalid="ignore"):
            drop = ground - shifted
        local = ((ground_n > 0) & (shifted_n > 0) & (drop > threshold))
        v = np.where(local, np.minimum(ground_n, 2).astype(np.float32) * 0.5,
                     0.0)
        vote = np.maximum(vote, v.astype(np.float32))
    haz_w += vote.reshape(-1)
    return haz_w


def resolve_labels(floor_w, wall_w, obstacle_w, hazard_w, coverage, ego,
                   cfg=None, grid=None):
    """Apply the class priority and contract.  Pure function for tests."""
    cfg = cfg or LabelConfig()
    grid = grid or GridSpec()
    shape = (grid.rows, grid.cols)
    floor_ok = (floor_w >= cfg.floor_min_w)
    wall_ok = (wall_w >= cfg.wall_min_w)
    obs_ok = (obstacle_w >= cfg.obstacle_min_w)
    haz_ok = (hazard_w >= cfg.hazard_min_w)
    semantic = np.full(shape, IGNORE, np.uint8)
    floor_m = coverage & floor_ok
    obs_m = coverage & obs_ok
    wall_m = coverage & wall_ok
    haz_m = coverage & haz_ok
    # Lowest priority first; later assignments overwrite.
    semantic[floor_m] = FLOOR
    semantic[obs_m] = OBSTACLE
    semantic[wall_m] = WALL
    semantic[haz_m] = HAZARD
    semantic[~coverage] = IGNORE
    semantic[ego] = IGNORE
    return semantic, {
        FLOOR: floor_m,
        WALL: wall_m,
        OBSTACLE: obs_m,
        HAZARD: haz_m,
    }


def fuse_camera_evidence(evidences, cfg, grid, ego=None, planes=None):
    """Sum camera evidence, fix compact walls, add drop hazards, resolve."""
    n_cells = grid.rows * grid.cols
    floor_w = np.zeros(n_cells, np.float32)
    wall_w = np.zeros(n_cells, np.float32)
    obs_w = np.zeros(n_cells, np.float32)
    haz_w = np.zeros(n_cells, np.float32)
    cov_n = np.zeros(n_cells, np.int32)
    cov_w = np.zeros(n_cells, np.float32)
    floor_n = np.zeros(n_cells, np.int32)
    floor_h_sum = np.zeros(n_cells, np.float32)
    wall_n = np.zeros(n_cells, np.int32)
    wall_h_max = np.full(n_cells, -np.inf, np.float32)
    for ev in evidences.values():
        if ev is None:
            continue
        floor_w += ev["floor_w"]
        wall_w += ev["wall_w"]
        obs_w += ev["occ_w"]
        haz_w += ev["haz_w"]
        cov_n += ev["cov_n"]
        cov_w += ev["cov_w"]
        floor_n += ev["floor_n"]
        floor_h_sum += ev["floor_h_sum"]
        wall_n += ev["wall_n"]
        np.maximum(wall_h_max, ev["wall_h_max"], out=wall_h_max)

    coverage = (cov_n > 0).reshape(grid.rows, grid.cols)
    if ego is None:
        ego = ego_mask(cfg, grid)
    ego = ego.reshape(grid.rows, grid.cols)
    coverage = coverage & ~ego

    wall_w2 = wall_w.reshape(grid.rows, grid.cols).copy()
    obs_w2 = obs_w.reshape(grid.rows, grid.cols).copy()
    wall_h_max2 = wall_h_max.reshape(grid.rows, grid.cols)
    wall_candidate = ((wall_w2 >= cfg.wall_min_w) & coverage)
    wall_w2, obs_w2 = _compact_wall_fix(
        wall_w2, obs_w2, wall_h_max2, wall_candidate, cfg, grid)

    floor2 = floor_w.reshape(grid.rows, grid.cols)
    haz_below2 = haz_w.reshape(grid.rows, grid.cols).copy()
    if cfg.local_drop_hazard:
        haz_drop2 = _local_drop_hazard(
            floor2, floor_n, floor_h_sum,
            np.zeros(grid.rows * grid.cols, np.float32), cfg,
            grid).reshape(grid.rows, grid.cols)
    else:
        haz_drop2 = np.zeros((grid.rows, grid.cols), np.float32)
    # Points below the plane under furniture are not drop hazards.  Keep
    # them only when the cell is not adjacent to an obstacle (a sofa/table
    # underside); a step down / hole has no obstacle ring, while a piece of
    # furniture does.  Local floor-height drops are always kept.
    obs_ok = (obs_w2 >= cfg.obstacle_min_w) & coverage
    obs_nb = _neighbor_count(obs_ok)
    below_gate = ((haz_below2 >= cfg.hazard_min_w)
                  & (obs_nb == 0) & coverage)
    haz_w2 = haz_drop2 + np.where(below_gate, haz_below2, 0.0)

    semantic, masks = resolve_labels(
        floor2, wall_w2, obs_w2, haz_w2,
        coverage, ego, cfg, grid)
    evidence = {
        "floor_w": floor2,
        "wall_w": wall_w2,
        "obstacle_w": obs_w2,
        "hazard_w": haz_w2,
        "hazard_below_w": haz_below2,
        "hazard_drop_w": haz_drop2,
        "cov_n": cov_n.reshape(grid.rows, grid.cols),
        "cov_w": cov_w.reshape(grid.rows, grid.cols),
        "wall_h_max": wall_h_max2,
    }
    return LabelArrays(
        semantic=semantic, ego=ego, coverage=coverage,
        evidence=evidence, planes=planes or {},
        stats=class_counts(semantic, ego, coverage))


def class_counts(semantic, ego, coverage):
    cells = int(semantic.size)
    counts = {name: int(np.sum(semantic == cls))
              for cls, name in CLASS_NAMES.items()}
    counts["ego"] = int(np.sum(ego))
    counts["covered"] = int(np.sum(coverage))
    counts["cells"] = cells
    return counts


def ego_mask(cfg, grid):
    xc, yc = grid.centers()
    x0, x1 = cfg.ego_x_range_m
    y0, y1 = cfg.ego_y_range_m
    return ((xc[:, None] >= x0) & (xc[:, None] <= x1)
            & (yc[None, :] >= y0) & (yc[None, :] <= y1))


class BevStudentLabeler:
    """SAM category maps x depth -> one semantic BEV label."""

    def __init__(self, rig_cfg=None, config_dir=None, roles=None,
                 seed=0):
        self.specs, self.cfg, self.grid, self.label_cfg = specs_from_config(
            rig_cfg, config_dir, roles)
        self.roles = sorted(self.specs)
        self.ego = ego_mask(self.label_cfg, self.grid)
        self._rng = np.random.default_rng(seed)

    def label(self, depth_by_role, masks_by_role=None):
        """Label one keyframe.

        ``depth_by_role``: role -> uint16 mm depth image.
        ``masks_by_role``: role -> {class: float confidence map} in color
        resolution.  Missing roles/classes are treated as no SAM evidence.
        """
        masks_by_role = masks_by_role or {}
        preps = {}
        for role in self.roles:
            depth = depth_by_role.get(role)
            if depth is None:
                continue
            maps = masks_by_role.get(role) or {}
            preps[role] = prepare_observation(
                self.specs[role], depth,
                floor_conf=maps.get(FLOOR),
                wall_conf=maps.get(WALL),
                obstacle_conf=maps.get(OBSTACLE),
                hazard_conf=maps.get(HAZARD),
                cfg=self.label_cfg, grid=self.grid)
        if not preps:
            return None

        joint = fit_joint_plane(preps, self.label_cfg, self._rng)
        planes = {}
        evidences = {}
        for role, prep in preps.items():
            plane = fit_plane(prep, self.label_cfg, self._rng)
            fallback = False
            if not plane["ok"]:
                if joint["ok"]:
                    plane = dict(joint)
                    plane["fallback"] = True
                    fallback = True
                else:
                    plane = _fallback_plane(False)
                    fallback = True
            planes[role] = {**plane, "role": role}
            evidences[role] = camera_evidence(
                prep, plane, self.label_cfg, self.grid, fallback=fallback)
        out = fuse_camera_evidence(
            evidences, self.label_cfg, self.grid, ego=self.ego, planes=planes)
        out.stats["plane_ok"] = {r: bool(planes[r]["ok"]) for r in planes}
        out.stats["plane_fallback"] = {
            r: bool(planes[r].get("fallback")) for r in planes}
        out.stats["roles"] = sorted(preps)
        return out


# ---------------------------------------------------------------------------
# Review images / stats
# ---------------------------------------------------------------------------


def label_stats(semantic, ego, coverage):
    counts = class_counts(np.asarray(semantic), np.asarray(ego),
                          np.asarray(coverage))
    cells = max(1, counts["cells"])
    return {
        "cells": counts["cells"],
        "class_cells": {k: counts[k] for k in
                        ("floor", "wall", "obstacle", "hazard", "ignore")},
        "class_fraction": {k: round(counts[k] / cells, 4) for k in
                           ("floor", "wall", "obstacle", "hazard", "ignore")},
        "ego_cells": counts["ego"],
        "covered_cells": counts["covered"],
        "ego_fraction": round(counts["ego"] / cells, 4),
        "coverage_fraction": round(counts["covered"] / cells, 4),
    }


def draw_legend(img, items, y=None):
    """Draw small colored squares + labels along the bottom of ``img``."""
    out = img
    if y is None:
        out = np.vstack([out, np.zeros((18, out.shape[1], 3), np.uint8)])
        y = out.shape[0] - 5
    x = 4
    for label, color in items:
        cv2.rectangle(out, (x, y - 9), (x + 8, y), color, -1)
        cv2.putText(out, label, (x + 11, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.32, (215, 215, 215), 1, cv2.LINE_AA)
        x += 11 + 7 * len(label) + 10
    return out


def render_label_bev(semantic, ego, coverage, grid, *, scale=4,
                     show_hud=True):
    """Semantic (R,C) -> BGR review image, mirrored like render_bev."""
    semantic = np.asarray(semantic)
    small = np.zeros((grid.rows, grid.cols, 3), np.uint8)
    for cls, color in LABEL_COLORS.items():
        small[semantic == cls] = color
    if ego is not None and np.asarray(ego).any():
        ego_u8 = np.asarray(ego, np.uint8)
        small[ego_u8 > 0] = EGO_FILL
        contours, _ = cv2.findContours(ego_u8, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(small, contours, -1, EGO_COLOR, 1)
    img = np.repeat(np.repeat(small, scale, axis=0), scale, axis=1)
    img = np.ascontiguousarray(img[:, ::-1])
    if show_hud:
        cells = semantic.size
        vals = {name: int(np.sum(semantic == cls))
                for cls, name in CLASS_NAMES.items()}
        bar = 18
        cv2.rectangle(img, (0, 0), (img.shape[1], bar), (0, 0, 0), -1)
        text = (f"BEV labels {grid.range_m:.1f}x{grid.width_m:.1f}m "
                f"{grid.cell_m*100:.0f}cm | floor "
                f"{100*vals['floor']/cells:.0f}% wall "
                f"{100*vals['wall']/cells:.0f}% obs "
                f"{100*vals['obstacle']/cells:.0f}% haz "
                f"{100*vals['hazard']/cells:.0f}% ign "
                f"{100*vals['ignore']/cells:.0f}%")
        cv2.putText(img, text, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.34,
                    (220, 220, 220), 1, cv2.LINE_AA)
        img = draw_legend(img, [
            ("floor", LABEL_COLORS[FLOOR]),
            ("wall", LABEL_COLORS[WALL]),
            ("obstacle", LABEL_COLORS[OBSTACLE]),
            ("hazard", LABEL_COLORS[HAZARD]),
            ("unknown", LABEL_COLORS[IGNORE]),
            ("ego", EGO_COLOR),
        ])
    return img


def render_color_masks(images, masks_by_role, *, max_width=640):
    """Color frame(s) with SAM category masks overlaid, side by side.

    ``images``: role -> BGR image.
    ``masks_by_role``: role -> {class: float/bool map}.
    """
    tiles = []
    for role in ("near", "far"):
        img = images.get(role)
        if img is None:
            continue
        img = np.asarray(img)
        h, w = img.shape[:2]
        scale = min(1.0, max_width / max(w, 1))
        if scale < 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_AREA)
        overlay = img.copy()
        maps = masks_by_role.get(role) or {}
        for cls, color in ((FLOOR, (60, 220, 60)),
                           (WALL, (235, 120, 40)),
                           (OBSTACLE, (40, 40, 220))):
            m = maps.get(cls)
            if m is None:
                continue
            m = np.asarray(m) > 0
            if m.shape != img.shape[:2]:
                m = cv2.resize(m.astype(np.uint8),
                               (img.shape[1], img.shape[0]),
                               interpolation=cv2.INTER_NEAREST) > 0
            overlay[m] = color
        img = cv2.addWeighted(img, 0.62, overlay, 0.38, 0)
        cv2.putText(img, role, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(img)
    if not tiles:
        return np.zeros((1, 1, 3), np.uint8)
    h = max(t.shape[0] for t in tiles)
    out = []
    for t in tiles:
        if t.shape[0] != h:
            t = cv2.resize(t, (int(t.shape[1] * h / t.shape[0]), h),
                           interpolation=cv2.INTER_AREA)
        out.append(t)
        out.append(np.zeros((h, 6, 3), np.uint8))
    img = np.concatenate(out[:-1], axis=1)
    return draw_legend(img, [
        ("floor", (60, 220, 60)),
        ("wall", (235, 120, 40)),
        ("obstacle", (40, 40, 220)),
    ])


def contact_sheet(images, cols=4, pad=4):
    """Grid of BGR tiles (contact sheet)."""
    if not images:
        return np.zeros((1, 1, 3), np.uint8)
    rows = int(np.ceil(len(images) / cols))
    h = max(i.shape[0] for i in images)
    w = max(i.shape[1] for i in images)
    sheet = np.zeros((rows * (h + pad) + pad, cols * (w + pad) + pad, 3),
                     np.uint8)
    for k, img in enumerate(images):
        r, c = divmod(k, cols)
        y = pad + r * (h + pad)
        x = pad + c * (w + pad)
        sheet[y:y + img.shape[0], x:x + img.shape[1]] = img
    return sheet


def save_label_npz(path, label):
    np.savez_compressed(path, semantic=label.semantic.astype(np.uint8),
                        ego=label.ego.astype(bool),
                        coverage=label.coverage.astype(bool))


def load_label_npz(path, grid=None):
    grid = grid or GridSpec()
    with np.load(path) as d:
        semantic = d["semantic"]
        ego = d["ego"]
        coverage = d["coverage"]
    validate_label_arrays(semantic, ego, coverage, grid)
    return semantic, ego, coverage


def validate_label_arrays(semantic, ego, coverage, grid=None):
    grid = grid or GridSpec()
    errors = []
    if semantic.shape != grid.shape:
        errors.append(f"semantic shape {semantic.shape} != {grid.shape}")
    if ego.shape != grid.shape:
        errors.append(f"ego shape {ego.shape} != {grid.shape}")
    if coverage.shape != grid.shape:
        errors.append(f"coverage shape {coverage.shape} != {grid.shape}")
    if semantic.dtype != np.uint8:
        errors.append(f"semantic dtype {semantic.dtype} != uint8")
    if ego.dtype != np.bool_:
        errors.append(f"ego dtype {ego.dtype} != bool")
    if coverage.dtype != np.bool_:
        errors.append(f"coverage dtype {coverage.dtype} != bool")
    if errors:
        raise ValueError("; ".join(errors))
    bad = set(np.unique(semantic).tolist()) - set(CLASS_NAMES)
    if bad:
        raise ValueError(f"semantic has invalid ids {sorted(bad)}")
    if np.any(semantic[ego] != IGNORE):
        raise ValueError("ego cells must be semantic=255")
    if np.any(semantic[~coverage] != IGNORE):
        raise ValueError("unobserved cells must be semantic=255")
    return True


def summarize_label_files(paths, grid=None):
    """Aggregate semantic/ego/coverage stats over label npz files."""
    grid = grid or GridSpec()
    counts = {name: 0 for name in CLASS_NAMES.values()}
    ego_cells = covered = cells = 0
    for p in paths:
        semantic, ego, coverage = load_label_npz(p, grid)
        for cls, name in CLASS_NAMES.items():
            counts[name] += int(np.sum(semantic == cls))
        ego_cells += int(np.sum(ego))
        covered += int(np.sum(coverage))
        cells += int(semantic.size)
    cells = max(1, cells)
    return {
        "frames": len(paths),
        "cells": cells,
        "class_cells": counts,
        "class_fraction": {k: round(v / cells, 4)
                           for k, v in counts.items()},
        "ego_fraction": round(ego_cells / cells, 4),
        "coverage_fraction": round(covered / cells, 4),
    }
