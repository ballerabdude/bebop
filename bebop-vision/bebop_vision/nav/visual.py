"""Visual heading from patch-feature matching (teach-and-repeat).

Given DINOv3 patch grids of the live forward view and of a taught keyframe,
mutual-nearest-neighbour matches give a horizontal column offset per match.
The robust (median) offset says where the taught scene sits in the live
frame: scene to the right means turn right.  Converted through the color
intrinsics this is a bearing correction the bearing controller can follow;
distance along the route stays with odometry.  Pure numpy, no camera, no
robot.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np


@dataclasses.dataclass
class MatchResult:
    n_matches: int
    n_ref: int
    inlier_frac: float          # matches / reference patches
    median_dcol: float          # live col - ref col (grid cells), + = right
    mad_dcol: float             # spread of the match offsets
    median_drow: float
    mean_sim: float
    bearing_rad: float          # correction: + = turn left
    confident: bool

    def to_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in dataclasses.asdict(self).items()}


@dataclasses.dataclass
class MatchConfig:
    # Measured on the Thor (DINOv3 ViT-L/16, 14x14 grid): the same view
    # shifted by up to 270 px keeps 67-86 % of patches as mutual matches;
    # an unrelated view still produced 16 % at a 0.55 cosine floor, so the
    # fraction gate is what separates "this scene" from "any carpet".
    min_sim: float = 0.60          # cosine floor for a match
    min_matches: int = 20
    min_inlier_frac: float = 0.30
    max_mad_cols: float = 2.5       # spread above this = ambiguous
    max_bearing_deg: float = 50.0
    image_width_px: float = 1280.0  # far color width
    fx_px: float = 610.5            # far color focal length


def match_patches(live: np.ndarray, ref: np.ndarray,
                  config: MatchConfig | None = None) -> MatchResult:
    """Match two [gh, gw, D] L2-normalised patch grids."""
    c = config or MatchConfig()
    gh, gw, d = ref.shape
    if live.shape != ref.shape:
        raise ValueError(f"grid shapes differ: {live.shape} vs {ref.shape}")
    a = live.reshape(-1, d)
    b = ref.reshape(-1, d)
    sim = a @ b.T                                   # [n_live, n_ref]
    best_for_live = np.argmax(sim, axis=1)
    best_for_ref = np.argmax(sim, axis=0)
    live_idx = np.arange(a.shape[0])
    mutual = best_for_ref[best_for_live] == live_idx
    s = sim[live_idx, best_for_live]
    ok = mutual & (s >= c.min_sim)
    n = int(np.count_nonzero(ok))
    n_ref = b.shape[0]
    if n == 0:
        return MatchResult(0, n_ref, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False)
    li = live_idx[ok]
    ri = best_for_live[ok]
    dcol = (li % gw) - (ri % gw)
    drow = (li // gw) - (ri // gw)
    med = float(np.median(dcol))
    mad = float(np.median(np.abs(dcol - med)))
    px = med * (c.image_width_px / gw)
    bearing = -math.atan2(px, c.fx_px)
    frac = n / n_ref
    confident = (n >= c.min_matches and frac >= c.min_inlier_frac
                 and mad <= c.max_mad_cols
                 and abs(bearing) <= math.radians(c.max_bearing_deg))
    return MatchResult(n, n_ref, float(frac), med, mad,
                       float(np.median(drow)), float(np.mean(s[ok])),
                       float(bearing), bool(confident))
