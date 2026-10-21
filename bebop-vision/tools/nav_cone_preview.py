#!/usr/bin/env python3
"""Offline check of the depth stop-cone on recorded depth PNGs.

    .venv/bin/python tools/nav_cone_preview.py --near near.png [--far far.png] \
        --out cone.png

Prints the body-frame height statistics (a well-calibrated rig puts the
floor at z ~ 0), the chassis footprint hits, and the clearance map, then
renders a top-down view: floor (grey), obstacles (red), drops (blue),
chassis footprint (box), sector clearances (green fan).
"""
from __future__ import annotations

import argparse
import math
import time
from types import SimpleNamespace

import cv2
import numpy as np

from bebop_vision.nav import (CameraGeometry, ConeConfig, DepthCone,
                              footprint_from_rig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--near")
    ap.add_argument("--far")
    ap.add_argument("--out", default="cone.png")
    ap.add_argument("--rig", default=None)
    ap.add_argument("--scale", type=float, default=150.0, help="px per m")
    args = ap.parse_args()

    roles = {}
    if args.near:
        roles["near"] = args.near
    if args.far:
        roles["far"] = args.far
    if not roles:
        ap.error("give --near and/or --far")
    geoms = [CameraGeometry.from_rig(r, rig_path=args.rig) for r in roles]
    cfg = ConeConfig(footprint_m=tuple(footprint_from_rig(rig_path=args.rig)))
    cone = DepthCone(geoms, cfg)

    frames = {}
    for role, path in roles.items():
        d = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if d is None or d.dtype != np.uint16:
            raise SystemExit(f"{path}: not a uint16 depth PNG")
        frames[role] = SimpleNamespace(depth=d, depth_scale_mm=1.0,
                                       stamp_us=0, recv_ts=time.monotonic())
        pts, frac = cone.points(role, d, 1.0)
        z = pts[:, 2]
        near_floor = np.abs(z) < 0.15
        print(f"[{role}] sampled {len(pts)} pts valid={frac:.2f} "
              f"z p05/p50/p95 = {np.percentile(z, 5):.3f}/"
              f"{np.median(z):.3f}/{np.percentile(z, 95):.3f} m; "
              f"|z|<0.15: {near_floor.mean()*100:.1f}% "
              f"(median z of those {np.median(z[near_floor]) if near_floor.any() else float('nan'):+.3f} m)")
        fx0, fx1, fy0, fy1 = cfg.footprint_m
        in_fp = ((pts[:, 0] >= fx0) & (pts[:, 0] <= fx1)
                 & (pts[:, 1] >= fy0) & (pts[:, 1] <= fy1))
        if in_fp.any():
            print(f"[{role}] footprint box holds {in_fp.sum()} pts, "
                  f"z p50={np.median(z[in_fp]):.2f} max={z[in_fp].max():.2f}")
    t0 = time.perf_counter()
    for _ in range(20):
        cm = cone.evaluate(frames)
    dt = (time.perf_counter() - t0) / 20
    print(f"evaluate: {dt*1000:.2f} ms/tick")
    print("clearance:", {k: v for k, v in cm.to_dict().items()
                         if k not in ("bearings_deg", "obstacle_m",
                                      "observed_m")})
    for b, c, o, f in zip(cm.bearings_rad, cm.clearance_m, cm.obstacle_m,
                          cm.observed_m):
        print(f"  {math.degrees(b):+6.1f} deg  clear {c:.2f}  "
              f"obst {'  -- ' if not math.isfinite(o) else f'{o:.2f}'}  "
              f"floor {f:.2f}")

    # --- render top-down ------------------------------------------------
    s = args.scale
    xr, yr = (-0.8, 4.0), (-2.0, 2.0)
    W = int((yr[1] - yr[0]) * s)
    H = int((xr[1] - xr[0]) * s)
    img = np.full((H, W, 3), 30, np.uint8)

    def px(x, y):
        return (int((yr[1] - y) * s), int((xr[1] - x) * s))

    for role in roles:
        pts, _ = cone.points(role, frames[role].depth, 1.0)
        z = pts[:, 2]
        cls = np.zeros(len(pts), np.uint8)
        cls[np.abs(z) <= cfg.floor_tol_m] = 1
        cls[(z >= cfg.obstacle_min_z_m) & (z <= cfg.obstacle_max_z_m)] = 2
        cls[z <= cfg.drop_z_m] = 3
        colors = {0: (70, 70, 70), 1: (150, 150, 150), 2: (60, 60, 230),
                  3: (230, 120, 60)}
        for k, col in colors.items():
            for x, y in pts[cls == k][:, :2]:
                u, v = px(x, y)
                if 0 <= u < W and 0 <= v < H:
                    img[v, u] = col
    fx0, fx1, fy0, fy1 = cfg.footprint_m
    cv2.rectangle(img, px(fx1, fy1), px(fx0, fy0), (0, 200, 255), 1)
    for b, c in zip(cm.bearings_rad, cm.clearance_m):
        u, v = px(c * math.cos(b), c * math.sin(b))
        cv2.line(img, px(0, 0), (u, v), (0, 200, 0), 1)
        cv2.circle(img, (u, v), 3, (0, 255, 0), -1)
    for r in (1.0, 2.0, 3.0):
        cv2.circle(img, px(0, 0), int(r * s), (60, 60, 60), 1)
    cv2.putText(img, "x fwd ^  y left <-", (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (200, 200, 200), 1)
    cv2.imwrite(args.out, img)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
