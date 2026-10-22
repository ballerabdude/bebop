#!/usr/bin/env python3
"""Measure near-camera depth quality under SDK presets and filter sets.

Stop the vision service first (the camera is exclusive), then:

    sudo systemctl stop bebop-vision
    .venv/bin/python tools/orbbec_depth_tune.py --role near --seconds 2.5
    sudo systemctl start bebop-vision

For each (preset, filters) it captures frames with the robot still and
scores what the stop-cone cares about:

    valid%        fraction of depth pixels that are non-zero
    fps           achieved frame rate through the filter chain
    floor_std_mm  median per-pixel temporal std over floor pixels, per range band
    flicker_obst  per-frame count of floor pixels (floor in the median frame)
                  that read as an obstacle (>= 0.10 m) within 1.2 m — phantom
                  obstacles the driver would brake for
    flicker_drop  the same for drops (<= -0.12 m)
    edge_obst     obstacle-class points in the 0.56-0.8 m corridor band

Lower flicker and std are better; valid% higher is better.  The floor is
found with the nav cone's own plane fit so the numbers match the driver.
"""
from __future__ import annotations

import argparse
import itertools
import time

import numpy as np

from bebop_vision.nav import CameraGeometry, ConeConfig, DepthCone, footprint_from_rig
from bebop_vision.orbbec import OrbbecCamera, load_rig_config

PRESETS = ["High Accuracy", "High Density", "Default"]
FILTER_SETS = [[], ["hole"], ["temporal", "hole"], ["noise", "temporal", "hole"],
               ["edge", "temporal", "hole"], ["spatial", "temporal", "hole"],
               ["noise", "spatial", "temporal", "hole"]]


def capture(serial, role, preset, filters, seconds, warmup=2.0,
            depth_range_mm=None, depth_precision_mm=None, noise_params=None):
    # The camera opens (and starts its capture thread) in the constructor.
    cam = OrbbecCamera(serial, role, depth_profile=(848, 480, 30),
                       color_profile=None, depth_preset=preset,
                       depth_filters=filters, imu=False,
                       depth_range_mm=depth_range_mm,
                       depth_precision_mm=depth_precision_mm,
                       depth_noise_params=noise_params)
    frames = []
    try:
        t0 = time.monotonic()
        last = None
        while time.monotonic() - t0 < warmup + seconds:
            fr = cam.read()
            if fr is None or fr.depth is None or fr.stamp_us == last:
                time.sleep(0.005)
                continue
            last = fr.stamp_us
            if time.monotonic() - t0 >= warmup:
                scale = float(fr.depth_scale_mm or 1.0)
                frames.append((fr.depth.astype(np.float32) * scale).astype(np.uint16)
                              if scale != 1.0 else fr.depth.copy())
    finally:
        cam.stop()
        time.sleep(1.0)            # let the USB device settle before reopening
    return frames, seconds


def score(frames, seconds, cone, role):
    g = cone.geometries[role]
    dirs, u, v, _n = cone._rays[role]
    stack = np.stack(frames)                              # [N, H, W]
    valid = (stack > 0).mean()
    med = np.median(stack, axis=0).astype(np.uint16)
    pts, _ = cone.points(role, med, 1.0)
    h = cone.fit_floor(role, pts)
    x, y = pts[:, 0], pts[:, 1]
    rng = np.hypot(x, y)
    ok_med = (med[v, u] >= 150) & (med[v, u] <= 5000)
    # Per-sample indices into the strided grid that are floor in the median.
    floor_idx = np.flatnonzero(ok_med)[np.abs(h) <= 0.06]
    floor_idx = floor_idx[(x[np.abs(h) <= 0.06] < 1.7) & (np.abs(y[np.abs(h) <= 0.06]) < 0.6)]
    fu, fv = u[floor_idx], v[floor_idx]
    d = stack[:, fv, fu].astype(np.float32)               # [N, F]
    d[d == 0] = np.nan
    std = np.nanstd(d, axis=0)
    frange = np.hypot(x[np.isin(np.flatnonzero(ok_med), floor_idx)][:0], 0)  # placeholder
    # range per floor sample
    sample_rng = {}
    rng_of = np.full(u.shape[0], np.nan, np.float32)
    rng_of[np.flatnonzero(ok_med)] = rng
    fr_rng = rng_of[floor_idx]
    bands = {"0.5-1.0": (0.5, 1.0), "1.0-1.7": (1.0, 1.7)}
    std_b = {k: float(np.nanmedian(std[(fr_rng >= a) & (fr_rng < b)])) if np.any((fr_rng >= a) & (fr_rng < b)) else float("nan")
             for k, (a, b) in bands.items()}
    # Flicker: per frame, floor samples that read obstacle / drop.
    flick_o, flick_d, edge_o = [], [], []
    for f in frames:
        p, _ = cone.points(role, f, 1.0)
        hh = p @ cone._floor[role][0] + cone._floor[role][1]
        okf = (f[v, u] >= 150) & (f[v, u] <= 5000)
        idx_all = np.flatnonzero(okf)
        # map floor_idx into this frame's samples
        hmap = np.full(u.shape[0], np.nan, np.float32)
        hmap[idx_all] = hh
        xmap = np.full(u.shape[0], np.nan, np.float32); xmap[idx_all] = p[:, 0]
        ymap = np.full(u.shape[0], np.nan, np.float32); ymap[idx_all] = p[:, 1]
        hf = hmap[floor_idx]; xf = xmap[floor_idx]
        near = xf < 1.2
        flick_o.append(int(np.sum((hf >= 0.10) & (hf <= 1.8) & near)))
        flick_d.append(int(np.sum((hf <= -0.12 - 0.03 * np.hypot(xf, ymap[floor_idx])) & near)))
        band = (xmap >= 0.56) & (xmap <= 0.8) & (np.abs(ymap) <= 0.42)
        edge_o.append(int(np.sum(band & (hmap >= 0.10) & (hmap <= 1.8))))
    return {
        "frames": len(frames), "fps": round(len(frames) / seconds, 1),
        "valid%": round(100 * float(valid), 1),
        "floor_px": int(floor_idx.size),
        "std_0.5-1.0_mm": round(std_b["0.5-1.0"] * 1, 1),
        "std_1.0-1.7_mm": round(std_b["1.0-1.7"] * 1, 1),
        "flicker_obst_mean": round(float(np.mean(flick_o)), 1),
        "flicker_obst_max": int(np.max(flick_o)),
        "flicker_drop_mean": round(float(np.mean(flick_d)), 1),
        "edge_obst_mean": round(float(np.mean(edge_o)), 1),
        "edge_obst_max": int(np.max(edge_o)),
        "fit": cone.floor_report()[role].get("source"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", default="near")
    ap.add_argument("--seconds", type=float, default=2.5)
    ap.add_argument("--presets", default=",".join(PRESETS))
    ap.add_argument("--quick", action="store_true", help="fewer filter sets")
    ap.add_argument("--range", default=None,
                    help="device depth clip min,max in mm, e.g. 150,2500")
    ap.add_argument("--precision", type=float, default=None,
                    help="device depth precision mm/unit: 1.0 0.8 0.4 0.2 0.1")
    ap.add_argument("--speckle", type=int, default=None,
                    help="device noise removal max speckle size (default 80)")
    ap.add_argument("--maxdiff", type=int, default=None,
                    help="device noise removal max diff (default 256)")
    ap.add_argument("--filters", default=None,
                    help="semicolon-separated filter sets, e.g. "
                         "'temporal;noise;noise+temporal' (overrides the defaults)")
    args = ap.parse_args()
    cfg = load_rig_config()["robots"]["default"]["cameras"]
    serial = next(s for s, c in cfg.items() if c["role"] == args.role)
    geom = CameraGeometry.from_rig(args.role)
    fsets = FILTER_SETS[:3] if args.quick else FILTER_SETS
    if args.filters:
        fsets = [[f for f in fs.split("+") if f and f != "none"]
                 for fs in args.filters.split(";")]
    rows = []
    for preset, filters in itertools.product(args.presets.split(","), fsets):
        cone = DepthCone([geom], ConeConfig(footprint_m=tuple(footprint_from_rig()),
                                             stride=2))
        try:
            rng = (tuple(int(v) for v in args.range.split(",")) if args.range
                   else None)
            noise = {k: v for k, v in (("max_speckle_size", args.speckle),
                                       ("max_diff", args.maxdiff)) if v is not None}
            frames, secs = capture(serial, args.role, preset.strip(), filters,
                                   args.seconds, depth_range_mm=rng,
                                   depth_precision_mm=args.precision,
                                   noise_params=noise or None)
            if len(frames) < 5:
                raise RuntimeError(f"only {len(frames)} frames")
            r = score(frames, secs, cone, args.role)
        except Exception as exc:  # noqa: BLE001
            r = {"error": f"{type(exc).__name__}: {exc}"[:80]}
        r = {"preset": preset.strip(), "filters": "+".join(filters) or "none",
             "range": args.range or "default", "precision": args.precision or "default",
             "speckle": args.speckle or "default", "maxdiff": args.maxdiff or "default", **r}
        rows.append(r)
        print(" ".join(f"{k}={v}" for k, v in r.items()), flush=True)
    print("\n== summary (sorted by flicker_obst_mean, then std 0.5-1.0)")
    good = [r for r in rows if "error" not in r]
    for r in sorted(good, key=lambda r: (r["flicker_obst_mean"], r["std_0.5-1.0_mm"])):
        print(f"{r['preset']:14s} {r['filters']:28s} valid={r['valid%']:5.1f} fps={r['fps']:4.1f} "
              f"std={r['std_0.5-1.0_mm']:5.1f}/{r['std_1.0-1.7_mm']:5.1f}mm "
              f"flickObst={r['flicker_obst_mean']:5.1f}(max {r['flicker_obst_max']}) "
              f"flickDrop={r['flicker_drop_mean']:4.1f} edge={r['edge_obst_mean']:4.1f}(max {r['edge_obst_max']})")


if __name__ == "__main__":
    main()
