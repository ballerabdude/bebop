"""Learn the chassis (ego) footprint from the robot's own recordings.

    .venv/bin/python tools/bev_ego_from_data.py --out config/ego_mask.npz

Self-supervised: the robot's own parts sit at the same body-frame
coordinates in every frame while the scene changes.  Over many frames,
cells near the origin that hold above-floor depth in nearly every frame are
chassis (``p_body``); cells the cameras never observe right beside that core
(``p_obs`` ~ 0) are occluded by the body -- wheels, underbody -- and belong to
it too.  The convex hull of both is the footprint.  No tape measure, no
hand-entered rectangle: the mask is a fine body-frame raster (1 cm) that any
BEV grid rasterises (``master_bev.rasterize_ego``).
"""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
import sys
import time

for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_key] = "1"

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from bebop_vision.bev_student_data import DEFAULT_ROOT, read_index, session_dirs
from bebop_vision.bev_student_labels import specs_from_config
from bebop_vision.orbbec import CONFIG_DIR, load_rig_config


def parse_range(text):
    lo, hi = (float(v) for v in str(text).split(","))
    if lo >= hi:
        raise argparse.ArgumentTypeError("range must be lo,hi with lo<hi")
    return lo, hi


def full_res_specs(roles=("near", "far")):
    """Camera specs at depth stride 1 and WITHOUT the pixel self mask.

    The self mask hides the robot's own parts from the labeler; here they
    are exactly the signal.
    """
    cfg = copy.deepcopy(load_rig_config())
    for cam in cfg["robots"]["default"]["cameras"].values():
        cam["label_stride_x"] = 1
        cam["label_stride_y"] = 1
        cam.pop("self_mask_pixels", None)
    specs, _, _, _ = specs_from_config(cfg, roles=roles)
    return specs


def accumulate(frames, specs, x_range, y_range, cell, *, min_z_m=0.15,
               body_h=(0.03, 1.2), log_every=100):
    """Per-cell observation and above-floor persistence over ``frames``.

    ``frames``: iterable of (session_dir, stamp_ns).  Returns
    (p_obs, p_body, n_frames) on the raster [nx, ny] with index 0 at
    (x_range[0], y_range[0]).
    """
    x0, x1 = x_range
    y0, y1 = y_range
    nx = int(round((x1 - x0) / cell))
    ny = int(round((y1 - y0) / cell))
    obs = np.zeros((nx, ny), np.int32)
    body = np.zeros((nx, ny), np.int32)
    n = 0
    started = time.monotonic()
    for session, stamp in frames:
        obs_f = np.zeros((nx, ny), bool)
        body_f = np.zeros((nx, ny), bool)
        seen = False
        for role, spec in specs.items():
            depth = cv2.imread(str(session / f"depth_{role}" / f"{stamp:020d}.png"),
                               cv2.IMREAD_UNCHANGED)
            if depth is None or depth.shape != (spec.height_d, spec.width_d):
                continue
            seen = True
            z = depth[spec.v_d, spec.u_d].astype(np.float32) * 1e-3
            ok = z > min_z_m
            pts = spec.dir_body[ok] * z[ok, None] + spec.t_body
            x, y, h = pts[:, 0], pts[:, 1], pts[:, 2]
            inb = (x >= x0) & (x < x1) & (y >= y0) & (y < y1) & (h > -0.2) & (h < 1.5)
            ix = ((x[inb] - x0) / cell).astype(np.int32)
            iy = ((y[inb] - y0) / cell).astype(np.int32)
            obs_f[ix, iy] = True
            above = (h[inb] > body_h[0]) & (h[inb] < body_h[1])
            body_f[ix[above], iy[above]] = True
        if not seen:
            continue
        obs += obs_f
        body += body_f
        n += 1
        if log_every and n % log_every == 0:
            print(f"  {n} frames, {(time.monotonic() - started) / n:.3f} s/frame",
                  flush=True)
    if n == 0:
        raise ValueError("no depth frames found")
    return obs / n, body / n, n


def derive_ego_mask(p_obs, p_body, cell, *, persist=0.85, unobserved=0.03,
                    min_component_m2=0.0025, shadow_reach_m=0.15):
    """Chassis footprint from the persistence maps.

    1. core: cells with above-floor depth in >= ``persist`` of frames
       (the scene changes, the robot does not), specks removed.
    2. shadow: never-observed cells (``p_obs`` < ``unobserved``) within
       ``shadow_reach_m`` of the core and inside its front/rear span --
       floor the body itself hides from the cameras beside and under it
       (wheels, underbody).  The strip the frame hides *ahead* of the
       bumper is occluded floor, not body, so it is not included.
    3. the convex hull of core + shadow.

    Returns (mask, core, shadow) boolean rasters.
    """
    core = (p_body >= persist).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(core, connectivity=8)
    min_cells = max(1, int(round(min_component_m2 / (cell * cell))))
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_cells
    core = keep[lab]
    if not core.any():
        raise ValueError("no persistent above-floor cells: lower --persist "
                         "or check the depth frames")
    xs = np.flatnonzero(core.any(axis=1))
    reach = int(round(shadow_reach_m / cell))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (2 * reach + 1, 2 * reach + 1))
    near_core = cv2.dilate(core.astype(np.uint8), kernel).astype(bool)
    x_ok = np.zeros_like(core)
    x_ok[max(0, xs.min() - 2):xs.max() + 3, :] = True   # rear..front edge only
    shadow = (p_obs < unobserved) & near_core & x_ok & ~core
    pts = np.argwhere(core | shadow)[:, ::-1].astype(np.int32)   # (col=y, row=x)
    hull = cv2.convexHull(pts)
    mask = np.zeros(core.shape, np.uint8)
    cv2.fillConvexPoly(mask, hull, 1)
    return mask.astype(bool), core, shadow


def extent_m(mask, x0, y0, cell):
    xs = np.flatnonzero(mask.any(axis=1))
    ys = np.flatnonzero(mask.any(axis=0))
    return (x0 + xs.min() * cell, x0 + (xs.max() + 1) * cell,
            y0 + ys.min() * cell, y0 + (ys.max() + 1) * cell)


def render_preview(p_obs, p_body, mask, core, shadow, x0, y0, cell, *,
                   ego_rect=None, cull_rect=None, scale=3):
    """Three panels, forward up, body +y (left) on the left."""
    def to_img(a):
        # raster [ix, iy] -> image rows = far..near (x descending), cols = +y..-y
        return np.ascontiguousarray(a[::-1, ::-1])

    def panel(gray_or_bgr):
        img = gray_or_bgr
        if img.ndim == 2:
            img = cv2.applyColorMap((np.clip(img, 0, 1) * 255).astype(np.uint8),
                                    cv2.COLORMAP_VIRIDIS)
        img = np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)
        return np.ascontiguousarray(img)

    def px(x, y):
        # metric -> pixel in the flipped, scaled image
        nx, ny = p_obs.shape
        ix = nx - 1 - int((x - x0) / cell)
        iy = ny - 1 - int((y - y0) / cell)
        return (int(iy * scale + scale // 2), int(ix * scale + scale // 2))

    def draw_rect(img, rect, color):
        if rect is None:
            return
        rx0, rx1, ry0, ry1 = rect
        cv2.rectangle(img, px(rx1, ry1), px(rx0, ry0), color, 1)

    m = np.zeros(p_obs.shape + (3,), np.uint8)
    m[:] = (40, 40, 40)
    m[mask] = (90, 90, 90)
    m[shadow] = (160, 120, 40)
    m[core] = (60, 200, 60)
    panels = [panel(to_img(p_obs)), panel(to_img(p_body)), panel(to_img(m))]
    for img in panels:
        cv2.drawMarker(img, px(0.0, 0.0), (255, 255, 255), cv2.MARKER_CROSS, 12, 1)
        draw_rect(img, ego_rect, (0, 255, 255))
        draw_rect(img, cull_rect, (255, 200, 0))
    titles = ["p(observed)", "p(above floor)", "ego: core / shadow / hull"]
    out = []
    for img, title in zip(panels, titles):
        bar = np.zeros((18, img.shape[1], 3), np.uint8)
        cv2.putText(bar, title, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (230, 230, 230), 1, cv2.LINE_AA)
        out.append(np.vstack([bar, img]))
    return np.hstack(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--sessions", nargs="*", default=None)
    ap.add_argument("--stride", type=int, default=10,
                    help="use every Nth index row of each drive")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--cell", type=float, default=0.01)
    ap.add_argument("--x-range", type=parse_range, default=(-0.60, 0.90))
    ap.add_argument("--y-range", type=parse_range, default=(-0.60, 0.60))
    ap.add_argument("--persist", type=float, default=0.85,
                    help="fraction of frames with above-floor depth for chassis")
    ap.add_argument("--unobserved", type=float, default=0.03,
                    help="max observed fraction for a body-shadow cell")
    ap.add_argument("--shadow-reach", type=float, default=0.15)
    ap.add_argument("--out", type=Path, default=CONFIG_DIR / "ego_mask.npz")
    ap.add_argument("--preview", type=Path, default=None)
    args = ap.parse_args()

    specs = full_res_specs()
    sessions = session_dirs(args.root)
    if args.sessions:
        sessions = [s for s in sessions if s.name in args.sessions]
    frames = []
    for s in sessions:
        for row in read_index(s)[::max(1, args.stride)]:
            frames.append((s, int(row["stamp_ns"])))
    if args.max_frames:
        frames = frames[:args.max_frames]
    print(f"accumulating {len(frames)} frames from {len(sessions)} drives",
          flush=True)
    p_obs, p_body, n = accumulate(frames, specs, args.x_range, args.y_range,
                                  args.cell)
    mask, core, shadow = derive_ego_mask(
        p_obs, p_body, args.cell, persist=args.persist,
        unobserved=args.unobserved, shadow_reach_m=args.shadow_reach)
    x0, y0 = args.x_range[0], args.y_range[0]
    ext = extent_m(mask, x0, y0, args.cell)
    core_ext = extent_m(core, x0, y0, args.cell)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out, mask=mask, core=core, shadow=shadow,
        p_obs=p_obs.astype(np.float32), p_body=p_body.astype(np.float32),
        x0=x0, y0=y0, cell=args.cell, frames=n,
        extent_m=np.asarray(ext, np.float32),
        created=time.strftime("%Y-%m-%dT%H:%M:%S"))
    print(f"frames {n}; core (seen chassis) x {core_ext[0]:+.2f}..{core_ext[1]:+.2f} "
          f"y {core_ext[2]:+.2f}..{core_ext[3]:+.2f} m; "
          f"footprint x {ext[0]:+.2f}..{ext[1]:+.2f} y {ext[2]:+.2f}..{ext[3]:+.2f} m "
          f"({(ext[1]-ext[0]):.2f} x {(ext[3]-ext[2]):.2f} m, "
          f"{mask.sum() * args.cell**2:.3f} m^2); wrote {args.out}", flush=True)
    robot = load_rig_config()["robots"]["default"]
    ego = (robot.get("bev_student") or {}).get("ego") or {}
    ego_rect = (tuple(ego["x_range_m"]) + tuple(ego["y_range_m"])
                if "x_range_m" in ego else None)
    cull = ((robot.get("bev_student") or {}).get("label") or {}).get("cull_rect_m")
    preview = args.preview or args.out.with_suffix(".png")
    cv2.imwrite(str(preview), render_preview(
        p_obs, p_body, mask, core, shadow, x0, y0, args.cell,
        ego_rect=ego_rect, cull_rect=tuple(cull) if cull else None))
    print(f"preview {preview}", flush=True)


if __name__ == "__main__":
    main()
