"""Fuse YOLO image-space masks with the geometric teacher into BEV labels.

Projection is a semantic FRUSTUM with depth-gated marking. A masked color
pixel defines a ray; the ray marks obstacle cells where it passes through
the height band [0.03, 0.30] m above the floor:

- depth valid at the pixel  -> mark only the +-0.15 m range window around
  the measured surface (tight, calibrated marking; floor bleed from mask
  edges cancels because the floor reading sits below the band)
- depth invalid (glass, dark, overexposed) -> mark the FULL band sweep up
  to the ground hit (conservative — this is the case the teacher cannot
  see at all)

Cells inside the body-frame self_mask footprint are dropped. Per session,
updates labels/{stamp}.npz in place:
    teacher   uint8 60x60  raw geometric grid (unchanged input)
    yolo_near uint8 60x60  near-camera frustum-blocked cells
    yolo_far  uint8 60x60  far-camera frustum-blocked cells
    fused     uint8 60x60  0 blocked / 1 navigable / 2 caution  <- training
    disagree  uint8 60x60  teacher-free but yolo-blocked (mining/hand-label)
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bebop_vision.bev import mount_rotation  # noqa: E402

# COCO ids that are floor-clutter / too small to block a wheeled robot;
# everything else marks blocked cells.
FLAT_IGNORE = {40, 41, 42, 43, 44, 45, 63, 64, 65, 66, 67, 73, 74, 79}
MIN_INSTANCE_PX = 1200
PIXEL_STRIDE = 4
BAND_LO_M, BAND_HI_M = 0.03, 0.30
SAMPLE_STEP_M = 0.025
MAX_SAMPLES = 60
DEPTH_TOL_M = 0.15
FLOOR_TOL_M = 0.25          # |floor landing range - measured depth| gate
MARGIN_CELLS = 4            # 0.20 m robot radius


def load_cfg():
    import yaml
    cfg = yaml.safe_load(open(ROOT / "config" / "orbbec_rig.yaml"))
    cams = cfg["robots"]["default"]["cameras"]
    bev = cfg["robots"]["default"]["bev"]
    sm = cfg["robots"]["default"]["robot"]["self_mask"]
    return cams, bev, sm


def build_lut(serial, cam_cfg, bev, intr):
    """Per color pixel (stride 4):
      t1, t2  band-entry/exit range along the ray (t = optical z, m)
      u_d, v_d pixel projected into the depth image
      cells/offsets  ragged list of swept ground cells (t-ordered)
    """
    fx, fy = intr["color_fx"], intr["color_fy"]
    cx, cy = intr["color_cx"], intr["color_cy"]
    W, H = int(intr["color_width"]), int(intr["color_height"])
    dfx, dfy = intr["fx"], intr["fy"]
    dcx, dcy = intr["cx"], intr["cy"]
    R = mount_rotation(float(cam_cfg["pitch_deg"]),
                       float(cam_cfg.get("yaw_deg", 0.0)))
    us = np.arange(0, W, PIXEL_STRIDE) - cx
    vs = np.arange(0, H, PIXEL_STRIDE) - cy
    u, v = np.meshgrid(us, vs)
    dirs = np.stack([u / fx, v / fy, np.ones_like(u, np.float32)],
                    axis=-1).reshape(-1, 3).astype(np.float32)  # optical
    n_px = len(dirs)
    d_body = dirs @ R.T
    H0 = float(cam_cfg["height_m"])
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = np.where(d_body[:, 2] < 0, (BAND_HI_M - H0) / d_body[:, 2], np.inf)
        t2 = np.where(d_body[:, 2] < 0, (BAND_LO_M - H0) / d_body[:, 2], np.inf)
    # depth-image projection of the same optical ray (color ~ depth origin)
    u_d = np.clip((dirs[:, 0] / dirs[:, 2] * dfx + dcx).astype(np.int64),
                  0, 847)
    v_d = np.clip((dirs[:, 1] / dirs[:, 2] * dfy + dcy).astype(np.int64),
                  0, 479)

    # ragged cell sweep between t1 and t2 (both inf for sky rays -> empty)
    ok = np.isfinite(t1) & np.isfinite(t2) & (t2 > t1)
    x1, y1 = t1 * d_body[:, 0], t1 * d_body[:, 1]
    x2, y2 = t2 * d_body[:, 0], t2 * d_body[:, 1]
    span = np.where(ok, np.hypot(x2 - x1, y2 - y1), 0.0)
    n = np.where(ok, np.minimum(
        np.ceil(span / SAMPLE_STEP_M).astype(np.int64) + 1, MAX_SAMPLES), 0)
    tot = int(n.sum())
    pix_of = np.repeat(np.arange(n_px, dtype=np.int64), n)
    k = np.arange(tot, dtype=np.float32) - np.repeat(
        np.concatenate([[0], np.cumsum(n)[:-1]]).astype(np.int64), n)
    t_of = t1[pix_of] + (t2[pix_of] - t1[pix_of]) * k
    a = (k / np.maximum(n[pix_of] - 1, 1)).astype(np.float32)
    xs = x1[pix_of] + (x2[pix_of] - x1[pix_of]) * a
    ys = y1[pix_of] + (y2[pix_of] - y1[pix_of]) * a
    rows = ((bev["range_m"] - xs) / bev["cell_m"]).astype(np.int64)
    cols = ((ys + bev["width_m"] / 2.0) / bev["cell_m"]).astype(np.int64)
    nrows = int(round(bev["range_m"] / bev["cell_m"]))
    ncols = int(round(bev["width_m"] / bev["cell_m"]))
    inb = (rows >= 0) & (rows < nrows) & (cols >= 0) & (cols < ncols)
    cells = (rows * ncols + cols)[inb]
    pix_of = pix_of[inb]
    t_of = t_of[inb]
    counts = np.bincount(pix_of, minlength=n_px)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    # ground-landing cell per pixel: where the ray meets the floor (z=0).
    # Used by the SAM floor pass: a floor pixel confirms its landing cell.
    with np.errstate(divide="ignore", invalid="ignore"):
        t_g = np.where(d_body[:, 2] < 0, H0 / (-d_body[:, 2]), np.inf)
    lx = np.where(np.isfinite(t_g), t_g * d_body[:, 0], 0.0)
    ly = np.where(np.isfinite(t_g), t_g * d_body[:, 1], 0.0)
    lr = ((bev["range_m"] - lx) / bev["cell_m"]).astype(np.int64)
    lc = ((ly + bev["width_m"] / 2.0) / bev["cell_m"]).astype(np.int64)
    land_ok = np.isfinite(t_g) & (lr >= 0) & (lr < nrows) & (lc >= 0) \
        & (lc < ncols)
    land_cells = np.where(land_ok, lr * ncols + lc, -1).astype(np.int32)
    return dict(t1=t1.astype(np.float32), t2=t2.astype(np.float32),
                u_d=u_d, v_d=v_d, cells=cells.astype(np.int32),
                offsets=offsets, n_px=n_px, land_cells=land_cells,
                t_g=t_g.astype(np.float32))


def load_lut(role):
    p = ROOT / "config" / f"raylut_{role}.npz"
    return {k: v for k, v in np.load(p).items()}


def session_fuse(sess_dir, cams, bev, sm, intr_by_serial, roles=("near", "far")):
    serials = {r: next(s for s, c in cams.items() if c["role"] == r)
               for r in roles}
    luts = {}
    for r in roles:
        p = ROOT / "config" / f"raylut_{r}.npz"
        if not p.exists():
            lut = build_lut(serials[r], cams[serials[r]], bev,
                            intr_by_serial[serials[r]])
            np.savez_compressed(p, **lut)
            print(f"[lut] {r}: {lut['n_px']} rays, {len(lut['cells'])} hits")
        luts[r] = load_lut(r)

    stats = {"n": 0, "blocked": [], "disagree": [], "caution": [],
             "near": [], "far": [], "floor": [], "nav": []}
    for npz_path in sorted((sess_dir / "labels").glob("*.npz")):
        stamp = int(npz_path.stem)
        dep = np.load(sess_dir / "depth" / f"{stamp:020d}.npz")
        d = dict(np.load(npz_path))
        teacher = d["teacher"].reshape(-1)
        union = np.zeros(60 * 60, bool)          # yolo obstacle marks
        floor_union = np.zeros(60 * 60, bool)    # sam floor confirmations
        for role in roles:
            lut = luts[role]
            blocked = np.zeros(60 * 60, bool)
            floor = np.zeros(60 * 60, bool)
            yp = sess_dir / ("yolo" if role == "near" else "yolo_far") \
                / f"{stamp:020d}.npz"
            if yp.exists():
                y = np.load(yp)
                H, W = int(y["shape"][0]), int(y["shape"][1])
                solid = (~np.isin(y["classes"], list(FLAT_IGNORE))) \
                    & (y["confs"] > 0.3)
                if len(y["classes"]):
                    areas = y["bits"].reshape(len(y["classes"]), -1) \
                        .astype(bool).sum(1)
                    solid &= areas >= MIN_INSTANCE_PX
                if solid.any():
                    um = np.zeros((H, W), bool)
                    for j in np.where(solid)[0]:
                        um |= np.unpackbits(y["bits"][j], count=H * W) \
                            .reshape(H, W).astype(bool)
                    act = um[::PIXEL_STRIDE, ::PIXEL_STRIDE].ravel()
                    sel = np.where(act)[0]
                    if len(sel):
                        depth_img = dep[role]
                        d_m = depth_img[lut["v_d"][sel],
                                        lut["u_d"][sel]].astype(np.float32) \
                            * 1e-3
                        valid = (d_m > 0.05) & (d_m < 6.0)
                        t1 = lut["t1"][sel]
                        t2 = lut["t2"][sel]
                        t_lo = np.where(valid, np.maximum(t1, d_m
                                                          - DEPTH_TOL_M), t1)
                        t_hi = np.where(valid, np.minimum(t2, d_m
                                                          + DEPTH_TOL_M), t2)
                        pix = sel
                        lens = (lut["offsets"][pix + 1] - lut["offsets"][pix])
                        ar0 = np.repeat(np.arange(sel.size), lens)
                        ac = np.concatenate(
                            [lut["cells"][lut["offsets"][i]:
                                          lut["offsets"][i + 1]]
                             for i in sel])
                        # t of each sample by linear interpolation across
                        # the pixel's [t1, t2] span
                        starts = np.concatenate(
                            [[0], np.cumsum(lens)[:-1]]).astype(np.int64)
                        k = (np.arange(int(lens.sum()), dtype=np.float32)
                             - np.repeat(starts, lens))
                        frac = k / np.maximum(
                            lens[ar0].astype(np.float32) - 1, 1)
                        t_sample = t1[ar0] + (t2[ar0] - t1[ar0]) * frac
                        keep = (t_sample >= t_lo[ar0]) \
                            & (t_sample <= t_hi[ar0])
                        hits = ac[keep]
                        if len(hits):
                            cnt = np.bincount(hits, minlength=60 * 60)
                            blocked = cnt > 0
            fp = sess_dir / ("sam_floor" if role == "near"
                             else "sam_floor_far") / f"{stamp:020d}.npz"
            if fp.exists():
                fm = np.load(fp)["mask"]
                fact = fm[::PIXEL_STRIDE, ::PIXEL_STRIDE].ravel()
                fsel = np.where(fact)[0]
                if len(fsel):
                    depth_img = dep[role]
                    d_m = depth_img[lut["v_d"][fsel],
                                    lut["u_d"][fsel]].astype(np.float32) * 1e-3
                    ok = (d_m > 0.05) & (d_m < 6.0) \
                        & (np.abs(lut["t_g"][fsel] - d_m) <= FLOOR_TOL_M)
                    lc = lut["land_cells"][fsel[ok]]
                    lc = lc[lc >= 0]
                    if len(lc):
                        floor[np.bincount(lc, minlength=60 * 60) > 0] = True
            x, y_ = cells_to_xy(np.where(blocked)[0], bev)
            infoot = (x > sm["x_range_m"][0]) & (x < sm["x_range_m"][1]) \
                & (y_ > sm["y_range_m"][0]) & (y_ < sm["y_range_m"][1])
            blocked[np.where(blocked)[0][infoot]] = False
            d[f"yolo_{role}"] = blocked.astype(np.uint8).reshape(60, 60)
            d[f"floor_{role}"] = floor.astype(np.uint8).reshape(60, 60)
            union |= blocked
            floor_union |= floor
            stats[role].append(blocked.mean())
        blocked_all = (teacher >= 1) | union
        # the robot's own footprint is never drivable
        rows_g, cols_g = np.mgrid[0:60, 0:60]
        xg = bev["range_m"] - (rows_g + 0.5) * bev["cell_m"]
        yg = (cols_g + 0.5) * bev["cell_m"] - bev["width_m"] / 2.0
        footprint = ((xg > sm["x_range_m"][0]) & (xg < sm["x_range_m"][1])
                     & (yg > sm["y_range_m"][0]) & (yg < sm["y_range_m"][1]))
        blocked_all |= footprint.reshape(-1)
        navigable = floor_union & ~blocked_all
        caution = ~blocked_all & ~navigable
        fused = np.ones(60 * 60, np.uint8)
        fused[blocked_all] = 0
        fused[caution] = 2
        d["fused"] = fused.reshape(60, 60)
        d["disagree"] = (union & (teacher == 0)).astype(np.uint8) \
            .reshape(60, 60)
        d["unconfirmed"] = (caution & (teacher == 0)).astype(np.uint8) \
            .reshape(60, 60)
        np.savez_compressed(npz_path, **d)
        stats["n"] += 1
        stats["blocked"].append(blocked_all.mean())
        stats["disagree"].append(d["disagree"].mean())
        stats["caution"].append(caution.mean())
        stats["floor"].append(floor_union.mean())
        stats["nav"].append(navigable.mean())
    return stats


def cells_to_xy(cells, bev):
    rows, cols = cells // 60, cells % 60
    x = bev["range_m"] - (rows + 0.5) * bev["cell_m"]
    y = (cols + 0.5) * bev["cell_m"] - bev["width_m"] / 2.0
    return x, y


def main():
    from bebop_vision.orbbec import load_intrinsics
    cams, bev, sm = load_cfg()
    intr_by_serial = {s: load_intrinsics(s, str(ROOT / "config"))
                      for s in cams}
    root = ROOT / "datasets" / "navd-v0"
    for sess in sorted(root.glob("navd_session_*")):
        st = session_fuse(sess, cams, bev, sm, intr_by_serial)
        n = st["n"]
        print(f"{sess.name}: {n} | blocked {100 * np.mean(st['blocked']):.1f}%"
              f" | floor {100 * np.mean(st['floor']):.1f}% | nav "
              f"{100 * np.mean(st['nav']):.1f}% | caution "
              f"{100 * np.mean(st['caution']):.1f}% | disagree "
              f"{100 * np.mean(st['disagree']):.2f}%")


if __name__ == "__main__":
    main()
