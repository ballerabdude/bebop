"""Fuse SAM 3.1 floor masks with depth into BEV labels (v2 — SAM+depth only).

v2 (2026-09-07, user decision): nothing geometric enters the fusion — no
RANSAC plane, no height-band occupancy classifier, no object detector.
The depth camera itself is the geometry; SAM 3.1 (floor/carpet/rug/ground
concepts only) is the only segmentation. Per color pixel (stride 4), per
camera, per tick:

  navigable  SAM floor mask AND the ray's measured depth lands within
             FLOOR_TOL of the flat-floor prediction -> the landing cell
             is drivable ground.
  blocked    non-floor pixel:
             - valid depth AND the measured surface is NOT within
               FLOOR_TOL of the ground plane: the ray hit something that
               is not ground -> mark the obstacle-band cells within
               +-DEPTH_TOL of the measured range
             - invalid depth (glass / dark / overexposed): full band
               sweep up to the ground hit (conservative — we know
               something is there, not where)
             - valid depth but surface == ground plane (SAM missed /
               shadow): no blocked mark — left unconfirmed, caution
  caution    everything else (unconfirmed floor, conflicts, out-of-FOV)

Cells inside the body-frame self_mask footprint are never drivable.
Known gap (accepted 2026-09-07): stair descents / holes that SAM masks as
ground read navigable — negative obstacles have no label signal; the
recorded /bev_teacher stays in each npz as bookkeeping so a drop-detection
term can be re-added later by re-running this tool (no re-recording).

Per session, updates labels/{stamp}.npz in place:
    teacher      uint8 60x60  raw geometric grid (recorded input, bookkeeping)
    sem_near     uint8 60x60  near-camera SAM+depth blocked cells
    sem_far      uint8 60x60  far-camera SAM+depth blocked cells
    floor_near   uint8 60x60  near-camera floor confirmations
    floor_far    uint8 60x60  far-camera floor confirmations
    fused        uint8 60x60  0 blocked / 1 navigable / 2 caution  <- training
    disagree     uint8 60x60  SAM/depth blocked where geometric saw nothing
    unconfirmed  uint8 60x60  caution where geometric saw nothing (mining)
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bebop_vision.bev import mount_rotation  # noqa: E402

PIXEL_STRIDE = 4
BAND_LO_M, BAND_HI_M = 0.03, 0.30
SAMPLE_STEP_M = 0.025
MAX_SAMPLES = 60
DEPTH_TOL_M = 0.15
FLOOR_TOL_M = 0.25          # |floor landing range - measured depth| gate


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
      land_cells, t_g  flat-floor landing cell / landing range
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


def _sweep_blocked(lut, sel, d_m, valid):
    """Vectorized ragged band sweep for the given ray indices.

    Rays with valid depth mark only the band cells within +-DEPTH_TOL of
    the measured range; rays without valid depth mark their whole band
    (something is there, we don't know where). Returns a bool cell map.
    """
    lens = lut["offsets"][sel + 1] - lut["offsets"][sel]
    nz = lens > 0
    pix = sel[nz]
    lens = lens[nz]
    if len(pix) == 0:
        return np.zeros(60 * 60, bool)
    starts = lut["offsets"][pix]
    tot = int(lens.sum())
    within = np.arange(tot, dtype=np.int64) - np.repeat(
        np.concatenate([[0], np.cumsum(lens)[:-1]]).astype(np.int64), lens)
    flat = np.repeat(starts, lens) + within
    hits_all = lut["cells"][flat]
    t1, t2 = lut["t1"][pix], lut["t2"][pix]
    dv, ok = d_m[nz], valid[nz]
    t_lo = np.where(ok, np.maximum(t1, dv - DEPTH_TOL_M), t1)
    t_hi = np.where(ok, np.minimum(t2, dv + DEPTH_TOL_M), t2)
    frac = (within / np.repeat(np.maximum(lens - 1, 1), lens)).astype(np.float32)
    t_sample = np.repeat(t1, lens) \
        + (np.repeat(t2, lens) - np.repeat(t1, lens)) * frac
    keep = (t_sample >= np.repeat(t_lo, lens)) \
        & (t_sample <= np.repeat(t_hi, lens))
    hits = hits_all[keep]
    if len(hits) == 0:
        return np.zeros(60 * 60, bool)
    return np.bincount(hits, minlength=60 * 60) > 0


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

    stats = {"n": 0, "blocked": [], "sem": [], "caution": [],
             "floor": [], "nav": [], "disagree": []}
    for npz_path in sorted((sess_dir / "labels").glob("*.npz")):
        stamp = int(npz_path.stem)
        dep = np.load(sess_dir / "depth" / f"{stamp:020d}.npz")
        d = dict(np.load(npz_path))
        teacher = d["teacher"].reshape(-1)   # bookkeeping only (not fused)
        sem_union = np.zeros(60 * 60, bool)      # SAM+depth blocked marks
        floor_union = np.zeros(60 * 60, bool)    # SAM floor confirmations
        for role in roles:
            lut = luts[role]
            blocked = np.zeros(60 * 60, bool)
            floor = np.zeros(60 * 60, bool)
            fp = sess_dir / ("sam_floor" if role == "near"
                             else "sam_floor_far") / f"{stamp:020d}.npz"
            if fp.exists():
                fm = np.load(fp)["mask"]
                act = fm[::PIXEL_STRIDE, ::PIXEL_STRIDE].ravel()
                depth_img = dep[role]
                # floor confirmations: SAM floor AND depth lands on the
                # flat-ground prediction
                fsel = np.where(act)[0]
                if len(fsel):
                    d_m = depth_img[lut["v_d"][fsel],
                                    lut["u_d"][fsel]].astype(np.float32) * 1e-3
                    ok = (d_m > 0.05) & (d_m < 6.0) \
                        & (np.abs(lut["t_g"][fsel] - d_m) <= FLOOR_TOL_M)
                    lc = lut["land_cells"][fsel[ok]]
                    lc = lc[lc >= 0]
                    if len(lc):
                        floor[np.bincount(lc, minlength=60 * 60) > 0] = True
                # blocked evidence: non-floor pixels whose measured surface
                # is not the ground plane (or that returned no depth at all)
                nsel = np.where(~act)[0]
                if len(nsel):
                    d_m = depth_img[lut["v_d"][nsel],
                                    lut["u_d"][nsel]].astype(np.float32) * 1e-3
                    valid = (d_m > 0.05) & (d_m < 6.0)
                    # depth-consistent ground that SAM did not confirm
                    # (shadow / missed mask) stays unconfirmed, not blocked
                    on_ground = valid \
                        & (np.abs(lut["t_g"][nsel] - d_m) <= FLOOR_TOL_M)
                    sel = nsel[~on_ground]   # the rest: sweep (windowed/full)
                    blocked = _sweep_blocked(lut, sel, d_m[~on_ground],
                                             valid[~on_ground])
            x, y_ = cells_to_xy(np.where(blocked)[0], bev)
            infoot = (x > sm["x_range_m"][0]) & (x < sm["x_range_m"][1]) \
                & (y_ > sm["y_range_m"][0]) & (y_ < sm["y_range_m"][1])
            blocked[np.where(blocked)[0][infoot]] = False
            d[f"sem_{role}"] = blocked.astype(np.uint8).reshape(60, 60)
            d[f"floor_{role}"] = floor.astype(np.uint8).reshape(60, 60)
            sem_union |= blocked
            floor_union |= floor
        # the robot's own footprint is never drivable
        rows_g, cols_g = np.mgrid[0:60, 0:60]
        xg = bev["range_m"] - (rows_g + 0.5) * bev["cell_m"]
        yg = (cols_g + 0.5) * bev["cell_m"] - bev["width_m"] / 2.0
        footprint = ((xg > sm["x_range_m"][0]) & (xg < sm["x_range_m"][1])
                     & (yg > sm["y_range_m"][0]) & (yg < sm["y_range_m"][1]))
        blocked_all = sem_union | footprint.reshape(-1)
        navigable = floor_union & ~blocked_all
        caution = ~blocked_all & ~navigable
        fused = np.ones(60 * 60, np.uint8)
        fused[blocked_all] = 0
        fused[caution] = 2
        d["fused"] = fused.reshape(60, 60)
        d["disagree"] = (sem_union & (teacher == 0)).astype(np.uint8) \
            .reshape(60, 60)
        d["unconfirmed"] = (caution & (teacher == 0)).astype(np.uint8) \
            .reshape(60, 60)
        np.savez_compressed(npz_path, **d)
        stats["n"] += 1
        stats["blocked"].append(blocked_all.mean())
        stats["sem"].append(sem_union.mean())
        stats["caution"].append(caution.mean())
        stats["floor"].append(floor_union.mean())
        stats["nav"].append(navigable.mean())
        stats["disagree"].append(d["disagree"].mean())
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
