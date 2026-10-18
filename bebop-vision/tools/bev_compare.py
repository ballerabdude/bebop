"""Capture a near/far color+depth pair, build the master BEV, and render a
side-by-side comparison for operator review.

Run on the robot (root, cameras free):

    sudo .venv/bin/python tools/bev_compare.py

Writes /tmp/bev_compare.png plus the individual near/far color, color-BEV
and occupancy images.  The color-BEV is the camera's own pixels re-projected
top-down through the same mount/intrinsics as the grid, so an occupied cell
can be checked against the object that produced it.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from bebop_vision.master_bev import MasterBevBuilder, mount_rotation, render_bev
from bebop_vision.orbbec import (OrbbecRig, load_intrinsics,
                                 load_rig_config)

OUT = Path("/tmp/bev_compare")
STRIDE = 4


def color_bev(frame, builder, role, cell_m, rows, cols, range_m, width_m):
    """Re-project one camera's color pixels into the master BEV grid."""
    intr = load_intrinsics(frame.serial)
    cfx, cfy = float(intr["color_fx"]), float(intr["color_fy"])
    ccx, ccy = float(intr["color_cx"]), float(intr["color_cy"])
    cw, ch = int(intr["color_width"]), int(intr["color_height"])
    dfx, dfy = float(intr["fx"]), float(intr["fy"])
    dcx, dcy = float(intr["cx"]), float(intr["cy"])
    dw, dh = int(intr["width"]), int(intr["height"])

    if not frame.color_jpeg:
        return None, None
    color = cv2.imdecode(np.frombuffer(frame.color_jpeg, np.uint8),
                         cv2.IMREAD_COLOR)
    if color is None:
        return None, None
    color = cv2.resize(color, (cw, ch), interpolation=cv2.INTER_AREA)
    color_small = color[::STRIDE, ::STRIDE]

    mount = builder.mounts[frame.serial]
    rot = mount_rotation(mount.pitch_deg, mount.yaw_deg)
    us = np.arange(0, cw, STRIDE) - ccx
    vs = np.arange(0, ch, STRIDE) - ccy
    u, v = np.meshgrid(us, vs)
    dirs = np.stack([u / cfx, v / cfy, np.ones_like(u)], axis=-1).astype(
        np.float32)
    d_body = dirs @ rot.T

    # The color and depth sensors are a few millimetres apart; for this
    # visual check we project the color ray into the depth frame directly.
    with np.errstate(divide="ignore", invalid="ignore"):
        ud = np.clip((dirs[..., 0] / dirs[..., 2] * dfx + dcx).astype(int),
                     0, dw - 1)
        vd = np.clip((dirs[..., 1] / dirs[..., 2] * dfy + dcy).astype(int),
                     0, dh - 1)
    scale = frame.depth_scale_mm if frame.depth_scale_mm is not None else 1.0
    depth = frame.depth[vd, ud].astype(np.float32) * (float(scale) * 1e-3)
    valid = (depth > 0.0) & (depth < builder.max_range_m)

    x = depth * d_body[..., 0]
    y = depth * d_body[..., 1]
    z = depth * d_body[..., 2] + mount.height_m
    rr = ((range_m - x) / cell_m).astype(np.int32)
    cc = ((y + width_m / 2.0) / cell_m).astype(np.int32)
    inb = valid & (rr >= 0) & (rr < rows) & (cc >= 0) & (cc < cols)
    if not inb.any():
        return None, None
    flat = rr[inb] * cols + cc[inb]
    n_cells = rows * cols
    color_small = color_small[:inb.shape[0], :inb.shape[1]]
    count = np.bincount(flat, minlength=n_cells).astype(np.float32)
    b = np.bincount(flat, weights=color_small[..., 0][inb],
                    minlength=n_cells)
    g = np.bincount(flat, weights=color_small[..., 1][inb],
                    minlength=n_cells)
    r = np.bincount(flat, weights=color_small[..., 2][inb],
                    minlength=n_cells)
    img = np.zeros((n_cells, 3), np.uint8)
    have = count > 0
    img[have, 0] = np.clip(b[have] / count[have], 0, 255)
    img[have, 1] = np.clip(g[have] / count[have], 0, 255)
    img[have, 2] = np.clip(r[have] / count[have], 0, 255)
    return img.reshape(rows, cols, 3), have.reshape(rows, cols)


def label(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 20), (0, 0, 0), -1)
    cv2.putText(out, text, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (235, 235, 235), 1, cv2.LINE_AA)
    return out


def fit_width(img, width=480, height=300):
    scale = min(width / img.shape[1], height / img.shape[0])
    return cv2.resize(img, (int(img.shape[1] * scale),
                            int(img.shape[0] * scale)),
                      interpolation=cv2.INTER_AREA)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rig = OrbbecRig(color=True)
    if not rig.wait_for_pair(10.0):
        rig.stop()
        raise SystemExit("cameras did not produce fresh frames")
    time.sleep(0.5)
    builder = MasterBevBuilder()
    frames = {role: cam.read() for role, cam in rig.cameras.items()}
    cps = {role: builder.prepare(f) for role, f in frames.items()}
    grid = builder.fuse(cps, {r: f.age_s() for r, f in frames.items()})
    rig.stop()
    if grid is None:
        raise SystemExit("no fused grid")

    occ = render_bev(grid, scale=4)
    cv2.imwrite(str(OUT / "occupancy.png"), occ)

    panels = []
    colors = {}
    for role in ("near", "far"):
        f = frames.get(role)
        if f is None or f.color_jpeg is None:
            continue
        color = cv2.imdecode(np.frombuffer(f.color_jpeg, np.uint8),
                             cv2.IMREAD_COLOR)
        if color is None:
            continue
        cv2.imwrite(str(OUT / f"{role}_color.png"), color)
        cbev, have = color_bev(f, builder, role, builder.cell_m,
                               builder.rows, builder.cols,
                               builder.range_m, builder.width_m)
        if cbev is None:
            continue
        colors[role] = (cbev, have)
        up = np.repeat(np.repeat(cbev, 4, axis=0), 4, axis=1)
        up = np.ascontiguousarray(up[:, ::-1])
        cv2.imwrite(str(OUT / f"{role}_color_bev.png"), up)
        panels.append((f"{role} color", fit_width(color)))
        panels.append((f"{role} color -> BEV", up))

    if colors:
        # Merge near/far photometric BEV: near below 1.5 m, far beyond.
        merged = np.zeros((builder.rows, builder.cols, 3), np.uint8)
        have = np.zeros((builder.rows, builder.cols), bool)
        x = builder.x_centers[:, None]
        for role in ("near", "far"):
            if role not in colors:
                continue
            img, hv = colors[role]
            take = hv & ~have
            if role == "near":
                take &= np.broadcast_to(x < 1.5, (builder.rows, builder.cols))
            elif "near" in colors:
                take &= np.broadcast_to(x >= 1.5, (builder.rows, builder.cols))
            merged[take] = img[take]
            have |= take
        up = np.repeat(np.repeat(merged, 4, axis=0), 4, axis=1)
        up = np.ascontiguousarray(up[:, ::-1])
        # Blend the occupancy classes over the photometric BEV.
        blend = up.copy()
        for cls, color in {1: (0, 0, 220), 2: (0, 140, 255),
                           3: (60, 60, 120)}.items():
            m = (grid.occ == cls) & grid.coverage
            m = np.repeat(np.repeat(m, 4, axis=0), 4, axis=1)[:, ::-1]
            blend[m] = (0.45 * blend[m] + 0.55 * np.array(color)).astype(
                np.uint8)
        cv2.imwrite(str(OUT / "merged_color_bev.png"), up)
        cv2.imwrite(str(OUT / "merged_overlay.png"), blend)
        panels.append(("merged color -> BEV", up))
        panels.append(("occupancy over color", blend))
    panels.append(("occupancy", occ))

    cols = 3
    cell_w, cell_h = 480, 520
    rows_n = (len(panels) + cols - 1) // cols
    canvas = np.zeros((rows_n * cell_h, cols * cell_w, 3), np.uint8)
    for i, (title, img) in enumerate(panels):
        im = label(img, title)
        h, w = im.shape[:2]
        r, c = divmod(i, cols)
        canvas[r * cell_h:r * cell_h + h, c * cell_w:c * cell_w + w] = im
    Path("/tmp/bev_compare.png").write_bytes(cv2.imencode(
        ".png", canvas)[1].tobytes())
    print("wrote /tmp/bev_compare.png")
    print("counts", grid.counts())
    print("plane_ok", grid.plane_ok)
    for role, c in grid.contributions.items():
        print(role, c)


if __name__ == "__main__":
    main()
