"""Regenerate SAM/depth labels directly at an explicit BEV grid resolution.

    .venv/bin/python tools/bev_relabel_grid.py --rows 360 --cols 640 --resume

Uses the same frame selection as labels_base2, samples native depth pixels,
and writes a separate labels_grid360x640 directory in each session. Existing
teacher label sets are preserved. No label resizing or unobserved-floor fill.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
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
from bebop_vision.bev_student_labels import (
    ALL_PROMPTS, BevStudentLabeler, class_counts, load_label_npz,
    render_label_bev, save_label_npz, validate_label_arrays)
from bebop_vision.orbbec import load_rig_config
from bebop_vision.sam3_concepts import Sam3ConceptSegmenter
from tools.bev_student_labels import category_maps


def parse_rect(text):
    """'x0,x1,y0,y1' -> (x0, x1, y0, y1) floats (body frame, metres)."""
    vals = tuple(float(v) for v in str(text).split(","))
    if len(vals) != 4 or vals[0] >= vals[1] or vals[2] >= vals[3]:
        raise argparse.ArgumentTypeError(
            "rectangle must be x0,x1,y0,y1 with x0<x1 and y0<y1")
    return vals


def make_labeler(rows, cols, depth_stride=1, width_m=None, range_m=None,
                 splat=False, cull_rect=None, ego_rect=None):
    cfg = copy.deepcopy(load_rig_config())
    robot = cfg["robots"]["default"]
    grid = robot.setdefault("bev_student", {}).setdefault("grid", {})
    grid.update(rows=rows, cols=cols)
    if width_m is not None:
        grid["width_m"] = float(width_m)
    if range_m is not None:
        grid["range_m"] = float(range_m)
    label = robot["bev_student"].setdefault("label", {})
    if splat:
        label["splat_footprint"] = True
    if cull_rect is not None:
        label["cull_rect_m"] = list(cull_rect)
    if ego_rect is not None:
        robot["bev_student"]["ego"] = {"x_range_m": [ego_rect[0], ego_rect[1]],
                                       "y_range_m": [ego_rect[2], ego_rect[3]]}
    for cam in robot["cameras"].values():
        cam["label_stride_x"] = depth_stride
        cam["label_stride_y"] = depth_stride
    return BevStudentLabeler(rig_cfg=cfg, roles=("near", "far"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--rows", type=int, required=True)
    ap.add_argument("--cols", type=int, required=True)
    ap.add_argument("--source-labels", default="labels_base2",
                    help="label the frames that have a label in this subdir "
                         "(ignored when --stride is given)")
    ap.add_argument("--stride", type=int, default=None,
                    help="label every Nth index row instead of mirroring "
                         "--source-labels (the base sets used --stride 3)")
    ap.add_argument("--out-labels", default=None)
    ap.add_argument("--depth-stride", type=int, default=1)
    ap.add_argument("--width-m", type=float, default=None,
                    help="BEV width in metres (default: rig config, 3.0)")
    ap.add_argument("--range-m", type=float, default=None,
                    help="BEV forward range in metres (default: rig config)")
    ap.add_argument("--splat", action="store_true",
                    help="footprint splat: each depth pixel covers the patch "
                         "to its neighbours instead of one cell")
    ap.add_argument("--cull-rect", type=parse_rect, default=None,
                    help="x0,x1,y0,y1 body-frame rectangle (chassis + margin) "
                         "that replaces the 0.55 m dead disc")
    ap.add_argument("--ego-rect", type=parse_rect, default=None,
                    help="x0,x1,y0,y1 physical chassis rectangle for the ego "
                         "layer (default: rig config)")
    ap.add_argument("--sessions", nargs="*", default=None)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    if args.stride is not None and args.stride < 1:
        ap.error("stride must be positive")
    if min(args.rows, args.cols, args.depth_stride) < 1:
        ap.error("rows, columns and depth stride must be positive")
    out_name = args.out_labels or f"labels_grid{args.rows}x{args.cols}"
    if (Path(out_name).name != out_name or not out_name.startswith("labels_grid")
            or out_name == args.source_labels):
        ap.error("output must be a separate labels_grid... directory name")
    lab = make_labeler(args.rows, args.cols, args.depth_stride,
                       width_m=args.width_m, range_m=args.range_m,
                       splat=args.splat, cull_rect=args.cull_rect,
                       ego_rect=args.ego_rect)
    metadata = json.loads(json.dumps({
        "grid": lab.grid.metadata(),
        "source_labels": None if args.stride else args.source_labels,
        "frame_stride": args.stride,
        "depth_stride": args.depth_stride, "sam_version": "sam3.1",
        "sam_conf": args.conf, "label_config": dataclasses.asdict(lab.label_cfg),
    }))
    seg = Sam3ConceptSegmenter(concepts=list(ALL_PROMPTS), conf=args.conf,
                               version="sam3.1", device="cuda")
    sessions = session_dirs(args.root)
    if args.sessions:
        sessions = [s for s in sessions if s.name in args.sessions]
    if not sessions:
        raise ValueError("no matching sessions")
    total = 0
    for session in sessions:
        if args.stride:
            rows = read_index(session)[::args.stride]
        else:
            rows = [r for r in read_index(session) if
                    (session / args.source_labels / f"{int(r['stamp_ns']):020d}.npz").exists()]
        if args.max_frames:
            rows = rows[:args.max_frames]
        if not rows:
            raise ValueError(f"no source labels in {session}")
        out = session / out_name
        out.mkdir(exist_ok=True)
        meta_path = out / "grid.json"
        if meta_path.exists() and json.loads(meta_path.read_text()) != metadata:
            raise ValueError(f"incompatible existing label metadata: {meta_path}")
        if not meta_path.exists() and list(out.glob("*.npz")):
            raise ValueError(f"existing labels have no grid metadata: {out}")
        meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
        counts = {}
        started = time.monotonic()
        with (out / "diagnostics.jsonl").open("a") as diag:
            for i, row in enumerate(rows):
                stamp = int(row["stamp_ns"])
                path = out / f"{stamp:020d}.npz"
                if args.resume and path.exists():
                    arrays = load_label_npz(path, lab.grid)
                else:
                    if path.exists():
                        raise FileExistsError(f"use --resume for existing label: {path}")
                    depths, masks = {}, {}
                    for role in ("near", "far"):
                        depth = cv2.imread(str(session / f"depth_{role}" /
                                               f"{stamp:020d}.png"), cv2.IMREAD_UNCHANGED)
                        color = cv2.imread(str(session / f"color_{role}" /
                                               f"{stamp:020d}.jpg"))
                        if depth is None or color is None:
                            raise FileNotFoundError(f"missing {role} RGB-D: {session}/{stamp}")
                        depths[role] = depth
                        masks[role] = category_maps(seg.segment(color), *color.shape[:2])
                    lab._rng = np.random.default_rng(0)
                    label = lab.label(depths, masks)
                    if label is None:
                        raise RuntimeError(f"no label for {session}/{stamp}")
                    arrays = (label.semantic, label.ego, label.coverage)
                    validate_label_arrays(*arrays, lab.grid)
                    temp = path.with_suffix(".tmp.npz")
                    save_label_npz(temp, label)
                    temp.replace(path)
                    diag.write(json.dumps({"stamp_ns": stamp,
                                           "plane_ok": label.stats["plane_ok"],
                                           "counts": class_counts(*arrays)}) + "\n")
                    diag.flush()
                frame_counts = class_counts(*arrays)
                for key, value in frame_counts.items():
                    counts[key] = counts.get(key, 0) + value
                if i == 0 or (i + 1) % 100 == 0:
                    cv2.imwrite(str(out / f"preview_{stamp}.png"),
                                render_label_bev(*arrays, lab.grid, scale=1))
                if i == 0 or (i + 1) % 10 == 0:
                    print(f"{session.name}: {i + 1}/{len(rows)} "
                          f"{(time.monotonic() - started)/(i + 1):.2f}s/frame "
                          f"covered={frame_counts['covered']/frame_counts['cells']:.1%}",
                          flush=True)
        (out / "stats.json").write_text(json.dumps({
            "frames": len(rows), "grid": lab.grid.metadata(),
            "class_cells": counts, "elapsed_s": time.monotonic() - started,
        }, indent=2) + "\n")
        total += len(rows)
    print(f"GRID_LABELS_DONE {out_name}: {total} frames", flush=True)


if __name__ == "__main__":
    main()
