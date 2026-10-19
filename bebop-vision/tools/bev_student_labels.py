"""SAM 3.1 x depth -> semantic BEV labels for the learned student.

Runs on the workstation (or the robot) against an extracted session::

    python tools/bev_student_labels.py \
        --root datasets/bev-student --session navd_session_20261018_193932 \
        --stride 3 --role both

The extractor (``tools/bev_student_dataset.py extract``) writes
``color_near/``, ``color_far/``, ``depth_near/``, ``depth_far/`` and
``index.jsonl``.  This tool writes ``labels/<stamp>.npz`` (semantic uint8
[120,60], ego bool, coverage bool), ``labels/diagnostics.jsonl`` and review
images under ``preview/``.

Label rate: frames are extracted at 10 Hz; ``--stride 3`` labels every third
frame, i.e. an effective 3.33 Hz.  Adjacent 10 Hz frames are nearly
identical, so this is the intended training rate.

SAM 3.1 is teacher-side only (Meta SAM License); it is never used at runtime.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bebop_vision.bev_student_labels import (  # noqa: E402
    ALL_PROMPTS, FLOOR, HAZARD, OBSTACLE, WALL, BevStudentLabeler,
    LabelConfig, GridSpec, contact_sheet, label_stats, load_label_npz,
    prompt_class, render_color_masks, render_label_bev, save_label_npz,
    summarize_label_files,
)


def session_dirs(root, sessions):
    root = Path(root)
    base = root / "sessions"
    if not sessions:
        return sorted(p for p in base.iterdir() if p.is_dir())
    out = []
    for s in sessions:
        p = Path(s)
        if not p.is_dir():
            p = base / s
        if not p.is_dir():
            raise SystemExit(f"session not found: {s}")
        out.append(p)
    return out


def read_index(session_dir):
    path = Path(session_dir) / "index.jsonl"
    if not path.exists():
        return build_index(session_dir)
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_index(session_dir):
    """Fallback index from on-disk color/depth PNG/JPG names."""
    session_dir = Path(session_dir)
    stamps = None
    for sub, pattern in (("depth_near", "*.png"), ("depth_far", "*.png")):
        d = session_dir / sub
        s = {int(p.stem) for p in d.glob(pattern)} if d.is_dir() else set()
        stamps = s if stamps is None else stamps & s
    stamps = sorted(stamps or [])
    rows = []
    for stamp in stamps:
        rows.append({
            "stamp_ns": stamp,
            "session": session_dir.name,
            "frame": len(rows),
            "color_near": f"color_near/{stamp:020d}.jpg",
            "color_far": f"color_far/{stamp:020d}.jpg",
            "depth_near": f"depth_near/{stamp:020d}.png",
            "depth_far": f"depth_far/{stamp:020d}.png",
            "label": None,
            "split": None,
        })
    return rows


def category_maps(segments, height, width):
    """InstanceSegment list -> per-class confidence maps."""
    maps = {
        FLOOR: np.zeros((height, width), np.float32),
        WALL: np.zeros((height, width), np.float32),
        OBSTACLE: np.zeros((height, width), np.float32),
        HAZARD: np.zeros((height, width), np.float32),
    }
    for seg in segments:
        cls = prompt_class(seg.label)
        if cls is None or seg.mask is None:
            continue
        m = np.asarray(seg.mask)
        m = np.squeeze(m)
        if m.ndim != 2:
            continue
        m = m.astype(bool)
        if m.shape != (height, width):
            m = cv2.resize(m.astype(np.uint8), (width, height),
                           interpolation=cv2.INTER_NEAREST) > 0
        conf = np.clip(float(seg.confidence), 0.0, 1.0)
        if conf <= 0:
            continue
        np.maximum(maps[cls], m.astype(np.float32) * conf,
                   out=maps[cls])
    return maps


def label_session(session_dir, labeler, segmenter, *, roles, stride,
                  resume=False, max_frames=0, preview_every=10,
                  out_dir=None, sampled_hz=10.0):
    session_dir = Path(session_dir)
    out_dir = Path(out_dir) if out_dir else session_dir
    labels_dir = out_dir / "labels"
    preview_dir = out_dir / "preview"
    labels_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    index = read_index(session_dir)
    meta = {}
    meta_path = session_dir / "meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except ValueError:
            meta = {}
    if len(index) > 1:
        span_s = (int(index[-1]["stamp_ns"]) - int(index[0]["stamp_ns"])) / 1e9
        extracted_hz = ((len(index) - 1) / span_s) if span_s > 0 else sampled_hz
    else:
        extracted_hz = sampled_hz
    source_hz = float(meta.get("source_tick_hz") or 0.0)
    diag_path = labels_dir / "diagnostics.jsonl"
    diag = open(diag_path, "a")
    paths = []
    sheet_bev, sheet_color = [], []
    t0 = time.time()
    n_new = n_skip = n_missing = 0
    for i, row in enumerate(index):
        if i % stride:
            continue
        stamp = int(row["stamp_ns"])
        npz_path = labels_dir / f"{stamp:020d}.npz"
        if resume and npz_path.exists():
            paths.append(npz_path)
            n_skip += 1
            continue
        if max_frames and n_new >= max_frames:
            break
        depths = {}
        for role in roles:
            p = session_dir / "depth" / f"{stamp:020d}.png"
            if not p.exists():
                p = session_dir / f"depth_{role}" / f"{stamp:020d}.png"
            img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if img is None:
                depths = None
                break
            depths[role] = img
        if not depths:
            n_missing += 1
            continue
        colors = {}
        masks = {}
        missing_color = False
        for role in roles:
            p = session_dir / "color" / f"{stamp:020d}.jpg"
            if not p.exists():
                p = session_dir / f"color_{role}" / f"{stamp:020d}.jpg"
            bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if bgr is None:
                missing_color = True
                break
            colors[role] = bgr
            segs = segmenter.segment(bgr)
            masks[role] = category_maps(segs, bgr.shape[0], bgr.shape[1])
        if missing_color:
            n_missing += 1
            continue

        label = labeler.label(depths, masks)
        if label is None:
            n_missing += 1
            continue
        save_label_npz(npz_path, label)
        paths.append(npz_path)
        n_new += 1
        diag.write(json.dumps({
            "stamp_ns": stamp,
            "roles": label.stats.get("roles", roles),
            "plane_ok": label.stats.get("plane_ok", {}),
            "plane_fallback": label.stats.get("plane_fallback", {}),
            "stats": label_stats(label.semantic, label.ego, label.coverage),
        }) + "\n")
        diag.flush()

        if preview_every and (n_new - 1) % preview_every == 0:
            bev = render_label_bev(label.semantic, label.ego,
                                   label.coverage, labeler.grid,
                                   scale=4, show_hud=True)
            cv2.imwrite(str(preview_dir / f"{stamp:020d}_bev_labels.png"),
                        bev)
            color_img = render_color_masks(colors, masks)
            cv2.imwrite(
                str(preview_dir / f"{stamp:020d}_color_masks.jpg"),
                color_img, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
            sheet_bev.append(render_label_bev(
                label.semantic, label.ego, label.coverage, labeler.grid,
                scale=2, show_hud=False))
            sheet_color.append(cv2.resize(
                color_img, (480, 300), interpolation=cv2.INTER_AREA))
        if n_new % 10 == 0:
            el = time.time() - t0
            print(f"  {session_dir.name}: {n_new} labeled "
                  f"({el / max(n_new, 1):.2f} s/frame)", flush=True)

    diag.close()
    if preview_every:
        if sheet_bev:
            cv2.imwrite(str(preview_dir / "contact_sheet.png"),
                        contact_sheet(sheet_bev[-24:], cols=4))
        if sheet_color:
            cv2.imwrite(str(preview_dir / "contact_sheet_colors.jpg"),
                        contact_sheet(sheet_color[-24:], cols=3),
                        [int(cv2.IMWRITE_JPEG_QUALITY), 88])

    agg = summarize_label_files(paths, labeler.grid) if paths else {
        "frames": 0, "cells": 1, "class_cells": {}, "class_fraction": {},
        "ego_fraction": 0.0, "coverage_fraction": 0.0,
    }
    plane_rows = []
    if diag_path.exists():
        with open(diag_path) as f:
            for line in f:
                try:
                    plane_rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    plane_ok = [r for r in plane_rows
                if r.get("plane_ok") and all(r["plane_ok"].values())]
    effective_hz = extracted_hz / max(1, stride)
    stats = {
        "session": session_dir.name,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "frames_extracted": len(index),
        "frames_labeled": len(paths),
        "frames_new_this_run": n_new,
        "frames_resumed": n_skip,
        "frames_missing_inputs": n_missing,
        "label_stride": int(stride),
        "sampled_hz": float(sampled_hz),
        "source_tick_hz": round(source_hz, 3),
        "extracted_hz": round(extracted_hz, 3),
        "effective_label_hz": round(effective_hz, 3),
        "label_rate": round(len(paths) / max(len(index), 1), 4),
        "roles": list(roles),
        **agg,
        "plane_ok_fraction": round(
            len(plane_ok) / max(len(plane_rows), 1), 4),
    }
    with open(preview_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"{session_dir.name}: {len(paths)}/{len(index)} frames labeled "
          f"({stats['label_rate']:.1%}, {effective_hz:.2f} Hz effective) | "
          f"floor {stats['class_fraction'].get('floor', 0):.1%} "
          f"wall {stats['class_fraction'].get('wall', 0):.1%} "
          f"obs {stats['class_fraction'].get('obstacle', 0):.1%} "
          f"haz {stats['class_fraction'].get('hazard', 0):.1%} "
          f"ign {stats['class_fraction'].get('ignore', 0):.1%}",
          flush=True)
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(ROOT / "datasets" / "bev-student"),
                    help="bev-student dataset root (default: %(default)s)")
    ap.add_argument("--session", action="append", default=None,
                    help="session name or path (repeatable; default: all)")
    ap.add_argument("--out", default=None,
                    help="output dir for a single session "
                         "(default: the session dir)")
    ap.add_argument("--role", choices=("near", "far", "both"), default="both",
                    help="camera(s) to run SAM on (default: both)")
    ap.add_argument("--stride", type=int, default=3,
                    help="label every Nth extracted frame (default: 3)")
    ap.add_argument("--sam-version", default="sam3.1",
                    help="SAM teacher checkpoint (default: sam3.1)")
    ap.add_argument("--conf", type=float, default=0.35,
                    help="SAM score threshold (default: 0.35)")
    ap.add_argument("--device", default="cuda",
                    help="torch device for SAM (default: cuda)")
    ap.add_argument("--trt-engine", default=None,
                    help="optional SAM 3.1 TensorRT encoder engine")
    ap.add_argument("--resume", action="store_true",
                    help="skip frames whose label npz already exists")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="maximum new labels per session (0 = all)")
    ap.add_argument("--preview-every", type=int, default=10,
                    help="write review images every N labeled frames "
                         "(0 disables)")
    ap.add_argument("--sampled-hz", type=float, default=10.0,
                    help="extraction frame rate used in the label-rate stat")
    args = ap.parse_args(argv)

    sessions = session_dirs(args.root, args.session)
    if args.out and len(sessions) != 1:
        ap.error("--out requires exactly one --session")
    roles = ("near", "far") if args.role == "both" else (args.role,)
    labeler = BevStudentLabeler(roles=roles)

    from bebop_vision.sam3_concepts import Sam3ConceptSegmenter
    seg = Sam3ConceptSegmenter(
        concepts=list(ALL_PROMPTS), conf=args.conf,
        version=args.sam_version, device=args.device,
        trt_engine=args.trt_engine)

    for session in sessions:
        label_session(
            session, labeler, seg, roles=roles, stride=args.stride,
            resume=args.resume, max_frames=args.max_frames,
            preview_every=args.preview_every, out_dir=args.out,
            sampled_hz=args.sampled_hz)


if __name__ == "__main__":
    main()
