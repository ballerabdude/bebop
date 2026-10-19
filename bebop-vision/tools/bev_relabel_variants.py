"""Re-label a session with several hazard rules using one SAM pass per frame.

Writes parallel label directories (``labels_base2``, ``labels_prompt``,
``labels_local``, ``labels_nohaz``) next to the original ``labels`` so the
training script can compare them without touching the old labels.

    .venv/bin/python tools/bev_relabel_variants.py --stride 3

The SAM masks are identical across variants; only the depth->hazard rule
changes.  Run with ``--sessions`` to limit a quick experiment.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from bebop_vision.bev_student_data import (  # noqa: E402
    DEFAULT_ROOT, read_index, session_dirs)
from bebop_vision.bev_student_labels import (  # noqa: E402
    ALL_PROMPTS, BevStudentLabeler, class_counts, save_label_npz)
from bebop_vision.sam3_concepts import Sam3ConceptSegmenter  # noqa: E402
from bev_student_labels import category_maps  # noqa: E402


VARIANTS = {
    # same rule as the original labels, current SAM checkpoint
    "base2": {},
    # below-plane hazard only where the SAM stairs/step/hole prompt fires
    "prompt": {"hazard_prompt_only": True},
    # no absolute below-plane hazard; only adjacent-cell floor drops
    "local": {"hazard_absolute": False, "local_drop_hazard": True},
    # no hazard at all (floor/obstacle/wall model)
    "nohaz": {"hazard_absolute": False, "local_drop_hazard": False},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--sessions", nargs="*", default=None)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    sessions = session_dirs(root)
    if args.sessions:
        sessions = [s for s in sessions if s.name in args.sessions]
    seg = Sam3ConceptSegmenter(
        concepts=list(ALL_PROMPTS), conf=args.conf, version="sam3.1",
        device="cuda")
    variants = {k: VARIANTS[k] for k in args.variants}

    for session in sessions:
        rows = [r for r in read_index(session)
                if (session / "labels" / f"{int(r['stamp_ns']):020d}.npz")
                .exists()]
        rows = rows[::max(1, args.stride)]
        if args.max_frames:
            rows = rows[:args.max_frames]
        outdirs = {v: session / f"labels_{v}" for v in variants}
        for d in outdirs.values():
            d.mkdir(parents=True, exist_ok=True)
        stats = {v: {"frames": 0, "class_cells": {}} for v in variants}
        diags = {v: open(outdirs[v] / "diagnostics.jsonl", "a")
                 for v in variants}
        t0 = time.time()
        n_done = 0
        for row in rows:
            stamp = int(row["stamp_ns"])
            if args.resume and all(
                    (outdirs[v] / f"{stamp:020d}.npz").exists()
                    for v in variants):
                continue
            depths, colors, masks = {}, {}, {}
            ok = True
            for role in ("near", "far"):
                d = cv2.imread(
                    str(session / f"depth_{role}" / f"{stamp:020d}.png"),
                    cv2.IMREAD_UNCHANGED)
                if d is None:
                    d = cv2.imread(
                        str(session / "depth" / f"{stamp:020d}.png"),
                        cv2.IMREAD_UNCHANGED)
                c = cv2.imread(
                    str(session / f"color_{role}" / f"{stamp:020d}.jpg"),
                    cv2.IMREAD_COLOR)
                if c is None:
                    c = cv2.imread(
                        str(session / "color" / f"{stamp:020d}.jpg"),
                        cv2.IMREAD_COLOR)
                if d is None or c is None:
                    ok = False
                    break
                depths[role] = d
                colors[role] = c
                masks[role] = category_maps(seg.segment(c), c.shape[0],
                                            c.shape[1])
            if not ok:
                continue
            for name, over in variants.items():
                cfg = copy.deepcopy(lab_cfg)
                for k, v in over.items():
                    setattr(cfg, k, v)
                lab.label_cfg = cfg
                lab._rng = np.random.default_rng(0)
                label = lab.label(depths, masks)
                if label is None:
                    continue
                save_label_npz(outdirs[name] / f"{stamp:020d}.npz", label)
                c = class_counts(label.semantic, label.ego, label.coverage)
                st = stats[name]
                st["frames"] += 1
                for k in ("floor", "wall", "obstacle", "hazard",
                          "ignore", "ego", "covered", "cells"):
                    st["class_cells"][k] = st["class_cells"].get(k, 0) + c[k]
                diags[name].write(json.dumps({
                    "stamp_ns": stamp,
                    "plane_ok": label.stats.get("plane_ok", {}),
                    "class_cells": {k: c[k] for k in
                                    ("floor", "wall", "obstacle", "hazard",
                                     "ignore", "covered")},
                }) + "\n")
                diags[name].flush()
            n_done += 1
            if n_done % 10 == 0:
                el = time.time() - t0
                print(f"  {session.name}: {n_done}/{len(rows)} "
                      f"({el / n_done:.2f} s/frame)", flush=True)
        for v in variants:
            diags[v].close()
            st = stats[v]
            cells = max(1, st["class_cells"].get("cells", 0))
            st["class_fraction"] = {
                k: round(st["class_cells"].get(k, 0) / cells, 4)
                for k in ("floor", "wall", "obstacle", "hazard", "ignore")}
            (outdirs[v] / "stats.json").write_text(json.dumps(st, indent=2))
        print(session.name, "done", {
            v: round(stats[v]["class_fraction"].get("hazard", 0), 4)
            for v in variants}, flush=True)


if __name__ == "__main__":
    # Build the labeler once; reused for every frame/variant.
    lab = BevStudentLabeler(roles=("near", "far"))
    lab_cfg = lab.label_cfg
    main()
