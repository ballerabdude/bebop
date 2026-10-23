"""Dataset plumbing for the learned master-BEV student.

The on-disk contract (all under ``bebop-vision/datasets/bev-student/``)::

    raw/mcap/<session>.mcap          original robot sessions (never modified)
    sessions/<session>/
        color_near/<stamp>.jpg
        color_far/<stamp>.jpg
        depth_near/<stamp>.png       uint16 mm, lossless
        depth_far/<stamp>.png
        labels/<stamp>.npz           semantic uint8 (120,60), ego, coverage
        index.jsonl                  one row per extracted 10 Hz frame
        calib.json
        meta.json
        preview/                     review artifacts
    splits/drive_split.json          train/val by drive, never by frame
    meta/label_stats.json            aggregate label stats


    runs/bev-student/<run_id>/
        tb/                          TensorBoard event files
        checkpoints/
        export/                      ONNX / TensorRT / metadata
        train_log.jsonl
        config.json
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np

from .bev_student_labels import (CLASS_NAMES, GridSpec, load_label_npz,
                                 summarize_label_files)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "datasets" / "bev-student"


def ensure_layout(root=DEFAULT_ROOT):
    """Create the dataset directory skeleton and return the paths."""
    root = Path(root)
    for sub in ("raw/mcap", "sessions", "splits", "meta"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    return {
        "root": root,
        "raw": root / "raw" / "mcap",
        "sessions": root / "sessions",
        "splits": root / "splits",
        "meta": root / "meta",
    }



def session_dirs(root=DEFAULT_ROOT):
    base = Path(root) / "sessions"
    if not base.is_dir():
        return []
    return sorted(p for p in base.iterdir() if p.is_dir())


def read_index(session_dir):
    """Read index.jsonl, or synthesize one from on-disk frames."""
    session_dir = Path(session_dir)
    path = session_dir / "index.jsonl"
    if path.exists():
        rows = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    stamps = None
    for sub in ("depth_near", "depth_far"):
        d = session_dir / sub
        s = {int(p.stem) for p in d.glob("*.png")} if d.is_dir() else set()
        stamps = s if stamps is None else stamps & s
    rows = []
    for stamp in sorted(stamps or []):
        rows.append({
            "stamp_ns": stamp,
            "stamp_us": stamp // 1000,
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


def label_path(session_dir, stamp_ns, labels_subdir="labels"):
    return (Path(session_dir) / str(labels_subdir)
            / f"{int(stamp_ns):020d}.npz")


def find_labels(session_dir):
    labels = Path(session_dir) / "labels"
    if not labels.is_dir():
        return []
    return sorted(labels.glob("*.npz"))


def validate_session(session_dir, grid=None, strict=True):
    """Check paths, label contract and census for one session."""
    session_dir = Path(session_dir)
    grid = grid or GridSpec()
    rows = read_index(session_dir)
    errors = []
    missing_color = 0
    missing_depth = 0
    labels = []
    stamps = {int(r["stamp_ns"]) for r in rows}
    for row in rows:
        stamp = int(row["stamp_ns"])
        for key in ("color_near", "color_far"):
            p = session_dir / (row.get(key) or
                               f"{key}/{stamp:020d}.jpg")
            if not p.exists():
                missing_color += 1
        for key in ("depth_near", "depth_far"):
            p = session_dir / (row.get(key) or
                               f"{key}/{stamp:020d}.png")
            if not p.exists():
                missing_depth += 1
        lp = label_path(session_dir, stamp)
        if lp.exists():
            try:
                load_label_npz(lp, grid)
                labels.append(lp)
            except ValueError as exc:
                errors.append(f"{lp.name}: {exc}")

    known = {p.stem for p in labels}
    orphan = [p.name for p in find_labels(session_dir)
              if p.stem not in {f"{s:020d}" for s in stamps}]
    if missing_color:
        errors.append(f"{missing_color} missing color files")
    if missing_depth:
        errors.append(f"{missing_depth} missing depth files")
    if orphan and strict:
        errors.append(f"{len(orphan)} labels without an index row")
    result = {
        "session": session_dir.name,
        "frames": len(rows),
        "labeled": len(labels),
        "label_rate": round(len(labels) / max(len(rows), 1), 4),
        "missing_color": missing_color,
        "missing_depth": missing_depth,
        "orphan_labels": orphan,
        "errors": errors,
    }
    sp = session_dir / "preview" / "stats.json"
    if sp.exists():
        try:
            result["label_stats"] = json.loads(sp.read_text())
        except ValueError:
            result["label_stats"] = {}
    if labels:
        result["stats"] = summarize_label_files(labels, grid)
    return result


def dataset_report(root=DEFAULT_ROOT, grid=None):
    grid = grid or GridSpec()
    sessions = []
    totals = {
        "frames": 0, "labeled": 0, "missing_color": 0,
        "missing_depth": 0, "errors": 0,
    }
    class_cells = {name: 0 for name in CLASS_NAMES.values()}
    cells = ego_covered = covered = 0
    for sd in session_dirs(root):
        r = validate_session(sd, grid)
        sessions.append(r)
        for key in totals:
            totals[key] += r.get(key, 0) if key != "errors" else len(r["errors"])
        st = r.get("stats")
        if st:
            for name, v in st["class_cells"].items():
                class_cells[name] += int(v)
            cells += int(st["cells"])
            covered += int(round(st["coverage_fraction"] * st["cells"]))
            ego_covered += int(round(st["ego_fraction"] * st["cells"]))
    cells = max(1, cells)
    totals["label_rate"] = round(totals["labeled"] / max(totals["frames"], 1),
                                 4)
    totals["class_fraction"] = {k: round(v / cells, 4)
                                for k, v in class_cells.items()}
    totals["coverage_fraction"] = round(covered / cells, 4)
    totals["ego_fraction"] = round(ego_covered / cells, 4)
    return {
        "root": str(root),
        "sessions": sessions,
        "totals": totals,
        "class_cells": class_cells,
        "cells": cells,
    }


def split_by_drive(root=DEFAULT_ROOT, val_sessions=None, val_fraction=0.2,
                   seed=0):
    """Assign whole sessions to train/val and stamp each index row.

    Splitting by frame leaks near-identical neighbouring frames; this tool
    only ever moves complete drives.
    """
    root = Path(root)
    grid = GridSpec()
    usable = []
    for sd in session_dirs(root):
        if find_labels(sd):
            usable.append(sd.name)
    if not usable:
        raise RuntimeError("no labeled sessions found")
    if val_sessions:
        val = [str(v) for v in val_sessions]
        unknown = sorted(set(val) - set(usable))
        if unknown:
            raise ValueError(f"unknown/unlabeled val sessions: {unknown}")
    elif len(usable) == 1:
        val = []
    else:
        n_val = min(len(usable) - 1,
                    max(1, int(round(len(usable) * float(val_fraction)))))
        val = sorted(random.Random(seed).sample(usable, n_val))
    train = sorted(set(usable) - set(val))
    split = {name: ("val" if name in val else "train") for name in usable}

    for sd in session_dirs(root):
        idx = sd / "index.jsonl"
        if not idx.exists():
            continue
        rows = read_index(sd)
        for row in rows:
            row["split"] = split.get(sd.name)
        tmp = idx.with_suffix(".jsonl.tmp")
        with open(tmp, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        tmp.replace(idx)

    payload = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seed": int(seed),
        "val_fraction": float(val_fraction),
        "sessions": split,
        "train": train,
        "val": val,
        "note": "split by drive/session; never by frame",
    }
    out = root / "splits" / "drive_split.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    return payload


def write_sessions_manifest(root=DEFAULT_ROOT):
    """One JSON row per drive: source size, frame rates, labels, split."""
    root = Path(root)
    split = {}
    split_path = root / "splits" / "drive_split.json"
    if split_path.exists():
        try:
            split = json.loads(split_path.read_text()).get("sessions", {})
        except ValueError:
            split = {}
    rows = []
    for sd in session_dirs(root):
        meta = {}
        mp = sd / "meta.json"
        if mp.exists():
            try:
                meta = json.loads(mp.read_text())
            except ValueError:
                meta = {}
        label_stats = {}
        sp = sd / "preview" / "stats.json"
        if sp.exists():
            try:
                label_stats = json.loads(sp.read_text())
            except ValueError:
                label_stats = {}
        rows.append({
            "session": sd.name,
            "source_mcap": meta.get("source_mcap"),
            "source_size": meta.get("source_size"),
            "duration_s": meta.get("duration_s"),
            "source_tick_hz": meta.get("source_tick_hz"),
            "extracted_hz": meta.get("extracted_hz"),
            "frames": meta.get("frames"),
            "labeled": len(find_labels(sd)),
            "effective_label_hz": label_stats.get("effective_label_hz"),
            "split": split.get(sd.name),
        })
    payload = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "root": str(root),
        "sessions": rows,
    }
    out = root / "meta" / "sessions.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    return out


def write_dataset_stats(report, root=DEFAULT_ROOT):
    out = Path(root) / "meta" / "label_stats.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    flat = []
    for s in report.get("sessions", []):
        row = {k: v for k, v in s.items() if k != "stats"}
        if "stats" in s:
            row.update({f"stats_{k}": v for k, v in s["stats"].items()})
        flat.append(row)
    payload = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "root": str(root),
        "totals": report.get("totals", {}),
        "class_cells": report.get("class_cells", {}),
        "sessions": flat,
    }
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    return out


def print_report(report):
    tot = report["totals"]
    print(f"{'session':32s} {'frames':>7s} {'labeled':>8s} "
          f"{'label%':>7s} {'Hz':>6s} {'floor':>6s} {'wall':>6s} "
          f"{'obs':>6s} {'haz':>6s} {'ign':>6s}")
    for s in report["sessions"]:
        f = (s.get("stats") or {}).get("class_fraction", {})
        hz = (s.get("label_stats") or {}).get("effective_label_hz", 0.0)
        print(f"{s['session'][:32]:32s} {s['frames']:7d} "
              f"{s['labeled']:8d} {100*s['label_rate']:6.1f}% "
              f"{hz:6.2f} "
              f"{100*f.get('floor', 0):5.1f}% "
              f"{100*f.get('wall', 0):5.1f}% "
              f"{100*f.get('obstacle', 0):5.1f}% "
              f"{100*f.get('hazard', 0):5.1f}% "
              f"{100*f.get('ignore', 0):5.1f}%")
        if s["errors"]:
            for e in s["errors"][:4]:
                print(f"  ! {e}")
    f = report["totals"].get("class_fraction", {})
    print(f"{'TOTAL':32s} {tot['frames']:7d} {tot['labeled']:8d} "
          f"{100*tot['label_rate']:6.1f}% "
          f"{100*f.get('floor', 0):5.1f}% {100*f.get('wall', 0):5.1f}% "
          f"{100*f.get('obstacle', 0):5.1f}% {100*f.get('hazard', 0):5.1f}% "
          f"{100*f.get('ignore', 0):5.1f}%")
    if tot.get("errors"):
        print(f"{tot['errors']} validation error(s)")
