"""Crop an existing label set to a shorter forward range (same cells).

    .venv/bin/python tools/bev_crop_labels.py --source labels_grid360x640w4 \
        --range-m 4.0 --out labels_grid240x640w4r4

Row 0 is the far edge, so a shorter range keeps the LAST rows of every
label: with 360 rows over 6 m (1.667 cm), 4 m is rows 120..359.  Nothing is
resampled; semantic/ego/coverage are copied cell for cell, and the new
``grid.json`` records the cropped grid with the same label configuration.
The requested range must be a whole number of source rows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from bebop_vision.bev_student_data import DEFAULT_ROOT, session_dirs
from bebop_vision.bev_student_labels import GridSpec, load_label_npz, validate_label_arrays


def crop_rows(src_grid, range_m):
    """Number of far rows to drop for ``range_m`` (exact, else ValueError)."""
    rows_keep = range_m / src_grid.cell_x_m
    if abs(rows_keep - round(rows_keep)) > 1e-6 or not (0 < range_m <= src_grid.range_m):
        raise ValueError(
            f"range {range_m} m is not a whole number of {src_grid.cell_x_m:.5f} m "
            f"rows within {src_grid.range_m} m")
    return src_grid.rows - int(round(rows_keep))


def crop_session(session, source, out_name, range_m):
    src = session / source
    meta_path = src / "grid.json"
    if not meta_path.exists():
        return 0
    meta = json.loads(meta_path.read_text())
    g = meta["grid"]
    src_grid = GridSpec(range_m=g["range_m"], width_m=g["width_m"],
                        rows_override=g["rows"], cols_override=g["cols"])
    drop = crop_rows(src_grid, range_m)
    grid = GridSpec(range_m=range_m, width_m=src_grid.width_m,
                    rows_override=src_grid.rows - drop,
                    cols_override=src_grid.cols)
    out = session / out_name
    out.mkdir(exist_ok=True)
    new_meta = dict(meta)
    new_meta["grid"] = grid.metadata()
    new_meta["cropped_from"] = {"labels": source, "range_m": src_grid.range_m,
                                "rows_dropped_far": drop}
    out_meta = out / "grid.json"
    if out_meta.exists() and json.loads(out_meta.read_text()) != new_meta:
        raise ValueError(f"incompatible existing label metadata: {out_meta}")
    out_meta.write_text(json.dumps(new_meta, indent=2) + "\n")
    n = 0
    for path in sorted(src.glob("*.npz")):
        sem, ego, cov = load_label_npz(path, src_grid)
        sem, ego, cov = sem[drop:], ego[drop:], cov[drop:]
        validate_label_arrays(sem, ego, cov, grid)
        tmp = (out / path.name).with_suffix(".tmp.npz")
        np.savez_compressed(tmp, semantic=sem.astype(np.uint8),
                            ego=ego.astype(bool), coverage=cov.astype(bool))
        tmp.replace(out / path.name)
        n += 1
    for extra in ("diagnostics.jsonl", "stats.json"):
        if (src / extra).exists():
            (out / extra).write_bytes((src / extra).read_bytes())
    print(f"{session.name}: {n} labels -> {out_name} "
          f"({grid.rows}x{grid.cols}, {range_m} m, dropped {drop} far rows)",
          flush=True)
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--source", required=True)
    ap.add_argument("--range-m", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sessions", nargs="*", default=None)
    args = ap.parse_args()
    if (Path(args.out).name != args.out or not args.out.startswith("labels_grid")
            or args.out == args.source):
        ap.error("output must be a separate labels_grid... directory name")
    total = 0
    for session in session_dirs(args.root):
        if args.sessions and session.name not in args.sessions:
            continue
        total += crop_session(session, args.source, args.out, args.range_m)
    print(f"CROP_LABELS_DONE {args.out}: {total} labels", flush=True)


if __name__ == "__main__":
    main()
