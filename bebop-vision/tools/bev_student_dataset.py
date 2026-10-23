"""Build and inspect the learned master-BEV dataset / training-run layout.

Typical flow on the workstation::

    # 1. unpack the six robot sessions (10 Hz, both cameras)
    python tools/bev_student_dataset.py extract \
        --mcap-dir datasets/bev-student/raw/mcap

    # 2. SAM 3.1 labels (see tools/bev_student_labels.py)
    python tools/bev_student_labels.py --root datasets/bev-student --stride 3

    # 3. validate + split by drive + report
    python tools/bev_student_dataset.py index
    python tools/bev_student_dataset.py split --val-sessions <one-session>
    python tools/bev_student_dataset.py report

    # 4. (removed 2026-10-23 with the BEV trainer) a run directory with TensorBoard
    python tools/bev_student_dataset.py init-run --run-id dino-vits16-m1
    tensorboard --logdir runs/bev-student

The original MCAPs are never modified.  Splits are by drive/session, never
by frame: adjacent 10 Hz frames are nearly identical and would leak.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bebop_vision.bev_student_data import (  # noqa: E402
    DEFAULT_ROOT, RUNS_ROOT, dataset_report, ensure_layout, ensure_run_layout,
    print_report, session_dirs, split_by_drive, write_dataset_stats,
    write_sessions_manifest,
)


def cmd_extract(args):
    from tools.mcap_extract import extract_bev_student
    paths = ensure_layout(args.root)
    mcaps = []
    if args.mcap:
        mcaps += [Path(p) for p in args.mcap]
    if args.mcap_dir:
        mcaps += sorted(Path(args.mcap_dir).glob("*.mcap"))
    if args.session:
        want = set(args.session)
        mcaps = [p for p in mcaps if p.stem in want]
    if not mcaps:
        raise SystemExit("no MCAP files found")
    for mcap in mcaps:
        out = paths["sessions"] / mcap.stem
        extract_bev_student(mcap, out, sample_hz=args.sample_hz,
                            force=args.force)
    write_sessions_manifest(args.root)
    print(f"layout: {paths['root']}")
    print(f"sessions: {paths['sessions']}")
    print(f"runs: {RUNS_ROOT}")


def cmd_index(args):
    report = dataset_report(args.root)
    write_dataset_stats(report, args.root)
    write_sessions_manifest(args.root)
    print_report(report)
    if report["totals"]["errors"]:
        raise SystemExit(1)


def cmd_split(args):
    payload = split_by_drive(
        args.root, val_sessions=args.val_sessions,
        val_fraction=args.val_fraction, seed=args.seed)
    print(f"train: {payload['train']}")
    print(f"val:   {payload['val']}")
    print(f"split: {Path(args.root) / 'splits' / 'drive_split.json'}")
    report = dataset_report(args.root)
    write_dataset_stats(report, args.root)
    write_sessions_manifest(args.root)
    print_report(report)


def cmd_report(args):
    report = dataset_report(args.root)
    write_dataset_stats(report, args.root)
    write_sessions_manifest(args.root)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_report(report)
    if report["totals"]["errors"]:
        raise SystemExit(1)


def cmd_init_run(args):
    paths = ensure_run_layout(args.run_id, args.runs_root)
    print(f"run:         {paths['run']}")
    print(f"tensorboard: {paths['tb']}")
    print(f"checkpoints: {paths['checkpoints']}")
    print(f"export:      {paths['export']}")
    print(f"view:       tensorboard --logdir {Path(args.runs_root)}")


def cmd_layout(args):
    paths = ensure_layout(args.root)
    print(f"root:     {paths['root']}")
    print(f"raw mcap: {paths['raw']}")
    print(f"sessions: {paths['sessions']}")
    print(f"splits:   {paths['splits']}")
    print(f"meta:     {paths['meta']}")
    print(f"runs:     {RUNS_ROOT}")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("extract", help="MCAP -> bev-student layout")
    p.add_argument("--mcap", action="append", default=None)
    p.add_argument("--mcap-dir", default=str(DEFAULT_ROOT / "raw" / "mcap"))
    p.add_argument("--session", action="append", default=None)
    p.add_argument("--sample-hz", type=float, default=10.0)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_extract)

    p = sub.add_parser("index", help="validate labels and write aggregate stats")
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("split", help="train/val by drive")
    p.add_argument("--val-sessions", nargs="*", default=None)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_split)

    p = sub.add_parser("report", help="print per-session label stats")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("init-run", help="create a TensorBoard run directory")
    p.add_argument("--run-id", required=True)
    p.add_argument("--runs-root", default=str(RUNS_ROOT))
    p.set_defaults(func=cmd_init_run)

    p = sub.add_parser("layout", help="print the dataset/run layout")
    p.set_defaults(func=cmd_layout)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
