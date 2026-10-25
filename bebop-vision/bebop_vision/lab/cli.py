"""Command line for bebop lab.

Thin: every subcommand resolves its session through :mod:`catalog`, does
one thing with one module and prints a table or a path. The notebooks and
scripts call the same functions; the CLI exists so the loop works from a
shell and from `just`.

    python -m bebop_vision.lab ls                      # sessions here (and in the ledger)
    python -m bebop_vision.lab remote                  # sessions on the robot (/captures)
    python -m bebop_vision.lab runs                    # drive runs: task, target, its sessions, outcome
    python -m bebop_vision.lab pull 'navd_session_20261024_*'
    python -m bebop_vision.lab info 20261018_191614
    python -m bebop_vision.lab index 20261018_191614   # build the per-session cache
    python -m bebop_vision.lab ticks 20261018_191614 --hz 10 --head 20
    python -m bebop_vision.lab label 20261018_191614 --variant hindsight
    python -m bebop_vision.lab backbones               # the architecture registry
    python -m bebop_vision.lab features 20261018_191614 --backbone timm:convnext_tiny.dinov3_lvd1689m --role far --size 416x256
    python -m bebop_vision.lab run 20261018_191614 --model cv
    python -m bebop_vision.lab score <run-dir> --labels hindsight
    python -m bebop_vision.lab show 20261018_191614 --run <run-dir> --labels hindsight
    python -m bebop_vision.lab bench export|parity|thor ...

Session arguments accept a name, a stem fragment, a glob or a path.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
from pathlib import Path
from typing import Optional


def _catalog(args):
    from bebop_vision.lab.catalog import Catalog
    kw = {}
    if getattr(args, "root", None):
        kw["roots"] = tuple(args.root)
    if getattr(args, "robot", None):
        kw["robot"] = args.robot
    if getattr(args, "cache", None):
        kw["cache_dir"] = args.cache
    return Catalog(**kw)


def _session(args, name: Optional[str] = None):
    from bebop_vision.lab.session import Session
    path = _catalog(args).resolve(name or args.session)
    overrides = None
    if getattr(args, "pitch_near", None) is not None or getattr(args, "pitch_far", None) is not None:
        overrides = {}
        if args.pitch_near is not None:
            overrides["near"] = {"pitch_deg": args.pitch_near}
        if args.pitch_far is not None:
            overrides["far"] = {"pitch_deg": args.pitch_far}
    return Session(path, cache_dir=getattr(args, "cache", None), overrides=overrides)


def _print_frame(df, limit: Optional[int] = None):
    import polars as pl
    with pl.Config(tbl_rows=limit or 200, tbl_cols=40, tbl_width_chars=200,
                   fmt_str_lengths=60):
        print(df if limit is None else df.head(limit))


def _cuda() -> bool:
    import torch
    return torch.cuda.is_available()


def _parse_size(text: Optional[str]):
    if not text:
        return None
    w, h = text.lower().split("x")
    return int(w), int(h)


# ---------------------------------------------------------------- commands

def cmd_ls(args):
    _print_frame(_catalog(args).local())


def cmd_runs(args):
    _print_frame(_catalog(args).runs(index=args.index))


def cmd_remote(args):
    _print_frame(_catalog(args).remote())


def cmd_pull(args):
    cat = _catalog(args)
    for name, state in cat.pull(args.pattern, finalize=not args.no_finalize):
        print(f"{state:8} {name}")
    # the run log: outcomes picked after Stop driving live only there
    try:
        path = cat.pull_runs()
        print(f"run log  {path}")
    except (OSError, ValueError) as exc:
        print(f"run log  not pulled ({exc})", file=sys.stderr)


def cmd_info(args):
    s = _session(args)
    print(json.dumps(s.summary(), indent=2, default=str))


def cmd_index(args):
    s = _session(args)
    out = s.index(force=args.force)
    print(out)


def cmd_ticks(args):
    s = _session(args)
    t = s.ticks(hz=args.hz)
    print(f"{len(t)} ticks" + (f" at {args.hz} Hz" if args.hz else " (raw)"))
    _print_frame(t.frame(), limit=args.head)


def cmd_label(args):
    from bebop_vision.lab import labels
    s = _session(args)
    ticks = s.ticks(hz=args.hz)
    cfg = labels.HindsightConfig(horizon_s=args.horizon, dt_s=args.dt, heading=args.heading)
    ls = labels.label_session(s, variant=args.variant, ticks=ticks, cfg=cfg)
    n_valid = int(np.asarray(ls.valid).sum()) if hasattr(ls, "valid") else len(ls.t_ns)
    print(f"{args.variant} v{ls.version}: {n_valid}/{len(ls.t_ns)} ticks labelled -> {labels.label_dir(s, args.variant)}")


def cmd_backbones(args):
    from bebop_vision.lab import backbones
    import polars as pl
    rows = backbones.available()
    _print_frame(pl.DataFrame(rows))


def cmd_features(args):
    from bebop_vision.lab import backbones
    from bebop_vision.lab.features import FeatureCache
    s = _session(args)
    ticks = s.ticks(hz=args.hz)
    bb = backbones.get(args.backbone, pretrained=not args.random, weights=args.weights)
    geometry = {"role": args.role, "size": _parse_size(args.size), "crop": None}
    cache = FeatureCache(s.index().parent)
    out = cache.compute(s, ticks, bb, geometry, what=tuple(args.what.split(",")),
                        device=args.device or ("cuda" if _cuda() else "cpu"), batch=args.batch)
    print(out)


def _model_from_args(args):
    from bebop_vision.lab import models
    if args.model == "cv":
        return models.ConstantVelocity(horizon_s=args.horizon, dt_s=args.dt)
    if args.model == "embed":
        from bebop_vision.lab import backbones
        bb = backbones.get(args.backbone, pretrained=not args.random, weights=args.weights)
        return models.EmbeddingModel(bb, modality=f"color_{args.role}", size=_parse_size(args.size))
    raise SystemExit(f"unknown model {args.model!r}: use cv or embed (trained heads run from "
                     "a notebook or script: train.RunDir.load_into(HeadModel) then runner.run)")


def cmd_run(args):
    from bebop_vision.lab import runner
    s = _session(args)
    model = _model_from_args(args)
    ticks = s.ticks(hz=args.hz)
    r = runner.run(s, model, ticks=ticks, device=args.device, progress=True)
    print(r.path)
    print(json.dumps(r.latency_ms(), indent=1))


def cmd_score(args):
    from bebop_vision.lab import labels, metrics, runner
    r = runner.Run.load(Path(args.run))
    sess = r.meta.get("session", {})
    s = _session(args, sess.get("path") or sess.get("name"))
    ls = labels.label_session(s, variant=args.labels, ticks=s.ticks(hz=args.hz))
    _print_frame(metrics.score_paths(r, ls, min_motion_m=args.min_motion))


def cmd_show(args):
    from bebop_vision.lab import rerun_log
    s = _session(args)
    runs = []
    if args.run:
        from bebop_vision.lab import runner
        runs = [runner.Run.load(Path(p)) for p in args.run]
    ls = None
    if args.labels:
        from bebop_vision.lab import labels
        ls = labels.label_session(s, variant=args.labels, ticks=s.ticks(hz=args.hz))
    rerun_log.show(s, runs=runs, labels=ls, spawn=not args.save, save=args.save)


def cmd_bench(args):
    from bebop_vision.lab import bench
    if args.what == "parse":
        print(json.dumps(bench.parse_trtexec(Path(args.file).read_text()), indent=1))
    elif args.what == "thor":
        print(json.dumps(bench.thor_latency(Path(args.file), host=args.host, user=args.user), indent=1))
    else:
        raise SystemExit("bench: use `parse <trtexec.log>` or `thor <model.onnx>`; "
                         "export and parity are library calls (see docs/lab.md)")


# ---------------------------------------------------------------- parser

def build_parser():
    ap = argparse.ArgumentParser(prog="python -m bebop_vision.lab",
                                 description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", action="append", help="session root(s) (repeatable)")
    ap.add_argument("--robot", default=None, help="firmware HTTP base, default http://bebop.local:9090")
    ap.add_argument("--cache", default=None, help="lab cache root (default datasets/lab or $BEBOP_LAB_CACHE)")
    ap.add_argument("--pitch-near", type=float, default=None, help="override the near mount pitch (deg)")
    ap.add_argument("--pitch-far", type=float, default=None, help="override the far mount pitch (deg)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("ls", help="sessions on this machine").set_defaults(fn=cmd_ls)
    sub.add_parser("remote", help="sessions on the robot").set_defaults(fn=cmd_remote)
    p = sub.add_parser("runs", help="drive runs (one task from start to arrived / gave up) over the local sessions")
    p.add_argument("--index", action="store_true", help="index sessions that have no cache yet (one pass each)")
    p.set_defaults(fn=cmd_runs)
    p = sub.add_parser("pull", help="mirror sessions from the robot")
    p.add_argument("pattern", nargs="*", default=["navd_session_*"],
                   help="stems, file names or globs (several allowed)")
    p.add_argument("--no-finalize", action="store_true")
    p.set_defaults(fn=cmd_pull)

    for name, fn, help_ in (("info", cmd_info, "channels, duration, calibration"),
                            ("index", cmd_index, "build the per-session cache")):
        p = sub.add_parser(name, help=help_)
        p.add_argument("session")
        if name == "index":
            p.add_argument("--force", action="store_true")
        p.set_defaults(fn=fn)

    p = sub.add_parser("ticks", help="per-tick table")
    p.add_argument("session")
    p.add_argument("--hz", type=float, default=None)
    p.add_argument("--head", type=int, default=20)
    p.set_defaults(fn=cmd_ticks)

    p = sub.add_parser("label", help="derive labels from the recording")
    p.add_argument("session")
    p.add_argument("--variant", default="hindsight")
    p.add_argument("--hz", type=float, default=10.0)
    p.add_argument("--horizon", type=float, default=3.0)
    p.add_argument("--dt", type=float, default=0.2)
    p.add_argument("--heading", choices=("gyro", "odom"), default="gyro")
    p.set_defaults(fn=cmd_label)

    sub.add_parser("backbones", help="architecture registry").set_defaults(fn=cmd_backbones)

    p = sub.add_parser("features", help="cache frozen features for a backbone")
    p.add_argument("session")
    p.add_argument("--backbone", required=True)
    p.add_argument("--weights", default=None)
    p.add_argument("--random", action="store_true", help="random init (no download)")
    p.add_argument("--role", default="far")
    p.add_argument("--size", default=None, help="WxH resize, e.g. 416x256")
    p.add_argument("--what", default="pooled")
    p.add_argument("--hz", type=float, default=10.0)
    p.add_argument("--device", default=None)
    p.add_argument("--batch", type=int, default=16)
    p.set_defaults(fn=cmd_features)

    p = sub.add_parser("run", help="run a model over a session")
    p.add_argument("session")
    p.add_argument("--model", default="cv", help="cv | embed | <checkpoint.pt>")
    p.add_argument("--backbone", default="timm:convnext_tiny.dinov3_lvd1689m")
    p.add_argument("--weights", default=None)
    p.add_argument("--random", action="store_true")
    p.add_argument("--role", default="far")
    p.add_argument("--size", default=None)
    p.add_argument("--hz", type=float, default=10.0)
    p.add_argument("--horizon", type=float, default=3.0)
    p.add_argument("--dt", type=float, default=0.2)
    p.add_argument("--device", default=None)
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("score", help="score a run against labels")
    p.add_argument("run")
    p.add_argument("--labels", default="hindsight")
    p.add_argument("--hz", type=float, default=10.0, help="tick rate the labels were made at")
    p.add_argument("--min-motion", type=float, default=0.05,
                   help="score only ticks whose labelled path leaves this radius (m); 0 scores all")
    p.set_defaults(fn=cmd_score)

    p = sub.add_parser("show", help="open the session (and runs, labels) in Rerun")
    p.add_argument("session")
    p.add_argument("--run", action="append", default=[])
    p.add_argument("--labels", default=None)
    p.add_argument("--hz", type=float, default=10.0)
    p.add_argument("--save", default=None, help="write an .rrd instead of spawning the viewer")
    p.set_defaults(fn=cmd_show)

    p = sub.add_parser("bench", help="trtexec parsing and on-Thor latency")
    p.add_argument("what", choices=("parse", "thor"))
    p.add_argument("file")
    p.add_argument("--host", default="bebop.local")
    p.add_argument("--user", default="bebop")
    p.set_defaults(fn=cmd_bench)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        args.fn(args)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0
