"""Path-model experiment over every local drive: which inputs and which
frozen backbone predict the robot's next 3 s best, on drives never seen in
training.

    .venv/bin/python tools/lab_path_experiment.py                 # all configs
    .venv/bin/python tools/lab_path_experiment.py --configs cv,cmd_only,cnxt_t_far

Protocol (bebop lab, docs/lab.md):

- Labels: hindsight paths at 10 Hz, 3 s horizon, gyro heading, with the
  measured mount pitch (near -59.5, far -8.6 deg; the rig YAML says -66 /
  -17.4, see docs/path-model.md section 4).
- Folds: drives are grouped into folds by hand so each fold holds a similar
  number of moving ticks (:data:`FOLDS`); a model is trained on the other
  folds and scored on the held-out one, and the four held-out scores are
  pooled. Never a split inside a drive.
- Validation for checkpoint selection: one drive of the training folds
  (:data:`VAL_DRIVE`), never the test fold.
- Scoring: FDE / ADE at 1, 2, 3 s over ticks whose labelled path leaves a
  5 cm radius (``metrics.per_tick_errors(min_motion_m=0.05)``); still ticks
  are trained on (stopping is behaviour) but not scored.
- Models: a ``PathHead`` on frozen pooled features (cached once per
  backbone x camera x size, ``features.FeatureCache``) plus the commanded
  twist; ``cmd_only`` is the same head on a constant feature, so the gap to
  it is what vision adds; ``cv`` is the constant-velocity arc of the
  commanded twist. Three seeds per learned config.

Writes ``datasets/lab/experiments/<name>/results.json`` and prints the
leaderboard.
"""

from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl  # noqa: E402
import torch  # noqa: E402

from bebop_vision.lab import backbones, labels, models, train  # noqa: E402
from bebop_vision.lab.catalog import Catalog  # noqa: E402
from bebop_vision.lab.dataset import LabDataset  # noqa: E402
from bebop_vision.lab.features import FeatureCache  # noqa: E402
from bebop_vision.lab.metrics import per_tick_errors  # noqa: E402
from bebop_vision.lab.session import Session  # noqa: E402
from bebop_vision.lab.types import InputSpec, Path2D  # noqa: E402

OVERRIDES = {"near": {"pitch_deg": -59.5}, "far": {"pitch_deg": -8.6}}
HZ, HORIZON, DT = 10.0, 3.0, 0.2
MIN_MOTION_M = 0.05
SEEDS = (0, 1, 2)

# Moving ticks per drive (10 Hz, path leaves 5 cm; labels v3), measured
# 2026-10-25: 191614 1409, 193224 1001, 191941 261, 193619 936, 192050 568,
# 043336 28, 043734 369, 212953 252, 193932 285, 215735 220, 214848 52, so
# the folds hold A 1409, B 1262, C 1532, D 1178. 043417 has none and is
# left out.
FOLDS = {
    "A": ["navd_session_20261018_191614"],
    "B": ["navd_session_20261018_193224", "navd_session_20261018_191941"],
    "C": ["navd_session_20261018_193619", "navd_session_20261018_192050",
          "navd_session_20261024_043336"],
    "D": ["navd_session_20261024_043734", "navd_session_20261023_212953",
          "navd_session_20261018_193932", "navd_session_20261023_215735",
          "navd_session_20261023_214848"],
}
# checkpoint selection drive per held-out fold (taken out of training)
VAL_DRIVE = {"A": "navd_session_20261018_193932", "B": "navd_session_20261018_192050",
             "C": "navd_session_20261018_191941", "D": "navd_session_20261018_192050"}

# name -> (backbone or None, camera role, (W, H), use the commanded twist)
CONFIGS = {
    "cv": None,                                                        # constant velocity
    "cmd_only": ("const", None, None, True),
    "cnxt_t_far": ("timm:convnext_tiny.dinov3_lvd1689m", "far", (416, 256), True),
    "cnxt_t_near": ("timm:convnext_tiny.dinov3_lvd1689m", "near", (416, 256), True),
    "cnxt_t_far_nocmd": ("timm:convnext_tiny.dinov3_lvd1689m", "far", (416, 256), False),
    "cnxt_s_far": ("timm:convnext_small.dinov3_lvd1689m", "far", (416, 256), True),
    "resnet50_far": ("timm:resnet50.a1_in1k", "far", (416, 256), True),
    "effvit_b1_far": ("timm:efficientvit_b1.r224_in1k", "far", (416, 256), True),
}


def load_drives(names):
    cat = Catalog()
    out = {}
    for n in names:
        s = Session(cat.resolve(n), overrides=OVERRIDES)
        t = s.ticks(hz=HZ)
        out[n] = (s, t, labels.label_session(s, ticks=t, cfg=labels.HindsightConfig(horizon_s=HORIZON, dt_s=DT)))
    return out


def features_for(drives, bb_name, role, size, device):
    """{drive: (t_ns, pooled)} for one backbone x camera x size."""
    if bb_name == "const":
        return {n: (t.t_ns, np.ones((len(t), 1), np.float32)) for n, (s, t, _) in drives.items()}
    bb = backbones.get(bb_name, pretrained=True)
    out = {}
    for n, (s, t, _) in drives.items():
        cache = FeatureCache(s.index().parent)
        geometry = {"role": role, "size": size, "crop": None}
        key = cache.key(bb, geometry)
        if not cache.has(s, key):
            t0 = time.time()
            cache.compute(s, t, bb, geometry, what=("pooled",), device=device, batch=32)
            print(f"  features {bb_name} {role} {n}: {time.time() - t0:.0f} s", flush=True)
        out[n] = cache.load(s, key, "pooled")
    del bb
    torch.cuda.empty_cache()
    return out


class _Const(torch.nn.Module):
    """Stand-in backbone for the cmd_only head (never run: features are cached)."""
    kind = "pyramid"
    id = "const"
    embed_dim = 1
    feature_info = []

    def pooled(self, feats):
        raise RuntimeError("cmd_only runs on cached features only")


def make_model(name, bb_name, role, size, use_cmd, in_dim, device, residual=False):
    head = models.PathHead(in_dim=in_dim, n_scalars=2 if use_cmd else 0,
                           n_points=int(round(HORIZON / DT)), dt_s=DT,
                           residual="cv" if (residual and use_cmd) else None)
    backbone = _Const() if bb_name == "const" else backbones.get(bb_name, pretrained=False)
    return models.HeadModel(backbone, head, kind="path", modality=f"color_{role or 'far'}", size=size,
                            scalar_modalities=("cmd_vel",) if use_cmd else (), mode="frozen",
                            horizon_s=HORIZON, dt_s=DT, name=name, device=device)


def dataset(drives, names, feats, use_cmd):
    spec = InputSpec(("cmd_vel",))   # frames never decoded: the head reads cached features
    ticks = [drives[n][1] for n in names]
    ds = LabDataset(ticks, spec, labels={n: drives[n][2] for n in names},
                    features={n: feats[n] for n in names})
    if ds.dropped_no_features:
        DROPPED.update({k: max(DROPPED.get(k, 0), v) for k, v in ds.dropped_no_features.items()})
    return ds


#: ticks per drive left out because their frame (so their feature) is missing
DROPPED: dict = {}


@torch.no_grad()
def predict(model, ds, batch=256):
    """{t_ns: Path2D} over a dataset (cached features)."""
    model.eval()
    rel_t = models.rel_t_grid(HORIZON, DT)
    out = {}
    for a in range(0, len(ds), batch):
        items = [ds[i] for i in range(a, min(a + batch, len(ds)))]
        samples = [s for s, _ in items]
        pts, w = model.forward_batch(samples)
        pts, w = pts.float().cpu().numpy(), w.float().cpu().numpy()   # PathHead weights are softmaxed
        for s, p, ww in zip(samples, pts, w):
            out[int(s.t_ns)] = Path2D(rel_t=rel_t, pts=p, weights=ww)
    return out


def cv_predict(drives, names):
    out = {}
    for n in names:
        s, t, _ = drives[n]
        cmd = s.series("cmd_vel")
        for x in t.t_ns:
            r = cmd.nearest(int(x), tol_ns=50_000_000)
            vx, wz = (float(r["vx"]), float(r["wz"])) if r is not None else (0.0, 0.0)
            out[(n, int(x))] = models.constant_velocity_path(vx, wz, HORIZON, DT)
    return out


def errors(preds_by_drive, drives):
    frames = []
    for n, preds in preds_by_drive.items():
        df = per_tick_errors(preds, drives[n][2], (1.0, 2.0, 3.0), min_motion_m=MIN_MOTION_M)
        frames.append(df.with_columns(pl.lit(n).alias("drive")))
    return pl.concat(frames)


def summarise(err: pl.DataFrame) -> dict:
    out = {"n": err.height}
    for c in ("ade@1.0", "ade@3.0", "fde@1.0", "fde@2.0", "fde@3.0"):
        v = err[c].drop_nulls().to_numpy()
        out[c] = {"p50": float(np.percentile(v, 50)), "mean": float(v.mean()),
                  "p90": float(np.percentile(v, 90)), "n": int(v.size)} if v.size else None
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--name", default="path_v0")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--residual", action="store_true",
                    help="heads with the twist predict a correction to the constant-velocity arc")
    args = ap.parse_args(argv)
    names = [n for fold in FOLDS.values() for n in fold]
    out_dir = Path(__file__).resolve().parents[1] / "datasets/lab/experiments" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    drives = load_drives(names)
    print(f"{len(drives)} drives, {sum(len(t) for _, t, _ in drives.values())} ticks", flush=True)

    results = {"protocol": {"folds": FOLDS, "val_drive": VAL_DRIVE, "overrides": OVERRIDES,
                            "hz": HZ, "horizon_s": HORIZON, "dt_s": DT, "min_motion_m": MIN_MOTION_M,
                            "seeds": list(SEEDS), "epochs": args.epochs, "residual": args.residual},
               "configs": {}}
    for cname in args.configs.split(","):
        spec = CONFIGS[cname]
        t0 = time.time()
        if spec is None:
            pooled = cv_predict(drives, names)
            by_drive = {n: {t: p for (d, t), p in pooled.items() if d == n} for n in names}
            err = errors(by_drive, drives)
            results["configs"][cname] = {"pooled": summarise(err), "per_seed": [], "per_fold": {
                f: summarise(err.filter(pl.col("drive").is_in(ds))) for f, ds in FOLDS.items()}}
        else:
            bb_name, role, size, use_cmd = spec
            feats = features_for(drives, bb_name, role, size, args.device)
            in_dim = next(iter(feats.values()))[1].shape[1]
            per_seed, fold_rows = [], {f: [] for f in FOLDS}
            for seed in SEEDS:
                errs = []
                for fold, test in FOLDS.items():
                    val = [VAL_DRIVE[fold]]
                    tr = [n for f, ds in FOLDS.items() if f != fold for n in ds if n not in val]
                    model = make_model(cname, bb_name, role, size, use_cmd, in_dim, args.device,
                                       residual=args.residual)
                    cfg = train.TrainConfig(epochs=args.epochs, lr=1e-3, batch=64, device=args.device,
                                            amp=False, mode="frozen", seed=seed)
                    run_dir = out_dir / "runs" / f"{cname}_s{seed}_{fold}"
                    rd = train.fit(model, dataset(drives, tr, feats, use_cmd),
                                   dataset(drives, val, feats, use_cmd), cfg, train.path_loss, run_dir=run_dir)
                    rd.load_into(model, "best")
                    preds = predict(model, dataset(drives, test, feats, use_cmd))
                    by_drive = {n: {t: p for t, p in preds.items()
                                    if t in set(map(int, drives[n][1].t_ns))} for n in test}
                    e = errors(by_drive, drives)
                    errs.append(e)
                    fold_rows[fold].append(e)
                per_seed.append(summarise(pl.concat(errs)))
            pooled_err = pl.concat([e for rows in fold_rows.values() for e in rows])
            results["configs"][cname] = {
                "backbone": bb_name, "role": role, "size": size, "cmd": use_cmd,
                "pooled": summarise(pooled_err), "per_seed": per_seed,
                "per_fold": {f: summarise(pl.concat(rows)) for f, rows in fold_rows.items()}}
        r = results["configs"][cname]
        if DROPPED:
            r["dropped_no_features"] = dict(DROPPED)
            print(f"  ticks without a frame (dropped): {DROPPED}", flush=True)
            DROPPED.clear()
        fde = r["pooled"]["fde@3.0"]
        seeds = [s["fde@3.0"]["p50"] for s in r["per_seed"]]
        print(f"{cname:18s} FDE@3s p50 {fde['p50']:.3f} m  mean {fde['mean']:.3f}  n {fde['n']}"
              + (f"  seeds {', '.join(f'{x:.3f}' for x in seeds)}" if seeds else "")
              + f"  ({time.time() - t0:.0f} s)", flush=True)
        (out_dir / "results.json").write_text(json.dumps(results, indent=1))

    rows = []
    for cname, r in results["configs"].items():
        p = r["pooled"]
        rows.append({"config": cname,
                     "fde3_p50": p["fde@3.0"]["p50"], "fde3_mean": p["fde@3.0"]["mean"],
                     "fde3_p90": p["fde@3.0"]["p90"], "fde1_p50": p["fde@1.0"]["p50"],
                     "ade3_p50": p["ade@3.0"]["p50"],
                     "seed_spread": (max(s["fde@3.0"]["p50"] for s in r["per_seed"])
                                     - min(s["fde@3.0"]["p50"] for s in r["per_seed"])) if r["per_seed"] else 0.0,
                     **{f"fold_{f}": r["per_fold"][f]["fde@3.0"]["p50"] if r["per_fold"][f]["fde@3.0"] else None
                        for f in FOLDS},
                     # scored ticks per seed (the pooled set repeats them per seed)
                     "n": r["per_seed"][0]["fde@3.0"]["n"] if r["per_seed"] else p["fde@3.0"]["n"]})
    board = pl.DataFrame(rows).sort("fde3_p50")
    with pl.Config(tbl_cols=20, tbl_width_chars=220, float_precision=3):
        print(board)
    (out_dir / "leaderboard.csv").write_text(board.write_csv())
    print(f"done in {time.time() - t_start:.0f} s -> {out_dir}")


if __name__ == "__main__":
    main()
