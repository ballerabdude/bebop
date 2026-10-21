"""Diagnostics for a trained BEV student checkpoint (no hardware).

Reports raw and coverage-masked semantic metrics, hazard precision/recall,
hazard predictions by target/range, and coverage quality.  Saves a montage
of teacher / raw prediction / masked prediction / coverage for review.

    .venv/bin/python tools/bev_eval_diagnostics.py \
        --run runs/bev-student/dino-vits16-full --split val
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from bebop_vision.bev_student import grid_from_model, student_from_checkpoint
from bebop_vision.bev_student_data import DEFAULT_ROOT
from bebop_vision.bev_student_labels import IGNORE
from train_bev_student import BevStudentDataset


CLASS_COLORS = np.array([[120, 120, 120], [255, 120, 40],
                         [40, 40, 230], [0, 160, 255],
                         [18, 18, 18]], np.uint8)  # BGR


def colorize(sem):
    out = np.zeros((*sem.shape, 3), np.uint8)
    for c in range(4):
        out[sem == c] = CLASS_COLORS[c]
    out[sem == 255] = CLASS_COLORS[4]
    return out


def metrics(conf):
    out = {}
    for c, name in enumerate(("floor", "wall", "obstacle", "hazard")):
        tp = conf[c, c]
        fp = conf[:, c].sum() - tp
        fn = conf[c, :].sum() - tp
        out[name] = {
            "iou": float(tp / max(tp + fp + fn, 1)),
            "precision": float(tp / max(tp + fp, 1)),
            "recall": float(tp / max(tp + fn, 1)),
            "tp": int(tp), "fp": int(fp), "fn": int(fn),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--split", default="val")
    ap.add_argument("--labels-subdir", default="labels")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    ap.add_argument("--montage", type=int, default=6)
    args = ap.parse_args()

    run = Path(args.run)
    ck = torch.load(run / "checkpoints" / args.ckpt, map_location="cpu",
                    weights_only=False)
    cfg = ck.get("args") or {}
    model = student_from_checkpoint(ck).to(args.device)
    grid = grid_from_model(model)
    model.eval()
    print("loaded", args.ckpt, "epoch", ck.get("epoch"),
          "stage", ck.get("stage"), flush=True)

    ds = BevStudentDataset(args.root, args.split, train=False,
                           labels_subdir=args.labels_subdir,
                           image_hw=model.input_hw, grid=grid)
    ld = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=4, pin_memory=True)
    conf_raw = np.zeros((4, 4), np.int64)
    conf_cov = np.zeros((4, 4), np.int64)
    cov_tp = cov_fp = cov_fn = 0
    # hazard predictions: (target class, covered?) counts
    haz_pred = np.zeros((5, 2), np.int64)
    haz_true = np.zeros((2,), np.int64)
    # forward-range bins (body x): row -> x
    x, _y = grid.centers()
    bins = [(0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 3.0),
            (3.0, 4.0), (4.0, 6.0)]
    bin_t = np.zeros((len(bins), 2), np.int64)   # target hazard, pred hazard
    bin_all = np.zeros((len(bins),), np.int64)
    mont = []
    frames = 0
    with torch.no_grad():
        for bi, batch in enumerate(ld):
            color = batch["color"].to(args.device)
            depth = batch["depth"].to(args.device)
            label = batch["semantic"].cpu().numpy()
            out = model(color, depth)
            sem = torch.softmax(out["semantic"].float(), 1).cpu().numpy()
            pred = sem.argmax(1).astype(np.uint8)
            if "coverage" in out:
                cov = (torch.sigmoid(out["coverage"][:, 0].float()) > 0.5
                       ).cpu().numpy()
            else:                      # single head: ego=4, unknown=5
                cov = (pred != 5) & (pred != 4)
            # Single-head / 5-class models also predict ego or unknown
            # (ids >= 4); those cells are not covered and stay out of the
            # 4x4 confusion.
            valid = (label != IGNORE) & (pred < 4)
            flat = (label[valid].astype(np.int64) * 4
                    + pred[valid].astype(np.int64))
            conf_raw += np.bincount(flat, minlength=16).reshape(4, 4)
            mask = valid & cov
            flat = (label[mask].astype(np.int64) * 4
                    + pred[mask].astype(np.int64))
            conf_cov += np.bincount(flat, minlength=16).reshape(4, 4)
            ct = batch["coverage"].numpy() > 0.5
            cov_tp += int((cov & ct).sum())
            cov_fp += int((cov & ~ct).sum())
            cov_fn += int((~cov & ct).sum())
            hpred = pred == 3
            for li in range(5):
                m = (label == li) | ((li == 4) & ~valid)
                haz_pred[li, 0] += int((hpred & m & cov).sum())
                haz_pred[li, 1] += int((hpred & m & ~cov).sum())
            haz_true[0] += int(((label == 3) & valid & cov).sum())
            haz_true[1] += int(((label == 3) & valid & ~cov).sum())
            for k, (lo, hi) in enumerate(bins):
                bm = (x >= lo) & (x < hi)
                m2 = valid & bm[None, :, None]
                bin_t[k, 0] += int(((label == 3) & m2).sum())
                bin_t[k, 1] += int((hpred & m2).sum())
                bin_all[k] += int(m2.sum())
            if len(mont) < args.montage:
                for j in range(min(color.shape[0], 2)):
                    teacher = label[j]
                    raw = pred[j]
                    msk = pred[j].copy()
                    msk[~cov[j]] = 255
                    panel = np.concatenate(
                        [colorize(teacher), np.full((grid.rows, 2, 3), 0, np.uint8),
                         colorize(raw), np.full((grid.rows, 2, 3), 0, np.uint8),
                         colorize(msk)], axis=1)
                    covimg = np.zeros((*grid.shape, 3), np.uint8)
                    covimg[cov[j]] = (255, 255, 255)
                    covimg[~cov[j]] = (30, 30, 30)
                    panel = np.concatenate([panel, covimg], axis=1)
                    scale = max(1, min(3, 480 // grid.rows))
                    panel = cv2.resize(panel, (panel.shape[1] * scale,
                                               panel.shape[0] * scale),
                                       interpolation=cv2.INTER_NEAREST)
                    cv2.putText(panel, f"{bi}-{j} teacher|raw|masked|cov",
                                (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                (0, 0, 255), 1, cv2.LINE_AA)
                    mont.append(panel)
                    if len(mont) >= args.montage:
                        break
            frames += color.shape[0]

    raw = metrics(conf_raw)
    cov = metrics(conf_cov)
    cp = cov_tp / max(cov_tp + cov_fp, 1)
    cr = cov_tp / max(cov_tp + cov_fn, 1)
    print(f"frames {frames}")
    print("raw", json.dumps(raw, indent=2))
    print("masked", json.dumps(cov, indent=2))
    print(f"coverage precision {cp:.3f} recall {cr:.3f}")
    print("hazard predictions by target class (covered, uncovered):")
    for i, name in enumerate(("floor", "wall", "obstacle", "hazard",
                              "ignore")):
        print(f"  {name:9s} {haz_pred[i].tolist()}")
    print("hazard target (covered, uncovered):", haz_true.tolist())
    print("range bins: [target hazard, pred hazard, valid cells]")
    for (lo, hi), t, n in zip(bins, bin_t, bin_all):
        print(f"  {lo:.1f}-{hi:.1f}m {t.tolist()} {int(n)}")
    if args.out:
        outdir = Path(args.out)
        outdir.mkdir(parents=True, exist_ok=True)
        res = {"raw": raw, "masked": cov,
               "coverage_precision": cp, "coverage_recall": cr,
               "hazard_pred_by_target": haz_pred.tolist(),
               "hazard_target_cov": haz_true.tolist(),
               "range_bins": [{"lo": lo, "hi": hi,
                               "target_hazard": int(t[0]),
                               "pred_hazard": int(t[1]),
                               "valid": int(n)}
                              for (lo, hi), t, n in zip(bins, bin_t, bin_all)]}
        (outdir / "diagnostics.json").write_text(json.dumps(res, indent=2))
        for i, img in enumerate(mont):
            cv2.imwrite(str(outdir / f"diag_{i}.jpg"), img,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        print("wrote", outdir)


if __name__ == "__main__":
    main()
