"""Sweep the hazard probability threshold with the live coverage mask.

The runtime currently uses semantic argmax.  Hazard is rare and poorly
calibrated, so a separate threshold can trade false orange for missed drops.
Reports precision/recall/F1/IoU on the held-out drive for each threshold.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader

from bebop_vision.bev_student import grid_from_model, student_from_checkpoint
from train_bev_student import BevStudentDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--labels-subdir", default="labels_base2")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    run = Path(args.run)
    ck = torch.load(run / "checkpoints" / args.ckpt, map_location="cpu",
                    weights_only=False)
    a = ck.get("args") or {}
    model = student_from_checkpoint(ck).to(args.device)
    model.eval()
    ds = BevStudentDataset(a.get("root"), "val", train=False,
                           labels_subdir=args.labels_subdir,
                           image_hw=model.input_hw, grid=grid_from_model(model))
    ld = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=4, pin_memory=True)
    ps, covs, labels = [], [], []
    with torch.no_grad():
        for batch in ld:
            out = model(batch["color"].to(args.device),
                        batch["depth"].to(args.device))
            p = torch.softmax(out["semantic"].float(), 1)[:, 3].cpu().numpy()
            if "coverage" in out:
                c = (torch.sigmoid(out["coverage"][:, 0].float()) > 0.5
                     ).cpu().numpy()
            else:                      # single head: ego=4, unknown=5
                am = out["semantic"].argmax(1).cpu().numpy()
                c = (am != 5) & (am != 4)
            ps.append(p)
            covs.append(c)
            labels.append(batch["semantic"].numpy())
    p = np.concatenate(ps)
    cov = np.concatenate(covs)
    lab = np.concatenate(labels)
    valid = lab != 255
    covered = cov & valid
    target = (lab == 3) & covered
    rows = []
    for t in np.arange(0.05, 0.96, 0.05):
        ph = (p > t) & covered
        tp = int((ph & target).sum())
        fp = int((ph & ~target).sum())
        fn = int((target & ~ph).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        iou = tp / max(tp + fp + fn, 1)
        rows.append({"t": round(float(t), 2), "tp": tp, "fp": fp, "fn": fn,
                     "precision": prec, "recall": rec, "f1": f1,
                     "iou": iou})
    best_f1 = max(rows, key=lambda r: r["f1"])
    best_iou = max(rows, key=lambda r: r["iou"])
    # argmax reference
    arg = (p > 0.5) & covered
    tp = int((arg & target).sum()); fp = int((arg & ~target).sum())
    fn = int((target & ~arg).sum())
    argmax = {"tp": tp, "fp": fp, "fn": fn,
              "precision": tp / max(tp + fp, 1),
              "recall": tp / max(tp + fn, 1),
              "f1": 2 * tp / max(2 * tp + fp + fn, 1),
              "iou": tp / max(tp + fp + fn, 1)}
    out = {"run": run.name, "target_cells": int(target.sum()),
           "coverage_cells": int(covered.sum()),
           "argmax": argmax, "sweep": rows,
           "best_f1": best_f1, "best_iou": best_iou}
    print(json.dumps(out, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
