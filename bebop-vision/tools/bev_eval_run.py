"""Evaluate a trained BEV checkpoint on a split/label variant.

    .venv/bin/python tools/bev_eval_run.py \
        --run runs/bev-student/exp-... --labels-subdir labels_base2

Useful for comparing checkpoints trained against different teacher-label
variants without retraining.  Writes metrics.json and val montages.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from bebop_vision.bev_student import grid_from_model, student_from_checkpoint
from train_bev_student import BevStudentDataset, evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--split", default="val")
    ap.add_argument("--labels-subdir", default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    run = Path(args.run)
    ck = torch.load(run / "checkpoints" / args.ckpt, map_location="cpu",
                    weights_only=False)
    a = ck.get("args") or {}
    labels_subdir = args.labels_subdir or a.get("labels_subdir", "labels")
    unknown = bool(a.get("unknown_class", False))
    single = bool(a.get("single_head", False))
    n = 6 if single else 5 if unknown else 4
    model = student_from_checkpoint(ck).to(args.device)
    model.eval()
    ds = BevStudentDataset(a.get("root"), args.split, train=False,
                           labels_subdir=labels_subdir,
                           unknown_class=unknown, single_head=single,
                           coverage_target=a.get("coverage_target", "depth"),
                           image_hw=model.input_hw, grid=grid_from_model(model))
    ld = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=4, pin_memory=True)
    weights = torch.ones(n, device=args.device)
    val = evaluate(model, ld, args.device, weights, num_classes=n)
    images = val.pop("images", [])
    print(json.dumps(val, indent=2))
    out = Path(args.out) if args.out else run / "eval" / labels_subdir
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.split}_{args.ckpt}.json").write_text(
        json.dumps(val, indent=2))
    for i, img in enumerate(images):
        cv2.imwrite(str(out / f"{args.split}_{args.ckpt}_{i}.jpg"), img,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    print("wrote", out)


if __name__ == "__main__":
    main()
