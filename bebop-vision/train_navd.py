"""Train the navd student model on navd-v0 teacher labels (plan §7.2).

    python train_navd.py --data datasets/navd-v0 --out weights/navd_v1

Sessions not named in --val-sessions are train; val reports mIoU over the
3 classes and the best checkpoint is kept.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bebop_vision.navd import (NavdDataset, NavdUNet, class_weights_from,
                               navd_loss)


def miou(logits, label):
    pred = logits.argmax(1)
    ious = []
    for c in range(3):
        inter = ((pred == c) & (label == c)).sum().item()
        union = ((pred == c) | (label == c)).sum().item()
        ious.append(inter / union if union else float("nan"))
    return float(np.nanmean(ious)), ious


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/navd-v0")
    ap.add_argument("--out", default="weights/navd_v1")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--imitation-lambda", type=float, default=0.2)
    ap.add_argument("--limit", type=int, default=0,
                    help="debug: cap train samples per session")
    ap.add_argument("--val-sessions", default="175638,175918")
    args = ap.parse_args()

    root = Path(args.data)
    sessions = sorted(p for p in root.glob("navd_session_*") if p.is_dir())
    val_ids = {s.strip() for s in args.val_sessions.split(",") if s.strip()}
    train_dirs = [s for s in sessions
                  if not any(v in s.name for v in val_ids)]
    val_dirs = [s for s in sessions if any(v in s.name for v in val_ids)]
    print(f"train {len(train_dirs)} sessions, val {len(val_dirs)}: "
          f"{[s.name for s in val_dirs]}")

    train_ds = NavdDataset(train_dirs, augment=True)
    if args.limit:
        train_ds.items = [it for it in train_ds.items
                          if train_ds.items.index(it) % 10 == 0]
    val_ds = NavdDataset(val_dirs, augment=False)
    print(f"train {len(train_ds)} samples, val {len(val_ds)} samples")
    w_cls = class_weights_from(train_dirs)
    print(f"class weights {w_cls.tolist()}")

    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True,
                          drop_last=True, persistent_workers=True)
    val_ld = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)

    device = "cuda"
    model = NavdUNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * max(len(train_ld), 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=steps, pct_start=0.2)
    scaler = torch.amp.GradScaler("cuda")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    best = -1.0
    log = open(out / "train_log.jsonl", "a")

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        for i, b in enumerate(train_ld):
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                logits = model(b["depth_near"].to(device),
                               b["depth_far"].to(device),
                               b["color"].to(device),
                               b["goal"].to(device))
                loss, ce, im = navd_loss(
                    logits.float(), b["label"].to(device),
                    b["imitation_target"].to(device), w_cls,
                    lam=args.imitation_lambda)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            if i % 25 == 0:
                print(f"e{epoch} {i}/{len(train_ld)} loss {loss.item():.3f} "
                      f"(ce {ce.item():.3f} im {im.item():.3f})", flush=True)
        model.eval()
        n, mi = 0, 0.0
        ious = np.zeros(3)
        with torch.inference_mode():
            for b in val_ld:
                logits = model(b["depth_near"].to(device),
                               b["depth_far"].to(device),
                               b["color"].to(device),
                               b["goal"].to(device))
                m, io = miou(logits.float(),
                             b["label"].to(logits.device))
                mi += m * b["label"].shape[0]
                ious += np.array(io) * b["label"].shape[0]
                n += b["label"].shape[0]
        mi /= max(n, 1)
        ious /= max(n, 1)
        row = {"epoch": epoch, "val_miou": mi, "ious": ious.tolist(),
               "secs": round(time.time() - t0, 1)}
        print(f"[epoch {epoch}] val mIoU {mi:.3f} | per-class "
              f"{np.round(ious, 3).tolist()} | {row['secs']}s")
        log.write(json.dumps(row) + "\n")
        log.flush()
        torch.save({"model": model.state_dict(), "epoch": epoch,
                    "val_miou": mi}, out / "last.pt")
        if mi > best:
            best = mi
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "val_miou": mi}, out / "best.pt")
            print(f"[epoch {epoch}] new best ({mi:.3f})")
    print(f"done; best val mIoU {best:.3f} -> {out / 'best.pt'}")


if __name__ == "__main__":
    main()
