"""Train the navd student model on navd-v0 teacher labels (plan §7.2).

    python train_navd.py --data datasets/navd-v0 --out weights/navd_v1

Sessions not named in --val-sessions are train; val reports mIoU over the
3 classes and the best checkpoint is kept.

TensorBoard (optional): pass --tb DIR and every run writes its events to
DIR/<run_tag> (run tag = --out basename + optional --tb-comment), so one
tb root holds one subdirectory per training. Inspect with:

    tensorboard --logdir <tb_root>       # default port 6006

Scalars: per-step total/ce/imitation loss; per-epoch val mIoU, per-class
IoU, learning rate and epoch wall time. Images (--tb-images val samples
of each epoch, stacked into one card per kind): color input, near|far
depth inputs, teacher|prediction, teacher|prediction with the model's
disagreements ringed red, and fusion|hand when the sample carries a
human correction.
Hyperparameters + best mIoU land in the TB hparams panel at the end. The
JSONL log (train_log.jsonl) is unchanged — other tooling reads it.
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bebop_vision.navd import (NavdDataset, NavdUNet, class_weights_from,
                               navd_loss)
from bebop_vision.navd_pre import IMAGENET_MEAN, IMAGENET_STD
import os

# BEV class palette (RGB order, matches add_image): 0 blocked (charcoal),
# 1 navigable (green), 2 caution (amber) — same color story as the
# operator BEV overlay so the panels read at a glance.
CLASS_PALETTE = np.array([[45, 45, 45], [70, 200, 90], [250, 190, 40]],
                         np.uint8)


def miou(logits, label):
    pred = logits.argmax(1)
    ious = []
    for c in range(3):
        inter = ((pred == c) & (label == c)).sum().item()
        union = ((pred == c) | (label == c)).sum().item()
        ious.append(inter / union if union else float("nan"))
    return float(np.nanmean(ious)), ious


def tb_run_tag(out, comment=""):
    """Run tag for the TB panels/directory: the --out basename plus the
    optional --tb-comment ('weights/navd_v3' -> 'navd_v3', or
    'navd_v3_lr1e-4'). Pure string logic, unit-testable."""
    tag = Path(out).name
    return f"{tag}_{comment}" if comment else tag


def tb_resolve_dir(tb_arg, run_tag):
    """--tb may point at a parent dir (the convention) or an explicit run
    dir. Parent (normal case): events go to <tb_arg>/<run_tag> so
    `tensorboard --logdir <tb_arg>` lists one run per training. If the
    path already is a run dir (named for the tag, or holding event files
    from a previous invocation), write there directly."""
    p = Path(tb_arg)
    if p.name == run_tag or any(p.glob("events.out.tfevents.*")):
        return p
    return p / run_tag


def tb_setup(tb_arg, out, comment=""):
    """SummaryWriter for --tb, or None when TB logging is off.

    The tensorboard import is guarded here: a missing optional dep should
    cost one actionable line at startup, not a mid-argparse traceback.
    """
    if not tb_arg:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as e:
        raise SystemExit(
            "--tb needs the tensorboard package; run: pip install tensorboard"
        ) from e
    log_dir = tb_resolve_dir(tb_arg, tb_run_tag(out, comment))
    return SummaryWriter(log_dir=str(log_dir))


def denorm_color(c):
    """[3,H,W] ImageNet-normalized f32 (NavdDataset 'color') -> HxWx3
    uint8 RGB, i.e. the inverse of navd_pre.prep_color, for TB viewing."""
    rgb = np.clip(c * IMAGENET_STD[:, None, None]
                  + IMAGENET_MEAN[:, None, None], 0.0, 1.0)
    return (rgb.transpose(1, 2, 0) * 255.0).round().astype(np.uint8)


def colorize_label(lab):
    """(H,W) int class map -> HxWx3 uint8 palette image (CLASS_PALETTE)."""
    return CLASS_PALETTE[np.asarray(lab).clip(0, 2)]


def label_panel(teacher, pred, upscale=6, sep_px=2):
    """Teacher | prediction as one TB image: palette-colored, nearest-
    upscaled 6x so the 60x60 grid is actually viewable, with a sep_px
    black separator column between the two halves."""
    t, p = colorize_label(teacher), colorize_label(pred)
    sep = np.zeros((t.shape[0], sep_px, 3), np.uint8)
    panel = np.concatenate([t, sep, p], axis=1)
    return cv2.resize(panel, (panel.shape[1] * upscale,
                              panel.shape[0] * upscale),
                      interpolation=cv2.INTER_NEAREST)


def depth_panel(d_near_m, d_far_m, upscale=1, sep_px=2):
    """near | far depth as one TB image: turbo-colormapped over the
    model's input range [0.3, 4] m (beyond 4 m everything reads the same
    — the far field matters as 'something is there', not its exact
    range), invalid (0) = black, half-res like the operator depth view,
    sep_px black separator between the two halves. Inputs are the batch
    tensors (f32 meters, 0 = invalid) exactly as the model sees them."""
    half = (424, 240)
    cols = []
    for d_m in (d_near_m, d_far_m):
        d = cv2.resize(np.asarray(d_m, np.float32), half,
                       interpolation=cv2.INTER_NEAREST)
        valid = d > 0
        norm = np.clip((d - 0.3) / (4.0 - 0.3), 0.0, 1.0)   # already meters
        img = (cv2.applyColorMap((norm * 255).astype(np.uint8),
                                 cv2.COLORMAP_TURBO)[:, :, ::-1]
               .astype(np.float32))   # BGR -> RGB
        img[~valid] = 0
        cols.append(img.astype(np.uint8))
    sep = np.zeros((half[1], sep_px, 3), np.uint8)
    panel = np.concatenate([cols[0], sep, cols[1]], axis=1)
    if upscale > 1:
        panel = cv2.resize(panel, (panel.shape[1] * upscale,
                                   panel.shape[0] * upscale),
                           interpolation=cv2.INTER_NEAREST)
    return panel


def error_panel(teacher, pred, upscale=6, sep_px=2):
    """Where the model disagrees with the teacher (hard-negative view):
    left = teacher, right = pred, and every cell where pred != teacher is
    overlaid bright red on both halves — the mining channel from
    fuse_navd_labels ('disagree') seen through the model's own eyes."""
    wrong = np.asarray(pred) != np.asarray(teacher)
    def half_img(lab):
        img = colorize_label(lab).astype(np.int32)
        img[wrong] = [255, 60, 60]
        return img.astype(np.uint8)
    t, p = half_img(teacher), half_img(pred)
    sep = np.zeros((t.shape[0], sep_px, 3), np.uint8)
    panel = np.concatenate([t, sep, p], axis=1)
    return cv2.resize(panel, (panel.shape[1] * upscale,
                              panel.shape[0] * upscale),
                      interpolation=cv2.INTER_NEAREST)


def hand_panel(fused, hand, upscale=6, sep_px=2):
    """Human correction review: SAM+depth fusion | hand-painted grid.
    Cells where the hand grid differs from the fusion are ringed white on
    the hand half — what the dashboard's paint tool actually changed."""
    changed = np.asarray(hand) != np.asarray(fused)
    def half_img(lab, ring):
        img = colorize_label(lab).astype(np.int32)
        img[ring] = [240, 240, 240]
        return img.astype(np.uint8)
    f, h = half_img(fused, np.zeros_like(changed)), half_img(hand, changed)
    sep = np.zeros((f.shape[0], sep_px, 3), np.uint8)
    panel = np.concatenate([f, sep, h], axis=1)
    return cv2.resize(panel, (panel.shape[1] * upscale,
                              panel.shape[0] * upscale),
                      interpolation=cv2.INTER_NEAREST)


def vstack(panels, sep_px=3):
    """Stack same-width TB panels vertically (sep_px black rows between)
    — one card per image kind showing N val samples instead of N cards."""
    rows = []
    for i, p in enumerate(panels):
        if i:
            rows.append(np.zeros((sep_px, p.shape[1], 3), np.uint8))
        rows.append(p)
    return np.concatenate(rows, axis=0)


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
    ap.add_argument("--tb", default=None, metavar="DIR",
                    help="TensorBoard root dir; events go to DIR/<run_tag>. "
                         "View with: tensorboard --logdir DIR")
    ap.add_argument("--tb-images", type=int, default=10, metavar="N",
                    help="val samples per epoch in the TB image panels, "
                         "stacked top-to-bottom into one card per kind "
                         "(0 disables the image panels)")
    ap.add_argument("--tb-comment", default="",
                    help="optional string merged into the TB run tag")
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
    best_epoch = -1
    log = open(out / "train_log.jsonl", "a")
    # One writer per training (None = TB off); everything is prefixed with
    # the run tag so panels group cleanly if runs share a tb root.
    run_tag = tb_run_tag(args.out, args.tb_comment)
    tb = tb_setup(args.tb, args.out, args.tb_comment)
    step = 0   # global optimizer-step counter across epochs (TB x-axis)

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
            step += 1
            if tb is not None:
                tb.add_scalar(f"{run_tag}/loss/total", loss.item(), step)
                tb.add_scalar(f"{run_tag}/loss/ce", ce.item(), step)
                tb.add_scalar(f"{run_tag}/loss/imitation", im.item(), step)
            if i % 25 == 0:
                print(f"e{epoch} {i}/{len(train_ld)} loss {loss.item():.3f} "
                      f"(ce {ce.item():.3f} im {im.item():.3f})", flush=True)
        model.eval()
        n, mi = 0, 0.0
        ious = np.zeros(3)
        # First --tb-images val samples kept aside for the TB panels
        # (spread across the val set — different scenes, so each epoch's
        # card shows a cross-section, not one scene N times).
        img_samples = []
        with torch.inference_mode():
            for b in val_ld:
                logits = model(b["depth_near"].to(device),
                               b["depth_far"].to(device),
                               b["color"].to(device),
                               b["goal"].to(device))
                if tb is not None and len(img_samples) < args.tb_images:
                    img_samples.append({
                        "color": denorm_color(b["color"][0].cpu().numpy()),
                        "d_near": b["depth_near"][0, 0].cpu().numpy(),
                        "d_far": b["depth_far"][0, 0].cpu().numpy(),
                        "teacher": b["label"][0].cpu().numpy(),
                        "pred": logits.float().argmax(1)[0].cpu().numpy(),
                    })
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
        if tb is not None:
            tb.add_scalar(f"{run_tag}/val/miou", mi, epoch)
            for name, v in zip(("blocked", "navigable", "caution"), ious):
                tb.add_scalar(f"{run_tag}/val/iou_{name}", float(v), epoch)
            tb.add_scalar(f"{run_tag}/train/lr",
                          opt.param_groups[0]["lr"], epoch)
            tb.add_scalar(f"{run_tag}/train/epoch_secs", row["secs"], epoch)
            if img_samples:
                # One card per kind, N val samples stacked top-to-bottom:
                # scroll the card to watch each scene's prediction evolve
                # across epochs (the step slider scrubs epochs).
                tb.add_image(
                    f"{run_tag}/img/color",
                    vstack([s["color"] for s in img_samples]),
                    epoch, dataformats="HWC")
                tb.add_image(
                    f"{run_tag}/img/depth_inputs",
                    vstack([depth_panel(s["d_near"], s["d_far"])
                            for s in img_samples]),
                    epoch, dataformats="HWC")
                tb.add_image(
                    f"{run_tag}/img/teacher_vs_pred",
                    vstack([label_panel(s["teacher"], s["pred"])
                            for s in img_samples]),
                    epoch, dataformats="HWC")
                tb.add_image(
                    f"{run_tag}/img/teacher_vs_pred_errors",
                    vstack([error_panel(s["teacher"], s["pred"])
                            for s in img_samples]),
                    epoch, dataformats="HWC")
            tb.flush()   # keep a live `tensorboard --logdir` in step
        torch.save({"model": model.state_dict(), "epoch": epoch,
                    "val_miou": mi}, out / "last.pt")
        if mi > best:
            best = mi
            best_epoch = epoch
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "val_miou": mi}, out / "best.pt")
            print(f"[epoch {epoch}] new best ({mi:.3f})")
    if tb is not None:
        # Final hparams card so runs are comparable in the TB hparams view.
        tb.add_hparams(
            {"epochs": args.epochs, "batch_size": args.batch_size,
             "lr": args.lr, "imitation_lambda": args.imitation_lambda,
             "data": args.data, "best_epoch": best_epoch},
            {"val/miou": best})
        tb.close()
    print(f"done; best val mIoU {best:.3f} -> {out / 'best.pt'}")


if __name__ == "__main__":
    main()
