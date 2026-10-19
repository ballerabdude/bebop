"""Train the learned semantic master-BEV student (M2/M3).

    python train_bev_student.py \
        --root datasets/bev-student \
        --run runs/bev-student/dino-vits16-full \
        --epochs 50 --stage1-epochs 5 --batch-size 8

Input: both cameras' RGB-D at 240x424.  Output: 120x60 semantic
(floor/wall/obstacle/hazard), ego and coverage heads.  The split is by drive
(`splits/drive_split.json`; the staircase drive is held out by default).

TensorBoard:

    tensorboard --logdir runs/bev-student --port 6006

The first stage freezes the DINOv3 encoder and trains the FPN/BEV lift/
decoder; stage 2 unfreezes the encoder at `lr * --encoder-lr-scale`.
Checkpoints and the run config land in `--run/{checkpoints,config.json}`.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

# Keep the Thor/wkst OpenBLAS pool asleep before numpy/torch.
for _thread_var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_thread_var, "1")

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from bebop_vision.bev_student import (BEV_HW, BevStudent, build_lift_lut,
                                      dummy_lift_lut)
from bebop_vision.bev_student_data import (DEFAULT_ROOT, RUNS_ROOT,
                                           ensure_run_layout, label_path,
                                           read_index, session_dirs)
from bebop_vision.bev_student_labels import (CLASS_NAMES, FLOOR, HAZARD,
                                             IGNORE, OBSTACLE, WALL,
                                             load_label_npz)

IMG_HW = (240, 424)
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
CLASS_PALETTE = np.array([[120, 120, 120], [255, 120, 40],
                          [40, 40, 230], [0, 160, 255],
                          [18, 18, 18]], np.uint8)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class BevStudentDataset(Dataset):
    """One labelled 10 Hz frame pair per item."""

    def __init__(self, root, split, train=False, max_items=0, seed=0):
        self.root = Path(root)
        self.split = split
        self.train = train
        self.rows = []
        split_file = self.root / "splits" / "drive_split.json"
        session_split = {}
        if split_file.exists():
            try:
                session_split = json.loads(
                    split_file.read_text()).get("sessions", {})
            except ValueError:
                session_split = {}
        for sd in session_dirs(root):
            want = session_split.get(sd.name)
            for row in read_index(sd):
                row_split = row.get("split") or want
                if row_split != split:
                    continue
                stamp = int(row["stamp_ns"])
                if not label_path(sd, stamp).exists():
                    continue
                self.rows.append((sd, row))
        if max_items:
            rng = random.Random(seed)
            self.rows = self.rows[:max_items]
            rng.shuffle(self.rows)
        if not self.rows:
            raise RuntimeError(f"no labelled rows for split {split!r}")
        self._rng = random.Random(seed + 17)

    def __len__(self):
        return len(self.rows)

    def _color(self, path):
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (IMG_HW[1], IMG_HW[0]),
                         interpolation=cv2.INTER_AREA)
        x = rgb.astype(np.float32) / 255.0
        x = (x - MEAN) / STD
        return x.transpose(2, 0, 1)

    def _depth(self, path):
        d = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if d is None:
            raise FileNotFoundError(path)
        d = cv2.resize(d, (IMG_HW[1], IMG_HW[0]),
                       interpolation=cv2.INTER_NEAREST)
        m = d.astype(np.float32) * 1e-3 / 8.0
        m = np.clip(m, 0.0, 1.0)
        return m[None]

    def __getitem__(self, idx):
        sd, row = self.rows[idx]
        stamp = int(row["stamp_ns"])
        name = f"{stamp:020d}"
        color, depth = [], []
        for role in ("near", "far"):
            color.append(self._color(sd / f"color_{role}" / f"{name}.jpg"))
            depth.append(self._depth(sd / f"depth_{role}" / f"{name}.png"))
        sem, ego, cov = load_label_npz(label_path(sd, stamp))
        sem = sem.astype(np.int64)

        if self.train:
            rng = self._rng
            if rng.random() < 0.5:
                color = [c[:, :, ::-1].copy() for c in color]
                depth = [d[:, :, ::-1].copy() for d in depth]
                sem = sem[:, ::-1].copy()
                ego = ego[:, ::-1].copy()
                cov = cov[:, ::-1].copy()
            # colour jitter
            gain = rng.uniform(0.88, 1.12)
            contrast = rng.uniform(0.90, 1.10)
            for i, c in enumerate(color):
                c = c * gain
                m = c.mean()
                c = (c - m) * contrast + m
                color[i] = np.clip(c, -2.5, 2.5)
            # depth noise + random dropouts, never on invalid cells
            np_rng = np.random.default_rng(rng.randrange(1 << 30))
            for i, d in enumerate(depth):
                noise = np_rng.normal(0.0, 0.0025, d.shape).astype(np.float32)
                d = np.where(d > 0, np.clip(d + noise, 0.0, 1.0), 0.0)
                for _ in range(int(np_rng.integers(0, 2))):
                    h, w = d.shape[1:]
                    eh = int(np_rng.integers(24, 72))
                    ew = int(np_rng.integers(24, 72))
                    y = int(np_rng.integers(0, h - eh))
                    x = int(np_rng.integers(0, w - ew))
                    d[:, y:y + eh, x:x + ew] = 0.0
                depth[i] = d

        return {
            "color": torch.from_numpy(np.stack(color)),
            "depth": torch.from_numpy(np.stack(depth)),
            "semantic": torch.from_numpy(sem),
            "ego": torch.from_numpy(ego.astype(np.float32)),
            "coverage": torch.from_numpy(cov.astype(np.float32)),
            "stamp_ns": stamp,
            "session": sd.name,
        }


def class_weights_for(dataset):
    counts = np.zeros(4, np.int64)
    for sd, row in dataset.rows:
        sem, _ego, _cov = load_label_npz(
            label_path(sd, int(row["stamp_ns"])))
        v = sem[sem != IGNORE]
        counts += np.bincount(v, minlength=4)[:4]
    freq = np.maximum(counts.astype(np.float64), 1.0)
    w = (freq.sum() / (4.0 * freq))
    w = np.clip(w, 0.5, 8.0)
    return w.astype(np.float32), counts


# ---------------------------------------------------------------------------
# Loss / metrics
# ---------------------------------------------------------------------------


def compute_loss(out, batch, class_weights, device):
    sem = out["semantic"].float()
    label = batch["semantic"].to(device)
    ce = F.cross_entropy(sem, label, weight=class_weights,
                         ignore_index=IGNORE)
    probs = torch.softmax(sem, dim=1)
    valid = (label != IGNORE)
    dice = sem.new_zeros(())
    for cls in (WALL, OBSTACLE):
        p = probs[:, cls] * valid
        t = ((label == cls) & valid).float()
        inter = (p * t).sum()
        denom = p.sum() + t.sum()
        dice = dice + (1.0 - (2.0 * inter + 1.0) / (denom + 1.0))
    dice = dice / 2.0
    ego = out["ego"][:, 0].float()
    cov = out["coverage"][:, 0].float()
    bce_ego = F.binary_cross_entropy_with_logits(
        ego, batch["ego"].to(device),
        pos_weight=torch.tensor(2.0, device=device))
    bce_cov = F.binary_cross_entropy_with_logits(
        cov, batch["coverage"].to(device))
    loss = ce + 0.5 * dice + 0.2 * bce_ego + 0.5 * bce_cov
    return loss, {"loss": float(loss.detach()),
                  "ce": float(ce.detach()),
                  "dice": float(dice.detach()),
                  "ego": float(bce_ego.detach()),
                  "coverage": float(bce_cov.detach())}


def _edge(mask):
    m = mask.float().unsqueeze(1)
    er = F.max_pool2d(m, 3, 1, 1)
    return (m > 0.5) & (er <= 0.5)


def _bands(rows, width=120):
    # x forward: row 0 = 6 m, row 119 = 0.025 m
    x = 6.0 - (torch.arange(rows, device="cpu") + 0.5) * 0.05
    return {
        "0_1m": x < 1.0,
        "1_2m": (x >= 1.0) & (x < 2.0),
        "2_4m": (x >= 2.0) & (x < 4.0),
    }


@torch.no_grad()
def evaluate(model, loader, device, class_weights, max_steps=0):
    model.eval()
    conf = torch.zeros(4, 4, dtype=torch.int64)
    bwall = torch.zeros(3, dtype=torch.int64)  # inter, pred, true
    ego_tp = ego_fp = ego_fn = 0
    cov_tp = cov_fp = cov_fn = 0
    band_stats = {k: [0, 0] for k in ("0_1m", "1_2m", "2_4m")}
    losses = []
    images = []
    bands = _bands(120)
    for i, batch in enumerate(loader):
        if max_steps and i >= max_steps:
            break
        color = batch["color"].to(device)
        depth = batch["depth"].to(device)
        label = batch["semantic"].to(device)
        out = model(color, depth)
        loss, _ = compute_loss(out, batch, class_weights, device)
        losses.append(float(loss) * label.shape[0])
        sem = out["semantic"].float()
        pred = sem.argmax(1)
        valid = label != IGNORE
        flat = (label[valid] * 4 + pred[valid]).cpu()
        conf += torch.bincount(flat, minlength=16).reshape(4, 4)
        # wall boundary
        tw = (label == WALL)
        pw = (pred == WALL)
        te, pe = _edge(tw), _edge(pw)
        bwall += torch.tensor([int((te & pe).sum()), int(pe.sum()),
                               int(te.sum())])
        # ego
        ep = torch.sigmoid(out["ego"][:, 0].float()) > 0.5
        et = batch["ego"].to(device) > 0.5
        ego_tp += int((ep & et).sum())
        ego_fp += int((ep & ~et).sum())
        ego_fn += int((~ep & et).sum())
        # coverage
        cp = torch.sigmoid(out["coverage"][:, 0].float()) > 0.5
        ct = batch["coverage"].to(device) > 0.5
        cov_tp += int((cp & ct).sum())
        cov_fp += int((cp & ~ct).sum())
        cov_fn += int((~cp & ct).sum())
        # obstacle recall by forward band
        for name, bm in bands.items():
            bm = bm.to(device)[None, :, None]
            t = (label == OBSTACLE) & valid & bm
            p = (pred == OBSTACLE) & bm
            band_stats[name][0] += int((t & p).sum())
            band_stats[name][1] += int(t.sum())
        if len(images) < 3:
            images.extend(_val_images(batch, pred))
    conf = conf.numpy()
    ious = {}
    recalls = {}
    for c in range(4):
        tp = conf[c, c]
        fp = conf[:, c].sum() - tp
        fn = conf[c, :].sum() - tp
        union = tp + fp + fn
        ious[CLASS_NAMES[c]] = float(tp / union) if union else float("nan")
        denom = tp + fn
        recalls[CLASS_NAMES[c]] = (float(tp / denom) if denom
                                   else float("nan"))
    vals = [v for v in ious.values() if not np.isnan(v)]
    miou = float(np.mean(vals)) if vals else 0.0
    wall_b = (2.0 * bwall[0] / (bwall[1] + bwall[2] + 1e-9)).item()
    prec = cov_tp / max(cov_tp + cov_fp, 1)
    rec = cov_tp / max(cov_tp + cov_fn, 1)
    return {
        "loss": sum(losses) / max(len(losses), 1),
        "miou": miou,
        "iou": ious,
        "recall": recalls,
        "wall_boundary_iou": wall_b,
        "obstacle_recall": recalls["obstacle"] if not np.isnan(
            recalls["obstacle"]) else 0.0,
        "hazard_recall": recalls["hazard"] if not np.isnan(
            recalls["hazard"]) else 0.0,
        "ego_fp_rate": ego_fp / max(ego_fp + ego_tp + ego_fn, 1),
        "coverage_precision": prec,
        "coverage_recall": rec,
        "obstacle_recall_bands": {
            k: (v[0] / max(v[1], 1)) for k, v in band_stats.items()},
        "confusion": conf.tolist(),
        "images": images,
    }


def _denorm(color):
    rgb = color * STD[:, None, None] + MEAN[:, None, None]
    return (np.clip(rgb, 0, 1).transpose(1, 2, 0) * 255).astype(np.uint8)


def _bev_panel(teacher, pred, scale=4):
    def colorize(lab):
        out = np.zeros((*lab.shape, 3), np.uint8)
        for c in range(4):
            out[lab == c] = CLASS_PALETTE[c]
        out[lab == IGNORE] = CLASS_PALETTE[4]
        return out
    t, p = colorize(teacher), colorize(pred)
    sep = np.zeros((t.shape[0], 2, 3), np.uint8)
    panel = np.concatenate([t, sep, p], axis=1)
    return cv2.resize(panel, (panel.shape[1] * scale,
                              panel.shape[0] * scale),
                      interpolation=cv2.INTER_NEAREST)


def _val_images(batch, pred):
    # one montage per val sample: near | far color, teacher | prediction
    out = []
    for i in range(min(batch["color"].shape[0], 1)):
        near = _denorm(batch["color"][i, 0].cpu().numpy())
        far = _denorm(batch["color"][i, 1].cpu().numpy())
        h = 240
        near = cv2.resize(near, (int(near.shape[1] * h / near.shape[0]), h))
        far = cv2.resize(far, (int(far.shape[1] * h / far.shape[0]), h))
        color = np.concatenate([near, np.zeros((h, 4, 3), np.uint8),
                                far], axis=1)
        panel = _bev_panel(batch["semantic"][i].cpu().numpy(),
                           pred[i].cpu().numpy())
        canvas = np.zeros((color.shape[0] + panel.shape[0] + 4,
                           max(color.shape[1], panel.shape[1]), 3), np.uint8)
        canvas[:color.shape[0], :color.shape[1]] = color
        canvas[color.shape[0] + 4:, :panel.shape[1]] = panel
        out.append(canvas)
    return out


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def make_tb(path):
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise SystemExit("pip install tensorboard") from exc
    return SummaryWriter(log_dir=str(path))


def save_ckpt(path, model, optimizer, scheduler, epoch, stage, best, args,
              metrics):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
        "epoch": epoch, "stage": stage, "best": best,
        "args": vars(args), "metrics": metrics,
    }, path)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--run", default=str(RUNS_ROOT / "dino-vits16-full"))
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--stage1-epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--encoder-lr-scale", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--pretrained-dir", default=None)
    ap.add_argument("--tiny", action="store_true",
                    help="small random DINOv3 backbone + dummy LUT (smoke)")
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--max-val", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    for sub in ("tb", "checkpoints", "export"):
        (run / sub).mkdir(parents=True, exist_ok=True)
    tb = make_tb(run / "tb")

    train_ds = BevStudentDataset(args.root, "train", train=True,
                                 max_items=args.max_train, seed=args.seed)
    val_ds = BevStudentDataset(args.root, "val", train=False,
                               max_items=args.max_val, seed=args.seed + 1)
    cw_np, counts = class_weights_for(train_ds)
    class_weights = torch.tensor(cw_np, device=args.device)
    print(f"train {len(train_ds)} val {len(val_ds)} "
          f"class weights {cw_np.round(2).tolist()} counts {counts.tolist()}",
          flush=True)
    train_ld = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=args.num_workers > 0)
    val_ld = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=max(2, args.num_workers // 2), pin_memory=True)

    lut = dummy_lift_lut() if args.tiny else build_lift_lut()
    model = BevStudent(pretrained_dir=args.pretrained_dir,
                       tiny=args.tiny, lut=lut).to(args.device)
    print(f"model {model.num_parameters()/1e6:.2f} M params; "
          f"LUT {model.lut_meta['groups']}", flush=True)

    start_epoch = 0
    best = -1.0
    stage = 1
    opt = sched = None
    if args.resume and (run / "checkpoints" / "last.pt").exists():
        ck = torch.load(run / "checkpoints" / "last.pt",
                        map_location=args.device, weights_only=False)
        model.load_state_dict(ck["model"])
        start_epoch = int(ck["epoch"]) + 1
        best = float(ck.get("best", -1.0))
        stage = int(ck.get("stage", 1))
        print(f"resumed at epoch {start_epoch} stage {stage} best {best}",
              flush=True)

    scaler = torch.amp.GradScaler(
        "cuda", enabled=args.device.startswith("cuda"))
    config = {
        "args": vars(args),
        "class_weights": cw_np.tolist(),
        "class_counts": counts.tolist(),
        "lut_meta": model.lut_meta,
        "model_params": model.num_parameters(),
        "train_items": len(train_ds),
        "val_items": len(val_ds),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(run / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    step = 0
    for epoch in range(start_epoch, args.epochs):
        if epoch < args.stage1_epochs:
            if stage != 1:
                stage = 1
            model.set_encoder_frozen(True)
            if opt is None or any(p.requires_grad for p in
                                  model.encoder_parameters()):
                params = model.head_parameters()
                opt = torch.optim.AdamW(params, lr=args.lr,
                                        weight_decay=args.weight_decay)
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt, T_max=max(1, args.stage1_epochs - start_epoch))
        else:
            if stage != 2 or opt is None:
                stage = 2
                model.set_encoder_frozen(False)
                opt = torch.optim.AdamW([
                    {"params": model.encoder_parameters(),
                     "lr": args.lr * args.encoder_lr_scale},
                    {"params": model.head_parameters(), "lr": args.lr},
                ], weight_decay=args.weight_decay)
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt, T_max=max(1, args.epochs - max(
                        start_epoch, args.stage1_epochs)))

        model.train()
        t0 = time.time()
        train_loss = 0.0
        n = 0
        for i, batch in enumerate(train_ld):
            if args.max_steps and i >= args.max_steps:
                break
            color = batch["color"].to(args.device, non_blocking=True)
            depth = batch["depth"].to(args.device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda",
                                    enabled=args.device.startswith("cuda")):
                out = model(color, depth)
                loss, parts = compute_loss(
                    out, batch, class_weights, args.device)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(opt)
            scaler.update()
            step += 1
            train_loss += parts["loss"]
            n += 1
            if i % 20 == 0:
                lr = opt.param_groups[0]["lr"]
                tb.add_scalar("train/loss", parts["loss"], step)
                tb.add_scalar("train/ce", parts["ce"], step)
                tb.add_scalar("train/dice", parts["dice"], step)
                tb.add_scalar("train/ego_bce", parts["ego"], step)
                tb.add_scalar("train/coverage_bce", parts["coverage"], step)
                tb.add_scalar("train/lr", lr, step)
                print(f"e{epoch} {i}/{len(train_ld)} "
                      f"loss {parts['loss']:.3f} ce {parts['ce']:.3f} "
                      f"dice {parts['dice']:.3f} lr {lr:.1e}", flush=True)
        if sched is not None:
            sched.step()
        val = evaluate(model, val_ld, args.device, class_weights)
        score = (0.30 * val["wall_boundary_iou"]
                 + 0.35 * val["obstacle_recall"]
                 + 0.30 * val["hazard_recall"]
                 + 0.05 * val["miou"])
        dt = time.time() - t0
        tb.add_scalar("val/loss", val["loss"], epoch)
        tb.add_scalar("val/miou", val["miou"], epoch)
        for name, v in val["iou"].items():
            tb.add_scalar(f"val/iou_{name}", v, epoch)
        tb.add_scalar("val/wall_boundary_iou",
                      val["wall_boundary_iou"], epoch)
        tb.add_scalar("val/obstacle_recall", val["obstacle_recall"], epoch)
        tb.add_scalar("val/hazard_recall", val["hazard_recall"], epoch)
        tb.add_scalar("val/ego_fp_rate", val["ego_fp_rate"], epoch)
        tb.add_scalar("val/coverage_precision",
                      val["coverage_precision"], epoch)
        tb.add_scalar("val/coverage_recall",
                      val["coverage_recall"], epoch)
        for band, v in val["obstacle_recall_bands"].items():
            tb.add_scalar(f"val/obstacle_recall_{band}", v, epoch)
        for k, img in enumerate(val["images"]):
            tb.add_image(f"val/sample_{k}", img, epoch, dataformats="HWC")
        tb.add_scalar("val/score", score, epoch)
        tb.flush()
        print(f"[epoch {epoch}] stage {stage} train {train_loss/max(n,1):.3f}"
              f" | val {val['loss']:.3f} mIoU {val['miou']:.3f} "
              f"wall {val['iou']['wall']:.3f} obsR {val['obstacle_recall']:.3f}"
              f" hazR {val['hazard_recall']:.3f} covP "
              f"{val['coverage_precision']:.3f} covR "
              f"{val['coverage_recall']:.3f} | {dt:.1f}s", flush=True)
        ckdir = run / "checkpoints"
        save_ckpt(ckdir / "last.pt", model, opt, sched, epoch, stage,
                  best, args, val)
        if score > best:
            best = score
            save_ckpt(ckdir / "best.pt", model, opt, sched, epoch, stage,
                      best, args, val)
            print(f"[epoch {epoch}] new best score {best:.3f}", flush=True)
    tb.add_hparams(
        {"epochs": args.epochs, "batch_size": args.batch_size,
         "lr": args.lr, "stage1_epochs": args.stage1_epochs,
         "run": run.name},
        {"val/score": best, "val/miou": val["miou"],
         "val/obstacle_recall": val["obstacle_recall"],
         "val/hazard_recall": val["hazard_recall"]})
    tb.close()
    print(f"done; best safety score {best:.3f}; "
          f"checkpoints {run / 'checkpoints'}", flush=True)


if __name__ == "__main__":
    main()
