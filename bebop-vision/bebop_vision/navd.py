"""navd student model (plan §7.2): depth+color+goal -> 60x60 BEV label.

Inputs (per spec):
  depth_near [1,1,240,424] f32 meters, clipped [0.3, 6.0], 0 = invalid
  depth_far  [1,1,240,424] same
  color      [1,3,240,424] f32, ImageNet-normalized
  goal       [1,1,60,60]  f32 unit-gradient fan toward the goal heading
Output:
  logits     [1,3,60,60]  (0 blocked / 1 navigable / 2 caution)

Training target = the v2 floor-anchored teacher labels produced by
tools/fuse_navd_labels.py (SAM floor + geometric + YOLO fusion). The
teacher stack (SAM, YOLO, RANSAC) runs workstation-side only; at runtime
this model replaces all of it with one forward pass (~10 Hz on the Orin).

Loss: class-weighted CE + imitation term — CE between the per-ray
navigable-probability distribution implied by the prediction and a soft
target on the ray the operator's twist was steering toward (v1 proxy:
atan2(0.5*wz, vx); skipped when vx ~ 0). The full planner-score
distillation can replace this later without touching the data.
"""

import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

IMG_H, IMG_W = 240, 424
GRID = 60
RANGE_M, WIDTH_M, CELL_M = 3.0, 3.0, 0.05
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
RAY_ANGLES = np.deg2rad(np.arange(-60.0, 61.0, 10.0))   # 13 rays


def build_goal_raster(goal, odom):
    """goal dict from the manifest {type: heading|point|none, ...} ->
    (60, 60) float32 fan: 1 along the goal bearing from the robot origin,
    fading with angular distance."""
    g = np.zeros((GRID, GRID), np.float32)
    if goal.get("type", "none") == "heading":
        bearing = float(goal["heading_rad"])
    elif goal.get("type") == "point":
        gx, gy = float(goal["x"]), float(goal["y"])
        ox, oy, oth = float(odom["x"]), float(odom["y"]), float(odom["theta"])
        bearing = math.atan2(gy - oy, gx - ox) - oth
    else:
        return g
    rows, cols = np.mgrid[0:GRID, 0:GRID]
    x = RANGE_M - (rows + 0.5) * CELL_M
    y = (cols + 0.5) * CELL_M - WIDTH_M / 2.0
    ang = np.arctan2(y, np.maximum(x, 1e-6))
    d = np.abs(np.angle(np.exp(1j * (ang - bearing))))
    return np.clip(1.0 - d / (math.pi / 2.0), 0.0, 1.0).astype(np.float32)


def _prep_depth(d_mm):
    d = cv2.resize(d_mm, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)
    m = (d > 0).astype(np.float32)
    out = np.clip(d.astype(np.float32) * 1e-3, 0.3, 6.0) * m
    return out[None], m[None]


def _prep_color(rgb):
    c = cv2.resize(rgb, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    c = (c.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return c.transpose(2, 0, 1)


class NavdDataset:
    """All ticks from the given session dirs (list of navd-v0 paths)."""

    def __init__(self, session_dirs, augment=False, sample_stride=1):
        self.items = []
        for sd in session_dirs:
            manifest = [json.loads(l) for l in
                        open(Path(sd) / "manifest.jsonl")]
            for row in manifest[::sample_stride]:
                stamp = row["stamp_ns"]
                p = Path(sd)
                if (p / "labels" / f"{stamp:020d}.npz").exists() \
                        and (p / "depth" / f"{stamp:020d}.npz").exists() \
                        and (p / "color" / f"{stamp:020d}.jpg").exists():
                    self.items.append((p, stamp, row))
        self.augment = augment
        self.rng = np.random.default_rng(0)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, stamp, row = self.items[i]
        depth = np.load(p / "depth" / f"{stamp:020d}.npz")
        dn, _ = _prep_depth(depth["near"])
        df, _ = _prep_depth(depth["far"])
        color = cv2.imread(str(p / "color" / f"{stamp:020d}.jpg"))
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        c = _prep_color(color)
        lab = np.load(p / "labels" / f"{stamp:020d}.npz")
        fused = lab["fused"].astype(np.int64)
        goal = build_goal_raster(row["goal"], row["odom"])
        cmd = row["cmd_vel"]
        vx, wz = float(cmd["vx"]), float(cmd["wz"])

        if self.augment:
            if self.rng.random() < 0.5:      # h-flip: mirror the world
                # the goal fan is spatial — mirroring it mirrors the goal
                dn, df, c, goal, fused = (dn[..., ::-1].copy(),
                                          df[..., ::-1].copy(),
                                          c[..., ::-1].copy(),
                                          goal[:, ::-1].copy(),
                                          fused[:, ::-1].copy())
                wz = -wz
            dn = dn + self.rng.normal(0, 0.01, dn.shape).astype(np.float32) \
                * (dn > 0)
            for _ in range(self.rng.integers(2, 6)):   # dropout blobs
                y0 = self.rng.integers(0, IMG_H - 20)
                x0 = self.rng.integers(0, IMG_W - 20)
                h0, w0 = self.rng.integers(8, 24, 2)
                dn[:, y0:y0 + h0, x0:x0 + w0] = 0.0
            b = 1.0 + self.rng.uniform(-0.25, 0.25)
            c = np.clip(c * b + self.rng.uniform(-0.05, 0.05), -3, 3)

        goal_ch = goal[None]
        imitation_target = -1
        if abs(vx) > 0.05:
            intend = math.atan2(0.5 * wz, vx)
            imitation_target = int(np.argmin(np.abs(RAY_ANGLES - intend)))
        return {
            "depth_near": dn.astype(np.float32),
            "depth_far": df.astype(np.float32),
            "color": c.astype(np.float32),
            "goal": goal_ch,
            "label": fused,
            "imitation_target": np.int64(imitation_target),
        }


def ray_navigable_probs(logits):
    """[B,3,60,60] logits -> [B,13] softmax over rays of mean P(navigable)
    along each ray (cells marched from the robot outward, bottom-center
    origin). Differentiable gather; no planner state."""
    B = logits.shape[0]
    p_nav = torch.softmax(logits, dim=1)[:, 1]            # [B,60,60]
    rows, cols = np.mgrid[0:GRID, 0:GRID]
    x = RANGE_M - (rows + 0.5) * CELL_M
    y = (cols + 0.5) * CELL_M - WIDTH_M / 2.0
    ang = np.arctan2(y, np.maximum(x, 1e-6))
    ray_hits = []
    for a in RAY_ANGLES:
        perp = np.abs(np.angle(np.exp(1j * (ang - a))))
        band = (perp < np.deg2rad(5.0)) & (x > 0)
        ray_hits.append(torch.from_numpy(
            np.where(band, 1.0 / max(band.sum(), 1), 0.0)
            .astype(np.float32)))
    W = torch.from_numpy(np.stack(ray_hits))              # [13,60,60]
    W = W / W.sum(dim=(1, 2), keepdim=True).clamp_min(1e-6)
    W = W.to(logits.device).unsqueeze(0)                  # [1,13,60,60]
    scores = (p_nav.unsqueeze(1) * W).sum(dim=(2, 3))     # [B,13]
    return torch.softmax(scores / 0.5, dim=1)


def class_weights_from(session_dirs, sample=400):
    """Inverse-frequency weights over the fused labels."""
    import random
    files = []
    for sd in session_dirs:
        files += list(Path(sd).glob("labels/*.npz"))
    random.seed(0)
    random.shuffle(files)
    counts = np.zeros(3, np.int64)
    for f in files[:sample]:
        lab = np.load(f)["fused"].ravel()
        counts += np.bincount(lab, minlength=3)
    w = counts.sum() / np.maximum(counts, 1) * 3.0
    return torch.tensor(w / w.mean(), dtype=torch.float32)


class Block(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(8, cin), nn.SiLU(),
            nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout),
            nn.SiLU(), nn.Conv2d(cout, cout, 3, padding=1))

    def forward(self, x):
        return self.net(x)


class NavdUNet(nn.Module):
    """3-level UNet on the 6-channel stem -> logits [B,3,60,60]."""

    def __init__(self, ch=(48, 96, 192)):
        super().__init__()
        self.stem = nn.Conv2d(6, ch[0], 3, padding=1)
        self.e1 = Block(ch[0], ch[0])
        self.d1 = nn.Conv2d(ch[0], ch[1], 4, stride=2, padding=1)
        self.e2 = Block(ch[1], ch[1])
        self.d2 = nn.Conv2d(ch[1], ch[2], 4, stride=2, padding=1)
        self.mid = Block(ch[2], ch[2])
        self.u2 = nn.ConvTranspose2d(ch[2], ch[1], 4, stride=2, padding=1)
        self.dec2 = Block(ch[2], ch[1])
        self.u1 = nn.ConvTranspose2d(ch[1], ch[0], 4, stride=2, padding=1)
        self.dec1 = Block(ch[1], ch[0])
        self.head = nn.Conv2d(ch[0], 3, 1)

    def forward(self, depth_near, depth_far, color, goal):
        x = torch.cat([depth_near, depth_far, color], dim=1)   # [B,6,240,424]
        g = F.interpolate(goal, size=x.shape[-2:], mode="nearest")
        x = torch.cat([x, g], dim=1)                            # 7ch
        s = x.shape[-2:]
        h1 = self.e1(self.stem(x))                              # 240x424
        h2 = self.e2(self.d1(h1))                               # 120x212
        m = self.mid(self.d2(h2))                               # 60x106
        y = torch.cat([self.u2(m)[:, :, :h2.shape[2], :h2.shape[3]], h2],
                      dim=1)
        y = self.dec2(y)
        y = torch.cat([self.u1(y)[:, :, :h1.shape[2], :h1.shape[3]], h1],
                      dim=1)
        y = self.dec1(y)
        logits = self.head(y)                                   # [B,3,240,424]
        # image frame -> grid frame: average-pool the FOV down to the 60x60
        # grid (interpolate to a divisible size first — adaptive pooling to
        # a non-factor output is not ONNX-exportable)
        logits = F.interpolate(logits, (GRID * 2, GRID * 2), mode="bilinear",
                               align_corners=False)
        logits = F.avg_pool2d(logits, 2)
        return logits


def navd_loss(logits, label, imitation_target, w_cls, lam=0.2):
    ce = F.cross_entropy(logits, label, weight=w_cls.to(logits.device))
    B = logits.shape[0]
    if imitation_target is not None and (imitation_target >= 0).any():
        probs = ray_navigable_probs(logits)
        tgt = torch.full_like(probs, 1.0 / probs.shape[1])
        m = imitation_target >= 0
        idx = imitation_target[m].clamp(0, probs.shape[1] - 1)
        tgt[m, idx] = 0.9
        tgt = tgt / tgt.sum(dim=1, keepdim=True)
        im = -(tgt * probs.clamp_min(1e-8).log()).sum(dim=1)[m].mean()
        return ce + lam * im, ce.detach(), im.detach()
    return ce, ce.detach(), torch.tensor(0.0)
