"""Train a navigable-path segmentation model (SegFormer-B0, from scratch, Apache stack).

Usage: python train_nav.py datasets/nav-v0 --epochs 60 --out weights/navseg
"""

import argparse
import glob
import os
import random
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import SegformerConfig, SegformerForSemanticSegmentation

from bebop_vision.navseg import CLASSES, IMAGENET_MEAN, IMAGENET_STD


class NavDataset(Dataset):
    def __init__(self, pairs, imgsz=512, train=True):
        self.pairs = pairs
        self.imgsz = imgsz
        self.train = train

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, label_path = self.pairs[idx]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        if img is None or label is None:
            return self.__getitem__((idx + 1) % len(self))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.train:
            if random.random() < 0.5:
                img = img[:, ::-1]
                label = label[:, ::-1]
            j = 0.6 + random.random() * 0.6
            img = np.clip(img.astype(np.float32) * j, 0, 255).astype(np.uint8)
        img = cv2.resize(img, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, (self.imgsz, self.imgsz), interpolation=cv2.INTER_NEAREST)
        x = img.astype(np.float32) / 255.0
        x = (x - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
        return (
            torch.from_numpy(x).permute(2, 0, 1).float(),
            torch.from_numpy(label.copy()).long(),
        )


def build_pairs(dataset_dir):
    images = sorted(glob.glob(os.path.join(dataset_dir, "images", "*.jpg")))
    pairs = []
    for p in images:
        stem = os.path.splitext(os.path.basename(p))[0]
        label = os.path.join(dataset_dir, "labels", f"{stem}.png")
        if os.path.exists(label):
            pairs.append((p, label))
    return pairs


def class_weights_from(pairs, imgsz):
    counts = np.zeros(3, dtype=np.int64)
    for _, lp in pairs:
        label = cv2.imread(lp, cv2.IMREAD_GRAYSCALE)
        counts += np.bincount(
            cv2.resize(label, (imgsz, imgsz), interpolation=cv2.INTER_NEAREST).ravel(),
            minlength=3,
        )
    freq = counts / max(1, counts.sum())
    weights = np.where(freq > 0, 1.0 / np.maximum(freq, 1e-4), 0.0)
    weights = weights / weights.sum() * 3
    return torch.tensor(weights, dtype=torch.float32)


def miou(logits, labels, n=3):
    pred = logits.argmax(dim=1)
    ious = []
    for c in range(n):
        inter = ((pred == c) & (labels == c)).sum().item()
        union = ((pred == c) | (labels == c)).sum().item()
        if union > 0:
            ious.append(inter / union)
    return sum(ious) / len(ious) if ious else 0.0


def main():
    parser = argparse.ArgumentParser(description="Train navigable-path SegFormer")
    parser.add_argument("dataset_dir")
    parser.add_argument("--out", default="weights/navseg")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    args = parser.parse_args()

    pairs = build_pairs(args.dataset_dir)
    if len(pairs) < 4:
        raise SystemExit(f"need at least 4 labeled samples, found {len(pairs)}")
    random.Random(0).shuffle(pairs)
    n_val = max(2, int(len(pairs) * args.val_fraction))
    val_pairs, train_pairs = pairs[:n_val], pairs[n_val:]

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    train_loader = DataLoader(
        NavDataset(train_pairs, args.imgsz, train=True), batch_size=args.batch,
        shuffle=True, num_workers=4, drop_last=True,
    )
    val_loader = DataLoader(
        NavDataset(val_pairs, args.imgsz, train=False), batch_size=args.batch, num_workers=2,
    )

    config = SegformerConfig(num_labels=3)
    model = SegformerForSemanticSegmentation(config).to(device)
    weights = class_weights_from(pairs, args.imgsz).to(device)
    print(f"[train] {len(train_pairs)} train / {len(val_pairs)} val | class weights: "
          f"{dict(zip(CLASSES, weights.tolist()))}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=args.epochs * max(1, len(train_loader)),
    )
    criterion = torch.nn.CrossEntropyLoss(weight=weights)

    best_miou = 0.0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        t0 = time.perf_counter()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(pixel_values=images).logits
            logits = torch.nn.functional.interpolate(
                logits.float(), size=labels.shape[-2:], mode="bilinear", align_corners=False
            )
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        model.eval()
        vals = []
        with torch.inference_mode():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(pixel_values=images).logits
                logits = torch.nn.functional.interpolate(
                    logits.float(), size=labels.shape[-2:], mode="bilinear", align_corners=False
                )
                vals.append(miou(logits, labels))
        v_miou = sum(vals) / len(vals) if vals else 0.0
        print(f"[train] epoch {epoch + 1:3d}/{args.epochs} | loss {total_loss / len(train_loader):.4f} "
              f"| val mIoU {v_miou:.4f} | {time.perf_counter() - t0:.1f}s")
        if v_miou > best_miou:
            best_miou = v_miou
            model.save_pretrained(args.out)
    print(f"[train] best val mIoU {best_miou:.4f} -> saved to {args.out}")


if __name__ == "__main__":
    main()