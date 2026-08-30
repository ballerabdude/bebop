"""Generate navigable-path labels from recorded SAM 3.1 concept masks.

Classes: 0 = blocked, 1 = navigable (floor, clear of obstacles),
2 = caution (floor within margin of obstacles/walls).

Usage: python -m bebop_vision.labelnav datasets/nav-v0 --margin 25
"""

import argparse
import glob
import os

import cv2
import numpy as np

CLASSES = ("blocked", "navigable", "caution")


def _dilate(mask, margin_px):
    if margin_px <= 0 or not mask.any():
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (margin_px, margin_px))
    return cv2.dilate(mask.astype(np.uint8), k).astype(bool)


def label_from_masks(masks, floor_labels="floor", margin_px=25):
    floor = None
    for label in [f.strip() for f in floor_labels.split(",") if f.strip()]:
        m = masks.get(label)
        if m is None:
            continue
        m = m.astype(bool)
        floor = m if floor is None else (floor | m)
    if floor is None:
        return None
    floor = floor.astype(bool)
    h, w = floor.shape
    obstacles = np.zeros((h, w), dtype=bool)
    for label, mask in masks.items():
        if label not in floor_labels:
            obstacles |= mask.astype(bool)
    danger = _dilate(obstacles, margin_px)
    label = np.zeros((h, w), dtype=np.uint8)
    label[floor & ~danger] = 1
    label[floor & danger] = 2
    return label


def make_labels(dataset_dir, floor_labels="floor", margin_px=25):
    masks_dir = os.path.join(dataset_dir, "masks")
    labels_dir = os.path.join(dataset_dir, "labels")
    os.makedirs(labels_dir, exist_ok=True)
    n = 0
    for path in sorted(glob.glob(os.path.join(masks_dir, "*.npz"))):
        masks = dict(np.load(path))
        label = label_from_masks(masks, floor_labels, margin_px)
        if label is None:
            continue
        stem = os.path.splitext(os.path.basename(path))[0]
        cv2.imwrite(os.path.join(labels_dir, f"{stem}.png"), label)
        n += 1
    print(f"[labelnav] wrote {n} labels to {labels_dir}")
    return n


def main():
    parser = argparse.ArgumentParser(description="Generate navigable-path labels")
    parser.add_argument("dataset_dir")
    parser.add_argument("--floor", default="floor",
                        help="drivable concept(s), comma-separated (union), e.g. 'pavement,driveway,concrete'")
    parser.add_argument("--margin", type=int, default=25,
                        help="safety margin in pixels around obstacles (robot radius)")
    args = parser.parse_args()
    make_labels(args.dataset_dir, args.floor, args.margin)


if __name__ == "__main__":
    main()