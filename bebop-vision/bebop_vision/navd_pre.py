"""Shared navd input preprocessing (torch-free).

Lives between training (`navd.py` / `NavdDataset`) and the runtime
provider (`navd_runtime.py`) so both paths run byte-identical
resize / clip / normalization — the ONNX contract assumes the
consumer-side preprocessing matches training exactly (export parity
gate checks the model, not the glue).

Grid geometry here (60x60 @ 5 cm, 3 m forward, 3 m wide) is the model's
native output frame; the runtime provider builds its BevGrid directly on
these constants rather than the rig YAML (the rig may be re-configured,
the exported model cannot).
"""

import math

import cv2
import numpy as np

IMG_H, IMG_W = 240, 424
GRID = 60
RANGE_M, WIDTH_M, CELL_M = 3.0, 3.0, 0.05
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


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


def prep_depth(d_mm):
    """Raw uint16 mm depth -> (meters f32 [1,H,W], validity mask [1,H,W]).

    Invalid (0) pixels stay 0 after clipping; values clip to [0.3, 6.0] m.
    """
    d = cv2.resize(d_mm, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)
    m = (d > 0).astype(np.float32)
    out = np.clip(d.astype(np.float32) * 1e-3, 0.3, 6.0) * m
    return out[None], m[None]


def prep_color(rgb):
    """RGB uint8 image -> ImageNet-normalized f32 [3,H,W] (training norm)."""
    c = cv2.resize(rgb, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    c = (c.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return c.transpose(2, 0, 1)
