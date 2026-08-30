"""Navigable-path segmentation with the trained SegFormer student model."""

import os
import time

import cv2
import numpy as np
import torch
from transformers import SegformerForSemanticSegmentation

from .runtime import autocast_ctx, resolve_device

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLASSES = ("blocked", "navigable", "caution")


class NavSegmenter:
    def __init__(self, model_dir, device="auto", imgsz=512):
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(
                f"{model_dir} not found. Train it first: python train_nav.py <dataset_dir> --out {model_dir}"
            )
        self.device = resolve_device(device)
        self.imgsz = imgsz
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_dir).to(self.device)
        self.model.eval()
        self._warmup(model_dir)

    def _warmup(self, model_dir):
        dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
        start = time.perf_counter()
        self.segment(dummy)
        print(f"[nav] SegFormer ready from {model_dir} on {self.device} "
              f"(warmup {time.perf_counter() - start:.2f}s)")

    def segment(self, frame):
        """Return (label_map (H, W) uint8, stats dict label -> fraction)."""
        h, w = frame.shape[:2]
        x = cv2.resize(frame, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = (x - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(IMAGENET_STD, dtype=np.float32)
        tensor = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        with torch.inference_mode(), autocast_ctx(self.device):
            logits = self.model(pixel_values=tensor).logits
            logits = torch.nn.functional.interpolate(
                logits.float(), size=(h, w), mode="bilinear", align_corners=False
            )
        label_map = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        counts = np.bincount(label_map.ravel(), minlength=3)
        total = max(1, label_map.size)
        stats = {CLASSES[i]: counts[i] / total for i in range(3) if counts[i] / total > 0.005}
        return label_map, stats