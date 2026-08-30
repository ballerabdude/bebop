"""Shared instance-segment data structure used by all segmentation backends."""

import cv2
import numpy as np


class InstanceSegment:
    __slots__ = ("bbox", "confidence", "class_id", "label", "polygon", "mask", "area", "mask_contours")

    def __init__(self, bbox, confidence, class_id, label, polygon=None, mask=None):
        self.bbox = bbox
        self.confidence = confidence
        self.class_id = class_id
        self.label = label
        self.polygon = polygon
        self.mask = mask
        self.mask_contours = None
        if mask is not None:
            self.area = float(mask.sum())
        elif polygon is not None:
            self.area = float(cv2.contourArea(polygon.astype(np.float32)))
        else:
            self.area = 0.0

    def to_dict(self):
        return {
            "bbox": self.bbox,
            "confidence": self.confidence,
            "class_id": self.class_id,
            "label": self.label,
            "mask_area": int(self.mask.sum()) if self.mask is not None else 0,
        }