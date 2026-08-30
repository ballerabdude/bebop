"""Drawing helpers for annotated frames."""

import cv2
import numpy as np

COLORS = [
    (60, 200, 60), (60, 100, 230), (60, 200, 230), (230, 60, 200),
    (230, 60, 60), (200, 130, 60), (130, 200, 230), (230, 200, 60),
]


def _label_box(frame, x1, y1, text, color):
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (x1, y1 - th - baseline - 6), (x1 + tw + 6, y1), color, -1)
    cv2.putText(
        frame, text, (x1 + 3, y1 - baseline - 3),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
    )


def draw_instances(frame, instances, alpha=0.35, with_confidence=True):
    overlay = frame.copy()
    have = False
    for inst in instances:
        color = COLORS[inst.class_id % len(COLORS)]
        if inst.mask is not None:
            overlay[inst.mask] = color
            have = True
        elif inst.polygon is not None and len(inst.polygon) >= 3:
            cv2.fillPoly(overlay, [inst.polygon.astype(np.int32)], color)
            have = True
    if have:
        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
    for inst in instances:
        color = COLORS[inst.class_id % len(COLORS)]
        if inst.mask is not None:
            if inst.mask_contours is None:
                inst.mask_contours, _ = cv2.findContours(
                    inst.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
            cv2.drawContours(frame, inst.mask_contours, -1, color, 2)
        elif inst.polygon is not None and len(inst.polygon) >= 3:
            cv2.polylines(frame, [inst.polygon.astype(np.int32)], True, color, 2)
        x1, y1, x2, y2 = (int(v) for v in inst.bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{inst.label} {inst.confidence:.2f}" if with_confidence else inst.label
        _label_box(frame, x1, y1, text, color)
    return frame


NAV_COLORS = {1: (60, 230, 60), 2: (60, 230, 230)}


def draw_nav(frame, label_map, alpha=0.35):
    if label_map is None or label_map.shape[:2] != frame.shape[:2]:
        return frame
    lut = np.zeros((3, 3), dtype=np.uint8)
    for k, c in NAV_COLORS.items():
        lut[k] = c
    colorized = lut[label_map]
    overlay = frame.copy()
    mask_any = (label_map > 0)
    if mask_any.any():
        overlay[mask_any] = colorized[mask_any]
        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
    return frame


def draw_hud(frame, fps, n_objects, task="", infer_fps=None):
    h, w = frame.shape[:2]
    hud_h = 118 if infer_fps is not None else 92
    cv2.rectangle(frame, (0, 0), (280, hud_h), (0, 0, 0), -1)
    cv2.putText(frame, f"FPS: {fps:5.1f}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 230, 60), 2)
    cv2.putText(frame, f"Objects: {n_objects}", (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 200, 230), 2)
    cv2.putText(frame, f"Task: {task}", (10, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 200, 60), 2)
    if infer_fps is not None:
        cv2.putText(frame, f"Infer: {infer_fps:5.1f} Hz", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 130, 60), 2)
    cv2.line(frame, (w // 2 - 20, h // 2), (w // 2 + 20, h // 2), (0, 255, 255), 1)
    cv2.line(frame, (w // 2, h // 2 - 20), (w // 2, h // 2 + 20), (0, 255, 255), 1)
    return frame
