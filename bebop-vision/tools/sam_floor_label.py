"""SAM 3.1 floor-segmentation pass over extracted navd sessions.

Prompts SAM 3.1 with drivable-ground concepts (floor / carpet / rug /
ground) per color frame and stores the union floor mask per camera:

    datasets/navd-v0/<session>/sam_floor/{stamp}.npz       (near camera)
    datasets/navd-v0/<session>/sam_floor_far/{stamp}.npz   (far camera)
        mask  bool (800, 1280) union floor mask

These are teacher-side labels only (SAM license note in sam3_concepts.py):
the runtime student model learns floor-ness from raw depth+color.
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CONCEPTS = ["floor", "carpet", "rug", "ground"]
CONF = 0.35


def label_session(model, sess_dir, role):
    sub = "" if role == "near" else "_far"
    color_dir = sess_dir / ("color" if role == "near" else "color_far")
    out_dir = sess_dir / f"sam_floor{sub}"
    out_dir.mkdir(exist_ok=True)
    frames = sorted(color_dir.glob("*.jpg"))
    done = {p.stem for p in out_dir.glob("*.npz")}
    n_new = 0
    t0 = time.time()
    for i, jpg in enumerate(frames):
        stamp = jpg.stem
        if stamp in done:
            continue
        img = cv2.imread(str(jpg))
        mask = np.zeros(img.shape[:2], bool)
        for seg in model.segment(img):
            if seg.label in CONCEPTS and seg.mask is not None:
                mask |= seg.mask
        np.savez_compressed(out_dir / f"{stamp}.npz", mask=mask)
        n_new += 1
        if n_new % 50 == 0:
            el = time.time() - t0
            print(f"  {role} {n_new} ({el / n_new:.2f} s/frame)", flush=True)
    return n_new


def main():
    from bebop_vision.sam3_concepts import Sam3ConceptSegmenter
    role = sys.argv[1] if len(sys.argv) > 1 else "near"
    model = Sam3ConceptSegmenter(concepts=CONCEPTS, conf=CONF,
                                 version="sam3.1", device="cuda")
    sessions = sorted((ROOT / "datasets" / "navd-v0").glob("navd_session_*"))
    for s in sessions:
        n = label_session(model, s, role)
        print(f"{s.name} [{role}]: {n} frames labeled", flush=True)


if __name__ == "__main__":
    main()
