"""Record a navigable-path dataset: frames + synchronized SAM 3.1 concept masks."""

import json
import os
import time

import cv2
import numpy as np

from . import config
from .camera import Camera
from .sam3_concepts import Sam3ConceptSegmenter
from .visualize import draw_instances


class DatasetRecorder:
    def __init__(self, source, out_dir, concepts, conf=config.DEFAULT_CONFIDENCE,
                 version="sam3.1", trt_engine=None, rate_hz=2.0, display=False):
        self.rate_hz = rate_hz
        self.display = display
        self.images_dir = os.path.join(out_dir, "images")
        self.masks_dir = os.path.join(out_dir, "masks")
        for d in (self.images_dir, self.masks_dir):
            os.makedirs(d, exist_ok=True)
        self.manifest_path = os.path.join(out_dir, "manifest.jsonl")
        self.camera = Camera(source)
        self.segmenter = Sam3ConceptSegmenter(
            concepts=concepts, conf=conf, version=version, trt_engine=trt_engine
        )

    def run(self, duration=None):
        self.camera.start()
        interval = 1.0 / self.rate_hz
        print(f"[recorder] saving to {os.path.dirname(self.images_dir)} @ {self.rate_hz} Hz")
        started = time.perf_counter()
        count = 0
        last_ts = 0.0
        try:
            while True:
                if duration is not None and time.perf_counter() - started >= duration:
                    break
                t0 = time.perf_counter()
                frame, ts = self.camera.read()
                if frame is None or ts == last_ts:
                    time.sleep(0.01)
                    continue
                last_ts = ts

                segments = self.segmenter.segment(frame)
                h, w = frame.shape[:2]
                concept_masks = {}
                for seg in segments:
                    if seg.mask is None:
                        continue
                    if seg.label in concept_masks:
                        concept_masks[seg.label] |= seg.mask
                    else:
                        concept_masks[seg.label] = seg.mask.copy()
                for concept in self.segmenter.concepts:
                    concept_masks.setdefault(concept, np.zeros((h, w), dtype=bool))

                stamp = f"{count:06d}_{int(ts * 1000)}"
                cv2.imwrite(
                    os.path.join(self.images_dir, f"{stamp}.jpg"), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 92],
                )
                np.savez_compressed(
                    os.path.join(self.masks_dir, f"{stamp}.npz"),
                    **{k: v for k, v in concept_masks.items() if k.isidentifier()},
                )
                with open(self.manifest_path, "a") as f:
                    f.write(json.dumps(
                        {"stamp": stamp, "px": {k: int(v.sum()) for k, v in concept_masks.items()}}
                    ) + "\n")

                count += 1
                if self.display:
                    annotated = frame.copy()
                    draw_instances(annotated, segments)
                    cv2.imshow("recorder", annotated)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
                time.sleep(max(0.0, interval - (time.perf_counter() - t0)))
        except KeyboardInterrupt:
            print("\n[recorder] interrupted")
        finally:
            self.camera.stop()
            if self.display:
                cv2.destroyAllWindows()
        print(f"[recorder] saved {count} samples")
        return count