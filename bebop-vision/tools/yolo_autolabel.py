"""YOLO-seg auto-label pass over extracted navd sessions (plan §7.1).

Runs yolo26l-seg over each session's near-camera color frames and stores
per-frame instance masks next to the extractor output:

    datasets/navd-v0/<session>/yolo/{stamp}.npz
        bits    uint8  packed (N, H, W) instance masks (row-major bits)
        classes int16  (N,) COCO class ids
        confs   f32    (N,) confidences
        shape   int16  (H, W)

    datasets/navd-v0/<session>/yolo_detections.jsonl
        one row per frame with detections: stamp, [(name, conf, area_px)]

The BEV projection / fusion with the geometric teacher happens in a later
step (the masks here are image-space; glass etc. have no depth to
deproject, so projection is a frustum problem, not a lookup).
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = Path("/tmp/opencode/yolo26l-seg.pt")
CONF = 0.30
IMG_SIZE = 1280


def pack(masks):
    """(N, H, W) bool -> packed uint8 bits."""
    return np.packbits(masks, axis=None).reshape(-1) if masks.size == 0 else \
        np.packbits(masks.reshape(masks.shape[0], -1), axis=1)


def label_session(model, names, sess_dir, role="near"):
    color_dir = sess_dir / ("color" if role == "near" else "color_far")
    sub = "" if role == "near" else "_far"
    out_dir = sess_dir / f"yolo{sub}"
    out_dir.mkdir(exist_ok=True)
    det_path = sess_dir / f"yolo_detections{sub}.jsonl"
    frames = sorted(color_dir.glob("*.jpg"))
    done = {json.loads(l)["stamp"] for l in open(det_path)} if det_path.exists() else set()
    n_det_frames = 0
    with open(det_path, "a") as det_f, open(sess_dir / "manifest.jsonl") as mf:
        stamp_by_jpg = {}
        for row in mf:
            r = json.loads(row)
            stamp_by_jpg[f"{r['stamp_ns']:020d}.jpg"] = r["stamp_ns"]
        t0 = time.time()
        for i, jpg in enumerate(frames):
            stamp = stamp_by_jpg.get(jpg.name)
            if stamp is None or stamp in done:
                continue
            results = model.predict(str(jpg), conf=CONF, imgsz=IMG_SIZE,
                                    verbose=False, device=0)
            r = results[0]
            dets = []
            if r.masks is not None and len(r.masks) > 0:
                m = r.masks.data.cpu().numpy().astype(bool)   # (N, h, w)
                if m.shape[1:] != (800, 1280):
                    import cv2
                    m = np.stack([cv2.resize(x.astype(np.uint8), (1280, 800),
                                             interpolation=cv2.INTER_NEAREST)
                                  for x in m]).astype(bool)
                cls = r.boxes.cls.cpu().numpy().astype(np.int16)
                conf = r.boxes.conf.cpu().numpy().astype(np.float32)
                np.savez_compressed(out_dir / f"{stamp:020d}.npz",
                                    bits=pack(m), classes=cls, confs=conf,
                                    shape=np.array(m.shape[1:], np.int32))
                for j in range(len(cls)):
                    area = int(m[j].sum())
                    if area > 200:
                        dets.append([str(names[int(cls[j])]),
                                     round(float(conf[j]), 2), area])
            if dets:
                det_f.write(json.dumps({"stamp": stamp, "det": dets}) + "\n")
                n_det_frames += 1
            if (i + 1) % 100 == 0:
                el = time.time() - t0
                print(f"  {i + 1}/{len(frames)} ({el / (i + 1):.2f} s/frame)",
                      flush=True)
    return len(frames), n_det_frames


def main():
    from ultralytics import YOLO
    role = sys.argv[1] if len(sys.argv) > 1 else "near"
    model = YOLO(str(WEIGHTS))
    names = model.names
    sessions = sorted((ROOT / "datasets" / "navd-v0").glob("navd_session_*"))
    for s in sessions:
        n, nd = label_session(model, names, s, role=role)
        print(f"{s.name}: {n} frames, {nd} with detections", flush=True)


if __name__ == "__main__":
    main()
