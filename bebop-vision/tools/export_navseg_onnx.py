"""Export the nav SegFormer student to ONNX, with a torch-vs-ONNX parity gate.

The exported artifact is what the firmware's nav runner consumes
(`bebop-linux/src/nav.rs`): input `pixel_values` [1,3,512,512] float32,
output `logits` [1,3,128,128] float32. Preprocessing (resize/normalize/NCHW)
stays on the Rust side; this script only exports the graph and proves the
two runtimes agree on labels.

Usage:
    python tools/export_navseg_onnx.py                  # export + 12-frame parity
    python tools/export_navseg_onnx.py --frames 30      # more parity frames
"""

import argparse
import glob
import os
import sys

import cv2
import numpy as np
import torch
from transformers import SegformerForSemanticSegmentation

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def preprocess(frame_bgr, imgsz):
    """Exactly what nav.rs does: square resize, RGB, /255, ImageNet norm, NCHW."""
    x = cv2.resize(frame_bgr, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = (x - np.array(IMAGENET_MEAN, np.float32)) / np.array(IMAGENET_STD, np.float32)
    return np.ascontiguousarray(np.transpose(x, (2, 0, 1)))[None]


def export(model, out_path, imgsz):
    dummy = torch.zeros((1, 3, imgsz, imgsz), dtype=torch.float32)
    try:
        torch.onnx.export(
            model, dummy, out_path,
            input_names=["pixel_values"], output_names=["logits"],
            opset_version=17, do_constant_folding=True,
        )
    except Exception as exc:
        print(f"[export] default exporter failed ({exc}); retrying dynamo=False")
        torch.onnx.export(
            model, dummy, out_path,
            input_names=["pixel_values"], output_names=["logits"],
            opset_version=17, do_constant_folding=True, dynamo=False,
        )
    print(f"[export] wrote {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB)")


def parity(model, sess, dataset, frames, imgsz, min_agreement):
    images = sorted(glob.glob(os.path.join(dataset, "images", "*.jpg")))
    if not images:
        raise SystemExit(f"no images under {dataset}/images for the parity check")
    step = max(1, len(images) // frames)
    picked = images[::step][:frames]
    agreements, max_diff = [], 0.0
    for path in picked:
        frame = cv2.imread(path, cv2.IMREAD_COLOR)
        if frame is None:
            continue
        x = preprocess(frame, imgsz)
        with torch.inference_mode():
            t_logits = model(pixel_values=torch.from_numpy(x)).logits.numpy()
        o_logits = sess.run(None, {"pixel_values": x})[0]
        max_diff = max(max_diff, float(np.abs(t_logits - o_logits).max()))
        agree = float((t_logits[0].argmax(0) == o_logits[0].argmax(0)).mean())
        agreements.append(agree)
        print(f"[parity] {os.path.basename(path)}: {agree:.4f}")
    mean = sum(agreements) / len(agreements) if agreements else 0.0
    print(f"[parity] mean label agreement {mean:.4%} | max |logit diff| {max_diff:.4f}")
    if mean < min_agreement:
        raise SystemExit(f"parity gate failed: {mean:.4%} < {min_agreement:.2%}")
    print(f"[parity] gate passed (>= {min_agreement:.0%})")


def main():
    parser = argparse.ArgumentParser(description="Export navseg student to ONNX")
    parser.add_argument("--model", default="weights/navseg")
    parser.add_argument("--out", default="weights/navseg.onnx")
    parser.add_argument("--dataset", default="datasets/nav-v0")
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--min-agreement", type=float, default=0.99)
    args = parser.parse_args()

    model = SegformerForSemanticSegmentation.from_pretrained(args.model)
    model.eval()

    export(model, args.out, args.imgsz)

    import onnxruntime as ort
    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    print(f"[parity] onnxruntime {ort.__version__}, providers: {sess.get_providers()}")
    parity(model, sess, args.dataset, args.frames, args.imgsz, args.min_agreement)
    print("[export] navseg.onnx is firmware-ready")


if __name__ == "__main__":
    main()
