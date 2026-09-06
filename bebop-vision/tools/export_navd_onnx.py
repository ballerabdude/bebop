"""Export the navd student to ONNX, with a torch-vs-ONNX parity gate.

Runtime contract (plan §7.2/§7.3), fixed names:
  inputs:  depth_near [1,1,240,424] f32 (meters, clipped [0.3,6], 0=invalid)
           depth_far  [1,1,240,424] f32
           color      [1,3,240,424] f32 (ImageNet-normalized)
           goal       [1,1,60,60]    f32
  output:  logits     [1,3,60,60]    f32 (0 blocked / 1 navigable / 2 caution)

Preprocessing stays consumer-side (same normalization as NavdDataset).
Parity gate: >= 0.99 cell-wise argmax agreement vs torch over dataset frames.

Usage:
    python tools/export_navd_onnx.py --ckpt weights/navd_v2/best.pt
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import cv2
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bebop_vision.navd import NavdDataset, NavdUNet  # noqa: E402

INPUT_NAMES = ["depth_near", "depth_far", "color", "goal"]


def export(model, out_path):
    dummy = (torch.zeros((1, 1, 240, 424)), torch.zeros((1, 1, 240, 424)),
             torch.zeros((1, 3, 240, 424)), torch.zeros((1, 1, 60, 60)))
    torch.onnx.export(
        model, dummy, out_path, input_names=INPUT_NAMES,
        output_names=["logits"], opset_version=17, do_constant_folding=True,
        dynamo=False)
    print(f"[export] wrote {out_path} "
          f"({Path(out_path).stat().st_size / 1e6:.1f} MB)")


def parity(model, sess, samples, min_agreement):
    agreements, max_diff = [], 0.0
    for it in samples:
        p, stamp, _row = it
        depth = np.load(p / "depth" / f"{stamp:020d}.npz")
        dn, _ = _prep(depth["near"])
        df, _ = _prep(depth["far"])
        import cv2
        cimg = cv2.imread(str(p / "color" / f"{stamp:020d}.jpg"))
        cimg = cv2.cvtColor(cimg, cv2.COLOR_BGR2RGB)
        c = _prep_color(cimg)
        lab = np.load(p / "labels" / f"{stamp:020d}.npz")
        goal = np.zeros((1, 1, 60, 60), np.float32)
        feed = {"depth_near": dn[None], "depth_far": df[None],
                "color": c[None], "goal": goal}
        with torch.inference_mode():
            t = model(torch.from_numpy(feed["depth_near"]),
                      torch.from_numpy(feed["depth_far"]),
                      torch.from_numpy(feed["color"]),
                      torch.from_numpy(feed["goal"])).numpy()
        o = sess.run(None, feed)[0]
        max_diff = max(max_diff, float(np.abs(t - o).max()))
        agree = float((t[0].argmax(0) == o[0].argmax(0)).mean())
        agreements.append(agree)
        gt = lab["fused"]
        acc_t = float((t[0].argmax(0) == gt).mean())
        print(f"[parity] {stamp}: agree {agree:.4f} | vs-teacher "
              f"{acc_t:.3f}")
    mean = sum(agreements) / len(agreements)
    print(f"[parity] mean label agreement {mean:.4%} | "
          f"max |logit diff| {max_diff:.4f}")
    if mean < min_agreement:
        raise SystemExit(f"parity gate failed: {mean:.4%} < {min_agreement:.0%}")
    print(f"[parity] gate passed (>= {min_agreement:.0%})")


def _prep(d_mm):
    d = cv2.resize(d_mm, (424, 240), interpolation=cv2.INTER_NEAREST)
    m = (d > 0).astype(np.float32)
    out = np.clip(d.astype(np.float32) * 1e-3, 0.3, 6.0) * m
    return out[None], m[None]


def _prep_color(rgb):
    c = cv2.resize(rgb, (424, 240), interpolation=cv2.INTER_AREA)
    c = (c.astype(np.float32) / 255.0
         - np.array([0.485, 0.456, 0.406], np.float32)) \
        / np.array([0.229, 0.224, 0.225], np.float32)
    return c.transpose(2, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="weights/navd_v2/best.pt")
    ap.add_argument("--out", default="weights/navd.onnx")
    ap.add_argument("--data", default="datasets/navd-v0")
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--min-agreement", type=float, default=0.99)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    model = NavdUNet()
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"[export] loaded {args.ckpt} (epoch {ck.get('epoch')}, "
          f"val_miou {ck.get('val_miou')})")

    export(model, args.out)

    import onnxruntime as ort
    sess = ort.InferenceSession(args.out,
                                providers=["CPUExecutionProvider"])
    print(f"[parity] onnxruntime {ort.__version__}, "
          f"providers: {sess.get_providers()}")

    sessions = sorted(p for p in Path(args.data).glob("navd_session_*")
                      if p.is_dir())
    ds = NavdDataset(sessions, augment=False)
    step = max(1, len(ds) // args.frames)
    samples = [ds.items[i] for i in range(0, len(ds), step)][:args.frames]
    parity(model, sess, samples, args.min_agreement)
    print("[export] navd.onnx is runtime-ready (§7.3)")


if __name__ == "__main__":
    main()
