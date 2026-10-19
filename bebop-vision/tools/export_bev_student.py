"""Export the trained DINOv3 BEV student to ONNX (and verify parity).

    python tools/export_bev_student.py \
        --checkpoint runs/bev-student/dino-vits16-full/checkpoints/best.pt \
        --out runs/bev-student/dino-vits16-full/export/bev_student.onnx

Input is the static pair ``[2,4,240,424]`` (near, far; RGB+depth/8),
outputs are ``semantic [4,120,60]``, ``ego [1,120,60]`` and
``coverage [1,120,60]``.  The fixed lift LUT is baked into the graph.

After this, build the TensorRT FP16 engine on the robot (TensorRT is not on
the workstation venv)::

    trtexec --onnx=bev_student.onnx --saveEngine=bev_student.engine \
        --fp16 --shapes=input:2x4x240x424

The JSON metadata next to the ONNX records names, preprocessing and grid
geometry for the runtime.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bebop_vision.bev_student import (BevStudent, FEAT_HW, HEIGHTS,
                                      build_lift_lut)


class ExportWrapper(torch.nn.Module):
    """One [2,4,H,W] pair -> the three heads for one sample."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        color = x[:, :3].unsqueeze(0)
        depth = x[:, 3:4].unsqueeze(0)
        out = self.model(color, depth)
        return (out["semantic"][0], out["ego"][0], out["coverage"][0])


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--pretrained-dir", default=None)
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--width", type=int, default=424)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--skip-verify", action="store_true")
    args = ap.parse_args(argv)
    ckpt = Path(args.checkpoint)
    out = (Path(args.out) if args.out
           else ckpt.parent.parent / "export" / "bev_student.onnx")
    out.parent.mkdir(parents=True, exist_ok=True)

    ck = torch.load(ckpt, map_location=args.device, weights_only=False)
    lut = build_lift_lut()
    model = BevStudent(pretrained_dir=args.pretrained_dir, lut=lut)
    model.load_state_dict(ck["model"])
    model.eval().to(args.device)
    wrapper = ExportWrapper(model).eval().to(args.device)
    print(f"checkpoint epoch {ck.get('epoch')} stage {ck.get('stage')}; "
          f"{model.num_parameters()/1e6:.2f} M params", flush=True)

    x = torch.randn(2, 4, args.height, args.width, device=args.device)
    with torch.inference_mode():
        torch_out = wrapper(x)
    names = ["semantic", "ego", "coverage"]
    torch.onnx.export(
        wrapper, (x,), str(out),
        input_names=["input"], output_names=names,
        opset_version=args.opset, do_constant_folding=True,
        dynamo=False)
    print(f"wrote {out} ({out.stat().st_size/1e6:.1f} MB)", flush=True)

    meta = {
        "checkpoint": str(ckpt),
        "checkpoint_epoch": ck.get("epoch"),
        "opset": args.opset,
        "input": {
            "name": "input", "shape": [2, 4, args.height, args.width],
            "camera_order": ["near", "far"],
            "channels": "RGB(3) + depth_m/8(1)",
            "color_mean": [0.485, 0.456, 0.406],
            "color_std": [0.229, 0.224, 0.225],
            "depth_scale_m": 8.0,
            "resize_hw": [args.height, args.width],
        },
        "outputs": {
            "semantic": [4, 120, 60], "ego": [1, 120, 60],
            "coverage": [1, 120, 60],
            "semantic_classes": ["floor", "wall", "obstacle", "hazard"],
        },
        "grid": {
            "rows": 120, "cols": 60, "cell_m": 0.05,
            "range_m": 6.0, "width_m": 3.0,
            "row0": "far", "col0": "right",
        },
        "lut": lut["meta"],
        "model_params": model.num_parameters(),
        "created_at": __import__("time").strftime("%Y-%m-%dT%H:%M:%S"),
    }
    meta_path = out.with_suffix(".json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    if not args.skip_verify:
        import onnx
        import onnxruntime as ort
        onnx.checker.check_model(str(out))
        sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        xin = x.cpu().numpy()
        got = sess.run(None, {"input": xin})
        print("ONNX outputs", [g.shape for g in got], flush=True)
        for name, ref, val in zip(names, torch_out, got):
            r = ref.detach().cpu().numpy()
            diff = float(np.max(np.abs(r - val)))
            cos = float(np.sum(r * val)
                        / (np.linalg.norm(r) * np.linalg.norm(val) + 1e-9))
            if name == "semantic":
                agree = float(np.mean(r.argmax(0) == val.argmax(0)))
            else:
                agree = float(np.mean((r > 0) == (val > 0)))
            print(f"  {name:8s} max_abs {diff:.3e} cos {cos:.6f} "
                  f"agree {agree:.4f}", flush=True)
        print(f"metadata {meta_path}", flush=True)
    return out


if __name__ == "__main__":
    main()
