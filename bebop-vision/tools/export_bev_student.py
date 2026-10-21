"""Export a trained BEV student to ONNX (and verify parity).

    python tools/export_bev_student.py \
        --checkpoint runs/bev-student/dino-vits16-full/checkpoints/best.pt \
        --out runs/bev-student/dino-vits16-full/export/bev_student.onnx

Input is a static pair ``[2,4,H,W]`` from the checkpoint (near, far; RGB+depth/8),
outputs are ``semantic [4,120,60]``, ``ego [1,120,60]`` and
``coverage [1,120,60]``.  The fixed lift LUT is baked into the graph.

After this, build the TensorRT FP16 engine on the robot (TensorRT is not on
the workstation venv)::

    trtexec --onnx=bev_student.onnx --saveEngine=bev_student.engine \
        --fp16

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

from bebop_vision.bev_student import (SINGLE_HEAD_CLASSES, grid_from_model,
                                      student_from_checkpoint)


class _SpatialSegformerMLP(torch.nn.Module):
    """Equivalent Mix-FFN in BCHW layout for TensorRT's convolution path.

    TRT 10.16.2 on Thor miscompiles the Linear -> transpose/reshape ->
    depthwise Conv path in MiT-B5 (also in FP32). Express the two per-token
    linears as 1x1 convolutions, keeping the MLP spatial throughout. This
    changes neither weights nor mathematics, and is export-only.
    """

    def __init__(self, mlp):
        super().__init__()
        self.fc1 = getattr(mlp, "fc1", getattr(mlp, "dense1", None))
        self.fc2 = getattr(mlp, "fc2", getattr(mlp, "dense2", None))
        self.dwconv = mlp.dwconv.dwconv
        self.activation = getattr(mlp, "activation_fn",
                                  getattr(mlp, "intermediate_act_fn", None))
        self.dropout = mlp.dropout

    def forward(self, hidden_states, height, width):
        b, _, c = hidden_states.shape
        x = hidden_states.transpose(1, 2).reshape(b, c, height, width)
        x = torch.nn.functional.conv2d(
            x, self.fc1.weight[:, :, None, None], self.fc1.bias)
        x = self.dwconv(x)
        x = self.dropout(self.activation(x))
        x = torch.nn.functional.conv2d(
            x, self.fc2.weight[:, :, None, None], self.fc2.bias)
        return self.dropout(x).flatten(2).transpose(1, 2)


class ExportWrapper(torch.nn.Module):
    """One [2,4,H,W] pair -> the three heads for one sample."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        if model.encoder == "segformer-b5":
            for module in list(model.backbone.modules()):
                mlp = getattr(module, "mlp", None)
                if (mlp is not None and hasattr(mlp, "dwconv")
                        and not isinstance(mlp, _SpatialSegformerMLP)):
                    module.mlp = _SpatialSegformerMLP(mlp)

    def forward(self, x):
        color = x[:, :3].unsqueeze(0)
        depth = x[:, 3:4].unsqueeze(0)
        out = self.model(color, depth)
        if "ego" not in out:            # single-head model
            return (out["semantic"][0],)
        return (out["semantic"][0], out["ego"][0], out["coverage"][0])


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--pretrained-dir", default=None)
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--height", type=int, default=None,
                    help="optional assertion; size is taken from checkpoint")
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--skip-verify", action="store_true")
    args = ap.parse_args(argv)
    ckpt = Path(args.checkpoint)
    out = (Path(args.out) if args.out
           else ckpt.parent.parent / "export" / "bev_student.onnx")
    out.parent.mkdir(parents=True, exist_ok=True)

    ck = torch.load(ckpt, map_location=args.device, weights_only=False)
    model = student_from_checkpoint(ck, args.pretrained_dir)
    height, width = model.input_hw
    grid = grid_from_model(model)
    if ((args.height is not None and args.height != height)
            or (args.width is not None and args.width != width)):
        ap.error("export size must match checkpoint's calibrated lift LUT")
    model.eval().to(args.device)
    wrapper = ExportWrapper(model).eval().to(args.device)
    print(f"checkpoint epoch {ck.get('epoch')} stage {ck.get('stage')}; "
          f"{model.num_parameters()/1e6:.2f} M params", flush=True)

    x = torch.randn(2, 4, height, width, device=args.device)
    with torch.inference_mode():
        torch_out = wrapper(x)
    single_head = bool(getattr(model, "single_head", False))
    names = ["semantic"] if single_head else ["semantic", "ego", "coverage"]
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
            "name": "input", "shape": [2, 4, height, width],
            "camera_order": ["near", "far"],
            "channels": "RGB(3) + depth_m/8(1)",
            "color_mean": [0.485, 0.456, 0.406],
            "color_std": [0.229, 0.224, 0.225],
            "depth_scale_m": 8.0,
            "resize_hw": [height, width],
        },
        "outputs": {
            "semantic": [model.num_classes, grid.rows, grid.cols],
            **({} if single_head else {
                "ego": [1, grid.rows, grid.cols],
                "coverage": [1, grid.rows, grid.cols]}),
            "semantic_classes": (
                list(SINGLE_HEAD_CLASSES) if single_head else
                ["floor", "wall", "obstacle", "hazard"]
                + (["unknown"] if model.num_classes == 5 else [])),
        },
        "single_head": single_head,
        "grid": grid.metadata(),
        "lut": model.lut_meta,
        "encoder": model.encoder,
        "spatial_segformer_mlp": model.encoder == "segformer-b5",
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
