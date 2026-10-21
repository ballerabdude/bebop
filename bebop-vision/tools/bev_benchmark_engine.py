"""Measure a TensorRT BEV tick on saved camera pairs, including JPEG + postprocess.

    python tools/bev_benchmark_engine.py --engine weights/candidate.engine \
        --frames /path/to/pair --out benchmark.json

The pair directory contains near.jpg/far.jpg and near.png/far.png (uint16
millimetres). No cameras are opened. Use --hz 10 for paced deadline testing;
--hz 0 measures maximum throughput. Report current competing GPU workloads
alongside results; this is a recorded-frame benchmark, not live-loop validation.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = "1"

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--runtime-file", type=Path,
                    help="test a staged runtime without installing it")
    ap.add_argument("--reference", type=Path,
                    help="NPZ of ONNX logits for the same saved pair")
    ap.add_argument("--iterations", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--hz", type=float, default=10.0)
    args = ap.parse_args()
    if args.iterations < 1 or args.warmup < 0 or args.hz < 0:
        ap.error("iterations must be positive; warmup and hz non-negative")
    torch.set_num_threads(1)
    if args.runtime_file:
        spec = importlib.util.spec_from_file_location(
            "bebop_vision._benchmark_runtime", args.runtime_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        runtime_cls = module.BevStudentRuntime
    else:
        from bebop_vision.bev_student_runtime import BevStudentRuntime
        runtime_cls = BevStudentRuntime
    frames = {}
    for role in ("near", "far"):
        depth = cv2.imread(str(args.frames / f"{role}.png"), cv2.IMREAD_UNCHANGED)
        if depth is None or depth.dtype != np.uint16:
            raise ValueError(f"{role}.png must be uint16 depth in millimetres")
        frames[role] = SimpleNamespace(
            color_jpeg=(args.frames / f"{role}.jpg").read_bytes(), color=None,
            depth=depth, depth_scale_mm=1.0, stamp_us=1)
    rt = runtime_cls(args.engine, hazard_source="model")
    for _ in range(args.warmup):
        rt.infer(frames)
    rt.infer(frames)
    parity = {}
    for name, tensor in rt.outputs.items():
        if not torch.isfinite(tensor).all():
            raise RuntimeError(f"non-finite TensorRT output: {name}")
    if args.reference:
        with np.load(args.reference) as reference:
            for name, tensor in rt.outputs.items():
                value = tensor.float().cpu().numpy()
                ref = reference[name]
                agree = ((value.argmax(0) == ref.argmax(0)) if name == "semantic"
                         else ((value > 0) == (ref > 0)))
                parity[name] = {"max_abs": float(np.max(np.abs(value - ref))),
                                "agreement": float(agree.mean())}
    totals, pre, infer = [], [], []
    period = 1.0 / args.hz if args.hz else 0.0
    started = time.perf_counter()
    for _ in range(args.iterations):
        t0 = time.perf_counter()
        grid = rt.infer(frames)
        totals.append((time.perf_counter() - t0) * 1000)
        stats = grid.contributions["model"]
        pre.append(stats["preprocess_ms"])
        infer.append(stats["infer_ms"])
        if period:
            time.sleep(max(0.0, period - (time.perf_counter() - t0)))
    elapsed = time.perf_counter() - started

    def summary(xs):
        return {key: round(float(value), 3) for key, value in zip(
            ("p50", "p95", "p99", "max"), np.percentile(xs, [50, 95, 99, 100]))}

    result = {
        "engine": args.engine, "input_shape": list(rt.in_shape),
        "iterations": args.iterations, "target_hz": args.hz,
        "achieved_hz": args.iterations / elapsed,
        "total_ms": summary(totals), "preprocess_ms": summary(pre),
        "transfer_inference_ms": summary(infer),
        "ticks_over_100ms": sum(t > 100 for t in totals),
        "source": "saved RGB-D pair; JPEG decode through postprocess",
        "onnx_parity": parity,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
