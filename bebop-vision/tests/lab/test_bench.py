"""bench.py: ONNX export + sidecar, onnxruntime parity, the trtexec log
parser on canned output, and thor_latency's refusal / command plumbing
through a fake runner. Nothing here reaches a GPU, the network or the robot."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from bebop_vision.lab import bench

# --------------------------------------------------------------------------
# A ConvNeXt-shaped toy: stem conv, one block (dwconv 7x7 -> LayerNorm ->
# MLP with GELU -> residual), pooled LayerNorm head. Two outputs so parity
# covers both a feature map and a pooled vector.
# --------------------------------------------------------------------------


class Block(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.dw = nn.Conv2d(c, c, 7, padding=3, groups=c)
        self.norm = nn.LayerNorm(c, eps=1e-6)
        self.fc1 = nn.Linear(c, 4 * c)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(4 * c, c)

    def forward(self, x):
        y = self.dw(x).permute(0, 2, 3, 1)
        y = self.fc2(self.act(self.fc1(self.norm(y)))).permute(0, 3, 1, 2)
        return x + y


class TinyConvNeXt(nn.Module):
    def __init__(self, c: int = 8):
        super().__init__()
        self.stem = nn.Conv2d(3, c, 4, stride=4)
        self.block = Block(c)
        self.head = nn.LayerNorm(c)

    def forward(self, x):
        f = self.block(self.stem(x))
        return f, self.head(f.mean((2, 3)))


@pytest.fixture
def net():
    torch.manual_seed(0)
    return TinyConvNeXt().eval()


@pytest.fixture
def example():
    torch.manual_seed(1)
    return torch.randn(1, 3, 32, 32)


def _onnx_ops(path):
    import onnx
    g = onnx.load(str(path))
    return {n.op_type for n in g.graph.node}, g


# --------------------------------------------------------------------------
# export_onnx
# --------------------------------------------------------------------------

def test_export_writes_static_opset17_and_sidecar(net, example, tmp_path):
    out = bench.export_onnx(net, example, tmp_path / "tiny.onnx", output_names=("c5", "pooled"))
    assert out == tmp_path / "tiny.onnx" and out.stat().st_size > 0
    assert not (tmp_path / "tiny.onnx.data").exists()   # one file for trtexec

    ops, g = _onnx_ops(out)
    assert "LayerNormalization" in ops          # opset 17 keeps LayerNorm as one op
    assert [o.version for o in g.opset_import if o.domain in ("", "ai.onnx")] == [17]
    dims = [d.dim_value for d in g.graph.input[0].type.tensor_type.shape.dim]
    assert dims == [1, 3, 32, 32]               # static shapes by default
    assert [o.name for o in g.graph.output] == ["c5", "pooled"]

    side = json.loads(bench.sidecar_path(out).read_text())
    assert side["onnx"] == "tiny.onnx"
    assert side["input_shape"] == [1, 3, 32, 32]
    assert side["inputs"] == [{"name": "input", "shape": [1, 3, 32, 32], "dtype": "float32"}]
    assert [o["name"] for o in side["outputs"]] == ["c5", "pooled"]
    assert side["outputs"][1]["shape"] == [1, 8]
    assert side["opset"] == 17 and side["opset_requested"] == 17
    assert side["exporter"] == "torchscript"
    assert side["dynamic"] is False
    assert "git" in side                        # None outside a checkout, a short hash inside
    assert side["torch"] == torch.__version__
    assert side["size_bytes"] == out.stat().st_size
    assert side["params"] == sum(p.numel() for p in net.parameters())
    assert net.training is False                # mode restored


def test_export_restores_training_mode(net, example, tmp_path):
    net.train()
    bench.export_onnx(net, example, tmp_path / "t.onnx")
    assert net.training is True


def test_export_default_output_names(net, example, tmp_path):
    out = bench.export_onnx(net, example, tmp_path / "names.onnx")
    _, g = _onnx_ops(out)
    assert [o.name for o in g.graph.output] == ["output0", "output1"]


def test_export_dynamic_batch(net, example, tmp_path):
    out = bench.export_onnx(net, example, tmp_path / "dyn.onnx", dynamic=True,
                            output_names=("c5", "pooled"))
    _, g = _onnx_ops(out)
    d0 = g.graph.input[0].type.tensor_type.shape.dim[0]
    assert d0.dim_param == "batch" and d0.dim_value == 0
    side = json.loads(bench.sidecar_path(out).read_text())
    assert side["dynamic"] is True and side["dynamic_axis"] == "batch"


def test_export_dynamo_path_single_file(net, example, tmp_path):
    """The fallback exporter: still one file, sidecar says which ran and
    what opset actually landed (torch 2.11 may not downgrade to 17)."""
    out = bench.export_onnx(net, example, tmp_path / "dy.onnx", exporter="dynamo",
                            output_names=("c5", "pooled"))
    assert out.exists() and not (tmp_path / "dy.onnx.data").exists()
    side = json.loads(bench.sidecar_path(out).read_text())
    assert side["exporter"] == "dynamo"
    assert side["opset"] is not None and side["opset"] >= 17
    ops, _ = _onnx_ops(out)
    assert "LayerNormalization" in ops
    par = bench.onnx_parity(net, example, out, atol=1e-4, rtol=1e-4,
                            providers=("CPUExecutionProvider",))
    assert par["ok"], par


class TokensLike(nn.Module):
    """The tokens-kind Backbone output shape: two tensors and a grid of ints."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(3, 4, 2, stride=2)

    def forward(self, x):
        f = self.proj(x)
        return {"tokens": f.flatten(2).transpose(1, 2), "cls": f.mean((2, 3)),
                "grid": (x.shape[2] // 2, x.shape[3] // 2)}


def test_export_skips_non_tensor_outputs(tmp_path):
    """A tokens Backbone's grid (Python ints) cannot be a graph output; the
    tensors export, the sidecar names what was dropped, parity still runs."""
    torch.manual_seed(0)
    mod = TokensLike().eval()
    x = torch.randn(1, 3, 8, 8)
    out = bench.export_onnx(mod, x, tmp_path / "tok.onnx")
    _, g = _onnx_ops(out)
    assert [o.name for o in g.graph.output] == ["output0", "output1"]
    side = json.loads(bench.sidecar_path(out).read_text())
    assert side["non_tensor_outputs"] == ["grid[0]", "grid[1]"]   # one entry per dropped leaf
    assert side["module"] == "TokensLike"                  # the real module, not the wrapper
    assert [o["shape"] for o in side["outputs"]] == [[1, 16, 4], [1, 4]]
    par = bench.onnx_parity(mod, x, out, atol=1e-5, rtol=1e-5, providers=("CPUExecutionProvider",))
    assert par["ok"] and set(par["outputs"]) == {"output0", "output1"}
    # explicit names count tensors only, and the error says what was skipped
    with pytest.raises(ValueError, match="grid"):
        bench.export_onnx(mod, x, tmp_path / "x.onnx", output_names=("a", "b", "c"))

    class NoTensors(nn.Module):
        def forward(self, x):
            return {"grid": (1, 2)}

    with pytest.raises(TypeError, match="no tensors"):
        bench.export_onnx(NoTensors().eval(), x, tmp_path / "x.onnx")


def test_export_surfaces_tracer_warnings(tmp_path):
    """A Python shape check in forward is frozen into the trace; the exporter's
    TracerWarning is the only sign, so it is re-emitted and kept in the sidecar
    while the TorchScript deprecation noise stays silenced."""

    class Branchy(nn.Module):
        def forward(self, x):
            if x.shape[1] != 3:
                raise RuntimeError("bad channels")
            return x * 2

    with pytest.warns(torch.jit.TracerWarning) as rec:
        out = bench.export_onnx(Branchy().eval(), torch.randn(1, 3, 4, 4), tmp_path / "b.onnx")
    assert not any(issubclass(w.category, DeprecationWarning) for w in rec)
    side = json.loads(bench.sidecar_path(out).read_text())
    assert len(side["trace_warnings"]) == 1 and side["trace_warnings"][0].startswith("TracerWarning:")

    import warnings
    with warnings.catch_warnings(record=True) as quiet:
        warnings.simplefilter("always")
        out = bench.export_onnx(TinyConvNeXt().eval(), torch.randn(1, 3, 32, 32), tmp_path / "q.onnx")
    assert [w for w in quiet if issubclass(w.category, torch.jit.TracerWarning)] == []
    assert json.loads(bench.sidecar_path(out).read_text())["trace_warnings"] == []


def test_export_rejects_bad_names(net, example, tmp_path):
    with pytest.raises(ValueError):
        bench.export_onnx(net, example, tmp_path / "x.onnx", input_names=("a", "b"))
    with pytest.raises(ValueError):
        bench.export_onnx(net, example, tmp_path / "x.onnx", output_names=("only_one",))
    with pytest.raises(ValueError):
        bench.export_onnx(net, example, tmp_path / "x.onnx", exporter="magic")


# --------------------------------------------------------------------------
# onnx_parity
# --------------------------------------------------------------------------

def test_parity_cpu_tight(net, example, tmp_path):
    out = bench.export_onnx(net, example, tmp_path / "tiny.onnx", output_names=("c5", "pooled"))
    res = bench.onnx_parity(net, example, out, atol=1e-5, rtol=1e-5,
                            providers=("CPUExecutionProvider",))
    assert res["ok"] is True
    assert res["provider"] == "CPUExecutionProvider"
    assert set(res["outputs"]) == {"c5", "pooled"}
    c5, pooled = res["outputs"]["c5"], res["outputs"]["pooled"]
    assert c5["shape"] == [1, 8, 8, 8] and c5["max_abs"] < 1e-5 and c5["ok"]
    assert "cosine" not in c5                    # 4-D map: no cosine
    assert pooled["shape"] == [1, 8] and pooled["ok"]
    assert pooled["max_abs"] < 1e-5
    assert pooled["cosine"] > 1 - 1e-6           # 1-D pooled: cosine reported
    assert pooled["max_rel"] >= 0.0


def test_parity_flags_a_different_model(net, example, tmp_path):
    out = bench.export_onnx(net, example, tmp_path / "tiny.onnx", output_names=("c5", "pooled"))
    torch.manual_seed(99)
    other = TinyConvNeXt().eval()
    res = bench.onnx_parity(other, example, out, atol=1e-3, rtol=1e-3,
                            providers=("CPUExecutionProvider",))
    assert res["ok"] is False
    assert res["outputs"]["c5"]["ok"] is False
    assert res["outputs"]["pooled"]["cosine"] < 1 - 1e-3


def test_parity_shape_mismatch_is_not_ok(net, example, tmp_path):
    out = bench.export_onnx(net, example, tmp_path / "tiny.onnx", output_names=("c5", "pooled"))

    class Bigger(nn.Module):
        def forward(self, x):
            f, p = net(x)
            return f, torch.cat([p, p], 1)

    res = bench.onnx_parity(Bigger().eval(), example, out, providers=("CPUExecutionProvider",))
    assert res["ok"] is False
    assert res["outputs"]["pooled"] == {"ok": False, "shape_onnx": [1, 8], "shape_torch": [1, 16]}


# --------------------------------------------------------------------------
# parse_trtexec on canned TensorRT 10.x output
# --------------------------------------------------------------------------

TRT_LOG = """\
&&&& RUNNING TensorRT.trtexec [TensorRT v101600] [b26] # /usr/src/tensorrt/bin/trtexec --onnx=/tmp/bebop-lab/tiny.onnx --fp16 --useCudaGraph --iterations=200 --avgRuns=50 --warmUp=500 --noDataTransfers --dumpProfile --separateProfileRun --saveEngine=/tmp/bebop-lab/tiny.plan
[10/24/2026-21:03:11] [I] TF32 is enabled by default. Add --noTF32 flag to further improve accuracy with some performance cost.
[10/24/2026-21:03:11] [I] === Model Options ===
[10/24/2026-21:03:11] [I] Format: ONNX
[10/24/2026-21:03:11] [I] Model: /tmp/bebop-lab/tiny.onnx
[10/24/2026-21:03:11] [I] === Device Information ===
[10/24/2026-21:03:11] [I] Available Devices:
[10/24/2026-21:03:11] [I]   Device 0: "NVIDIA Thor" UUID: GPU-0d3f
[10/24/2026-21:03:11] [I] Selected Device: NVIDIA Thor
[10/24/2026-21:03:11] [I] Selected Device ID: 0
[10/24/2026-21:03:11] [I] Compute Capability: 11.0
[10/24/2026-21:03:11] [I] TensorRT version: 10.16.2
[10/24/2026-21:03:11] [I] Loading standard plugins
[10/24/2026-21:03:11] [I] [TRT] [MemUsageChange] Init CUDA: CPU +2, GPU +0, now: CPU 31, GPU 4850 (MiB)
[10/24/2026-21:03:11] [I] [TRT] [MemUsageChange] Init builder kernel library: CPU +1240, GPU +8, now: CPU 1301, GPU 4858 (MiB)
[10/24/2026-21:03:12] [I] Start parsing network model.
[10/24/2026-21:03:12] [I] [TRT] ----------------------------------------------------------------
[10/24/2026-21:03:12] [I] [TRT] Input filename:   /tmp/bebop-lab/tiny.onnx
[10/24/2026-21:03:12] [I] [TRT] ONNX IR version:  0.0.8
[10/24/2026-21:03:12] [I] [TRT] Opset version:    17
[10/24/2026-21:03:12] [I] [TRT] Producer name:    pytorch
[10/24/2026-21:03:12] [I] [TRT] Producer version: 2.11.0
[10/24/2026-21:03:12] [I] [TRT] Domain:
[10/24/2026-21:03:12] [I] [TRT] Model version:    0
[10/24/2026-21:03:12] [I] [TRT] Doc string:
[10/24/2026-21:03:12] [I] [TRT] ----------------------------------------------------------------
[10/24/2026-21:03:12] [I] Finished parsing network model. Parse time: 0.0123
[10/24/2026-21:03:12] [I] Set shape of input tensor input for optimization profile 0 to: MIN=1x3x32x32 OPT=1x3x32x32 MAX=1x3x32x32
[10/24/2026-21:03:12] [I] [TRT] Local timing cache in use. Profiling results in this builder pass will not be stored.
[10/24/2026-21:03:19] [I] [TRT] Detected 1 inputs and 2 output network tensors.
[10/24/2026-21:03:19] [I] [TRT] Total Host Persistent Memory: 36496 bytes
[10/24/2026-21:03:19] [I] [TRT] Total Device Persistent Memory: 0 bytes
[10/24/2026-21:03:19] [I] [TRT] Max Scratch Memory: 0 bytes
[10/24/2026-21:03:19] [I] [TRT] [BlockAssignment] Started assigning block shifts. This will take 9 steps to complete.
[10/24/2026-21:03:19] [I] [TRT] [BlockAssignment] Algorithm ShiftNTopDown took 0.0321ms to assign 3 blocks to 9 nodes requiring 8704 bytes.
[10/24/2026-21:03:19] [I] [TRT] Total Activation Memory: 8704 bytes
[10/24/2026-21:03:19] [I] [TRT] Total Weights Memory: 21632 bytes
[10/24/2026-21:03:19] [I] [TRT] Engine generation completed in 6.81 seconds.
[10/24/2026-21:03:19] [I] [TRT] [MemUsageStats] Peak memory usage of TRT CPU/GPU memory allocators: CPU 1 MiB, GPU 4 MiB
[10/24/2026-21:03:19] [I] [TRT] [MemUsageStats] Peak memory usage during Engine building and serialization: CPU: 1612 MiB
[10/24/2026-21:03:19] [I] Created engine with size: 0.145561 MiB
[10/24/2026-21:03:19] [I] Engine built in 7.02 sec.
[10/24/2026-21:03:19] [I] Created engine with size: 0.145561 MiB
[10/24/2026-21:03:19] [I] [TRT] Loaded engine size: 0 MiB
[10/24/2026-21:03:19] [I] Engine deserialized in 0.00291 sec.
[10/24/2026-21:03:19] [I] [TRT] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +0, GPU +1, now: CPU 0, GPU 1 (MiB)
[10/24/2026-21:03:19] [I] Setting persistentCacheLimit to 0 bytes.
[10/24/2026-21:03:19] [I] Created execution context with device memory size: 0.0083 MiB
[10/24/2026-21:03:19] [I] Using random values for input input
[10/24/2026-21:03:19] [I] Input binding for input with dimensions 1x3x32x32 is created.
[10/24/2026-21:03:19] [I] Output binding for c5 with dimensions 1x8x8x8 is created.
[10/24/2026-21:03:19] [I] Output binding for pooled with dimensions 1x8 is created.
[10/24/2026-21:03:19] [I] Starting inference
[10/24/2026-21:03:22] [I] Warmup completed 1032 queries over 500 ms
[10/24/2026-21:03:22] [I] Timing trace has 6248 queries over 3.00096 s
[10/24/2026-21:03:22] [I]
[10/24/2026-21:03:22] [I] === Trace details ===
[10/24/2026-21:03:22] [I] Trace averages of 50 runs:
[10/24/2026-21:03:22] [I] Average on 50 runs - GPU latency: 0.47631 ms - Host latency: 0.47631 ms (enqueue 0.0 ms)
[10/24/2026-21:03:22] [I] Average on 50 runs - GPU latency: 0.47912 ms - Host latency: 0.47912 ms (enqueue 0.0 ms)
[10/24/2026-21:03:22] [I]
[10/24/2026-21:03:22] [I] === Performance summary ===
[10/24/2026-21:03:22] [I] Throughput: 2081.83 qps
[10/24/2026-21:03:22] [I] Latency: min = 0.431152 ms, max = 1.15283 ms, mean = 0.479256 ms, median = 0.476562 ms, percentile(90%) = 0.491211 ms, percentile(95%) = 0.498047 ms, percentile(99%) = 0.541748 ms
[10/24/2026-21:03:22] [I] Enqueue Time: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(90%) = 0 ms, percentile(95%) = 0 ms, percentile(99%) = 0 ms
[10/24/2026-21:03:22] [I] H2D Latency: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(90%) = 0 ms, percentile(95%) = 0 ms, percentile(99%) = 0 ms
[10/24/2026-21:03:22] [I] GPU Compute Time: min = 0.431152 ms, max = 1.15283 ms, mean = 0.479256 ms, median = 0.476562 ms, percentile(90%) = 0.491211 ms, percentile(95%) = 0.498047 ms, percentile(99%) = 0.541748 ms
[10/24/2026-21:03:22] [I] D2H Latency: min = 0 ms, max = 0 ms, mean = 0 ms, median = 0 ms, percentile(90%) = 0 ms, percentile(95%) = 0 ms, percentile(99%) = 0 ms
[10/24/2026-21:03:22] [I] Total Host Walltime: 3.00096 s
[10/24/2026-21:03:22] [I] Total GPU Compute Time: 2.99439 s
[10/24/2026-21:03:22] [W] * GPU compute time is unstable, with coefficient of variance = 4.42112%.
[10/24/2026-21:03:22] [W]   If not already set, locking GPU clock frequency or adding --useSpinWait may improve the stability.
[10/24/2026-21:03:22] [I] Explanations of the performance metrics are printed in the verbose logs.
[10/24/2026-21:03:22] [I]
[10/24/2026-21:03:22] [I] The e2e network timing is not reported since it is inaccurate due to the extra synchronizations when the profiler is enabled.
[10/24/2026-21:03:22] [I] To show e2e network timing report, add --separateProfileRun to profile layer timing in a separate run or remove --dumpProfile to disable the profiler.
[10/24/2026-21:03:25] [I]
[10/24/2026-21:03:25] [I] === Profile (6214 iterations ) ===
[10/24/2026-21:03:25] [I]                                                               Layer   Time (ms)   Avg. Time (ms)   Median Time (ms)   Time %
[10/24/2026-21:03:25] [I]                           Reformatting CopyNode for Input Tensor 0 to /stem/Conv      181.25           0.0292             0.0287     10.2
[10/24/2026-21:03:25] [I]                                                          /stem/Conv      402.17           0.0647             0.0640     22.7
[10/24/2026-21:03:25] [I]                                                      /block/dw/Conv      503.81           0.0811             0.0803     28.4
[10/24/2026-21:03:25] [I]                               /block/norm/LayerNormalization + PWN(/block/act/Erf)      271.02           0.0436             0.0431     15.3
[10/24/2026-21:03:25] [I]                                                     /block/fc1/MatMul      165.44           0.0266             0.0262      9.3
[10/24/2026-21:03:25] [I]                                                     /block/fc2/MatMul      143.90           0.0232             0.0228      8.1
[10/24/2026-21:03:25] [I]                                                     /head/ReduceMean       59.60           0.0096             0.0094      3.4
[10/24/2026-21:03:25] [I]                                             /head/LayerNormalization       46.33           0.0075             0.0073      2.6
[10/24/2026-21:03:25] [I]                                                               Total     1773.52           0.2854             0.2818    100.0
[10/24/2026-21:03:25] [I]
&&&& PASSED TensorRT.trtexec [TensorRT v101600] [b26] # /usr/src/tensorrt/bin/trtexec --onnx=/tmp/bebop-lab/tiny.onnx --fp16 --useCudaGraph --iterations=200 --avgRuns=50 --warmUp=500 --noDataTransfers --dumpProfile --separateProfileRun --saveEngine=/tmp/bebop-lab/tiny.plan
"""


def test_parse_trtexec_summary():
    r = bench.parse_trtexec(TRT_LOG, top_n=3)
    assert r["passed"] is True
    assert r["trt_version"] == "10.16.2"      # the info line carries the patch level
    assert r["throughput_qps"] == pytest.approx(2081.83)
    gpu = r["gpu_compute_ms"]
    assert gpu == {"min": 0.431152, "max": 1.15283, "mean": 0.479256, "median": 0.476562,
                   "p90": 0.491211, "p95": 0.498047, "p99": 0.541748}
    assert r["median_ms"] == 0.476562 and r["mean_ms"] == 0.479256 and r["p99_ms"] == 0.541748
    assert r["latency_ms"]["median"] == 0.476562
    assert r["enqueue_ms"]["max"] == 0.0 and r["h2d_ms"]["p99"] == 0.0 and r["d2h_ms"]["min"] == 0.0
    assert r["total_host_walltime_s"] == pytest.approx(3.00096)
    assert r["total_gpu_compute_s"] == pytest.approx(2.99439)
    assert r["engine_size_mib"] == pytest.approx(0.145561)
    assert r["engine_build_s"] == pytest.approx(7.02)
    assert r["peak_gpu_mib"] == 4.0 and r["peak_cpu_mib"] == 1.0
    assert r["context_gpu_mib"] == 1.0
    assert r["activation_bytes"] == 8704 and r["weights_bytes"] == 21632
    assert any("unstable" in w for w in r["warnings"])


def test_parse_trtexec_profile_top_layers():
    r = bench.parse_trtexec(TRT_LOG, top_n=3)
    prof = r["profile"]
    assert prof["iterations"] == 6214
    assert prof["layers"] == 8
    assert prof["total_avg_ms"] == pytest.approx(0.2854)
    assert prof["total_median_ms"] == pytest.approx(0.2818)
    top = prof["top"]
    assert [t["layer"] for t in top] == ["/block/dw/Conv", "/stem/Conv",
                                         "/block/norm/LayerNormalization + PWN(/block/act/Erf)"]
    assert top[0] == {"layer": "/block/dw/Conv", "time_ms": 503.81, "avg_ms": 0.0811,
                      "median_ms": 0.0803, "pct": 28.4}
    # a layer name with spaces survives
    r_all = bench.parse_trtexec(TRT_LOG, top_n=20)
    names = {t["layer"] for t in r_all["profile"]["top"]}
    assert "Reformatting CopyNode for Input Tensor 0 to /stem/Conv" in names
    assert "Total" not in names


def test_parse_trtexec_without_profile_or_summary():
    r = bench.parse_trtexec("")
    assert r["passed"] is None and r["median_ms"] is None and r["gpu_compute_ms"] is None
    assert r["profile"] is None and r["warnings"] == []
    failed = "[10/24/2026-21:03:12] [E] Error[4]: unsupported op\n&&&& FAILED TensorRT.trtexec [TensorRT v101600]\n"
    r = bench.parse_trtexec(failed)
    assert r["passed"] is False and r["trt_version"] == "10.16.0"


def test_parse_trtexec_verbose_explanations_keep_the_numbers():
    """Under --verbose trtexec prints "[V] GPU Compute Time: the GPU latency
    to ..." explanations after the summary; those key lines carry no numbers
    and must not erase the parsed block."""
    verbose = TRT_LOG.replace(
        "[10/24/2026-21:03:22] [I] Explanations of the performance metrics are printed in the verbose logs.",
        "[10/24/2026-21:03:22] [V] === Explanations of the performance metrics ===\n"
        "[10/24/2026-21:03:22] [V] GPU Compute Time: the GPU latency to execute the kernels for a query.\n"
        "[10/24/2026-21:03:22] [V] Enqueue Time: the host latency to enqueue a query. If this is longer than GPU Compute Time, the GPU may be under-utilized.\n"
        "[10/24/2026-21:03:22] [V] H2D Latency: the latency for host-to-device data transfers for input tensors of a single query.\n"
        "[10/24/2026-21:03:22] [V] D2H Latency: the latency for device-to-host data transfers for output tensors of a single query.\n"
        "[10/24/2026-21:03:22] [V] Latency: the summation of H2D Latency, GPU Compute Time, and D2H Latency. This is the latency to infer a single query.")
    assert verbose != TRT_LOG
    r = bench.parse_trtexec(verbose)
    assert r["median_ms"] == 0.476562 and r["p99_ms"] == 0.541748
    assert r["latency_ms"]["median"] == 0.476562
    assert r["enqueue_ms"]["max"] == 0.0 and r["h2d_ms"]["p99"] == 0.0 and r["d2h_ms"]["min"] == 0.0
    assert r == bench.parse_trtexec(TRT_LOG)


def test_parse_trtexec_version_banners():
    # 8.x banner packs Mmpb; 10.x packs MMmmpp; the info line wins when present.
    assert bench.parse_trtexec("&&&& PASSED TensorRT.trtexec [TensorRT v8601] # trtexec\n")["trt_version"] == "8.6.0"
    assert bench.parse_trtexec("&&&& PASSED TensorRT.trtexec [TensorRT v101600] # trtexec\n")["trt_version"] == "10.16.0"
    text = "[01/01/2026-00:00:00] [I] TensorRT version: 10.16.2\n&&&& PASSED TensorRT.trtexec [TensorRT v101600]\n"
    assert bench.parse_trtexec(text)["trt_version"] == "10.16.2"
    assert bench.parse_trtexec("no banner here")["trt_version"] is None


def test_parse_trtexec_old_profile_table_without_median():
    text = """\
[01/01/2025-00:00:00] [I] === Profile (100 iterations ) ===
[01/01/2025-00:00:00] [I]    Layer   Time (ms)   Avg. Time (ms)   Time %
[01/01/2025-00:00:00] [I]    conv1       10.00           0.1000     40.0
[01/01/2025-00:00:00] [I]    conv2       15.00           0.1500     60.0
[01/01/2025-00:00:00] [I]    Total       25.00           0.2500    100.0
"""
    prof = bench.parse_trtexec(text)["profile"]
    assert prof["layers"] == 2 and prof["total_avg_ms"] == 0.25 and prof["total_median_ms"] is None
    assert prof["top"][0]["layer"] == "conv2" and prof["top"][0]["median_ms"] is None


# --------------------------------------------------------------------------
# thor_latency through a fake runner
# --------------------------------------------------------------------------

class FakeRunner:
    """Scripted ssh/scp replies keyed on the remote command; records calls."""

    def __init__(self, recorder_running=False, trtexec_log=TRT_LOG, trtexec_rc=0):
        self.calls = []
        self.recorder_running = recorder_running
        self.trtexec_log = trtexec_log
        self.trtexec_rc = trtexec_rc

    def __call__(self, cmd, *, timeout, env=None):
        self.calls.append({"cmd": list(cmd), "timeout": timeout, "env": env})
        remote = cmd[-1]
        if remote.startswith("pgrep"):
            if self.recorder_running:
                return subprocess.CompletedProcess(cmd, 0,
                    "4321 /home/bebop/bebop/bebop-vision/.venv/bin/python main.py --record-navd\n", "")
            return subprocess.CompletedProcess(cmd, 1, "", "")
        if remote.startswith("mkdir"):
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if cmd[0] in ("scp",) or (cmd[0] == "sshpass" and cmd[2] == "scp"):
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if "trtexec" in remote:
            return subprocess.CompletedProcess(cmd, self.trtexec_rc, self.trtexec_log, "")
        if remote.startswith("stat"):
            return subprocess.CompletedProcess(cmd, 0, "152632\n", "")
        raise AssertionError(f"unexpected command {cmd}")


@pytest.fixture
def onnx_file(net, example, tmp_path):
    return bench.export_onnx(net, example, tmp_path / "tiny.onnx", output_names=("c5", "pooled"))


def test_thor_latency_refuses_when_recorder_runs(onnx_file, monkeypatch):
    monkeypatch.delenv("BEBOP_SSH_PASSWORD", raising=False)
    fake = FakeRunner(recorder_running=True)
    with pytest.raises(bench.RecorderRunning) as ei:
        bench.thor_latency(onnx_file, runner=fake)
    msg = str(ei.value)
    assert "record-navd" in msg and "bebop.local" in msg and "refusing" in msg
    # exactly one ssh call (the pgrep), no scp, no trtexec
    assert len(fake.calls) == 1
    cmd = fake.calls[0]["cmd"]
    assert cmd[0] == "ssh" and "BatchMode=yes" in cmd and cmd[-1] == bench.RECORDER_PGREP
    assert cmd[-2] == "bebop@bebop.local"


def test_thor_latency_refusal_via_monkeypatched_default_runner(onnx_file, monkeypatch):
    fake = FakeRunner(recorder_running=True)
    monkeypatch.setattr(bench, "_run", fake)
    with pytest.raises(bench.RecorderRunning):
        bench.thor_latency(onnx_file, host="thor.test", user="u")
    assert fake.calls[0]["cmd"][-2] == "u@thor.test"


def test_thor_latency_happy_path_key_auth(onnx_file, monkeypatch):
    monkeypatch.delenv("BEBOP_SSH_PASSWORD", raising=False)
    fake = FakeRunner()
    r = bench.thor_latency(onnx_file, iterations=123, fp16=True, cuda_graph=False, timeout_s=321,
                           runner=fake, top_n=2)
    kinds = [(c["cmd"][0], c["cmd"][-1].split()[0]) for c in fake.calls]
    assert kinds == [("ssh", "pgrep"), ("ssh", "mkdir"), ("scp", f"bebop@bebop.local:/tmp/bebop-lab/tiny.onnx"),
                     ("ssh", bench.TRTEXEC), ("ssh", "stat")]
    for c in fake.calls:
        assert "BatchMode=yes" in c["cmd"] and "SSHPASS" not in (c["env"] or {})
    trt = fake.calls[3]["cmd"][-1]
    assert "--onnx=/tmp/bebop-lab/tiny.onnx" in trt and "--fp16" in trt
    assert "--useCudaGraph" not in trt
    assert "--iterations=123" in trt and "--avgRuns=50" in trt and "--warmUp=500" in trt
    assert "--noDataTransfers" in trt and "--dumpProfile" in trt and "--separateProfileRun" in trt
    assert "--saveEngine=/tmp/bebop-lab/tiny.plan" in trt
    assert fake.calls[2]["timeout"] == 321 and fake.calls[3]["timeout"] == 321
    # nothing that drives: no curl / firmware / systemctl / CAN anywhere
    joined = " ".join(" ".join(c["cmd"]) for c in fake.calls)
    for forbidden in ("curl", "9090", "systemctl", "cansend", "odrive", "cmd_vel"):
        assert forbidden not in joined

    assert r["passed"] is True and r["median_ms"] == 0.476562 and r["p99_ms"] == 0.541748
    assert r["engine"] == "/tmp/bebop-lab/tiny.plan" and r["engine_bytes"] == 152632
    assert r["remote_onnx"] == "/tmp/bebop-lab/tiny.onnx" and r["host"] == "bebop.local"
    assert r["iterations"] == 123 and r["fp16"] is True and r["cuda_graph"] is False
    assert r["export"]["opset"] == 17                  # sidecar travels with the result
    assert len(r["profile"]["top"]) == 2
    assert r["command"].startswith(bench.TRTEXEC)
    assert "=== Performance summary ===" in r["log"]


def test_thor_latency_uses_sshpass_when_password_set(onnx_file, monkeypatch):
    monkeypatch.setenv("BEBOP_SSH_PASSWORD", "hunter2")
    fake = FakeRunner()
    bench.thor_latency(onnx_file, remote_dir="/data/lab/", runner=fake)
    for c in fake.calls:
        assert c["cmd"][:2] == ["sshpass", "-e"]
        assert c["cmd"][2] in ("ssh", "scp")
        assert "BatchMode=yes" not in c["cmd"]          # would suppress the password login
        assert c["env"]["SSHPASS"] == "hunter2"         # sshpass -e reads this, not BEBOP_*
    assert fake.calls[2]["cmd"][-1] == "bebop@bebop.local:/data/lab/tiny.onnx"
    assert "--saveEngine=/data/lab/tiny.plan" in fake.calls[3]["cmd"][-1]


def test_thor_latency_trtexec_failure_raises_with_tail(onnx_file, monkeypatch):
    monkeypatch.delenv("BEBOP_SSH_PASSWORD", raising=False)
    log = "[10/24/2026-21:03:12] [E] Error[4]: /block/dw/Conv: unsupported\n&&&& FAILED TensorRT.trtexec [TensorRT v101600]\n"
    fake = FakeRunner(trtexec_log=log, trtexec_rc=1)
    with pytest.raises(bench.BenchError) as ei:
        bench.thor_latency(onnx_file, runner=fake)
    assert "unsupported" in str(ei.value) and "exit 1" in str(ei.value)
    assert not any(c["cmd"][-1].startswith("stat") for c in fake.calls)


def test_thor_latency_ssh_failure_is_not_a_refusal(onnx_file, monkeypatch):
    monkeypatch.delenv("BEBOP_SSH_PASSWORD", raising=False)

    def runner(cmd, *, timeout, env=None):
        return subprocess.CompletedProcess(cmd, 255, "", "ssh: connect to host bebop.local port 22: No route to host")

    with pytest.raises(bench.BenchError) as ei:
        bench.thor_latency(onnx_file, runner=runner)
    assert not isinstance(ei.value, bench.RecorderRunning)
    assert "No route to host" in str(ei.value)


def test_thor_latency_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        bench.thor_latency(tmp_path / "nope.onnx", runner=FakeRunner())


def test_thor_latency_local_failures_are_bench_errors(onnx_file, monkeypatch):
    """A timeout or a missing ssh/sshpass binary is a BenchError, not the
    FileNotFoundError that means 'export first'."""
    monkeypatch.delenv("BEBOP_SSH_PASSWORD", raising=False)

    def slow(cmd, *, timeout, env=None):
        raise subprocess.TimeoutExpired(cmd, timeout)

    with pytest.raises(bench.BenchTimeout) as ei:
        bench.thor_latency(onnx_file, runner=slow)
    assert isinstance(ei.value, bench.BenchError) and "60 s" in str(ei.value)

    def no_binary(cmd, *, timeout, env=None):
        raise FileNotFoundError(2, "No such file or directory", cmd[0])

    with pytest.raises(bench.BenchError) as ei:
        bench.thor_latency(onnx_file, runner=no_binary)
    assert not isinstance(ei.value, FileNotFoundError)
    assert "'ssh' is not installed" in str(ei.value)


def test_thor_latency_rejects_unsafe_remote_paths(onnx_file, tmp_path):
    """scp hands the remote path to a shell under the legacy protocol but not
    under SFTP (OpenSSH 9.0+ default), so no quoting matches both; paths that
    could make scp, mkdir and trtexec disagree are refused before any ssh."""
    fake = FakeRunner()
    with pytest.raises(ValueError, match="remote_dir"):
        bench.thor_latency(onnx_file, remote_dir="/tmp/my lab", runner=fake)
    with pytest.raises(ValueError, match="remote_dir"):
        bench.thor_latency(onnx_file, remote_dir="/tmp/$(reboot)", runner=fake)
    spaced = tmp_path / "my model.onnx"
    spaced.write_bytes(onnx_file.read_bytes())
    with pytest.raises(ValueError, match="ONNX filename"):
        bench.thor_latency(spaced, runner=fake)
    assert fake.calls == []
    with pytest.raises(ValueError):
        bench.scp_command("/x/a.onnx", "u@h", "/tmp/my lab/a.onnx")
    assert bench.scp_command("/x/a.onnx", "u@h", "/tmp/lab-1/a_b.onnx")[-1] == "u@h:/tmp/lab-1/a_b.onnx"


# --------------------------------------------------------------------------
# import-time hygiene
# --------------------------------------------------------------------------

def test_bench_imports_without_torch_or_onnxruntime():
    """The package must import on the robot without the workstation's
    export stack; torch / onnx / onnxruntime load lazily."""
    code = ("import sys; import bebop_vision.lab.bench as b; "
            "bad = [m for m in ('torch', 'onnx', 'onnxruntime') if m in sys.modules]; "
            "print(','.join(bad))")
    root = Path(__file__).resolve().parents[2]
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=str(root), env={**os.environ, "PYTHONPATH": str(root)}, check=True)
    assert out.stdout.strip() == "", f"eagerly imported: {out.stdout.strip()}"
