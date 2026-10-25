"""ONNX export, onnxruntime parity and on-Thor ``trtexec`` latency.

The deployment path for every lab model is the one ``docs/path-model.md``
pins: export to ONNX on the workstation, check it against the torch module
with onnxruntime, then build and time the TensorRT engine on the robot
itself (TensorRT 10.16, sm_110; an engine built on the RTX 5090 does not
deserialize there). This module is those three steps as functions, so a CLI
or a notebook can run them and keep the numbers next to the run manifest.

``thor_latency`` is the only thing here that touches the robot. It talks to
the Thor over ssh/scp, refuses to run while the recorder owns the cameras
and the GPU (AGENTS.md: never test mid-session), and never sends anything
that could move the robot: it copies a file and runs ``trtexec``.

Import-time dependencies are numpy and the standard library; torch,
onnx and onnxruntime are imported inside the functions that need them so
the package imports on the robot without them.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import re
import shlex
import subprocess
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from bebop_vision.lab.types import git_hash

#: The recorder check from AGENTS.md. ``pgrep`` exits 0 with matches, 1 without;
#: the bracket trick keeps the pgrep shell itself out of the match.
RECORDER_PGREP = "pgrep -af 'record-nav[d]|main.p[y]'"
#: Where JetPack installs trtexec on the Thor.
TRTEXEC = "/usr/src/tensorrt/bin/trtexec"
#: Default remote scratch directory for ONNX files and engines.
REMOTE_DIR = "/tmp/bebop-lab"

#: ``runner(cmd, timeout=..., env=...) -> subprocess.CompletedProcess``.
Runner = Callable[..., "subprocess.CompletedProcess[str]"]


class BenchError(RuntimeError):
    """A bench step failed (ssh, scp or trtexec)."""


class RecorderRunning(BenchError):
    """The robot's recorder (``bebop-vision``) is running; the GPU is not ours."""


class BenchTimeout(BenchError):
    """An ssh/scp/trtexec call ran past its timeout."""


# --------------------------------------------------------------------------
# ONNX export
# --------------------------------------------------------------------------

def _as_args(example: Any) -> tuple:
    return tuple(example) if isinstance(example, (tuple, list)) else (example,)


def _leaves(out: Any, path: str = "") -> List[Tuple[str, Any]]:
    """Every leaf of a module output (tuples / lists / dicts in declaration
    order, which is also the order the exporters use) as ``(path, value)``."""
    if isinstance(out, dict):
        items = [(f"{path}.{k}" if path else str(k), v) for k, v in out.items()]
    elif isinstance(out, (tuple, list)):
        items = [(f"{path}[{i}]", v) for i, v in enumerate(out)]
    else:
        return [(path or type(out).__name__, out)]
    flat: List[Tuple[str, Any]] = []
    for p, v in items:
        flat.extend(_leaves(v, p))
    return flat


def _flatten_outputs(out: Any) -> Tuple[list, List[str], List[int]]:
    """Module outputs as a flat list of tensors, the paths of the non-tensor
    leaves that were skipped, and the leaf positions of the tensors.

    A tokens-kind Backbone returns ``{"tokens", "cls", "grid": (h, w)}``; the
    grid is Python ints, which neither exporter can emit as a graph output,
    so :func:`export_onnx` exports the tensors only and records the dropped
    names in the sidecar.
    """
    import torch
    tensors, dropped, keep = [], [], []
    for i, (p, v) in enumerate(_leaves(out)):
        if isinstance(v, torch.Tensor):
            tensors.append(v)
            keep.append(i)
        else:
            dropped.append(p)
    return tensors, dropped, keep


def _tensor_only(module, keep: Sequence[int]):
    """Wrap ``module`` so its forward returns the leaves at positions
    ``keep`` as a tuple. The positions come from the eager run: under the
    TorchScript tracer ``x.shape[2]`` is a 0-d tensor, so filtering by type
    at trace time would keep the ints the eager module returns."""
    import torch

    class TensorOutputs(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, *args):
            leaves = _leaves(self.inner(*args))
            return tuple(leaves[i][1] for i in keep)

    return TensorOutputs(module)


def _tensor_meta(t) -> dict:
    return {"shape": [int(s) for s in t.shape], "dtype": str(t.dtype).replace("torch.", "")}


def _onnx_opset(path: Path) -> Optional[int]:
    """The default-domain opset actually written to the file (the dynamo
    exporter may not honour the request; see ``export_onnx``)."""
    try:
        import onnx
        model = onnx.load(str(path), load_external_data=False)
    except Exception:
        return None
    for imp in model.opset_import:
        if imp.domain in ("", "ai.onnx"):
            return int(imp.version)
    return None


def sidecar_path(onnx_path: Path | str) -> Path:
    """``model.onnx`` -> ``model.json`` beside it."""
    return Path(onnx_path).with_suffix(".json")


def export_onnx(module, example, path, opset: int = 17, dynamic: bool = False,
                input_names: Sequence[str] = ("input",), output_names: Optional[Sequence[str]] = None,
                exporter: str = "auto") -> Path:
    """Export ``module`` to ONNX at ``path`` and write ``<path>.json`` beside it.

    Opset 17 so LayerNorm is one ``LayerNormalization`` op (TensorRT fuses
    it; the opset-16 decomposition is a dozen kernels). Shapes are static by
    default, which is what trtexec builds the fastest engine for; ``dynamic``
    frees the batch axis only.

    ``exporter``: ``"torchscript"`` (``dynamo=False``, the default under
    ``"auto"`` because on torch 2.11 it writes exactly the requested opset
    into one file), ``"dynamo"`` (``torch.export`` based; it may emit a
    higher opset when the version converter cannot downgrade, measured
    opset 18 for a ConvNeXt block asked for 17), or ``"auto"`` which falls
    back to dynamo when the torchscript exporter raises. The sidecar records
    which one ran and the opset that landed in the file.
    """
    import torch

    if exporter not in ("auto", "torchscript", "dynamo"):
        raise ValueError(f"exporter must be auto, torchscript or dynamo, not {exporter!r}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    args = _as_args(example)
    input_names = list(input_names)
    if len(input_names) != len(args):
        raise ValueError(f"{len(input_names)} input names for {len(args)} example inputs")

    was_training = module.training
    module.eval()
    trace_warnings: List[str] = []
    try:
        with torch.no_grad():
            ref, dropped, keep = _flatten_outputs(module(*args))
        if not ref:
            raise TypeError(f"{type(module).__qualname__} returned no tensors "
                            f"(got {', '.join(dropped) or 'nothing'}); nothing to export")
        if output_names is None:
            output_names = ["output"] if len(ref) == 1 else [f"output{i}" for i in range(len(ref))]
        output_names = list(output_names)
        if len(output_names) != len(ref):
            raise ValueError(f"{len(output_names)} output names for {len(ref)} tensor outputs"
                             + (f" (non-tensor outputs skipped: {', '.join(dropped)})" if dropped else ""))
        # Ints in the output structure (a tokens Backbone's grid) cannot be
        # graph outputs; trace a tensor-only view of the module instead.
        exported = _tensor_only(module, keep) if dropped else module

        used, errors = None, []
        caught: list = []
        if exporter in ("auto", "torchscript"):
            try:
                dynamic_axes = ({n: {0: "batch"} for n in input_names + output_names}
                                if dynamic else None)
                with warnings.catch_warnings(record=True) as caught:
                    # torch 2.11 deprecates the TorchScript exporter loudly;
                    # it is still the one that writes the opset we ask for.
                    # TracerWarnings stay: they are the only sign that a
                    # shape check or Python branch was frozen into the graph.
                    warnings.simplefilter("always")
                    warnings.filterwarnings("ignore", category=DeprecationWarning)
                    warnings.filterwarnings("ignore", category=FutureWarning)
                    torch.onnx.export(exported, args, str(path), opset_version=opset, dynamo=False,
                                      input_names=input_names, output_names=output_names,
                                      dynamic_axes=dynamic_axes, do_constant_folding=True)
                used = "torchscript"
            except Exception as e:  # noqa: BLE001 - any exporter failure falls through
                errors.append(f"torchscript: {type(e).__name__}: {e}")
                if exporter == "torchscript":
                    raise
            for w in caught:
                msg = f"{w.category.__name__}: {str(w.message).strip().splitlines()[0]}"
                trace_warnings.append(msg)
                warnings.warn(msg, w.category, stacklevel=2)
        if used is None:
            dynamic_shapes = None
            if dynamic:
                batch = torch.export.Dim("batch")
                dynamic_shapes = tuple({0: batch} for _ in args)
            try:
                # external_data=False: the default splits weights into a
                # ``.data`` file, and trtexec wants one file.
                torch.onnx.export(exported, args, str(path), opset_version=opset, dynamo=True,
                                  input_names=input_names, output_names=output_names,
                                  dynamic_shapes=dynamic_shapes, external_data=False)
            except Exception as e:  # noqa: BLE001
                errors.append(f"dynamo: {type(e).__name__}: {e}")
                raise BenchError("ONNX export failed: " + "; ".join(errors)) from e
            used = "dynamo"
    finally:
        module.train(was_training)

    if not path.exists():
        raise BenchError(f"exporter {used} wrote nothing at {path}")
    n_params = sum(int(p.numel()) for p in module.parameters())
    meta = {
        "onnx": path.name,
        "inputs": [dict(name=n, **_tensor_meta(t)) for n, t in zip(input_names, args)],
        "outputs": [dict(name=n, **_tensor_meta(t)) for n, t in zip(output_names, ref)],
        "input_shape": _tensor_meta(args[0])["shape"],
        "opset_requested": int(opset),
        "opset": _onnx_opset(path),
        "dynamic": bool(dynamic),
        "dynamic_axis": "batch" if dynamic else None,
        "exporter": used,
        "exporter_errors": errors,
        "trace_warnings": trace_warnings,
        "non_tensor_outputs": dropped,
        "module": type(module).__qualname__,
        "params": n_params,
        "torch": torch.__version__,
        "git": git_hash(Path(__file__).resolve().parent),
        "created": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "size_bytes": path.stat().st_size,
    }
    sidecar_path(path).write_text(json.dumps(meta, indent=2) + "\n")
    return path


# --------------------------------------------------------------------------
# onnxruntime parity
# --------------------------------------------------------------------------

def _default_providers() -> List[str]:
    import onnxruntime as ort
    available = ort.get_available_providers()
    out = []
    if "CUDAExecutionProvider" in available:
        out.append("CUDAExecutionProvider")
    out.append("CPUExecutionProvider")
    return out


def _cosine_rows(a: np.ndarray, b: np.ndarray) -> float:
    """Minimum over batch rows of the cosine between ``a`` and ``b`` seen as
    ``[B, D]`` (a ``[D]`` vector is one row)."""
    a2 = a.reshape(-1, a.shape[-1]) if a.ndim >= 1 else a.reshape(1, 1)
    b2 = b.reshape(-1, b.shape[-1]) if b.ndim >= 1 else b.reshape(1, 1)
    num = np.sum(a2 * b2, axis=1)
    den = np.linalg.norm(a2, axis=1) * np.linalg.norm(b2, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        cos = np.where(den > 0, num / den, 1.0)
    return float(np.min(cos)) if cos.size else 1.0


def onnx_parity(module, example, onnx_path, atol: float = 1e-2, rtol: float = 1e-2,
                providers: Optional[Sequence[str]] = None) -> dict:
    """Run ``module`` and the ONNX file on the same ``example`` and compare
    every output.

    Per output: ``max_abs``, ``max_rel`` (``|a-b| / max(|b|, 1e-6)``, so it
    is noisy where the reference is near zero and ``ok`` is the allclose
    check ``|a-b| <= atol + rtol*|b|`` instead), and ``cosine`` for pooled
    outputs (``[B, D]`` or ``[D]``, the minimum over the batch). The
    session uses ``CUDAExecutionProvider`` when onnxruntime has it, else
    CPU; ``provider`` says which one actually ran.
    """
    import torch
    import onnxruntime as ort

    onnx_path = Path(onnx_path)
    args = _as_args(example)
    was_training = module.training
    module.eval()
    try:
        with torch.no_grad():
            ref = [t.detach().cpu().float().numpy() for t in _flatten_outputs(module(*args))[0]]
    finally:
        module.train(was_training)

    wanted = list(providers) if providers else _default_providers()
    so = ort.SessionOptions()
    so.log_severity_level = 3   # errors only; a missing cuDNN is reported via `provider`
    try:
        sess = ort.InferenceSession(str(onnx_path), so, providers=wanted)
    except Exception as e:  # noqa: BLE001 - CUDA EP without its libraries
        if wanted == ["CPUExecutionProvider"]:
            raise
        warnings.warn(f"onnxruntime {wanted[0]} unavailable ({e}); using CPU")
        sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
    inputs = sess.get_inputs()
    if len(inputs) != len(args):
        raise ValueError(f"{onnx_path.name} has {len(inputs)} inputs, example has {len(args)}")
    feeds = {inp.name: a.detach().cpu().numpy() for inp, a in zip(inputs, args)}
    got = sess.run(None, feeds)
    names = [o.name for o in sess.get_outputs()]

    outputs: Dict[str, dict] = {}
    all_ok = len(got) == len(ref)
    for i, (name, a) in enumerate(zip(names, got)):
        a = np.asarray(a, dtype=np.float64)
        if i >= len(ref):
            outputs[name] = {"ok": False, "shape_onnx": list(a.shape), "shape_torch": None}
            all_ok = False
            continue
        b = np.asarray(ref[i], dtype=np.float64)
        if a.shape != b.shape:
            outputs[name] = {"ok": False, "shape_onnx": list(a.shape), "shape_torch": list(b.shape)}
            all_ok = False
            continue
        diff = np.abs(a - b)
        rec: Dict[str, Any] = {
            "shape": list(a.shape),
            "max_abs": float(diff.max()) if diff.size else 0.0,
            "max_rel": float((diff / np.maximum(np.abs(b), 1e-6)).max()) if diff.size else 0.0,
            "ok": bool(np.allclose(a, b, atol=atol, rtol=rtol)),
        }
        if a.ndim <= 2:
            rec["cosine"] = _cosine_rows(a, b)
        outputs[name] = rec
        all_ok = all_ok and rec["ok"]
    return {
        "onnx": str(onnx_path),
        "provider": sess.get_providers()[0],
        "atol": atol, "rtol": rtol,
        "ok": bool(all_ok),
        "outputs": outputs,
    }


# --------------------------------------------------------------------------
# On-Thor trtexec
# --------------------------------------------------------------------------

_SSH_OPTS = ["-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=15"]


def ssh_command(target: str, remote_cmd: str, password: Optional[str] = None) -> List[str]:
    """``ssh`` (or ``sshpass -e ssh``) argv for ``remote_cmd`` on ``target``.

    With a password, ``sshpass -e`` reads it from ``SSHPASS`` in the child's
    environment (``_env``); without one, ``BatchMode=yes`` makes a missing
    key fail fast instead of prompting.
    """
    if password:
        return ["sshpass", "-e", "ssh", *_SSH_OPTS, target, remote_cmd]
    return ["ssh", "-o", "BatchMode=yes", *_SSH_OPTS, target, remote_cmd]


#: Characters a remote path may not contain. Whether ``scp`` hands the remote
#: path to a shell depends on its protocol (legacy scp does, the SFTP mode
#: that OpenSSH 9.0+ defaults to does not), so no quoting is right for both;
#: instead the path is kept to characters that mean the same under ``scp``,
#: the quoted ``mkdir`` and the quoted ``trtexec --onnx=`` argument.
_REMOTE_PATH_OK = re.compile(r"^[A-Za-z0-9._~/@%+,:=-]+$")


def check_remote_path(path: str, what: str = "remote path") -> str:
    """Reject a remote path that whitespace or shell metacharacters would
    make ``scp``, ``mkdir`` and ``trtexec`` disagree about."""
    if not path or not _REMOTE_PATH_OK.match(path):
        raise ValueError(f"{what} {path!r} contains whitespace or shell metacharacters; "
                         "use plain [A-Za-z0-9._-/] names on the robot")
    return path


def scp_command(local: Path | str, target: str, remote_path: str,
                password: Optional[str] = None) -> List[str]:
    """``scp`` (or ``sshpass -e scp``) argv copying ``local`` to
    ``target:remote_path``. ``remote_path`` is passed verbatim (see
    :func:`check_remote_path`)."""
    check_remote_path(remote_path)
    if password:
        return ["sshpass", "-e", "scp", *_SSH_OPTS, str(local), f"{target}:{remote_path}"]
    return ["scp", "-o", "BatchMode=yes", *_SSH_OPTS, str(local), f"{target}:{remote_path}"]


def trtexec_command(remote_onnx: str, engine: str, fp16: bool = True, cuda_graph: bool = True,
                    iterations: int = 200, trtexec: str = TRTEXEC,
                    extra_args: Sequence[str] = ()) -> List[str]:
    """The trtexec argv. Flags kept to ones present in TensorRT 10.x:
    ``--noDataTransfers`` times the GPU alone (the recorder keeps frames on
    device anyway), ``--separateProfileRun`` keeps the per-layer profile from
    disturbing the timing run, ``--saveEngine`` leaves the plan on the robot
    for the runtime to pick up."""
    cmd = [trtexec, f"--onnx={remote_onnx}"]
    if fp16:
        cmd.append("--fp16")
    if cuda_graph:
        cmd.append("--useCudaGraph")
    cmd += [f"--iterations={int(iterations)}", "--avgRuns=50", "--warmUp=500",
            "--noDataTransfers", "--dumpProfile", "--separateProfileRun",
            f"--saveEngine={engine}"]
    cmd += list(extra_args)
    return cmd


def _env(password: Optional[str]) -> Dict[str, str]:
    env = dict(os.environ)
    if password:
        env["SSHPASS"] = password
    return env


def _run(cmd: Sequence[str], *, timeout: float, env: Optional[Dict[str, str]] = None):
    """The default runner: capture both streams as text."""
    return subprocess.run(list(cmd), capture_output=True, text=True, timeout=timeout, env=env)


def _check(res, what: str) -> None:
    if res.returncode != 0:
        tail = (res.stderr or res.stdout or "").strip().splitlines()[-5:]
        raise BenchError(f"{what} failed (exit {res.returncode}): " + " | ".join(tail))


def _call(run: Runner, cmd: Sequence[str], what: str, *, timeout: float, env: Dict[str, str]):
    """Run one ssh/scp command, turning the two ways the local side can fail
    before anything is heard from the robot into :class:`BenchError`: a
    timeout, and a missing ``ssh``/``scp``/``sshpass`` binary (which would
    otherwise surface as the same ``FileNotFoundError`` a missing ONNX file
    raises)."""
    try:
        return run(cmd, timeout=timeout, env=env)
    except subprocess.TimeoutExpired as e:
        raise BenchTimeout(f"{what} timed out after {timeout:g} s") from e
    except FileNotFoundError as e:
        raise BenchError(f"{what}: {cmd[0]!r} is not installed on this machine ({e})") from e


def thor_latency(onnx_path, host: str = "bebop.local", user: str = "bebop",
                 remote_dir: str = REMOTE_DIR, fp16: bool = True, cuda_graph: bool = True,
                 iterations: int = 200, timeout_s: float = 900.0, *, top_n: int = 10,
                 extra_args: Sequence[str] = (), runner: Optional[Runner] = None) -> dict:
    """Copy ``onnx_path`` to the Thor, build and time its TensorRT engine
    with ``trtexec``, and return :func:`parse_trtexec` of the log plus the
    remote paths and the engine size on disk.

    Order of operations, each over its own ssh/scp call:

    1. ``pgrep -af 'record-nav[d]|main.p[y]'`` on the robot. A match means the
       recorder holds the cameras and the GPU; this raises
       :class:`RecorderRunning` and nothing else is sent. Stop
       ``bebop-vision`` from the app (or ``systemctl stop bebop-vision`` with
       the user's go) and run again.
    2. ``mkdir -p remote_dir`` and ``scp`` the ONNX file.
    3. ``trtexec`` (see :func:`trtexec_command`); a non-zero exit or a log
       without ``&&&& PASSED`` raises :class:`BenchError` with the log tail.
    4. ``stat`` the saved engine for its size.

    Nothing here moves the robot: no firmware endpoint, no CAN, no service
    restart. The password comes from ``BEBOP_SSH_PASSWORD`` (``sshpass -e``)
    when set, else ssh key auth in BatchMode. ``runner`` replaces
    ``subprocess.run`` (tests); ``timeout_s`` bounds the scp and the trtexec
    call each (an engine build for a ViT-L can take minutes on the Thor).
    Every local failure is a :class:`BenchError` (:class:`BenchTimeout` for
    a timeout); only a missing ``onnx_path`` raises ``FileNotFoundError``.
    ``remote_dir`` and the ONNX filename must pass :func:`check_remote_path`.
    """
    onnx_path = Path(onnx_path)
    if not onnx_path.is_file():
        raise FileNotFoundError(onnx_path)
    remote_dir = check_remote_path(remote_dir.rstrip("/") or "/", "remote_dir")
    check_remote_path(onnx_path.name, "ONNX filename")
    run = runner or _run
    password = os.environ.get("BEBOP_SSH_PASSWORD") or None
    env = _env(password)
    target = f"{user}@{host}"
    t0 = time.monotonic()

    # 1. The recorder owns the GPU during a session; never bench beside it.
    res = _call(run, ssh_command(target, RECORDER_PGREP, password),
                f"ssh {target} recorder check", timeout=60, env=env)
    if res.returncode == 0 and res.stdout.strip():
        procs = " | ".join(res.stdout.strip().splitlines()[:4])
        raise RecorderRunning(
            f"the recorder is running on {host} ({procs}); it owns the cameras and the GPU. "
            "Stop bebop-vision (with the user's go) before benchmarking; refusing to run trtexec.")
    if res.returncode not in (0, 1):   # 1 = pgrep found nothing
        _check(res, f"ssh {target} recorder check")

    # 2. Stage the ONNX file.
    remote_onnx = f"{remote_dir}/{onnx_path.name}"
    engine = f"{remote_dir}/{onnx_path.stem}.plan"
    res = _call(run, ssh_command(target, f"mkdir -p {shlex.quote(remote_dir)}", password),
                f"mkdir {remote_dir} on {host}", timeout=60, env=env)
    _check(res, f"mkdir {remote_dir} on {host}")
    res = _call(run, scp_command(onnx_path, target, remote_onnx, password),
                f"scp {onnx_path.name} to {host}", timeout=timeout_s, env=env)
    _check(res, f"scp {onnx_path.name} to {host}")

    # 3. Build + time.
    cmd = trtexec_command(remote_onnx, engine, fp16=fp16, cuda_graph=cuda_graph,
                          iterations=iterations, extra_args=extra_args)
    remote_cmd = " ".join(shlex.quote(c) for c in cmd)
    res = _call(run, ssh_command(target, remote_cmd, password),
                f"trtexec on {host}", timeout=timeout_s, env=env)
    text = (res.stdout or "") + ("\n" + res.stderr if res.stderr else "")
    result = parse_trtexec(text, top_n=top_n)
    if res.returncode != 0 or not result.get("passed"):
        tail = "\n".join(text.strip().splitlines()[-30:])
        raise BenchError(f"trtexec on {host} failed (exit {res.returncode}):\n{tail}")

    # 4. The plan on disk (trtexec prints the engine size, but not always).
    res = _call(run, ssh_command(target, f"stat -c %s {shlex.quote(engine)}", password),
                f"stat {engine} on {host}", timeout=60, env=env)
    engine_bytes = int(res.stdout.strip()) if res.returncode == 0 and res.stdout.strip().isdigit() else None

    sidecar = sidecar_path(onnx_path)
    result.update({
        "host": host,
        "onnx": str(onnx_path),
        "remote_onnx": remote_onnx,
        "engine": engine,
        "engine_bytes": engine_bytes,
        "fp16": bool(fp16), "cuda_graph": bool(cuda_graph), "iterations": int(iterations),
        "command": remote_cmd,
        "export": json.loads(sidecar.read_text()) if sidecar.exists() else None,
        "elapsed_s": round(time.monotonic() - t0, 1),
        "log": text,
    })
    return result


# --------------------------------------------------------------------------
# trtexec log parser
# --------------------------------------------------------------------------

# "[10/24/2026-12:00:00] [I] GPU Compute Time: ..." -> level, payload.
_LOG_LINE = re.compile(r"^\[\d\d/\d\d/\d{4}-\d\d:\d\d:\d\d\]\s+\[(\w)\]\s?(.*)$")
_STAT = re.compile(r"(min|max|mean|median|percentile\((\d+)%\))\s*=\s*([-+0-9.eE]+)\s*ms")
_TIMING_KEYS = {
    "Latency": "latency_ms",
    "Enqueue Time": "enqueue_ms",
    "H2D Latency": "h2d_ms",
    "GPU Compute Time": "gpu_compute_ms",
    "D2H Latency": "d2h_ms",
}
_TIMING_LINE = re.compile(r"^(" + "|".join(re.escape(k) for k in _TIMING_KEYS) + r"):\s*(.*)$")
_PROFILE_HEAD = re.compile(r"=== Profile \((\d+) iterations\s*\) ===")
_PROFILE_ROW = re.compile(
    r"^\s*(?P<layer>.+?)\s+(?P<time>[0-9.]+)\s+(?P<avg>[0-9.]+)(?:\s+(?P<median>[0-9.]+))?\s+(?P<pct>[0-9.]+)\s*$")
_FLOAT = r"([-+0-9.eE]+)"


def _strip(line: str) -> Tuple[Optional[str], str]:
    m = _LOG_LINE.match(line.rstrip())
    if m:
        return m.group(1), m.group(2)
    return None, line.rstrip()


def _timing(payload: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, pct, value in _STAT.findall(payload):
        name = f"p{pct}" if pct else key
        out[name] = float(value)
    return out


def _search(pattern: str, text: str, cast=float):
    m = re.search(pattern, text)
    return cast(m.group(1)) if m else None


def _trt_version(text: str, body: str) -> Optional[str]:
    """``"10.16.2"`` from the ``TensorRT version:`` info line (it carries the
    patch level), else decoded from the ``[TensorRT vNNNNNN]`` banner, whose
    layout changed with the major: 10.x packs ``MMmmpp`` (``v101600`` is
    10.16.0), 8.x/9.x pack ``Mmpb`` (``v8601`` is 8.6.0 build 1)."""
    m = re.search(r"TensorRT version:\s*(\d+(?:\.\d+)+)", body)
    if m:
        return m.group(1)
    m = re.search(r"\[TensorRT v(\d+)\]", text)
    if not m:
        return None
    digits = m.group(1)
    if len(digits) >= 6:
        major, minor, patch = digits[:-4], digits[-4:-2], digits[-2:]
    elif len(digits) == 4:
        major, minor, patch = digits[0], digits[1], digits[2]
    else:
        return None
    return f"{int(major)}.{int(minor)}.{int(patch)}"


def parse_trtexec(text: str, top_n: int = 10) -> dict:
    """Pull the numbers out of a ``trtexec`` log (TensorRT 10.x format; the
    8.x profile table without the median column parses too).

    ``gpu_compute_ms`` / ``latency_ms`` / ``enqueue_ms`` / ``h2d_ms`` /
    ``d2h_ms`` are ``{min, max, mean, median, p90, p95, p99}`` from the
    ``=== Performance summary ===`` block; ``median_ms`` / ``mean_ms`` /
    ``p99_ms`` repeat the GPU compute figures at the top level (the ones the
    budget in ``docs/path-model.md`` is written against). ``profile`` holds
    the ``--dumpProfile`` table: the top ``top_n`` layers by total time and
    the layer count. Missing pieces are ``None``, never a guess.
    """
    lines = text.splitlines()
    stripped = [_strip(ln) for ln in lines]
    payloads = [p for _, p in stripped]
    body = "\n".join(payloads)

    out: Dict[str, Any] = {
        "passed": None,
        "trt_version": None,
        "throughput_qps": None,
        "median_ms": None, "mean_ms": None, "p99_ms": None,
        "total_host_walltime_s": None,
        "total_gpu_compute_s": None,
        "engine_size_mib": None,
        "engine_build_s": None,
        "peak_gpu_mib": None, "peak_cpu_mib": None,
        "context_gpu_mib": None,
        "activation_bytes": None, "weights_bytes": None,
        "profile": None,
        "warnings": [],
    }
    for key in _TIMING_KEYS.values():
        out[key] = None

    if "&&&& PASSED" in text:
        out["passed"] = True
    elif "&&&& FAILED" in text:
        out["passed"] = False

    out["trt_version"] = _trt_version(text, body)
    out["throughput_qps"] = _search(r"Throughput:\s*" + _FLOAT + r"\s*qps", body)

    for level, payload in stripped:
        m = _TIMING_LINE.match(payload)
        if m:
            # Under --verbose trtexec also prints "[V] GPU Compute Time: the
            # GPU latency to execute ..." explanations after the summary;
            # a key line without numbers must not erase the parsed block.
            stats = _timing(m.group(2))
            if stats:
                out[_TIMING_KEYS[m.group(1)]] = stats
        if level == "W" and payload.strip():
            out["warnings"].append(payload.strip())

    gpu = out.get("gpu_compute_ms") or {}
    out["median_ms"] = gpu.get("median")
    out["mean_ms"] = gpu.get("mean")
    out["p99_ms"] = gpu.get("p99")

    out["total_host_walltime_s"] = _search(r"Total Host Walltime:\s*" + _FLOAT + r"\s*s", body)
    out["total_gpu_compute_s"] = _search(r"Total GPU Compute Time:\s*" + _FLOAT + r"\s*s", body)
    out["engine_size_mib"] = (_search(r"Created engine with size:\s*" + _FLOAT + r"\s*MiB", body)
                              or _search(r"Loaded engine size:\s*" + _FLOAT + r"\s*MiB", body))
    out["engine_build_s"] = _search(r"Engine built in\s*" + _FLOAT + r"\s*sec", body)
    m = re.search(r"Peak memory usage of TRT CPU/GPU memory allocators:\s*CPU\s*" + _FLOAT
                  + r"\s*MiB,\s*GPU\s*" + _FLOAT + r"\s*MiB", body)
    if m:
        out["peak_cpu_mib"], out["peak_gpu_mib"] = float(m.group(1)), float(m.group(2))
    m = re.search(r"TensorRT-managed allocation in IExecutionContext creation:.*?GPU\s*\+?" + _FLOAT, body)
    if m:
        out["context_gpu_mib"] = float(m.group(1))
    out["activation_bytes"] = _search(r"Total Activation Memory:\s*(\d+)\s*bytes", body, int)
    out["weights_bytes"] = _search(r"Total Weights Memory:\s*(\d+)\s*bytes", body, int)

    out["profile"] = _parse_profile(payloads, top_n)
    return out


def _parse_profile(payloads: List[str], top_n: int) -> Optional[dict]:
    """The ``--dumpProfile`` table: header, one row per layer (names may
    contain spaces, so the row is matched from its trailing numeric columns),
    a ``Total`` row."""
    start = None
    iterations = None
    for i, p in enumerate(payloads):
        m = _PROFILE_HEAD.search(p)
        if m:
            start, iterations = i, int(m.group(1))
            break
    if start is None:
        return None
    rows = []
    total = None
    seen_header = False
    for p in payloads[start + 1:]:
        if not seen_header:
            if "Layer" in p and "Time" in p:
                seen_header = True
            continue
        m = _PROFILE_ROW.match(p)
        if not m:
            if rows:
                break
            continue
        rec = {
            "layer": m.group("layer").strip(),
            "time_ms": float(m.group("time")),
            "avg_ms": float(m.group("avg")),
            "median_ms": float(m.group("median")) if m.group("median") else None,
            "pct": float(m.group("pct")),
        }
        if rec["layer"] == "Total":
            total = rec
            break
        rows.append(rec)
    if not rows:
        return None
    rows.sort(key=lambda r: r["time_ms"], reverse=True)
    return {
        "iterations": iterations,
        "layers": len(rows),
        "total_avg_ms": total["avg_ms"] if total else round(sum(r["avg_ms"] for r in rows), 4),
        "total_median_ms": total["median_ms"] if total else None,
        "top": rows[:top_n],
    }
