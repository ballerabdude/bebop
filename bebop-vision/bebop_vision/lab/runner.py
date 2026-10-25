"""Run a Model over one session and keep its outputs per tick.

The runner is the one place that walks a model over recorded ticks, so the
viewer, the metrics and the bench all read the same ``Run``: outputs stacked
over ticks (one array per output name), the tick stamps, the InputSpec the
model asked for and the wall time of every step. A Run saves to a directory
(``outputs.npz`` + ``run.json``) and loads back into the same object, so a
model that took an hour on the GPU is scored and shown as often as needed
without running again.

Only ``bebop_vision.lab.types`` is imported from the package; the Ticks and
Session objects are duck-typed against the contract (``select``, ``__len__``,
``__getitem__`` -> Sample, ``t_ns``, ``session.name`` / ``session.path``).
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from bebop_vision.lab import types
from bebop_vision.lab.types import (
    Classes,
    Embedding,
    Grid,
    InputSpec,
    Mask,
    Model,
    Path2D,
    Scalars,
)

#: Bumped when the on-disk layout of a run directory changes.
PRODUCER = "runner@1"

#: Samples materialised and handed to ``batch_step`` per call.
DEFAULT_BATCH = 16

_KIND_NAMES = {Path2D: "Path2D", Mask: "Mask", Embedding: "Embedding",
               Classes: "Classes", Grid: "Grid", Scalars: "Scalars"}
_KIND_BY_NAME = {v: k for k, v in _KIND_NAMES.items()}


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------

class Run:
    """A model's outputs over the ticks of one session.

    ``outputs[name]`` is a list with one entry per tick (an output object,
    or ``None`` when the model produced nothing for that tick); ``t_ns`` the
    tick stamps; ``step_ms`` / ``sample_ms`` the per-tick model time and the
    time spent materialising the Sample (decode, alignment); ``meta`` the
    run.json payload (model, spec, session identity, timing summary, ...).
    ``spec`` is the model's InputSpec as an object.
    """

    def __init__(self, t_ns: np.ndarray, outputs: Dict[str, List[Any]], meta: Dict[str, Any],
                 step_ms: Optional[np.ndarray] = None, sample_ms: Optional[np.ndarray] = None,
                 path: Optional[Path] = None):
        self.t_ns = np.asarray(t_ns, dtype=np.int64)
        self.outputs = outputs
        self.meta = meta
        n = len(self.t_ns)
        self.step_ms = (np.asarray(step_ms, dtype=np.float64) if step_ms is not None
                        else np.full(n, np.nan))
        self.sample_ms = (np.asarray(sample_ms, dtype=np.float64) if sample_ms is not None
                          else np.full(n, np.nan))
        self.path = Path(path) if path is not None else None
        for name, items in outputs.items():
            if len(items) != n:
                raise ValueError(f"output {name!r}: {len(items)} entries for {n} ticks")

    # -- identity ------------------------------------------------------------
    def __len__(self) -> int:
        return int(len(self.t_ns))

    def __repr__(self) -> str:
        return (f"Run({self.name!r}, session={self.session_name!r}, ticks={len(self)}, "
                f"outputs={list(self.outputs)})")

    @property
    def name(self) -> str:
        return str(self.meta.get("model", ""))

    @property
    def session_name(self) -> str:
        s = self.meta.get("session", {})
        return str(s.get("name", "")) if isinstance(s, Mapping) else str(s)

    @property
    def spec(self) -> Optional[InputSpec]:
        return _spec_from_dict(self.meta.get("spec"))

    # -- outputs -------------------------------------------------------------
    def kinds(self) -> Dict[str, Optional[type]]:
        """Output name -> kind class, from the first non-None entry (or the
        saved meta when every entry is None)."""
        out: Dict[str, Optional[type]] = {}
        for name, items in self.outputs.items():
            kind = next((type(x) for x in items if x is not None), None)
            if kind is None:
                kind = _KIND_BY_NAME.get(self.meta.get("outputs", {}).get(name, {}).get("kind"))
            out[name] = kind
        return out

    def names_of(self, kind: type) -> List[str]:
        return [n for n, k in self.kinds().items() if k is kind]

    def valid(self, name: str) -> np.ndarray:
        """``bool[T]``: ticks where the model produced output ``name``."""
        return np.array([x is not None for x in self.outputs[name]], dtype=bool)

    def stacked(self, name: str) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Output ``name`` stacked over ticks: ``(arrays, meta)`` exactly as
        saved in ``outputs.npz`` (minus the ``<name>.`` prefix) and run.json."""
        items = self.outputs[name]
        kind = self.kinds()[name]
        if kind is None:
            return {}, {"kind": None}
        return _pack(kind, items, self.valid(name))

    def paths(self, name: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The Path2D output as ``(rel_t [N], pts [T, K, N, 2], weights [T, K])``;
        NaN rows for ticks without output. ``name`` defaults to the single
        Path2D output."""
        name = self._pick(name, Path2D)
        arrays, _ = self.stacked(name)
        return arrays["rel_t"], arrays["pts"], arrays["weights"]

    def _pick(self, name: Optional[str], kind: type) -> str:
        if name is not None:
            if name not in self.outputs:
                raise KeyError(f"no output {name!r}; have {list(self.outputs)}")
            return name
        names = self.names_of(kind)
        if len(names) != 1:
            raise KeyError(f"need exactly one {_KIND_NAMES.get(kind, kind.__name__)} output "
                           f"to pick by default, have {names}")
        return names[0]

    # -- timing --------------------------------------------------------------
    def latency_ms(self) -> Dict[str, float]:
        """Model step time per tick: p50 / p90 / p99 / mean / max over the
        ticks that have a measurement (batched models spread the batch time
        evenly over its ticks, see :func:`run`)."""
        v = self.step_ms[np.isfinite(self.step_ms)]
        if v.size == 0:
            return {"p50": float("nan"), "p90": float("nan"), "p99": float("nan"),
                    "mean": float("nan"), "max": float("nan"), "n": 0}
        return {"p50": float(np.percentile(v, 50)), "p90": float(np.percentile(v, 90)),
                "p99": float(np.percentile(v, 99)), "mean": float(v.mean()),
                "max": float(v.max()), "n": int(v.size)}

    # -- persistence ---------------------------------------------------------
    def save(self, dir: Path) -> Path:
        """Write ``outputs.npz`` (every output stacked, ``<name>.<field>``
        keys plus ``<name>.valid``), ``run.json`` and the lab ``meta.json``."""
        d = Path(dir)
        d.mkdir(parents=True, exist_ok=True)
        arrays: Dict[str, np.ndarray] = {"t_ns": self.t_ns, "step_ms": self.step_ms,
                                         "sample_ms": self.sample_ms}
        out_meta: Dict[str, Any] = {}
        for name, items in self.outputs.items():
            kind = self.kinds()[name]
            valid = self.valid(name)
            arrays[f"{name}.valid"] = valid
            if kind is None:
                out_meta[name] = {"kind": None}
                continue
            packed, m = _pack(kind, items, valid)
            for k, a in packed.items():
                arrays[f"{name}.{k}"] = a
            out_meta[name] = m
        np.savez(d / "outputs.npz", **arrays)

        meta = dict(self.meta)
        meta["outputs"] = out_meta
        meta["n_ticks"] = len(self)
        meta["t_ns"] = [int(t) for t in self.t_ns]
        meta["timing"] = {"step_ms": _nan_to_none(self.step_ms),
                          "sample_ms": _nan_to_none(self.sample_ms),
                          "latency": self.latency_ms()}
        meta["producer"] = PRODUCER
        (d / "run.json").write_text(json.dumps(meta, indent=2, default=_json_default))
        sess = meta.get("session", {}) if isinstance(meta.get("session"), Mapping) else {}
        (d / "meta.json").write_text(json.dumps({
            "source": sess.get("path"), "identity": sess.get("identity"),
            "producer": PRODUCER, "model": meta.get("model"), "n_ticks": len(self),
        }, indent=2, default=_json_default))
        self.path = d
        return d

    @classmethod
    def load(cls, dir: Path) -> "Run":
        d = Path(dir)
        meta = json.loads((d / "run.json").read_text())
        if meta.get("producer") != PRODUCER:
            raise ValueError(f"{d}: written by {meta.get('producer')!r}, expected {PRODUCER!r}")
        with np.load(d / "outputs.npz", allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        t_ns = arrays.pop("t_ns")
        step_ms = arrays.pop("step_ms", None)
        sample_ms = arrays.pop("sample_ms", None)
        outputs: Dict[str, List[Any]] = {}
        for name, m in meta.get("outputs", {}).items():
            prefix = f"{name}."
            fields = {k[len(prefix):]: a for k, a in arrays.items() if k.startswith(prefix)}
            valid = fields.pop("valid", np.ones(len(t_ns), dtype=bool))
            kind = _KIND_BY_NAME.get(m.get("kind"))
            if kind is None:
                outputs[name] = [None] * len(t_ns)
            else:
                outputs[name] = _unpack(kind, fields, m, valid)
        return cls(t_ns, outputs, meta, step_ms=step_ms, sample_ms=sample_ms, path=d)


# --------------------------------------------------------------------------
# run()
# --------------------------------------------------------------------------

def run(session, model: Model, ticks=None, out_dir: Optional[Path] = None,
        device: Optional[str] = None, progress: bool = False, *,
        batch: int = DEFAULT_BATCH, save: bool = True) -> Run:
    """Run ``model`` over ``ticks`` (default: every raw tick of ``session``).

    The ticks are narrowed to the model's InputSpec, ``model.prepare`` runs
    once with the session calibration (and ``{"device": device}`` when a
    device is given), then :func:`types.batch_step` consumes the samples in
    chunks of ``batch``. Per-tick timing: a model without ``batch_step`` is
    stepped one sample at a time so each tick gets its own wall time; a
    batched model gets the chunk time divided evenly, which is the right
    per-tick figure for throughput but hides the batch's first-call cost.

    ``save=True`` writes the Run to ``out_dir`` or, when that is None, to
    ``<cache>/<session>/runs/<model.name>-<stamp>/`` (``session.cache_dir``,
    else ``BEBOP_LAB_CACHE`` or ``datasets/lab/<session>``).
    """
    if ticks is None:
        ticks = session.ticks()
    spec: InputSpec = model.spec
    ticks = ticks.select(*spec.modalities, history_s=dict(spec.history_s), tol_ns=spec.tol_ns)
    n = len(ticks)

    calib = getattr(session, "calib", None)
    cfg = {"device": device} if device is not None else None
    model.prepare(calib, cfg)

    t_ns = np.asarray(ticks.t_ns, dtype=np.int64)
    outputs: Dict[str, List[Any]] = {name: [None] * n for name in model.outputs}
    step_ms = np.full(n, np.nan)
    sample_ms = np.full(n, np.nan)
    # A model without batch_step is timed per tick (chunks of one); a batched
    # one is timed per chunk and the time spread over its ticks.
    chunk = batch if getattr(model, "batch_step", None) is not None else 1
    chunk = max(1, int(chunk))

    started = _dt.datetime.now(_dt.timezone.utc)
    done = 0
    for lo in range(0, n, chunk):
        idx = list(range(lo, min(n, lo + chunk)))
        samples = []
        for i in idx:
            t0 = time.perf_counter()
            samples.append(ticks[i])
            sample_ms[i] = (time.perf_counter() - t0) * 1e3
        t0 = time.perf_counter()
        results = types.batch_step(model, samples)
        dt_ms = (time.perf_counter() - t0) * 1e3
        if len(results) != len(idx):
            raise RuntimeError(f"{model.name}: batch_step returned {len(results)} results "
                               f"for {len(idx)} samples")
        for i, res in zip(idx, results):
            step_ms[i] = dt_ms / len(idx)
            if res is None:
                continue
            for name, value in res.items():
                if name not in outputs:
                    # an undeclared output is kept rather than dropped; the
                    # declaration is advisory for the viewer and the trainer
                    outputs[name] = [None] * n
                outputs[name][i] = value
        done += len(idx)
        if progress:
            _print_progress(model.name, done, n, step_ms[:done])
    finished = _dt.datetime.now(_dt.timezone.utc)

    meta = {
        "model": str(model.name),
        "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
        "backbone": _backbone_id(model),
        "mode": getattr(model, "mode", None),
        "spec": _spec_to_dict(spec),
        "declared_outputs": {k: _KIND_NAMES.get(v, getattr(v, "__name__", str(v)))
                             for k, v in model.outputs.items()},
        "session": _session_identity(session),
        "device": device,
        "batch": int(chunk),
        "started": started.isoformat(timespec="seconds"),
        "finished": finished.isoformat(timespec="seconds"),
        "git": types.git_hash(),
    }
    extra = getattr(model, "meta", None)
    try:
        # a model's own metadata must never cost a finished run: it is
        # dropped when it cannot be produced or written as JSON
        if callable(extra):
            extra = extra()
        if isinstance(extra, Mapping):
            meta["model_meta"] = json.loads(json.dumps(dict(extra), default=_json_default))
    except Exception:
        pass

    r = Run(t_ns, outputs, meta, step_ms=step_ms, sample_ms=sample_ms)
    if save:
        r.save(Path(out_dir) if out_dir is not None else default_run_dir(session, model.name, started))
    return r


def default_run_dir(session, model_name: str, stamp: Optional[_dt.datetime] = None) -> Path:
    """``<cache>/<session>/runs/<model>-<YYYYmmdd_HHMMSS>/`` (a suffix is
    added when that directory already exists)."""
    stamp = stamp or _dt.datetime.now(_dt.timezone.utc)
    base = _session_cache_dir(session) / "runs"
    name = f"{model_name}-{stamp.strftime('%Y%m%d_%H%M%S')}"
    d = base / name
    k = 1
    while d.exists():
        d = base / f"{name}-{k}"
        k += 1
    return d


def list_runs(session) -> List[Path]:
    """Run directories under the session's cache, oldest first."""
    base = _session_cache_dir(session) / "runs"
    if not base.is_dir():
        return []
    return sorted(p for p in base.iterdir() if (p / "run.json").is_file())


# --------------------------------------------------------------------------
# Packing per output kind
# --------------------------------------------------------------------------

def _pack(kind: type, items: Sequence[Any], valid: np.ndarray) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Stack per-tick outputs of one kind into arrays over ticks; invalid
    ticks get NaN (float) or 0 (int) rows. Shapes must agree across ticks:
    one model, one output geometry per run."""
    first = next((x for x in items if x is not None), None)
    if first is None:
        return {}, {"kind": _KIND_NAMES[kind]}
    n = len(items)
    if kind is Path2D:
        rel_t = np.asarray(first.rel_t, dtype=np.float32)
        K, N = first.pts.shape[:2]
        pts = np.full((n, K, N, 2), np.nan, np.float32)
        weights = np.full((n, K), np.nan, np.float32)
        has_psi = first.psi is not None
        has_v = first.v is not None
        psi = np.full((n, K, N), np.nan, np.float32) if has_psi else None
        vel = np.full((n, K, N), np.nan, np.float32) if has_v else None
        for i, p in enumerate(items):
            if p is None:
                continue
            if p.pts.shape != (K, N, 2):
                raise ValueError(f"Path2D shape {p.pts.shape} at tick {i}, expected {(K, N, 2)}")
            if p.rel_t.shape != rel_t.shape or not np.allclose(p.rel_t, rel_t, atol=1e-6):
                raise ValueError(f"Path2D rel_t grid differs at tick {i}")
            pts[i] = p.pts
            weights[i] = p.weights
            if has_psi and p.psi is not None:
                psi[i] = p.psi
            if has_v and p.v is not None:
                vel[i] = p.v
        arrays = {"rel_t": rel_t, "pts": pts, "weights": weights}
        if has_psi:
            arrays["psi"] = psi
        if has_v:
            arrays["v"] = vel
        return arrays, {"kind": "Path2D", "K": int(K), "N": int(N)}

    if kind is Mask:
        v0 = np.asarray(first.values)
        is_int = np.issubdtype(v0.dtype, np.integer) or v0.dtype == np.bool_
        # class ids fit uint8 (the contract's int masks); probabilities /
        # logits go to float16, which halves the file for no scoring loss
        dtype = np.uint8 if is_int else np.float16
        values = np.zeros((n,) + v0.shape, dtype)
        if not is_int:
            values[:] = np.nan
        for i, m in enumerate(items):
            if m is None:
                continue
            a = np.asarray(m.values)
            if a.shape != v0.shape:
                raise ValueError(f"Mask shape {a.shape} at tick {i}, expected {v0.shape}")
            if is_int and a.size and (int(a.max()) > 255 or int(a.min()) < 0):
                # a negative id (the usual -1 ignore index) would wrap to 255
                # silently and be scored as a class
                raise ValueError(f"Mask class ids {int(a.min())}..{int(a.max())} at tick {i} "
                                 f"do not fit uint8 (0..255)")
            values[i] = a
        meta = {"kind": "Mask", "camera": first.camera, "classes": list(first.classes),
                "stride": int(first.stride),
                "crop": list(first.crop) if first.crop is not None else None,
                "dtype": str(np.dtype(dtype))}
        return {"values": values}, meta

    if kind is Embedding:
        D = int(np.asarray(first.vector).shape[0])
        vec = np.full((n, D), np.nan, np.float32)
        for i, e in enumerate(items):
            if e is not None:
                vec[i] = np.asarray(e.vector, dtype=np.float32)
        return {"vector": vec}, {"kind": "Embedding", "source": first.source, "D": D}

    if kind is Classes:
        C = int(np.asarray(first.logits).shape[0])
        logits = np.full((n, C), np.nan, np.float32)
        for i, c in enumerate(items):
            if c is not None:
                logits[i] = np.asarray(c.logits, dtype=np.float32)
        return {"logits": logits}, {"kind": "Classes", "names": list(first.names), "C": C}

    if kind is Grid:
        shape = tuple(np.asarray(first.values).shape)
        values = np.full((n,) + shape, np.nan, np.float32)
        for i, g in enumerate(items):
            if g is not None:
                values[i] = np.asarray(g.values, dtype=np.float32)
        return {"values": values}, {"kind": "Grid", "cell_m": float(first.cell_m),
                                    "origin_xy": [float(first.origin_xy[0]), float(first.origin_xy[1])],
                                    "channels": list(first.channels)}

    if kind is Scalars:
        keys: List[str] = []
        for s in items:
            if s is not None:
                for k in s.values:
                    if k not in keys:
                        keys.append(k)
        dtype = np.dtype([(k, "f8") for k in keys])
        values = np.full(n, np.nan, dtype=dtype) if keys else np.zeros(n, dtype=dtype)
        for i, s in enumerate(items):
            if s is None:
                continue
            for k, v in s.values.items():
                values[k][i] = float(v)
        return {"values": values}, {"kind": "Scalars", "keys": keys}

    raise TypeError(f"cannot stack outputs of type {kind.__name__}; "
                    f"supported: {[k.__name__ for k in _KIND_NAMES]}")


def _unpack(kind: type, arrays: Dict[str, np.ndarray], meta: Dict[str, Any],
            valid: np.ndarray) -> List[Any]:
    """Inverse of :func:`_pack`: per-tick output objects (None where invalid)."""
    n = int(len(valid))
    out: List[Any] = [None] * n
    if not arrays:
        return out
    if kind is Path2D:
        rel_t, pts, weights = arrays["rel_t"], arrays["pts"], arrays["weights"]
        psi, vel = arrays.get("psi"), arrays.get("v")
        for i in range(n):
            if valid[i]:
                out[i] = Path2D(rel_t.copy(), pts[i], weights[i],
                                psi=None if psi is None else psi[i],
                                v=None if vel is None else vel[i])
    elif kind is Mask:
        values = arrays["values"]
        crop = meta.get("crop")
        for i in range(n):
            if valid[i]:
                out[i] = Mask(camera=meta.get("camera", ""), values=values[i],
                              classes=tuple(meta.get("classes", ())), stride=int(meta.get("stride", 1)),
                              crop=tuple(crop) if crop is not None else None)
    elif kind is Embedding:
        vec = arrays["vector"]
        for i in range(n):
            if valid[i]:
                out[i] = Embedding(vector=vec[i], source=meta.get("source", ""))
    elif kind is Classes:
        logits = arrays["logits"]
        for i in range(n):
            if valid[i]:
                out[i] = Classes(logits=logits[i], names=tuple(meta.get("names", ())))
    elif kind is Grid:
        values = arrays["values"]
        for i in range(n):
            if valid[i]:
                out[i] = Grid(values=values[i], cell_m=float(meta["cell_m"]),
                              origin_xy=tuple(meta["origin_xy"]), channels=tuple(meta.get("channels", ())))
    elif kind is Scalars:
        values = arrays["values"]
        keys = list(values.dtype.names or ())
        for i in range(n):
            if valid[i]:
                out[i] = Scalars({k: float(values[k][i]) for k in keys})
    else:  # pragma: no cover - _pack refuses these first
        raise TypeError(f"cannot load outputs of type {kind.__name__}")
    return out


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _spec_to_dict(spec: InputSpec) -> Dict[str, Any]:
    return {"modalities": list(spec.modalities),
            "history_s": {k: float(v) for k, v in spec.history_s.items()},
            "tol_ns": int(spec.tol_ns)}


def _spec_from_dict(d: Optional[Mapping[str, Any]]) -> Optional[InputSpec]:
    if not d:
        return None
    return InputSpec(modalities=tuple(d.get("modalities", ())),
                     history_s=dict(d.get("history_s", {})),
                     tol_ns=int(d.get("tol_ns", 50_000_000)))


def _backbone_id(model) -> Optional[str]:
    """The backbone id a model carries, if any: ``model.backbone_id`` or
    ``model.backbone.id`` (HeadModel / EmbeddingModel in the contract)."""
    bid = getattr(model, "backbone_id", None)
    if bid is None:
        bb = getattr(model, "backbone", None)
        bid = getattr(bb, "id", None) if bb is not None else None
    return None if bid is None else str(bid)


def _session_identity(session) -> Dict[str, Any]:
    name = getattr(session, "name", None)
    path = getattr(session, "path", None)
    ident = None
    if path is not None:
        try:
            if Path(path).is_file():
                ident = types.file_identity(Path(path))
        except OSError:
            ident = None
    return {"name": str(name) if name is not None else None,
            "path": str(path) if path is not None else None,
            "identity": ident}


def _session_cache_dir(session) -> Path:
    """The session's cache directory: ``session.cache_dir`` when the Session
    carries one (the real Session does; it is the per-session directory,
    explicit or ``<root>/<name>``), else the lab default root / ``<name>``.
    ``session.index()`` is deliberately not called: it builds the whole
    MCAP index as a side effect, which listing runs must not trigger."""
    cache_dir = getattr(session, "cache_dir", None)
    if cache_dir is not None:
        return Path(cache_dir)
    name = str(getattr(session, "name", None) or Path(str(getattr(session, "path", "session"))).stem)
    root = os.environ.get("BEBOP_LAB_CACHE")
    if root:
        return Path(root) / name
    return Path(__file__).resolve().parents[2] / "datasets" / "lab" / name


def _print_progress(name: str, done: int, n: int, step_ms: np.ndarray) -> None:
    v = step_ms[np.isfinite(step_ms)]
    med = float(np.median(v)) if v.size else float("nan")
    end = "\n" if done >= n else "\r"
    sys.stderr.write(f"[lab run] {name}: {done}/{n} ticks, step p50 {med:.1f} ms{end}")
    sys.stderr.flush()


def _nan_to_none(a: np.ndarray) -> list:
    return [None if not np.isfinite(x) else float(x) for x in np.asarray(a, dtype=np.float64)]


def _json_default(o):
    if isinstance(o, np.bool_):      # not an np.integer subclass
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, (set, tuple)):
        return list(o)
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


__all__ = ["Run", "run", "default_run_dir", "list_runs", "PRODUCER", "DEFAULT_BATCH"]
