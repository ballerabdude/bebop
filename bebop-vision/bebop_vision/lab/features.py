"""Per-session frozen-feature cache keyed by backbone and image geometry.

Frozen-backbone experiments (linear probes, head sweeps, embedding studies)
re-run the same backbone over the same frames many times; this caches the
result once per ``(backbone.id, role, size, crop)`` under
``<cache>/<session>/features/<key>/`` and lets the dataset read it back by
tick time.

Layout of one feature directory::

    pooled.npy   [N, D]           float16 (default) or float32
    t_ns.npy     [N]              int64, the ticks with a frame (missing frames are dropped)
    c3.npy ...   [N, C, h, w]     optional stage maps written as memmaps
    tokens.npy   [N, T, D]        optional ViT tokens
    meta.json    source, identity, producer, backbone, geometry, shapes, counts

``compute`` pulls frames through the Session/Ticks surface
(``ticks.select("color_<role>")[i]["color_<role>"]``); it never decodes
video itself. The geometry's ``crop`` is in recorded-image pixels and is
applied before the ``size`` resize, the same order :class:`types.Mask` uses
to relate a model's output to the recorded frame.

A feature directory is replaced whole: ``compute`` writes into a sibling
``<key>.tmp`` directory and renames it over the old one only after
``meta.json`` is written. An interrupted recompute therefore leaves the
previous complete directory (or nothing), never a half-filled memmap under a
valid ``meta.json``, and no stage file from an earlier run of the same key
can outlive it (the key does not encode the tick selection, so such a file
would otherwise be served against the new ``t_ns.npy``).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from bebop_vision.lab.types import file_identity

log = logging.getLogger(__name__)

#: Bump when the on-disk layout or the computation changes; readers rebuild.
PRODUCER = "features@1"
_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _geometry_norm(geometry: Dict[str, Any]) -> Dict[str, Any]:
    """Canonical geometry: role (str), size (W, H) or None, crop 4-tuple or None."""
    role = str(geometry.get("role", "near"))
    size = geometry.get("size")
    crop = geometry.get("crop")
    size_t = None if size is None else (int(size[0]), int(size[1]))
    crop_t = None if crop is None else tuple(int(v) for v in crop)
    if crop_t is not None and len(crop_t) != 4:
        raise ValueError(f"crop must be (r0, r1, c0, c1), got {crop!r}")
    return {"role": role, "size": size_t, "crop": crop_t}


def _module_device(module) -> Optional[Any]:
    """Device of the module's first parameter or buffer; None when it has none."""
    for t in module.parameters():
        return t.device
    for t in module.buffers():
        return t.device
    return None


def _trim_rows(path: Path, n: int, chunk_bytes: int = 64 << 20) -> None:
    """Shrink a ``.npy`` memmap to its first ``n`` rows by a chunked copy into
    a fresh file (bounded RAM: a c2 map over an 18k-tick drive is ~6 GB, a
    full in-memory copy is not an option) and an atomic replace."""
    src = np.load(path, mmap_mode="r")
    if len(src) == n:
        return
    tmp = path.with_name(path.name + ".trim")
    dst = np.lib.format.open_memmap(tmp, mode="w+", dtype=src.dtype, shape=(n,) + src.shape[1:])
    row_bytes = max(1, int(np.prod(src.shape[1:])) * src.dtype.itemsize)
    step = max(1, chunk_bytes // row_bytes)
    for a in range(0, n, step):
        b = min(a + step, n)
        dst[a:b] = src[a:b]
    dst.flush()
    del dst, src
    os.replace(tmp, path)


class FeatureCache:
    """Compute, store and read back frozen features for a session."""

    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)

    # -- keys and paths ----------------------------------------------------
    def key(self, backbone, geometry: Dict[str, Any]) -> str:
        """Readable, filesystem-safe and collision-resistant: the slugged
        backbone id and geometry, plus 8 hex of a hash of the exact
        ``(id, geometry)`` so two ids that slug alike cannot share a dir."""
        g = _geometry_norm(geometry)
        parts = [_SAFE.sub("_", str(backbone.id)), g["role"]]
        if g["size"] is not None:
            parts.append(f"{g['size'][0]}x{g['size'][1]}")
        else:
            parts.append("native")
        if g["crop"] is not None:
            parts.append("crop" + "-".join(str(v) for v in g["crop"]))
        canon = json.dumps({"id": str(backbone.id), "geometry": g}, sort_keys=True)
        parts.append(hashlib.sha1(canon.encode()).hexdigest()[:8])
        return "__".join(parts)

    def dir(self, session, key: str) -> Path:
        return self.cache_dir / str(session.name) / "features" / key

    # -- compute -----------------------------------------------------------
    def compute(self, session, ticks, backbone, geometry: Dict[str, Any],
                what: Sequence[str] = ("pooled",), device: str = "cuda", batch: int = 16,
                fp16: bool = True) -> Path:
        """Run ``backbone`` over every tick's ``color_<role>`` frame and write
        the feature directory; returns its path.

        ``what`` names the arrays to keep: ``"pooled"`` always makes sense;
        pyramid stages (``"c3"`` ...) or ``"tokens"`` are stored as memmaps,
        which get large (a c3 map at 224x128 is 192x16x28 fp16 = 172 KB per
        tick). ``fp16`` is the storage dtype; on CUDA it also enables autocast
        for the forward pass. On CPU the forward runs in fp32 regardless.
        """
        import torch

        g = _geometry_norm(geometry)
        what = tuple(what)
        if "pooled" not in what:
            what = ("pooled",) + what
        modality = f"color_{g['role']}"
        key = self.key(backbone, g)
        out_dir = self.dir(session, key)
        tmp_dir = out_dir.with_name(out_dir.name + ".tmp")
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)   # a previous run died mid-write
        tmp_dir.mkdir(parents=True)

        if str(device).startswith("cuda") and not torch.cuda.is_available():
            log.warning("features: %s requested but CUDA is unavailable; using cpu", device)
            device = "cpu"
        dev = torch.device(device)
        use_amp = bool(fp16) and dev.type == "cuda"
        store_dtype = np.float16 if fp16 else np.float32

        # The backbone may be shared with a head on another device (a
        # HeadModel on CPU); put it back where it was, mode and device.
        was_training = backbone.training
        home_dev = _module_device(backbone)
        backbone.eval().to(dev)
        sel = ticks.select(modality)
        n_total = len(sel)

        t_out: List[int] = []
        pooled_chunks: List[np.ndarray] = []
        stage_mm: Dict[str, Any] = {}
        stage_shapes: Dict[str, Tuple[int, ...]] = {}
        n_missing = 0
        written = 0
        t_start = time.time()

        def flush(frames: List[np.ndarray], stamps: List[int]) -> None:
            nonlocal written
            if not frames:
                return
            x = torch.cat([backbone.preprocess(f, size=g["size"]) for f in frames], 0).to(dev)
            with torch.inference_mode():
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    feats = backbone(x)
                    pooled = backbone.pooled(feats)
            pooled_chunks.append(pooled.float().cpu().numpy().astype(store_dtype))
            for name in what:
                if name == "pooled":
                    continue
                if name not in feats:
                    raise KeyError(f"backbone {backbone.id} has no feature {name!r}; "
                                   f"available: {sorted(k for k in feats if hasattr(feats[k], 'shape'))}")
                arr = feats[name].float().cpu().numpy().astype(store_dtype)
                if name not in stage_mm:
                    # The per-tick shape is only known after the first forward;
                    # size the memmap for every tick that still may come and
                    # trim it once the count is final.
                    shape = (n_total,) + tuple(arr.shape[1:])
                    stage_mm[name] = np.lib.format.open_memmap(tmp_dir / f"{name}.npy", mode="w+",
                                                               dtype=store_dtype, shape=shape)
                    stage_shapes[name] = tuple(arr.shape[1:])
                stage_mm[name][written:written + len(arr)] = arr
            t_out.extend(int(t) for t in stamps)
            written += len(frames)

        frames: List[np.ndarray] = []
        stamps: List[int] = []
        try:
            try:
                for i in range(n_total):
                    sample = sel[i]
                    frame = sample[modality]
                    if frame is None:
                        n_missing += 1
                        continue
                    if g["crop"] is not None:
                        r0, r1, c0, c1 = g["crop"]
                        frame = frame[r0:r1, c0:c1]
                    frames.append(frame)
                    stamps.append(int(sample.t_ns))
                    if len(frames) >= batch:
                        flush(frames, stamps)
                        frames, stamps = [], []
                flush(frames, stamps)
            finally:
                if was_training:
                    backbone.train()
                if home_dev is not None and home_dev != dev:
                    backbone.to(home_dev)

            pooled_all = (np.concatenate(pooled_chunks, 0) if pooled_chunks
                          else np.zeros((0, backbone.embed_dim), store_dtype))
            np.save(tmp_dir / "pooled.npy", pooled_all)
            np.save(tmp_dir / "t_ns.npy", np.asarray(t_out, dtype=np.int64))
            for name in list(stage_mm):
                # Pop so the writable mapping is really released before the
                # file is trimmed or renamed (a loop variable's del is not).
                mm = stage_mm.pop(name)
                mm.flush()
                del mm
                if written < n_total:
                    _trim_rows(tmp_dir / f"{name}.npy", written)
        except BaseException:
            stage_mm.clear()
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise

        src = Path(getattr(session, "path", ""))
        meta = {
            "source": str(src),
            "identity": file_identity(src) if src.is_file() else None,
            "producer": PRODUCER,
            "session": str(session.name),
            "backbone": str(backbone.id),
            "kind": backbone.kind,
            "embed_dim": int(backbone.embed_dim),
            "mean": list(backbone.mean), "std": list(backbone.std),
            "geometry": {"role": g["role"], "size": list(g["size"]) if g["size"] else None,
                         "crop": list(g["crop"]) if g["crop"] else None},
            "what": list(what),
            "dtype": np.dtype(store_dtype).name,
            "device": str(dev),
            "n": int(written), "n_ticks": int(n_total), "n_missing": int(n_missing),
            "shapes": {"pooled": [int(written), int(backbone.embed_dim)],
                       **{k: [int(written)] + list(v) for k, v in stage_shapes.items()}},
            "elapsed_s": round(time.time() - t_start, 3),
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        (tmp_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        if out_dir.exists():
            shutil.rmtree(out_dir)
        os.replace(tmp_dir, out_dir)
        return out_dir

    # -- read back ---------------------------------------------------------
    def meta(self, session, key: str) -> Optional[dict]:
        p = self.dir(session, key) / "meta.json"
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text())
        except (OSError, ValueError):
            return None

    def has(self, session, key: str) -> bool:
        """True when a complete, current feature dir exists: same producer
        version, same MCAP identity (when the MCAP is on this machine) and
        the ``pooled.npy`` / ``t_ns.npy`` arrays present."""
        meta = self.meta(session, key)
        if meta is None or meta.get("producer") != PRODUCER:
            return False
        d = self.dir(session, key)
        if not (d / "pooled.npy").exists() or not (d / "t_ns.npy").exists():
            return False
        src = Path(getattr(session, "path", ""))
        if src.is_file() and meta.get("identity") not in (None, file_identity(src)):
            return False
        return True

    def load(self, session, key: str, what: str = "pooled") -> Tuple[np.ndarray, np.ndarray]:
        """``(t_ns [N], array)`` for ``what``; stage maps come back as
        read-only memmaps, ``pooled`` fully in memory. Raises when the cache
        is absent or stale (see :meth:`has`)."""
        if not self.has(session, key):
            raise FileNotFoundError(f"no current feature cache {key} for {session.name} "
                                    f"under {self.cache_dir}")
        d = self.dir(session, key)
        t_ns = np.load(d / "t_ns.npy")
        p = d / f"{what}.npy"
        computed = (self.meta(session, key) or {}).get("what", [])
        if what not in computed or not p.exists():
            # meta['what'] is the authority: a file the last compute did not
            # write is a leftover, whatever its row count.
            raise KeyError(f"{key}: {what!r} was not computed (what={computed})")
        arr = np.load(p) if what == "pooled" else np.load(p, mmap_mode="r")
        if len(arr) != len(t_ns):
            raise RuntimeError(f"{key}: {what}.npy has {len(arr)} rows for {len(t_ns)} ticks")
        return t_ns, arr

    def list(self, session) -> List[dict]:
        """Every feature dir of the session with its meta (for ``lab features``)."""
        root = self.cache_dir / str(session.name) / "features"
        out = []
        if root.is_dir():
            for d in sorted(root.iterdir()):
                m = self.meta(session, d.name)
                if m is not None:
                    out.append(dict(m, key=d.name, current=self.has(session, d.name)))
        return out
