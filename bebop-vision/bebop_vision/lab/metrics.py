"""Scores for runs: path ADE / FDE against hindsight labels, mask IoU, and a
leaderboard that lines runs up side by side.

Conventions
-----------
* A path is scored against the label's ``rel_t`` grid: the prediction's
  best hypothesis is linearly interpolated onto the label times, so two
  models with different output grids (and the CV baseline) are measured on
  the same points. A body-frame path starts at the robot, so ``(0, (0, 0))``
  is prepended to the prediction before interpolating; a label time beyond
  the prediction's last point is not extrapolated and leaves that horizon
  unscored (NaN) for the tick.
* ``ade@h`` is the mean point error over label points with ``0 < rel_t <= h``;
  ``fde@h`` the error at the label point nearest ``h`` (within half a grid
  step). A label that does not reach ``h`` leaves both NaN for that tick, so
  every horizon is reported over the ticks that actually have it, with its
  own ``n``.
* A truncated label is scored only up to its truncation. ``labels.py``
  keeps a label whose window holds a reverse valid, cuts it at the first
  backward step (``truncated_at_s``) and holds the stop pose over the rest
  of the ``rel_t`` grid with ``v = 0``: that tail is stop supervision for
  training, not where the robot went, so a path error against it would
  measure nothing. The label record's ``truncated_at_s`` (top level, or
  under ``"flags"``) drops the points past it before scoring; a label with
  no real point left is treated as missing for that tick.
* Statistics are p50 / mean / p90 over those ticks (per metric, per
  horizon); p50 is the headline, mean and p90 show the tail.
* Mask IoU is dataset-level per class: summed intersection over summed
  union across ticks (the usual semantic segmentation mIoU), not a mean of
  per-image IoUs, which would weight empty frames oddly. A ``Mask.crop``
  is honoured: prediction and reference are compared over the recorded
  image region both cover (a mask without a crop covers the whole image at
  its stride), at the reference's resolution; pixels only one side covers
  are not scored. Without crops on either side a prediction of another
  shape is nearest-resampled onto the reference.

Only ``bebop_vision.lab.types`` is imported from the package; Run and
LabelSet objects are duck-typed (Run: ``t_ns``, ``outputs``; LabelSet:
``t_ns``, ``get(t_ns) -> dict | Path2D | None``).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import polars as pl

from bebop_vision.lab.types import Mask, Path2D

DEFAULT_HORIZONS: Tuple[float, ...] = (1.0, 2.0, 3.0)
_STATS = ("p50", "mean", "p90")


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

def metric_name(metric: str, horizon: float) -> str:
    """``"fde@3.0"`` style column names (one decimal, so horizons sort and
    read the same everywhere)."""
    return f"{metric}@{float(horizon):.1f}"


def parse_metric(name: str) -> Tuple[str, float]:
    """Inverse of :func:`metric_name`: ``"fde@3.0" -> ("fde", 3.0)``."""
    if "@" not in name:
        raise ValueError(f"metric {name!r} must look like 'fde@3.0'")
    m, h = name.split("@", 1)
    m = m.strip().lower()
    if m not in ("ade", "fde"):
        raise ValueError(f"unknown path metric {m!r} (ade | fde)")
    return m, float(h)


def path_errors(pred: Path2D, label: Path2D, horizons: Sequence[float] = DEFAULT_HORIZONS) -> Dict[str, float]:
    """ADE / FDE of one prediction against one label on the label's grid.

    Returns ``{"ade@1.0": ..., "fde@1.0": ..., ...}`` with NaN where the
    label does not reach the horizon or the prediction does not reach the
    label's points there.
    """
    tl = np.asarray(label.rel_t, dtype=np.float64)
    pl_pts = np.asarray(label.best, dtype=np.float64)
    tp = np.asarray(pred.rel_t, dtype=np.float64)
    pp = np.asarray(pred.best, dtype=np.float64)
    out: Dict[str, float] = {}
    if tl.size == 0 or tp.size == 0:
        for h in horizons:
            out[metric_name("ade", h)] = math.nan
            out[metric_name("fde", h)] = math.nan
        return out

    # the body-frame path starts at the robot: anchor the prediction at the
    # origin so label points before its first sample are interpolated, not
    # clamped to the first predicted point
    if tp[0] > 1e-9:
        tp = np.r_[0.0, tp]
        pp = np.vstack([np.zeros((1, 2)), pp])
    order = np.argsort(tp, kind="stable")
    tp, pp = tp[order], pp[order]
    x = np.interp(tl, tp, pp[:, 0])
    y = np.interp(tl, tp, pp[:, 1])
    err = np.hypot(x - pl_pts[:, 0], y - pl_pts[:, 1])
    err[tl > tp[-1] + 1e-6] = np.nan        # prediction does not reach
    step = float(np.median(np.diff(tl))) if tl.size > 1 else float(tl[0])
    half = 0.5 * step + 1e-6

    for h in horizons:
        h = float(h)
        sel = (tl > 1e-9) & (tl <= h + 1e-6)
        j = int(np.argmin(np.abs(tl - h)))
        reaches = abs(float(tl[j]) - h) <= half
        if not reaches or not np.any(sel):
            ade = fde = math.nan
        else:
            e = err[sel]
            ade = math.nan if np.any(np.isnan(e)) else float(e.mean())
            fde = float(err[j]) if np.isfinite(err[j]) else math.nan
        out[metric_name("ade", h)] = ade
        out[metric_name("fde", h)] = fde
    return out


def label_motion_m(label: Path2D) -> float:
    """How far the labelled path gets from the robot (metres): the largest
    distance of any of its points from the origin."""
    pts = np.asarray(label.best, dtype=np.float64)
    return float(np.hypot(pts[:, 0], pts[:, 1]).max()) if pts.size else 0.0


def per_tick_errors(run_or_paths, labels, horizons: Sequence[float] = DEFAULT_HORIZONS,
                    name: Optional[str] = None, min_motion_m: float = 0.0) -> pl.DataFrame:
    """One row per scored tick: ``t_ns`` plus ``ade@h`` / ``fde@h`` columns
    (null where the tick lacks that horizon). Ticks without a valid label
    or without a prediction are left out: they are not "valid ticks".

    ``min_motion_m`` also leaves out ticks whose labelled path never gets
    that far from the robot. Standing still is most of a drive (73 % of the
    10 Hz ticks of navd_session_20261024_043734), and every model that
    predicts "stay" scores zero there, so an ungated mean mostly measures
    how long the robot stood. The row carries ``motion_m`` either way.
    """
    preds = _iter_paths(run_or_paths, name)
    cols = [metric_name(m, h) for h in horizons for m in ("ade", "fde")]
    rows: List[Dict[str, Any]] = []
    for t_ns, pred in preds:
        if pred is None:
            continue
        label = _label_path(labels, t_ns)
        if label is None:
            continue
        motion = label_motion_m(label)
        if motion < min_motion_m:
            continue
        e = path_errors(pred, label, horizons)
        rows.append({"t_ns": int(t_ns), "motion_m": motion,
                     **{c: (None if math.isnan(e[c]) else e[c]) for c in cols}})
    schema = {"t_ns": pl.Int64, "motion_m": pl.Float64, **{c: pl.Float64 for c in cols}}
    return pl.DataFrame(rows, schema=schema)


def score_paths(run_or_paths, labels, horizons: Sequence[float] = DEFAULT_HORIZONS, *,
                name: Optional[str] = None, baseline=None,
                min_motion_m: float = 0.0) -> pl.DataFrame:
    """ADE / FDE per horizon over valid ticks.

    Columns ``metric, horizon, p50, mean, p90, n``; ``n`` is the number of
    ticks that had the horizon. When ``baseline`` (another run or path set,
    typically the constant-velocity model) is given, its rows are appended
    and a ``source`` column (``"model"`` | ``"baseline"``) tells them apart.
    ``min_motion_m`` scores only ticks where the robot moves at least that
    far (see :func:`per_tick_errors`); both rows use the same ticks.
    """
    frame = _summarise(per_tick_errors(run_or_paths, labels, horizons, name, min_motion_m), horizons)
    if baseline is None:
        return frame
    base = _summarise(per_tick_errors(baseline, labels, horizons, min_motion_m=min_motion_m), horizons)
    return pl.concat([frame.with_columns(pl.lit("model").alias("source")),
                      base.with_columns(pl.lit("baseline").alias("source"))])


def worst(run, labels, n: int = 20, metric: str = "fde@3.0", name: Optional[str] = None,
          min_motion_m: float = 0.0) -> List[Tuple[int, float]]:
    """The ``n`` ticks with the largest ``metric`` as ``(t_ns, value)``,
    worst first. Ticks that lack the horizon are not candidates."""
    m, h = parse_metric(metric)
    col = metric_name(m, h)
    df = per_tick_errors(run, labels, (h,), name, min_motion_m).drop_nulls(col).sort(col, descending=True).head(n)
    return [(int(t), float(v)) for t, v in zip(df["t_ns"].to_list(), df[col].to_list())]


def _summarise(errors: pl.DataFrame, horizons: Sequence[float]) -> pl.DataFrame:
    rows = []
    for m in ("ade", "fde"):
        for h in horizons:
            col = metric_name(m, h)
            v = errors[col].drop_nulls().to_numpy() if col in errors.columns else np.zeros(0)
            if v.size:
                rows.append({"metric": m, "horizon": float(h), "p50": float(np.percentile(v, 50)),
                             "mean": float(v.mean()), "p90": float(np.percentile(v, 90)), "n": int(v.size)})
            else:
                rows.append({"metric": m, "horizon": float(h), "p50": None, "mean": None,
                             "p90": None, "n": 0})
    return pl.DataFrame(rows, schema={"metric": pl.Utf8, "horizon": pl.Float64, "p50": pl.Float64,
                                      "mean": pl.Float64, "p90": pl.Float64, "n": pl.Int64})


# --------------------------------------------------------------------------
# Masks
# --------------------------------------------------------------------------

def score_masks(run, labels_or_ref, classes, *, name: Optional[str] = None,
                ignore: Optional[int] = None) -> pl.DataFrame:
    """Dataset-level IoU per class from integer class-id masks.

    ``run`` and ``labels_or_ref`` each may be a Run (its Mask output; float
    ``[C, H, W]`` values are argmaxed), a ``{t_ns: mask}`` mapping, a
    sequence of ``(t_ns, mask)`` pairs, an array ``[T, H, W]`` or a
    LabelSet-like object (``t_ns`` + ``get(t_ns)`` -> dict with ``"mask"`` /
    a Mask / an array). Pairs are matched by ``t_ns`` when both sides carry
    stamps, else by tick position (a Run's ``None`` ticks keep their place,
    so an unstamped ``[T, H, W]`` reference lines up with the run's ticks);
    a tick missing on either side is skipped. A Mask's ``crop`` (and the
    stride, once a crop is in play) places it in the recorded image: both
    sides are resampled onto the region they share, at the reference's
    resolution (see the module docstring). ``classes`` is a sequence of
    names (index = class id) or ``{id: name}``. Reference pixels equal to
    ``ignore`` are excluded.

    Columns ``class_id, class, iou, intersection, union, n`` (``n`` = ticks
    scored) plus a final ``mean`` row (``class_id`` -1) with the mIoU over
    classes that occur in prediction or reference.
    """
    names = (dict(classes) if isinstance(classes, Mapping)
             else {i: str(c) for i, c in enumerate(classes)})
    pairs = _pair_masks(_iter_masks(run, name), _iter_masks(labels_or_ref, name))
    ids = sorted(int(k) for k in names)
    inter = {c: 0 for c in ids}
    union = {c: 0 for c in ids}
    n = 0
    for pred_m, ref_m in pairs:
        aligned = _align_masks(pred_m, ref_m)
        if aligned is None:
            continue
        pred, ref = aligned
        keep = np.ones(ref.shape, dtype=bool) if ignore is None else ref != ignore
        for c in ids:
            p = (pred == c) & keep
            r = (ref == c) & keep
            inter[c] += int(np.count_nonzero(p & r))
            union[c] += int(np.count_nonzero(p | r))
        n += 1
    rows = []
    ious = []
    for c in ids:
        iou = (inter[c] / union[c]) if union[c] > 0 else None
        if iou is not None:
            ious.append(iou)
        rows.append({"class_id": c, "class": names[c], "iou": iou,
                     "intersection": inter[c], "union": union[c], "n": n})
    rows.append({"class_id": -1, "class": "mean", "iou": float(np.mean(ious)) if ious else None,
                 "intersection": sum(inter.values()), "union": sum(union.values()), "n": n})
    return pl.DataFrame(rows, schema={"class_id": pl.Int64, "class": pl.Utf8, "iou": pl.Float64,
                                      "intersection": pl.Int64, "union": pl.Int64, "n": pl.Int64})


# --------------------------------------------------------------------------
# Leaderboard
# --------------------------------------------------------------------------

def leaderboard(runs_scores: Sequence[Tuple[Mapping[str, Any], pl.DataFrame]],
                sort_by: str = "fde@3.0 p50") -> pl.DataFrame:
    """One wide row per run: ``model, backbone, modalities, mode, session,
    n_ticks`` from the run meta (``Run.meta`` or any dict with those keys;
    ``spec.modalities`` is flattened to a comma-joined string) joined with
    its score frame: a path frame becomes ``"<metric>@<h> <stat>"`` columns
    plus ``"n@<h>"``, a mask frame ``"iou/<class>"`` columns. Baseline rows
    of a path frame (``source == "baseline"``) are skipped: score the
    baseline as its own run to put it on the board. Sorted ascending by
    ``sort_by`` when that column exists (nulls last).
    """
    rows: List[Dict[str, Any]] = []
    for meta, frame in runs_scores:
        row: Dict[str, Any] = {
            "model": _meta_get(meta, "model", "name"),
            "backbone": _meta_get(meta, "backbone", "backbone_id"),
            "modalities": _modalities(meta),
            "mode": _meta_get(meta, "mode"),
            "session": _session_name(meta),
            "n_ticks": _meta_get(meta, "n_ticks"),
        }
        if frame is not None and {"metric", "horizon"} <= set(frame.columns):
            f = frame
            if "source" in f.columns:
                f = f.filter(pl.col("source") != "baseline")
            for r in f.iter_rows(named=True):
                base = metric_name(str(r["metric"]), float(r["horizon"]))
                for s in _STATS:
                    row[f"{base} {s}"] = r.get(s)
                row[f"n@{float(r['horizon']):.1f}"] = r.get("n")
        elif frame is not None and {"class", "iou"} <= set(frame.columns):
            for r in frame.iter_rows(named=True):
                row[f"iou/{r['class']}"] = r["iou"]
            if "n" in frame.columns and frame.height:
                row["n_masks"] = frame["n"][0]
        rows.append(row)
    if not rows:
        return pl.DataFrame({"model": pl.Series([], dtype=pl.Utf8)})
    keys: List[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    df = pl.DataFrame([{k: r.get(k) for k in keys} for r in rows], infer_schema_length=None)
    if sort_by in df.columns:
        df = df.sort(sort_by, nulls_last=True)
    return df


# --------------------------------------------------------------------------
# Input adapters
# --------------------------------------------------------------------------

def _iter_paths(src, name: Optional[str]) -> List[Tuple[int, Optional[Path2D]]]:
    """``(t_ns, Path2D | None)`` per tick from a Run, a ``{t_ns: Path2D}``
    mapping or a sequence of ``(t_ns, Path2D)`` pairs."""
    if hasattr(src, "outputs") and hasattr(src, "t_ns"):
        outputs = src.outputs
        if name is None:
            cands = [k for k, v in outputs.items()
                     if any(isinstance(x, Path2D) for x in v)]
            if len(cands) != 1:
                raise KeyError(f"run has {len(cands)} Path2D outputs {cands}; pass name=")
            name = cands[0]
        return list(zip((int(t) for t in src.t_ns), outputs[name]))
    if isinstance(src, Mapping):
        return sorted(((int(t), p) for t, p in src.items()), key=lambda tp: tp[0])
    items = list(src)
    if items and not (isinstance(items[0], tuple) and len(items[0]) == 2):
        raise TypeError("paths must be a Run, a {t_ns: Path2D} mapping or (t_ns, Path2D) pairs")
    return [(int(t), p) for t, p in items]


def _label_path(labels, t_ns: int) -> Optional[Path2D]:
    """The valid label path at ``t_ns`` or None, cut at the record's
    ``truncated_at_s`` when it carries one. Accepts a LabelSet (``get(t_ns)``
    -> dict with ``path`` and optional ``valid`` / ``truncated_at_s``, or a
    Path2D) and a plain ``{t_ns: Path2D}`` mapping."""
    if isinstance(labels, Mapping):
        rec = labels.get(int(t_ns))
    else:
        get = getattr(labels, "get", None)
        if get is None:
            raise TypeError("labels must have get(t_ns) or be a {t_ns: Path2D} mapping")
        rec = get(int(t_ns))
    if rec is None:
        return None
    if isinstance(rec, Path2D):
        return rec
    if isinstance(rec, Mapping):
        path = rec.get("path")
        if path is None or not isinstance(path, Path2D):
            return None
        if not bool(rec.get("valid", True)):
            return None
        trunc = rec.get("truncated_at_s")
        flags = rec.get("flags")
        if trunc is None and isinstance(flags, Mapping):
            trunc = flags.get("truncated_at_s")
        if trunc is not None and math.isfinite(float(trunc)):
            return _truncate_path(path, float(trunc))
        return path
    return None


def _truncate_path(path: Path2D, at_s: float) -> Optional[Path2D]:
    """``path`` without the points past ``at_s`` (the held stop pose of a
    truncated hindsight label); None when no point is left."""
    keep = np.asarray(path.rel_t, dtype=np.float64) <= at_s + 1e-6
    if keep.all():
        return path
    if not keep.any():
        return None
    return Path2D(rel_t=path.rel_t[keep], pts=path.pts[:, keep], weights=path.weights,
                  psi=None if path.psi is None else np.asarray(path.psi)[:, keep],
                  v=None if path.v is None else np.asarray(path.v)[:, keep])


def _iter_masks(src, name: Optional[str]) -> List[Tuple[Optional[int], Any]]:
    """``(t_ns | None, Mask | array | None)`` per tick. Ticks without a mask
    stay in the list as None so positional pairing keeps the tick index."""
    if isinstance(src, np.ndarray):
        if src.ndim < 3:
            raise ValueError("a mask array must be [T, H, W] (or [T, C, H, W])")
        return [(None, src[i]) for i in range(src.shape[0])]
    if hasattr(src, "outputs") and hasattr(src, "t_ns"):
        outputs = src.outputs
        if name is None:
            cands = [k for k, v in outputs.items() if any(isinstance(x, Mask) for x in v)]
            if len(cands) != 1:
                raise KeyError(f"run has {len(cands)} Mask outputs {cands}; pass name=")
            name = cands[0]
        return [(int(t), m) for t, m in zip(src.t_ns, outputs[name])]
    if isinstance(src, Mapping):
        return sorted(((int(t), m) for t, m in src.items()), key=lambda tm: tm[0])
    get = getattr(src, "get", None)
    stamps = getattr(src, "t_ns", None)
    if callable(get) and stamps is not None and not isinstance(src, (list, tuple)):
        out = []
        for t in np.asarray(stamps):
            rec = get(int(t))
            v = rec.get("mask") if isinstance(rec, Mapping) else rec
            out.append((int(t), v))
        return out
    items = list(src)
    if items and isinstance(items[0], tuple) and len(items[0]) == 2:
        return [(int(t), m) for t, m in items]
    return [(None, m) for m in items]


def _pair_masks(preds, refs) -> List[Tuple[Any, Any]]:
    """Prediction / reference pairs by ``t_ns`` when both sides are stamped,
    else by position; a pair with a None on either side is dropped."""
    stamped = (preds and refs and all(t is not None for t, _ in preds)
               and all(t is not None for t, _ in refs))
    if stamped:
        by_t = {t: m for t, m in refs}
        pairs = [(p, by_t[t]) for t, p in preds if t in by_t]
    else:
        k = min(len(preds), len(refs))
        pairs = [(preds[i][1], refs[i][1]) for i in range(k)]
    return [(p, r) for p, r in pairs if p is not None and r is not None]


def _align_masks(pred, ref) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Class-id arrays of a prediction and its reference on one pixel grid.

    Without a crop on either side the prediction is resampled onto the
    reference's shape. With a crop, each side's extent in recorded-image
    pixels is its crop, or ``shape * stride`` for a crop-less Mask (a bare
    array is stride 1); both are resampled onto the intersection at the
    reference's pixel density. None when the extents do not overlap.
    """
    p = _to_class_ids(pred.values if isinstance(pred, Mask) else pred)
    r = _to_class_ids(ref.values if isinstance(ref, Mask) else ref)
    pc = tuple(int(x) for x in pred.crop) if isinstance(pred, Mask) and pred.crop is not None else None
    rc = tuple(int(x) for x in ref.crop) if isinstance(ref, Mask) and ref.crop is not None else None
    if pc is None and rc is None:
        if p.shape != r.shape:
            p = _resample(p, (0, p.shape[0], 0, p.shape[1]), (0, p.shape[0], 0, p.shape[1]), r.shape)
        return p, r
    pe = pc if pc is not None else _full_extent(pred, p.shape)
    re_ = rc if rc is not None else _full_extent(ref, r.shape)
    region = (max(pe[0], re_[0]), min(pe[1], re_[1]), max(pe[2], re_[2]), min(pe[3], re_[3]))
    if region[0] >= region[1] or region[2] >= region[3]:
        return None
    h = max(1, int(round((region[1] - region[0]) * r.shape[0] / (re_[1] - re_[0]))))
    w = max(1, int(round((region[3] - region[2]) * r.shape[1] / (re_[3] - re_[2]))))
    return _resample(p, pe, region, (h, w)), _resample(r, re_, region, (h, w))


def _full_extent(m, shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
    stride = int(getattr(m, "stride", 1) or 1)
    return (0, shape[0] * stride, 0, shape[1] * stride)


def _resample(a: np.ndarray, extent: Tuple[int, int, int, int], region: Tuple[int, int, int, int],
              shape: Tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour sample of ``a`` (covering ``extent`` in image
    pixels) at the centres of a ``shape`` grid laid over ``region``."""
    h, w = shape
    r0, r1, c0, c1 = extent
    R0, R1, C0, C1 = region
    img_rows = R0 + (np.arange(h) + 0.5) * (R1 - R0) / h
    img_cols = C0 + (np.arange(w) + 0.5) * (C1 - C0) / w
    rows = np.clip(np.floor((img_rows - r0) * a.shape[0] / (r1 - r0)).astype(np.int64), 0, a.shape[0] - 1)
    cols = np.clip(np.floor((img_cols - c0) * a.shape[1] / (c1 - c0)).astype(np.int64), 0, a.shape[1] - 1)
    return a[rows[:, None], cols[None, :]]


def _to_class_ids(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim == 3:
        return np.argmax(a, axis=0).astype(np.int64)
    if a.ndim != 2:
        raise ValueError(f"mask must be [H, W] or [C, H, W], got {a.shape}")
    return a.astype(np.int64, copy=False)


def _meta_get(meta: Mapping[str, Any], *keys: str):
    for k in keys:
        if k in meta and meta[k] is not None:
            return meta[k]
    return None


def _modalities(meta: Mapping[str, Any]) -> Optional[str]:
    spec = meta.get("spec")
    mods = None
    if isinstance(spec, Mapping):
        mods = spec.get("modalities")
    elif spec is not None and hasattr(spec, "modalities"):
        mods = spec.modalities
    if mods is None:
        mods = meta.get("modalities")
    if mods is None:
        return None
    return ",".join(str(m) for m in mods)


def _session_name(meta: Mapping[str, Any]) -> Optional[str]:
    s = meta.get("session")
    if isinstance(s, Mapping):
        return s.get("name")
    return None if s is None else str(s)


__all__ = ["DEFAULT_HORIZONS", "metric_name", "parse_metric", "path_errors", "per_tick_errors",
           "score_paths", "score_masks", "leaderboard", "worst"]
