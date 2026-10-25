"""torch Dataset over (session, tick) pairs honouring an InputSpec.

The dataset is a thin index over one or more ``Ticks`` (``Session.ticks``):
every item is the tick's ``Sample`` exactly as the session materialises it
(raw BGR frames, structured odom / cmd records) plus the per-tick labels as
tensors. Nothing is resized or normalised here: the model's backbone owns
its geometry (``Backbone.preprocess``), so one dataset serves every
architecture and the trainer batches Samples as a list.

Labels come from ``LabelSet`` objects keyed by session name; a tick without
a label (the recording ended before the horizon, a reverse window, ...) is
served with ``valid=False`` and a zero path of the right shape, so a batch
always stacks and the loss masks it out. That promise needs every drive to
be labeled: a dataset with labels for only some of its drives is refused at
construction instead of failing (or silently yielding empty label dicts) at
collate time.

Frozen-feature shortcut: when ``features`` carries a per-session
``(t_ns, pooled[N, D])`` pair (what ``FeatureCache.load`` returns), the
matching row is attached to the Sample as ``sample["features"]`` and a
``HeadModel`` in frozen mode feeds it to the head instead of re-running the
backbone over the frame. Ticks of a featured drive with no feature within
``spec.tol_ns`` are dropped (counted in ``dropped_no_features``).
"""

from __future__ import annotations

import random
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from bebop_vision.lab.types import InputSpec, Path2D, Sample


def _session_name(ticks: Any) -> str:
    """The drive a Ticks belongs to: ``ticks.session.name`` (a Session) or
    the session attribute itself when it is a plain string (fakes)."""
    session = getattr(ticks, "session", None)
    name = getattr(session, "name", session)
    if name is None:
        name = getattr(ticks, "name", None)
    if name is None:
        raise ValueError("Ticks without a session name cannot be placed in a dataset")
    return str(name)


def _normalize_labels(labels: Optional[Mapping[str, Any]], drives: Sequence[str]) -> Dict[str, List[Any]]:
    """``labels`` -> session name -> list of LabelSets.

    Accepted shapes: ``{session: LabelSet}``, ``{session: [LabelSet, ...]}``,
    ``{session: {variant: LabelSet}}`` and, for a single-session dataset only,
    ``{variant: LabelSet}`` (no key matches a drive, so the keys must be
    variants). Mixed keys are an error.
    """
    out: Dict[str, List[Any]] = {d: [] for d in drives}
    if not labels:
        return out
    keys = list(labels.keys())
    by_session = [k for k in keys if k in out]
    if by_session and len(by_session) != len(keys):
        unknown = [k for k in keys if k not in out]
        raise KeyError(f"label keys {unknown} match no drive in {list(drives)}")
    if not by_session:
        if len(drives) != 1:
            raise KeyError(f"label keys {keys} match no drive in {list(drives)}")
        out[drives[0]] = [_as_labelset(v) for v in labels.values()]
        return out
    for k in keys:
        v = labels[k]
        if isinstance(v, Mapping):
            out[k] = [_as_labelset(x) for x in v.values()]
        elif isinstance(v, (list, tuple)):
            out[k] = [_as_labelset(x) for x in v]
        else:
            out[k] = [_as_labelset(v)]
    return out


def _as_labelset(obj: Any) -> Any:
    if not hasattr(obj, "get") or not hasattr(obj, "t_ns"):
        raise TypeError(f"{obj!r} is not a LabelSet (needs .t_ns and .get)")
    return obj


class _LabelView:
    """One LabelSet's tensors per tick, with a cached template so an unlabeled
    tick still yields arrays of the set's shape."""

    def __init__(self, labelset: Any, prefix: str = ""):
        self.labelset = labelset
        self.prefix = prefix
        self.variant = str(getattr(labelset, "variant", "label"))
        self.version = int(getattr(labelset, "version", 0))
        self._template: Optional[Dict[str, np.ndarray]] = None
        self._scanned = False

    def _path_arrays(self, path: Path2D) -> Dict[str, np.ndarray]:
        out = {"path": np.asarray(path.best, dtype=np.float32),
               "rel_t": np.asarray(path.rel_t, dtype=np.float32)}
        k = int(np.argmax(path.weights))
        if path.psi is not None:
            out["psi"] = np.asarray(path.psi, dtype=np.float32).reshape(path.pts.shape[0], -1)[k]
        if path.v is not None:
            out["v"] = np.asarray(path.v, dtype=np.float32).reshape(path.pts.shape[0], -1)[k]
        return out

    def _template_from_arrays(self) -> Dict[str, np.ndarray]:
        """The set's stacked arrays (``rel_t[N]``, ``pts[T, N, 2]``, ``psi`` and
        ``v`` ``[T, N]``, what ``labels.LabelSet`` carries whatever ``valid``
        says) fix the shapes even when no tick is valid; empty when the set
        does not expose them (fakes, foreign label types)."""
        rel_t = getattr(self.labelset, "rel_t", None)
        pts = getattr(self.labelset, "pts", None)
        if rel_t is None or pts is None:
            return {}
        rel_t = np.asarray(rel_t, dtype=np.float32)
        pts = np.asarray(pts)
        if rel_t.ndim != 1 or len(rel_t) == 0 or pts.ndim != 3 or pts.shape[1] != len(rel_t):
            return {}
        out = {"path": np.zeros((len(rel_t), int(pts.shape[2])), np.float32), "rel_t": rel_t}
        for key in ("psi", "v"):
            if getattr(self.labelset, key, None) is not None:
                out[key] = np.zeros(len(rel_t), np.float32)
        return out

    def template(self) -> Dict[str, np.ndarray]:
        """Arrays an unlabeled tick yields: zeros shaped like the first valid
        label (``rel_t`` keeps the grid), else like the set's stacked arrays;
        empty when neither is available (the dataset may then hand in a
        sibling drive's template through :meth:`set_template`)."""
        if self._scanned:
            return dict(self._template or {})
        self._scanned = True
        self._template = {}
        for t in np.asarray(self.labelset.t_ns).tolist():
            rec = self.labelset.get(int(t))
            if rec and rec.get("path") is not None:
                arrays = self._path_arrays(rec["path"])
                self._template = {k: (v if k == "rel_t" else np.zeros_like(v)) for k, v in arrays.items()}
                break
        if not self._template:
            self._template = self._template_from_arrays()
        return dict(self._template)

    def set_template(self, template: Mapping[str, np.ndarray]) -> None:
        self._template = {k: np.asarray(v) for k, v in template.items()}
        self._scanned = True

    def at(self, t_ns: int) -> Dict[str, torch.Tensor]:
        rec = self.labelset.get(int(t_ns))
        valid = bool(rec is not None and rec.get("valid", True) and rec.get("path") is not None)
        arrays = self._path_arrays(rec["path"]) if valid else self.template()
        out = {self.prefix + k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in arrays.items()}
        out[self.prefix + "valid"] = torch.tensor(valid)
        return out


class LabDataset(torch.utils.data.Dataset):
    """Items are ``(Sample, dict[str, Tensor])``; batch with :func:`lab_collate`.

    ``ticks_list``: one Ticks per drive (``Session.ticks(hz)``); each is
    re-selected with ``spec`` so the Samples carry exactly the requested
    modalities. ``labels``: see :func:`_normalize_labels`; with one LabelSet
    per drive the label keys are ``path``, ``rel_t``, ``valid`` (plus ``psi``
    and ``v`` when the set carries them); with several variants per drive
    they are prefixed ``<variant>.``. ``transform(sample, labels)`` runs last
    and returns the pair. ``features``: session name -> ``(t_ns, array)``.

    With labels, every drive must have one (``ValueError`` otherwise), and a
    drive whose set has no valid tick borrows the template shape of a sibling
    drive with the same variant, so every item yields the same label keys
    and shapes and :func:`lab_collate` always stacks.
    """

    collate_fn: Callable

    def __init__(self, ticks_list: Sequence[Any], spec: InputSpec,
                 labels: Optional[Mapping[str, Any]] = None,
                 transform: Optional[Callable[[Sample, Dict[str, Any]], Tuple[Sample, Dict[str, Any]]]] = None,
                 features: Optional[Mapping[str, Any]] = None):
        if not isinstance(spec, InputSpec):
            raise TypeError("spec must be a types.InputSpec")
        self.spec = spec
        self.transform = transform
        self.sources: List[Any] = []
        self.drives: List[str] = []
        for ticks in ticks_list:
            self.drives.append(_session_name(ticks))
            self.sources.append(self._select(ticks, spec))
        if len(set(self.drives)) != len(self.drives):
            raise ValueError(f"duplicate drives in dataset: {self.drives}")
        # flat index -> (source, tick index)
        self._items: List[Tuple[int, int]] = [(j, k) for j, src in enumerate(self.sources) for k in range(len(src))]
        self._offsets = np.cumsum([0] + [len(s) for s in self.sources])
        by_session = _normalize_labels(labels, self.drives)
        if labels:
            unlabeled = [d for d, sets in by_session.items() if not sets]
            if unlabeled:
                raise ValueError(f"labels cover {[d for d in self.drives if d not in unlabeled]} "
                                 f"but not {unlabeled}; label every drive or none")
        self._labels: Dict[str, List[_LabelView]] = {}
        for drive, sets in by_session.items():
            prefix = len(sets) > 1
            self._labels[drive] = [_LabelView(s, f"{getattr(s, 'variant', i)}." if prefix else "")
                                   for i, s in enumerate(sets)]
        # a set with no valid tick (a short drive entirely inside the horizon,
        # a reverse window) has no shape of its own: lend it a sibling's
        shared: Dict[str, Dict[str, np.ndarray]] = {}
        for views in self._labels.values():
            for v in views:
                t = v.template()
                if t and v.prefix not in shared:
                    shared[v.prefix] = t
        for views in self._labels.values():
            for v in views:
                if not v.template() and v.prefix in shared:
                    v.set_template(shared[v.prefix])
        self._features: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for drive, value in (features or {}).items():
            if drive not in by_session:
                raise KeyError(f"features for unknown drive {drive!r}")
            if isinstance(value, Mapping):
                t, arr = value["t_ns"], value.get("pooled", value.get("array"))
            else:
                t, arr = value
            self._features[drive] = (np.asarray(t, dtype=np.int64), np.asarray(arr))
        # A tick with no cached feature (its frame was missing: the far camera
        # drops colour under load) is dropped from a featured drive. Kept, it
        # made HeadModel run the whole batch through the backbone on blank
        # frames (the dataset never decodes them), silently replacing every
        # cached embedding in that batch with junk.
        self.dropped_no_features: Dict[str, int] = {}
        if self._features:
            kept: List[Tuple[int, int]] = []
            for j, k in self._items:
                drive = self.drives[j]
                if drive in self._features and \
                        self._feature_at(drive, int(self.sources[j].t_ns[k])) is None:
                    self.dropped_no_features[drive] = self.dropped_no_features.get(drive, 0) + 1
                    continue
                kept.append((j, k))
            self._items = kept

    @staticmethod
    def _select(ticks: Any, spec: InputSpec) -> Any:
        """Ticks already carrying this spec are used as they are."""
        if getattr(ticks, "spec", None) == spec:
            return ticks
        select = getattr(ticks, "select", None)
        if select is None:
            return ticks
        return select(*spec.modalities, history_s=dict(spec.history_s) or None, tol_ns=spec.tol_ns)

    # -- Dataset protocol ---------------------------------------------------
    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, i: int) -> Tuple[Sample, Dict[str, Any]]:
        j, k = self._items[i]
        sample: Sample = self.sources[j][k]
        drive = self.drives[j]
        labels: Dict[str, Any] = {}
        for view in self._labels.get(drive, []):
            labels.update(view.at(int(sample.t_ns)))
        feat = self._feature_at(drive, int(sample.t_ns))
        if feat is not None:
            sample.data["features"] = feat
        if self.transform is not None:
            sample, labels = self.transform(sample, labels)
        return sample, labels

    def _feature_at(self, drive: str, t_ns: int) -> Optional[np.ndarray]:
        pair = self._features.get(drive)
        if pair is None:
            return None
        t, arr = pair
        if len(t) == 0:
            return None
        i = int(np.searchsorted(t, t_ns))
        best = min((j for j in (i - 1, i) if 0 <= j < len(t)), key=lambda j: abs(int(t[j]) - t_ns))
        if abs(int(t[best]) - t_ns) > self.spec.tol_ns:
            return None
        return np.asarray(arr[best])

    # -- introspection ------------------------------------------------------
    @property
    def t_ns(self) -> np.ndarray:
        """All tick stamps, in item order."""
        if not self._items:
            return np.empty(0, np.int64)
        return np.array([int(self.sources[j].t_ns[k]) for j, k in self._items], dtype=np.int64)

    def drive_of(self, i: int) -> str:
        return self.drives[self._items[i][0]]

    def label_variants(self) -> List[Dict[str, Any]]:
        """``[{"variant", "version"}]`` over every LabelSet, for the manifest."""
        seen: Dict[Tuple[str, int], Dict[str, Any]] = {}
        for views in self._labels.values():
            for v in views:
                seen.setdefault((v.variant, v.version), {"variant": v.variant, "version": v.version})
        return list(seen.values())

    def label_keys(self) -> List[str]:
        """Label dict keys an item yields (empty without labels), the union
        over drives in first-seen order."""
        keys: List[str] = []
        for views in self._labels.values():
            for v in views:
                for k in list(v.template().keys()) + ["valid"]:
                    if v.prefix + k not in keys:
                        keys.append(v.prefix + k)
        return keys


def lab_collate(batch: Sequence[Tuple[Sample, Mapping[str, Any]]]) -> Tuple[List[Sample], Dict[str, Any]]:
    """Samples stay a list (ragged images, None frames, structured records);
    label values that are tensors or numbers are stacked along a new batch
    axis, anything else is kept as a list."""
    samples = [s for s, _ in batch]
    labels: Dict[str, Any] = {}
    if not batch:
        return samples, labels
    keys = list(batch[0][1].keys())
    for key in keys:
        vals = [lab[key] for _, lab in batch]
        if all(isinstance(v, (torch.Tensor, np.ndarray, np.generic, bool, int, float)) for v in vals):
            try:
                labels[key] = torch.stack([torch.as_tensor(v) for v in vals])
                continue
            except RuntimeError as e:
                # ragged shapes are a dataset bug (every tick must yield the
                # template shape), so name the key instead of hiding it in a list
                raise RuntimeError(f"label {key!r} does not stack: {e}") from e
        labels[key] = vals
    return samples, labels


LabDataset.collate_fn = staticmethod(lab_collate)


def make_splits(session_names: Sequence[str], holdout: Any, seed: int = 0) -> Dict[str, List[str]]:
    """Train / val split **by drive**, never by frame: consecutive ticks of
    one recording are near-duplicates, so a frame-level split leaks.

    ``holdout``: an int (number of val drives), a float in (0, 1) (fraction,
    at least one drive) or an explicit list of val drive names. The draw is
    a seeded shuffle of the sorted names, so the same inputs always give the
    same split. ``holdout=0`` keeps every drive in train.
    """
    names = sorted(set(str(n) for n in session_names))
    if len(names) != len(list(session_names)):
        raise ValueError("session_names must be unique")
    if isinstance(holdout, (list, tuple, set)):
        val = sorted(str(n) for n in holdout)
        unknown = [n for n in val if n not in names]
        if unknown:
            raise ValueError(f"holdout drives not in session_names: {unknown}")
    else:
        if isinstance(holdout, bool):
            raise TypeError("holdout must be an int, a float or a list of names")
        if isinstance(holdout, float) and not float(holdout).is_integer():
            if not 0.0 < holdout < 1.0:
                raise ValueError("fractional holdout must be in (0, 1)")
            n_val = max(1, int(round(holdout * len(names))))
        else:
            n_val = int(holdout)
        if n_val < 0 or (n_val >= len(names) and n_val > 0):
            raise ValueError(f"holdout of {n_val} drives leaves no training drive out of {len(names)}")
        shuffled = list(names)
        random.Random(int(seed)).shuffle(shuffled)
        val = sorted(shuffled[:n_val])
    val_set = set(val)
    return {"train": [n for n in names if n not in val_set], "val": val}


def _run_rows(source: Any) -> Tuple[List[Mapping[str, Any]], List[str], List[str]]:
    """``(run rows, extra session names, unindexed names)`` from a Catalog
    (anything with ``runs()`` and ``names()``), a ``runs()`` DataFrame or an
    iterable of row dicts with ``run_id`` and ``sessions``."""
    if hasattr(source, "runs") and callable(source.runs):
        rows = source.runs().to_dicts()
        names = list(source.names()) if hasattr(source, "names") else []
        return rows, names, list(getattr(source, "runs_unindexed", []) or [])
    if hasattr(source, "to_dicts"):
        return source.to_dicts(), [], []
    return [dict(r) for r in source], [], []


def make_run_splits(catalog_or_rows: Any, holdout: Any, seed: int = 0,
                    sessions: Optional[Sequence[str]] = None) -> Dict[str, List[str]]:
    """Train / val split **by drive run**: every session of a run lands on
    the same side. A run spans several segment files (the recorder rolls
    segments), and repeating a run drives the same route again, so a
    :func:`make_splits` by file would put most of one run in train and the
    rest in val.

    ``catalog_or_rows``: a ``catalog.Catalog`` (its ``runs()`` over the
    sessions it has indexed, plus every local session name), the DataFrame
    ``Catalog.runs()`` returns, or an iterable of dicts with ``run_id`` and
    ``sessions``. ``sessions``: further session names to split; a session in
    no run (recorded before drive runs, or with no intent) is a group of its
    own. A session in two runs (one ended mid-segment and the next began)
    joins those runs into one group. A Catalog session without an index has
    no known run, so it is its own group and a warning names it (index it,
    or call ``Catalog.runs(index=True)`` first, to keep its run together).

    ``holdout`` counts groups, not sessions: an int (number of val groups),
    a float in (0, 1) (fraction of groups, at least one) or an explicit list
    of session names and / or run ids whose groups go to val. The draw is a
    seeded shuffle of the groups sorted by their first session, so the same
    inputs give the same split. Returns ``{"train": [...], "val": [...],
    "val_runs": [...]}``: sorted session names, and the run ids in val.

    Per-run labels (a hindsight path that crosses a segment boundary) are
    future work; ``labels.py`` still labels each session on its own.
    """
    rows, names, unindexed = _run_rows(catalog_or_rows)
    if unindexed:
        import warnings
        warnings.warn(f"{len(unindexed)} session(s) have no lab index, so their runs are unknown and "
                      f"each is its own split group: {unindexed}", RuntimeWarning, stacklevel=2)
    parent: Dict[str, str] = {}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def add(x: str) -> None:
        parent.setdefault(x, x)

    run_sessions: Dict[str, List[str]] = {}
    for r in rows:
        rid = str(r.get("run_id") or "")
        members = [str(s) for s in (r.get("sessions") or [])]
        if not rid or not members:
            continue
        run_sessions.setdefault(rid, [])
        for m in members:
            add(m)
            if m not in run_sessions[rid]:
                run_sessions[rid].append(m)
        root = find(members[0])
        for m in members[1:]:
            parent[find(m)] = root
    for n in list(names) + [str(s) for s in (sessions or [])]:
        add(n)
    groups: Dict[str, List[str]] = {}
    for n in parent:
        groups.setdefault(find(n), []).append(n)
    ordered = sorted((sorted(g) for g in groups.values()), key=lambda g: g[0])
    session_group = {n: i for i, g in enumerate(ordered) for n in g}

    if isinstance(holdout, (list, tuple, set)):
        val_groups = set()
        unknown = []
        for key in holdout:
            key = str(key)
            if key in session_group:
                val_groups.add(session_group[key])
            elif key in run_sessions:
                val_groups.update(session_group[m] for m in run_sessions[key])
            else:
                unknown.append(key)
        if unknown:
            raise ValueError(f"holdout names are neither sessions nor runs: {sorted(unknown)}")
    else:
        if isinstance(holdout, bool):
            raise TypeError("holdout must be an int, a float or a list of names")
        if isinstance(holdout, float) and not float(holdout).is_integer():
            if not 0.0 < holdout < 1.0:
                raise ValueError("fractional holdout must be in (0, 1)")
            n_val = max(1, int(round(holdout * len(ordered))))
        else:
            n_val = int(holdout)
        if n_val < 0 or (n_val >= len(ordered) and n_val > 0):
            raise ValueError(f"holdout of {n_val} groups leaves no training group out of {len(ordered)}")
        idx = list(range(len(ordered)))
        random.Random(int(seed)).shuffle(idx)
        val_groups = set(idx[:n_val])
    val = sorted(n for i in val_groups for n in ordered[i])
    val_set = set(val)
    train = sorted(n for n in session_group if n not in val_set)
    val_runs = sorted(rid for rid, ms in run_sessions.items() if ms and ms[0] in val_set)
    return {"train": train, "val": val, "val_runs": val_runs}
