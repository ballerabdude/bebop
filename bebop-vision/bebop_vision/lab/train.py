"""Training helper with run manifests.

``fit`` trains a :class:`models.HeadModel` on a :class:`dataset.LabDataset`
in one of three modes:

* ``frozen``: the head only (linear probe; the backbone never sees a
  gradient, so cached pooled features can stand in for frames);
* ``lpft``: linear probe for ``lp_epochs`` (stage 0), then head plus the
  backbone's last ``unfreeze_blocks`` block groups (stage 1) with a fresh,
  lower-lr optimizer: fine-tuning from a random head destroys pretrained
  features, probing first does not;
* ``full``: everything from the first epoch.

Every run leaves a directory the rest of the lab can read back:
``manifest.json`` (who, what, on which drives), ``metrics.jsonl`` (one line
per epoch), ``best.pt`` and ``last.pt`` (state_dict + manifest). A run is
reproducible from its manifest and seed.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

import numpy as np
import torch
import torch.nn.functional as F

from bebop_vision.lab.types import InputSpec, git_hash


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TrainConfig:
    epochs: int = 5
    lr: float = 1e-3
    batch: int = 16
    device: str = field(default_factory=_default_device)
    amp: bool = True
    mode: str = "frozen"            # frozen | lpft | full
    unfreeze_blocks: int = 2
    lp_epochs: int = 1
    seed: int = 0
    notes: str = ""
    weight_decay: float = 1e-2
    #: lr for the fine-tune stage of ``lpft`` (and all of ``full``); None = lr / 10
    ft_lr: Optional[float] = None
    num_workers: int = 0
    #: metric (a metrics.jsonl column) that picks best.pt; lower is better
    select: str = "val_loss"

    def __post_init__(self):
        if self.mode not in ("frozen", "lpft", "full"):
            raise ValueError("mode must be 'frozen', 'lpft' or 'full'")
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1")
        if not isinstance(self.select, str) or not self.select:
            raise ValueError("select must name a metrics.jsonl column")


# --------------------------------------------------------------------------
# Losses (ready-made for the two heads)
# --------------------------------------------------------------------------


def path_loss(out: Any, labels: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Winner-takes-all L1 over the K hypotheses, masked by ``valid``; with
    K > 1 a cross-entropy pushes the weights toward the winner."""
    pts, weights = out
    target = labels["path"].to(pts.device, pts.dtype)
    valid = labels["valid"].to(pts.device).float() if "valid" in labels else torch.ones(pts.shape[0], device=pts.device)
    err = (pts - target[:, None]).abs().mean(dim=(-1, -2))          # [B, K]
    best = err.argmin(dim=1)
    l1 = err.gather(1, best[:, None])[:, 0]
    denom = valid.sum().clamp(min=1.0)
    loss = (l1 * valid).sum() / denom
    if pts.shape[1] > 1:
        ce = F.nll_loss(torch.log(weights.clamp_min(1e-8)), best, reduction="none")
        loss = loss + (ce * valid).sum() / denom
    return loss


def seg_loss(logits: torch.Tensor, labels: Mapping[str, torch.Tensor], ignore_index: int = 255) -> torch.Tensor:
    """Cross-entropy against ``labels["mask"]`` (``[B, h, w]`` class ids,
    resized with nearest to the logits' grid when the strides differ)."""
    target = labels["mask"].to(logits.device).long()
    if target.shape[-2:] != logits.shape[-2:]:
        target = F.interpolate(target[:, None].float(), size=logits.shape[-2:], mode="nearest")[:, 0].long()
    return F.cross_entropy(logits, target, ignore_index=ignore_index)


# --------------------------------------------------------------------------
# Run directory
# --------------------------------------------------------------------------


class RunDir:
    """A training run on disk: ``manifest.json``, ``metrics.jsonl``,
    ``best.pt`` / ``last.pt``."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

    @property
    def manifest_path(self) -> Path:
        return self.path / "manifest.json"

    @property
    def metrics_path(self) -> Path:
        return self.path / "metrics.jsonl"

    @property
    def manifest(self) -> Dict[str, Any]:
        if not self.manifest_path.exists():
            return {}
        return json.loads(self.manifest_path.read_text())

    def write_manifest(self, manifest: Mapping[str, Any]) -> None:
        self.manifest_path.write_text(json.dumps(_jsonable(manifest), indent=2, sort_keys=True))

    def append_metrics(self, row: Mapping[str, Any]) -> None:
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(_jsonable(row), sort_keys=True) + "\n")

    def metrics(self) -> List[Dict[str, Any]]:
        if not self.metrics_path.exists():
            return []
        return [json.loads(l) for l in self.metrics_path.read_text().splitlines() if l.strip()]

    def checkpoint_path(self, name: str) -> Path:
        return self.path / f"{name}.pt"

    def save_checkpoint(self, name: str, state_dict: Mapping[str, torch.Tensor],
                        manifest: Mapping[str, Any], **extra: Any) -> Path:
        """``{"state_dict", "manifest", **extra}``; everything but the tensors
        is plain JSON-able data so ``torch.load(weights_only=True)`` accepts it."""
        payload = {"state_dict": {k: v.detach().cpu() for k, v in state_dict.items()},
                   "manifest": _jsonable(manifest), **_jsonable(extra)}
        p = self.checkpoint_path(name)
        tmp = p.with_suffix(".pt.tmp")
        torch.save(payload, tmp)
        tmp.replace(p)
        return p

    def load_checkpoint(self, name: str = "best", map_location: str = "cpu") -> Dict[str, Any]:
        return torch.load(self.checkpoint_path(name), map_location=map_location, weights_only=True)

    def load_into(self, model: Any, name: str = "best", map_location: str = "cpu") -> Dict[str, Any]:
        """Restore a checkpoint's weights into ``model`` (anything with
        ``load_state_dict``); returns the checkpoint without its tensors."""
        ckpt = self.load_checkpoint(name, map_location)
        model.load_state_dict(ckpt["state_dict"])
        return {k: v for k, v in ckpt.items() if k != "state_dict"}

    def best_metric(self) -> Optional[Dict[str, Any]]:
        return self.manifest.get("best")


def _jsonable(obj: Any) -> Any:
    """Plain JSON data from dataclasses, paths, numpy and torch values.
    NaN / inf become null (``json.dumps`` would otherwise emit bare ``NaN``
    literals that strict readers such as polars and jq reject), so numpy and
    torch values are unwrapped first and then re-checked as Python floats."""
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.generic,)):
        return _jsonable(obj.item())
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, torch.Tensor):
        return _jsonable(obj.detach().cpu().tolist() if obj.ndim else obj.item())
    if hasattr(obj, "__dataclass_fields__"):
        return _jsonable(asdict(obj))
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


# --------------------------------------------------------------------------
# fit
# --------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def seed_everything(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _spec_dict(spec: Optional[InputSpec]) -> Optional[Dict[str, Any]]:
    if spec is None:
        return None
    return {"modalities": list(spec.modalities), "history_s": dict(spec.history_s), "tol_ns": spec.tol_ns}


def _stage_for(cfg: TrainConfig, epoch: int) -> Any:
    if cfg.mode == "full":
        return "all"
    if cfg.mode == "lpft" and epoch >= cfg.lp_epochs:
        return 1
    return 0


def _collate_for(ds: Any) -> Callable:
    """The dataset's own collate (``LabDataset.collate_fn`` keeps Samples a
    list); torch's default would try to stack them, so a dataset without
    one is refused rather than guessed at."""
    fn = getattr(ds, "collate_fn", None)
    if fn is None:
        raise TypeError(f"{type(ds).__name__} has no collate_fn; LabDataset provides lab_collate")
    return fn


def _labels_to(labels: Mapping[str, Any], device) -> Dict[str, Any]:
    return {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in labels.items()}


def _loader(ds: Any, cfg: TrainConfig, shuffle: bool, generator: Optional[torch.Generator]) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(ds, batch_size=cfg.batch, shuffle=shuffle, collate_fn=_collate_for(ds),
                                       num_workers=cfg.num_workers, generator=generator, drop_last=False)


def evaluate(model: Any, ds: Any, cfg: TrainConfig, loss_fn: Callable,
             metrics_fn: Optional[Callable] = None) -> Dict[str, float]:
    """Mean loss (and ``metrics_fn(out, labels)`` means, sample-weighted)
    over ``ds`` with the model in eval mode. An empty ``ds`` is an error: a
    mean over nothing is not 0.0."""
    if len(ds) == 0:
        raise ValueError("evaluate: the dataset is empty")
    model.eval()
    device = torch.device(cfg.device)
    total, n = 0.0, 0
    sums: Dict[str, float] = {}
    with torch.inference_mode():
        for samples, labels in _loader(ds, cfg, shuffle=False, generator=None):
            labels = _labels_to(labels, device)
            with _autocast(cfg, device):
                out = model.forward_batch(samples)
                loss = loss_fn(out, labels)
            b = len(samples)
            total += float(loss) * b
            n += b
            if metrics_fn is not None:
                for k, v in (metrics_fn(out, labels) or {}).items():
                    sums[k] = sums.get(k, 0.0) + float(v) * b
    res = {"val_loss": total / max(n, 1)}
    res.update({k: v / max(n, 1) for k, v in sums.items()})
    return res


def _autocast(cfg: TrainConfig, device: torch.device):
    """AMP is a CUDA concern: on CPU bf16 autocast makes tiny heads slower
    and changes numerics, so it is left off there whatever ``cfg.amp`` says."""
    enabled = bool(cfg.amp and device.type == "cuda")
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled)


def fit(model: Any, train_ds: Any, val_ds: Any, cfg: TrainConfig, loss_fn: Callable,
        run_dir: Path, metrics_fn: Optional[Callable] = None) -> RunDir:
    """Train ``model`` (a ``HeadModel``) and leave a :class:`RunDir` behind.

    ``loss_fn(out, labels) -> Tensor`` sees the head's raw ``forward_batch``
    output and the collated label tensors on ``cfg.device``;
    ``metrics_fn(out, labels) -> dict[str, float]`` is averaged over the
    validation set per epoch. ``val_ds`` may be None or empty (what
    ``make_splits(holdout=0)`` wires up); then nothing is evaluated and
    ``train_loss`` selects ``best.pt``. ``cfg.select`` must be a column the
    first epoch's row carries (``train_loss``, ``val_loss`` or a
    ``metrics_fn`` key), else ``ValueError`` after that epoch; an absent
    column would otherwise never write ``best.pt``. Same seed, same data,
    same device -> same numbers.
    """
    run = RunDir(run_dir)
    if val_ds is not None and len(val_ds) == 0:
        val_ds = None
    generator = seed_everything(cfg.seed)
    device = torch.device(cfg.device)
    model.to(device)
    backbone = getattr(model, "backbone", None)
    manifest: Dict[str, Any] = {
        "git": git_hash(),
        "cfg": asdict(cfg),
        "model": getattr(model, "name", type(model).__name__),
        "backbone": getattr(backbone, "id", None),
        "head": type(getattr(model, "head", model)).__name__,
        "spec": _spec_dict(getattr(model, "spec", None)),
        "labels": _call(train_ds, "label_variants", []),
        "label_keys": _call(train_ds, "label_keys", []),
        "split": {"train": list(getattr(train_ds, "drives", [])),
                  "val": list(getattr(val_ds, "drives", [])) if val_ds is not None else []},
        "sizes": {"train": len(train_ds), "val": len(val_ds) if val_ds is not None else 0},
        "started": _now(), "finished": None, "epochs_done": 0, "best": None,
    }
    run.write_manifest(manifest)

    select = cfg.select
    if val_ds is None:
        if select != "val_loss":
            raise ValueError(f"select={select!r} needs a validation set; none (or an empty one) was given")
        select = "train_loss"
    best_value = math.inf
    optimizer: Optional[torch.optim.Optimizer] = None
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg.amp and device.type == "cuda"))
    stage: Any = None
    for epoch in range(cfg.epochs):
        t0 = time.perf_counter()
        new_stage = _stage_for(cfg, epoch)
        model.train()
        params = model.set_stage(new_stage)
        if new_stage != stage:
            # a stage change means a new parameter set: a fresh optimizer, with
            # the fine-tune lr once the backbone joins in
            stage = new_stage
            lr = cfg.lr if stage == 0 else (cfg.ft_lr if cfg.ft_lr is not None else cfg.lr * 0.1)
            optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=cfg.weight_decay)
        assert optimizer is not None
        total, n = 0.0, 0
        for samples, labels in _loader(train_ds, cfg, shuffle=True, generator=generator):
            labels = _labels_to(labels, device)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(cfg, device):
                out = model.forward_batch(samples)
                loss = loss_fn(out, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach()) * len(samples)
            n += len(samples)
        row: Dict[str, Any] = {"epoch": epoch, "stage": stage, "train_loss": total / max(n, 1),
                               "lr": optimizer.param_groups[0]["lr"], "n_train": n}
        if val_ds is not None:
            row.update(evaluate(model, val_ds, cfg, loss_fn, metrics_fn))
        row["time_s"] = time.perf_counter() - t0
        run.append_metrics(row)
        if select not in row:
            raise ValueError(f"select={select!r} is not a metrics column; this run logs {sorted(row)}")
        manifest["epochs_done"] = epoch + 1
        value = float(row[select])
        if value < best_value:
            best_value = value
            manifest["best"] = {"epoch": epoch, "metric": select, "value": value}
            run.save_checkpoint("best", model.state_dict(), manifest, epoch=epoch, metrics=row)
        run.save_checkpoint("last", model.state_dict(), manifest, epoch=epoch, metrics=row)
        run.write_manifest(manifest)
    manifest["finished"] = _now()
    run.write_manifest(manifest)
    model.eval()
    return run


def _call(obj: Any, method: str, default: Any) -> Any:
    fn = getattr(obj, method, None)
    return fn() if callable(fn) else default
