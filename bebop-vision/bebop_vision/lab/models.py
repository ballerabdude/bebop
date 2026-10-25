"""Model protocol implementations: baselines and backbone + head.

* :class:`ConstantVelocity` rolls the commanded (or odometry-derived)
  velocity forward on a unicycle: the baseline every learned path model has
  to beat (the CV ADE on the 2026-10-18 drives is 0.053 m at 3 s).
* :class:`EmbeddingModel` is a backbone's pooled vector per tick, for
  retrieval and the Rerun PCA view.
* :class:`PathHead` / :class:`SegHead` are the two heads a backbone drives;
  :class:`HeadModel` wraps (backbone, head) into the ``Model`` protocol and
  owns the linear-probe / fine-tune parameter staging.

torch is imported at the top here by contract (models, dataset, train and
backbones are the only lab modules allowed to).
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from bebop_vision.lab.types import (
    Calib,
    Embedding,
    InputSpec,
    Mask,
    Path2D,
    Sample,
)

# --------------------------------------------------------------------------
# Constant-velocity baseline
# --------------------------------------------------------------------------


def rel_t_grid(horizon_s: float, dt_s: float) -> np.ndarray:
    """``dt, 2 dt, ..., horizon`` (the hindsight-label grid: no point at 0)."""
    n = int(round(horizon_s / dt_s))
    if n < 1:
        raise ValueError("horizon_s must be at least one dt_s")
    return (np.arange(1, n + 1) * dt_s).astype(np.float32)


def constant_velocity_path(vx: float, wz: float, horizon_s: float, dt_s: float) -> Path2D:
    """Unicycle roll-out of a constant (vx, wz) in the body frame.

    The same formula lives in ``labels.py`` (``constant_velocity_path``) by
    design: the two modules must not import each other and the closed form
    is six lines, so the duplication is cheaper than a shared home.
    """
    tau = rel_t_grid(horizon_s, dt_s).astype(np.float64)
    if abs(wz) < 1e-6:
        x = vx * tau
        y = np.zeros_like(tau)
    else:
        x = vx / wz * np.sin(wz * tau)
        y = vx / wz * (1.0 - np.cos(wz * tau))
    psi = (wz * tau)[None]
    v = np.full((1, len(tau)), vx)
    return Path2D(rel_t=tau, pts=np.stack([x, y], -1)[None], psi=psi, v=v)


def _rec_fields(rec: Any, fields: Sequence[str]) -> Optional[List[float]]:
    """Fields of a structured record (or the last row of a window); None when
    the record is missing or any field is NaN."""
    if rec is None:
        return None
    arr = np.asarray(rec)
    if arr.ndim >= 1:
        if arr.shape[0] == 0:
            return None
        arr = arr[-1]
    try:
        vals = [float(arr[f]) for f in fields]
    except (ValueError, KeyError, IndexError):
        return None
    return None if any(math.isnan(v) for v in vals) else vals


class ConstantVelocity:
    """``Model``: ``{"path": Path2D}`` from the current velocity.

    ``source="cmd"`` reads ``cmd_vel`` (what the operator asked for, which the
    drive follows with a 0.08 s / 0.30 s lag); ``source="odom"`` differences
    the last two odometry records, which needs ``odom`` history in the spec.
    Falls back from one to the other when the preferred one is absent, and to
    a standstill when both are.
    """

    outputs = {"path": Path2D}

    def __init__(self, horizon_s: float = 3.0, dt_s: float = 0.2, source: str = "cmd",
                 name: str = "cv"):
        if source not in ("cmd", "odom"):
            raise ValueError("source must be 'cmd' or 'odom'")
        self.horizon_s = float(horizon_s)
        self.dt_s = float(dt_s)
        self.source = source
        self.name = name
        history = {"odom": 0.5} if source == "odom" else {}
        self.spec = InputSpec(("odom", "cmd_vel"), history_s=history)

    def prepare(self, calib: Optional[Calib], cfg: Optional[Mapping[str, Any]] = None) -> None:
        return None

    def velocity(self, sample: Sample) -> Tuple[float, float]:
        cmd = _rec_fields(sample.get("cmd_vel"), ("vx", "wz"))
        odom = self._odom_velocity(sample.get("odom"))
        order = (cmd, odom) if self.source == "cmd" else (odom, cmd)
        for v in order:
            if v is not None:
                return float(v[0]), float(v[1])
        return 0.0, 0.0

    @staticmethod
    def _odom_velocity(odom: Any) -> Optional[List[float]]:
        """Body speed and yaw rate from the last two odom records (needs a
        window, i.e. odom history in the spec)."""
        if odom is None:
            return None
        arr = np.asarray(odom)
        if arr.ndim == 0 or arr.shape[0] < 2:
            return None
        a, b = arr[-2], arr[-1]
        dt = (int(b["t_ns"]) - int(a["t_ns"])) / 1e9
        if dt <= 0:
            return None
        dx, dy = float(b["x"] - a["x"]), float(b["y"] - a["y"])
        th = float(a["theta"])
        fwd = math.cos(th) * dx + math.sin(th) * dy   # body-frame forward step
        dth = math.atan2(math.sin(b["theta"] - a["theta"]), math.cos(b["theta"] - a["theta"]))
        return [fwd / dt, dth / dt]

    def step(self, sample: Sample) -> Dict[str, Any]:
        vx, wz = self.velocity(sample)
        return {"path": constant_velocity_path(vx, wz, self.horizon_s, self.dt_s)}


# --------------------------------------------------------------------------
# Shared backbone helpers
# --------------------------------------------------------------------------


def _safe_name(s: str) -> str:
    """A model name that is also a directory name (runs/<name>-<stamp>/)."""
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", s).strip("_") or "model"


def _module_device(m: nn.Module) -> torch.device:
    p = next(iter(m.parameters()), None)
    return p.device if p is not None else torch.device("cpu")


def _camera_role(modality: str) -> str:
    """``color_near`` -> ``near``."""
    return modality.split("_", 1)[1] if "_" in modality else modality


_INDEXED = re.compile(r"^(?:\d+|[A-Za-z_]+\d+)$")


def block_groups(backbone: nn.Module) -> List[Tuple[str, List[str]]]:
    """Parameter groups that stand for one "block" of a backbone, in
    registration order: the prefix up to and including the first indexed
    path component of each parameter name (``blocks.11``,
    ``encoder.layer.11``; ``stages_3`` / ``layer4`` for timm pyramids whose
    features-only wrapper flattens ``stages.3`` into ``stages_3``). A
    parameter without an indexed component (stem, pos_embed, final norm)
    belongs to no group. For a timm pyramid a group is therefore a whole
    stage, which is the granularity LP-FT wants there anyway (one stage of
    ConvNeXt-T is 3 or 9 residual blocks sharing a resolution).
    """
    groups: Dict[str, List[str]] = {}
    for name, _ in backbone.named_parameters():
        parts = name.split(".")
        for i, part in enumerate(parts[:-1]):
            if _INDEXED.match(part):
                groups.setdefault(".".join(parts[: i + 1]), []).append(name)
                break
    return list(groups.items())


def tail_parameters(backbone: nn.Module, n_blocks: int) -> List[nn.Parameter]:
    """The parameters from the start of the ``n_blocks``-th-last block group
    to the end of the backbone (so a final norm after the last block is
    included). With no block groups at all the whole backbone is returned;
    ``n_blocks <= 0`` returns nothing."""
    if n_blocks <= 0:
        return []
    groups = block_groups(backbone)
    names = [n for n, _ in backbone.named_parameters()]
    params = dict(backbone.named_parameters())
    if not groups:
        return [params[n] for n in names]
    start_name = groups[max(0, len(groups) - n_blocks)][1][0]
    start = names.index(start_name)
    return [params[n] for n in names[start:]]


def _set_training_by_grad(module: nn.Module, parent_trainable: bool) -> None:
    """``module.training`` = whether any parameter under it requires a
    gradient; a parameterless module (GELU, Dropout, DropPath) follows its
    nearest ancestor with parameters, so Dropout stays active inside a
    trainable block and off inside a frozen one."""
    params = list(module.parameters())
    trainable = any(p.requires_grad for p in params) if params else parent_trainable
    module.training = trainable
    for child in module.children():
        _set_training_by_grad(child, trainable)


def _preprocess_batch(backbone: Any, frames: Sequence[Optional[np.ndarray]],
                      size: Optional[Tuple[int, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Stack ``backbone.preprocess`` over frames; a None frame (the far
    camera drops colour under load) becomes zeros and ``present=False``."""
    tensors: List[Optional[torch.Tensor]] = []
    for f in frames:
        tensors.append(None if f is None else backbone.preprocess(f, size=size))
    ref = next((t for t in tensors if t is not None), None)
    if ref is None:
        if size is None:
            raise ValueError("every frame in the batch is missing and no size is configured")
        ref = torch.zeros(1, 3, int(size[1]), int(size[0]))
    x = torch.cat([ref.new_zeros(ref.shape) if t is None else t for t in tensors], 0)
    present = torch.tensor([t is not None for t in tensors])
    return x, present


# --------------------------------------------------------------------------
# Embedding model
# --------------------------------------------------------------------------


class EmbeddingModel:
    """``Model``: ``{"embedding": Embedding}`` = the backbone's pooled vector.

    ``size`` is ``(W, H)`` for the resize done in ``backbone.preprocess``
    (None keeps the recorded resolution). A missing frame yields a NaN
    vector so the run keeps one row per tick.
    """

    outputs = {"embedding": Embedding}

    def __init__(self, backbone: nn.Module, modality: str = "color_far",
                 size: Optional[Tuple[int, int]] = None, name: Optional[str] = None,
                 device: Optional[str] = None):
        self.backbone = backbone
        self.modality = modality
        self.size = tuple(size) if size is not None else None
        self.spec = InputSpec((modality,))
        self.name = name or _safe_name(f"emb-{getattr(backbone, 'id', 'backbone')}")
        self.device = device

    def prepare(self, calib: Optional[Calib], cfg: Optional[Mapping[str, Any]] = None) -> None:
        device = (cfg or {}).get("device", self.device)
        if device is not None:
            self.backbone.to(device)
        self.backbone.eval()

    @torch.inference_mode()
    def batch_step(self, samples: Sequence[Sample]) -> List[Dict[str, Any]]:
        frames = [s.get(self.modality) for s in samples]
        dim = int(getattr(self.backbone, "embed_dim", 0))
        source = str(getattr(self.backbone, "id", ""))
        out: List[Optional[np.ndarray]] = [None] * len(frames)
        idx = [i for i, f in enumerate(frames) if f is not None]
        if idx:
            x, _ = _preprocess_batch(self.backbone, [frames[i] for i in idx], self.size)
            feats = self.backbone(x.to(_module_device(self.backbone)))
            pooled = self.backbone.pooled(feats).float().cpu().numpy()
            dim = pooled.shape[1]
            for row, i in enumerate(idx):
                out[i] = pooled[row]
        nan = np.full(dim, np.nan, np.float32)
        return [{"embedding": Embedding(vector=np.asarray(v if v is not None else nan, np.float32),
                                        source=source)} for v in out]

    def step(self, sample: Sample) -> Dict[str, Any]:
        return self.batch_step([sample])[0]


# --------------------------------------------------------------------------
# Heads
# --------------------------------------------------------------------------


class PathHead(nn.Module):
    """Pooled features ``[B, D]`` (+ scalars ``[B, S]``) -> ``K`` timed paths.

    Returns ``(pts [B, K, N, 2], weights [B, K])``. The MLP predicts per-step
    body-frame increments that are cumulatively summed along N: a path is
    then smooth by construction and a constant arc is a bias.

    The features and the scalars are encoded separately and joined after:
    the features through a LayerNorm (DINOv3 ConvNeXt pooled vectors have a
    norm near 50, ResNet-50's near 4), the scalars through their own small
    layer. Concatenated raw, two twist numbers around 0.3 next to 768
    features of that size were ignored: a DINOv3 ConvNeXt-T head scored the
    same with and without the twist (path_v0, 2026-10-25).

    ``residual="cv"`` adds the prediction to the constant-twist arc of the
    first two scalars, which must be ``(vx, wz)`` (``cmd_vel`` first in
    ``HeadModel.scalar_modalities``): the head then learns where the robot
    departs from its current motion instead of the motion itself, and a
    zero output is the constant-velocity baseline. ``dt_s`` is the point
    spacing (point ``i`` at ``(i + 1) * dt_s``).
    """

    def __init__(self, in_dim: int, n_scalars: int = 0, n_points: int = 15, k: int = 1,
                 hidden: int = 256, dropout: float = 0.0, residual: Optional[str] = None,
                 dt_s: float = 0.2, scalar_hidden: int = 64):
        super().__init__()
        self.in_dim, self.n_scalars, self.n_points, self.k = int(in_dim), int(n_scalars), int(n_points), int(k)
        if residual not in (None, "cv"):
            raise ValueError("residual must be None or 'cv'")
        if residual == "cv" and self.n_scalars < 2:
            raise ValueError("residual='cv' needs the (vx, wz) scalars first (n_scalars >= 2)")
        self.residual, self.dt_s = residual, float(dt_s)
        self.feat = nn.Sequential(nn.LayerNorm(self.in_dim), nn.Linear(self.in_dim, hidden), nn.GELU())
        self.scal = (nn.Sequential(nn.Linear(self.n_scalars, scalar_hidden), nn.GELU())
                     if self.n_scalars else None)
        joined = hidden + (scalar_hidden if self.n_scalars else 0)
        self.mlp = nn.Sequential(nn.Dropout(dropout), nn.Linear(joined, hidden), nn.GELU())
        self.delta = nn.Linear(hidden, self.k * self.n_points * 2)
        self.logit = nn.Linear(hidden, self.k)
        if residual == "cv":
            # start exactly at the baseline
            nn.init.zeros_(self.delta.weight)
            nn.init.zeros_(self.delta.bias)

    def cv_arc(self, vx: torch.Tensor, wz: torch.Tensor) -> torch.Tensor:
        """The constant-twist arc ``[B, N, 2]`` (left turn = positive wz)."""
        t = torch.arange(1, self.n_points + 1, device=vx.device, dtype=vx.dtype) * self.dt_s
        vx, wz = vx[:, None], wz[:, None]
        small = wz.abs() < 1e-4
        wz_safe = torch.where(small, torch.ones_like(wz), wz)
        x = torch.where(small, vx * t, vx / wz_safe * torch.sin(wz * t))
        y = torch.where(small, 0.5 * vx * wz * t * t, vx / wz_safe * (1.0 - torch.cos(wz * t)))
        return torch.stack([x, y], -1)

    def forward(self, pooled: torch.Tensor, scalars: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = pooled.flatten(1)
        parts = [self.feat(x)]
        if self.n_scalars:
            if scalars is None:
                scalars = x.new_zeros(x.shape[0], self.n_scalars)
            scalars = scalars.to(x.dtype)
            parts.append(self.scal(scalars))
        h = self.mlp(torch.cat(parts, 1))
        delta = self.delta(h).view(-1, self.k, self.n_points, 2)
        pts = torch.cumsum(delta, dim=2)
        if self.residual == "cv":
            pts = pts + self.cv_arc(scalars[:, 0], scalars[:, 1])[:, None]
        weights = torch.softmax(self.logit(h), dim=1)
        return pts, weights


def _level_info(info: Any) -> Tuple[str, int, int]:
    """``FeatureInfo``-like (name, stride, channels) from an object, a dict or
    a tuple, so this module needs no import from backbones.py."""
    if isinstance(info, Mapping):
        return str(info["name"]), int(info["stride"]), int(info["channels"])
    if isinstance(info, (tuple, list)):
        return str(info[0]), int(info[1]), int(info[2])
    return str(info.name), int(info.stride), int(info.channels)


class SegHead(nn.Module):
    """FPN-lite: lateral 1x1 to ``channels``, nearest top-down sum, one 3x3,
    classifier; logits at stride 4 of the network input.

    Accepts a pyramid ``{"c2".."c5"}`` or ViT tokens ``{"tokens", "grid"}``
    (one level at the patch stride, upsampled). ``channels`` stays <= 128 so
    the head is cheap next to any backbone.
    """

    OUT_STRIDE = 4

    def __init__(self, feature_info: Sequence[Any], n_classes: int, channels: int = 128,
                 classes: Sequence[str] = ()):
        super().__init__()
        if channels > 128:
            raise ValueError("SegHead keeps channels <= 128 by contract")
        levels = sorted((_level_info(i) for i in feature_info), key=lambda l: l[1])
        if not levels:
            raise ValueError("feature_info is empty")
        self.levels = [name for name, _, _ in levels]
        self.strides = {name: stride for name, stride, _ in levels}
        self.lateral = nn.ModuleDict({name: nn.Conv2d(ch, channels, 1) for name, _, ch in levels})
        self.smooth = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1), nn.GELU())
        self.cls = nn.Conv2d(channels, int(n_classes), 1)
        self.n_classes = int(n_classes)
        self.classes = tuple(classes) if classes else tuple(f"c{i}" for i in range(self.n_classes))

    def forward(self, feats: Mapping[str, Any]) -> torch.Tensor:
        tokens = feats.get("tokens")
        if tokens is not None and tokens.ndim == 3:
            # [B, N, D] patch tokens -> one [B, D, h, w] level at the patch stride
            h, w = feats["grid"]
            feats = {self.levels[0]: tokens.transpose(1, 2).reshape(tokens.shape[0], -1, int(h), int(w))}
        fine_to_coarse = [n for n in self.levels if n in feats]
        if not fine_to_coarse:
            raise KeyError(f"features carry none of {self.levels}")
        y = None
        for name in reversed(fine_to_coarse):
            lat = self.lateral[name](feats[name])
            y = lat if y is None else lat + F.interpolate(y, size=lat.shape[-2:], mode="nearest")
        logits = self.cls(self.smooth(y))
        finest = self.strides[fine_to_coarse[0]]
        if finest != self.OUT_STRIDE:
            logits = F.interpolate(logits, scale_factor=finest / self.OUT_STRIDE,
                                   mode="bilinear", align_corners=False)
        return logits


# --------------------------------------------------------------------------
# Backbone + head as a Model
# --------------------------------------------------------------------------

#: Scalar inputs a HeadModel can read from a Sample's state / latest records.
SCALAR_FIELDS: Dict[str, Tuple[str, ...]] = {
    "cmd_vel": ("vx", "wz"),
    "runtime": ("wheel_l_vel", "wheel_r_vel"),
}


class HeadModel:
    """``Model`` around ``(backbone, head)`` with LP-FT parameter staging.

    ``kind`` is ``"path"`` (``PathHead``: pooled features + scalars ->
    ``Path2D`` on the ``horizon_s / dt_s`` grid) or ``"seg"`` (``SegHead`` ->
    ``Mask`` for the modality's camera); it is inferred from the head class.
    ``modality`` names the image the backbone sees; ``size`` is its ``(W, H)``
    resize; ``scalar_modalities`` are extra Sample records flattened through
    :data:`SCALAR_FIELDS` (``cmd_vel`` -> vx, wz) for a PathHead.

    ``mode`` and ``unfreeze_blocks`` describe how :func:`train.fit` stages
    the parameters: ``trainable_parameters(0)`` is the head, ``(1)`` adds the
    backbone's last ``unfreeze_blocks`` block groups (see
    :func:`block_groups`) and ``("all")`` everything.
    """

    def __init__(self, backbone: nn.Module, head: nn.Module, kind: Optional[str] = None,
                 modality: str = "color_near", size: Optional[Tuple[int, int]] = None,
                 scalar_modalities: Sequence[str] = (), mode: str = "frozen",
                 unfreeze_blocks: int = 2, horizon_s: float = 3.0, dt_s: float = 0.2,
                 name: Optional[str] = None, device: Optional[str] = None):
        if kind is None:
            kind = "path" if isinstance(head, PathHead) else "seg" if isinstance(head, SegHead) else None
        if kind not in ("path", "seg"):
            raise ValueError("kind must be 'path' or 'seg' (or the head a PathHead / SegHead)")
        if mode not in ("frozen", "lpft", "full"):
            raise ValueError("mode must be 'frozen', 'lpft' or 'full'")
        for m in scalar_modalities:
            if m not in SCALAR_FIELDS:
                raise ValueError(f"no scalar fields for {m!r}; known: {sorted(SCALAR_FIELDS)}")
        self.backbone = backbone
        self.head = head
        self.net = nn.ModuleDict({"backbone": backbone, "head": head})
        self.kind = kind
        self.modality = modality
        self.size = tuple(size) if size is not None else None
        self.scalar_modalities = tuple(scalar_modalities)
        self.mode = mode
        self.unfreeze_blocks = int(unfreeze_blocks)
        self.horizon_s, self.dt_s = float(horizon_s), float(dt_s)
        self.rel_t = rel_t_grid(horizon_s, dt_s)
        if kind == "path":
            if int(getattr(head, "n_points", len(self.rel_t))) != len(self.rel_t):
                raise ValueError(f"PathHead has {head.n_points} points but horizon_s / dt_s "
                                 f"gives {len(self.rel_t)}")
            want = self.scalar_dim(self.scalar_modalities)
            if int(getattr(head, "n_scalars", want)) != want:
                raise ValueError(f"PathHead expects {head.n_scalars} scalars but "
                                 f"{self.scalar_modalities} give {want}")
            self.outputs = {"path": Path2D}
        else:
            self.outputs = {"mask": Mask}
        self.spec = InputSpec((modality,) + self.scalar_modalities)
        self.name = name or _safe_name(f"{kind}-{getattr(backbone, 'id', 'backbone')}")
        if device is not None:
            self.net.to(device)

    # -- torch delegation ---------------------------------------------------
    def parameters(self) -> Iterable[nn.Parameter]:
        return self.net.parameters()

    def train(self, mode: bool = True) -> "HeadModel":
        self.net.train(mode)
        return self

    def eval(self) -> "HeadModel":
        self.net.eval()
        return self

    def to(self, device) -> "HeadModel":
        self.net.to(device)
        return self

    @property
    def device(self) -> torch.device:
        return _module_device(self.net)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return self.net.state_dict()

    def load_state_dict(self, state: Mapping[str, torch.Tensor], strict: bool = True):
        return self.net.load_state_dict(state, strict=strict)

    # -- staging ------------------------------------------------------------
    def trainable_parameters(self, stage: Any) -> List[nn.Parameter]:
        """Stage 0: head. Stage 1: head + the backbone's last
        ``unfreeze_blocks`` block groups (:func:`tail_parameters`). ``"all"``:
        everything (mode ``full``)."""
        head = list(self.head.parameters())
        if stage == "all":
            return head + list(self.backbone.parameters())
        if stage == 0:
            return head
        if stage == 1:
            return head + tail_parameters(self.backbone, self.unfreeze_blocks)
        raise ValueError(f"unknown stage {stage!r}")

    def set_stage(self, stage: Any) -> List[nn.Parameter]:
        """Freeze everything but the stage's parameters, and put every frozen
        part of the backbone in eval mode (when the model is training) so its
        BatchNorm statistics, Dropout and DropPath stay put: freezing the
        parameters alone leaves a timm ResNet's early stages drifting their
        running means under the fine-tune batches."""
        params = self.trainable_parameters(stage)
        ids = {id(p) for p in params}
        for p in self.net.parameters():
            p.requires_grad_(id(p) in ids)
        if self.net.training:
            _set_training_by_grad(self.backbone, parent_trainable=True)
        return params

    # -- inputs -------------------------------------------------------------
    @staticmethod
    def scalar_dim(scalar_modalities: Sequence[str]) -> int:
        return sum(len(SCALAR_FIELDS[m]) for m in scalar_modalities)

    def scalars(self, samples: Sequence[Sample]) -> Optional[torch.Tensor]:
        """``[B, S]`` from the samples' records (zeros for a missing one)."""
        if not self.scalar_modalities:
            return None
        rows = []
        for s in samples:
            row: List[float] = []
            for m in self.scalar_modalities:
                vals = _rec_fields(s.get(m), SCALAR_FIELDS[m])
                row += vals if vals is not None else [0.0] * len(SCALAR_FIELDS[m])
            rows.append(row)
        return torch.tensor(rows, dtype=torch.float32)

    def uses_cached_features(self, samples: Sequence[Sample]) -> bool:
        """Frozen path head + every sample carrying ``features`` (the dataset's
        pooled-feature shortcut): skip the backbone."""
        return (self.kind == "path" and self.mode == "frozen"
                and all(s.get("features") is not None for s in samples))

    def forward_batch(self, samples: Sequence[Sample]) -> Any:
        """The head's raw output for a list of Samples (differentiable)."""
        device = self.device
        n_feat = sum(s.get("features") is not None for s in samples)
        if 0 < n_feat < len(samples):
            # Never fall back to the backbone for a mixed batch: the samples
            # without features usually carry no frame either (a cached-feature
            # dataset does not decode images), and the fallback would replace
            # the good cached embeddings of the whole batch.
            raise ValueError(f"{len(samples) - n_feat} of {len(samples)} samples have no cached "
                             "features; drop them from the dataset (LabDataset does this) or "
                             "cache features for every tick")
        if self.uses_cached_features(samples):
            pooled = torch.as_tensor(np.stack([np.asarray(s["features"], np.float32) for s in samples])).to(device)
            return self.head(pooled, self._scalars_to(samples, device))
        x, _ = _preprocess_batch(self.backbone, [s.get(self.modality) for s in samples], self.size)
        feats = self.backbone(x.to(device))
        if self.kind == "path":
            return self.head(self.backbone.pooled(feats), self._scalars_to(samples, device))
        return self.head(feats)

    def _scalars_to(self, samples: Sequence[Sample], device) -> Optional[torch.Tensor]:
        sc = self.scalars(samples)
        return sc.to(device) if sc is not None else None

    # -- Model protocol -----------------------------------------------------
    def prepare(self, calib: Optional[Calib], cfg: Optional[Mapping[str, Any]] = None) -> None:
        device = (cfg or {}).get("device")
        if device is not None:
            self.net.to(device)
        self.net.eval()

    def to_outputs(self, out: Any, samples: Sequence[Sample]) -> List[Dict[str, Any]]:
        """Typed outputs per sample from a raw ``forward_batch`` result."""
        if self.kind == "path":
            pts, weights = out
            pts = pts.detach().float().cpu().numpy()
            weights = weights.detach().float().cpu().numpy()
            return [{"path": Path2D(rel_t=self.rel_t, pts=pts[i], weights=weights[i])}
                    for i in range(len(samples))]
        logits = out.detach().float().cpu().numpy()
        role = _camera_role(self.modality)
        results = []
        for i, s in enumerate(samples):
            frame = s.get(self.modality)
            # stride relative to the recorded image: 4 at native resolution,
            # scaled when ``size`` resizes the network input
            stride = int(round(frame.shape[1] / logits.shape[-1])) if frame is not None else SegHead.OUT_STRIDE
            results.append({"mask": Mask(camera=role, values=logits[i], classes=self.head.classes,
                                         stride=max(1, stride))})
        return results

    def batch_step(self, samples: Sequence[Sample]) -> List[Dict[str, Any]]:
        samples = list(samples)
        # per-module flags: under LP-FT the backbone is in eval while the head
        # trains, and a blanket train(True) afterwards would undo that
        modes = [(m, m.training) for m in self.net.modules()]
        self.net.eval()
        try:
            with torch.inference_mode():
                out = self.forward_batch(samples)
        finally:
            for m, flag in modes:
                m.training = flag
        return self.to_outputs(out, samples)

    def step(self, sample: Sample) -> Dict[str, Any]:
        return self.batch_step([sample])[0]
