"""Stand-ins for session.py, labels.py and backbones.py used by the dataset,
models and train tests (those modules are written concurrently; everything
here codes against the CONTRACT interfaces only)."""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from bebop_vision.lab.types import SERIES_DTYPES, InputSpec, Path2D, Sample

T0_NS = 1_792_800_000_000_000_000
TICK_NS = 50_000_000  # 20 Hz


# --------------------------------------------------------------------------
# Ticks
# --------------------------------------------------------------------------


class FakeSession:
    def __init__(self, name: str):
        self.name = name


class FakeTicks:
    """``Ticks`` per the contract: t_ns, __len__, __getitem__ -> Sample,
    select(). Frames are deterministic noise seeded by (session, index); the
    cmd / odom records describe a constant arc (vx, wz)."""

    def __init__(self, name: str, n: int = 24, vx: float = 0.3, wz: float = 0.4,
                 spec: Optional[InputSpec] = None, hw: Tuple[int, int] = (64, 96),
                 drop_far_every: int = 0, t0_ns: int = T0_NS):
        self.session = FakeSession(name)
        self.n, self.vx, self.wz, self.hw = n, vx, wz, hw
        self.drop_far_every = drop_far_every
        self.t_ns = (t0_ns + np.arange(n, dtype=np.int64) * TICK_NS).astype(np.int64)
        self.spec = spec
        self.getitem_calls = 0

    def __len__(self) -> int:
        return self.n

    def select(self, *modalities, history_s=None, tol_ns=None) -> "FakeTicks":
        spec = InputSpec(tuple(modalities), history_s=dict(history_s or {}),
                         tol_ns=tol_ns if tol_ns is not None else 50_000_000)
        out = FakeTicks(self.session.name, self.n, self.vx, self.wz, spec, self.hw,
                        self.drop_far_every, int(self.t_ns[0]))
        return out

    def frame(self, i: int) -> np.ndarray:
        rng = np.random.default_rng(zlib.crc32(f"{self.session.name}:{i}".encode()))
        return rng.integers(0, 256, size=(*self.hw, 3), dtype=np.uint8)

    def odom_rec(self, i: int) -> np.ndarray:
        t = i * TICK_NS / 1e9
        rec = np.zeros((), dtype=SERIES_DTYPES["odom"])
        th = self.wz * t
        if abs(self.wz) < 1e-9:
            x, y = self.vx * t, 0.0
        else:
            x, y = self.vx / self.wz * np.sin(th), self.vx / self.wz * (1 - np.cos(th))
        rec["t_ns"], rec["x"], rec["y"], rec["theta"] = int(self.t_ns[i]), x, y, th
        return rec

    def __getitem__(self, i: int) -> Sample:
        self.getitem_calls += 1
        if i < 0 or i >= self.n:
            raise IndexError(i)
        spec = self.spec or InputSpec(("color_near", "odom", "cmd_vel"))
        data: Dict[str, Any] = {}
        for m in spec.modalities:
            if m in ("color_near", "color_far"):
                if m == "color_far" and self.drop_far_every and i % self.drop_far_every == 0:
                    data[m] = None
                else:
                    data[m] = self.frame(i)
            elif m == "cmd_vel":
                rec = np.zeros((), dtype=SERIES_DTYPES["cmd_vel"])
                rec["t_ns"], rec["vx"], rec["wz"] = int(self.t_ns[i]), self.vx, self.wz
                data[m] = rec
            elif m == "odom":
                if spec.history(m) > 0:
                    k = int(round(spec.history(m) * 1e9 / TICK_NS))
                    rows = [self.odom_rec(j) for j in range(max(0, i - k), i + 1)]
                    data[m] = np.stack(rows).astype(SERIES_DTYPES["odom"])
                else:
                    data[m] = self.odom_rec(i)
            else:
                data[m] = None
        return Sample(t_ns=int(self.t_ns[i]), session=self.session.name, index=i, data=data,
                      flags={"has_color_near": True})


# --------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------


def arc(vx: float, wz: float, horizon_s: float = 3.0, dt_s: float = 0.2) -> Path2D:
    tau = np.arange(1, int(round(horizon_s / dt_s)) + 1) * dt_s
    if abs(wz) < 1e-9:
        x, y = vx * tau, np.zeros_like(tau)
    else:
        x, y = vx / wz * np.sin(wz * tau), vx / wz * (1 - np.cos(wz * tau))
    return Path2D(rel_t=tau, pts=np.stack([x, y], -1)[None], psi=(wz * tau)[None],
                  v=np.full((1, len(tau)), vx))


class FakeLabelSet:
    """``LabelSet`` per the contract: variant, version, t_ns, get(t_ns)."""

    def __init__(self, t_ns: np.ndarray, path: Path2D, invalid_last: int = 0,
                 variant: str = "hindsight", version: int = 1):
        self.variant, self.version = variant, version
        self.t_ns = np.asarray(t_ns, dtype=np.int64)
        self.path = path
        self.invalid_last = invalid_last

    def get(self, t_ns: int) -> Optional[dict]:
        idx = np.flatnonzero(self.t_ns == int(t_ns))
        if len(idx) == 0:
            return None
        i = int(idx[0])
        valid = i < len(self.t_ns) - self.invalid_last
        return {"path": self.path if valid else None, "valid": valid,
                "reverse_window": False, "truncated_at_s": None}


# --------------------------------------------------------------------------
# Backbones
# --------------------------------------------------------------------------


@dataclass
class FeatureInfo:
    name: str
    stride: int
    channels: int


class _Block(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, stride=stride, padding=1)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.conv(x))


class _Preprocess:
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    input_range = "01"

    def preprocess(self, bgr: np.ndarray, size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        import cv2
        img = np.asarray(bgr)
        if size is not None:
            img = cv2.resize(img, (int(size[0]), int(size[1])), interpolation=cv2.INTER_AREA)
        rgb = img[..., ::-1].astype(np.float32) / 255.0
        rgb = (rgb - np.array(self.mean, np.float32)) / np.array(self.std, np.float32)
        return torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))[None]


class FakePyramid(_Preprocess, nn.Module):
    """timm-features_only-like naming: stem_0, stages_0..3 (each a Sequential
    of two blocks), a pooling norm after."""

    id = "fake:pyramid"
    kind = "pyramid"

    def __init__(self, widths: Sequence[int] = (8, 16, 24, 32)):
        nn.Module.__init__(self)
        self.stem_0 = nn.Conv2d(3, widths[0], 3, stride=2, padding=1)
        cin = widths[0]
        for i, w in enumerate(widths):
            setattr(self, f"stages_{i}", nn.Sequential(_Block(cin, w, 2), _Block(w, w, 1)))
            cin = w
        self.norm = nn.LayerNorm(widths[-1])
        self.feature_info = [FeatureInfo(f"c{i + 2}", 4 * 2 ** i, w) for i, w in enumerate(widths)]
        self.embed_dim = widths[-1]

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.stem_0(x)
        out = {}
        for i in range(4):
            x = getattr(self, f"stages_{i}")(x)
            out[f"c{i + 2}"] = x
        return out

    def pooled(self, feats: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.norm(feats["c5"].mean(dim=(2, 3)))


class FakeTokens(_Preprocess, nn.Module):
    """ViT-like: patch conv at stride 16, ``blocks`` ModuleList, final norm."""

    id = "fake:tokens"
    kind = "tokens"

    def __init__(self, dim: int = 32, depth: int = 3, patch: int = 16):
        nn.Module.__init__(self)
        self.patch_embed = nn.Conv2d(3, dim, patch, stride=patch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(dim, dim), nn.GELU()) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.patch = patch
        self.feature_info = [FeatureInfo("tokens", patch, dim)]
        self.embed_dim = dim

    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        p = self.patch_embed(x)
        h, w = p.shape[-2:]
        t = p.flatten(2).transpose(1, 2)
        for b in self.blocks:
            t = t + b(t)
        t = self.norm(t)
        return {"tokens": t, "cls": t.mean(1) + self.cls_token[:, 0], "grid": (int(h), int(w))}

    def pooled(self, feats: Dict[str, Any]) -> torch.Tensor:
        return feats["cls"]
