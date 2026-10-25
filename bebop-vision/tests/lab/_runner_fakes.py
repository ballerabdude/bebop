"""Fakes for the runner / metrics tests: a Ticks over the synthetic arc, a
constant-velocity model and a label set built from the same arc.

Everything here is analytic: the robot drives a circle segment at speed
``v`` and yaw rate ``w`` (the trajectory ``synth.write_synthetic_session``
writes), so the body-frame future path at any tick is

    x(tau) = v/w sin(w tau),  y(tau) = v/w (1 - cos(w tau))

for every tick, and a constant-velocity model that reads ``cmd_vel`` must
reproduce the hindsight label exactly. ``FakeLabelSet(offset=...)`` shifts
every label point by a known vector so ADE and FDE have a known value.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

import synth
from bebop_vision.lab.types import (
    SERIES_DTYPES,
    Calib,
    Classes,
    Embedding,
    Grid,
    InputSpec,
    Mask,
    Path2D,
    Sample,
    Scalars,
)

ARC = dict(duration_s=4.0, hz=20.0, v=0.3, w=0.4)


def arc_body(v: float, w: float, rel_t: np.ndarray) -> np.ndarray:
    """Body-frame points ``[N, 2]`` of the arc at ``rel_t`` seconds ahead."""
    tau = np.asarray(rel_t, dtype=np.float64)
    if abs(w) < 1e-9:
        return np.stack([v * tau, np.zeros_like(tau)], axis=-1)
    return np.stack([v / w * np.sin(w * tau), v / w * (1.0 - np.cos(w * tau))], axis=-1)


def to_body(x, y, theta, xs, ys) -> np.ndarray:
    """World points -> body frame of pose ``(x, y, theta)`` (x forward, y left)."""
    dx = np.asarray(xs) - x
    dy = np.asarray(ys) - y
    c, s = math.cos(theta), math.sin(theta)
    return np.stack([c * dx + s * dy, -s * dx + c * dy], axis=-1)


class FakeSession:
    def __init__(self, name: str = "navd_session_20260101_000000", cache: Optional[Path] = None,
                 path: Optional[Path] = None, **arc):
        self.name = name
        self.path = Path(path) if path is not None else Path(f"/nonexistent/{name}.mcap")
        self.calib = Calib(roles={"near": "S-NEAR", "far": "S-FAR"}, mounts=synth.MOUNTS)
        self.arc = {**ARC, **arc}
        self._cache = cache
        # the real Session's per-session cache directory; None here stands
        # for a duck-typed session without one
        self.cache_dir: Optional[Path] = cache / name if cache is not None else None
        self.index_calls = 0

    def index(self, force: bool = False) -> Path:
        self.index_calls += 1
        if self.cache_dir is None:
            raise RuntimeError("fake session has no cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return self.cache_dir

    def ticks(self, hz=None, start_ns=None, end_ns=None) -> "FakeTicks":
        return FakeTicks(self)


class FakeTicks:
    """The contract's Ticks surface over the synthetic arc (Euler-integrated
    odom exactly as ``synth.expected_trajectory``; ``cmd_vel`` = (v, w))."""

    def __init__(self, session: FakeSession, spec: Optional[InputSpec] = None,
                 indices: Optional[Sequence[int]] = None):
        self.session = session
        a = session.arc
        self.traj = synth.expected_trajectory(**a)
        n = len(self.traj["t"])
        dt_ns = int(round(1e9 / a["hz"]))
        self._all_t_ns = np.array([synth.T0_NS + i * dt_ns for i in range(n)], dtype=np.int64)
        self.indices = list(range(n)) if indices is None else list(indices)
        self.t_ns = self._all_t_ns[self.indices]
        self.spec = spec
        self.selected: List[tuple] = []

    def select(self, *modalities, history_s=None, tol_ns=None) -> "FakeTicks":
        spec = InputSpec(tuple(modalities), dict(history_s or {}),
                         50_000_000 if tol_ns is None else int(tol_ns))
        t = FakeTicks(self.session, spec, self.indices)
        t.selected = self.selected + [(modalities, history_s, tol_ns)]
        return t

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> Sample:
        j = self.indices[i]
        t_ns = int(self._all_t_ns[j])
        mods = self.spec.modalities if self.spec is not None else ()
        data: Dict[str, object] = {}
        for m in mods:
            if m == "odom":
                rec = np.zeros(1, SERIES_DTYPES["odom"])
                rec["t_ns"], rec["x"], rec["y"], rec["theta"] = (
                    t_ns, self.traj["x"][j], self.traj["y"][j], self.traj["theta"][j])
                data[m] = rec[0]
            elif m == "cmd_vel":
                rec = np.zeros(1, SERIES_DTYPES["cmd_vel"])
                rec["t_ns"], rec["vx"], rec["wz"] = t_ns, self.traj["vx"][j], self.traj["w"]
                data[m] = rec[0]
            elif m in ("color_near", "color_far"):
                data[m] = synth.colour_frame(j)
            elif m in ("depth_near", "depth_far"):
                data[m] = synth.depth_frame(j)
            else:
                data[m] = None
        return Sample(t_ns=t_ns, session=self.session.name, index=i, data=data,
                      calib=self.session.calib, flags={"has_color_near": True, "has_color_far": True})

    def flags(self, i: int) -> dict:
        return {"has_color_near": True, "has_color_far": True}


class FakeCV:
    """Constant-velocity Model: the arc from the current ``cmd_vel``."""

    outputs = {"path": Path2D}

    def __init__(self, name: str = "cv", horizon_s: float = 3.0, dt_s: float = 0.1,
                 skip_ticks: Iterable[int] = (), with_psi: bool = False):
        self.name = name
        self.spec = InputSpec(("odom", "cmd_vel"))
        self.rel_t = np.round(np.arange(1, int(round(horizon_s / dt_s)) + 1) * dt_s, 6)
        self.prepared: List[tuple] = []
        self.steps = 0
        self.skip = set(skip_ticks)
        self.with_psi = with_psi

    def prepare(self, calib, cfg=None) -> None:
        self.prepared.append((calib, cfg))

    def step(self, sample: Sample) -> dict:
        self.steps += 1
        if sample.index in self.skip:
            return {}
        cmd = sample["cmd_vel"]
        pts = arc_body(float(cmd["vx"]), float(cmd["wz"]), self.rel_t)
        psi = (float(cmd["wz"]) * self.rel_t)[None] if self.with_psi else None
        return {"path": Path2D(self.rel_t, pts[None], psi=psi)}


class FakeBatchedCV(FakeCV):
    """Same model with a ``batch_step`` that counts its calls."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.batches: List[int] = []

    def batch_step(self, samples):
        samples = list(samples)
        self.batches.append(len(samples))
        return [self.step(s) for s in samples]


class FakeMultiModel:
    """Every output kind at once, for the save / load round trip."""

    name = "multi"
    spec = InputSpec(("color_near", "odom"), history_s={"odom": 0.5})
    outputs = {"path": Path2D, "seg": Mask, "prob": Mask, "emb": Embedding,
               "cls": Classes, "grid": Grid, "stats": Scalars}
    backbone_id = "timm:fake_tiny"
    mode = "frozen"

    def __init__(self, none_at: int = -1):
        self.none_at = none_at
        self.rel_t = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0], np.float32)

    def prepare(self, calib, cfg=None) -> None:
        self.calib = calib

    def meta(self) -> dict:
        return {"note": "fake"}

    def step(self, sample: Sample) -> dict:
        i = sample.index
        rng = np.random.default_rng(i)
        out = {
            "path": Path2D(self.rel_t, rng.normal(size=(3, 6, 2)), weights=np.array([0.5, 0.3, 0.2])),
            "seg": Mask("near", (rng.integers(0, 3, size=(8, 12))).astype(np.int64),
                        classes=("floor", "wall", "thing"), stride=4, crop=(0, 32, 0, 48)),
            "prob": Mask("far", rng.random((3, 4, 6)).astype(np.float32), classes=("a", "b", "c")),
            "emb": Embedding(rng.normal(size=16).astype(np.float32), source="timm:fake_tiny"),
            "cls": Classes(rng.normal(size=4).astype(np.float32), names=("w", "x", "y", "z")),
            "grid": Grid(rng.random((2, 5, 7)).astype(np.float32), 0.1, (0.0, -0.35), ("occ", "free")),
            "stats": Scalars({"speed": float(i), "turn": 0.5 * i} if i % 2 == 0
                             else {"speed": float(i), "extra": 1.0}),
        }
        if i == self.none_at:
            out["path"] = None
            del out["emb"]
        return out


class FakeLabelSet:
    """Hindsight labels from the analytic arc on a ``dt_s`` grid up to
    ``horizon_s``, truncated at the end of the recording like the real label
    set truncates at a reverse; ``offset`` shifts every point (known error);
    ``per_tick_offset`` maps tick index -> extra offset for a few ticks."""

    variant = "fake"
    version = 1

    def __init__(self, ticks: FakeTicks, horizon_s: float = 3.0, dt_s: float = 0.2,
                 offset: Tuple[float, float] = (0.0, 0.0),
                 per_tick_offset: Optional[Mapping[int, Tuple[float, float]]] = None,
                 source: str = "arc"):
        self.ticks = ticks
        self.t_ns = ticks.t_ns.copy()
        a = ticks.session.arc
        traj = ticks.traj
        n_all = len(traj["t"])
        self.records: Dict[int, dict] = {}
        self.valid = np.zeros(len(ticks), dtype=bool)
        for i, j in enumerate(ticks.indices):
            remaining = (n_all - 1 - j) / a["hz"]
            n_pts = int(math.floor(min(horizon_s, remaining) / dt_s + 1e-9))
            t_ns = int(ticks.t_ns[i])
            if n_pts < 1:
                self.records[t_ns] = {"path": None, "valid": False, "flags": {"truncated_at_s": 0.0}}
                continue
            rel_t = np.round(np.arange(1, n_pts + 1) * dt_s, 6)
            if source == "arc":
                pts = arc_body(a["v"], a["w"], rel_t)
            else:
                # from the Euler-integrated odom samples (every dt_s is a
                # whole number of ticks on the 20 Hz grid)
                steps = np.round(rel_t * a["hz"]).astype(int)
                pts = to_body(traj["x"][j], traj["y"][j], traj["theta"][j],
                              traj["x"][j + steps], traj["y"][j + steps])
            pts = pts + np.asarray(offset, dtype=np.float64)
            if per_tick_offset and i in per_tick_offset:
                pts = pts + np.asarray(per_tick_offset[i], dtype=np.float64)
            self.records[t_ns] = {"path": Path2D(rel_t, pts[None]), "valid": True,
                                  "flags": {"truncated_at_s": float(rel_t[-1]) if remaining < horizon_s else None}}
            self.valid[i] = True

    def get(self, t_ns: int) -> Optional[dict]:
        return self.records.get(int(t_ns))

    def paths(self) -> Dict[int, Path2D]:
        return {t: r["path"] for t, r in self.records.items() if r["valid"]}
