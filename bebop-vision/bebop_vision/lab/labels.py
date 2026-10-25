"""Labels derived from the recording itself: fused pose and hindsight paths.

A drive MCAP already contains the answer to "where did the robot go next":
wheel odometry gives the distance travelled per tick and the camera gyros
give the heading. The two are fused here (odom step distance + gyro
integrated heading, bias from still periods) because odom heading alone
over-reads in-place turns 2-3.5x on carpet and shows phantom rotation while
blocked, while the camera gyros agree with each other to 1-2 deg over 3 s
(docs/path-model.md section 4, measured on the 2026-10-18 sessions). The
hindsight path of a tick is the fused pose over the next ``horizon_s``
seconds expressed in that tick's body frame, sampled every ``dt_s``.

Reverse rule (docs/path-model.md section 4): a window whose future holds a
backward step (odom forward step < -1 cm or cmd vx < -1 cm/s) is a recorded
course correction, not a path to imitate. The label is truncated at the
first backward step and the remaining samples hold the stop pose (stop
supervision); ticks taken during the reverse are invalid as inputs (the
forward cameras do not see where the robot goes); the forward ticks right
after a reverse are flagged ``recovery_start``. The doc says "cmd vx < 0";
the 1 cm/s threshold is the prototype's (``common.py`` ``reverse_cmd``): on
navd_session_20261018_191941 30 of the 485 negative commands lie in
(-0.01, 0), joystick deadzone values that would otherwise invalidate 30
ticks and truncate 97 windows. ``HindsightConfig.reverse_cmd_m_s = 0``
restores the literal rule.

Gyro timing: the recorder stamps IMU samples at host arrival (``log_time``),
which the SDK delivers in bursts (median log dt 0.03 ms, p99 37 ms on the
2026-10-18 sessions) 18 ms (p50) to 32 ms (p90) after the device stamp.
Integration therefore uses the device clock dt, and each sample is placed
on the odom (host) axis at ``device time + fitted offset``, where the offset
is the lower envelope of ``log - device`` with its drift (measured -7.5 to
+10.5 ms/min across sessions) fitted linearly. A device clock reset (camera
reconnect) falls back to the log dt for that one step and to log-time
placement for the file; ``meta["fused"]`` records which clock was used.

Body frame: x forward, y left, heading theta counter-clockwise, as in
``datasets/bev-student/hindsight-prototype/common.py``. Everything here is
numpy only; nothing heavier is imported.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from bebop_vision.lab.types import Path2D, Series, file_identity, mount_rotation

#: Bump when the label arrays or their meaning change; readers rebuild on a
#: different producer string. 3 (2026-10-25): the dead-reckoning step sign
#: comes from the odom heading, not the fused one (see :func:`_fuse`), and
#: windows crossing a stall or gyro gap are invalid by default.
LABELS_VERSION = 3
PRODUCER = f"labels@{LABELS_VERSION}"

FUSED_POSE_DTYPE = np.dtype([
    ("t_ns", "i8"), ("x", "f8"), ("y", "f8"), ("theta_fused", "f8"),
    ("theta_odom", "f8"), ("still", "?"), ("gap", "?"),
])

LABEL_FLAGS_DTYPE = np.dtype([
    ("reverse_window", "?"),    # a backward step lies inside the window
    ("reversing", "?"),         # the robot is moving backward at the tick itself
    ("truncated_at_s", "f4"),   # rel time of the first backward step (NaN: none)
    ("still", "?"),             # the robot is still at the tick
    ("stall_gap", "?"),         # a gyro gap or odom stall overlaps the window (any part of it)
    ("recovery_start", "?"),    # forward tick within recovery_s after a reverse
])


@dataclass
class HindsightConfig:
    """Knobs of the hindsight label; the defaults are the path-model plan."""

    horizon_s: float = 3.0
    dt_s: float = 0.2
    still_s: float = 0.5          # odom unchanged this long around a sample -> still
    reverse_truncate: bool = True
    heading: str = "gyro"         # "gyro" | "odom"
    gyro_role: str = "near"       # preferred gyro; the other camera is the fallback
    reverse_step_m: float = 0.01  # odom forward step below -this is a backward step
    reverse_cmd_m_s: float = 0.01  # cmd vx below -this is a backward command (0: literal "vx < 0")
    recovery_s: float = 1.0       # forward ticks this soon after a reverse are recovery starts
    gyro_gap_s: float = 0.02      # device dt above this is a gyro stream gap
    gyro_max_dt_s: float = 0.05   # a gap integrates at most this much of the post-gap rate
    odom_gap_s: float = 0.25      # odom dt above this is a recorder stall (seen 0.5-1.2 s)
    # A window crossing a stall or gyro gap is interpolated across the hole,
    # not measured; docs/path-model.md keeps those windows out of the path
    # loss. False keeps them valid (flagged stall_gap) for inspection.
    drop_stall_windows: bool = True
    min_still_samples: int = 100  # fewer still gyro samples -> no bias estimate

    def as_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------

def to_body(xi, yi, thi, xs, ys):
    """World (odom) points -> frame i body frame: ``R(-theta_i) (p - p_i)``."""
    c, s = np.cos(thi), np.sin(thi)
    dx = np.asarray(xs, dtype=np.float64) - xi
    dy = np.asarray(ys, dtype=np.float64) - yi
    return c * dx + s * dy, -s * dx + c * dy


def _rel_grid(horizon_s: float, dt_s: float) -> np.ndarray:
    n = int(round(horizon_s / dt_s))
    if n < 1:
        raise ValueError(f"horizon_s={horizon_s} shorter than dt_s={dt_s}")
    return np.arange(1, n + 1, dtype=np.float64) * dt_s


def _series(session, name: str) -> Series:
    s = session.series(name)
    return s if s is not None else Series.empty(name)


# --------------------------------------------------------------------------
# Fused pose
# --------------------------------------------------------------------------

def fused_pose(session, cfg: Optional[HindsightConfig] = None) -> Series:
    """Odom step distance + gyro-integrated heading on the odom timeline.

    Fields: ``t_ns, x, y, theta_fused, theta_odom, still, gap``. ``theta_odom``
    is the unwrapped odom heading; ``theta_fused`` starts from it and follows
    the bias-corrected gyro (``cfg.heading="odom"``, or no usable gyro, makes
    the two equal). ``x, y`` dead-reckon the odom step length (sign from the
    forward component in the fused heading) along the fused heading from
    the first odom pose. ``still``: odom unchanged over ``still_s`` and cmd
    zero. ``gap``: the step ending at this sample overlaps a gyro stream gap
    (every step inside a gap, not just the one that closes it), lies outside
    the gyro's coverage, or is an odom stall longer than ``odom_gap_s``.
    """
    series, _ = _fuse(session, cfg or HindsightConfig())
    return series


def _fuse(session, cfg: HindsightConfig) -> Tuple[Series, Dict[str, Any]]:
    odom = _series(session, "odom")
    if len(odom) == 0:
        raise ValueError("fused_pose: session has no /odom")
    t_ns = odom.t_ns.astype(np.int64)
    t = (t_ns - t_ns[0]).astype(np.float64) * 1e-9
    x = odom["x"].astype(np.float64)
    y = odom["y"].astype(np.float64)
    th_od = np.unwrap(odom["theta"].astype(np.float64))
    n = len(t)

    # --- stillness: no odom motion within +-still_s and cmd zero ------------
    dx, dy, dth = np.diff(x), np.diff(y), np.diff(th_od)
    moved_step = (np.hypot(dx, dy) > 1e-4) | (np.abs(dth) > 1e-4)
    cm = np.concatenate([[0], np.cumsum(moved_step)])
    lo = np.searchsorted(t, t - cfg.still_s, side="left")
    hi = np.searchsorted(t, t + cfg.still_s, side="right")
    moved = (cm[np.maximum(hi - 1, lo)] - cm[lo]) > 0
    cmd = _series(session, "cmd_vel")
    if len(cmd):
        t_cmd = (cmd.t_ns.astype(np.int64) - t_ns[0]).astype(np.float64) * 1e-9
        zero = (np.abs(cmd["vx"]) < 1e-6) & (np.abs(cmd["wz"]) < 1e-6)
        cmd_zero = np.interp(t, t_cmd, zero.astype(np.float64)) > 0.5
    else:
        cmd_zero = np.ones(n, bool)
    still = (~moved) & cmd_zero

    # --- odom stalls ------------------------------------------------------
    gap = np.zeros(n, bool)
    gap[1:] = np.diff(t) > cfg.odom_gap_s

    meta: Dict[str, Any] = {"heading_source": "odom", "gyro_role": None,
                            "bias_rad_s": 0.0, "still_gyro_samples": 0,
                            "gyro_gaps": 0, "odom_stalls": int(gap.sum())}
    th_f = th_od.copy()
    if cfg.heading == "gyro":
        g = _gyro_heading(session, cfg, t_ns[0], t, still, meta)
        if g is not None:
            theta_g, gap_g = g
            th_f = th_od[0] + theta_g
            gap |= gap_g
    elif cfg.heading != "odom":
        raise ValueError(f"heading must be 'gyro' or 'odom', got {cfg.heading!r}")

    # --- dead reckoning: odom step length along the fused heading ----------
    # The sign (forward or backward) comes from the odom heading: an odom
    # step lies along odom's own heading (lateral/step 0.004-0.007 on the
    # 2026-10-18 drives). Projecting it onto the fused heading flips forward
    # driving into "reverse" once odom's turn over-read has drifted the two
    # headings more than 90 deg apart (seen: -117 deg in 193619, 873 flipped
    # steps, 353 forward-commanded ticks flagged reversing).
    fwd = np.cos(th_od[:-1]) * dx + np.sin(th_od[:-1]) * dy
    sd = np.where(fwd < 0, -1.0, 1.0) * np.hypot(dx, dy)
    xf = x[0] + np.concatenate([[0.0], np.cumsum(sd * np.cos(th_f[:-1]))])
    yf = y[0] + np.concatenate([[0.0], np.cumsum(sd * np.sin(th_f[:-1]))])

    out = np.empty(n, dtype=FUSED_POSE_DTYPE)
    out["t_ns"] = t_ns
    out["x"], out["y"] = xf, yf
    out["theta_fused"], out["theta_odom"] = th_f, th_od
    out["still"], out["gap"] = still, gap
    meta["still_odom_samples"] = int(still.sum())
    return Series(out, "fused_pose"), meta


#: Timing slack (1 ns) so a sample exactly on a window or gap boundary does
#: not count as inside it after the ns -> float conversions.
_EPS_S = 1e-9


def _gyro_heading(session, cfg: HindsightConfig, t0_ns: int, t_od: np.ndarray,
                  still: np.ndarray, meta: Dict[str, Any]):
    """Relative heading from the camera gyro at the odom times, or None.

    Body yaw rate = third row of ``mount_rotation(pitch, yaw)`` applied to
    the optical (gx, gy, gz); bias = mean rate over still samples (needs
    ``min_still_samples``); integration on the device clock dt, clipped so a
    stream gap contributes at most ``gyro_max_dt_s`` of the post-gap rate.
    Samples stay in log (arrival) order, which the recorder's drain keeps
    equal to device order; re-sorting by device time would interleave the
    two epochs of a camera reconnect. See the module docstring for the
    clock choice. Returns ``(theta_rel [n_odom], gap [n_odom])`` with
    ``theta_rel[0] = 0``; ``gap[k]`` means the odom step ending at ``k``
    overlaps a gyro stream gap or lies outside the gyro's coverage.
    """
    roles = [cfg.gyro_role] + [r for r in ("near", "far") if r != cfg.gyro_role]
    for role in roles:
        gyro = _series(session, f"imu_gyro_{role}")
        if len(gyro) < 2:
            continue
        mount = _mount(session, role)
        if mount is None:
            continue
        row = mount_rotation(float(mount["pitch_deg"]), float(mount.get("yaw_deg", 0.0)))[2]
        g = np.stack([gyro["x"], gyro["y"], gyro["z"]], axis=1).astype(np.float64)
        rate = g @ row
        t_log = (gyro.t_ns.astype(np.int64) - t0_ns).astype(np.float64) * 1e-9
        t_dev = gyro["device_us"].astype(np.float64) * 1e-6
        dt, t_g, clock = _gyro_clock(t_log, t_dev)

        still_g = np.interp(t_g, t_od, still.astype(np.float64)) > 0.5
        n_still = int(still_g.sum())
        bias = float(rate[still_g].mean()) if n_still >= cfg.min_still_samples else 0.0

        dth = np.zeros(len(rate))
        dth[1:] = (rate[1:] - bias) * np.clip(dt, 0.0, cfg.gyro_max_dt_s)
        theta_g = np.cumsum(dth)
        th_at_od = np.interp(t_od, t_g, theta_g)
        theta_rel = th_at_od - th_at_od[0]

        # Every odom step that overlaps a gap interval carries the flag: the
        # heading of a sample inside the gap is np.interp's straight line
        # across the hole, and the clip spreads at most 50 ms of rate over
        # it. So does any odom sample outside the gyro's time coverage by
        # more than a gap's worth (np.interp held the end value there; the
        # first tick routinely precedes the first gyro arrival by a few ms).
        j = np.nonzero(dt > cfg.gyro_gap_s)[0]
        gap = _steps_overlapping(t_od, t_g[j], t_g[j + 1])
        gap |= (t_od < t_g[0] - cfg.gyro_gap_s) | (t_od > t_g[-1] + cfg.gyro_gap_s)

        meta.update(heading_source="gyro", gyro_role=role, bias_rad_s=bias,
                    still_gyro_samples=n_still, gyro_gaps=int(j.size),
                    gyro_rate_hz=float(1.0 / np.median(dt)) if dt.size and np.median(dt) > 0 else 0.0,
                    **clock)
        return theta_rel, gap
    return None


def _gyro_clock(t_log: np.ndarray, t_dev: np.ndarray):
    """Per-step integration dt and host-axis placement of the gyro samples.

    Returns ``(dt [n-1], t_host [n], clock_meta)``. The device clock is the
    truth for the sample spacing; the host log time is bursty. A step whose
    device stamp goes backwards (a reset) takes the log dt instead; a
    repeated stamp integrates nothing (the next step covers two periods). A
    device clock that never advances (absent field) means log time for
    everything, as does a reset for the placement, since one linear offset
    cannot map two device epochs. ``clock_meta`` records the choice.
    """
    dt_log = np.maximum(np.diff(t_log), 0.0)
    dt_dev = np.diff(t_dev)
    resets = int((dt_dev < 0).sum())
    duplicates = int((dt_dev == 0).sum())
    if not np.any(dt_dev > 0):
        dt, t_host, source, placement = dt_log, t_log, "log", "log"
    else:
        dt = np.where(dt_dev < 0, dt_log, dt_dev)
        source = "device" if resets == 0 else "mixed"
        if resets == 0:
            t_host, fit = _device_to_host(t_log, t_dev)
            placement = "device"
        else:
            t_host, fit, placement = t_log, {}, "log"
    clock = {"gyro_clock": source, "gyro_placement": placement,
             "gyro_clock_resets": resets, "gyro_clock_duplicates": duplicates}
    if placement == "device":
        clock.update(fit)
    return dt, t_host, clock


def _device_to_host(t_log: np.ndarray, t_dev: np.ndarray):
    """Map device stamps onto the host axis: ``t_dev + (offset + drift t)``.

    ``t_log - t_dev`` is the clock offset plus the arrival latency, whose
    floor is the delivery time and whose excess is the SDK's batching. A
    least-squares line gives the drift; it is then lowered to the 2nd
    percentile of its residuals so it sits on the latency floor rather than
    the batch average (p50 18.5 ms vs the floor on the 2026-10-18 sessions).
    Returns ``(t_host, diag)``; ``diag`` carries the host-minus-device offset
    at the start (``gyro_host_offset_s``), the fitted drift
    (``gyro_drift_ms_per_min``) and the median arrival excess over the floor
    (``gyro_arrival_excess_p50_s``, the batching that log-time placement
    would have folded into the heading).
    """
    r = t_log - t_dev
    tc = t_dev - t_dev[0]
    slope = float(np.polyfit(tc, r, 1)[0]) if len(tc) >= 10 and tc[-1] > tc[0] else 0.0
    line = r.mean() + slope * (tc - tc.mean())
    shift = float(np.percentile(r - line, 2.0))
    t_host = t_dev + line + shift
    return t_host, {"gyro_host_offset_s": float(line[0] + shift),
                    "gyro_drift_ms_per_min": slope * 60e3,
                    "gyro_arrival_excess_p50_s": float(np.median(r - line - shift))}


def _steps_overlapping(tp: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Flag sample ``k`` when the step ``(tp[k-1], tp[k])`` overlaps any open
    interval ``(lo, hi)``: ``tp[k] > lo`` and ``tp[k-1] < hi``."""
    n = len(tp)
    out = np.zeros(n, bool)
    lo, hi = np.asarray(lo, np.float64), np.asarray(hi, np.float64)
    if lo.size == 0 or n < 2:
        return out
    k_lo = np.maximum(np.searchsorted(tp, lo + _EPS_S, side="right"), 1)
    k_hi = np.minimum(np.searchsorted(tp, hi - _EPS_S, side="left") + 1, n)
    keep = k_lo < k_hi
    mark = np.zeros(n + 1, np.int64)
    np.add.at(mark, k_lo[keep], 1)
    np.add.at(mark, k_hi[keep], -1)
    out[:] = np.cumsum(mark)[:n] > 0
    return out


def _mount(session, role: str) -> Optional[Dict[str, float]]:
    calib = getattr(session, "calib", None)
    if calib is None:
        return None
    try:
        m = calib.mount(role)
    except Exception:  # noqa: BLE001 - a calib without the role is "no mount"
        return None
    return m if m and "pitch_deg" in m else None


# --------------------------------------------------------------------------
# Label set
# --------------------------------------------------------------------------

@dataclass
class LabelSet:
    """Per-tick labels keyed by ``t_ns``.

    Arrays are stacked over the ``T`` ticks: ``pts[T, N, 2]`` body-frame
    points at ``rel_t[N]`` seconds after the tick (NaN where ``valid`` is
    False), ``psi[T, N]`` heading change, ``v[T, N]`` signed speed over the
    preceding ``dt``. ``flags`` is structured per tick (see
    :data:`LABEL_FLAGS_DTYPE`). A truncated label holds the stop pose from
    ``truncated_at_s`` on with ``v = 0``, so every label has ``N`` points.
    """

    variant: str
    version: int
    t_ns: np.ndarray
    rel_t: np.ndarray
    pts: np.ndarray
    psi: np.ndarray
    v: np.ndarray
    valid: np.ndarray
    flags: np.ndarray
    meta: Dict[str, Any]

    def __len__(self) -> int:
        return int(len(self.t_ns))

    def _path(self, i: int) -> Optional[Path2D]:
        """The label of tick ``i`` as a one-hypothesis :class:`Path2D`, or None.

        The arrays are copies: a consumer that edits the returned path (a
        model's augmentation, a viewer's reprojection) must not corrupt the
        label set.
        """
        if not self.valid[i]:
            return None
        return Path2D(rel_t=self.rel_t.copy(), pts=self.pts[i].copy()[None],
                      psi=self.psi[i].copy()[None], v=self.v[i].copy()[None])

    @property
    def path(self) -> "_Paths":
        """Per-tick paths: ``labels.path[i]`` and ``labels.path(i)`` both give
        tick ``i``'s :class:`Path2D` or None (the contract reads either way)."""
        return _Paths(self)

    def paths(self) -> list:
        return [self._path(i) for i in range(len(self))]

    def index_of(self, t_ns: int) -> Optional[int]:
        i = int(np.searchsorted(self.t_ns, int(t_ns)))
        if i < len(self.t_ns) and int(self.t_ns[i]) == int(t_ns):
            return i
        return None

    def get(self, t_ns: int) -> Optional[dict]:
        """``{"t_ns", "index", "valid", "path", <flags>...}`` for a label
        tick, None when ``t_ns`` is not one."""
        i = self.index_of(t_ns)
        if i is None:
            return None
        d = {"t_ns": int(self.t_ns[i]), "index": i, "valid": bool(self.valid[i]),
             "path": self._path(i)}
        for name in self.flags.dtype.names:
            val = self.flags[name][i]
            d[name] = float(val) if name == "truncated_at_s" else bool(val)
        return d

    def census(self) -> Dict[str, int]:
        f = self.flags
        return {"ticks": len(self), "valid": int(self.valid.sum()),
                "reverse_window": int(f["reverse_window"].sum()),
                "reversing": int(f["reversing"].sum()),
                "truncated": int(np.isfinite(f["truncated_at_s"]).sum()),
                "still": int(f["still"].sum()), "stall_gap": int(f["stall_gap"].sum()),
                "recovery_start": int(f["recovery_start"].sum())}

    # -- persistence -------------------------------------------------------
    def save(self, dir) -> Path:
        d = Path(dir)
        d.mkdir(parents=True, exist_ok=True)
        np.savez(d / "labels.npz", t_ns=self.t_ns, rel_t=self.rel_t, pts=self.pts,
                 psi=self.psi, v=self.v, valid=self.valid, flags=self.flags)
        meta = dict(self.meta)
        meta.update(variant=self.variant, version=self.version, producer=PRODUCER,
                    census=self.census(), written_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
        (d / "meta.json").write_text(json.dumps(meta, indent=2, default=_json_default))
        return d

    @classmethod
    def load(cls, dir) -> "LabelSet":
        d = Path(dir)
        meta = json.loads((d / "meta.json").read_text())
        with np.load(d / "labels.npz") as z:
            arrays = {k: z[k] for k in ("t_ns", "rel_t", "pts", "psi", "v", "valid", "flags")}
        return cls(variant=str(meta["variant"]), version=int(meta["version"]),
                   meta=meta, **arrays)


class _Paths:
    """Indexable and callable view of a :class:`LabelSet`'s per-tick paths."""

    __slots__ = ("_ls",)

    def __init__(self, ls: LabelSet):
        self._ls = ls

    def __call__(self, i: int) -> Optional[Path2D]:
        return self._ls._path(int(i))

    def __getitem__(self, i) -> Optional[Path2D]:
        if isinstance(i, slice):
            return [self._ls._path(k) for k in range(*i.indices(len(self._ls)))]
        return self._ls._path(int(i))

    def __len__(self) -> int:
        return len(self._ls)

    def __iter__(self):
        return (self._ls._path(k) for k in range(len(self._ls)))


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


# --------------------------------------------------------------------------
# Hindsight paths
# --------------------------------------------------------------------------

def hindsight_paths(session, ticks, cfg: Optional[HindsightConfig] = None) -> LabelSet:
    """The fused pose over the next ``horizon_s`` in each tick's body frame.

    ``ticks`` is a ``Ticks`` (anything with ``t_ns``) or an int64 array. A
    tick is valid when the pose covers its whole horizon and the robot is
    not reversing at the tick. See the module docstring for the reverse
    rule; ``cfg.reverse_truncate=False`` keeps the full path and only flags.
    """
    cfg = cfg or HindsightConfig()
    t_ticks = np.asarray(getattr(ticks, "t_ns", ticks), dtype=np.int64)
    pose, fmeta = _fuse(session, cfg)
    rel_t = _rel_grid(cfg.horizon_s, cfg.dt_s)
    N, T = len(rel_t), len(t_ticks)

    t0_ns = int(pose.t_ns[0])
    tp = (pose.t_ns.astype(np.int64) - t0_ns).astype(np.float64) * 1e-9
    px, py, pth = pose["x"], pose["y"], pose["theta_fused"]
    pstill, pgap = pose["still"], pose["gap"]
    # Signed arclength of the fused path, so v is a true path speed rather
    # than a chord over dt_s.
    step = np.hypot(np.diff(px), np.diff(py))
    fwd = np.cos(pth[:-1]) * np.diff(px) + np.sin(pth[:-1]) * np.diff(py)
    signed = np.where(fwd < 0, -step, step)
    arc = np.concatenate([[0.0], np.cumsum(signed)])

    # Backward motion: odom steps (time = step start) and backward cmd samples
    # (below -reverse_cmd_m_s; see the module docstring for the threshold).
    back_t = list(tp[:-1][fwd < -cfg.reverse_step_m])
    cmd = _series(session, "cmd_vel")
    if len(cmd):
        t_cmd = (cmd.t_ns.astype(np.int64) - t0_ns).astype(np.float64) * 1e-9
        cmd_neg = cmd["vx"] < -cfg.reverse_cmd_m_s
        back_t.extend(t_cmd[cmd_neg])
    back_t = np.unique(np.asarray(back_t, dtype=np.float64))
    # Reversing "at" a time: the odom step covering it is backward, or the
    # cmd in force is backward.
    step_back = fwd < -cfg.reverse_step_m

    pts = np.full((T, N, 2), np.nan, np.float32)
    psi = np.full((T, N), np.nan, np.float32)
    v = np.full((T, N), np.nan, np.float32)
    valid = np.zeros(T, bool)
    flags = np.zeros(T, dtype=LABEL_FLAGS_DTYPE)
    flags["truncated_at_s"] = np.nan

    t_tick = (t_ticks - t0_ns).astype(np.float64) * 1e-9
    eps = 1e-9
    for i, ti in enumerate(t_tick):
        k = int(np.searchsorted(tp, ti, side="right")) - 1   # pose sample at or before ti
        if k < 0 or ti > tp[-1] + eps:
            continue                                         # tick outside the pose
        k_near = k if (k + 1 >= len(tp) or abs(tp[k] - ti) <= abs(tp[k + 1] - ti)) else k + 1
        flags["still"][i] = bool(pstill[k_near])
        reversing = bool(k < len(step_back) and step_back[k])
        if len(cmd):
            j = int(np.searchsorted(t_cmd, ti, side="right")) - 1
            if j >= 0 and cmd_neg[j]:
                reversing = True
        flags["reversing"][i] = reversing
        # recovery start: forward tick within recovery_s after the last backward step
        if not reversing and back_t.size:
            b = int(np.searchsorted(back_t, ti, side="left")) - 1
            if b >= 0 and ti - back_t[b] <= cfg.recovery_s + eps:
                flags["recovery_start"][i] = True
        t_end = ti + cfg.horizon_s
        # window flags whatever the validity
        b0 = int(np.searchsorted(back_t, ti, side="right"))
        has_rev = b0 < back_t.size and back_t[b0] <= t_end + eps
        flags["reverse_window"][i] = bool(has_rev or reversing)
        # A flagged pose sample k stands for the step (tp[k-1], tp[k]); the
        # window (ti, t_end) is affected when any such step overlaps it,
        # including the step that straddles t_end: the label's tail is then
        # interpolated across the hole although no sample inside the window
        # carries the flag.
        ka = int(np.searchsorted(tp, ti + _EPS_S, side="right"))       # first k with tp[k] > ti
        kb = int(np.searchsorted(tp, t_end - _EPS_S, side="left")) + 1  # first k with tp[k-1] >= t_end
        flags["stall_gap"][i] = bool(pgap[ka:kb].any())
        if reversing or t_end > tp[-1] + eps:
            continue
        if cfg.drop_stall_windows and flags["stall_gap"][i]:
            continue
        tq = ti + rel_t
        xi, yi, thi = (np.interp(ti, tp, px), np.interp(ti, tp, py), np.interp(ti, tp, pth))
        xs, ys, ths = np.interp(tq, tp, px), np.interp(tq, tp, py), np.interp(tq, tp, pth)
        s = np.interp(np.concatenate([[ti], tq]), tp, arc)
        vq = np.diff(s) / cfg.dt_s
        if has_rev and cfg.reverse_truncate:
            t_rev = float(back_t[b0])
            keep = tq <= t_rev + eps
            xr, yr, thr = np.interp(t_rev, tp, px), np.interp(t_rev, tp, py), np.interp(t_rev, tp, pth)
            xs, ys, ths = np.where(keep, xs, xr), np.where(keep, ys, yr), np.where(keep, ths, thr)
            vq = np.where(keep, vq, 0.0)
            flags["truncated_at_s"][i] = t_rev - ti
        bx, by = to_body(xi, yi, thi, xs, ys)
        pts[i, :, 0], pts[i, :, 1] = bx, by
        psi[i] = ths - thi
        v[i] = vq
        valid[i] = True

    meta = {"config": cfg.as_dict(), "fused": fmeta, "session": getattr(session, "name", None),
            "source": str(getattr(session, "path", "")), "identity": _identity(session),
            "ticks": T}
    return LabelSet(variant="hindsight", version=LABELS_VERSION, t_ns=t_ticks, rel_t=rel_t,
                    pts=pts, psi=psi, v=v, valid=valid, flags=flags, meta=meta)


def _identity(session) -> Optional[str]:
    p = getattr(session, "path", None)
    try:
        return file_identity(Path(p)) if p is not None and Path(p).is_file() else None
    except OSError:
        return None


# --------------------------------------------------------------------------
# Session entry point with the cache layout
# --------------------------------------------------------------------------

def default_cache_root() -> Path:
    env = os.environ.get("BEBOP_LAB_CACHE")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "datasets" / "lab"


def label_dir(session, variant: str = "hindsight", cache_root: Optional[Path] = None) -> Path:
    """``<session cache dir>/labels/<variant>_v<version>``.

    A ``Session`` carries its own ``cache_dir`` (``<root>/<name>`` by default,
    or whatever it was constructed with); that wins over the default root so
    labels land next to the session's index. ``cache_root`` overrides both.
    """
    sub = Path("labels") / f"{variant}_v{LABELS_VERSION}"
    if cache_root is not None:
        return Path(cache_root) / str(getattr(session, "name", "session")) / sub
    own = getattr(session, "cache_dir", None)
    if own is not None:
        return Path(own) / sub
    return default_cache_root() / str(getattr(session, "name", "session")) / sub


def label_session(session, variant: str = "hindsight", ticks=None,
                  cfg: Optional[HindsightConfig] = None, out_dir=None,
                  force: bool = False) -> LabelSet:
    """Label a session and persist the result.

    Writes ``<cache>/<name>/labels/<variant>_v<version>/`` (``out_dir``
    names that directory directly when given). An existing directory whose
    ``meta.json`` carries the same identity, producer, config, camera mounts
    and ticks is loaded instead of recomputed unless ``force``. The mounts
    are part of the key because the gyro heading rotates through the mount
    pitch, and a session opened with a pitch override (the rig YAML is ~7
    deg stale) must not reuse labels made without it.
    """
    if variant != "hindsight":
        raise ValueError(f"unknown label variant {variant!r} (have: hindsight)")
    cfg = cfg or HindsightConfig()
    if ticks is None:
        ticks = session.ticks()
    t_ticks = np.asarray(getattr(ticks, "t_ns", ticks), dtype=np.int64)
    d = Path(out_dir) if out_dir is not None else label_dir(session, variant)
    mounts = _mounts_key(session)

    if not force and (d / "meta.json").is_file() and (d / "labels.npz").is_file():
        try:
            meta = json.loads((d / "meta.json").read_text())
            if (meta.get("producer") == PRODUCER and meta.get("identity") == _identity(session)
                    and meta.get("config") == cfg.as_dict()
                    and meta.get("mounts", {}) == mounts):
                ls = LabelSet.load(d)
                if len(ls) == len(t_ticks) and np.array_equal(ls.t_ns, t_ticks):
                    return ls
        except (OSError, ValueError, KeyError):
            pass  # unreadable cache: rebuild below

    ls = hindsight_paths(session, t_ticks, cfg)
    ls.meta["dir"] = str(d)
    ls.meta["mounts"] = mounts
    ls.save(d)
    return ls


def _mounts_key(session) -> Dict[str, Dict[str, float]]:
    """The mounts the labels were made with, JSON-comparable."""
    out = {}
    for role in ("near", "far"):
        m = _mount(session, role)
        if m is not None:
            out[role] = {k: round(float(v), 6) for k, v in sorted(m.items())}
    return out


# --------------------------------------------------------------------------
# Constant-velocity baseline
# --------------------------------------------------------------------------

def cv_arc(vx: float, wz: float, horizon_s: float, dt_s: float):
    """The exact constant-twist arc: ``(rel_t, pts[N, 2], psi[N], v[N])``.

    ``x = vx/wz sin(wz t)``, ``y = vx/wz (1 - cos(wz t))``; straight when
    ``wz`` is ~0. The same formula is re-implemented in ``models.py`` on
    purpose (the contract forbids a cross import between the two modules).
    """
    rel_t = _rel_grid(horizon_s, dt_s)
    if abs(wz) < 1e-9:
        xs, ys = vx * rel_t, np.zeros_like(rel_t)
    else:
        r = vx / wz
        xs, ys = r * np.sin(wz * rel_t), r * (1.0 - np.cos(wz * rel_t))
    pts = np.stack([xs, ys], axis=-1)
    return rel_t, pts, wz * rel_t, np.full_like(rel_t, float(vx))


def constant_velocity_path(sample_or_series, horizon_s: float = 3.0, dt_s: float = 0.2) -> Path2D:
    """The CV baseline from the current commanded twist.

    Accepts a ``Sample`` (its ``cmd_vel`` record), a structured record or
    dict with ``vx``/``wz``, or a ``(vx, wz)`` pair. A missing twist means
    the robot is taken as stationary (a zero path).
    """
    vx, wz = _twist_of(sample_or_series)
    rel_t, pts, psi, v = cv_arc(vx, wz, horizon_s, dt_s)
    return Path2D(rel_t=rel_t, pts=pts[None], psi=psi[None], v=v[None])


def _twist_of(obj) -> Tuple[float, float]:
    if obj is None:
        return 0.0, 0.0
    get = getattr(obj, "get", None)
    if get is not None and hasattr(obj, "data"):           # Sample
        rec = get("cmd_vel")
        if rec is None:
            return 0.0, 0.0
        return _twist_of(rec)
    if isinstance(obj, dict):
        return float(obj.get("vx", 0.0)), float(obj.get("wz", 0.0))
    names = getattr(getattr(obj, "dtype", None), "names", None)
    if names and "vx" in names and "wz" in names:            # structured record / array
        if np.ndim(obj) > 0:
            if len(obj) == 0:                                # empty history window
                return 0.0, 0.0
            obj = obj[-1]
        return float(obj["vx"]), float(obj["wz"])
    if isinstance(obj, Series):
        return (0.0, 0.0) if len(obj) == 0 else _twist_of(obj.data[-1])
    if hasattr(obj, "vx") and hasattr(obj, "wz"):            # protobuf Twist or alike
        return float(obj.vx), float(obj.wz)
    vx, wz = obj                                             # (vx, wz)
    return float(vx), float(wz)


__all__ = [
    "LABELS_VERSION", "PRODUCER", "FUSED_POSE_DTYPE", "LABEL_FLAGS_DTYPE",
    "HindsightConfig", "LabelSet", "fused_pose", "hindsight_paths", "label_session",
    "label_dir", "default_cache_root", "constant_velocity_path", "cv_arc", "to_body",
]
