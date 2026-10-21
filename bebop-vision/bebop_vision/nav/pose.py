"""Gyro + wheel-odometry pose for the waypoint driver.

Wheel odometry on this robot over-reads in-place rotation badly (carpet +
castors: 2026-10-21 it reported 170 deg for a physical 82 deg turn, both
camera gyros agreeing).  Heading therefore comes from the Orbbec camera
gyros (200 Hz, rad/s, camera IMU frame rotated into the body frame with the
rig extrinsics) and only the *distance* comes from wheel odometry:

    theta += (yaw_rate - bias) * dt           (per gyro sample)
    ds     = odom displacement projected on the odom heading (sign = fwd/back)
    x, y  += ds * (cos theta, sin theta)

Gyro bias is learned while the robot is demonstrably still (no commanded
twist, odometry unchanged) and tracked slowly afterwards.  Until enough
still samples exist, or when the gyro stream stops, heading falls back to
odometry deltas and the pose reports that.  The estimate is local to the
process; it starts at the firmware odom pose seen first.
"""
from __future__ import annotations

import dataclasses
import math
import time

import numpy as np

from .depth_cone import CameraGeometry


def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


@dataclasses.dataclass
class PoseConfig:
    still_cmd_eps: float = 1e-3        # commanded |vx|,|wz| below this = still
    still_odom_eps_m: float = 2e-3     # odom motion below this per tick = still
    still_min_s: float = 0.3           # stillness required before biasing
    bias_min_samples: int = 100        # gyro samples before heading is trusted
    bias_alpha: float = 0.02           # slow tracking of the still bias
    max_sample_dt_s: float = 0.05      # gap above this is not integrated
    gyro_stale_s: float = 0.5          # no samples this long -> odom fallback
    max_rate_rad_s: float = 6.0        # reject obviously broken samples


@dataclasses.dataclass
class Pose:
    x: float
    y: float
    theta: float
    heading_source: str                # "gyro" | "odom"
    calibrated: bool
    bias_rad_s: float
    gyro_hz: float
    gyro_age_s: float | None
    odom: tuple

    def as_tuple(self):
        return (self.x, self.y, self.theta)

    def to_dict(self) -> dict:
        return {"x": round(self.x, 4), "y": round(self.y, 4),
                "theta": round(self.theta, 4),
                "theta_deg": round(math.degrees(self.theta), 1),
                "heading_source": self.heading_source,
                "calibrated": self.calibrated,
                "bias_rad_s": round(self.bias_rad_s, 5),
                "gyro_hz": round(self.gyro_hz, 1),
                "gyro_age_s": (None if self.gyro_age_s is None
                               else round(self.gyro_age_s, 3)),
                "odom": [round(float(v), 4) for v in self.odom]}


class PoseEstimator:
    """Fuses per-tick odometry with raw gyro samples from one or more
    camera IMU taps.  `taps` maps role -> deque of
    (kind, x, y, z, device_us, recv_ts); `geometries` gives each role's
    optical->body rotation."""

    def __init__(self, geometries, taps: dict, config: PoseConfig | None = None):
        self.config = config or PoseConfig()
        self._rows = {}
        for g in geometries:
            if g.role in taps and taps[g.role] is not None:
                # Body yaw rate = third row of the optical->body rotation
                # applied to the optical-frame gyro vector.
                self._rows[g.role] = g.rotation()[2].astype(np.float64)
        self.taps = {r: taps[r] for r in self._rows}
        self.x = self.y = self.theta = 0.0
        self._last_odom = None
        self._init = False
        # Per-camera gyro state: bias, sample count, last device stamp.
        self._bias = {r: 0.0 for r in self._rows}
        self._bias_n = {r: 0 for r in self._rows}
        self._last_ts = {r: None for r in self._rows}
        self._still_since: float | None = None
        self._last_sample_recv: float | None = None
        self._rate_n = 0
        self._rate_t0 = time.monotonic()
        self._gyro_hz = 0.0
        self._source = "odom"

    @property
    def has_gyro(self) -> bool:
        return bool(self.taps)

    def reset_to(self, odom) -> None:
        self.x, self.y, self.theta = float(odom[0]), float(odom[1]), float(odom[2])
        self._last_odom = tuple(odom)
        self._init = True

    # -- per tick -------------------------------------------------------------

    def update(self, odom, cmd=(0.0, 0.0), now=None) -> Pose:
        c = self.config
        now = time.monotonic() if now is None else now
        odom = tuple(float(v) for v in odom)
        if not self._init:
            self.reset_to(odom)
        # --- odometry distance (sign along the odom heading) -------------
        px, py, pth = self._last_odom
        dx, dy = odom[0] - px, odom[1] - py
        ds = dx * math.cos(pth) + dy * math.sin(pth)
        moved = math.hypot(dx, dy) > c.still_odom_eps_m or \
            abs(wrap_pi(odom[2] - pth)) > 1e-3
        d_odom_theta = wrap_pi(odom[2] - pth)
        self._last_odom = odom
        # --- stillness window for bias -----------------------------------
        still = (abs(cmd[0]) < c.still_cmd_eps and abs(cmd[1]) < c.still_cmd_eps
                 and not moved)
        if still:
            if self._still_since is None:
                self._still_since = now
        else:
            self._still_since = None
        biasing = still and (now - (self._still_since or now)) >= c.still_min_s
        # --- gyro samples (per camera: own bias, own integral) ---------
        c_min = c.bias_min_samples
        d_thetas = []
        for role, tap in self.taps.items():
            row = self._rows[role]
            samples = []
            while tap:
                kind, gx, gy, gz, dev_us, recv_ts = tap.popleft()
                if kind != "gyro":
                    continue
                rate = float(row @ np.array([gx, gy, gz]))
                if not math.isfinite(rate) or abs(rate) > c.max_rate_rad_s:
                    continue
                samples.append((dev_us * 1e-6, recv_ts, rate))
            samples.sort()
            d_theta = 0.0
            for t_dev, recv_ts, rate in samples:
                self._rate_n += 1
                if biasing:
                    n = self._bias_n[role]
                    if n < c_min:
                        self._bias[role] = (self._bias[role] * n + rate) / (n + 1)
                    else:
                        self._bias[role] += c.bias_alpha * (rate - self._bias[role])
                    self._bias_n[role] = n + 1
                last = self._last_ts[role]
                if last is not None:
                    dt = t_dev - last
                    if 0 < dt <= c.max_sample_dt_s:
                        d_theta += (rate - self._bias[role]) * dt
                self._last_ts[role] = t_dev
                self._last_sample_recv = recv_ts
            if samples and self._bias_n[role] >= c_min:
                d_thetas.append(d_theta)
        d_theta_gyro = float(np.mean(d_thetas)) if d_thetas else 0.0
        if now - self._rate_t0 >= 1.0:
            self._gyro_hz = self._rate_n / (now - self._rate_t0)
            self._rate_n = 0
            self._rate_t0 = now
        gyro_age = (None if self._last_sample_recv is None
                    else max(0.0, now - self._last_sample_recv))
        calibrated = any(n >= c_min for n in self._bias_n.values())
        gyro_ok = (calibrated and gyro_age is not None
                   and gyro_age <= c.gyro_stale_s)
        # --- integrate ---------------------------------------------------
        if gyro_ok:
            self.theta = wrap_pi(self.theta + d_theta_gyro)
            self._source = "gyro"
        else:
            self.theta = wrap_pi(self.theta + d_odom_theta)
            self._source = "odom"
        self.x += ds * math.cos(self.theta)
        self.y += ds * math.sin(self.theta)
        bias = (float(np.mean([self._bias[r] for r in self._bias]))
                if self._bias else 0.0)
        return Pose(self.x, self.y, self.theta, self._source, calibrated,
                    bias, self._gyro_hz, gyro_age, odom)
