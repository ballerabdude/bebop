"""Sector planner: navigable-path mask -> differential-drive twist.

Splits the nav label map into angular sectors, scores each by how far a
connected navigable corridor reaches up the image (bottom rows = ground near
the robot), then emits a body-frame twist. In-place rotation handles large
heading offsets — the differential-drive advantage.
"""

import math
import time

import numpy as np

from .proto.bebop.runtime.v1 import bebop_runtime_pb2 as pb


class SectorPlanner:
    def __init__(self, num_sectors=15, v_max=0.4, wz_max=1.2, wz_turn=1.8,
                 row_coverage=0.85, near_zone=0.18, center_bias=1.5,
                 turn_threshold=0.5, min_clearance=0.12, deadman_s=0.5):
        self.num_sectors = num_sectors
        self.v_max = v_max
        self.wz_max = wz_max
        self.wz_turn = wz_turn
        self.row_coverage = row_coverage
        self.near_zone = near_zone
        self.center_bias = center_bias
        self.turn_threshold = turn_threshold
        self.min_clearance = min_clearance
        self.deadman_s = deadman_s
        self._search_dir = 1.0

    def compute(self, label_map):
        """label_map: (H, W) uint8 nav labels. Returns (vx, wz, info)."""
        h, w = label_map.shape
        drivable = (label_map == 1).astype(np.uint8)

        k = self.num_sectors
        band_w = math.ceil(w / k)
        wpad = band_w * k
        padded = np.zeros((h, wpad), dtype=np.uint8)
        padded[:, :w] = drivable
        frac = padded.reshape(h, k, band_w).mean(axis=2)  # (H, K)
        row_ok = frac >= self.row_coverage
        consec = np.cumprod(row_ok[::-1], axis=0)
        clearance = consec.sum(axis=0) / h  # (K,) in [0, 1]

        near = drivable[int(h * (1.0 - self.near_zone)):, w // 3: 2 * w // 3]
        near_clear = near.mean() >= self.row_coverage

        offsets = (np.arange(k) - (k - 1) / 2) / ((k - 1) / 2)  # (K,) in [-1, 1], + right
        scores = (clearance ** 1.5) * (1.0 + self.center_bias * np.exp(-(offsets ** 2) / 0.18))
        best = int(np.argmax(scores))
        s = float(offsets[best])
        c = float(clearance[best])

        if c < self.min_clearance:
            return 0.0, self.wz_turn * self._search_dir, {
                "state": "search", "clearance": round(c, 2)}

        if not near_clear:
            self._search_dir = math.copysign(1.0, s)
            return 0.0, -math.copysign(self.wz_turn, s), {
                "state": "rotate", "clearance": round(c, 2), "target": round(s, 2)}

        if abs(s) > self.turn_threshold:
            return 0.0, -math.copysign(self.wz_turn, s), {
                "state": "rotate", "clearance": round(c, 2), "target": round(s, 2)}

        v = self.v_max * float(np.clip(c / 0.5, 0.0, 1.0))
        wz = -s * self.wz_max
        return v, wz, {"state": "drive", "clearance": round(c, 2), "target": round(s, 2)}


class DriveNode:
    """Pipeline frame_sink -> planner -> robot commands, at a fixed rate."""

    def __init__(self, robot, planner=None, command_hz=10.0, require_mode=pb.MODE_RUN_POLICY):
        self.robot = robot
        self.planner = planner or SectorPlanner()
        self.require_mode = require_mode
        self._interval = 1.0 / command_hz
        self._last_cmd_ts = 0.0
        self._last_nav_ts = 0.0
        self._last_label = None
        self.last_info = {}

    def on_frame(self, frame, results, stats):
        nav = (results or {}).get("nav")
        if nav is not None and nav[0] is not None:
            self._last_label = nav[0]
            self._last_nav_ts = time.monotonic()
        now = time.monotonic()
        if now - self._last_cmd_ts < self._interval:
            return
        self._last_cmd_ts = now
        vx, wz, info = self._decide(now)
        self.last_info = info
        self.robot.send_twist(vx, wz)

    def _decide(self, now):
        if self._last_label is None or now - self._last_nav_ts > self.planner.deadman_s:
            return 0.0, 0.0, {"state": "waiting"}
        st = self.robot.state
        if not st.connected:
            return 0.0, 0.0, {"state": "disconnected"}
        if st.estop_latched:
            return 0.0, 0.0, {"state": "estop", "reason": st.estop_reason}
        if self.require_mode is not None and st.mode != self.require_mode:
            return 0.0, 0.0, {"state": "hold",
                               "mode": pb.Mode.Name(st.mode)}
        return self.planner.compute(self._last_label)

    def stop(self):
        self.robot.send_twist(0.0, 0.0)