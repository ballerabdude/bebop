"""Goal-conditioned polar planner + drive node (navd Phase A, Section 6.4).

A generalization of `SectorPlanner` (planner.py): the same clearance^1.5 *
(1 + bias * exp(-dpsi^2/0.18)) scoring curve, with the image-center bias
replaced by a goal bias, and the 15 vertical image bands replaced by 13
polar rays (-60..+60 deg) marched through the BEV occupancy grid.

Goals are either a body-frame heading offset (`GoalHeading`) or an odom
waypoint (`GoalPoint`, resolved to a body-frame bearing each tick from
DriveState odometry). States mirror SectorPlanner: drive / rotate /
search, plus a hard stop when the near cone is blocked and "reached" for
waypoints. All limits default to the SectorPlanner values.

Sign conventions: body x forward, y left; wz + = left (matches
RobotClient.send_twist). dpsi = ray_angle - goal_bearing; rotating with
wz = k * dpsi moves a world-fixed corridor onto the goal direction.
"""

import dataclasses
import math
import time

import numpy as np

from .bev import OCCUPIED
from .planner import DriveNode
from .proto.bebop.runtime.v1 import bebop_runtime_pb2 as pb


@dataclasses.dataclass
class GoalHeading:
    """Body-frame heading offset (rad, + left). Never 'reaches'."""
    heading_rad: float


@dataclasses.dataclass
class GoalPoint:
    """Odom-frame waypoint (m). 'reached' within goal_reach_m."""
    x: float
    y: float


class GoalSlot:
    """Latest-wins goal slot shared between the stdin reader and planner."""

    def __init__(self):
        self._goal = None

    def set(self, goal):
        self._goal = goal

    def clear(self):
        self._goal = None

    def get(self):
        return self._goal


def parse_goal(line):
    """Parse a bench stdin goal command: 'heading <deg>' | 'xy <x> <y>' | 'stop'."""
    parts = line.strip().lower().split()
    if not parts:
        return None
    if parts[0] == "stop":
        return "stop"
    if parts[0] == "heading" and len(parts) == 2:
        return GoalHeading(math.radians(float(parts[1])))
    if parts[0] == "xy" and len(parts) == 3:
        return GoalPoint(float(parts[1]), float(parts[2]))
    raise ValueError(f"bad goal command: {line!r} (use 'heading <deg>', "
                     f"'xy <x> <y>' or 'stop')")


def _wrap(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class GoalPlanner:
    def __init__(self, num_rays=13, fov_deg=120.0, v_max=0.4, wz_max=1.2,
                 wz_turn=1.8, goal_bias=1.5, k_yaw=2.0, turn_threshold=0.5,
                 min_clearance=0.12, r_min=0.35, goal_reach_m=0.3,
                 near_cone_deg=20.0, near_cone_r=0.5):
        self.num_rays = num_rays
        self.angles = np.radians(np.linspace(
            -fov_deg / 2.0, fov_deg / 2.0, num_rays))
        self.v_max = v_max
        self.wz_max = wz_max
        self.wz_turn = wz_turn
        self.goal_bias = goal_bias
        self.k_yaw = k_yaw
        self.turn_threshold = turn_threshold
        self.min_clearance = min_clearance
        self.r_min = r_min
        self.goal_reach_m = goal_reach_m
        self.near_cone_deg = near_cone_deg
        self.near_cone_r = near_cone_r
        self._search_dir = 1.0
        self._searching = False

    def compute(self, grid, goal, odom=(0.0, 0.0, 0.0)):
        """grid: BevGrid; goal: GoalHeading | GoalPoint | None.

        Returns (vx, wz, info). goal None -> (0, 0, waiting).
        """
        if goal is None:
            return 0.0, 0.0, {"state": "waiting"}
        if isinstance(goal, GoalPoint):
            bearing = _wrap(math.atan2(goal.y - odom[1], goal.x - odom[0]) - odom[2])
            dist = math.hypot(goal.x - odom[0], goal.y - odom[1])
            if dist < self.goal_reach_m:
                return 0.0, 0.0, {"state": "reached", "dist": round(dist, 2)}
        else:
            bearing = _wrap(goal.heading_rad)
            dist = None

        clear_raw, r_max = self._march_clearances(grid)
        clear_norm = clear_raw / r_max

        dpsi = _wrap(self.angles - bearing)
        scores = (clear_norm ** 1.5) * (
            1.0 + self.goal_bias * np.exp(-(dpsi ** 2) / 0.18))
        best = int(np.argmax(scores))
        c_best = float(clear_norm[best])
        dpsi_best = float(dpsi[best])

        info = {"state": "drive", "clearance": round(c_best, 2),
                "dpsi": round(math.degrees(dpsi_best), 1),
                "bearing": round(math.degrees(bearing), 1)}
        if dist is not None:
            info["dist"] = round(dist, 2)

        # State priority mirrors SectorPlanner: search first (nothing usable
        # in the whole span), then rotate (pivot toward the best corridor —
        # safe even with the near cone blocked: the pivot turns AWAY from
        # the blockage and its sweep radius is the robot's own footprint),
        # then drive with the near-cone hard stop as its gate (do not
        # advance into something). Hard-stopping before rotate deadlocked
        # the robot facing a small obstacle with a clear corridor beside it
        # (bench run 2, 2026-09-06).
        if self._searching:
            if c_best >= self.min_clearance + 0.05:
                self._searching = False
            else:
                info["state"] = "search"
                return 0.0, self.wz_turn * self._search_dir, info

        if c_best < self.min_clearance:
            self._searching = True
            if abs(bearing) > 1e-3:
                self._search_dir = math.copysign(1.0, bearing)
            info["state"] = "search"
            return 0.0, self.wz_turn * self._search_dir, info

        cone = np.abs(self.angles) <= math.radians(self.near_cone_deg)
        cone_blocked = bool(np.any(cone & (clear_raw < self.near_cone_r)))
        best_near_blocked = bool(clear_raw[best] < self.near_cone_r)

        if cone_blocked and best_near_blocked:
            # Boxed: the near cone AND the best corridor are both
            # close-blocked — no useful pivot exists, stand still.
            info["state"] = "hard_stop"
            return 0.0, 0.0, info

        if abs(dpsi_best) > self.turn_threshold:
            info["state"] = "rotate"
            return 0.0, float(np.clip(self.k_yaw * dpsi_best,
                                      -self.wz_max, self.wz_max)), info

        if cone_blocked:
            # About to advance with something inside the near cone: stand.
            info["state"] = "hard_stop"
            return 0.0, 0.0, info

        if best_near_blocked:
            # Best corridor itself is near-blocked (outside the cone):
            # rotate toward it per spec.
            info["state"] = "rotate"
            return 0.0, float(np.clip(self.k_yaw * dpsi_best,
                                      -self.wz_max, self.wz_max)), info

        vx = self.v_max * float(np.clip(c_best / 0.5, 0.0, 1.0))
        wz = float(np.clip(self.k_yaw * dpsi_best, -self.wz_max, self.wz_max))
        return vx, wz, info

    def _march_clearances(self, grid):
        """First blocked range along each ray on the inflated grid.

        Returns (clearances_m, r_max). Out-of-grid ray samples count as
        clear — the grid is the planner's horizon (plan Section 6.4).
        """
        occ = grid.occ
        rows, cols = occ.shape
        cell = grid.cell_m
        range_m = rows * cell
        width_m = cols * cell
        step = cell / 2.0
        rs = np.arange(self.r_min, range_m + step / 2, step, dtype=np.float32)
        x = np.cos(self.angles)[:, None] * rs[None, :]
        y = np.sin(self.angles)[:, None] * rs[None, :]
        r_idx = ((range_m - x) / cell).astype(np.int32)
        c_idx = ((y + width_m / 2.0) / cell).astype(np.int32)
        inb = (r_idx >= 0) & (r_idx < rows) & (c_idx >= 0) & (c_idx < cols)
        blocked = np.zeros(x.shape, dtype=bool)
        blocked[inb] = occ[r_idx[inb], c_idx[inb]] > 0
        hit = np.where(blocked, rs[None, :], np.inf)
        first = hit.argmin(axis=1)
        best_hit = hit[np.arange(len(first)), first]
        return np.where(np.isfinite(best_hit), rs[first], range_m), range_m


class GoalDriveNode(DriveNode):
    """DriveNode gating (mode / estop / deadman) over BEV grids.

    Shared with the legacy pipeline: same rate limiter, same zero-twist
    semantics, same mode gate. Adds the navd-specific floor gate: both
    cameras' plane fits failing for > plane_fail_s -> zero twist.
    """

    def __init__(self, robot, planner, grid_provider, goal_slot=None,
                 command_hz=10.0, require_mode=None, plane_fail_s=1.0,
                 deadman_s=0.5):
        super().__init__(robot, planner, command_hz, require_mode)
        self.grid_provider = grid_provider
        self.goal_slot = goal_slot
        self.plane_fail_s = plane_fail_s
        self.deadman_s = deadman_s
        self._plane_fail_since = None

    def on_grid(self):
        """Tick the node: latest grid -> gates -> twist. Call at command_hz."""
        now = time.monotonic()
        if now - self._last_cmd_ts < self._interval:
            return
        self._last_cmd_ts = now
        vx, wz, info = self._decide(now)
        self.last_info = info
        self.robot.send_twist(vx, wz)

    def _decide(self, now):
        grid = self.grid_provider()
        if grid is None or now - grid.recv_ts > self.deadman_s:
            return 0.0, 0.0, {"state": "waiting"}
        st = self.robot.state
        if not st.connected:
            return 0.0, 0.0, {"state": "disconnected"}
        if st.estop_latched:
            return 0.0, 0.0, {"state": "estop", "reason": st.estop_reason}
        if self.require_mode is not None and st.mode != self.require_mode:
            return 0.0, 0.0, {"state": "hold", "mode": pb.Mode.Name(st.mode)}
        if grid.plane_ok and not any(grid.plane_ok.values()):
            if self._plane_fail_since is None:
                self._plane_fail_since = now
            elif now - self._plane_fail_since >= self.plane_fail_s:
                return 0.0, 0.0, {"state": "no_floor"}
        else:
            self._plane_fail_since = None
        goal = self.goal_slot.get() if self.goal_slot is not None else None
        return self.planner.compute(grid, goal, odom=st.odom)
