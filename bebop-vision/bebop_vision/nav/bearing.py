"""Bearing controller: drive to an odom-frame waypoint through the
clearance map, one twist per tick.

The goal is set as a bearing + distance from the robot's pose at that
moment (or directly as odom x, y) and then closed on wheel odometry.
Each tick picks a steering direction from the `ClearanceMap`:

* if the goal itself is inside the observed clearance along its bearing,
  head straight for it;
* otherwise score every sector by clearance, biased toward the goal
  bearing, and head for the best one (simple reactive avoidance);
* a sector is eligible only when its clearance exceeds `stop_m`, and
  forward motion additionally needs the chassis-width corridor clear.

Large heading errors rotate in place; small ones drive with speed scaled
by clearance and by the remaining distance.  No clearance map, a stale
one, or an untrusted frame holds the robot still.  Being blocked for
longer than `blocked_timeout_s` aborts the goal.

Pure logic: no threads, sockets, or robot calls.  `drive.py` owns those.
"""
from __future__ import annotations

import dataclasses
import math
import time

import numpy as np

from .depth_cone import ClearanceMap


def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


@dataclasses.dataclass
class ControllerConfig:
    v_max: float = 0.30
    w_max: float = 0.9
    w_min: float = 0.2          # rotate-in-place floor (static friction)
    k_heading: float = 1.8
    reach_tol_m: float = 0.08
    # Clearances are measured from the body origin (mast); the footprint
    # front is ~0.32 m ahead, so 0.60 keeps ~0.28 m between the bumper and
    # the nearest return before forward motion stops.
    stop_m: float = 0.60
    slow_m: float = 1.30
    rotate_deg: float = 45.0
    goal_sigma_deg: float = 20.0
    goal_bias: float = 2.0
    turn_penalty: float = 0.4    # score factor lost at the edge of the FOV
    # Rotating in place sweeps the footprint corners (~0.47 m from the
    # mast); refuse when any sector has a return closer than this.
    turn_clearance_m: float = 0.50
    max_map_age_s: float = 0.4
    blocked_timeout_s: float = 4.0
    max_goal_m: float = 6.0
    approach_m: float = 0.5      # start slowing this far from the goal
    # Straight reverse (the rear is unobserved): slow, short, heading held.
    reverse_v: float = 0.15
    max_reverse_m: float = 1.0

    def validate(self) -> None:
        if not (0 < self.v_max <= 1.0):
            raise ValueError("v_max must be in (0, 1] m/s")
        if not (0 < self.w_max <= 2.0):
            raise ValueError("w_max must be in (0, 2] rad/s")
        if self.stop_m >= self.slow_m:
            raise ValueError("stop_m must be < slow_m")
        if self.reach_tol_m <= 0:
            raise ValueError("reach_tol_m must be > 0")


@dataclasses.dataclass(frozen=True)
class Goal:
    """An odom-frame target point.

    `reverse_m` > 0 makes it a straight backward move of that length
    along the heading at set time (x, y then mark the end point).
    """

    x: float
    y: float
    label: str = ""
    reverse_m: float = 0.0
    start_x: float = 0.0
    start_y: float = 0.0
    start_th: float = 0.0

    @classmethod
    def reverse(cls, odom, distance_m: float, label: str = "") -> "Goal":
        ox, oy, th = odom
        return cls(ox - distance_m * math.cos(th), oy - distance_m * math.sin(th),
                   label, reverse_m=float(distance_m),
                   start_x=ox, start_y=oy, start_th=th)

    @classmethod
    def from_bearing(cls, odom, bearing_rad: float, distance_m: float,
                     label: str = "") -> "Goal":
        ox, oy, th = odom
        h = th + bearing_rad
        return cls(ox + distance_m * math.cos(h),
                   oy + distance_m * math.sin(h), label)


@dataclasses.dataclass
class Command:
    vx: float
    wz: float
    state: str
    detail: dict = dataclasses.field(default_factory=dict)


TERMINAL_STATES = ("reached", "aborted_blocked", "stopped")


class BearingController:
    def __init__(self, config: ControllerConfig | None = None):
        self.config = config or ControllerConfig()
        self.config.validate()
        self.goal: Goal | None = None
        self._blocked_since: float | None = None
        self.last: Command = Command(0.0, 0.0, "idle")

    # -- goal management ------------------------------------------------------

    def set_goal(self, goal: Goal, odom) -> Goal:
        d = math.hypot(goal.x - odom[0], goal.y - odom[1])
        if goal.reverse_m > 0 and goal.reverse_m > self.config.max_reverse_m:
            raise ValueError(
                f"reverse {goal.reverse_m:.2f} m exceeds max_reverse_m "
                f"{self.config.max_reverse_m}")
        if d > self.config.max_goal_m:
            raise ValueError(
                f"goal {d:.2f} m away exceeds max_goal_m "
                f"{self.config.max_goal_m}")
        self.goal = goal
        self._blocked_since = None
        return goal

    def clear(self, state: str = "stopped") -> None:
        self.goal = None
        self._blocked_since = None
        self.last = Command(0.0, 0.0, state)

    # -- one tick -------------------------------------------------------------

    def step(self, odom, cmap: ClearanceMap | None, now=None) -> Command:
        c = self.config
        now = time.monotonic() if now is None else now
        if self.goal is None:
            self.last = Command(0.0, 0.0, "idle")
            return self.last
        ox, oy, th = odom
        if self.goal.reverse_m > 0:
            return self._reverse_step(odom)
        dx, dy = self.goal.x - ox, self.goal.y - oy
        dist = math.hypot(dx, dy)
        goal_b = wrap_pi(math.atan2(dy, dx) - th)
        detail = {"dist_m": round(dist, 3),
                  "goal_bearing_deg": round(math.degrees(goal_b), 1)}
        if dist <= c.reach_tol_m:
            self.goal = None
            self.last = Command(0.0, 0.0, "reached", detail)
            return self.last
        if cmap is None:
            return self._hold("no_depth", detail)
        age = cmap.age_s(now)
        if age > c.max_map_age_s:
            detail["map_age_s"] = round(age, 3)
            return self._hold("stale_depth", detail)
        if not cmap.trusted:
            detail["valid_frac"] = cmap.valid_frac
            return self._hold("untrusted_depth", detail)

        half_fov = float(cmap.bearings_rad[0])
        min_obstacle = float(np.min(cmap.obstacle_m))
        can_turn = min_obstacle >= c.turn_clearance_m
        detail["min_obstacle_m"] = (None if not math.isfinite(min_obstacle)
                                    else round(min_obstacle, 3))

        # --- pick a heading -------------------------------------------------
        if abs(goal_b) > half_fov:
            # Goal outside the depth field of view: turn toward it first.
            if not can_turn:
                return self._blocked(now, "turn_blocked", detail)
            return self._rotate(goal_b, detail)

        goal_clear = cmap.clearance_at(goal_b)
        if cmap.drop:
            # A drop in the corridor is never steered around: the near
            # camera is the only drop sensor and it only sees the corridor.
            detail["drop"] = True
            return self._blocked(now, "drop_in_corridor", detail)
        corridor_ok = cmap.corridor_min_m >= c.stop_m
        # Head straight for the goal only while the chassis-width corridor
        # is clear; otherwise fall through to sector steering so a return
        # just inside the corridor edge is driven around, not parked at.
        if dist <= goal_clear and (corridor_ok
                                   or abs(goal_b) > math.radians(15.0)):
            chosen, chosen_clear = goal_b, goal_clear
            detail["mode"] = "direct"
        else:
            eligible = cmap.clearance_m >= c.stop_m
            if not corridor_ok:
                eligible = eligible & (np.abs(cmap.bearings_rad)
                                       >= math.radians(15.0))
            if not np.any(eligible):
                if can_turn and abs(goal_b) > math.radians(c.rotate_deg):
                    # Nothing ahead is clear but the goal is well off to
                    # the side: rotate to look that way.
                    return self._rotate(goal_b, detail)
                return self._blocked(now, "no_clear_sector", detail)
            sigma = math.radians(c.goal_sigma_deg)
            w = 1.0 + c.goal_bias * np.exp(
                -0.5 * ((cmap.bearings_rad - goal_b) / sigma) ** 2)
            # Clearance beyond the goal is worthless, and a big heading
            # change costs time and a blind sweep: cap and penalise so a
            # merely-adequate sector near the goal beats a wide-open one
            # far off to the side.
            useful = np.minimum(cmap.clearance_m, dist + c.approach_m)
            turn_pen = 1.0 - c.turn_penalty * np.abs(cmap.bearings_rad) / half_fov
            if not corridor_ok:
                # Straight-ahead sectors would be refused by the corridor
                # check below; do not pick them.
                eligible = eligible & (np.abs(cmap.bearings_rad)
                                       >= math.radians(15.0))
            score = np.where(eligible, useful ** 1.5 * w * turn_pen, -1.0)
            i = int(np.argmax(score))
            chosen = float(cmap.bearings_rad[i])
            chosen_clear = float(cmap.clearance_m[i])
            detail["mode"] = "sector"
        detail["heading_deg"] = round(math.degrees(chosen), 1)
        detail["clearance_m"] = round(chosen_clear, 3)

        # --- act --------------------------------------------------------------
        if abs(chosen) > math.radians(c.rotate_deg):
            if not can_turn:
                return self._blocked(now, "turn_blocked", detail)
            return self._rotate(chosen, detail)

        detail["corridor_min_m"] = (None if not math.isfinite(cmap.corridor_min_m)
                                    else round(cmap.corridor_min_m, 3))
        if cmap.drop:
            detail["drop"] = True
        if not corridor_ok and abs(chosen) < math.radians(15.0):
            # The chassis-width corridor is blocked and the chosen heading
            # is essentially straight: do not creep forward.
            return self._blocked(now, "corridor_blocked", detail)
        if chosen_clear < c.stop_m:
            return self._blocked(now, "too_close", detail)

        self._blocked_since = None
        speed = c.v_max * min(1.0, (chosen_clear - c.stop_m)
                              / max(1e-6, c.slow_m - c.stop_m))
        speed *= min(1.0, max(0.3, dist / c.approach_m))
        vx = speed * max(0.0, math.cos(chosen))
        wz = max(-c.w_max, min(c.w_max, c.k_heading * chosen))
        self.last = Command(round(vx, 4), round(wz, 4), "drive", detail)
        return self.last

    def _reverse_step(self, odom) -> Command:
        """Back straight up along the heading held at set time; no depth
        is consulted because nothing looks backward.  Bounded by
        max_reverse_m and reverse_v."""
        c = self.config
        g = self.goal
        x, y, th = odom
        travelled = ((x - g.start_x) * math.cos(g.start_th)
                     + (y - g.start_y) * math.sin(g.start_th))
        remaining = g.reverse_m + travelled      # travelled is negative
        detail = {"reverse_m": round(g.reverse_m, 3),
                  "remaining_m": round(remaining, 3)}
        if remaining <= c.reach_tol_m:
            self.goal = None
            self.last = Command(0.0, 0.0, "reached", detail)
            return self.last
        hold = wrap_pi(g.start_th - th)
        vx = -c.reverse_v * min(1.0, max(0.3, remaining / c.approach_m))
        wz = max(-0.4, min(0.4, 1.5 * hold))
        self.last = Command(round(vx, 4), round(wz, 4), "reverse", detail)
        return self.last

    # -- helpers ----------------------------------------------------------------

    def _hold(self, state: str, detail: dict) -> Command:
        self.last = Command(0.0, 0.0, state, detail)
        return self.last

    def _rotate(self, err: float, detail: dict) -> Command:
        c = self.config
        self._blocked_since = None
        wz = c.k_heading * err
        wz = math.copysign(max(c.w_min, min(c.w_max, abs(wz))), err)
        detail["heading_deg"] = round(math.degrees(err), 1)
        self.last = Command(0.0, round(wz, 4), "rotate", detail)
        return self.last

    def _blocked(self, now: float, reason: str, detail: dict) -> Command:
        c = self.config
        if self._blocked_since is None:
            self._blocked_since = now
        detail["reason"] = reason
        detail["blocked_s"] = round(now - self._blocked_since, 2)
        if now - self._blocked_since >= c.blocked_timeout_s:
            self.goal = None
            self._blocked_since = None
            self.last = Command(0.0, 0.0, "aborted_blocked", detail)
            return self.last
        self.last = Command(0.0, 0.0, "blocked", detail)
        return self.last
