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
    w_min: float = 0.35         # rotate-in-place floor (carpet friction)
    k_heading: float = 1.8
    reach_tol_m: float = 0.08
    turn_tol_rad: float = 0.07      # ~4 deg, rotate-only goals
    # Clearances are measured from the body origin (mast); the footprint
    # front is ~0.32 m ahead, so 0.60 keeps ~0.28 m between the bumper and
    # the nearest return before forward motion stops.
    stop_m: float = 0.60
    slow_m: float = 1.30
    rotate_deg: float = 45.0
    goal_sigma_deg: float = 20.0
    goal_bias: float = 2.0
    turn_penalty: float = 0.4    # score factor lost at the edge of the FOV
    # Rotating in place sweeps the footprint corners through arcs; a turn is
    # refused only when a return sits inside an arc a corner will actually
    # sweep (see `sweep_blocked`), with this margin on the corner radius.
    # `turn_floor_m` is a hard floor: anything closer than this anywhere in
    # view refuses every turn.
    footprint_m: tuple = (-0.13, 0.32, -0.34, 0.19)
    sweep_margin_m: float = 0.04
    turn_floor_m: float = 0.28
    # Escape when blocked: turn toward the most open sector whose sweep is
    # clear, or back up this far once per goal, before giving up.
    escape: bool = True
    escape_reverse_m: float = 0.30
    escape_min_clear_m: float = 0.85
    escape_timeout_s: float = 4.0
    arc_margin_rad: float = 0.08      # about half a sector on each arc end
    # Commitment: an escape turn is driven to its heading as one unit, and
    # sector steering keeps its side for this long, so the choice does not
    # flip every tick as the view rotates (the robot used to oscillate).
    commit_s: float = 3.0
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


def sweep_blocked(cmap: ClearanceMap, delta_rad: float, footprint_m,
                  margin_m: float = 0.06, floor_m: float = 0.30,
                  arc_margin_rad: float = 0.10) -> bool:
    """Would rotating in place by `delta_rad` (+ left) hit a return?

    Each footprint corner at polar (b_c, r_c) sweeps the arc from b_c to
    b_c + delta at radius r_c.  A sector return at (b, r) blocks the turn
    when r < r_c + margin and b lies on that arc.  Returns outside every
    arc do not block, so the robot may turn away from a wall it is facing.
    Anything closer than `floor_m` blocks regardless.
    """
    obst = cmap.obstacle_m
    if np.any(obst < floor_m):
        return True
    x0, x1, y0, y1 = footprint_m
    corners = [(x1, y1), (x1, y0), (x0, y1), (x0, y0)]
    for cx, cy in corners:
        bc = math.atan2(cy, cx)
        rc = math.hypot(cx, cy) + margin_m
        for b, r in zip(cmap.bearings_rad, obst):
            if not math.isfinite(r) or r >= rc:
                continue
            # Is bearing b on the arc from bc to bc + delta?
            rel = wrap_pi(float(b) - bc)
            if delta_rad >= 0:
                if -arc_margin_rad <= rel <= delta_rad + arc_margin_rad:
                    return True
            else:
                if delta_rad - arc_margin_rad <= rel <= arc_margin_rad:
                    return True
    return False


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
    # Rotate-only goal: turn in place to this absolute heading (rad).
    rotate_to: float | None = None

    @classmethod
    def turn(cls, odom, delta_rad: float, label: str = "") -> "Goal":
        ox, oy, th = odom
        return cls(ox, oy, label, rotate_to=wrap_pi(th + delta_rad))

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
        self._escape = None              # ("reverse", start_odom, dist_m)
        self._escape_t0 = None
        self._escapes_used = 0
        self._turn_commit = None         # (target_heading, deadline)
        self._side_commit = None         # (+1 | -1, deadline)

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
        self._escape = None
        self._escapes_used = 0
        self._turn_commit = None
        self._side_commit = None
        return goal

    def clear(self, state: str = "stopped") -> None:
        self.goal = None
        self._blocked_since = None
        self._escape = None
        self._turn_commit = None
        self._side_commit = None
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
        if self.goal.rotate_to is not None:
            return self._turn_step(odom, cmap, now)
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
        detail["min_obstacle_m"] = (None if not math.isfinite(min_obstacle)
                                    else round(min_obstacle, 3))

        def can_turn(delta):
            return not sweep_blocked(cmap, delta, c.footprint_m,
                                     c.sweep_margin_m, c.turn_floor_m,
                                     c.arc_margin_rad)

        # --- an escape in progress (reverse / committed turn) ---------------
        esc = self._escape_step(odom, cmap, now, detail)
        if esc is not None:
            return esc
        if self._turn_commit is not None:
            target, deadline = self._turn_commit
            err = wrap_pi(target - th)
            if abs(err) <= c.turn_tol_rad or now > deadline:
                self._turn_commit = None
            elif can_turn(err):
                detail["escape"] = f"turn to {math.degrees(target):+.0f} deg"
                return self._rotate(err, detail)
            else:
                self._turn_commit = None     # the committed side closed up

        # --- pick a heading -------------------------------------------------
        if abs(goal_b) > half_fov:
            # Goal outside the depth field of view: turn toward it first.
            if not can_turn(goal_b):
                return self._escape_or_block(odom, cmap, now, goal_b,
                                             "turn_blocked", detail, can_turn)
            return self._rotate(goal_b, detail)

        goal_clear = cmap.clearance_at(goal_b)
        if cmap.drop:
            # A drop in the corridor is never driven across.  Turning away
            # toward an open sector (drops count as blocks in the sector
            # map) or backing up is allowed; creeping forward is not.
            detail["drop"] = True
            return self._escape_or_block(odom, cmap, now, goal_b,
                                         "drop_in_corridor", detail, can_turn)
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
                if (abs(goal_b) > math.radians(c.rotate_deg)
                        and can_turn(goal_b)):
                    # Nothing ahead is clear but the goal is well off to
                    # the side: rotate to look that way.
                    return self._rotate(goal_b, detail)
                return self._escape_or_block(odom, cmap, now, goal_b,
                                             "no_clear_sector", detail,
                                             can_turn)
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
            if self._side_commit is not None:
                side, deadline = self._side_commit
                same = eligible & (np.sign(cmap.bearings_rad) == side)
                if now > deadline or not np.any(same):
                    self._side_commit = None
                else:
                    eligible = same
            score = np.where(eligible, useful ** 1.5 * w * turn_pen, -1.0)
            i = int(np.argmax(score))
            chosen = float(cmap.bearings_rad[i])
            chosen_clear = float(cmap.clearance_m[i])
            if abs(chosen) >= math.radians(15.0) and self._side_commit is None:
                self._side_commit = (float(np.sign(chosen)), now + c.commit_s)
            detail["mode"] = "sector"
        detail["heading_deg"] = round(math.degrees(chosen), 1)
        detail["clearance_m"] = round(chosen_clear, 3)

        # --- act --------------------------------------------------------------
        if abs(chosen) > math.radians(c.rotate_deg):
            if not can_turn(chosen):
                return self._escape_or_block(odom, cmap, now, goal_b,
                                             "turn_blocked", detail, can_turn)
            # Commit to the whole rotation: finish turning to that heading
            # before re-choosing, so a 180 is one confident turn.
            self._turn_commit = (wrap_pi(th + chosen), now + c.commit_s * 2)
            return self._rotate(chosen, detail)

        detail["corridor_min_m"] = (None if not math.isfinite(cmap.corridor_min_m)
                                    else round(cmap.corridor_min_m, 3))
        if cmap.drop:
            detail["drop"] = True
        if not corridor_ok and abs(chosen) < math.radians(15.0):
            # The chassis-width corridor is blocked and the chosen heading
            # is essentially straight: do not creep forward.
            return self._escape_or_block(odom, cmap, now, goal_b,
                                         "corridor_blocked", detail, can_turn)
        if chosen_clear < c.stop_m:
            return self._escape_or_block(odom, cmap, now, goal_b,
                                         "too_close", detail, can_turn)

        self._blocked_since = None
        speed = c.v_max * min(1.0, (chosen_clear - c.stop_m)
                              / max(1e-6, c.slow_m - c.stop_m))
        speed *= min(1.0, max(0.3, dist / c.approach_m))
        vx = speed * max(0.0, math.cos(chosen))
        wz = max(-c.w_max, min(c.w_max, c.k_heading * chosen))
        self.last = Command(round(vx, 4), round(wz, 4), "drive", detail)
        return self.last

    def _turn_step(self, odom, cmap, now) -> Command:
        """Rotate in place to the goal heading; the same sweep-clearance
        rule as any other in-place turn, and the depth map must be live."""
        c = self.config
        err = wrap_pi(self.goal.rotate_to - odom[2])
        detail = {"turn_remaining_deg": round(math.degrees(err), 1)}
        esc = self._escape_step(odom, cmap, now, detail)
        if esc is not None:
            return esc
        if abs(err) <= c.turn_tol_rad:
            self.goal = None
            self.last = Command(0.0, 0.0, "reached", detail)
            return self.last
        if cmap is None:
            return self._hold("no_depth", detail)
        if cmap.age_s(now) > c.max_map_age_s:
            return self._hold("stale_depth", detail)
        if sweep_blocked(cmap, err, c.footprint_m, c.sweep_margin_m,
                         c.turn_floor_m, c.arc_margin_rad):
            detail["min_obstacle_m"] = round(float(np.min(cmap.obstacle_m)), 3)
            # Back up once to free the sweep, then retry on later ticks.
            esc = self._start_reverse_escape(odom, detail)
            if esc is not None:
                return esc
            return self._blocked(now, "turn_blocked", detail)
        return self._rotate(err, detail)

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

    # -- escapes ------------------------------------------------------------------

    def _escape_or_block(self, odom, cmap, now, goal_b, reason, detail,
                         can_turn) -> Command:
        """Blocked while driving: turn toward the most open sector whose
        sweep is clear, else back up once, else count toward the abort."""
        c = self.config
        if c.escape:
            best = None
            for b, cl in zip(cmap.bearings_rad, cmap.clearance_m):
                b = float(b)
                if cl < c.escape_min_clear_m or abs(b) < math.radians(12.0):
                    continue
                score = float(cl) - 0.4 * abs(wrap_pi(b - goal_b))
                if (best is None or score > best[0]) and can_turn(b):
                    best = (score, b, float(cl))
            if best is not None:
                detail["escape"] = f"turn {math.degrees(best[1]):+.0f} deg"
                self._blocked_since = None
                self._turn_commit = (wrap_pi(odom[2] + best[1]),
                                     now + c.commit_s * 2)
                self._side_commit = (float(np.sign(best[1])), now + c.commit_s)
                return self._rotate(best[1], detail)
            esc = self._start_reverse_escape(odom, detail)
            if esc is not None:
                return esc
        return self._blocked(now, reason, detail)

    def _start_reverse_escape(self, odom, detail) -> Command | None:
        c = self.config
        if not c.escape or self._escapes_used >= 1:
            return None
        self._escapes_used += 1
        self._escape = ("reverse", tuple(odom), c.escape_reverse_m)
        self._escape_t0 = None
        self._blocked_since = None
        detail["escape"] = f"reverse {c.escape_reverse_m} m"
        return self._reverse_cmd(odom, detail)

    def _escape_step(self, odom, cmap, now, detail) -> Command | None:
        if self._escape is None:
            return None
        if self._escape_t0 is None:
            self._escape_t0 = now
        if now - self._escape_t0 > self.config.escape_timeout_s:
            self._escape = None            # did not get anywhere: give up
            return None
        _kind, start, dist = self._escape
        x, y, th = odom
        travelled = -((x - start[0]) * math.cos(start[2])
                      + (y - start[1]) * math.sin(start[2]))
        if travelled >= dist - 0.03:
            self._escape = None
            return None
        detail["escape"] = f"reverse {dist} m ({travelled:.2f})"
        return self._reverse_cmd(odom, detail, start_th=start[2])

    def _reverse_cmd(self, odom, detail, start_th=None) -> Command:
        c = self.config
        hold = 0.0 if start_th is None else wrap_pi(start_th - odom[2])
        self.last = Command(-c.reverse_v, max(-0.4, min(0.4, 1.5 * hold)),
                            "escape", detail)
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
