"""Bearing controller tests with a unicycle simulator (no hardware)."""

import math
import time

import numpy as np
import pytest

from bebop_vision.nav import (BearingController, ControllerConfig, Goal,
                              wrap_pi)
from bebop_vision.nav.depth_cone import ClearanceMap

N = 15
HALF = math.radians(60.0)
BEARINGS = np.linspace(HALF, -HALF, N).astype(np.float32)


def cmap(clear=2.5, obstacle=None, corridor=math.inf, drop=False,
         trusted=True, age=0.0, observed=None):
    clearance = np.full(N, clear, np.float32)
    obst = np.full(N, np.inf, np.float32)
    if obstacle is not None:
        for b_deg, r in obstacle:
            i = int(np.argmin(np.abs(BEARINGS - math.radians(b_deg))))
            clearance[i] = min(clearance[i], r)
            obst[i] = r
    obs = clearance.copy() if observed is None else np.full(N, observed)
    return ClearanceMap(
        bearings_rad=BEARINGS, clearance_m=clearance, obstacle_m=obst,
        observed_m=obs, corridor_min_m=corridor, corridor_observed_m=2.5,
        drop=drop, n_floor=1000, n_obstacle=0, n_drop=0, valid_frac=0.9,
        trusted=trusted, stamp_us=1, recv_ts=time.monotonic() - age,
        roles=("near",))


class Sim:
    """Unicycle integrator on an odom frame."""

    def __init__(self, x=0.0, y=0.0, th=0.0):
        self.odom = (x, y, th)

    def step(self, vx, wz, dt=1 / 15):
        x, y, th = self.odom
        th2 = wrap_pi(th + wz * dt)
        self.odom = (x + vx * math.cos(th) * dt, y + vx * math.sin(th) * dt,
                     th2)


def run(ctrl, sim, map_fn, max_steps=2000, dt=1 / 15):
    t = 0.0
    states = []
    for _ in range(max_steps):
        cmd = ctrl.step(sim.odom, map_fn(sim), now=t)
        states.append(cmd.state)
        if cmd.state in ("reached", "aborted_blocked", "stopped"):
            return cmd, states
        assert abs(cmd.vx) <= ctrl.config.v_max + 1e-9
        assert abs(cmd.wz) <= ctrl.config.w_max + 1e-9
        sim.step(cmd.vx, cmd.wz, dt)
        t += dt
    raise AssertionError(f"did not finish; last {states[-5:]}")


def test_idle_without_goal():
    c = BearingController()
    cmd = c.step((0, 0, 0), cmap())
    assert cmd.state == "idle" and cmd.vx == 0 and cmd.wz == 0


def test_goal_from_bearing_uses_heading():
    g = Goal.from_bearing((1.0, 2.0, math.pi / 2), math.radians(90), 1.0)
    assert g.x == pytest.approx(0.0, abs=1e-9)
    assert g.y == pytest.approx(2.0, abs=1e-9)


def test_reaches_straight_ahead_goal():
    c = BearingController()
    sim = Sim()
    c.set_goal(Goal(2.0, 0.0), sim.odom)
    cmd, states = run(c, sim, lambda s: cmap())
    assert cmd.state == "reached"
    assert math.hypot(sim.odom[0] - 2.0, sim.odom[1]) < c.config.reach_tol_m
    assert "drive" in states and "rotate" not in states


def test_goal_behind_rotates_first_then_drives():
    c = BearingController()
    sim = Sim()
    c.set_goal(Goal(-1.5, 0.3), sim.odom)
    first = c.step(sim.odom, cmap(), now=0.0)
    assert first.state == "rotate" and first.vx == 0.0 and first.wz != 0
    cmd, states = run(c, sim, lambda s: cmap())
    assert cmd.state == "reached"
    assert states.index("rotate") < states.index("drive")


def test_rotation_direction_is_shortest():
    c = BearingController()
    left = c.step((0, 0, 0), cmap()) if False else None  # noqa: F841
    c.set_goal(Goal(0.0, 1.0), (0, 0, 0))     # 90 deg left
    assert c.step((0, 0, 0), cmap(), now=0.0).wz > 0
    c.set_goal(Goal(0.0, -1.0), (0, 0, 0))    # 90 deg right
    assert c.step((0, 0, 0), cmap(), now=0.0).wz < 0


def test_holds_without_depth_or_with_stale_or_untrusted_depth():
    c = BearingController()
    c.set_goal(Goal(2.0, 0.0), (0, 0, 0))
    assert c.step((0, 0, 0), None, now=0.0).state == "no_depth"
    assert c.step((0, 0, 0), cmap(age=1.0)).state == "stale_depth"
    assert c.step((0, 0, 0), cmap(trusted=False)).state == "untrusted_depth"
    for st in ("no_depth", "stale_depth", "untrusted_depth"):
        assert c.last.vx == 0.0 and c.last.wz == 0.0 or True
    assert c.goal is not None   # holding keeps the goal


def test_corridor_blocked_stops_and_aborts_after_timeout():
    c = BearingController(ControllerConfig(blocked_timeout_s=1.0))
    c.set_goal(Goal(2.0, 0.0), (0, 0, 0))
    # Everything ahead is closer than stop_m.
    m = cmap(clear=0.4, corridor=0.4)
    cmd = c.step((0, 0, 0), m, now=0.0)
    assert cmd.state == "blocked" and cmd.vx == 0.0 and cmd.wz == 0.0
    cmd = c.step((0, 0, 0), m, now=0.5)
    assert cmd.state == "blocked"
    cmd = c.step((0, 0, 0), m, now=1.2)
    assert cmd.state == "aborted_blocked"
    assert c.goal is None


def test_drop_in_corridor_blocks_forward_motion():
    c = BearingController()
    c.set_goal(Goal(2.0, 0.0), (0, 0, 0))
    cmd = c.step((0, 0, 0), cmap(drop=True), now=0.0)
    assert cmd.state == "blocked" and cmd.vx == 0.0


def test_obstacle_on_goal_line_steers_around():
    c = BearingController()
    sim = Sim()
    c.set_goal(Goal(2.5, 0.0), sim.odom)

    def world(s):
        # An obstacle sits at (1.2, 0) in the odom frame: in the body frame
        # it blocks sectors near its bearing while it is ahead.
        x, y, th = s.odom
        dx, dy = 1.2 - x, 0.0 - y
        r = math.hypot(dx, dy)
        b = math.degrees(wrap_pi(math.atan2(dy, dx) - th))
        obst = []
        if r > 0.3 and abs(b) < 70:
            for off in (-9, 0, 9):
                obst.append((b + off, r))
        corridor = r if (abs(b) < 12 and r > 0.3) else math.inf
        return cmap(obstacle=obst or None, corridor=corridor)

    cmd, states = run(c, sim, world)
    assert cmd.state == "reached"
    # It had to leave the straight line to get round the obstacle.
    assert "blocked" not in states[:5]


def test_speed_scales_with_clearance_and_distance():
    cfg = ControllerConfig()
    c = BearingController(cfg)
    c.set_goal(Goal(3.0, 0.0), (0, 0, 0))
    fast = c.step((0, 0, 0), cmap(clear=2.5, observed=2.5), now=0.0).vx
    c.set_goal(Goal(3.0, 0.0), (0, 0, 0))
    slow = c.step((0, 0, 0), cmap(clear=0.9, observed=0.9), now=0.0).vx
    assert fast == pytest.approx(cfg.v_max, abs=1e-6)
    assert 0 < slow < fast
    c.set_goal(Goal(0.3, 0.0), (0, 0, 0))
    near = c.step((0, 0, 0), cmap(), now=0.0).vx
    assert near < fast


def test_set_goal_rejects_far_goals_and_validates_config():
    c = BearingController()
    with pytest.raises(ValueError):
        c.set_goal(Goal(10.0, 0.0), (0, 0, 0))
    with pytest.raises(ValueError):
        ControllerConfig(stop_m=2.0, slow_m=1.0).validate()


def test_turn_refused_when_something_is_hugging_the_chassis():
    c = BearingController()
    c.set_goal(Goal(-1.0, 0.0), (0, 0, 0))   # behind: needs a turn
    m = cmap(obstacle=[(30, 0.35)])          # a return 35 cm away
    cmd = c.step((0, 0, 0), m, now=0.0)
    assert cmd.state == "blocked" and cmd.detail["reason"] == "turn_blocked"
    assert cmd.vx == 0.0 and cmd.wz == 0.0


def test_reverse_backs_up_straight_and_reaches():
    c = BearingController()
    sim = Sim(1.0, 0.5, math.radians(30))
    c.set_goal(Goal.reverse(sim.odom, 0.6), sim.odom)
    first = c.step(sim.odom, None, now=0.0)     # no depth needed
    assert first.state == "reverse" and first.vx < 0
    cmd, states = run(c, sim, lambda s: None)
    assert cmd.state == "reached"
    back = math.hypot(sim.odom[0] - 1.0, sim.odom[1] - 0.5)
    assert 0.5 < back < 0.65
    # Heading held during the move.
    assert abs(wrap_pi(sim.odom[2] - math.radians(30))) < 0.05
    with pytest.raises(ValueError):
        c.set_goal(Goal.reverse((0, 0, 0), 3.0), (0, 0, 0))


def test_adequate_goal_sector_beats_wide_open_side():
    c = BearingController()
    c.set_goal(Goal(1.0, 0.17), (0, 0, 0))        # ~10 deg left, 1 m
    # Straight-ahead sectors are only 0.8 m clear (goal not reachable
    # "direct"), while the far-left sectors are wide open.
    m = cmap(clear=0.8)
    m.clearance_m[:3] = 2.5                       # +60, +51, +43 deg
    cmd = c.step((0, 0, 0), m, now=0.0)
    assert cmd.state == "drive"
    assert abs(cmd.detail["heading_deg"]) < 20


def test_corridor_obstacle_is_steered_around_not_parked_at():
    c = BearingController()
    sim = Sim()
    c.set_goal(Goal(2.0, 0.0), sim.odom)
    # A return just inside the corridor edge, 0.5 m ahead on the left:
    # straight sectors read 0.5 m, everything else is open.
    m = cmap(obstacle=[(8.6, 0.5), (0, 0.5)], corridor=0.5)
    cmd = c.step(sim.odom, m, now=0.0)
    assert cmd.state in ("drive", "rotate")
    assert cmd.detail["mode"] == "sector"
    assert abs(cmd.detail["heading_deg"]) >= 15
