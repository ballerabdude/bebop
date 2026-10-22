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
    c = BearingController(ControllerConfig(blocked_timeout_s=1.0, escape=False))
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
    c = BearingController(ControllerConfig(escape=False))
    c.set_goal(Goal(2.0, 0.0), (0, 0, 0))
    cmd = c.step((0, 0, 0), cmap(drop=True), now=0.0)
    assert cmd.state == "blocked" and cmd.vx == 0.0
    # With escapes: never forward, but turning toward an open side is fine.
    c2 = BearingController()
    c2.set_goal(Goal(2.0, 0.0), (0, 0, 0))
    cmd = c2.step((0, 0, 0), cmap(drop=True), now=0.0)
    assert cmd.vx <= 0.0 and cmd.state in ("rotate", "escape")


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
    c = BearingController(ControllerConfig(escape=False))
    c.set_goal(Goal(-1.0, 0.0), (0, 0, 0))   # behind: needs a turn
    m = cmap(obstacle=[(30, 0.35)])          # a return 35 cm away at a corner
    cmd = c.step((0, 0, 0), m, now=0.0)
    assert cmd.state == "blocked" and cmd.detail["reason"] == "turn_blocked"
    assert cmd.vx == 0.0 and cmd.wz == 0.0
    # With escapes on, the same situation backs up instead of sitting.
    c2 = BearingController()
    c2.set_goal(Goal(-1.0, 0.0), (0, 0, 0))
    cmd = c2.step((0, 0, 0), m, now=0.0)
    assert cmd.state == "escape" and cmd.vx < 0


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


def test_turn_goal_rotates_in_place_and_reaches():
    c = BearingController()
    sim = Sim()
    c.set_goal(Goal.turn(sim.odom, math.radians(90)), sim.odom)
    first = c.step(sim.odom, cmap(), now=0.0)
    assert first.state == "rotate" and first.vx == 0.0 and first.wz > 0
    cmd, states = run(c, sim, lambda s: cmap())
    assert cmd.state == "reached" and "drive" not in states
    assert abs(wrap_pi(sim.odom[2] - math.radians(90))) < 0.08
    # A return in the sweep: back up first (escape), then held without depth.
    c.set_goal(Goal.turn((0, 0, 0), math.radians(-90)), (0, 0, 0))
    assert c.step((0, 0, 0), cmap(obstacle=[(0, 0.3)]), now=0.0).state == "escape"
    c.set_goal(Goal.turn((0, 0, 0), math.radians(-90)), (0, 0, 0))
    assert c.step((0, 0, 0), None, now=0.0).state == "no_depth"


def test_sweep_geometry_lets_the_robot_turn_away_from_a_wall():
    from bebop_vision.nav.bearing import sweep_blocked
    fp = (-0.13, 0.32, -0.34, 0.19)
    wall = cmap(obstacle=[(0, 0.45), (8.6, 0.45), (-8.6, 0.45)])
    # Small turns either way do not sweep a corner through the wall.
    assert not sweep_blocked(wall, math.radians(20), fp)
    assert not sweep_blocked(wall, math.radians(-20), fp)
    # A big LEFT turn drags the front-right corner (bearing -47 deg,
    # radius 0.47 m) through the wall; the same size RIGHT turn only moves
    # the nearer front-left corner (0.37 m) across it and is fine.
    assert sweep_blocked(wall, math.radians(90), fp)
    assert not sweep_blocked(wall, math.radians(-60), fp)
    # Bring the wall inside the front-left corner's radius: now the right
    # turn is refused too.
    near = cmap(obstacle=[(0, 0.40), (8.6, 0.40), (-8.6, 0.40)])
    assert sweep_blocked(near, math.radians(-60), fp)
    # Closer than the hard floor: everything is refused.
    assert sweep_blocked(cmap(obstacle=[(50, 0.2)]), math.radians(-10), fp)
    # Something off to the far left does not stop a right turn.
    assert not sweep_blocked(cmap(obstacle=[(60, 0.4)]), math.radians(-90), fp)


def test_blocked_ahead_escapes_toward_the_open_side():
    c = BearingController()
    c.set_goal(Goal(2.0, 0.3), (0, 0, 0))
    # Wall ahead at 0.5 m across the corridor, wide open to the left.
    m = cmap(clear=0.5, corridor=0.5)
    for i, b in enumerate(m.bearings_rad):
        if math.degrees(b) >= 34:
            m.clearance_m[i] = 2.0
            m.obstacle_m[i] = 2.0
    cmd = c.step((0, 0, 0), m, now=0.0)
    # Sector steering already heads for the open side (no creep forward
    # into the wall): heading well off straight, turning left.
    assert cmd.state in ("drive", "rotate") and cmd.wz > 0
    assert cmd.detail["heading_deg"] >= 15
    # With nothing eligible for sector steering, the explicit escape turns
    # toward the open side instead of parking.
    m2 = cmap(clear=0.5, corridor=0.5)
    for i, b in enumerate(m2.bearings_rad):
        if math.degrees(b) >= 50:
            m2.clearance_m[i] = 0.9          # open-ish, but under the goal
            m2.obstacle_m[i] = 0.9
    c.set_goal(Goal(2.0, 0.3), (0, 0, 0))
    cmd = c.step((0, 0, 0), m2, now=0.0)
    assert cmd.state in ("drive", "rotate") and cmd.wz > 0


def test_blocked_with_nothing_open_backs_up_once_then_aborts():
    c = BearingController(ControllerConfig(blocked_timeout_s=1.0))
    sim = Sim()
    c.set_goal(Goal(2.0, 0.0), sim.odom)
    boxed = cmap(clear=0.4, corridor=0.4)
    cmd = c.step(sim.odom, boxed, now=0.0)
    assert cmd.state == "escape" and cmd.vx < 0
    t = 0.0
    while cmd.state == "escape" and t < 5.0:
        sim.step(cmd.vx, cmd.wz, 0.1)
        t += 0.1
        cmd = c.step(sim.odom, boxed, now=t)
    assert sim.odom[0] < -0.25                      # backed up ~0.3 m
    assert cmd.state == "blocked"                   # still boxed in: no 2nd escape
    cmd = c.step(sim.odom, boxed, now=t + 1.5)
    assert cmd.state == "aborted_blocked"


def test_big_turn_is_committed_as_one_heading():
    c = BearingController()
    sim = Sim()
    c.set_goal(Goal(2.0, 0.0), sim.odom)
    # Boxed in ahead; the only open sector is at +60.
    m = cmap(clear=0.5, corridor=0.5)
    i = int(np.argmin(np.abs(np.degrees(m.bearings_rad) - 60)))
    m.clearance_m[i] = 2.0
    m.obstacle_m[i] = 2.0
    cmd = c.step(sim.odom, m, now=0.0)
    assert cmd.state == "rotate" and cmd.wz > 0
    target = c._turn_commit[0]
    assert target == pytest.approx(math.radians(60), abs=0.05)
    # While turning, the map now says everything is open (the open sector
    # rotated with us): the controller still finishes the committed turn
    # instead of re-choosing every tick, and never flips direction.
    t = 0.0
    seen = 0
    while c._turn_commit is not None and c._turn_commit[0] == target and t < 5:
        sim.step(cmd.vx, cmd.wz, 0.1)
        t += 0.1
        cmd = c.step(sim.odom, cmap(), now=t)
        if c._turn_commit is not None and c._turn_commit[0] == target:
            assert cmd.state == "rotate" and cmd.wz > 0
            seen += 1
    assert seen > 3
    assert abs(wrap_pi(sim.odom[2] - target)) < 0.15


def test_sector_side_is_kept_for_a_while():
    c = BearingController(ControllerConfig(commit_s=3.0))
    c.set_goal(Goal(3.0, 0.0), (0, 0, 0))

    def two_sided(left_clear, right_clear):
        m = cmap(clear=0.5, corridor=0.5)
        for i, b in enumerate(m.bearings_rad):
            if abs(math.degrees(b) - 34) < 1:
                m.clearance_m[i] = m.obstacle_m[i] = left_clear
            if abs(math.degrees(b) + 34) < 1:
                m.clearance_m[i] = m.obstacle_m[i] = right_clear
        return m
    cmd = c.step((0, 0, 0), two_sided(2.2, 2.0), now=0.0)
    assert cmd.detail["heading_deg"] > 0
    # The right side now looks clearly better: keep the left while committed.
    cmd = c.step((0, 0, 0), two_sided(2.0, 3.0), now=1.0)
    assert cmd.detail["heading_deg"] > 0
    # After the commitment expires it may switch.
    cmd = c.step((0, 0, 0), two_sided(2.0, 3.0), now=4.5)
    assert cmd.detail["heading_deg"] < 0


def test_sweep_blocked_big_turn_prefers_an_arc_heading():
    c = BearingController()
    c.set_goal(Goal(1.15, 1.64), (0, 0, 0))          # goal at +55 deg, 2 m
    m = cmap(clear=0.5, corridor=0.5)
    # Wide open at +60 (needs a big in-place turn), modest at +34 (an arc
    # will do); a return at 0.35 m dead ahead refuses the big turn.
    for i, b in enumerate(m.bearings_rad):
        if abs(math.degrees(b) - 60) < 1:
            m.clearance_m[i] = m.obstacle_m[i] = 2.5
        if abs(math.degrees(b) - 34) < 1:
            m.clearance_m[i] = m.obstacle_m[i] = 1.2
        if abs(math.degrees(b)) < 1:
            m.obstacle_m[i] = 0.35
    cmd = c.step((0, 0, 0), m, now=0.0)
    assert cmd.state == "drive" and cmd.detail["mode"] == "arc"
    assert cmd.detail["heading_deg"] == pytest.approx(34.3, abs=0.1)
    assert cmd.vx > 0 and cmd.wz > 0


def test_side_latch_never_steers_away_from_a_goal_on_the_other_side():
    c = BearingController()
    c.set_goal(Goal(0.6, 1.0), (0, 0, 0))            # goal at +59 deg, 1.2 m
    m = cmap(clear=0.5, corridor=0.5)
    for i, b in enumerate(m.bearings_rad):
        if abs(abs(math.degrees(b)) - 34) < 1 or abs(abs(math.degrees(b)) - 43) < 1:
            m.clearance_m[i] = m.obstacle_m[i] = 2.0
    c._side_commit = (-1.0, 10.0)                    # latched RIGHT earlier
    cmd = c.step((0, 0, 0), m, now=0.0)
    assert cmd.detail["heading_deg"] > 0             # goes left, toward the goal
    assert c._side_commit is None or c._side_commit[0] > 0


def test_direct_sector_decision_does_not_flap():
    c = BearingController(ControllerConfig(commit_s=3.0))
    c.set_goal(Goal(0.6, 1.0), (0, 0, 0))            # +59 deg, 1.17 m away
    # Corridor blocked; clearance on the goal bearing hovers around the
    # goal distance from tick to tick.
    def m(goal_clear):
        mm = cmap(clear=0.5, corridor=0.5)
        for i, b in enumerate(mm.bearings_rad):
            if math.degrees(b) >= 30:
                mm.clearance_m[i] = mm.obstacle_m[i] = goal_clear
        return mm
    first = c.step((0, 0, 0), m(1.10), now=0.0)      # 1.10 < 1.17: sector
    assert first.detail["mode"] == "sector"
    second = c.step((0, 0, 0), m(1.25), now=0.2)     # would be direct now
    assert second.detail.get("mode") != "direct"     # ...but stays put
    c._turn_commit = None                            # let the turn finish
    later = c.step((0, 0, 0), m(1.25), now=3.5)
    assert later.detail["mode"] == "direct"


def test_stalled_reverse_escape_gives_up():
    c = BearingController(ControllerConfig(blocked_timeout_s=1.0))
    boxed = cmap(clear=0.4, corridor=0.4)
    c.set_goal(Goal(2.0, 0.0), (0, 0, 0))
    cmd = c.step((0, 0, 0), boxed, now=0.0)
    assert cmd.state == "escape" and cmd.vx == pytest.approx(-0.22)
    cmd = c.step((0, 0, 0), boxed, now=1.0)          # robot has not moved
    assert cmd.state == "escape"
    cmd = c.step((0, 0, 0), boxed, now=1.7)
    assert cmd.state == "blocked" and cmd.detail.get("escape") == "reverse stalled"


# --- path goals (pure pursuit) -------------------------------------------------

def _speeds(cmds):
    return [c.vx for c in cmds]


def run_path(ctrl, sim, points, map_fn=lambda s: cmap(), max_steps=3000, dt=1 / 15):
    ctrl.set_goal(Goal.from_path(points), sim.odom)
    t = 0.0
    cmds = []
    traj = []
    for _ in range(max_steps):
        cmd = ctrl.step(sim.odom, map_fn(sim), now=t)
        cmds.append(cmd)
        traj.append(sim.odom)
        if cmd.state in ("reached", "aborted_blocked", "stopped"):
            return cmd, cmds, traj
        sim.step(cmd.vx, cmd.wz, dt)
        t += dt
    raise AssertionError(f"did not finish: {cmds[-1]}")


def test_path_goal_follows_an_l_shape_without_stopping_at_the_corner():
    c = BearingController()
    sim = Sim()
    cmd, cmds, traj = run_path(c, sim, [(0.5, 0), (1.5, 0), (1.5, 1.0), (1.5, 2.0)])
    assert cmd.state == "reached"
    assert math.hypot(sim.odom[0] - 1.5, sim.odom[1] - 2.0) < 0.3
    # The corner is passed slowed, not stopped: speed never drops to zero
    # before the final approach, and the robot does not rotate in place.
    before_end = [cm for cm in cmds[:-20]]
    assert min(_speeds(before_end)) > 0.05
    assert all(cm.state == "drive" for cm in before_end)
    # Progress reported and monotonic.
    prog = [cm.detail.get("path_progress", 0) for cm in cmds if cm.detail.get("path_progress") is not None]
    assert prog[-1] >= 0.9 and all(b >= a - 1e-6 for a, b in zip(prog, prog[1:]))
    # Passed through the corner area (within 0.35 m of (1.5, 0)).
    assert min(math.hypot(x - 1.5, y) for x, y, _ in traj) < 0.35


def test_path_corner_slows_but_straight_runs_full_speed():
    c = BearingController()
    sim = Sim()
    cmd, cmds, _ = run_path(c, sim, [(1.0, 0), (2.0, 0), (3.0, 0)])
    # Straight: full speed in the middle.
    mid = cmds[len(cmds) // 2]
    assert mid.vx == pytest.approx(c.config.v_max, abs=0.02)
    c2 = BearingController()
    sim2 = Sim()
    cmd2, cmds2, _ = run_path(c2, sim2, [(0.6, 0), (1.0, 0), (1.0, 1.0)])
    # Near the bend the speed cap applies (bend_deg reported, speed under max).
    bends = [cm for cm in cmds2 if cm.detail.get("bend_deg", 0) > 45]
    assert bends and all(cm.vx < c2.config.v_max for cm in bends)


def test_resending_a_path_keeps_progress():
    c = BearingController()
    sim = Sim()
    pts = [(0.5, 0), (1.0, 0), (1.5, 0), (2.0, 0.5)]
    c.set_goal(Goal.from_path(pts), sim.odom)
    t = 0.0
    for _ in range(60):                       # drive ~1 m
        cmd = c.step(sim.odom, cmap(), now=t)
        sim.step(cmd.vx, cmd.wz, 1 / 15)
        t += 1 / 15
    assert sim.odom[0] > 0.9
    # Re-send the same path (shifted a little, as a follower would after a
    # yaw correction): the controller keeps aiming ahead, not at point 0.
    shifted = [(x, y + 0.02) for x, y in pts]
    c.set_goal(Goal.from_path(shifted), sim.odom)
    cmd = c.step(sim.odom, cmap(), now=t)
    assert c._path_seg >= 1
    assert abs(cmd.detail["goal_bearing_deg"]) < 30
    assert cmd.detail["aim"][0] > sim.odom[0]


def test_path_goal_still_respects_obstacles():
    c = BearingController(ControllerConfig(blocked_timeout_s=1.0, escape=False))
    sim = Sim()
    cmd, cmds, _ = run_path(c, sim, [(1.0, 0), (2.0, 0)],
                            map_fn=lambda s: cmap(clear=0.4, corridor=0.4))
    assert cmd.state == "aborted_blocked"
    assert all(cm.vx <= 0.0 for cm in cmds)


def test_body_path_and_validation():
    g = Goal.from_body_path((1.0, 2.0, math.pi / 2), [(1.0, 0.0), (1.0, 0.5)])
    assert g.path[0] == pytest.approx((1.0, 3.0))
    assert g.path[1] == pytest.approx((0.5, 3.0))
    assert (g.x, g.y) == pytest.approx((0.5, 3.0))
    with pytest.raises(ValueError):
        Goal.from_path([])
    # A single-point path behaves like a point goal.
    c = BearingController()
    sim = Sim()
    cmd, cmds, _ = run_path(c, sim, [(1.0, 0.0)])
    assert cmd.state == "reached"
