"""GoalPlanner + GoalDriveNode unit tests against synthetic grids."""

import math
import time
from types import SimpleNamespace

import numpy as np
import pytest

from bebop_vision.bev import BevGrid
from bebop_vision.goal_planner import (GoalDriveNode, GoalHeading, GoalPlanner,
                                       GoalPoint, GoalSlot, parse_goal)
from bebop_vision.proto.bebop.runtime.v1 import bebop_runtime_pb2 as pb

ROWS = COLS = 60
CELL = 0.05


def make_grid(occ=None):
    if occ is None:
        occ = np.zeros((ROWS, COLS), np.uint8)
    else:
        occ = np.asarray(occ, np.uint8).copy()
    return BevGrid(occ=occ, raw=occ.copy(), stamp_us=1,
                   per_camera_age_s={"near": 0.01}, plane_ok={"near": True},
                   roles=["near"], cell_m=CELL, recv_ts=time.monotonic())


def paint(grid, x0, x1, y0, y1, cls):
    r0 = int(round((ROWS * CELL - x1) / CELL))
    r1 = int(round((ROWS * CELL - x0) / CELL))
    c0 = int(round((y0 + COLS * CELL / 2) / CELL))
    c1 = int(round((y1 + COLS * CELL / 2) / CELL))
    grid.occ[max(r0, 0):r1, max(c0, 0):c1] = cls
    grid.raw[max(r0, 0):r1, max(c0, 0):c1] = cls


def cell_at(x, y):
    return (int((ROWS * CELL - x) / CELL), int((y + COLS * CELL / 2) / CELL))


@pytest.fixture
def planner():
    return GoalPlanner()


class FakeRobot:
    def __init__(self, mode=pb.MODE_RUN_POLICY):
        self.state = SimpleNamespace(connected=True, estop_latched=False,
                                     estop_reason="", mode=mode,
                                     odom=(0.0, 0.0, 0.0))
        self.twists = []

    def send_twist(self, vx, wz):
        self.twists.append((vx, wz))


# --- planner ---------------------------------------------------------------

def test_no_goal_waits(planner):
    assert planner.compute(make_grid(), None) == (0.0, 0.0, {"state": "waiting"})


def test_drive_clear_toward_goal(planner):
    vx, wz, info = planner.compute(make_grid(), GoalHeading(0.0))
    assert info["state"] == "drive"
    assert vx == pytest.approx(0.4)   # full clearance -> v_max
    assert wz == pytest.approx(0.0)


def test_drive_curves_toward_offset_goal(planner):
    vx, wz, info = planner.compute(make_grid(), GoalHeading(math.radians(25)))
    assert info["state"] == "drive"
    # Best corridor sits left of the goal bearing: steer right (negative wz).
    assert -0.5 < wz < 0.0
    assert vx > 0.0


def test_rotate_toward_corridor_despite_near_cone(planner):
    # Obstacle dead-ahead blocks the cone, but a clear corridor sits at
    # -30 deg: the planner must pivot toward it (hard stop here would
    # deadlock — bench run 2) rather than stand forever.
    grid = make_grid()
    paint(grid, 0.35, 0.55, -0.15, 0.15, 3)   # blocks cone rays at ~0.4 m
    vx, wz, info = planner.compute(grid, GoalHeading(0.0))
    assert info["state"] == "rotate"
    assert vx == 0.0
    assert wz < -0.5  # pivots right toward the open channel


def test_hard_stop_cone_blocked_no_usable_pivot(planner):
    # Cone blocked dead-ahead AND the best corridor is itself near-blocked:
    # no useful pivot exists -> stand still.
    grid = make_grid()
    paint(grid, 0.42, 0.55, -0.5, 0.4, 1)     # blocks cone rays at ~0.40-0.47
    paint(grid, 0.18, 0.9, -1.5, -0.35, 1)    # blocks right diagonals near-field
    paint(grid, 0.18, 0.9, 0.35, 1.5, 1)      # blocks left diagonals near-field
    vx, wz, info = planner.compute(grid, GoalHeading(math.radians(-25)))
    assert info["state"] == "hard_stop"
    assert (vx, wz) == (0.0, 0.0)


def test_search_when_fully_blocked():
    grid = make_grid()
    grid.occ[:] = 1
    _, wz, info = GoalPlanner().compute(grid, GoalHeading(0.0))
    assert info["state"] == "search"
    assert wz == pytest.approx(1.8)
    # Search rotates toward the goal side, latched at episode entry.
    _, wz_left, _ = GoalPlanner().compute(grid, GoalHeading(math.radians(30)))
    assert wz_left == pytest.approx(1.8)
    _, wz_right, _ = GoalPlanner().compute(grid, GoalHeading(math.radians(-30)))
    assert wz_right == pytest.approx(-1.8)


def test_search_latch_no_thrash():
    # Mid-episode bearing swings (e.g. from the robot's own rotation) must
    # not flip the search direction until a corridor actually opens.
    grid = make_grid()
    grid.occ[:] = 1
    p = GoalPlanner()
    _, wz1, info1 = p.compute(grid, GoalHeading(math.radians(30)))
    assert info1["state"] == "search" and wz1 == pytest.approx(1.8)
    _, wz2, info2 = p.compute(grid, GoalHeading(math.radians(-30)))
    assert info2["state"] == "search"
    assert wz2 == pytest.approx(1.8)  # same direction: episode persists
    # A clear corridor ends the episode.
    open_grid = make_grid()
    _, wz3, info3 = p.compute(open_grid, GoalHeading(math.radians(-30)))
    assert info3["state"] != "search"
    # Fully blocked again -> fresh episode, direction follows the new side.
    _, wz4, info4 = p.compute(grid, GoalHeading(math.radians(-30)))
    assert info4["state"] == "search" and wz4 == pytest.approx(-1.8)


def test_rotate_when_goal_far_off_corridor(planner):
    # All clear but the goal bearing is beyond the ray span: rotate in place.
    # dpsi = ray(-60 deg) - bearing(-90 deg) = +30 deg -> wz = k_yaw * dpsi > 0
    # (turning left moves the corridor's body-frame angle down toward -90).
    vx, wz, info = planner.compute(make_grid(), GoalHeading(-math.pi / 2))
    assert info["state"] == "rotate"
    assert vx == 0.0
    assert wz == pytest.approx(1.05, abs=0.05)


def test_ray_march_clearance(planner):
    grid = make_grid()
    grid.occ[cell_at(1.475, 0.025)] = 1
    clear, r_max = planner._march_clearances(grid)
    assert clear[6] == pytest.approx(1.475, abs=0.03)   # center ray blocked
    assert clear[0] == pytest.approx(3.0)               # -60 deg ray clear
    assert clear[-1] == pytest.approx(3.0)


def test_goal_point_reached(planner):
    vx, wz, info = planner.compute(make_grid(), GoalPoint(0.2, 0.0))
    assert info["state"] == "reached"
    assert (vx, wz) == (0.0, 0.0)


def test_goal_point_bearing_uses_odom(planner):
    # Waypoint 1 m to the left: bearing +90 deg, beyond the ray span -> rotate.
    vx, wz, info = planner.compute(make_grid(), GoalPoint(0.0, 1.0))
    assert info["state"] == "rotate"
    assert wz < 0
    # Same waypoint behind: bearing flips with odometry.
    vx, wz, info = planner.compute(make_grid(), GoalPoint(0.0, 1.0),
                                   odom=(0.0, 0.0, math.pi))
    assert info["state"] == "rotate"
    assert wz > 0


def test_parse_goal():
    assert parse_goal("heading 25") == GoalHeading(math.radians(25))
    assert parse_goal("xy 1.5 0.5") == GoalPoint(1.5, 0.5)
    assert parse_goal("stop") == "stop"
    assert parse_goal("") is None
    with pytest.raises(ValueError):
        parse_goal("heading")
    with pytest.raises(ValueError):
        parse_goal("warp 9")


# --- GoalDriveNode gating ---------------------------------------------------

def make_node(grid_holder, goal_slot=None, **kwargs):
    robot = FakeRobot()
    node = GoalDriveNode(robot, GoalPlanner(), lambda: grid_holder[0],
                         goal_slot=goal_slot, **kwargs)
    return node, robot


def test_node_waiting_without_grid():
    node, robot = make_node([None])
    node.on_grid()
    assert node.last_info["state"] == "waiting"
    assert robot.twists[-1] == (0.0, 0.0)


def test_node_deadman_on_stale_grid():
    grid = make_grid()
    grid.recv_ts = time.monotonic() - 1.0
    node, robot = make_node([grid], goal_slot=GoalSlot())
    node.goal_slot.set(GoalHeading(0.0))
    node.on_grid()
    assert node.last_info["state"] == "waiting"
    assert robot.twists[-1] == (0.0, 0.0)


def test_node_estop_gate():
    node, robot = make_node([make_grid()], goal_slot=GoalSlot())
    node.goal_slot.set(GoalHeading(0.0))
    robot.state.estop_latched = True
    node.on_grid()
    assert node.last_info["state"] == "estop"
    assert robot.twists[-1] == (0.0, 0.0)


def test_node_mode_gate():
    node, robot = make_node([make_grid()], goal_slot=GoalSlot(),
                            require_mode=pb.MODE_RUN_POLICY)
    node.goal_slot.set(GoalHeading(0.0))
    robot.state.mode = pb.MODE_IDLE
    node.on_grid()
    assert node.last_info["state"] == "hold"
    assert robot.twists[-1] == (0.0, 0.0)


def test_node_drives_when_gates_open():
    slot = GoalSlot()
    node, robot = make_node([make_grid()], goal_slot=slot)
    slot.set(GoalHeading(0.0))
    node.on_grid()
    assert node.last_info["state"] == "drive"
    assert robot.twists[-1][0] == pytest.approx(0.4)


def test_node_no_floor_gate():
    grid = make_grid()
    grid.plane_ok = {"near": False, "far": False}
    node, robot = make_node([grid], goal_slot=GoalSlot(), plane_fail_s=0.0)
    node.goal_slot.set(GoalHeading(0.0))
    node.on_grid()               # first tick: timer starts, planner output
    node._last_cmd_ts = 0.0      # bypass the rate limiter for the second tick
    node.on_grid()               # second tick: both cameras still floorless
    assert node.last_info["state"] == "no_floor"
    assert robot.twists[-1] == (0.0, 0.0)


def test_node_plane_fail_tolerated_below_timeout():
    grid = make_grid()
    grid.plane_ok = {"near": False, "far": False}
    node, robot = make_node([grid], goal_slot=GoalSlot(), plane_fail_s=10.0)
    node.goal_slot.set(GoalHeading(0.0))
    node.on_grid()
    assert node.last_info["state"] == "drive"  # not yet timed out
