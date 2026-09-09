"""Navigation-goal plumbing for data collection.

The app / stdin set a goal; the firmware broadcasts it as
NavigationGoalState; main.py's record path holds it here and the MCAP
recorder writes it per tick — so teleop sessions toward a waypoint carry
the goal in the dataset (the signal the learned models train on).

Extracted from goal_planner.py when the drive/planner stack moved to the
exp/navd-ai branch; the goal *slot* is recording infrastructure, not
driving.
"""

import dataclasses
import math


@dataclasses.dataclass
class GoalHeading:
    """Body-frame heading offset (rad, + left). Never 'reaches'."""
    heading_rad: float


@dataclasses.dataclass
class GoalPoint:
    """Odom-frame waypoint (m)."""
    x: float
    y: float


class GoalSlot:
    """Latest-wins goal slot shared between the app bridge and recorder."""

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
