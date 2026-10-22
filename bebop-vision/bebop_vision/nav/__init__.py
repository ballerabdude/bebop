"""Waypoint navigation primitives (depth stop-cone + bearing controller).

Fresh, small, and testable: nothing here depends on the archived navd /
BEV stack.  The depth cameras provide obstacle avoidance; higher layers
(place recognition, VLM) choose waypoints as a bearing + distance.

    depth_cone.py   body-frame clearance map from one or both depth cameras
    bearing.py      odom-closed-loop waypoint controller (twist per tick)
    drive.py        NavDrive loop (arming, deadman, HTTP goal surface)
    synth.py        synthetic depth rendering for tests/tools
"""
from .depth_cone import (CameraGeometry, ClearanceMap, ConeConfig, DepthCone,
                         footprint_from_rig)
from .bearing import (BearingController, Command, ControllerConfig, Goal,
                      wrap_pi)
from .drive import DEFAULT_PORT, NavDrive, NavServer
from .pose import Pose, PoseConfig, PoseEstimator
from .visual import MatchConfig, MatchResult, match_patches
from .follow import FollowConfig, RouteFollower
from .trip import TripRunner

__all__ = [
    "CameraGeometry", "ClearanceMap", "ConeConfig", "DepthCone",
    "footprint_from_rig",
    "BearingController", "Command", "ControllerConfig", "Goal", "wrap_pi",
    "NavDrive", "NavServer", "DEFAULT_PORT",
    "Pose", "PoseConfig", "PoseEstimator",
    "MatchConfig", "MatchResult", "match_patches",
    "FollowConfig", "RouteFollower", "TripRunner",
]
