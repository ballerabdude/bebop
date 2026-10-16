"""Semantic place memory for the wheeled Bebop.

A lightweight, appearance-based "where am I / what do I know about this
room" layer:

  - named places taught by the operator,
  - DINOv3 descriptors for appearance-based place recognition,
  - operator-stated and vision-derived facts per place,
  - an HTTP surface (:9095) for the app and the on-device LLM tool layer.

This is deliberately **not** metric SLAM. Localisation is place recognition
among rooms the robot has been taught, not a coordinate in a global map; a
metric/VIO backend (e.g. cuVSLAM) can be added later behind the same seam if
the robot ever needs to *drive* to a room.
"""
from .capture import Capture, RigCapture, StaticCapture, View
from .encoder import DinoV3Encoder, PlaceEncoder, find_weights_dir
from .server import DEFAULT_PORT, DEFAULT_ROOT, PlacesService
from .store import Fact, Keyframe, Match, Place, PlaceStore, Tag

__all__ = [
    "Capture",
    "RigCapture",
    "StaticCapture",
    "View",
    "DinoV3Encoder",
    "PlaceEncoder",
    "find_weights_dir",
    "PlacesService",
    "DEFAULT_PORT",
    "DEFAULT_ROOT",
    "PlaceStore",
    "Place",
    "Fact",
    "Keyframe",
    "Tag",
    "Match",
]
