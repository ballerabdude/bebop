"""Semantic place memory for the wheeled Bebop.

A lightweight, appearance-based "where am I / what do I know about this
room" layer:

  - named places taught by the operator,
  - DINOv3 descriptors for appearance-based place recognition,
  - operator-stated and vision-derived facts per place,
  - an HTTP surface (:9095) for the app and the on-device LLM tool layer.

This is deliberately **not** global metric SLAM.  Each taught place can own
a local, gravity-aligned structure submap accumulated across captures in one
vision-session frame (the first capture's pose is the pinned anchor), and
places are linked by route knowledge (traversal/success counts), not by a
single global map.  Re-capturing a place unions its occupied/free evidence
instead of replacing it; `replace=True` is the explicit reset.
Localisation is still primarily place recognition among taught rooms; the
submaps and graph are the navigation material, and relocalization is the
missing piece that would tie session frames together.
"""
from .capture import Capture, RigCapture, StaticCapture, View
from .encoder import DinoV3Encoder, PlaceEncoder, find_weights_dir
from .importer import PlaceImporter
from .scout import PlaceScout, ScoutConfig
from .server import DEFAULT_PORT, DEFAULT_ROOT, PlacesService
from .store import (Fact, Keyframe, Match, Place, PlaceEdge, PlaceMap,
                    PlaceStore, Route, RouteNode, Tag)
from .trail import TrailConfig, TrailTeacher

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
    "PlaceScout",
    "ScoutConfig",
    "PlaceImporter",
    "Place",
    "PlaceMap",
    "PlaceEdge",
    "Route",
    "RouteNode",
    "TrailConfig",
    "TrailTeacher",
    "Fact",
    "Keyframe",
    "Tag",
    "Match",
]
