"""Semantic place memory for the wheeled Bebop.

A lightweight, appearance-based "where am I / what do I know about this
room" layer:

  - named places taught by the operator,
  - DINOv3 descriptors for appearance-based place recognition,
  - operator-stated and vision-derived facts per place,
  - an HTTP surface (:9095) for the app and the on-device LLM tool layer.

This is deliberately **not** global metric SLAM: localisation is place
recognition among taught rooms, and places are linked by traversal knowledge
(success counts per A->B edge), not by a single global map.
"""
from .capture import RigCapture, StaticCapture, View
from .encoder import DinoV3Encoder, find_weights_dir
from .importer import PlaceImporter
from .scout import PlaceScout, ScoutConfig
from .server import DEFAULT_PORT, PlacesService
from .store import Fact, Keyframe, Match, Place, PlaceEdge, PlaceStore

__all__ = [
    "RigCapture",
    "StaticCapture",
    "View",
    "DinoV3Encoder",
    "find_weights_dir",
    "PlacesService",
    "DEFAULT_PORT",
    "PlaceStore",
    "PlaceScout",
    "ScoutConfig",
    "PlaceImporter",
    "Place",
    "PlaceEdge",
    "Fact",
    "Keyframe",
    "Match",
]
