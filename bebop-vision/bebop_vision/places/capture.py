"""Read-only capture of the current camera view(s) for teaching a place.

Wraps the Orbbec rig without ever opening a camera: the `bebop-vision`
process owns the rig and each camera exposes a latest-wins frame slot, so a
read here is the same cheap operation the videoserver already does. A stale
frame (older than `max_age_s`) is skipped rather than taught, so a place is
never anchored to a frozen view.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class View:
    role: str
    jpeg: bytes
    stamp_us: int


class Capture(Protocol):
    def grab(self) -> list[View]: ...


class RigCapture:
    """Pulls the freshest camera-encoded JPEG from each configured role."""

    def __init__(self, rig, roles=("near", "far"), max_age_s: float = 0.5):
        self.rig = rig
        self.roles = tuple(roles)
        self.max_age_s = float(max_age_s)

    def grab(self) -> list[View]:
        views: list[View] = []
        cameras = getattr(self.rig, "cameras", {}) or {}
        for role in self.roles:
            cam = cameras.get(role)
            if cam is None:
                continue
            fr = cam.read()
            if fr is None:
                continue
            if fr.age_s() > self.max_age_s:
                continue
            jpeg = getattr(fr, "color_jpeg", None)
            if not jpeg:
                continue
            views.append(View(role, bytes(jpeg), int(getattr(fr, "stamp_us", 0))))
        return views


class StaticCapture:
    """A capture source that returns a fixed set of views (tests / replay)."""

    def __init__(self, views: list[View]):
        self._views = list(views)

    def grab(self) -> list[View]:
        return list(self._views)
