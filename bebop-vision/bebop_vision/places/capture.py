"""Read-only capture of the current camera view(s) for teaching a place.

Wraps the Orbbec rig without ever opening a camera: the `bebop-vision`
process owns the rig and each camera exposes a latest-wins frame slot, so a
read here is the same cheap operation the videoserver already does. A stale
frame (older than `max_age_s`) is skipped rather than taught, so a place is
never anchored to a frozen view.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class View:
    role: str
    jpeg: bytes
    stamp_us: int


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
            fr = self._newest_color_frame(cam)
            if fr is None:
                continue
            jpeg = getattr(fr, "color_jpeg", None)
            if not jpeg:
                continue
            views.append(View(role, bytes(jpeg), int(getattr(fr, "stamp_us", 0))))
        return views

    def _newest_color_frame(self, cam):
        """The freshest frame that actually carries a color JPEG.

        The SDK publishes depth-only FrameSets alongside complete ones (the
        far camera drops ~30% of its color frames under load), so the
        latest slot is often depth-only.  Scan the camera's recent history
        newest-first for a color frame within `max_age_s`; fall back to the
        latest slot when there is no history.
        """
        recent = getattr(cam, "recent", None)
        frames = []
        if recent is not None:
            try:
                frames = list(recent())
            except Exception:  # noqa: BLE001
                frames = []
        if not frames:
            fr = cam.read()
            frames = [fr] if fr is not None else []
        for fr in reversed(frames):
            if fr is None or not getattr(fr, "color_jpeg", None):
                continue
            if fr.age_s() > self.max_age_s:
                return None
            return fr
        return None


class StaticCapture:
    """A capture source that returns a fixed set of views (tests / replay)."""

    def __init__(self, views: list[View]):
        self._views = list(views)

    def grab(self) -> list[View]:
        return list(self._views)
