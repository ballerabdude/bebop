"""Deterministic in-memory tracker for hardware-free tests."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from .backend import Pose, TrackResult


class FakeTracker:
    """A tracker driven by explicit per-call results.

    Seams:
      results           full TrackResult sequence (takes precedence)
      poses             Pose|None per call
      corrected_poses   Pose|None per call
      states            tracking-state string per call
      errors            error string|None per call

    A provided sequence is consumed one item per ``track`` call.  When it
    runs out the call reports ``lost`` (``error="fake results exhausted"``)
    rather than silently repeating a pose.  With no sequences at all the
    fake returns a straight +x path at 0.01 m per call, which keeps the
    default live-status path easy to smoke-test.
    """

    def __init__(self, results: Iterable[TrackResult] | None = None,
                 poses: Sequence[Pose | None] | None = None,
                 corrected_poses: Sequence[Pose | None] | None = None,
                 states: Sequence[str] | None = None,
                 errors: Sequence[str | None] | None = None):
        self._results = list(results) if results is not None else None
        self._poses = list(poses) if poses is not None else None
        self._corrected = (list(corrected_poses)
                           if corrected_poses is not None else None)
        self._states = list(states) if states is not None else None
        self._errors = list(errors) if errors is not None else None
        self._i = 0
        self.calls: list[dict] = []
        self.closed = False
        self.name = "fake"

    @staticmethod
    def _at(seq, i, default=None):
        return seq[i] if seq is not None and i < len(seq) else default

    def track(self, timestamp_ns: int, rgb: np.ndarray,
              depth_mm: np.ndarray) -> TrackResult:
        i = self._i
        self._i += 1
        self.calls.append({
            "timestamp_ns": int(timestamp_ns),
            "rgb_shape": tuple(np.asarray(rgb).shape),
            "depth_shape": tuple(np.asarray(depth_mm).shape),
            "depth_dtype": str(np.asarray(depth_mm).dtype),
        })
        if self._results is not None:
            if i >= len(self._results):
                return TrackResult(None, None, "lost",
                                   "fake results exhausted")
            return self._results[i]

        if self._poses is None and self._corrected is None \
                and self._states is None and self._errors is None:
            pose = Pose(int(timestamp_ns),
                        (0.01 * i, 0.0, 0.0),
                        (0.0, 0.0, 0.0, 1.0))
            return TrackResult(pose, None, "tracking", None)

        pose = self._at(self._poses, i) if self._poses is not None else None
        corrected = (self._at(self._corrected, i)
                     if self._corrected is not None else None)
        state = self._at(self._states, i,
                         "tracking" if pose is not None else "lost")
        error = self._at(self._errors, i)
        return TrackResult(pose, corrected, state, error)

    def close(self) -> None:
        self.closed = True
