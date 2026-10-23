"""Operator notes that ride along in the drive MCAP: intent, events, places.

One ``SessionNotes`` lives in the vision process and outlives recorder
segments.  Writers: the video server's HTTP routes (``POST /intent``,
``POST /event``), the place scout (``add_place`` on every confirmed
recognition) and, through them, the voice tools.  Reader: the active
``NavdRecorder`` drains it every tick and writes ``/intent``, ``/event``
and ``/place`` (``bebop.navd.Intent`` / ``Event`` / ``PlaceSeen``).

- The intent is sticky and may be set at any time, armed or not: it is
  written at the start of every segment and again whenever it changes, so
  each file carries the mission it was driven under even when a long drive
  rolls over several segments.
- Events are accepted only while a segment is open (``begin_segment`` /
  ``end_segment``, called by the recorder).  Otherwise ``add_event`` raises
  ``NotRecording`` and the HTTP route answers 409, so an operator who says
  "mark that" after an E-STOP closed the segment is told the marker was not
  saved instead of getting a success that lands nowhere.  The recorder
  ends the segment before its final drain, so every accepted event is in
  the file it was accepted for.
- Place sightings are queued only while a segment is open and silently
  ignored otherwise (the scout runs at 1 Hz whether or not anyone drives).
"""

from __future__ import annotations

import collections
import re
import threading
import time
from dataclasses import dataclass

MAX_EVENTS = 256
MAX_PLACES = 2048
MAX_TEXT = 500
_KIND = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")


class NotRecording(RuntimeError):
    """An event arrived while no recording segment is open."""


def _text(value) -> str:
    """Strip, cap and make UTF-8 safe.  Valid JSON such as "\\ud800" decodes
    to a lone surrogate that protobuf refuses to encode in the recorder
    thread; replace it here, at the boundary, instead."""
    s = str(value or "").strip()[:MAX_TEXT]
    return s.encode("utf-8", "replace").decode("utf-8")


@dataclass(frozen=True)
class Intent:
    text: str
    from_place: str
    to_place: str
    stamp_ns: int
    version: int

    @property
    def empty(self) -> bool:
        return not (self.text or self.from_place or self.to_place)

    def to_dict(self) -> dict:
        return {"text": self.text, "from": self.from_place, "to": self.to_place,
                "stamp_ns": self.stamp_ns, "version": self.version}


class SessionNotes:
    """Thread-safe intent slot plus bounded event / place queues."""

    def __init__(self, clock_ns=time.time_ns):
        self._clock_ns = clock_ns
        self._lock = threading.Lock()
        self._intent: Intent | None = None
        self._version = 0
        self._segment = None
        self._events = collections.deque(maxlen=MAX_EVENTS)
        self._places = collections.deque(maxlen=MAX_PLACES)
        self.events_dropped = 0
        self.places_dropped = 0

    # -- recording segment ------------------------------------------------

    def begin_segment(self, token) -> None:
        """The recorder opened a segment; events are accepted from now on."""
        with self._lock:
            self._segment = token

    def end_segment(self, token) -> None:
        """The recorder is closing ``token``'s segment (before its final
        drain).  A stale token, e.g. a segment that already handed over to a
        newer one, changes nothing."""
        with self._lock:
            if self._segment is token:
                self._segment = None

    @property
    def recording(self) -> bool:
        with self._lock:
            return self._segment is not None

    # -- intent -----------------------------------------------------------

    def set_intent(self, text, from_place="", to_place="") -> Intent:
        """Replace the current intent; empty text and places clear it.  Every
        call bumps the version so the recorder re-emits it."""
        with self._lock:
            self._version += 1
            self._intent = Intent(_text(text), _text(from_place), _text(to_place),
                                  int(self._clock_ns()), self._version)
            return self._intent

    def clear_intent(self) -> Intent:
        return self.set_intent("")

    def intent(self) -> Intent | None:
        with self._lock:
            return self._intent

    # -- events / places --------------------------------------------------

    @staticmethod
    def normalize_kind(kind) -> str:
        kind = str(kind or "").strip().lower()
        if not _KIND.match(kind):
            raise ValueError("event kind must match [a-z][a-z0-9_-]{0,31}")
        return kind

    def add_event(self, kind, note="") -> dict:
        """Queue a marker for the open segment; raise NotRecording if none."""
        kind = self.normalize_kind(kind)
        note = _text(note)
        with self._lock:
            if self._segment is None:
                raise NotRecording(
                    "no drive is being recorded (arm the wheels in DialIn "
                    "first); the marker was not saved")
            ev = {"kind": kind, "note": note, "stamp_ns": int(self._clock_ns())}
            if len(self._events) == self._events.maxlen:
                self.events_dropped += 1
            self._events.append(ev)
        return ev

    def add_place(self, place_id, name, score, margin=0.0) -> dict | None:
        """Queue a place sighting for the open segment; None when idle."""
        with self._lock:
            if self._segment is None:
                return None
            seen = {"place_id": _text(place_id), "name": _text(name),
                    "score": float(score), "margin": float(margin),
                    "stamp_ns": int(self._clock_ns())}
            if len(self._places) == self._places.maxlen:
                self.places_dropped += 1
            self._places.append(seen)
        return seen

    def drain_events(self) -> list:
        with self._lock:
            out = list(self._events)
            self._events.clear()
        return out

    def drain_places(self) -> list:
        with self._lock:
            out = list(self._places)
            self._places.clear()
        return out

    def snapshot(self) -> dict:
        with self._lock:
            intent = self._intent.to_dict() if self._intent is not None else None
            return {"recording": self._segment is not None,
                    "intent": intent,
                    "events_pending": len(self._events),
                    "places_pending": len(self._places),
                    "events_dropped": self.events_dropped,
                    "places_dropped": self.places_dropped}
