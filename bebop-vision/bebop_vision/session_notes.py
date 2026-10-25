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

Runs.  A non-empty intent belongs to a *run*: one attempt at a task, named
by a ``run_id`` ("r" + local YYYYMMDD_HHMMSS + "_" + 4 hex) and stamped
with ``run_started_ns``.  The run id is what tells two drives with the same
text apart, and what groups the several segments one attempt can span (a
disarm, an E-STOP or a size roll-over closes the segment, not the run).
``set_intent(mode="new"|"repeat")`` starts a run, ``mode="continue"``
edits the open one, ``end_run`` closes it with an outcome (arrived, found,
gave_up, unfinished) or only counts an interruption.  A run that closes
without arrived / found / gave_up is ``unfinished``: an explicit decline
(``end_run("unfinished")``), a ``clear_intent``, or a new or repeated run
started while it was still open.  Older state files and recordings closed
such runs with outcome ``""``; ``recent()`` and the lab read that as
unfinished too.  The last few runs are kept for ``recent()`` so the app
and the voice agent can repeat one.

Breaks.  A run spans recording segments, and the operator decides what a
break in it means; nothing here infers it.  ``note_stop`` records why a
segment ended (the recorder loop computes the cause from the firmware
state: a disarm, an E-STOP, a lost link...) as a ``stop`` event in the
closing file and in the run's ``stops`` list.  ``resume_run`` records the
operator's "continue this run" answer (with an optional note) as a
``resume`` event: written at once while a segment records, otherwise held
and written as the first event of the next segment, which is the drive it
belongs to.

Persistence.  With ``state_path`` the open intent, the recent runs (with
each closed run's ``end_note``) and the held resume events are saved as JSON (tmp file + ``os.replace``) on every change and loaded at
start, so a vision restart mid-run keeps the run id instead of silently
dropping the mission.  A missing or unreadable file only prints a warning.
"""

from __future__ import annotations

import collections
import json
import os
import re
import secrets
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path

MAX_EVENTS = 256
MAX_PLACES = 2048
MAX_TEXT = 500
_KIND = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")
MAX_RECENT = 8
MAX_STOPS = 64          # per run, oldest dropped
MAX_RESUMES = 32        # per run, oldest dropped
MAX_HELD = 32           # resume events waiting for the next segment
STATE_FORMAT = 1

# The task vocabulary (mirrored in bebop-app's driveNotes.ts).  The value
# is the closed set of targets a task takes, None for free text (a place
# name, a step, an object) and () for a task that takes no target.
TASKS: dict[str, tuple[str, ...] | None] = {
    "hold": (),
    "go_to_place": None,
    "go_through_door": ("left", "right", "ahead"),
    "follow_step": None,
    "look_for": None,
    "explore": None,
    "return": None,
    # The path-model intent one-hot the app's maneuver presets set.
    "maneuver": ("keep", "left", "right", "stop", "turn-around"),
}
# A run closed without arrived / found / gave_up: declined at the resume
# prompt, cleared, or replaced by a new run while open.  Mirrored in
# bebop-app's driveNotes.ts and the lab (the app shows it through a display
# map, so the word on screen can change without touching recordings).
UNFINISHED = "unfinished"
# Outcomes that end a run; "interrupted" is only a marker on the open run.
ENDING_OUTCOMES = ("arrived", "found", "gave_up", UNFINISHED)
OUTCOMES = (*ENDING_OUTCOMES, "interrupted")
MODES = ("new", "repeat", "continue")
# Why a recording segment ended (``note_stop``; computed in main.py's --auto
# loop from the firmware state).  Disarming every wheel is the normal stop;
# an E-STOP is an emergency, split by who latched it.
STOP_CAUSES = (
    "disarmed",         # every wheel disarmed
    "wheel_disarmed",   # some but not all wheels disarmed
    "idle",             # mode left DialIn
    "estop_operator",   # E-STOP latched by the operator
    "estop_auto",       # E-STOP latched for any other reason
    "link_lost",        # robot runtime not connected
    "rolled",           # segment rolled over by size or time
    "recorder_stop",    # the recorder process is stopping
)


class NotRecording(RuntimeError):
    """An event arrived while no recording segment is open."""


class NoRun(ValueError):
    """``set_intent(mode="continue")`` or ``resume_run`` with no open run
    (HTTP 409)."""


def new_run_id(clock_ns: int) -> str:
    """``r`` + local YYYYMMDD_HHMMSS + ``_`` + 4 lowercase hex."""
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(clock_ns / 1e9))
    return f"r{stamp}_{secrets.token_hex(2)}"


def normalize_task(task) -> str:
    """``""`` or one of TASKS; ValueError otherwise."""
    task = str(task or "").strip().lower()
    if task and task not in TASKS:
        raise ValueError(f"task must be one of {', '.join(TASKS)} (or empty)")
    return task


def normalize_target(task: str, target) -> str:
    """Cap the target text; check it against a task's closed set.  A task
    that takes no target (hold) drops whatever came with it."""
    target = _text(target)
    allowed = TASKS.get(task)
    if allowed is None or not target:
        return target
    if not allowed:
        return ""
    low = target.lower()
    if low not in allowed:
        raise ValueError(f"target for {task} must be one of {', '.join(allowed)}")
    return low


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
    task: str = ""
    target: str = ""
    run_id: str = ""            # "" when empty (no run)
    run_started_ns: int = 0

    @property
    def empty(self) -> bool:
        return not (self.text or self.from_place or self.to_place
                    or self.task or self.target)

    def to_dict(self) -> dict:
        return {"text": self.text, "from": self.from_place, "to": self.to_place,
                "task": self.task, "target": self.target,
                "run_id": self.run_id, "run_started_ns": self.run_started_ns,
                "stamp_ns": self.stamp_ns, "version": self.version}


def _run_entry(it: Intent) -> dict:
    """A recent-runs row for the open run of ``it``."""
    return {"run_id": it.run_id, "text": it.text, "from": it.from_place,
            "to": it.to_place, "task": it.task, "target": it.target,
            "started_ns": it.run_started_ns, "ended_ns": 0, "outcome": "",
            "interruptions": 0, "stops": [], "last_stop": None, "resumes": [],
            "end_note": ""}


def _public_row(row: dict) -> dict:
    """The recent-runs row as ``recent()`` / ``/notes`` show it: the stop and
    resume logs become counts (plus ``last_stop``), and a closed run with no
    outcome (saved before ``unfinished`` existed) reads as unfinished."""
    out = dict(row)
    out["stops"] = len(row.get("stops") or ())
    out["resumes"] = len(row.get("resumes") or ())
    last = row.get("last_stop")
    out["last_stop"] = dict(last) if last else None
    if out.get("ended_ns") and not out.get("outcome"):
        out["outcome"] = UNFINISHED
    return out


class SessionNotes:
    """Thread-safe intent slot plus bounded event / place queues."""

    def __init__(self, clock_ns=time.time_ns, state_path=None):
        self._clock_ns = clock_ns
        self._lock = threading.Lock()
        self._intent: Intent | None = None
        self._run: dict | None = None        # recent-runs row of the open run
        self._recent: collections.deque = collections.deque(maxlen=MAX_RECENT)
        self._version = 0
        self._segment = None
        self._events = collections.deque(maxlen=MAX_EVENTS)
        # Resume events made while no segment records, for the next one.
        self._held: collections.deque = collections.deque(maxlen=MAX_HELD)
        self._places = collections.deque(maxlen=MAX_PLACES)
        self.events_dropped = 0
        self.places_dropped = 0
        self._state_path = Path(state_path) if state_path else None
        # Every closed run, appended forever (the recent list keeps 8): the
        # lab's record of how runs ended. An outcome picked after Stop
        # driving lands in no drive file (disarming closed it), so this log
        # is where it lives. Served as GET /runs.
        self._runs_log_path = (self._state_path.with_name("runs.jsonl")
                               if self._state_path is not None else None)
        if self._state_path is not None:
            self._load()

    # -- recording segment ------------------------------------------------

    def begin_segment(self, token) -> None:
        """The recorder opened a segment; events are accepted from now on.

        Held resume events go first into the queue, stamped now (the file's
        event stream starts at its open; the operator's own time stays in the
        run's ``resumes`` list) so the recorder's pre-segment filter keeps
        them.  One whose run has closed since is dropped: it would mark the
        wrong run.
        """
        with self._lock:
            self._segment = token
            now = int(self._clock_ns())
            open_id = self._run["run_id"] if self._run is not None else None
            had_held = bool(self._held)
            held = [dict(ev, stamp_ns=max(now, ev["stamp_ns"]))
                    for ev in self._held if ev.get("run_id") == open_id]
            self._held.clear()
            for ev in reversed(held):
                ev.pop("run_id", None)
                self._events.appendleft(ev)
            if had_held:
                self._save_locked()     # delivered: a restart must not redo them

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

    # -- intent / runs ----------------------------------------------------

    def set_intent(self, text, from_place=None, to_place=None, task=None,
                   target=None, mode="new") -> Intent:
        """Set the intent; every call bumps the version so the recorder
        re-emits it.

        ``mode="new"`` (and ``"repeat"``, the same thing said for a re-run)
        starts a fresh run, closing any open one as unfinished.
        ``mode="continue"`` keeps the open run's id and start and only
        replaces the fields that are not None (a task change without a
        target clears the target, which belonged to the old task); it raises
        ``NoRun`` (a ValueError) when no run is open.  A result with empty
        text, places, task and target clears the intent: there is no run
        then.
        """
        mode = str(mode or "new").strip().lower()
        if mode not in MODES:
            raise ValueError(f"mode must be one of {', '.join(MODES)}")
        with self._lock:
            now = int(self._clock_ns())
            cur = self._intent
            if mode == "continue":
                if cur is None or not cur.run_id:
                    raise NoRun("no run is open to continue; start a new one")
                text = cur.text if text is None else text
                from_place = cur.from_place if from_place is None else from_place
                to_place = cur.to_place if to_place is None else to_place
                if target is None:
                    # The old target only means something for the old task;
                    # a task change without a target drops it.
                    same = task is None or normalize_task(task) == cur.task
                    target = cur.target if same else ""
                task = cur.task if task is None else task
            task = normalize_task(task)
            fields = dict(text=_text(text), from_place=_text(from_place),
                          to_place=_text(to_place), task=task,
                          target=normalize_target(task, target))
            probe = Intent(stamp_ns=now, version=0, **fields)
            if probe.empty:
                self._close_run_locked(UNFINISHED, now)
                run_id, started = "", 0
            elif mode == "continue":
                run_id, started = cur.run_id, cur.run_started_ns
            else:
                self._close_run_locked(UNFINISHED, now)
                run_id, started = new_run_id(now), now
            self._version += 1
            self._intent = replace(probe, version=self._version,
                                   run_id=run_id, run_started_ns=started)
            if run_id:
                if self._run is None or self._run["run_id"] != run_id:
                    self._run = _run_entry(self._intent)
                else:
                    self._run.update({k: v for k, v in _run_entry(self._intent).items()
                                      if k in ("text", "from", "to", "task", "target")})
            self._save_locked()
            return self._intent

    def clear_intent(self) -> Intent:
        """Clear the intent; an open run ends as unfinished."""
        return self.set_intent("", "", "", "", "")

    def intent(self) -> Intent | None:
        with self._lock:
            return self._intent

    def intent_dict(self) -> dict | None:
        """The intent as ``/notes`` shows it: ``Intent.to_dict`` plus the open
        run's ``stops`` and ``resumes`` counts and ``last_stop`` (0, 0, None
        with no run).  None before any intent was ever set."""
        with self._lock:
            return self._intent_dict_locked()

    def _intent_dict_locked(self) -> dict | None:
        if self._intent is None:
            return None
        d = self._intent.to_dict()
        run = self._run
        same = run is not None and run["run_id"] == self._intent.run_id
        view = _public_row(run) if same else {}
        d.update(stops=view.get("stops", 0), resumes=view.get("resumes", 0),
                 last_stop=view.get("last_stop"))
        return d

    def repeat_run(self, run_id) -> Intent:
        """Start a new run with the text, places, task and target of a recent
        run (the open one included); ValueError if ``run_id`` is unknown."""
        run_id = str(run_id or "").strip()
        with self._lock:
            src = next((r for r in self._recent_locked() if r["run_id"] == run_id), None)
        if src is None:
            raise ValueError(f"unknown run {run_id!r}")
        return self.set_intent(src["text"], src["from"], src["to"], src["task"],
                               src["target"], mode="repeat")

    def end_run(self, outcome, note="") -> dict:
        """End the open run with ``outcome`` (arrived, found, gave_up, or
        unfinished: the operator declined to continue it), or count an
        interruption on it (interrupted keeps it open).

        While a segment records, the outcome is also queued as an event
        (kind = outcome) so the file shows where the run ended; ending a run
        never needs a recording, so ``event_recorded`` reports which it was.
        The note of an ending outcome is kept on the run row as ``end_note``
        (persisted) either way: the app's end flows (the start prompt's
        "close it", the stop sheet) run while nothing records, and the
        operator's context text must not depend on a segment being open.
        With no run open nothing happens (``run`` None, no event), so a
        double tap on an end button does not write a second, runless ending.
        Returns ``{"run": <row>, "event_recorded": bool}``.
        """
        outcome = str(outcome or "").strip().lower()
        if outcome not in OUTCOMES:
            raise ValueError(f"outcome must be one of {', '.join(OUTCOMES)}")
        note = _text(note)
        with self._lock:
            now = int(self._clock_ns())
            if self._run is None:
                return {"run": None, "event_recorded": False}
            recorded = self._segment is not None
            if recorded:
                self._queue_event_locked({"kind": outcome, "note": note, "stamp_ns": now})
            if outcome == "interrupted":
                self._run["interruptions"] += 1
                run = _public_row(self._run)
            else:
                run = self._close_run_locked(outcome, now, note)
                self._version += 1
                self._intent = Intent("", "", "", now, self._version)
            self._save_locked()
        return {"run": run, "event_recorded": recorded}

    def resume_run(self, note="") -> dict:
        """The operator chose to continue the open run (the app asks every
        time a drive starts with a run open).  Logs ``{t_ns, note}`` in the
        run's ``resumes`` and queues a ``resume`` event: now while a segment
        records, otherwise held for the first event of the next segment.
        Raises ``NoRun`` with no run open.  Returns ``{"run": <row>,
        "event_recorded": bool, "held": bool}``.
        """
        note = _text(note)
        with self._lock:
            if self._run is None:
                raise NoRun("no run is open to resume; start a new one")
            now = int(self._clock_ns())
            resumes = list(self._run.get("resumes") or [])
            resumes.append({"t_ns": now, "note": note})
            self._run["resumes"] = resumes[-MAX_RESUMES:]
            ev = {"kind": "resume", "note": note, "stamp_ns": now}
            recorded = self._segment is not None
            if recorded:
                self._queue_event_locked(ev)
            else:
                self._held.append(dict(ev, run_id=self._run["run_id"]))
            run = _public_row(self._run)
            self._save_locked()
        return {"run": run, "event_recorded": recorded, "held": not recorded}

    def note_stop(self, cause, detail="") -> dict:
        """Record why the recording segment is ending (one of STOP_CAUSES;
        ``detail`` is the E-STOP reason or the disarmed wheels).  Call it
        before the recorder closes the segment so the ``stop`` event (note
        ``"<cause>"`` or ``"<cause>: <detail>"``) lands in the closing file.
        The open run, if any, logs ``{t_ns, cause, detail}`` in ``stops``
        and as ``last_stop`` either way.  ValueError on an unknown cause.
        Returns ``{"stop": {...}, "event_recorded": bool, "run": <row>|None}``.
        """
        cause = str(cause or "").strip().lower()
        if cause not in STOP_CAUSES:
            raise ValueError(f"stop cause must be one of {', '.join(STOP_CAUSES)}")
        detail = _text(detail)
        with self._lock:
            now = int(self._clock_ns())
            stop = {"t_ns": now, "cause": cause, "detail": detail}
            recorded = self._segment is not None
            if recorded:
                note = f"{cause}: {detail}" if detail else cause
                self._queue_event_locked({"kind": "stop", "note": _text(note),
                                          "stamp_ns": now})
            run = None
            if self._run is not None:
                stops = list(self._run.get("stops") or [])
                stops.append(stop)
                self._run["stops"] = stops[-MAX_STOPS:]
                self._run["last_stop"] = dict(stop)
                run = _public_row(self._run)
                self._save_locked()
        return {"stop": stop, "event_recorded": recorded, "run": run}

    def recent(self) -> list[dict]:
        """Newest first, at most MAX_RECENT rows; the open run comes first
        (``ended_ns`` 0, ``outcome`` "").  Stops and resumes are counts here
        (``last_stop`` the newest stop or None)."""
        with self._lock:
            return self._recent_locked()

    def _recent_locked(self) -> list[dict]:
        rows = [self._run] if self._run is not None else []
        rows.extend(self._recent)
        return [_public_row(r) for r in rows[:MAX_RECENT]]

    def _close_run_locked(self, outcome: str, now: int, note: str = "") -> dict | None:
        """Move the open run (if any) into the recent list, with ``note`` as
        its ``end_note``; return its row."""
        run, self._run = self._run, None
        self._held.clear()          # a held resume belonged to this run
        if run is None:
            return None
        run.update(ended_ns=now, outcome=outcome, end_note=note)
        self._recent.appendleft(run)
        self._append_run_log_locked(run)
        return _public_row(run)

    def _append_run_log_locked(self, run: dict) -> None:
        """Append one closed run (full stop / resume logs) to ``runs.jsonl``;
        a failure only warns."""
        if self._runs_log_path is None:
            return
        try:
            with open(self._runs_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(run) + "\n")
                f.flush()
                os.fsync(f.fileno())
        except OSError as exc:
            print(f"[session-notes] could not append {self._runs_log_path}: {exc}")

    def run_log(self, since_ns: int = 0) -> list[dict]:
        """Every closed run in ``runs.jsonl`` that ended after ``since_ns``,
        oldest first (unreadable lines skipped); ``[]`` without persistence."""
        path = self._runs_log_path
        if path is None or not path.is_file():
            return []
        out = []
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    try:
                        row = json.loads(line)
                    except ValueError:
                        continue        # a line cut by a power loss
                    if isinstance(row, dict) and int(row.get("ended_ns") or 0) > since_ns:
                        out.append(row)
        except OSError as exc:
            print(f"[session-notes] could not read {path}: {exc}")
        return out

    # -- persistence ------------------------------------------------------

    def _save_locked(self) -> None:
        """Write the open intent and recent runs atomically; a failure only
        warns (the drive goes on, the run is just not restart-safe)."""
        if self._state_path is None:
            return
        it = self._intent
        state = {"format": STATE_FORMAT,
                 "intent": it.to_dict() if it is not None and it.run_id else None,
                 "run": self._run,
                 "recent": list(self._recent),
                 # held resume events, so a restart before the next segment
                 # still writes them into it
                 "held": list(self._held)}
        tmp = self._state_path.with_name(self._state_path.name + ".tmp")
        try:
            # fsync the data before the rename (and the directory after), so
            # a power cut leaves the old file or the new one, never an empty
            # one: the restart case is what this file is for.
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(json.dumps(state))
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self._state_path)
            try:
                fd = os.open(self._state_path.parent, os.O_RDONLY)
            except OSError:
                fd = None
            if fd is not None:
                try:
                    os.fsync(fd)
                except OSError:
                    pass
                finally:
                    os.close(fd)
        except OSError as exc:
            print(f"[session-notes] could not save {self._state_path}: {exc}")

    def _load(self) -> None:
        """Restore the open intent, recent runs and held resume events (only
        those of the restored open run).  The intent keeps its run id and
        start but takes a fresh version, so the recorder writes it into the
        next segment."""
        path = self._state_path
        try:
            state = json.loads(path.read_text())
        except FileNotFoundError:
            return
        except (OSError, ValueError) as exc:
            print(f"[session-notes] ignoring unreadable {path}: {exc}")
            return
        try:
            if not isinstance(state, dict):
                raise ValueError("not a JSON object")
            recent = [self._row(r) for r in (state.get("recent") or [])][:MAX_RECENT]
            d = state.get("intent")
            intent = run = None
            if d:
                task = normalize_task(d.get("task", ""))
                intent = Intent(_text(d.get("text")), _text(d.get("from")),
                                _text(d.get("to")), int(d.get("stamp_ns") or 0), 1,
                                task, normalize_target(task, d.get("target")),
                                _text(d.get("run_id")), int(d.get("run_started_ns") or 0))
                if intent.empty or not intent.run_id:
                    intent = None
                else:
                    saved = state.get("run")
                    run = (self._row(saved) if isinstance(saved, dict)
                           and saved.get("run_id") == intent.run_id
                           else _run_entry(intent))
            held = []
            if intent is not None:
                for ev in state.get("held") or []:
                    if isinstance(ev, dict) and ev.get("run_id") == intent.run_id:
                        held.append({"kind": "resume", "note": _text(ev.get("note")),
                                     "stamp_ns": int(ev.get("stamp_ns") or 0),
                                     "run_id": intent.run_id})
        except (TypeError, ValueError, AttributeError) as exc:
            print(f"[session-notes] ignoring malformed {path}: {exc}")
            return
        self._recent.extend(recent)
        if intent is not None:
            self._intent, self._run, self._version = intent, run, 1
            self._held.extend(held)

    @staticmethod
    def _row(r: dict) -> dict:
        """A recent-runs row from saved JSON, with every key typed."""
        if not isinstance(r, dict) or not r.get("run_id"):
            raise ValueError("run row without run_id")
        return {"run_id": _text(r["run_id"]), "text": _text(r.get("text")),
                "from": _text(r.get("from")), "to": _text(r.get("to")),
                "task": _text(r.get("task")), "target": _text(r.get("target")),
                "started_ns": int(r.get("started_ns") or 0),
                "ended_ns": int(r.get("ended_ns") or 0),
                "outcome": _text(r.get("outcome")),
                "interruptions": int(r.get("interruptions") or 0),
                "stops": SessionNotes._log(r.get("stops"), MAX_STOPS,
                                           ("cause", "detail")),
                "last_stop": next(iter(SessionNotes._log(
                    [r.get("last_stop")], 1, ("cause", "detail"))), None),
                "resumes": SessionNotes._log(r.get("resumes"), MAX_RESUMES,
                                             ("note",)),
                "end_note": _text(r.get("end_note"))}

    @staticmethod
    def _log(items, cap: int, keys: tuple) -> list[dict]:
        """A saved stop / resume log, typed; anything that is not a list of
        objects (a state file from before the logs existed) is empty."""
        if not isinstance(items, list):
            return []
        out = [{"t_ns": int(x.get("t_ns") or 0), **{k: _text(x.get(k)) for k in keys}}
               for x in items if isinstance(x, dict)]
        return out[-cap:]

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
            self._queue_event_locked(ev)
        return ev

    def _queue_event_locked(self, ev: dict) -> None:
        if len(self._events) == self._events.maxlen:
            self.events_dropped += 1
        self._events.append(ev)

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
            return {"recording": self._segment is not None,
                    "intent": self._intent_dict_locked(),
                    "recent": self._recent_locked(),
                    "events_pending": len(self._events),
                    "events_held": len(self._held),
                    "places_pending": len(self._places),
                    "events_dropped": self.events_dropped,
                    "places_dropped": self.places_dropped}
