"""One command in, continuous waypoints out.

    navigator.command("go to the kitchen")
    navigator.command("go to the dresser")
    navigator.command("forward 1 meter") / ("back half a metre") / ("turn left 90")
    navigator.command("stop")

The navigator resolves the words to a mechanism and runs it; the driver
keeps receiving waypoints until the mission ends:

* a taught **place** name (fuzzy match against place memory) -> a multi-leg
  trip over the place graph when a route chain exists from where the robot
  recognises itself to be; with no chain, the place name is handed to the
  object seeker as a description ("the kitchen doorway") so vision can still
  try;
* anything else -> the **object seeker** (VLM box -> bearing + range, one
  look per leg);
* **relative** moves -> one bounded driver goal.

`status()` is a single view over whichever runner is active, and `stop()`
stops everything.  Voice and text both land here.
"""
from __future__ import annotations

import difflib
import re
import threading
import time

_LEAD = re.compile(
    r"^\s*(?:please\s+)?(?:can you\s+|could you\s+|would you\s+)?"
    r"(?:go|navigate|drive|head|take me|bring me|move|walk|come|get|find|"
    r"locate|roll)\s*(?:over\s+)?(?:to|towards|toward|into|in to|at)?\s*",
    re.I)
_THE = re.compile(r"^\s*(?:the|a|an|my|our)\s+", re.I)
_STOP = re.compile(r"^\s*(?:stop|halt|cancel|abort|wait|freeze)\b", re.I)
_REL = re.compile(
    r"^\s*(?:go\s+|move\s+|drive\s+|roll\s+)?(forward|forwards|ahead|straight|"
    r"back|backward|backwards|reverse)\s*(?:by\s+)?([\d.]+|a|one|half a|half)?"
    r"\s*(m|meter|meters|metre|metres|cm|centimeters|centimetres|feet|foot|ft)?",
    re.I)
_TURN = re.compile(
    r"^\s*(?:turn|rotate|spin|face)\s*(?:to the\s+)?(left|right|around)\s*"
    r"(?:by\s+)?([\d.]+)?\s*(deg|degrees|°)?", re.I)
_WORDNUM = {"a": 1.0, "one": 1.0, "half a": 0.5, "half": 0.5}


def parse_command(text: str, place_names: list[str]) -> dict:
    """Pure resolver: text -> {kind, target, ...}.  No I/O."""
    raw = (text or "").strip()
    if not raw:
        raise ValueError("empty command")
    if _STOP.match(raw):
        return {"kind": "stop", "command": raw}
    m = _TURN.match(raw)
    if m:
        side, amount, _unit = m.groups()
        deg = float(amount) if amount else (180.0 if side == "around" else 90.0)
        deg = max(5.0, min(180.0, deg))
        sign = 1.0 if side == "left" else (-1.0 if side == "right" else 1.0)
        return {"kind": "turn", "turn_deg": sign * deg, "command": raw}
    m = _REL.match(raw)
    if m:
        direction, amount, unit = m.groups()
        val = 1.0
        if amount:
            val = _WORDNUM.get(amount.lower(), None)
            if val is None:
                val = float(amount)
        unit = (unit or "m").lower()
        if unit.startswith("c"):
            val /= 100.0
        elif unit in ("feet", "foot", "ft"):
            val *= 0.3048
        val = max(0.1, min(3.0, val))
        if direction.lower().startswith(("back", "rev")):
            return {"kind": "reverse", "distance_m": min(val, 1.0),
                    "command": raw}
        return {"kind": "relative", "bearing_deg": 0.0, "distance_m": val,
                "command": raw}
    target = _LEAD.sub("", raw, count=1).strip().rstrip(".!?")
    target = _THE.sub("", target).strip() or raw
    # Place match: exact / substring / fuzzy against taught names.
    low = target.lower()
    best, best_score = None, 0.0
    for name in place_names:
        n = name.lower()
        score = 1.0 if n == low else (0.9 if (n in low or low in n) and
                                       min(len(n), len(low)) >= 4 else
                                       difflib.SequenceMatcher(None, n, low).ratio())
        if score > best_score:
            best, best_score = name, score
    if best is not None and best_score >= 0.72:
        return {"kind": "place", "target": best, "score": round(best_score, 2),
                "command": raw}
    return {"kind": "object", "target": target, "command": raw}


class Navigator:
    def __init__(self, drive, store, *, trip=None, seeker=None, recognize=None,
                 log=print):
        self.drive = drive
        self.store = store
        self.trip = trip
        self.seeker = seeker
        self.recognize = recognize
        self._log = log
        self._lock = threading.RLock()
        self._mission: dict | None = None
        self._t0 = 0.0
        self._history: list = []

    # -- commands -------------------------------------------------------------

    def command(self, text: str) -> dict:
        names = [p.name for p in self.store.list_places(status="confirmed")]
        intent = parse_command(text, names)
        with self._lock:
            if intent["kind"] == "stop":
                self.stop()
                return self.status()
            if self._active():
                self._log(f"[navigate] replacing active mission with {text!r}")
                self._stop_runners()
            intent["started_at"] = time.time()
            self._t0 = time.monotonic()
            if intent["kind"] == "place":
                self._start_place(intent)
            elif intent["kind"] == "object":
                self._start_object(intent)
            elif intent["kind"] == "turn":
                self.drive.set_goal(turn_deg=intent["turn_deg"],
                                    label=f"cmd:{text[:24]}")
                intent["runner"] = "goal"
            elif intent["kind"] == "reverse":
                self.drive.set_goal(reverse_m=intent["distance_m"],
                                    label=f"cmd:{text[:24]}")
                intent["runner"] = "goal"
            else:
                self.drive.set_goal(bearing_deg=intent["bearing_deg"],
                                    distance_m=intent["distance_m"],
                                    label=f"cmd:{text[:24]}")
                intent["runner"] = "goal"
            self._mission = intent
            self._log(f"[navigate] {text!r} -> {intent['kind']} "
                      f"{intent.get('target', '')} via {intent['runner']}")
            return self.status()

    def _start_place(self, intent):
        target = intent["target"]
        here = None
        if self.recognize is not None:
            try:
                rec = self.recognize() or {}
            except Exception:  # noqa: BLE001
                rec = {}
            if rec.get("place"):
                here = rec["place"]
                intent["here"] = here.get("name")
        if here and here.get("name", "").lower() == target.lower():
            intent["runner"] = "none"
            intent["note"] = "already there"
            return
        if here and self.trip is not None:
            try:
                plan = self.store.plan_trip(here["id"], target)
            except KeyError:
                plan = {"legs": []}
            if plan["legs"]:
                self.trip.start(target, from_place=here["id"])
                intent["runner"] = "trip"
                intent["legs"] = len(plan["legs"])
                return
            intent["note"] = (f"no taught route from {here['name']!r} to "
                              f"{target!r}; looking for it instead")
        else:
            intent["note"] = ("current place not recognised; looking for "
                              "it instead")
        if self.seeker is None:
            raise RuntimeError(intent["note"] + " (no object seeker)")
        self.seeker.seek(f"the {target}")
        intent["runner"] = "seek"

    def _start_object(self, intent):
        if self.seeker is None:
            raise RuntimeError("object goals are not configured")
        what = intent["target"]
        if not what.lower().startswith(("the ", "a ", "an ")):
            what = "the " + what
        self.seeker.seek(what)
        intent["runner"] = "seek"

    # -- lifecycle ------------------------------------------------------------------

    def _stop_runners(self):
        for r in (self.trip, self.seeker):
            if r is not None:
                try:
                    r.stop()
                except Exception:  # noqa: BLE001
                    pass
        try:
            self.drive.cancel()
        except Exception:  # noqa: BLE001
            pass

    def stop(self) -> dict:
        with self._lock:
            was = self._active()
            self._stop_runners()
            if self._mission is not None and was:
                self._mission["stopped"] = True
            return self.status()

    def _runner_state(self) -> str:
        m = self._mission
        if m is None:
            return "idle"
        if m.get("stopped"):
            return "stopped"
        r = m.get("runner")
        if r == "none":
            return "arrived"
        if r == "trip" and self.trip is not None:
            s = self.trip.status()["state"]
            return "running" if s == "running" else s
        if r == "seek" and self.seeker is not None:
            s = self.seeker.status()["state"]
            return "running" if s == "seeking" else s
        if r == "goal":
            d = self.drive.status()
            if d.get("active") or d.get("arming"):
                return "running"
            st = d.get("state", "idle")
            return {"reached": "arrived", "idle": "running"}.get(st, st)
        return "idle"

    def _active(self) -> bool:
        return self._runner_state() == "running"

    def status(self) -> dict:
        with self._lock:
            m = self._mission
            state = self._runner_state()
            runner = None
            if m:
                if m.get("runner") == "trip" and self.trip is not None:
                    runner = self.trip.status()
                elif m.get("runner") == "seek" and self.seeker is not None:
                    runner = self.seeker.status()
                elif m.get("runner") == "goal":
                    d = self.drive.status()
                    runner = {"state": d.get("state"),
                              "detail": d.get("command", {}).get("detail")}
            return {
                "state": state,
                "mission": dict(m) if m else None,
                "elapsed_s": (round(time.monotonic() - self._t0, 1)
                              if m and state == "running" else None),
                "runner": runner,
                "summary": self.summary(state, m, runner),
            }

    @staticmethod
    def summary(state: str, m: dict | None, runner: dict | None) -> str:
        """One spoken sentence for the voice tool."""
        if m is None:
            return "No navigation command has been given."
        what = m.get("target") or m.get("command")
        if state == "running":
            if m.get("runner") == "trip" and runner:
                leg = runner.get("leg")
                legs = runner.get("legs")
                return (f"On the way to {what}, leg "
                        f"{(leg or 0) + 1} of {legs}.")
            if m.get("runner") == "seek" and runner:
                last = runner.get("last") or {}
                rng = last.get("range_m")
                if runner.get("scanning"):
                    return f"Looking around for {what}."
                return (f"Heading for {what}"
                        + (f", about {rng:.1f} metres away." if rng else
                           ", looking for it."))
            return f"Driving: {m.get('command')}."
        if state == "arrived":
            if m.get("runner") == "goal":
                return f"Done: {m.get('command')}."
            if m.get("note") == "already there":
                return f"Already at {what}."
            return f"Arrived at {what}."
        if state == "arrived_unconfirmed":
            return f"Reached the end of the route to {what}, but I do not recognise it."
        if state == "stopped":
            return f"Stopped on the way to {what}."
        return f"Could not get to {what}: {state.replace('_', ' ')}."
