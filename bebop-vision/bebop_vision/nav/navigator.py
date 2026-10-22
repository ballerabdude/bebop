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
_SEQ_SPLIT = re.compile(r"(\s*(?:,\s*)?\b(?:and then|then|and|after that)\b\s*|\s*[.;]\s+)", re.I)
_FILLER = [
    re.compile(r"^(?:way |far |over |off )?(?:in|into|in to) the distance,?\s*", re.I),
    re.compile(r"^there(?:'s| is| are)\s+", re.I),
    re.compile(r"[.,]?\s*(?:and\s+)?(?:please\s+)?(?:navigate|go|drive|head|move)\s+(?:to|towards|toward)\s+(?:it|that|there)\.?$", re.I),
    re.compile(r"^(?:you(?:'ll| will) see|you can see)\s+", re.I),
]


def clean_target(text: str) -> str:
    out = text.strip()
    for _ in range(2):
        for rx in _FILLER:
            out = rx.sub("", out).strip(" ,.")
    return out or text.strip()


def split_steps(text: str) -> list[str]:
    """'turn around and go to the window' -> ['turn around', 'go to the window'].

    Splits on then/and/sentence breaks, but only keeps a split when the
    first part is a move or turn (so 'the chair and the table' stays one
    object description)."""
    raw = (text or "").strip()
    pieces = _SEQ_SPLIT.split(raw)           # text, sep, text, sep, ...
    texts = [pieces[i] for i in range(0, len(pieces), 2)]
    seps = [pieces[i] for i in range(1, len(pieces), 2)]
    if len(texts) <= 1:
        return [raw]
    steps = []
    buf = texts[0].strip(" ,.")
    for sep, nxt in zip(seps, texts[1:]):
        nxt = nxt.strip(" ,.")
        if not nxt:
            continue
        if not buf or _TURN.match(buf) or _REL.match(buf) or _STOP.match(buf):
            if buf:
                steps.append(buf)
            buf = nxt
        else:
            buf = f"{buf}{sep}{nxt}"         # keep the original joiner
    if buf:
        steps.append(buf)
    return steps or [raw]


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
    target = _LEAD.sub("", clean_target(raw), count=1).strip().rstrip(".!?")
    target = clean_target(target)
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
        self._steps: list = []            # remaining step texts
        self._step_no = 0
        self._step_total = 0
        self._command_text = ""
        self._stop_evt = threading.Event()
        self._thread: threading.Thread | None = None

    # -- thread: advances multi-step commands --------------------------------

    def start_thread(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, name="navigator",
                                            daemon=True)
            self._thread.start()
        return self

    def stop_thread(self, timeout: float = 3.0):
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def _loop(self):
        while not self._stop_evt.is_set():
            try:
                self.tick()
            except Exception as exc:  # noqa: BLE001
                self._log(f"[navigate] tick error: {type(exc).__name__}: {exc}")
            time.sleep(0.5)

    def tick(self) -> None:
        """Recover a blocked turn, then start the next step when the
        current one has arrived."""
        with self._lock:
            if self._mission is None:
                return
            state = self._runner_state()
            if state == "running":
                return
            if self._recover_turn(state):
                return
            if not self._steps:
                return
            if state in ("arrived", "arrived_unconfirmed"):
                nxt = self._steps.pop(0)
                self._step_no += 1
                self._log(f"[navigate] step {self._step_no}/{self._step_total}: "
                          f"{nxt!r}")
                try:
                    self._start_step(nxt)
                except Exception as exc:  # noqa: BLE001
                    self._mission = dict(self._mission or {}, failed=str(exc))
                    self._steps = []
            else:
                # The step failed or was stopped: drop the rest.
                self._steps = []

    def _recover_turn(self, state: str) -> bool:
        """A turn refused for lack of sweep clearance: back up a little and
        try once more (the seeker does the same for its scan turns)."""
        m = self._mission
        if m is None or m.get("runner") != "goal" or m.get("kind") != "turn":
            return False
        phase = m.get("recover")
        if state == "aborted_blocked" and phase is None:
            try:
                self.drive.set_goal(reverse_m=0.4, label="unblock:turn")
            except RuntimeError:
                return False
            m["recover"] = "reversing"
            self._log("[navigate] turn blocked: backing up 0.4 m to retry")
            return True
        if phase == "reversing" and state == "arrived":
            try:
                self.drive.set_goal(turn_deg=m["turn_deg"],
                                    label=f"cmd:{m['command'][:24]}")
            except RuntimeError:
                return False
            m["recover"] = "retried"
            self._log("[navigate] retrying the turn")
            return True
        return False

    # -- commands -------------------------------------------------------------

    def command(self, text: str) -> dict:
        text = (text or "").strip()
        if _STOP.match(text):
            with self._lock:
                self.stop()
                return self.status()
        steps = split_steps(text)
        with self._lock:
            if self._active():
                self._log(f"[navigate] replacing active mission with {text!r}")
                self._stop_runners()
            self._command_text = text
            self._steps = steps[1:]
            self._step_no = 1
            self._step_total = len(steps)
            self._t0 = time.monotonic()
            self._start_step(steps[0])
            return self.status()

    def _start_step(self, text: str) -> None:
        names = [p.name for p in self.store.list_places(status="confirmed")]
        intent = parse_command(text, names)
        if intent["kind"] == "stop":
            self._steps = []
            self._stop_runners()
            self._mission = dict(intent, runner="none", stopped=True)
            return
        intent["started_at"] = time.time()
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
        intent["step"] = self._step_no
        intent["steps"] = self._step_total
        intent["full_command"] = self._command_text
        self._mission = intent
        self._log(f"[navigate] {text!r} -> {intent['kind']} "
                  f"{intent.get('target', '')} via {intent['runner']} "
                  f"(step {self._step_no}/{self._step_total})")

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
            self._steps = []
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
        if m.get("failed"):
            return "failed: " + m["failed"]
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
            if state in ("arrived", "arrived_unconfirmed") and self._steps:
                state = "running"           # next step starts on the tick
            if (m and m.get("kind") == "turn" and m.get("recover") == "reversing"
                    and state == "arrived"):
                state = "running"           # the retry starts on the tick
            if (m and m.get("kind") == "turn" and m.get("recover") is None
                    and state == "aborted_blocked"):
                state = "running"           # the recovery starts on the tick
            return {
                "state": state,
                "mission": dict(m) if m else None,
                "steps_left": len(self._steps),
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
        prefix = ""
        if (m.get("steps") or 1) > 1:
            prefix = f"Step {m.get('step')} of {m.get('steps')}: "
        if state == "running":
            return prefix + Navigator._running_summary(m, runner, what)
        if state == "arrived":
            if m.get("runner") == "goal":
                return prefix + f"Done: {m.get('command')}."
            if m.get("note") == "already there":
                return prefix + f"Already at {what}."
            return prefix + f"Arrived at {what}."
        if state == "arrived_unconfirmed":
            return prefix + (f"Reached the end of the route to {what}, but I do "
                             "not recognise it.")
        if state == "stopped":
            return prefix + f"Stopped on the way to {what}."
        return prefix + f"Could not get to {what}: {state.replace('_', ' ')}."

    @staticmethod
    def _running_summary(m, runner, what) -> str:
        if True:
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
