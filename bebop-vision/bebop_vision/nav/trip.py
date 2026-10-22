"""Trip runner: a multi-leg journey over the place graph.

    plan   -> `PlaceStore.plan_trip(from, to)` (Dijkstra over taught routes)
    drive  -> one `RouteFollower.follow(route_id)` per leg, in order
    handover: when a leg ends `arrived` (or `arrived_unconfirmed`), the next
              leg starts from where the robot stands; the follower re-anchors
              its route frame there and the visual correction absorbs the
              small heading difference between one route's end and the
              next route's start
    stop   -> any failed leg fails the trip; the operator can stop any time

The start place comes from the request or from place recognition.  Each
leg's outcome is already recorded on the graph by the follower, so later
plans prefer legs that have actually worked.
"""
from __future__ import annotations

import threading
import time

TERMINAL_OK = ("arrived", "arrived_unconfirmed")


class TripRunner:
    def __init__(self, follower, store, *, recognize=None, hz: float = 2.0,
                 min_success_rate: float = 0.0, log=print):
        self.follower = follower
        self.store = store
        self.recognize = recognize
        self.hz = max(0.5, float(hz))
        self.min_success_rate = float(min_success_rate)
        self._log = log
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._state = "idle"
        self._plan: dict | None = None
        self._leg = 0
        self._leg_started = False
        self._t0 = 0.0
        self._history: list = []
        self._detail: dict = {}

    # -- thread ---------------------------------------------------------------

    def start_thread(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, name="trip",
                                            daemon=True)
            self._thread.start()
        return self

    def stop_thread(self, timeout: float = 3.0):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def _loop(self):
        period = 1.0 / self.hz
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                self.tick()
            except Exception as exc:  # noqa: BLE001
                self._log(f"[trip] tick error: {type(exc).__name__}: {exc}")
                with self._lock:
                    if self._state == "running":
                        self._finish(f"failed: {exc}")
            time.sleep(max(0.0, period - (time.monotonic() - t0)))

    # -- API ----------------------------------------------------------------------

    def start(self, to_place: str, *, from_place: str | None = None) -> dict:
        with self._lock:
            if self._state == "running":
                raise RuntimeError("a trip is already running")
            start = from_place
            if not start and self.recognize is not None:
                rec = self.recognize() or {}
                if rec.get("place"):
                    start = rec["place"]["id"]
                    self._detail = {"start_recognized": rec["place"].get("name"),
                                    "start_score": rec.get("score")}
            if not start:
                raise ValueError("cannot tell where we are: pass from_place")
            plan = self.store.plan_trip(start, to_place,
                                        min_success_rate=self.min_success_rate)
            if not plan["legs"]:
                if plan["reachable"]:
                    self._state = "arrived"
                    self._plan = plan
                    self._history.append({"state": "arrived", "legs": 0,
                                          "to": plan["to_name"],
                                          "duration_s": 0.0, "ts": time.time()})
                    return self.status()
                raise LookupError(
                    f"no taught route chain from {plan['from_name']!r} to "
                    f"{plan['to_name']!r}")
            self._plan = plan
            self._leg = 0
            self._leg_started = False
            self._state = "running"
            self._t0 = time.monotonic()
            self._log(f"[trip] {plan['from_name']!r} -> {plan['to_name']!r}: "
                      f"{len(plan['legs'])} legs, ~{plan['cost_s']} s")
            return self.status()

    def stop(self) -> dict:
        with self._lock:
            if self._state == "running":
                try:
                    self.follower.stop()
                except Exception:  # noqa: BLE001
                    pass
                self._finish("stopped")
            return self.status()

    def status(self) -> dict:
        with self._lock:
            p = self._plan
            return {
                "state": self._state,
                "plan": p,
                "leg": (self._leg if self._state == "running" else None),
                "legs": (len(p["legs"]) if p else 0),
                "elapsed_s": (round(time.monotonic() - self._t0, 1)
                              if self._state == "running" else None),
                "follower": self.follower.status() if self.follower else None,
                "detail": dict(self._detail),
                "history": list(self._history[-10:]),
            }

    # -- tick ---------------------------------------------------------------------

    def tick(self) -> None:
        with self._lock:
            if self._state != "running":
                return
            legs = self._plan["legs"]
            leg = legs[self._leg]
            if not self._leg_started:
                self._log(f"[trip] leg {self._leg + 1}/{len(legs)}: "
                          f"{leg['from_name']!r} -> {leg['to_name']!r} "
                          f"({leg['route_id']})")
                try:
                    self.follower.follow(leg["route_id"])
                except Exception as exc:  # noqa: BLE001
                    self._finish(f"failed: leg {self._leg + 1} could not "
                                 f"start: {exc}")
                    return
                self._leg_started = True
                return
            fstate = self.follower.status().get("state")
            if fstate == "following":
                return
            if fstate in TERMINAL_OK:
                self._detail[f"leg{self._leg + 1}"] = fstate
                if self._leg + 1 >= len(legs):
                    self._finish("arrived" if fstate == "arrived"
                                 else "arrived_unconfirmed")
                    return
                self._leg += 1
                self._leg_started = False
                return
            self._finish(f"failed: leg {self._leg + 1} {fstate}")

    def _finish(self, state: str):
        p = self._plan
        entry = {"state": state, "to": p["to_name"] if p else None,
                 "from": p["from_name"] if p else None,
                 "legs": len(p["legs"]) if p else 0,
                 "leg_reached": self._leg,
                 "duration_s": round(time.monotonic() - self._t0, 1),
                 "ts": time.time()}
        self._history.append(entry)
        del self._history[:-50]
        self._state = state
        self._log(f"[trip] {entry}")
