"""RouterService: the System 1 loop that runs while the robot is powered.

It is event-triggered by a slow clock (not per-frame): every `interval_s` it
takes one fresh view bundle, asks the skill to decide, and lets the executor
act if the decision is actionable.  Only one assessment runs at a time — a
slow VLM must never let decisions pile up, and a manual (app) assessment takes
precedence over the background tick.

Bounded by design:
  * one in-flight decision (stale ticks are dropped, not queued),
  * a short rolling event history fed into the next observation,
  * a bounded decision log for the app and for future training data.
"""
from __future__ import annotations

import threading
import time
from collections import deque

from .schema import DEFER, SKIP


def _log(msg: str) -> None:
    print(f"[router] {msg}", flush=True)


class RouterService:
    """Owns the router thread and the decision log."""

    def __init__(
        self,
        skill,
        executor,
        capture=None,
        *,
        interval_s: float = 20.0,
        initial_delay_s: float = 5.0,
        history: int = 50,
        events: int = 12,
        clock=time.monotonic,
    ) -> None:
        self.skill = skill
        self.executor = executor
        self.capture = capture
        self.interval_s = float(interval_s)
        self.initial_delay_s = float(initial_delay_s)
        self._clock = clock
        self._history: deque[dict] = deque(maxlen=int(history))
        self._events: deque[str] = deque(maxlen=int(events))
        self._lock = threading.Lock()
        self._busy = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._assessments = 0
        self._applied = 0

    # -- lifecycle --------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="router-loop", daemon=True
        )
        self._thread.start()
        _log(f"started (every {self.interval_s:.0f}s, vlm={self.skill.vlm.describe})")

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
        self._thread = None

    def _loop(self) -> None:
        delay = self.initial_delay_s
        while not self._stop.wait(delay):
            delay = self.interval_s
            try:
                self.assess_once(source="loop")
            except Exception as exc:  # noqa: BLE001 - a tick must never kill the loop
                _log(f"tick error: {exc}")

    # -- one decision -----------------------------------------------------

    def assess_once(
        self,
        *,
        instruction: str | None = None,
        views=None,
        apply: bool = True,
        source: str = "manual",
    ) -> dict:
        """Take one decision. Returns a JSON-ready result; never raises."""
        with self._lock:
            if self._busy:
                return {"ok": False, "skipped": "busy"}
            self._busy = True
        try:
            if views is None:
                views = list(self.capture.grab()) if self.capture is not None else []
            if not views:
                rec = self._record(
                    source=source, decision=DEFER, reason_code="no_frames",
                    latency_ms=0, applied=None,
                )
                return {"ok": False, "reason": "no_frames", "record": rec}

            assessment = self.skill.assess(
                views,
                instruction=instruction,
                recent_events=list(self._events),
            )
            outcome = None
            if apply:
                outcome = self.executor.apply(
                    assessment.decision, views, instruction=instruction
                )
                assessment.applied = outcome
            self._assessments += 1
            if outcome and outcome.get("applied"):
                self._applied += 1
            self._note_event(assessment, outcome)
            rec = self._record(
                source=source,
                decision=assessment.decision.decision,
                reason_code=assessment.decision.reason_code,
                latency_ms=assessment.latency_ms,
                applied=outcome or {},
                place_id=(outcome or {}).get("place_id"),
                error=assessment.error,
            )
            out = {"ok": True, "record": rec, **assessment.to_dict()}
            return out
        finally:
            with self._lock:
                self._busy = False

    def _note_event(self, assessment, outcome) -> None:
        decision = assessment.decision
        if outcome and outcome.get("applied"):
            action = outcome.get("action")
            name = outcome.get("place_name") or outcome.get("place_id")
            self._events.append(f"{action} {name!r} ({decision.reason_code})")
        elif decision.decision in (SKIP, DEFER):
            self._events.append(f"{decision.decision}: {decision.reason_code}")
        else:
            self._events.append(
                f"{decision.decision} not applied: "
                f"{(outcome or {}).get('reason_code', 'skipped')}"
            )

    def _record(
        self,
        *,
        source: str,
        decision: str,
        reason_code: str,
        latency_ms: int,
        applied: dict | None,
        place_id: str | None = None,
        error: str | None = None,
    ) -> dict:
        rec = {
            "ts": time.time(),
            "source": source,
            "decision": decision,
            "reason_code": reason_code,
            "latency_ms": int(latency_ms),
            "applied": bool(applied and applied.get("applied")),
            "place_id": place_id,
        }
        if applied:
            rec["outcome"] = applied
        if error:
            rec["error"] = error[:200]
        self._history.append(rec)
        return rec

    # -- introspection ----------------------------------------------------

    def decisions(self, limit: int = 20) -> list[dict]:
        items = list(self._history)
        if limit > 0:
            items = items[-int(limit):]
        return list(reversed(items))

    def events(self) -> list[str]:
        return list(self._events)

    def status(self) -> dict:
        vlm = getattr(self.skill, "vlm", None)
        running = bool(self._thread is not None and self._thread.is_alive())
        return {
            "enabled": True,
            "running": running,
            "interval_s": self.interval_s,
            "cooldown_s": getattr(self.executor, "cooldown_s", None),
            "vlm": getattr(vlm, "describe", None),
            "assessments": self._assessments,
            "applied": self._applied,
            "busy": self._busy,
            "events": self.events(),
        }
