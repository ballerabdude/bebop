"""RouterService: the System 1 loop that runs while the robot is powered.

It is triggered by a slow clock (not per-frame): every `interval_s` it takes
one fresh view bundle and evaluates each registered skill in turn.  Place
memory and map capture are both assessed on every tick, so the shared brain
gets to weigh in even when the scene is unchanged; an optional
``event_trigger`` hook remains for a future skill that genuinely needs event
gating.

Multiple skills share one thread, one in-flight guard, one event history and
one decision log.  The constructor keeps the original single ``(skill,
executor)`` form; pass extra pairs via ``skills=[...]`` (or a list of pairs as
the only positional argument) to register more.

Bounded by design:
  * one in-flight decision across all skills (stale ticks are dropped, not
    queued; skills on a tick are evaluated sequentially, never concurrently),
  * a short rolling event history shared by every skill,
  * a bounded decision log for the app and for future training data,
  * background ticks are skipped while a voice turn is in flight.
"""
from __future__ import annotations

import re
import threading
import time
from collections import deque
from dataclasses import dataclass

from .schema import DEFER, REJECT_CAPTURE, SKIP


def _log(msg: str) -> None:
    print(f"[router] {msg}", flush=True)


def _skill_name(skill) -> str:
    """A stable short name for status/logs (`CaptureMapSkill` -> capture_map)."""
    cls = type(skill).__name__
    if cls.endswith("Skill"):
        cls = cls[:-5]
    snake = re.sub(r"(?<!^)(?=[A-Z])", "_", cls).lower()
    return snake or "skill"


@dataclass(frozen=True)
class SkillRegistration:
    """One skill + the executor that applies its decisions."""

    skill: object
    executor: object
    name: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            object.__setattr__(self, "name", _skill_name(self.skill))


def _as_registration(item, *, name: str = "") -> SkillRegistration:
    if isinstance(item, SkillRegistration):
        return item
    if isinstance(item, dict):
        return SkillRegistration(
            item["skill"], item["executor"], item.get("name") or name)
    if isinstance(item, (tuple, list)) and len(item) == 2:
        return SkillRegistration(item[0], item[1], name)
    raise ValueError(
        "a router registration must be (skill, executor), SkillRegistration, "
        f"or a mapping; got {type(item).__name__}")


def _registration_items(value) -> list:
    """Normalise one or many registrations into a list of items."""
    if value is None:
        return []
    if isinstance(value, (SkillRegistration, dict)):
        return [value]
    if (isinstance(value, (tuple, list)) and len(value) == 2
            and not isinstance(value[0], (tuple, list, SkillRegistration, dict))):
        return [value]
    return list(value)


class RouterService:
    """Owns the router thread and the shared decision log."""

    def __init__(
        self,
        skill,
        executor=None,
        capture=None,
        *,
        skills=None,
        interval_s: float = 20.0,
        initial_delay_s: float = 5.0,
        history: int = 50,
        events: int = 12,
        voice_busy=None,
        clock=time.monotonic,
    ) -> None:
        # Backward-compatible forms:
        #   RouterService(skill, executor, capture=...)
        #   RouterService(skill, executor, skills=[(skill2, executor2)], ...)
        #   RouterService([(skill1, executor1), (skill2, executor2)], ...)
        if executor is None and isinstance(skill, (SkillRegistration, dict)):
            items = [skill]
        elif executor is None and isinstance(skill, (list, tuple)):
            items = _registration_items(skill)
        else:
            items = [(skill, executor)]
        items.extend(_registration_items(skills))
        registrations = [_as_registration(item) for item in items]
        if not registrations:
            raise ValueError("RouterService needs at least one skill registration")

        self.registrations: list[SkillRegistration] = registrations
        # The first registration is the primary: manual assessments and the
        # legacy `skill`/`executor` attributes keep pointing at it.
        self.skill = registrations[0].skill
        self.executor = registrations[0].executor
        self.capture = capture
        self.interval_s = float(interval_s)
        self.initial_delay_s = float(initial_delay_s)
        self._clock = clock
        self._voice_busy = voice_busy
        self._history: deque[dict] = deque(maxlen=int(history))
        self._events: deque[str] = deque(maxlen=int(events))
        self._lock = threading.Lock()
        self._busy = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._assessments = 0
        self._applied = 0
        self._last_trigger: str | None = None
        self._last_triggers: dict[str, str] = {}

    # -- lifecycle --------------------------------------------------------

    @property
    def skills(self) -> list[SkillRegistration]:
        """All registered skills (alias for ``registrations``)."""
        return list(self.registrations)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="router-loop", daemon=True
        )
        self._thread.start()
        names = ", ".join(r.name for r in self.registrations)
        vlm = getattr(self.registrations[0].skill, "vlm", None)
        _log(f"started (every {self.interval_s:.0f}s, skills=[{names}], "
             f"vlm={getattr(vlm, 'describe', None)})")

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
                self.tick()
            except Exception as exc:  # noqa: BLE001 - a tick must never kill the loop
                _log(f"tick error: {exc}")

    def tick(self) -> dict:
        """One background evaluation of every registered skill.

        Skills without ``event_trigger`` (place memory **and** map capture)
        are assessed on **every** tick, so the brain weighs in even on an
        unchanged scene.  The ``event_trigger`` hook remains available for a
        future skill that genuinely wants event gating.  The whole tick is
        skipped while a voice turn is in flight.  Registrations are evaluated
        sequentially under the shared one-in-flight guard, so a slow brain
        call can never be queued behind or raced by another skill.
        """
        if self._voice_busy is not None:
            try:
                if self._voice_busy():
                    return {"ok": False, "skipped": "voice_busy"}
            except Exception as exc:  # noqa: BLE001 - never block on telemetry
                _log(f"voice_busy check failed: {exc}")

        views = list(self.capture.grab()) if self.capture is not None else []
        results: list[dict] = []
        for registration in self.registrations:
            trigger_hook = getattr(registration.skill, "event_trigger", None)
            if callable(trigger_hook):
                try:
                    trigger = trigger_hook(views)
                except Exception as exc:  # noqa: BLE001 - one skill must not
                    _log(f"{registration.name} event_trigger failed: {exc}")
                    trigger = None
                if not trigger:
                    results.append({
                        "ok": False,
                        "skipped": "not_triggered",
                        "skill": registration.name,
                    })
                    continue
                self._last_triggers[registration.name] = str(trigger)
                self._last_trigger = str(trigger)
                result = self._assess_registration(
                    registration, views=views, source=f"loop:{trigger}")
            else:
                result = self._assess_registration(
                    registration, views=views, source="loop")
            result.setdefault("skill", registration.name)
            results.append(result)

        if len(self.registrations) == 1:
            return results[0]
        return {
            "ok": any(bool(result.get("ok")) for result in results),
            "evaluated": [result.get("skill") for result in results],
            "results": results,
            "skipped": [result for result in results
                        if not result.get("ok")],
        }

    # -- one decision -----------------------------------------------------

    def assess_once(
        self,
        *,
        instruction: str | None = None,
        views=None,
        apply: bool = True,
        source: str = "manual",
        skill=None,
    ) -> dict:
        """Take one decision from a skill (the primary unless `skill` is set).

        Returns a JSON-ready result; never raises.
        """
        registration = self._registration_for(skill)
        return self._assess_registration(
            registration, instruction=instruction, views=views,
            apply=apply, source=source)

    def _registration_for(self, skill) -> SkillRegistration:
        if skill is None:
            return self.registrations[0]
        if isinstance(skill, str):
            for registration in self.registrations:
                if registration.name == skill:
                    return registration
        for registration in self.registrations:
            if registration.skill is skill:
                return registration
        raise KeyError(f"no such router skill: {skill!r}")

    def skill_names(self) -> list[str]:
        """Registered skill names, in evaluation order."""
        return [registration.name for registration in self.registrations]

    def _assess_registration(
        self,
        registration: SkillRegistration,
        *,
        instruction: str | None = None,
        views=None,
        apply: bool = True,
        source: str = "manual",
    ) -> dict:
        with self._lock:
            if self._busy:
                return {"ok": False, "skipped": "busy",
                        "skill": registration.name}
            self._busy = True
        try:
            if views is None:
                views = list(self.capture.grab()) if self.capture is not None else []
            if not views:
                rec = self._record(
                    source=source, decision=DEFER, reason_code="no_frames",
                    latency_ms=0, applied=None, skill=registration.name,
                )
                return {"ok": False, "reason": "no_frames", "record": rec,
                        "skill": registration.name}

            assessment = registration.skill.assess(
                views,
                instruction=instruction,
                recent_events=list(self._events),
            )
            outcome = None
            if apply:
                outcome = registration.executor.apply(
                    assessment.decision, views, instruction=instruction
                )
                assessment.applied = outcome
                note = getattr(registration.skill, "note_outcome", None)
                if callable(note):
                    note(outcome)
            self._assessments += 1
            if outcome and outcome.get("applied"):
                self._applied += 1
            self._note_event(assessment, outcome, registration.name)
            rec = self._record(
                source=source,
                decision=assessment.decision.decision,
                reason_code=assessment.decision.reason_code,
                latency_ms=assessment.latency_ms,
                applied=outcome or {},
                place_id=(outcome or {}).get("place_id"),
                error=assessment.error,
                skill=registration.name,
                gate_reason=(assessment.decision.arguments or {}).get(
                    "gate_reason"),
            )
            out = {"ok": True, "record": rec, **assessment.to_dict()}
            return out
        finally:
            with self._lock:
                self._busy = False

    def _note_event(self, assessment, outcome, name: str) -> None:
        decision = assessment.decision
        prefix = f"[{name}] " if len(self.registrations) > 1 else ""
        if outcome and outcome.get("applied"):
            action = outcome.get("action")
            place = outcome.get("place_name") or outcome.get("place_id")
            self._events.append(f"{prefix}{action} {place!r} "
                                f"({decision.reason_code})")
        elif decision.decision in (SKIP, DEFER, REJECT_CAPTURE):
            self._events.append(
                f"{prefix}{decision.decision}: {decision.reason_code}")
        else:
            self._events.append(
                f"{prefix}{decision.decision} not applied: "
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
        skill: str | None = None,
        gate_reason: str | None = None,
    ) -> dict:
        rec = {
            "ts": time.time(),
            "source": source,
            "skill": skill,
            "decision": decision,
            "reason_code": reason_code,
            "latency_ms": int(latency_ms),
            "applied": bool(applied and applied.get("applied")),
            "place_id": place_id,
        }
        if gate_reason:
            rec["gate_reason"] = str(gate_reason)[:80]
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
        primary = self.registrations[0]
        vlm = getattr(primary.skill, "vlm", None)
        running = bool(self._thread is not None and self._thread.is_alive())
        skills = []
        for registration in self.registrations:
            reg_vlm = getattr(registration.skill, "vlm", None)
            entry = {
                "name": registration.name,
                "event_gated": callable(
                    getattr(registration.skill, "event_trigger", None)),
                "cooldown_s": getattr(registration.executor, "cooldown_s", None),
                "last_trigger": self._last_triggers.get(registration.name),
                "vlm": getattr(reg_vlm, "describe", None),
            }
            gate_fn = getattr(registration.skill, "gate_status", None)
            if callable(gate_fn):
                try:
                    entry["gate"] = gate_fn()
                except Exception as exc:  # noqa: BLE001 - status is read-only
                    entry["gate"] = {"evaluated": False, "error": str(exc)[:200]}
            skills.append(entry)
        return {
            "enabled": True,
            "running": running,
            "interval_s": self.interval_s,
            "skills": skills,
            "registrations": len(self.registrations),
            "cooldown_s": getattr(primary.executor, "cooldown_s", None),
            "vlm": getattr(vlm, "describe", None),
            "assessments": self._assessments,
            "applied": self._applied,
            "busy": self._busy,
            "event_gated": callable(getattr(primary.skill, "event_trigger", None)),
            "last_trigger": self._last_trigger,
            "events": self.events(),
        }
