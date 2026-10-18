"""Place-memory skill: the first capability the System 1 router owns.

Two pieces, deliberately separated:

- `PlaceMemorySkill.assess` — *decide*.  Retrieves candidate places by DINOv3
  appearance, builds the observation bundle, asks the VLM, and returns a
  validated decision.  It writes nothing.
- `PlaceMemoryExecutor.apply` — *act*.  Turns a validated decision into
  keyframes/descriptors/facts in the SQLite place store, using the exact
  frames the decision was made from.

The split matters: the decision layer stays cheap and side-effect free, the
executor holds the write guards (name/duplicate handling, a creation cooldown,
per-place growth caps), and both are testable without a camera or a GPU.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .prompt import build_messages
from .schema import (
    DEFER,
    REMEMBER_PLACE,
    SKIP,
    UPDATE_PLACE,
    Candidate,
    DecisionParseError,
    Observation,
    SkillDecision,
    View,
    defer_fallback,
    normalize_decision,
    parse_decision,
)
from .vlm import VLMClient


def _log(msg: str) -> None:
    print(f"[router] {msg}", flush=True)


def _preferred_view(views: list[View]) -> View | None:
    """The forward-looking camera sees the room; `near` is angled at the floor."""
    for v in views:
        if v.role == "far":
            return v
    return views[0] if views else None


@dataclass
class PlaceAssessment:
    """One routing decision plus the context it was made in."""

    decision: SkillDecision
    candidates: list[Candidate] = field(default_factory=list)
    latency_ms: int = 0
    error: str | None = None
    applied: dict | None = None

    def to_dict(self) -> dict:
        out = {
            "decision": self.decision.to_dict(),
            "candidates": [
                {"place_id": c.place_id, "name": c.name, "score": round(c.score, 4)}
                for c in self.candidates
            ],
            "latency_ms": self.latency_ms,
        }
        if self.error:
            out["error"] = self.error
        if self.applied is not None:
            out["applied"] = self.applied
        return out


class PlaceMemorySkill:
    """Decide whether the current view is worth remembering."""

    def __init__(
        self,
        store,
        encoder,
        vlm: VLMClient,
        *,
        k: int = 3,
        min_evidence_views: int = 1,
        max_observations: int = 6,
        max_tokens: int = 384,
    ) -> None:
        self.store = store
        self.encoder = encoder
        self.vlm = vlm
        self.k = int(k)
        self.min_evidence_views = int(min_evidence_views)
        self.max_observations = int(max_observations)
        self.max_tokens = int(max_tokens)

    # -- retrieval --------------------------------------------------------

    def candidates(self, views: list[View]) -> list[Candidate]:
        if self.encoder is None:
            return []
        view = _preferred_view(views)
        if view is None:
            return []
        try:
            vec = self.encoder.encode_jpeg(view.jpeg)
        except Exception as exc:  # noqa: BLE001 - retrieval is best-effort
            _log(f"candidate retrieval failed: {exc}")
            return []
        try:
            matches = self.store.match(vec, k=self.k, status="confirmed")
        except Exception as exc:  # noqa: BLE001
            _log(f"candidate match failed: {exc}")
            return []
        return [
            Candidate(
                place_id=m.place.id,
                name=m.place.name,
                score=float(m.score),
                facts=tuple(f.text for f in m.place.facts),
                view_count=len(m.place.keyframes),
            )
            for m in matches
        ]

    # -- decide -----------------------------------------------------------

    def assess(
        self,
        views: list[View],
        *,
        instruction: str | None = None,
        active_skill: str | None = None,
        recent_events: list[str] | None = None,
    ) -> PlaceAssessment:
        t0 = time.monotonic()
        candidates = self.candidates(views)
        obs = Observation(
            views=list(views),
            candidates=candidates,
            instruction=instruction,
            active_skill=active_skill,
            recent_events=list(recent_events or []),
            places_known=self.store.count(status="confirmed"),
        )
        try:
            text = self.vlm.complete(build_messages(obs), max_tokens=self.max_tokens)
        except Exception as exc:  # noqa: BLE001 - degrade to defer
            latency = int((time.monotonic() - t0) * 1000)
            _log(f"vlm error ({latency} ms): {exc}")
            return PlaceAssessment(
                decision=defer_fallback("vlm_error"),
                candidates=candidates,
                latency_ms=latency,
                error=str(exc),
            )
        try:
            raw = parse_decision(text)
        except DecisionParseError as exc:
            latency = int((time.monotonic() - t0) * 1000)
            _log(f"unparseable decision: {exc}")
            return PlaceAssessment(
                decision=defer_fallback("unparseable_output", raw=text),
                candidates=candidates,
                latency_ms=latency,
                error=str(exc),
            )
        decision = self.sanitize(raw, obs)
        latency = int((time.monotonic() - t0) * 1000)
        _log(
            f"{decision.decision} ({decision.reason_code or '-'}) "
            f"conf={decision.confidence:.2f} {latency} ms"
        )
        return PlaceAssessment(
            decision=decision, candidates=candidates, latency_ms=latency
        )

    # -- validation -------------------------------------------------------

    def sanitize(self, decision: SkillDecision, obs: Observation) -> SkillDecision:
        """Reject anything the executor could not safely apply."""
        canonical = normalize_decision(decision.decision)
        if canonical is None or canonical not in (
                REMEMBER_PLACE, UPDATE_PLACE, SKIP, DEFER):
            return defer_fallback(
                "unknown_decision",
                raw=decision.raw,
                requested=decision.decision,
            )
        decision.decision = canonical
        decision.observations = decision.observations[: self.max_observations]

        if len(obs.views) < self.min_evidence_views:
            return defer_fallback("insufficient_evidence", raw=decision.raw)

        if canonical == REMEMBER_PLACE:
            name = _argument_name(decision)
            if not name:
                return defer_fallback("missing_name", raw=decision.raw)
            decision.arguments["name"] = name
            return decision

        if canonical == UPDATE_PLACE:
            place_id = _argument_place_id(decision)
            resolved = _resolve_candidate(place_id, obs.candidates)
            if resolved is None:
                return defer_fallback(
                    "unknown_place",
                    raw=decision.raw,
                    requested=place_id,
                )
            decision.arguments["place_id"] = resolved.place_id
            decision.arguments["place_name"] = resolved.name
            return decision

        # skip / defer: nothing to validate.
        return decision


def _argument_name(decision: SkillDecision) -> str:
    args = decision.arguments
    for key in ("name", "suggested_name", "place_name", "label"):
        value = args.get(key)
        if value:
            return " ".join(str(value).split())[:64]
    return ""


def _argument_place_id(decision: SkillDecision) -> str:
    args = decision.arguments
    for key in ("place_id", "place", "id"):
        value = args.get(key)
        if value:
            return str(value).strip()
    return ""


def _resolve_candidate(place_id: str, candidates: list[Candidate]) -> Candidate | None:
    if not place_id:
        return None
    for c in candidates:
        if c.place_id == place_id or c.name == place_id:
            return c
    return None


class PlaceMemoryExecutor:
    """Apply validated place-memory decisions to the place store."""

    def __init__(
        self,
        store,
        encoder=None,
        *,
        cooldown_s: float = 180.0,
        max_views_per_call: int = 4,
        max_keyframes_per_place: int = 32,
        max_facts_per_call: int = 6,
        clock=time.monotonic,
    ) -> None:
        self.store = store
        self.encoder = encoder
        self.cooldown_s = float(cooldown_s)
        self.max_views_per_call = int(max_views_per_call)
        self.max_keyframes_per_place = int(max_keyframes_per_place)
        self.max_facts_per_call = int(max_facts_per_call)
        self._clock = clock
        self._last_created: tuple[float, str] | None = None

    def apply(
        self,
        decision: SkillDecision,
        views: list[View],
        *,
        instruction: str | None = None,
    ) -> dict:
        """Execute a sanitized decision. Never raises; returns an outcome."""
        if decision.decision == SKIP:
            return {"applied": False, "action": SKIP,
                    "reason_code": decision.reason_code}
        if decision.decision == DEFER:
            return {"applied": False, "action": DEFER,
                    "reason_code": decision.reason_code}
        if decision.decision == REMEMBER_PLACE:
            return self._remember(decision, views, instruction=instruction)
        if decision.decision == UPDATE_PLACE:
            return self._update(decision, views, instruction=instruction)
        return {"applied": False, "action": "unknown",
                "reason_code": "not_actionable"}

    # -- actions ----------------------------------------------------------

    def _remember(
        self,
        decision: SkillDecision,
        views: list[View],
        *,
        instruction: str | None,
    ) -> dict:
        name = str(decision.arguments.get("name") or "").strip()
        if not name:
            return {"applied": False, "action": REMEMBER_PLACE,
                    "reason_code": "missing_name"}

        # Creation cooldown: if we just created a place, treat a second
        # "remember" as more evidence for it instead of proliferating
        # duplicates on a slow walk through one room.
        now = self._clock()
        if self._last_created and now - self._last_created[0] < self.cooldown_s:
            prev_id = self._last_created[1]
            if self.store.get_place(prev_id) is not None:
                upd = SkillDecision(
                    decision=UPDATE_PLACE,
                    reason_code="cooldown_dedupe",
                    confidence=decision.confidence,
                    arguments={"place_id": prev_id},
                    observations=decision.observations,
                    evidence=decision.evidence,
                    raw=decision.raw,
                )
                out = self._update(upd, views, instruction=instruction)
                out["deduped_from"] = REMEMBER_PLACE
                return out

        try:
            place = self.store.create_place(name)
        except ValueError as exc:
            # Duplicate name: fold into the existing place instead of failing.
            existing = self.store.get_place(name)
            if existing is None:
                return {"applied": False, "action": REMEMBER_PLACE,
                        "reason_code": "invalid_name", "error": str(exc)}
            upd = SkillDecision(
                decision=UPDATE_PLACE,
                reason_code="deduped_by_name",
                confidence=decision.confidence,
                arguments={"place_id": existing.id},
                observations=decision.observations,
                evidence=decision.evidence,
                raw=decision.raw,
            )
            out = self._update(upd, views, instruction=instruction)
            out["deduped_from"] = REMEMBER_PLACE
            return out

        added_kf, added_desc = self._add_evidence(place.id, views)
        added_facts = self._add_facts(place.id, decision, instruction)
        self._last_created = (now, place.id)
        _log(f"remembered {name!r} as {place.id} "
             f"({added_kf} keyframes, {added_desc} descriptors, {added_facts} facts)")
        return {
            "applied": True,
            "action": REMEMBER_PLACE,
            "reason_code": decision.reason_code,
            "place_id": place.id,
            "place_name": place.name,
            "keyframes": added_kf,
            "descriptors": added_desc,
            "facts": added_facts,
        }

    def _update(
        self,
        decision: SkillDecision,
        views: list[View],
        *,
        instruction: str | None,
    ) -> dict:
        place_id = str(decision.arguments.get("place_id") or "").strip()
        place = self.store.get_place(place_id) if place_id else None
        if place is None:
            return {"applied": False, "action": UPDATE_PLACE,
                    "reason_code": "unknown_place"}
        added_kf, added_desc = self._add_evidence(place.id, views)
        added_facts = self._add_facts(place.id, decision, instruction)
        if added_kf:
            self.store.touch_seen(place.id)
        _log(f"updated {place.name!r} ({place.id}) "
             f"({added_kf} keyframes, {added_desc} descriptors, {added_facts} facts)")
        return {
            "applied": True,
            "action": UPDATE_PLACE,
            "reason_code": decision.reason_code,
            "place_id": place.id,
            "place_name": place.name,
            "keyframes": added_kf,
            "descriptors": added_desc,
            "facts": added_facts,
        }

    # -- helpers ----------------------------------------------------------

    def _add_evidence(self, place_id: str, views: list[View]) -> tuple[int, int]:
        place = self.store.get_place(place_id)
        if place is None:
            return (0, 0)
        room = self.max_keyframes_per_place - len(place.keyframes)
        if room <= 0:
            return (0, 0)
        added_kf = added_desc = 0
        for view in views[: self.max_views_per_call]:
            if added_kf >= room:
                break
            try:
                self.store.add_keyframe(
                    place_id, view.jpeg, role=view.role, stamp_us=view.stamp_us
                )
                added_kf += 1
            except Exception as exc:  # noqa: BLE001 - keep the other frames
                _log(f"keyframe store failed for {view.role}: {exc}")
                continue
            if self.encoder is not None:
                try:
                    self.store.add_descriptor(
                        place_id,
                        self.encoder.encode_jpeg(view.jpeg),
                        source=f"dinov3:{view.role}",
                    )
                    added_desc += 1
                except Exception as exc:  # noqa: BLE001
                    _log(f"descriptor encode failed for {view.role}: {exc}")
        return (added_kf, added_desc)

    def _add_facts(
        self,
        place_id: str,
        decision: SkillDecision,
        instruction: str | None,
    ) -> int:
        added = 0
        texts: list[tuple[str, str]] = []
        if instruction:
            texts.append((f"user: {instruction}", "operator"))
        for obs_text in decision.observations:
            texts.append((obs_text, "router"))
        texts.append((f"router: {decision.reason_code or decision.decision}", "router"))
        seen: set[str] = set()
        for text, source in texts:
            norm = " ".join(text.split())[:240]
            if not norm or norm.lower() in seen:
                continue
            seen.add(norm.lower())
            if added >= self.max_facts_per_call:
                break
            try:
                self.store.add_fact(place_id, norm, source=source)
                added += 1
            except Exception as exc:  # noqa: BLE001
                _log(f"fact store failed: {exc}")
        return added
