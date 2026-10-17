"""System 1 router: wire types, decision vocabulary, and output parsing.

The router is a *fast, bounded* visual decision layer: it looks at a small
observation bundle (a few frames, a short task history, retrieved memories)
and picks the next skill to run.  It is deliberately not a planner and not a
controller — it does not sequence multi-step work, and it never touches the
motors.  Its whole output is one small structured decision.

This module is transport-free and dependency-light (stdlib only) so the
schema, the decision vocabulary, and the model-output parser are testable
without torch, a camera, or a VLM server.

The first skill is **place memory**: decide whether the current view is
worth remembering, and if so whether it is a new place or new evidence for
one we already know.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

# --- decision vocabulary (place memory) ------------------------------------

#: Create a new place from the observed view(s).
REMEMBER_PLACE = "remember_place"
#: Add the observed view(s) to a place we already know.
UPDATE_PLACE = "update_place"
#: Nothing to remember here (already represented, or not a place).
SKIP = "skip"
#: Insufficient/ambiguous evidence — look longer before deciding.
DEFER = "defer"

PLACE_MEMORY_DECISIONS: tuple[str, ...] = (
    REMEMBER_PLACE,
    UPDATE_PLACE,
    SKIP,
    DEFER,
)

#: Tolerated synonyms the model may emit; normalised to the canonical name.
_DECISION_ALIASES: dict[str, str] = {
    "create_place": REMEMBER_PLACE,
    "create": REMEMBER_PLACE,
    "remember": REMEMBER_PLACE,
    "new_place": REMEMBER_PLACE,
    "save_place": REMEMBER_PLACE,
    "enrich_place": UPDATE_PLACE,
    "enrich": UPDATE_PLACE,
    "update": UPDATE_PLACE,
    "add_view": UPDATE_PLACE,
    "add_place_view": UPDATE_PLACE,
    "none": SKIP,
    "no_action": SKIP,
    "ignore": SKIP,
    "nothing": SKIP,
    "wait": DEFER,
    "uncertain": DEFER,
    "unsure": DEFER,
    "need_more_evidence": DEFER,
    "gather_more": DEFER,
}


class DecisionParseError(ValueError):
    """The model's output did not contain a usable decision object."""


# --- observation bundle ----------------------------------------------------


@dataclass(frozen=True)
class View:
    """One camera frame offered to the router as evidence."""

    role: str
    jpeg: bytes
    stamp_us: int = 0


@dataclass(frozen=True)
class Candidate:
    """A place retrieval returned for the current view (appearance match)."""

    place_id: str
    name: str
    score: float
    facts: tuple[str, ...] = ()
    view_count: int = 0

    @property
    def fact_text(self) -> str:
        return "; ".join(self.facts[:4])


@dataclass
class Observation:
    """Everything the router sees for one decision.

    Kept intentionally small: a handful of frames, the active task/instruction
    if any, a short list of recent events, and the retrieved candidate places.
    No continuous video, no full transcript.
    """

    views: list[View] = field(default_factory=list)
    candidates: list[Candidate] = field(default_factory=list)
    instruction: str | None = None
    active_skill: str | None = None
    recent_events: list[str] = field(default_factory=list)
    places_known: int = 0


@dataclass
class SkillDecision:
    """One routing decision, normalised."""

    decision: str
    reason_code: str = ""
    confidence: float = 0.0
    arguments: dict[str, Any] = field(default_factory=dict)
    observations: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    raw: str = ""

    @property
    def actionable(self) -> bool:
        return self.decision in (REMEMBER_PLACE, UPDATE_PLACE)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "decision": self.decision,
            "reason_code": self.reason_code,
            "confidence": round(float(self.confidence), 3),
            "arguments": dict(self.arguments),
        }
        if self.observations:
            out["observations"] = list(self.observations)
        if self.evidence:
            out["evidence"] = list(self.evidence)
        return out


def normalize_decision(name: Any) -> str | None:
    """Map a model-emitted decision name onto the canonical vocabulary."""
    text = str(name or "").strip().lower().replace(" ", "_").replace("-", "_")
    if not text:
        return None
    if text in PLACE_MEMORY_DECISIONS:
        return text
    return _DECISION_ALIASES.get(text)


def defer_fallback(reason: str, *, raw: str = "", **arguments: Any) -> SkillDecision:
    """A `defer` decision used whenever the router cannot safely act."""
    return SkillDecision(
        decision=DEFER,
        reason_code=reason,
        confidence=0.0,
        arguments=dict(arguments),
        raw=raw,
    )


# --- parsing ---------------------------------------------------------------


def extract_json_object(text: str) -> dict[str, Any]:
    """Pull the first balanced JSON object out of a model reply.

    Models wrap JSON in prose or ```json fences; this scans for the first
    ``{`` and its matching ``}`` (string-aware) and parses that slice.
    """
    if not text:
        raise DecisionParseError("empty model output")
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    chunk = text[start : i + 1]
                    try:
                        parsed = json.loads(chunk)
                    except ValueError as exc:
                        raise DecisionParseError(f"invalid JSON: {exc}") from exc
                    if not isinstance(parsed, dict):
                        raise DecisionParseError("decision object must be a mapping")
                    return parsed
        start = text.find("{", start + 1)
    raise DecisionParseError("no JSON object found in model output")


def _as_str_list(value: Any, *, limit: int = 8) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        items = [str(value)]
    out: list[str] = []
    for item in items:
        text = " ".join(str(item).split())
        if text:
            out.append(text[:240])
        if len(out) >= limit:
            break
    return out


def _as_arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    return {}


def parse_decision(text: str) -> SkillDecision:
    """Parse the router's JSON reply into a `SkillDecision`.

    Raises `DecisionParseError` when no decision object is present; an object
    with an unrecognised `decision` field is returned as-is (the caller's
    sanitiser turns it into a `defer`).
    """
    obj = extract_json_object(text)
    raw_decision = obj.get("decision")
    decision = normalize_decision(raw_decision)
    if decision is None:
        # Keep the raw token so the sanitiser can report it.
        decision = str(raw_decision or "").strip().lower()

    arguments = _as_arguments(obj.get("arguments"))
    # Tolerate flat fields (name/place_id) instead of a nested `arguments`.
    for key in ("name", "place_id", "place_name", "suggested_name"):
        if key in obj and key not in arguments:
            arguments[key] = obj[key]

    try:
        confidence = float(obj.get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0

    return SkillDecision(
        decision=decision,
        reason_code=" ".join(str(obj.get("reason_code") or "").split())[:80],
        confidence=max(0.0, min(1.0, confidence)),
        arguments=arguments,
        observations=_as_str_list(obj.get("observations")),
        evidence=_as_str_list(obj.get("evidence")),
        raw=text,
    )
