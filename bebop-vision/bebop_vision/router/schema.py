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

# --- decision vocabulary (place map capture) -------------------------------

#: Persist the first local structure map for the recognized place.
CAPTURE_MAP = "capture_map"
#: Add this capture's evidence to the place's existing accumulated map.
REFINE_MAP = "refine_map"
#: Refuse the capture: the live map is not worth persisting. Writes nothing.
REJECT_CAPTURE = "reject_capture"

PLACE_MEMORY_DECISIONS: tuple[str, ...] = (
    REMEMBER_PLACE,
    UPDATE_PLACE,
    SKIP,
    DEFER,
)

#: Decisions the map-capture skill may emit.
CAPTURE_DECISIONS: tuple[str, ...] = (
    CAPTURE_MAP,
    REFINE_MAP,
    REJECT_CAPTURE,
    SKIP,
    DEFER,
)

#: Every canonical decision the router understands.
ALL_DECISIONS: tuple[str, ...] = PLACE_MEMORY_DECISIONS + (
    CAPTURE_MAP,
    REFINE_MAP,
    REJECT_CAPTURE,
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
    # map capture
    "capture": CAPTURE_MAP,
    "map_capture": CAPTURE_MAP,
    "save_map": CAPTURE_MAP,
    "create_map": CAPTURE_MAP,
    "record_map": CAPTURE_MAP,
    "refine": REFINE_MAP,
    "update_map": REFINE_MAP,
    "improve_map": REFINE_MAP,
    "extend_map": REFINE_MAP,
    "reject": REJECT_CAPTURE,
    "reject_map": REJECT_CAPTURE,
    "discard": REJECT_CAPTURE,
    "discard_capture": REJECT_CAPTURE,
    "not_worth_saving": REJECT_CAPTURE,
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


@dataclass(frozen=True)
class StructureSummary:
    """The small, prompt-sized view of one live ``:9096/structure`` snapshot.

    The full snapshot is never shown to the model (and is never accepted from
    it): the executor fetches it itself.  This is only the evidence the brain
    needs to judge whether a capture is worth persisting.
    """

    gravity_valid: bool = False
    pose_valid: bool = False
    pose_x: float | None = None
    pose_y: float | None = None
    pose_yaw_deg: float | None = None
    occupied_cells: int = 0
    free_cells: int = 0
    wall_count: int = 0
    pose_delta_m: float = 0.0
    pose_delta_yaw_deg: float = 0.0
    revision: int | None = None
    segment: int | None = None

    @property
    def known_cells(self) -> int:
        return int(self.occupied_cells) + int(self.free_cells)


@dataclass(frozen=True)
class Recognition:
    """The appearance match for "where am I", with its thresholds visible."""

    matched: bool = False
    place_id: str | None = None
    place_name: str | None = None
    score: float = 0.0
    margin: float = 0.0
    min_score: float = 0.0
    min_margin: float = 0.0

    @property
    def reason(self) -> str:
        if self.matched:
            return "recognized"
        if not self.place_id:
            return "not_recognized_here"
        return "below_threshold"


@dataclass(frozen=True)
class CaptureContext:
    """Everything the map-capture skill knows before it may call the brain.

    ``gate_ok``/``gate_reason`` come from the deterministic pre-gate: when the
    gate fails the skill deflects to ``defer`` and makes **no** VLM request.
    """

    structure: StructureSummary
    recognition: Recognition
    place_id: str | None = None
    place_name: str | None = None
    has_map: bool = False
    map_revision: int | None = None
    map_age_s: float | None = None
    map_known_cells: int = 0
    cells_gained: int = 0
    gate_ok: bool = False
    gate_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "gate": {"ok": bool(self.gate_ok), "reason": self.gate_reason},
            "place": {
                "id": self.place_id,
                "name": self.place_name,
                "recognized": self.recognition.matched,
                "score": round(float(self.recognition.score), 4),
                "margin": round(float(self.recognition.margin), 4),
            },
            "structure": {
                "gravity_valid": self.structure.gravity_valid,
                "pose_valid": self.structure.pose_valid,
                "pose": [self.structure.pose_x, self.structure.pose_y,
                         self.structure.pose_yaw_deg],
                "occupied_cells": self.structure.occupied_cells,
                "free_cells": self.structure.free_cells,
                "known_cells": self.structure.known_cells,
                "wall_count": self.structure.wall_count,
                "pose_delta_m": round(float(self.structure.pose_delta_m), 4),
                "pose_delta_yaw_deg": round(
                    float(self.structure.pose_delta_yaw_deg), 3),
                "revision": self.structure.revision,
                "segment": self.structure.segment,
            },
            "map": {
                "has_map": self.has_map,
                "revision": self.map_revision,
                "age_s": self.map_age_s,
                "known_cells": self.map_known_cells,
                "cells_gained": self.cells_gained,
            },
        }


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
        return self.decision in (
            REMEMBER_PLACE, UPDATE_PLACE, CAPTURE_MAP, REFINE_MAP)

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
    if text in ALL_DECISIONS:
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
