"""Map-capture skill: let the brain decide capture / refine / reject.

Two pieces, deliberately separated, exactly like the place-memory skill:

- `CaptureMapSkill.assess` — *decide*.  It probes the live ``:9096/structure``
  snapshot, the current ``/whereami`` recognition and the place's stored map,
  then **always** asks the VLM to choose among ``capture_map | refine_map |
  reject_capture | skip | defer`` (the capture skill is prompted every tick).
  A deterministic pre-gate still runs, but it is now a **write precondition**,
  not a prompt filter: if it is not OK, the model's decision is degraded to
  ``defer`` with a ``gate_*`` reason and nothing is written.
- `CaptureMapExecutor.apply` — *act*.  It fetches the live structure snapshot
  **itself** and calls ``PlaceStore.upsert_place_map`` (merge semantics).  The
  model never authors a geometry payload; it only picks a decision.  A
  per-place cooldown prevents the same every-tick decision from re-writing the
  map on every tick.

The gate telemetry is exposed read-only through :meth:`gate_status` and
``/router/status``; the VLM still gets prompted even when the gate is closed,
so its decision record explains the defer.
"""
from __future__ import annotations

import json
import math
import time
import urllib.request
from dataclasses import dataclass, replace

from bebop_vision.places.submap import rle_true_count

from .prompt import build_capture_messages
from .schema import (
    CAPTURE_MAP,
    DEFER,
    REFINE_MAP,
    REJECT_CAPTURE,
    SKIP,
    CaptureContext,
    DecisionParseError,
    Recognition,
    SkillDecision,
    StructureSummary,
    defer_fallback,
    normalize_decision,
    parse_decision,
)
from .vlm import VLMClient

DEFAULT_STRUCTURE_URL = "http://127.0.0.1:9096/structure"

#: The capture skill may only emit these (plus skip/defer).
_CAPTURE_ALLOWED = (CAPTURE_MAP, REFINE_MAP, REJECT_CAPTURE, SKIP, DEFER)

#: Pre-gate reasons (as reported by CaptureContext.gate_reason) mapped to the
#: explicit ``defer`` vocabulary recorded when a write is blocked.
_GATE_DEFER_REASONS: dict[str, str] = {
    "gravity_invalid": "gate_gravity_invalid",
    "pose_invalid": "gate_pose_invalid",
    "too_few_cells": "gate_too_few_cells",
    "robot_moving": "gate_moving",
    "not_recognized_here": "gate_not_recognized_here",
    "unknown_place": "gate_unknown_place",
    "no_structure": "gate_no_structure",
}


def gate_defer_reason(reason: str) -> str:
    """Canonical ``gate_*`` defer reason for a failed write precondition."""
    reason = " ".join(str(reason or "").split()) or "gate_failed"
    if reason == "gate_failed" or reason.startswith("gate_"):
        return reason
    return _GATE_DEFER_REASONS.get(reason, f"gate_{reason}")


def _log(msg: str) -> None:
    print(f"[router] {msg}", flush=True)


# --- live structure --------------------------------------------------------


def fetch_structure(
    url: str = DEFAULT_STRUCTURE_URL, *, timeout: float = 2.0
) -> dict:
    """Read one live structure snapshot (stdlib only, bounded)."""
    with urllib.request.urlopen(url, timeout=float(timeout)) as response:
        return json.loads(response.read().decode("utf-8"))


def _body(payload) -> dict:
    if not isinstance(payload, dict):
        return {}
    inner = payload.get("structure")
    return inner if isinstance(inner, dict) else payload


def _number(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    out = float(value)
    return out if math.isfinite(out) else None


def _int_count(value) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value >= 0 else None


def _mask_count(grid: dict, key: str) -> int:
    direct = _int_count(grid.get(f"{key}_count"))
    if direct is not None:
        return direct
    encoded = grid.get(f"{key}_rle")
    if not isinstance(encoded, str) or not encoded:
        return 0
    try:
        return rle_true_count(encoded)
    except (ValueError, TypeError):
        return 0


def summarize_structure(
    payload,
    *,
    pose_delta_m: float = 0.0,
    pose_delta_yaw_deg: float = 0.0,
) -> StructureSummary:
    """Extract the prompt-sized evidence from a structure snapshot."""
    body = _body(payload)
    grid = body.get("grid") if isinstance(body.get("grid"), dict) else {}
    gravity = body.get("gravity") if isinstance(body.get("gravity"), dict) else {}
    pose = body.get("pose")
    if not isinstance(pose, dict) and isinstance(payload, dict):
        pose = payload.get("pose")
    pose = pose if isinstance(pose, dict) else {}
    walls = body.get("walls") if isinstance(body.get("walls"), list) else []

    # A caller may supply an explicit motion sample (tests, or a future
    # /status integration); otherwise the skill measures it from two fetches.
    explicit_delta = None
    explicit_yaw = None
    for source in (body, payload if isinstance(payload, dict) else {}):
        if explicit_delta is None:
            explicit_delta = _number(source.get("pose_delta_m"))
        if explicit_yaw is None:
            explicit_yaw = _number(source.get("pose_delta_yaw_deg"))
    delta_m = explicit_delta if explicit_delta is not None else pose_delta_m
    delta_yaw = (explicit_yaw if explicit_yaw is not None
                 else pose_delta_yaw_deg)

    revision = _int_count(body.get("revision"))
    segment = _int_count(pose.get("segment"))
    wall_count = len(walls)
    if not walls:
        diagnostics = body.get("diagnostics")
        if isinstance(diagnostics, dict):
            wall_count = max(0, _int_count(
                diagnostics.get("wall_segments")) or 0)
    return StructureSummary(
        gravity_valid=gravity.get("valid") is True,
        pose_valid=pose.get("valid") is True,
        pose_x=_number(pose.get("x")),
        pose_y=_number(pose.get("y")),
        pose_yaw_deg=_number(pose.get("yaw_deg")),
        occupied_cells=_mask_count(grid, "occupied"),
        free_cells=_mask_count(grid, "free"),
        wall_count=wall_count,
        pose_delta_m=max(0.0, float(delta_m)),
        pose_delta_yaw_deg=max(0.0, float(delta_yaw)),
        revision=revision,
        segment=segment,
    )


def _has_explicit_motion(payload) -> bool:
    body = _body(payload)
    extra = payload if isinstance(payload, dict) else {}
    return ("pose_delta_m" in body or "pose_delta_yaw_deg" in body
            or "pose_delta_m" in extra or "pose_delta_yaw_deg" in extra)


def _pose_delta(a: StructureSummary, b: StructureSummary) -> tuple[float, float]:
    """Distance/heading change between two structure samples."""
    if not (a.pose_valid and b.pose_valid):
        return (0.0, 0.0)
    if (a.pose_x is None or a.pose_y is None
            or b.pose_x is None or b.pose_y is None):
        return (0.0, 0.0)
    dist = math.hypot(b.pose_x - a.pose_x, b.pose_y - a.pose_y)
    if (a.segment is not None and b.segment is not None
            and a.segment != b.segment):
        # A new session/segment means the poses are not comparable; treat the
        # capture as moving rather than pretending it is stationary.
        return (max(dist, 1e6), 1e6)
    dyaw = 0.0
    if a.pose_yaw_deg is not None and b.pose_yaw_deg is not None:
        dyaw = abs((b.pose_yaw_deg - a.pose_yaw_deg + 180.0) % 360.0 - 180.0)
    return (dist, dyaw)


def _recognition_from_whereami(raw) -> Recognition:
    """Normalise a ``/whereami`` reply (or a simple dict) for the gate."""
    if not isinstance(raw, dict):
        return Recognition()
    matched = raw.get("matched") is True
    place = raw.get("place") if isinstance(raw.get("place"), dict) else {}
    place_id = place.get("id") or raw.get("place_id")
    place_name = place.get("name") or raw.get("place_name")
    score = _number(raw.get("score")) or 0.0
    margin = _number(raw.get("margin")) or 0.0
    return Recognition(
        matched=bool(matched and place_id),
        place_id=str(place_id) if place_id else None,
        place_name=str(place_name) if place_name else None,
        score=score,
        margin=margin,
        min_score=_number(raw.get("min_score")) or 0.0,
        min_margin=_number(raw.get("min_margin")) or 0.0,
    )


def _preferred_view(views):
    for view in views:
        if view.role == "far":
            return view
    return views[0] if views else None


@dataclass
class CaptureAssessment:
    """One capture decision plus the evidence bundle it was made from."""

    decision: SkillDecision
    context: CaptureContext | None = None
    latency_ms: int = 0
    error: str | None = None
    applied: dict | None = None

    def to_dict(self) -> dict:
        out = {
            "decision": self.decision.to_dict(),
            "latency_ms": self.latency_ms,
        }
        if self.context is not None:
            out["capture"] = self.context.to_dict()
        if self.error:
            out["error"] = self.error
        if self.applied is not None:
            out["applied"] = self.applied
        return out


class CaptureMapSkill:
    """Decide whether the live structure map is worth persisting."""

    def __init__(
        self,
        store,
        encoder,
        vlm: VLMClient,
        *,
        structure_fetch=None,
        structure_url: str = DEFAULT_STRUCTURE_URL,
        structure_timeout: float = 2.0,
        recognizer=None,
        whereami=None,
        min_score: float = 0.5,
        min_margin: float = 0.05,
        min_cells: int = 10,
        max_pose_delta_m: float = 0.15,
        max_pose_delta_yaw_deg: float = 8.0,
        motion_window_s: float = 0.3,
        place_stable_s: float = 10.0,
        min_cells_gained: int = 100,
        cooldown_s: float = 180.0,
        max_observations: int = 6,
        max_tokens: int = 192,
        clock=time.monotonic,
        wall_clock=time.time,
        sleep=time.sleep,
    ) -> None:
        self.store = store
        self.encoder = encoder
        self.vlm = vlm
        self.structure_url = structure_url
        self.structure_timeout = float(structure_timeout)
        self._structure_fetch = structure_fetch
        self._recognizer = (recognizer if recognizer is not None
                            else whereami)
        self.min_score = float(min_score)
        self.min_margin = float(min_margin)
        self.min_cells = int(min_cells)
        self.max_pose_delta_m = float(max_pose_delta_m)
        self.max_pose_delta_yaw_deg = float(max_pose_delta_yaw_deg)
        self.motion_window_s = max(0.0, float(motion_window_s))
        self.place_stable_s = max(0.0, float(place_stable_s))
        self.min_cells_gained = int(min_cells_gained)
        self.cooldown_s = float(cooldown_s)
        self.max_observations = int(max_observations)
        self.max_tokens = int(max_tokens)
        self._clock = clock
        self._wall_clock = wall_clock
        self._sleep = sleep
        # Gate-tracking state (monotonic clock) and write cooldown.
        self._stable_place: str | None = None
        self._stable_since: float | None = None
        self._last_recognized_place: str | None = None
        self._last_capture_at: dict[str, float] = {}
        # Last gate evaluation, exposed read-only through gate_status().
        self._last_gate_state: dict | None = None
        self._last_gate_at: float | None = None

    # -- probe (no brain) --------------------------------------------------

    def probe(
        self,
        views=None,
        *,
        target_place_id: str | None = None,
        now: float | None = None,
    ) -> CaptureContext:
        """Read live evidence and run the deterministic pre-gate.

        This never calls the VLM.  It always returns a context; inspect
        ``gate_ok``/``gate_reason`` before allowing a brain call.
        """
        wall_now = float(self._wall_clock() if now is None else now)
        try:
            payload = self._fetch_payload()
            summary = summarize_structure(payload)
            explicit_motion = _has_explicit_motion(payload)
            if (self.motion_window_s > 0 and summary.pose_valid
                    and not explicit_motion):
                try:
                    self._sleep(self.motion_window_s)
                    later = summarize_structure(self._fetch_payload())
                    dist, dyaw = _pose_delta(summary, later)
                    summary = replace(
                        summary, pose_delta_m=dist,
                        pose_delta_yaw_deg=dyaw)
                except Exception as exc:  # noqa: BLE001 - motion is best effort
                    _log(f"motion sample failed: {exc}")
        except Exception as exc:  # noqa: BLE001 - no structure -> defer
            _log(f"structure fetch failed: {exc}")
            return CaptureContext(
                structure=StructureSummary(),
                recognition=Recognition(),
                gate_ok=False,
                gate_reason="no_structure",
            )

        recognition = self._recognize(views, target_place_id)
        place = None
        if recognition.place_id:
            place = self.store.get_place(recognition.place_id)
        place_map = None
        if place is not None:
            place_map = self.store.get_place_map(place.id)

        map_known = 0
        has_map = place_map is not None
        if place_map is not None:
            map_known = (rle_true_count(place_map.occupied_rle)
                         + rle_true_count(place_map.free_rle))
        cells_gained = max(0, summary.known_cells - map_known)

        gate_ok, gate_reason = self._gate(
            summary, recognition, place, target_place_id)

        return CaptureContext(
            structure=summary,
            recognition=recognition,
            place_id=place.id if place is not None else recognition.place_id,
            place_name=place.name if place is not None else recognition.place_name,
            has_map=has_map,
            map_revision=place_map.revision if place_map is not None else None,
            map_age_s=(max(0.0, wall_now - place_map.updated_at)
                       if place_map is not None else None),
            map_known_cells=map_known,
            cells_gained=cells_gained,
            gate_ok=bool(gate_ok),
            gate_reason=gate_reason,
        )

    def _fetch_payload(self) -> dict:
        if self._structure_fetch is not None:
            return self._structure_fetch()
        return fetch_structure(self.structure_url,
                               timeout=self.structure_timeout)

    def _gate(
        self,
        summary: StructureSummary,
        recognition: Recognition,
        place,
        target_place_id: str | None,
    ) -> tuple[bool, str]:
        if not summary.gravity_valid:
            return (False, "gravity_invalid")
        if not summary.pose_valid:
            return (False, "pose_invalid")
        if summary.known_cells < self.min_cells:
            return (False, "too_few_cells")
        if (summary.pose_delta_m > self.max_pose_delta_m
                or summary.pose_delta_yaw_deg > self.max_pose_delta_yaw_deg):
            return (False, "robot_moving")
        if not recognition.matched or recognition.place_id is None:
            return (False, "not_recognized_here")
        if place is None:
            return (False, "unknown_place")
        if target_place_id is not None and place.id != target_place_id:
            return (False, "not_recognized_here")
        return (True, "")

    def _recognize(self, views, target_place_id: str | None) -> Recognition:
        if self._recognizer is not None:
            try:
                try:
                    raw = self._recognizer(views)
                except TypeError:
                    # A zero-argument recognizer (e.g. PlacesService.whereami)
                    # is also acceptable.
                    raw = self._recognizer()
                return _recognition_from_whereami(raw)
            except Exception as exc:  # noqa: BLE001 - defer, don't capture blind
                _log(f"recognizer failed: {exc}")
                return Recognition()
        if self.encoder is None:
            return Recognition()
        view = _preferred_view(list(views or []))
        if view is None:
            return Recognition()
        try:
            vec = self.encoder.encode_jpeg(view.jpeg)
            matches = self.store.match(vec, k=3, status="confirmed")
        except Exception as exc:  # noqa: BLE001
            _log(f"recognition failed: {exc}")
            return Recognition()
        if not matches:
            return Recognition()
        top = matches[0]
        runner = matches[1].score if len(matches) > 1 else 0.0
        margin = float(top.score) - float(runner)
        matched = (top.score >= self.min_score and margin >= self.min_margin)
        return Recognition(
            matched=bool(matched),
            place_id=top.place.id,
            place_name=top.place.name,
            score=float(top.score),
            margin=margin,
            min_score=self.min_score,
            min_margin=self.min_margin,
        )

    # -- decide ------------------------------------------------------------

    def assess(
        self,
        views,
        *,
        instruction: str | None = None,
        active_skill: str | None = None,
        recent_events: list[str] | None = None,
        target_place_id: str | None = None,
        context: CaptureContext | None = None,
    ) -> CaptureAssessment:
        t0 = time.monotonic()
        ctx = context or self.probe(views, target_place_id=target_place_id)
        # Gate telemetry is computed on every assessment, even when the gate is
        # closed: it is the write precondition, not a prompt filter.
        self._observe_gate(ctx, float(self._clock()))

        # ALWAYS ask the brain.  Capture is prompted every tick, like place
        # memory; the deterministic gate only decides whether a write is
        # allowed (below).
        try:
            text = self.vlm.complete(
                build_capture_messages(ctx, list(views or [])),
                max_tokens=self.max_tokens,
            )
        except Exception as exc:  # noqa: BLE001 - degrade to defer
            latency = int((time.monotonic() - t0) * 1000)
            _log(f"vlm error ({latency} ms): {exc}")
            return CaptureAssessment(
                decision=defer_fallback("vlm_error"),
                context=ctx,
                latency_ms=latency,
                error=str(exc),
            )
        try:
            raw = parse_decision(text)
        except DecisionParseError as exc:
            latency = int((time.monotonic() - t0) * 1000)
            _log(f"unparseable capture decision: {exc}")
            return CaptureAssessment(
                decision=defer_fallback("unparseable_output", raw=text),
                context=ctx,
                latency_ms=latency,
                error=str(exc),
            )

        if not ctx.gate_ok:
            # The model may have said capture_map/refine_map; a closed gate
            # degrades it to defer and the executor will write nothing.
            reason = gate_defer_reason(ctx.gate_reason)
            decision = defer_fallback(
                reason,
                raw=text,
                requested=raw.decision,
                gate_reason=ctx.gate_reason,
            )
            latency = int((time.monotonic() - t0) * 1000)
            _log(f"capture gate blocked ({ctx.gate_reason}); "
                 f"model said {raw.decision} -> {reason} ({latency} ms)")
            return CaptureAssessment(
                decision=decision, context=ctx, latency_ms=latency)

        decision = self.sanitize(raw, ctx)
        latency = int((time.monotonic() - t0) * 1000)
        _log(f"capture {decision.decision} ({decision.reason_code or '-'}) "
             f"conf={decision.confidence:.2f} {latency} ms")
        return CaptureAssessment(
            decision=decision, context=ctx, latency_ms=latency)

    def sanitize(
        self, decision: SkillDecision, ctx: CaptureContext
    ) -> SkillDecision:
        """Reject anything the executor could not safely apply."""
        canonical = normalize_decision(decision.decision)
        if canonical is None or canonical not in _CAPTURE_ALLOWED:
            return defer_fallback(
                "unknown_decision",
                raw=decision.raw,
                requested=decision.decision,
            )
        decision.decision = canonical
        decision.observations = decision.observations[: self.max_observations]

        if canonical in (CAPTURE_MAP, REFINE_MAP):
            if not ctx.gate_ok or ctx.place_id is None:
                return defer_fallback(
                    gate_defer_reason(ctx.gate_reason),
                    raw=decision.raw,
                    requested=decision.decision,
                    gate_reason=ctx.gate_reason,
                )
            requested = _argument_place_id(decision)
            if (requested
                    and requested not in (ctx.place_id, ctx.place_name)):
                return defer_fallback(
                    "unknown_place", raw=decision.raw, requested=requested)
            # The executor may only target the place recognized as "here".
            decision.arguments["place_id"] = ctx.place_id
            decision.arguments["place_name"] = ctx.place_name
        return decision

    # -- gate telemetry (the write precondition) ---------------------------

    def in_cooldown(self, place_id: str | None, now: float | None = None) -> bool:
        return self.cooldown_remaining(place_id, now) > 0.0

    def cooldown_remaining(self, place_id: str | None,
                           now: float | None = None) -> float:
        if not place_id:
            return 0.0
        t = float(self._clock() if now is None else now)
        last = self._last_capture_at.get(place_id)
        if last is None:
            return 0.0
        return max(0.0, self.cooldown_s - (t - last))

    def gate_thresholds(self) -> dict:
        """The exact deterministic gate thresholds in force.

        ``place_stable_s``/``min_cells_gained`` are retained as tuning
        telemetry; they no longer suppress the prompt (capture is assessed
        every tick).
        """
        return {
            "min_cells": self.min_cells,
            "max_pose_delta_m": self.max_pose_delta_m,
            "max_pose_delta_yaw_deg": self.max_pose_delta_yaw_deg,
            "motion_window_s": self.motion_window_s,
            "min_score": self.min_score,
            "min_margin": self.min_margin,
            "place_stable_s": self.place_stable_s,
            "min_cells_gained": self.min_cells_gained,
            "cooldown_s": self.cooldown_s,
        }

    def gate_status(self) -> dict:
        """Read-only snapshot of the last gate evaluation.

        Populated by every ``assess`` call (and therefore by every router
        tick).  This method never probes the structure service or calls the
        brain, so it is cheap even though a real tick costs 5-10 s.
        ``age_s`` says how stale the snapshot is.
        """
        if self._last_gate_state is None:
            return {
                "evaluated": False,
                "gate_ok": None,
                "gate_reason": "",
                "place_id": None,
                "stable_for_s": 0.0,
                "place_stable_s": self.place_stable_s,
                "place_changed": False,
                "cells_gained": 0,
                "min_cells_gained": self.min_cells_gained,
                "in_cooldown": False,
                "cooldown_s": self.cooldown_s,
                "cooldown_remaining_s": 0.0,
                "thresholds": self.gate_thresholds(),
            }
        state = dict(self._last_gate_state)
        if self._last_gate_at is not None:
            state["age_s"] = round(
                max(0.0, self._wall_clock() - self._last_gate_at), 3)
        return state

    def _observe_gate(self, ctx: CaptureContext, now: float) -> None:
        """Update gate telemetry. This is observation only, never a filter."""
        cooldown = self.in_cooldown(ctx.place_id, now)
        if not ctx.gate_ok or not ctx.place_id:
            self._stable_place = None
            self._stable_since = None
            self._last_recognized_place = None
            stable_for = 0.0
            place_changed = False
        else:
            if ctx.place_id != self._stable_place:
                self._stable_place = ctx.place_id
                self._stable_since = now
            stable_for = now - (self._stable_since
                                if self._stable_since is not None else now)
            place_changed = ctx.place_id != self._last_recognized_place
            self._last_recognized_place = ctx.place_id
        self._remember_gate_state(
            ctx, now, stable_for=stable_for,
            place_changed=place_changed, cooldown=cooldown)

    def _remember_gate_state(
        self,
        ctx: CaptureContext,
        now: float,
        *,
        stable_for: float,
        place_changed: bool,
        cooldown: bool,
    ) -> None:
        self._last_gate_state = {
            "evaluated": True,
            "gate_ok": ctx.gate_ok,
            "gate_reason": ctx.gate_reason,
            "place_id": ctx.place_id,
            "place_name": ctx.place_name,
            "recognized": ctx.recognition.matched,
            "score": round(float(ctx.recognition.score), 4),
            "margin": round(float(ctx.recognition.margin), 4),
            "has_map": ctx.has_map,
            "map_revision": ctx.map_revision,
            "map_age_s": (round(float(ctx.map_age_s), 1)
                          if ctx.map_age_s is not None else None),
            "stable_place": self._stable_place,
            "stable_for_s": round(max(0.0, float(stable_for)), 3),
            "place_stable_s": self.place_stable_s,
            "place_changed": bool(place_changed),
            "cells_gained": int(ctx.cells_gained),
            "min_cells_gained": self.min_cells_gained,
            "in_cooldown": bool(cooldown),
            "cooldown_s": self.cooldown_s,
            "cooldown_remaining_s": round(
                self.cooldown_remaining(ctx.place_id, now), 3),
            "gravity_valid": ctx.structure.gravity_valid,
            "pose_valid": ctx.structure.pose_valid,
            "pose_delta_m": round(float(ctx.structure.pose_delta_m), 3),
            "pose_delta_yaw_deg": round(
                float(ctx.structure.pose_delta_yaw_deg), 2),
            "known_cells": ctx.structure.known_cells,
            "wall_count": ctx.structure.wall_count,
            "evaluated_at": round(float(self._wall_clock()), 3),
            "thresholds": self.gate_thresholds(),
        }
        self._last_gate_at = float(self._wall_clock())

    def note_outcome(self, outcome: dict | None) -> None:
        """Record a successful capture so the cooldown state advances."""
        if not outcome or not outcome.get("applied"):
            return
        place_id = outcome.get("place_id")
        if not place_id:
            return
        self._last_capture_at[str(place_id)] = float(self._clock())


def _argument_place_id(decision: SkillDecision) -> str:
    args = decision.arguments
    for key in ("place_id", "place", "id"):
        value = args.get(key)
        if value:
            return str(value).strip()
    return ""


class CaptureMapExecutor:
    """Apply capture decisions; the model never supplies geometry."""

    def __init__(
        self,
        store,
        *,
        structure_fetch=None,
        structure_url: str = DEFAULT_STRUCTURE_URL,
        structure_timeout: float = 2.0,
        cooldown_s: float = 180.0,
        clock=time.monotonic,
    ) -> None:
        self.store = store
        self.structure_url = structure_url
        self.structure_timeout = float(structure_timeout)
        self._structure_fetch = structure_fetch
        self.cooldown_s = float(cooldown_s)
        self._clock = clock
        self._last_capture_at: dict[str, float] = {}
        self._last_outcome: dict | None = None

    def _fetch_payload(self) -> dict:
        if self._structure_fetch is not None:
            return self._structure_fetch()
        return fetch_structure(self.structure_url,
                               timeout=self.structure_timeout)

    def cooldown_remaining(self, place_id: str | None,
                           now: float | None = None) -> float:
        if not place_id:
            return 0.0
        t = float(self._clock() if now is None else now)
        last = self._last_capture_at.get(place_id)
        if last is None:
            return 0.0
        return max(0.0, self.cooldown_s - (t - last))

    def apply(
        self,
        decision: SkillDecision,
        views=None,
        *,
        instruction: str | None = None,
    ) -> dict:
        """Execute a sanitized decision. Never raises; returns an outcome."""
        action = normalize_decision(decision.decision) or decision.decision
        if action in (SKIP, DEFER, REJECT_CAPTURE):
            return {
                "applied": False,
                "action": action,
                "reason_code": decision.reason_code,
                "written": False,
            }
        if action not in (CAPTURE_MAP, REFINE_MAP):
            return {"applied": False, "action": "unknown",
                    "reason_code": "not_actionable", "written": False}

        place_id = _argument_place_id(decision)
        place = self.store.get_place(place_id) if place_id else None
        if place is None:
            return {"applied": False, "action": action,
                    "reason_code": "unknown_place", "written": False}

        now = float(self._clock())
        remaining = self.cooldown_remaining(place.id, now)
        if remaining > 0:
            return {
                "applied": False,
                "action": action,
                "reason_code": "cooldown",
                "written": False,
                "cooldown_remaining_s": round(remaining, 3),
            }

        # Geometry comes ONLY from the live service, never from the decision.
        try:
            snapshot = self._fetch_payload()
        except Exception as exc:  # noqa: BLE001 - no map without structure
            _log(f"capture aborted, no structure: {exc}")
            return {"applied": False, "action": action,
                    "reason_code": "no_structure", "written": False,
                    "error": str(exc)[:200]}
        try:
            place_map, created = self.store.upsert_place_map(
                place.id, snapshot)
        except KeyError:
            return {"applied": False, "action": action,
                    "reason_code": "unknown_place", "written": False}
        except ValueError as exc:
            _log(f"capture rejected, invalid structure: {exc}")
            return {"applied": False, "action": action,
                    "reason_code": "invalid_structure", "written": False,
                    "error": str(exc)[:200]}

        self._last_capture_at[place.id] = now
        outcome = {
            "applied": True,
            "written": True,
            "action": action,
            "reason_code": decision.reason_code,
            "place_id": place.id,
            "place_name": place.name,
            "created": bool(created),
            "revision": place_map.revision,
            "width": place_map.width,
            "height": place_map.height,
            "occupied_cells": rle_true_count(place_map.occupied_rle),
            "free_cells": rle_true_count(place_map.free_rle),
            "known_cells": (rle_true_count(place_map.occupied_rle)
                            + rle_true_count(place_map.free_rle)),
            "walls": len(place_map.walls),
        }
        self._last_outcome = outcome
        _log(f"{action} {place.name!r} ({place.id}) revision "
             f"{place_map.revision}: {place_map.width}x{place_map.height} "
             f"cells, {len(place_map.walls)} walls")
        return outcome
