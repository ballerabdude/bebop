"""Operator-driven import for pending places.

Capture is deliberately cheap (see :mod:`bebop_vision.places.scout`).  The
VLM is only used here, when the operator asks for a name suggestion for a
pending place:

    suggest  -> one LLM call over the pending place's stored keyframes
             -> suggested_name / suggested_facts on the pending row
             -> status stays "pending"

    import   -> deterministic promotion with the operator-edited name
             -> no LLM call

The operator can type a name and skip the model entirely; the import path
must therefore work with ``vlm=None``.
"""
from __future__ import annotations

import base64
import json
import threading
import time

from .capture import View
from .store import Place, PlaceStore, is_placeholder_name

IMPORT_SYSTEM_PROMPT = """\
You are naming one place the robot visited. You see one to three keyframes
from that place. Suggest a short, specific place name and the stable visual
facts that would help a person recognize it later.

Rules:
- Prefer room or landmark names: workshop, kitchen, stairwell, front door.
- Do not name a single movable object, a person, a pet, or a transient view.
- Keep the name at most 48 characters.
- Give 3 to 5 short, stable visual facts. No people, pets, motion blur, or
  camera artifacts.
- If the images are too poor to name the place, return an empty name.

Reply with ONE JSON object and nothing else:
{
  "name": "short place name",
  "facts": ["stable visual fact", "..."],
  "confidence": 0.0
}
"""

_MAX_IMAGE_SIDE = 640
_MAX_FACTS = 5
_MAX_TOKENS = 160
_SUGGESTION_CACHE_S = 3600.0


def _log(msg: str) -> None:
    print(f"[import] {msg}", flush=True)


def _extract_json(text: str) -> dict:
    """Pull the first balanced JSON object out of a model reply."""
    if not text:
        raise ValueError("empty model output")
    start = text.find("{")
    while start >= 0:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            ch = text[index]
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
                    try:
                        value = json.loads(text[start:index + 1])
                    except ValueError as exc:
                        raise ValueError(f"invalid JSON: {exc}") from exc
                    if not isinstance(value, dict):
                        raise ValueError("model reply is not a JSON object")
                    return value
        start = text.find("{", start + 1)
    raise ValueError("no JSON object in model reply")


def _image_data_url(jpeg: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")


def _downscale_jpeg(jpeg: bytes, max_side: int = _MAX_IMAGE_SIDE) -> bytes:
    try:
        import cv2
        import numpy as np

        image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8),
                             cv2.IMREAD_COLOR)
        if image is None:
            return jpeg
        height, width = image.shape[:2]
        longest = max(height, width)
        if longest <= int(max_side) or longest <= 0:
            return jpeg
        scale = float(max_side) / float(longest)
        resized = cv2.resize(
            image,
            (max(1, int(round(width * scale))),
             max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
        ok, buf = cv2.imencode(
            ".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        return buf.tobytes() if ok else jpeg
    except Exception:  # noqa: BLE001 - never fail a suggestion over resizing
        return jpeg


def _clean_facts(facts) -> list[str]:
    if isinstance(facts, str):
        facts = [facts]
    out: list[str] = []
    seen: set[str] = set()
    for raw in list(facts or []):
        text = " ".join(str(raw).split())[:240]
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            out.append(text)
    return out[: _MAX_FACTS]


def _summary(place: Place) -> dict:
    return {
        "id": place.id,
        "name": place.name,
        "display_name": place.display_name,
        "status": place.status,
        "seen_count": int(place.seen_count),
        "keyframe_count": len(place.keyframes),
        "fact_count": len(place.facts),
    }


class PlaceImporter:
    """Name suggestions + deterministic pending-place promotion."""

    def __init__(
        self,
        store: PlaceStore,
        vlm=None,
        *,
        max_images: int = 3,
        cache_s: float = _SUGGESTION_CACHE_S,
        clock=time.time,
    ) -> None:
        self.store = store
        self.vlm = vlm
        self.max_images = max(1, int(max_images))
        self.cache_s = float(cache_s)
        self._clock = clock
        self._lock = threading.Lock()

    @property
    def describe(self) -> str:
        return (getattr(self.vlm, "describe", None)
                or "no VLM configured")

    # -- suggestions ------------------------------------------------------

    def suggest(self, id_or_name: str, *, refresh: bool = False) -> dict:
        """Ask the model for a name/facts suggestion; do not change status."""
        with self._lock:
            place = self.store.get_place(id_or_name)
            if place is None:
                raise KeyError(id_or_name)
            if not place.pending:
                raise ValueError(f"place {place.id!r} is already confirmed")
            self._refresh_best_match(place)
            place = self.store.get_place(place.id)
            now = float(self._clock())
            if (not refresh and place.suggested_name
                    and place.suggested_at is not None
                    and now - float(place.suggested_at) < self.cache_s):
                return self._result(place, cached=True, latency_ms=0,
                                    confidence=None)
            if self.vlm is None:
                return {
                    "id": place.id,
                    "error": "vlm_not_configured",
                    "suggested_name": place.suggested_name,
                    "facts": list(place.suggested_facts),
                }
            views = self._views(place)
            if not views:
                return {
                    "id": place.id,
                    "error": "no_images",
                    "suggested_name": place.suggested_name,
                    "facts": list(place.suggested_facts),
                }
            t0 = time.monotonic()
            try:
                text = self.vlm.complete(
                    self._messages(views), max_tokens=_MAX_TOKENS)
            except Exception as exc:  # noqa: BLE001 - report, keep pending
                latency = int((time.monotonic() - t0) * 1000)
                _log(f"suggestion failed for {place.id}: {exc}")
                return {
                    "id": place.id,
                    "error": "vlm_error",
                    "detail": str(exc)[:200],
                    "suggested_name": place.suggested_name,
                    "facts": list(place.suggested_facts),
                    "latency_ms": latency,
                }
            latency = int((time.monotonic() - t0) * 1000)
            try:
                data = _extract_json(text)
            except ValueError as exc:
                _log(f"unparseable suggestion for {place.id}: {exc}")
                return {
                    "id": place.id,
                    "error": "unparseable_output",
                    "suggested_name": place.suggested_name,
                    "facts": list(place.suggested_facts),
                    "latency_ms": latency,
                }
            name = " ".join(str(data.get("name") or "").split())[:64]
            facts = _clean_facts(data.get("facts"))
            if not name:
                return {
                    "id": place.id,
                    "error": "empty_name",
                    "suggested_name": place.suggested_name,
                    "facts": list(place.suggested_facts),
                    "latency_ms": latency,
                }
            try:
                self.store.set_pending_suggestion(place.id, name, facts)
            except ValueError as exc:
                return {
                    "id": place.id,
                    "error": "invalid_name",
                    "detail": str(exc)[:200],
                    "latency_ms": latency,
                }
            place = self.store.get_place(place.id)
            confidence = data.get("confidence")
            try:
                confidence = max(0.0, min(1.0, float(confidence)))
            except (TypeError, ValueError):
                confidence = None
            result = self._result(place, cached=False, latency_ms=latency,
                                  confidence=confidence)
            _log(f"suggested {name!r} for {place.id} ({latency} ms)")
            return result

    def _refresh_best_match(self, place: Place) -> None:
        try:
            matches = self.store.match_place(
                place.id, status="confirmed", k=1)
        except Exception:  # noqa: BLE001 - the hint is best effort
            return
        if matches and float(matches[0].score) >= 0.5:
            self.store.set_pending_match(
                place.id, matches[0].place.id, float(matches[0].score))

    def _views(self, place: Place) -> list[View]:
        keyframes = list(place.keyframes)
        if not keyframes:
            return []
        far = [kf for kf in keyframes if kf.role == "far"]
        pool = far or keyframes
        if len(pool) > self.max_images:
            if self.max_images == 1:
                picks = [pool[0]]
            else:
                step = (len(pool) - 1) / float(self.max_images - 1)
                picks = [pool[min(len(pool) - 1, int(round(i * step)))]
                         for i in range(self.max_images)]
        else:
            picks = pool
        out: list[View] = []
        for kf in picks:
            try:
                with open(kf.path, "rb") as handle:
                    data = handle.read()
            except OSError:
                continue
            if data:
                out.append(View(kf.role, data, int(kf.stamp_us)))
        return out

    def _messages(self, views: list[View]) -> list[dict]:
        content: list[dict] = [{
            "type": "text",
            "text": "These keyframes are all from the same place.",
        }]
        for index, view in enumerate(views, start=1):
            content.append({
                "type": "text",
                "text": f"View {index} ({view.role}):",
            })
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": _image_data_url(_downscale_jpeg(view.jpeg)),
                },
            })
        return [
            {"role": "system", "content": IMPORT_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]

    def _result(
        self,
        place: Place,
        *,
        cached: bool,
        latency_ms: int,
        confidence: float | None,
    ) -> dict:
        conflict = None
        if place.suggested_name:
            existing = self.store.get_place(place.suggested_name)
            if existing is not None and existing.id != place.id:
                conflict = {"id": existing.id, "name": existing.name}
        return {
            "id": place.id,
            "suggested_name": place.suggested_name,
            "facts": list(place.suggested_facts),
            "confidence": confidence,
            "cached": bool(cached),
            "latency_ms": int(latency_ms),
            "conflict": conflict,
            "best_match": self._best_match_hint(place),
        }

    def _best_match_hint(self, place: Place) -> dict | None:
        if not place.best_match_id or place.best_match_score is None:
            return None
        target = self.store.get_place(place.best_match_id)
        if target is None:
            return None
        return {
            "id": target.id,
            "name": target.name,
            "score": round(float(place.best_match_score), 4),
        }

    # -- import -----------------------------------------------------------

    def import_pending(
        self,
        ids,
        *,
        names=None,
        merge_into=None,
        facts=None,
    ) -> dict:
        """Promote or merge pending places using operator-confirmed names.

        This path is deterministic and never calls the VLM.  Every item gets
        a result; one failure does not block the rest of the batch.
        """
        ids = [str(value) for value in (ids or []) if str(value).strip()]
        if not ids:
            raise ValueError("ids must contain at least one pending place")
        if len(ids) > 100:
            raise ValueError("at most 100 places can be imported per batch")
        names = dict(names or {})
        merge_into = dict(merge_into or {})
        facts_by_id = dict(facts or {})
        results: list[dict] = []
        imported = merged = failed = 0
        with self._lock:
            for pid in ids:
                place = self.store.get_place(pid)
                if place is None:
                    results.append({"id": pid, "status": "failed",
                                    "error": "not_found"})
                    failed += 1
                    continue
                if not place.pending:
                    results.append({"id": pid, "status": "failed",
                                    "error": "already_confirmed",
                                    "place": _summary(place)})
                    failed += 1
                    continue
                target = merge_into.get(pid)
                if target:
                    try:
                        outcome = self.store.merge_pending_into(
                            place.id, str(target))
                    except (KeyError, ValueError) as exc:
                        results.append({"id": pid, "status": "failed",
                                        "error": "merge_failed",
                                        "detail": str(exc)[:200]})
                        failed += 1
                        continue
                    confirmed = self.store.get_place(outcome["target_id"])
                    results.append({
                        "id": pid,
                        "status": "merged",
                        "merge": outcome,
                        "place": _summary(confirmed) if confirmed else None,
                    })
                    merged += 1
                    continue

                final_name = " ".join(
                    str(names.get(pid) or place.suggested_name or "").split())
                if not final_name and not is_placeholder_name(place.name):
                    final_name = place.name
                if not final_name:
                    results.append({"id": pid, "status": "failed",
                                    "error": "needs_name",
                                    "suggested_name": place.suggested_name})
                    failed += 1
                    continue
                name_source = (
                    "llm" if final_name == place.suggested_name
                    else "operator")
                fact_values = facts_by_id.get(pid)
                if fact_values is None:
                    fact_values = place.suggested_facts
                try:
                    promoted = self.store.promote_place(
                        place.id, final_name,
                        facts=fact_values, name_source=name_source)
                except ValueError as exc:
                    conflict = self.store.get_place(final_name)
                    results.append({
                        "id": pid,
                        "status": "failed",
                        "error": "name_conflict" if conflict else "invalid_name",
                        "detail": str(exc)[:200],
                        "conflict": ({"id": conflict.id, "name": conflict.name}
                                     if conflict is not None else None),
                    })
                    failed += 1
                    continue
                results.append({
                    "id": pid,
                    "status": "imported",
                    "name_source": name_source,
                    "place": _summary(promoted),
                })
                imported += 1
        return {
            "results": results,
            "imported": imported,
            "merged": merged,
            "failed": failed,
        }
