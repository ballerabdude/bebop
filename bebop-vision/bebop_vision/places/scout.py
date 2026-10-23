"""Lofi pending-place capture: no LLM, just DINOv3 + thresholds.

The scout runs beside the recorder (sharing the already-open camera slots)
and is deliberately cheap:

    far frame -> DINOv3 descriptor (46 ms)
       |-- confident match to a confirmed place -> touch seen, do nothing
       |-- match to a pending place            -> add evidence, no LLM
       `-- novel view -> in-memory candidate
                          -> stays 3 sightings / 2 s
                          -> write a PENDING place row (no LLM)

Only an operator import triggers the VLM (see :mod:`bebop_vision.places.importer`);
pending places are never recognized or navigable until promoted.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np

from .capture import View
from .store import Place, PlaceStore


def _log(msg: str) -> None:
    print(f"[scout] {msg}", flush=True)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    if a.size == 0 or a.size != b.size:
        return -1.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return -1.0
    return float(a @ b / denom)


@dataclass
class ScoutConfig:
    """Thresholds for the lofi capture loop.

    ``known_score``/``known_margin`` decide "this is a confirmed room".
    ``pending_merge_score`` decides "this is evidence for a pending room".
    ``candidate_merge_score`` clusters consecutive novel frames before a
    pending row is written.
    """

    interval_s: float = 1.0
    known_score: float = 0.80
    known_margin: float = 0.15
    pending_merge_score: float = 0.82
    candidate_merge_score: float = 0.80
    min_hint_score: float = 0.50
    min_hits: int = 3
    min_dwell_s: float = 2.0
    candidate_ttl_s: float = 6.0
    new_pending_cooldown_s: float = 5.0
    max_pending: int = 50
    pending_ttl_s: float = 14 * 86400
    max_keyframes: int = 8
    min_keyframe_delta: float = 0.03
    seen_update_s: float = 5.0
    prune_interval_s: float = 3600.0
    max_candidates: int = 8


@dataclass
class _Frame:
    view: View
    vec: np.ndarray


@dataclass
class _Candidate:
    centroid: np.ndarray
    hits: int
    first_seen: float
    last_seen: float
    frames: list[_Frame] = field(default_factory=list)
    best_match_id: str | None = None
    best_match_score: float | None = None


class PlaceScout:
    """Always-on pending-place capture (no VLM calls)."""

    def __init__(
        self,
        store: PlaceStore,
        encoder,
        capture,
        *,
        config: ScoutConfig | None = None,
        clock=time.monotonic,
        on_known=None,
    ) -> None:
        self.store = store
        self.encoder = encoder
        self.capture = capture
        self.config = config or ScoutConfig()
        # Called on every confident match to a confirmed place:
        # on_known(place_id, name, score, margin).  The recorder logs these
        # as /place so a drive carries its own from/to anchors.
        self.on_known = on_known
        self._clock = clock
        self._candidates: list[_Candidate] = []
        self._last_pending_at: float | None = None
        self._last_prune_at: float | None = None
        self._last_seen_write: dict[str, float] = {}
        self._lock = threading.Lock()
        self._tick_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._ticks = 0
        self._known = 0
        self._pending_hits = 0
        self._pending_created = 0
        self._errors = 0
        self._last_tick_ms = 0.0
        self._last_error: str | None = None
        self._last_action = ""

    # -- lifecycle --------------------------------------------------------

    @property
    def running(self) -> bool:
        return bool(self._thread is not None and self._thread.is_alive())

    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="place-scout", daemon=True)
        self._thread.start()
        _log(f"started (every {self.config.interval_s:.1f}s, "
             f"pending cap {self.config.max_pending})")

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
        self._thread = None

    def _loop(self) -> None:
        while not self._stop.wait(max(0.0, float(self.config.interval_s))):
            try:
                self.tick()
            except Exception as exc:  # noqa: BLE001 - never kill the loop
                self._errors += 1
                self._last_error = f"{type(exc).__name__}: {exc}"
                _log(f"tick error: {self._last_error}")

    # -- one scan ---------------------------------------------------------

    def tick(self, views=None, *, now: float | None = None) -> dict:
        """One capture scan. Tests call this directly with a fake clock."""
        with self._tick_lock:
            t0 = time.monotonic()
            now = float(self._clock() if now is None else now)
            try:
                views = list(views if views is not None else self.capture.grab())
            except Exception as exc:  # noqa: BLE001 - camera is best effort
                self._errors += 1
                self._last_error = f"capture: {exc}"
                return {"action": "error", "error": self._last_error}
            if not views:
                self._ticks += 1
                self._last_action = "no_frames"
                return {"action": "no_frames"}

            view = next((v for v in views if v.role == "far"), views[0])
            try:
                vec = np.asarray(
                    self.encoder.encode_jpeg(view.jpeg), dtype=np.float32)
            except Exception as exc:  # noqa: BLE001 - skip a bad frame
                self._errors += 1
                self._last_error = f"encode: {exc}"
                self._last_action = "encode_error"
                return {"action": "encode_error", "error": self._last_error}

            self._ticks += 1
            result = self._scan(view, vec, now)
            result.setdefault("action", self._last_action)
            self._last_tick_ms = round((time.monotonic() - t0) * 1000.0, 1)
            self._last_action = str(result["action"])
            self._maybe_prune(now)
            return result

    def _scan(self, view: View, vec: np.ndarray, now: float) -> dict:
        confirmed = self.store.match(vec, k=3, status="confirmed")
        top = confirmed[0] if confirmed else None
        runner = confirmed[1].score if len(confirmed) > 1 else 0.0
        margin = (float(top.score) - float(runner)) if top is not None else 0.0
        known = (top is not None
                 and float(top.score) >= self.config.known_score
                 and margin >= self.config.known_margin)

        if known:
            self._known += 1
            self._touch_confirmed(top.place.id, now)
            self._drop_candidate(vec)
            if self.on_known is not None:
                try:
                    self.on_known(top.place.id, top.place.name,
                                  float(top.score), float(margin))
                except Exception as exc:  # noqa: BLE001 - never kill the scan
                    self._last_error = f"on_known: {exc}"
            return {
                "action": "known",
                "place_id": top.place.id,
                "score": round(float(top.score), 4),
                "margin": round(margin, 4),
            }

        pending = self.store.match(vec, k=1, status="pending")
        if pending and float(pending[0].score) >= self.config.pending_merge_score:
            match = pending[0]
            self._pending_hits += 1
            self._drop_candidate(vec)
            hint_id = hint_score = None
            if (top is not None
                    and float(top.score) >= self.config.min_hint_score):
                hint_id = top.place.id
                hint_score = float(top.score)
            self._add_pending_evidence(
                match.place, view, vec,
                best_id=hint_id, best_score=hint_score,
            )
            return {
                "action": "pending",
                "place_id": match.place.id,
                "score": round(float(match.score), 4),
            }

        best_id = None
        best_score = None
        if top is not None and float(top.score) >= self.config.min_hint_score:
            best_id = top.place.id
            best_score = float(top.score)
        candidate = self._update_candidate(vec, view, now, best_id, best_score)
        self._prune_candidates(now)

        if (candidate.hits >= max(1, int(self.config.min_hits))
                and (now - candidate.first_seen) >= self.config.min_dwell_s
                and self._cooldown_ready(now)):
            place = self._create_pending(candidate)
            if place is not None:
                self._candidates.remove(candidate)
                self._last_pending_at = now
                self._pending_created += 1
                _log(f"pending {place.id} from {candidate.hits} sightings "
                     f"({len(place.keyframes)} keyframe(s))")
                return {
                    "action": "created",
                    "place_id": place.id,
                    "hits": candidate.hits,
                }
        return {
            "action": "candidate",
            "hits": candidate.hits,
            "best_match_score": best_score,
        }

    # -- candidate helpers ------------------------------------------------

    def _update_candidate(
        self,
        vec: np.ndarray,
        view: View,
        now: float,
        best_id: str | None,
        best_score: float | None,
    ) -> _Candidate:
        best: _Candidate | None = None
        best_sim = -1.0
        for candidate in self._candidates:
            sim = _cosine(candidate.centroid, vec)
            if sim > best_sim:
                best_sim = sim
                best = candidate
        if best is None or best_sim < self.config.candidate_merge_score:
            best = _Candidate(
                centroid=np.asarray(vec, dtype=np.float32).copy(),
                hits=1,
                first_seen=now,
                last_seen=now,
            )
            self._candidates.append(best)
            if len(self._candidates) > max(1, int(self.config.max_candidates)):
                self._candidates.remove(
                    min(self._candidates, key=lambda c: c.last_seen))
        else:
            updated = (best.centroid * best.hits
                       + np.asarray(vec, dtype=np.float32))
            norm = float(np.linalg.norm(updated))
            best.centroid = (updated / norm if norm > 1e-12 else updated)
            best.hits += 1
            best.last_seen = now
        if best_score is not None and (
                best.best_match_score is None
                or best_score > best.best_match_score):
            best.best_match_id = best_id
            best.best_match_score = best_score
        if len(best.frames) < 3:
            if not best.frames or all(
                    _cosine(frame.vec, vec)
                    < 1.0 - self.config.min_keyframe_delta
                    for frame in best.frames):
                best.frames.append(_Frame(view, np.asarray(vec, np.float32)))
        return best

    def _prune_candidates(self, now: float) -> None:
        ttl = max(0.0, float(self.config.candidate_ttl_s))
        if ttl <= 0:
            return
        self._candidates = [
            c for c in self._candidates if now - c.last_seen <= ttl
        ]

    def _drop_candidate(self, vec: np.ndarray) -> None:
        self._candidates = [
            c for c in self._candidates
            if _cosine(c.centroid, vec) < self.config.candidate_merge_score
        ]

    def _cooldown_ready(self, now: float) -> bool:
        if self._last_pending_at is None:
            return True
        return (now - self._last_pending_at
                >= max(0.0, float(self.config.new_pending_cooldown_s)))

    # -- writes -----------------------------------------------------------

    def _create_pending(self, candidate: _Candidate) -> Place | None:
        if self.store.count(status="pending") >= int(self.config.max_pending):
            self.store.prune_pending(
                ttl_s=self.config.pending_ttl_s,
                max_count=self.config.max_pending)
            if self.store.count(status="pending") >= int(self.config.max_pending):
                _log("pending queue full; dropping a candidate")
                return None
        try:
            place = self.store.create_pending_place(source="scout")
        except Exception as exc:  # noqa: BLE001 - keep the loop alive
            _log(f"could not create pending place: {exc}")
            return None
        for frame in candidate.frames[: max(1, int(self.config.max_keyframes))]:
            try:
                self.store.add_keyframe(
                    place.id, frame.view.jpeg,
                    role=frame.view.role, stamp_us=frame.view.stamp_us)
                self.store.add_descriptor(
                    place.id, frame.vec, source="dinov3:scout")
            except Exception as exc:  # noqa: BLE001 - keep other evidence
                _log(f"pending evidence failed: {exc}")
        if (candidate.best_match_score is not None
                and candidate.best_match_score >= self.config.min_hint_score):
            self.store.set_pending_match(
                place.id, candidate.best_match_id, candidate.best_match_score)
        self.store.note_pending_seen(
            place.id, seen_at=time.time())
        return self.store.get_place(place.id)

    def _add_pending_evidence(
        self,
        place: Place,
        view: View,
        vec: np.ndarray,
        *,
        best_id: str | None,
        best_score: float | None,
    ) -> None:
        try:
            self.store.note_pending_seen(
                place.id, score=best_score, match_id=best_id,
                seen_at=time.time())
        except Exception as exc:  # noqa: BLE001
            _log(f"pending seen update failed: {exc}")
        if len(place.keyframes) >= int(self.config.max_keyframes):
            return
        try:
            similarity = self.store.max_descriptor_similarity(place.id, vec)
            if similarity >= 1.0 - float(self.config.min_keyframe_delta):
                return
            self.store.add_keyframe(
                place.id, view.jpeg, role=view.role,
                stamp_us=view.stamp_us)
            self.store.add_descriptor(
                place.id, vec, source="dinov3:scout")
        except Exception as exc:  # noqa: BLE001
            _log(f"pending evidence failed: {exc}")

    def _touch_confirmed(self, place_id: str, now: float) -> None:
        last = self._last_seen_write.get(place_id)
        if (last is not None
                and now - last < float(self.config.seen_update_s)):
            return
        self._last_seen_write[place_id] = now
        try:
            self.store.touch_seen(place_id)
        except Exception as exc:  # noqa: BLE001
            _log(f"touch_seen failed: {exc}")
        if len(self._last_seen_write) > 256:
            cutoff = now - 300.0
            self._last_seen_write = {
                key: ts for key, ts in self._last_seen_write.items()
                if ts >= cutoff
            }

    def _maybe_prune(self, now: float) -> None:
        interval = max(60.0, float(self.config.prune_interval_s))
        if (self._last_prune_at is not None
                and now - self._last_prune_at < interval):
            return
        self._last_prune_at = now
        try:
            self.store.prune_pending(
                ttl_s=self.config.pending_ttl_s,
                max_count=self.config.max_pending)
        except Exception as exc:  # noqa: BLE001
            _log(f"prune failed: {exc}")

    # -- introspection ----------------------------------------------------

    def status(self) -> dict:
        with self._lock:
            pending = self.store.count(status="pending")
            return {
                "running": self.running,
                "interval_s": self.config.interval_s,
                "ticks": self._ticks,
                "known": self._known,
                "pending_hits": self._pending_hits,
                "pending_created": self._pending_created,
                "pending_count": pending,
                "candidates": len(self._candidates),
                "last_action": self._last_action,
                "last_tick_ms": self._last_tick_ms,
                "errors": self._errors,
                "last_error": self._last_error,
                "thresholds": {
                    "known_score": self.config.known_score,
                    "known_margin": self.config.known_margin,
                    "pending_merge_score": self.config.pending_merge_score,
                    "min_hits": self.config.min_hits,
                    "min_dwell_s": self.config.min_dwell_s,
                    "max_pending": self.config.max_pending,
                    "pending_ttl_s": self.config.pending_ttl_s,
                },
            }
