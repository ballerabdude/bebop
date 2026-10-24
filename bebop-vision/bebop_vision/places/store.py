"""SQLite-backed semantic place memory for the Places service.

A *place* is a named location the robot has been taught. It carries:

  - a name (operator-set, unique) and timestamps,
  - one or more keyframe JPEGs (the view at teach time),
  - DINOv3 descriptors (L2-normalised float32) for appearance matching,
  - free-text facts, each attributed to its source ("operator", or "import"
    for the facts confirmed when a pending place is imported),
  - route edges between places with traversal/success counters.

Deliberately dependency-light: stdlib sqlite3 plus numpy for the descriptor
vectors. This module knows nothing about the camera or the LLM, so it is
fully unit-testable offline.

Localisation here is *appearance-based place recognition*: a query
descriptor is matched against the stored ones by cosine similarity, and the
best place wins only if it clears a similarity margin over the runner-up.
Places are linked by *route knowledge* (edges), not by a global metric map.
"""
from __future__ import annotations

import json
import math
import sqlite3
import statistics
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Bounds a caller-supplied guide path (`record_traversal`).
MAX_GUIDE_POINTS = 10_000

# odom_x/odom_y/odom_theta are never written or read; they stay in the DDL so
# databases created with them need no table rebuild.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS places (
    id           TEXT PRIMARY KEY,
    name         TEXT NOT NULL COLLATE NOCASE UNIQUE,
    created_at   REAL NOT NULL,
    updated_at   REAL NOT NULL,
    last_seen_at REAL,
    odom_x       REAL,
    odom_y       REAL,
    odom_theta   REAL,
    status       TEXT NOT NULL DEFAULT 'confirmed',
    source       TEXT NOT NULL DEFAULT 'operator',
    confirmed_at REAL,
    seen_count   INTEGER NOT NULL DEFAULT 1,
    best_match_id    TEXT,
    best_match_score REAL,
    suggested_name   TEXT,
    suggested_facts_json TEXT,
    suggested_at     REAL,
    name_source      TEXT
);
CREATE TABLE IF NOT EXISTS descriptors (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    place_id   TEXT NOT NULL REFERENCES places(id) ON DELETE CASCADE,
    dim        INTEGER NOT NULL,
    vec        BLOB NOT NULL,
    source     TEXT NOT NULL DEFAULT 'dinov3',
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS facts (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    place_id   TEXT NOT NULL REFERENCES places(id) ON DELETE CASCADE,
    text       TEXT NOT NULL,
    source     TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS keyframes (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    place_id   TEXT NOT NULL REFERENCES places(id) ON DELETE CASCADE,
    path       TEXT NOT NULL,
    role       TEXT NOT NULL,
    stamp_us   INTEGER NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS place_edges (
    from_place        TEXT NOT NULL REFERENCES places(id) ON DELETE CASCADE,
    to_place          TEXT NOT NULL REFERENCES places(id) ON DELETE CASCADE,
    traversals        INTEGER NOT NULL DEFAULT 0,
    successes         INTEGER NOT NULL DEFAULT 0,
    last_attempt_at   REAL,
    last_success_at   REAL,
    median_duration_s REAL,
    guide_path_json   TEXT,
    durations_json    TEXT,
    created_at        REAL NOT NULL,
    updated_at        REAL NOT NULL,
    PRIMARY KEY (from_place, to_place)
);
CREATE INDEX IF NOT EXISTS idx_desc_place  ON descriptors(place_id);
CREATE INDEX IF NOT EXISTS idx_facts_place ON facts(place_id);
CREATE INDEX IF NOT EXISTS idx_kf_place    ON keyframes(place_id);
CREATE INDEX IF NOT EXISTS idx_edges_to    ON place_edges(to_place);
"""


@dataclass
class Fact:
    text: str
    source: str
    created_at: float


@dataclass
class Keyframe:
    path: str
    role: str
    stamp_us: int
    created_at: float


@dataclass
class PlaceEdge:
    """Route knowledge: "you got from A to B by this route"."""

    from_place: str
    to_place: str
    from_name: str
    to_name: str
    traversals: int
    successes: int
    last_attempt_at: float | None
    last_success_at: float | None
    median_duration_s: float | None
    guide_path: list[list[float]] | None
    created_at: float
    updated_at: float

    @property
    def success_rate(self) -> float:
        if self.traversals <= 0:
            return 0.0
        return self.successes / self.traversals

    def to_dict(self) -> dict:
        return {
            "from": self.from_place,
            "to": self.to_place,
            "from_name": self.from_name,
            "to_name": self.to_name,
            "traversals": self.traversals,
            "successes": self.successes,
            "success_rate": round(self.success_rate, 4),
            "last_attempt_at": self.last_attempt_at,
            "last_success_at": self.last_success_at,
            "median_duration_s": self.median_duration_s,
            "guide_path": self.guide_path,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


#: Pending places get a hidden, unique placeholder name until the operator
#: imports them.  The prefix keeps them distinguishable from real names.
PENDING_NAME_PREFIX = "__pending__"


def is_placeholder_name(name: str | None) -> bool:
    return str(name or "").startswith(PENDING_NAME_PREFIX)


@dataclass
class Place:
    id: str
    name: str
    created_at: float
    updated_at: float
    last_seen_at: float | None = None
    facts: list[Fact] = field(default_factory=list)
    keyframes: list[Keyframe] = field(default_factory=list)
    status: str = "confirmed"
    source: str = "operator"
    confirmed_at: float | None = None
    seen_count: int = 1
    best_match_id: str | None = None
    best_match_score: float | None = None
    suggested_name: str | None = None
    suggested_facts: list[str] = field(default_factory=list)
    suggested_at: float | None = None
    name_source: str | None = None

    @property
    def pending(self) -> bool:
        return self.status == "pending"

    @property
    def display_name(self) -> str:
        if not self.pending:
            return self.name
        if self.suggested_name:
            return self.suggested_name
        if not is_placeholder_name(self.name):
            return self.name
        return "Unnamed place"

    def to_dict(self, *, root: Path | None = None) -> dict:
        """JSON-ready view. Keyframe paths are relative to `root` when given,
        so the app can fetch them over HTTP without leaking host paths."""
        kf = []
        for k in self.keyframes:
            path = k.path
            if root is not None:
                try:
                    path = str(Path(k.path).relative_to(root))
                except ValueError:
                    pass
            kf.append({"path": path, "role": k.role, "stamp_us": k.stamp_us})
        return {
            "id": self.id,
            "name": self.name,
            "display_name": self.display_name,
            "status": self.status,
            "source": self.source,
            "confirmed_at": self.confirmed_at,
            "seen_count": int(self.seen_count),
            "best_match_id": self.best_match_id,
            "best_match_score": self.best_match_score,
            "suggested_name": self.suggested_name,
            "suggested_facts": list(self.suggested_facts),
            "suggested_at": self.suggested_at,
            "name_source": self.name_source,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_seen_at": self.last_seen_at,
            "facts": [
                {"text": f.text, "source": f.source, "created_at": f.created_at}
                for f in self.facts
            ],
            "keyframes": kf,
        }


@dataclass
class Match:
    """One place's best score for a query descriptor."""

    place: Place
    score: float


def _now() -> float:
    return time.time()


def _parse_duration(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("duration_s must be a finite number >= 0 or null")
    out = float(value)
    if not math.isfinite(out) or out < 0:
        raise ValueError("duration_s must be a finite number >= 0 or null")
    return out


def _number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be a finite number")
    return out


def parse_guide_path(path) -> list[list[float]] | None:
    """Validate a caller-supplied guide path ([[x, y], ...], metres)."""
    if path is None:
        return None
    if not isinstance(path, (list, tuple)):
        raise ValueError("guide_path must be a list of [x, y] pairs")
    if len(path) > MAX_GUIDE_POINTS:
        raise ValueError(f"guide_path must have <= {MAX_GUIDE_POINTS} points")
    out: list[list[float]] = []
    for i, point in enumerate(path):
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError(f"guide_path[{i}] must be an [x, y] pair")
        out.append([_number(point[0], f"guide_path[{i}].x"),
                    _number(point[1], f"guide_path[{i}].y")])
    return out


def _load_durations(raw) -> list[float]:
    if not raw:
        return []
    try:
        values = json.loads(raw)
    except (TypeError, ValueError):
        return []
    if not isinstance(values, list):
        return []
    return [float(v) for v in values
            if isinstance(v, (int, float)) and not isinstance(v, bool)
            and math.isfinite(float(v))]


def _clean_name(name: str) -> str:
    name = " ".join(str(name).split())
    if not name:
        raise ValueError("place name must be non-empty")
    if len(name) > 64:
        raise ValueError("place name must be <= 64 characters")
    return name


def _clean_status(status: str) -> str:
    value = str(status or "").strip().lower()
    if value not in ("pending", "confirmed"):
        raise ValueError("place status must be 'pending' or 'confirmed'")
    return value


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
    return out[:8]


class PlaceStore:
    """A place registry backed by one SQLite file.

    Thread-safe: the HTTP service handles requests on a thread pool, so the
    single connection is guarded by a lock. `root` also holds the keyframe
    JPEGs under `keyframes/<place_id>/`.
    """

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.keyframe_dir = self.root / "keyframes"
        self.keyframe_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "places.db"
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(_SCHEMA)
        self._migrate()
        self._conn.commit()

    def _migrate(self) -> None:
        """Additive migrations for databases created before pending places.

        SQLite has no portable ``ADD COLUMN IF NOT EXISTS``; inspect the
        table and add only what is missing.  Existing rows stay confirmed.
        """
        columns = {
            str(row["name"])
            for row in self._conn.execute("PRAGMA table_info(places)")
        }
        additions = {
            "status": "TEXT NOT NULL DEFAULT 'confirmed'",
            "source": "TEXT NOT NULL DEFAULT 'operator'",
            "confirmed_at": "REAL",
            "seen_count": "INTEGER NOT NULL DEFAULT 1",
            "best_match_id": "TEXT",
            "best_match_score": "REAL",
            "suggested_name": "TEXT",
            "suggested_facts_json": "TEXT",
            "suggested_at": "REAL",
            "name_source": "TEXT",
        }
        for name, decl in additions.items():
            if name not in columns:
                self._conn.execute(
                    f"ALTER TABLE places ADD COLUMN {name} {decl}")
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_places_status ON places(status)")

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- places -----------------------------------------------------------

    def create_place(
        self,
        name: str,
        *,
        status: str = "confirmed",
        source: str = "operator",
    ) -> Place:
        name = _clean_name(name)
        status = _clean_status(status)
        pid = uuid_like(name)
        ts = _now()
        confirmed_at = ts if status == "confirmed" else None
        with self._lock:
            try:
                self._conn.execute(
                    "INSERT INTO places (id, name, created_at, updated_at,"
                    " status, source, confirmed_at, name_source)"
                    " VALUES (?,?,?,?,?,?,?,?)",
                    (pid, name, ts, ts, status, source, confirmed_at,
                     "operator" if status == "confirmed" else None),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"a place named {name!r} already exists") from exc
            self._conn.commit()
        return self.get_place(pid)

    def create_pending_place(self, *, source: str = "scout") -> Place:
        """Create a hidden candidate place for operator review.

        The name is a unique placeholder; ``display_name`` hides it.  The
        place is not recognized until ``promote_place`` is called.
        """
        import uuid

        name = f"{PENDING_NAME_PREFIX}{uuid.uuid4().hex[:12]}"
        return self.create_place(name, status="pending", source=source)

    def list_places(self, *, status: str | None = None) -> list[Place]:
        with self._lock:
            if status is None:
                rows = self._conn.execute(
                    "SELECT id FROM places ORDER BY name COLLATE NOCASE"
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT id FROM places WHERE status = ?"
                    " ORDER BY name COLLATE NOCASE", (_clean_status(status),)
                ).fetchall()
        return [self.get_place(r["id"]) for r in rows]

    def list_pending(self, *, limit: int = 100) -> list[Place]:
        """Pending places, newest evidence first (for the review queue)."""
        limit = max(1, int(limit))
        with self._lock:
            rows = self._conn.execute(
                "SELECT id FROM places WHERE status = 'pending'"
                " ORDER BY COALESCE(last_seen_at, created_at) DESC,"
                " created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self.get_place(r["id"]) for r in rows]

    def get_place(self, id_or_name: str) -> Place | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM places WHERE id = ? OR name = ? COLLATE NOCASE",
                (id_or_name, id_or_name),
            ).fetchone()
            if row is None:
                return None
            place = self._row_to_place(row)
            place.facts = [
                Fact(r["text"], r["source"], r["created_at"])
                for r in self._conn.execute(
                    "SELECT text, source, created_at FROM facts WHERE place_id = ?"
                    " ORDER BY created_at", (place.id,),
                )
            ]
            place.keyframes = [
                Keyframe(r["path"], r["role"], r["stamp_us"], r["created_at"])
                for r in self._conn.execute(
                    "SELECT path, role, stamp_us, created_at FROM keyframes"
                    " WHERE place_id = ? ORDER BY created_at", (place.id,),
                )
            ]
        return place

    def rename_place(self, id_or_name: str, new_name: str) -> Place:
        new_name = _clean_name(new_name)
        place = self.get_place(id_or_name)
        if place is None:
            raise KeyError(id_or_name)
        with self._lock:
            try:
                self._conn.execute(
                    "UPDATE places SET name = ?, updated_at = ? WHERE id = ?",
                    (new_name, _now(), place.id),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"a place named {new_name!r} already exists") from exc
            self._conn.commit()
        return self.get_place(place.id)

    def delete_place(self, id_or_name: str) -> bool:
        place = self.get_place(id_or_name)
        if place is None:
            return False
        with self._lock:
            self._conn.execute("DELETE FROM places WHERE id = ?", (place.id,))
            self._conn.commit()
        for kf in place.keyframes:
            Path(kf.path).unlink(missing_ok=True)
        kdir = self.keyframe_dir / place.id
        if kdir.exists():
            for leftover in kdir.iterdir():
                leftover.unlink(missing_ok=True)
            kdir.rmdir()
        return True

    def touch_seen(self, id_or_name: str) -> None:
        place = self.get_place(id_or_name)
        if place is None:
            return
        with self._lock:
            self._conn.execute(
                "UPDATE places SET last_seen_at = ?, seen_count = seen_count + 1"
                " WHERE id = ?", (_now(), place.id)
            )
            self._conn.commit()

    def note_pending_seen(
        self,
        id_or_name: str,
        *,
        score: float | None = None,
        match_id: str | None = None,
        seen_at: float | None = None,
    ) -> None:
        """Increment a pending place's evidence counters.

        ``score``/``match_id`` update the best confirmed-place hint when the
        new observation is a better match than the stored one.
        """
        place = self.get_place(id_or_name)
        if place is None or not place.pending:
            return
        ts = _now() if seen_at is None else float(seen_at)
        with self._lock:
            row = self._conn.execute(
                "SELECT best_match_score FROM places WHERE id = ?", (place.id,)
            ).fetchone()
            old = (row["best_match_score"] if row is not None else None)
            if score is not None and (old is None or float(score) > float(old)):
                self._conn.execute(
                    "UPDATE places SET last_seen_at = ?,"
                    " seen_count = seen_count + 1, best_match_id = ?,"
                    " best_match_score = ? WHERE id = ?",
                    (ts, match_id, float(score), place.id),
                )
            else:
                self._conn.execute(
                    "UPDATE places SET last_seen_at = ?,"
                    " seen_count = seen_count + 1 WHERE id = ?",
                    (ts, place.id),
                )
            self._conn.commit()

    def set_pending_match(
        self, id_or_name: str, match_id: str | None, score: float | None
    ) -> None:
        place = self.get_place(id_or_name)
        if place is None or not place.pending:
            return
        if score is None:
            return
        with self._lock:
            row = self._conn.execute(
                "SELECT best_match_score FROM places WHERE id = ?", (place.id,)
            ).fetchone()
            old = (row["best_match_score"] if row is not None else None)
            if old is None or float(score) > float(old):
                self._conn.execute(
                    "UPDATE places SET best_match_id = ?, best_match_score = ?"
                    " WHERE id = ?", (match_id, float(score), place.id),
                )
                self._conn.commit()

    def set_pending_suggestion(
        self,
        id_or_name: str,
        name: str | None,
        facts: list[str] | None = None,
        *,
        at: float | None = None,
    ) -> None:
        place = self.get_place(id_or_name)
        if place is None or not place.pending:
            return
        clean_name = _clean_name(name) if name else None
        clean_facts = _clean_facts(facts or [])
        with self._lock:
            self._conn.execute(
                "UPDATE places SET suggested_name = ?,"
                " suggested_facts_json = ?, suggested_at = ?,"
                " updated_at = ? WHERE id = ?",
                (clean_name,
                 json.dumps(clean_facts, separators=(",", ":")),
                 _now() if at is None else float(at), _now(), place.id),
            )
            self._conn.commit()

    def promote_place(
        self,
        id_or_name: str,
        name: str,
        *,
        facts: list[str] | None = None,
        name_source: str = "operator",
    ) -> Place:
        """Promote a pending place to a confirmed, recognizable place."""
        place = self.get_place(id_or_name)
        if place is None:
            raise KeyError(id_or_name)
        if not place.pending:
            raise ValueError(f"place {place.id!r} is already confirmed")
        name = _clean_name(name)
        if is_placeholder_name(name):
            raise ValueError("a confirmed place needs a real name")
        ts = _now()
        with self._lock:
            try:
                self._conn.execute(
                    "UPDATE places SET name = ?, status = 'confirmed',"
                    " confirmed_at = ?, updated_at = ?, name_source = ?"
                    " WHERE id = ?",
                    (name, ts, ts, str(name_source or "operator"), place.id),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"a place named {name!r} already exists") from exc
            self._conn.commit()
        self.add_facts_unique(place.id, facts or [], source="import")
        return self.get_place(place.id)

    def count(self, *, status: str | None = None) -> int:
        with self._lock:
            if status is None:
                return int(
                    self._conn.execute(
                        "SELECT COUNT(*) FROM places").fetchone()[0])
            return int(
                self._conn.execute(
                    "SELECT COUNT(*) FROM places WHERE status = ?",
                    (_clean_status(status),)).fetchone()[0])

    # -- descriptors ------------------------------------------------------

    def add_descriptor(
        self, id_or_name: str, vec: np.ndarray, *, source: str = "dinov3"
    ) -> None:
        place = self.get_place(id_or_name)
        if place is None:
            raise KeyError(id_or_name)
        v = np.asarray(vec, dtype=np.float32).reshape(-1)
        if v.size == 0:
            raise ValueError("empty descriptor")
        v = v / (np.linalg.norm(v) + 1e-12)
        with self._lock:
            self._conn.execute(
                "INSERT INTO descriptors (place_id, dim, vec, source, created_at)"
                " VALUES (?,?,?,?,?)",
                (place.id, int(v.size), v.tobytes(), source, _now()),
            )
            self._conn.commit()

    def descriptor_count(self, id_or_name: str | None = None) -> int:
        with self._lock:
            if id_or_name is None:
                return int(
                    self._conn.execute("SELECT COUNT(*) FROM descriptors").fetchone()[0]
                )
            place = self.get_place(id_or_name)
            if place is None:
                return 0
            return int(
                self._conn.execute(
                    "SELECT COUNT(*) FROM descriptors WHERE place_id = ?", (place.id,)
                ).fetchone()[0]
            )

    def match(
        self, vec: np.ndarray, *, k: int = 3, status: str | None = None
    ) -> list[Match]:
        """Rank places by best cosine similarity to `vec`.

        Descriptors are stored L2-normalised, so the score is a dot product.
        Returns up to `k` places, best first.  ``status`` restricts the
        candidates (the scout matches confirmed and pending separately).
        """
        q = np.asarray(vec, dtype=np.float32).reshape(-1)
        if q.size == 0:
            return []
        q = q / (np.linalg.norm(q) + 1e-12)
        with self._lock:
            if status is None:
                rows = self._conn.execute(
                    "SELECT place_id, dim, vec FROM descriptors"
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT d.place_id, d.dim, d.vec FROM descriptors d"
                    " JOIN places p ON p.id = d.place_id"
                    " WHERE p.status = ?", (_clean_status(status),)
                ).fetchall()
        if not rows:
            return []
        best: dict[str, float] = {}
        for r in rows:
            if r["dim"] != q.size:
                continue
            v = np.frombuffer(r["vec"], dtype=np.float32)
            score = float(v @ q)
            pid = r["place_id"]
            if pid not in best or score > best[pid]:
                best[pid] = score
        ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)[:k]
        out: list[Match] = []
        for pid, score in ranked:
            place = self.get_place(pid)
            if place is not None:
                out.append(Match(place=place, score=score))
        return out

    def match_place(
        self, id_or_name: str, *, status: str | None = "confirmed", k: int = 1
    ) -> list[Match]:
        """Best target places for one place's stored descriptors.

        Used by import/suggestion to refresh the "closest confirmed place"
        hint without re-encoding the pending keyframes.
        """
        place = self.get_place(id_or_name)
        if place is None:
            return []
        with self._lock:
            rows = self._conn.execute(
                "SELECT dim, vec FROM descriptors WHERE place_id = ?",
                (place.id,),
            ).fetchall()
        if not rows:
            return []
        candidates: list[np.ndarray] = [
            np.frombuffer(r["vec"], dtype=np.float32) for r in rows
        ]
        best: dict[str, float] = {}
        with self._lock:
            if status is None:
                target_rows = self._conn.execute(
                    "SELECT place_id, dim, vec FROM descriptors"
                ).fetchall()
            else:
                target_rows = self._conn.execute(
                    "SELECT d.place_id, d.dim, d.vec FROM descriptors d"
                    " JOIN places p ON p.id = d.place_id"
                    " WHERE p.status = ?", (_clean_status(status),)
                ).fetchall()
        for tr in target_rows:
            if tr["place_id"] == place.id:
                continue
            target = np.frombuffer(tr["vec"], dtype=np.float32)
            for q in candidates:
                if q.size != target.size:
                    continue
                score = float(target @ q)
                pid = tr["place_id"]
                if pid not in best or score > best[pid]:
                    best[pid] = score
        ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)[:k]
        out: list[Match] = []
        for pid, score in ranked:
            target = self.get_place(pid)
            if target is not None:
                out.append(Match(place=target, score=score))
        return out

    def max_descriptor_similarity(
        self, id_or_name: str, vec: np.ndarray
    ) -> float:
        """Highest cosine similarity between `vec` and a place's descriptors."""
        place = self.get_place(id_or_name)
        if place is None:
            return -1.0
        q = np.asarray(vec, dtype=np.float32).reshape(-1)
        if q.size == 0:
            return -1.0
        q = q / (np.linalg.norm(q) + 1e-12)
        with self._lock:
            rows = self._conn.execute(
                "SELECT dim, vec FROM descriptors WHERE place_id = ?",
                (place.id,),
            ).fetchall()
        best = -1.0
        for r in rows:
            if r["dim"] != q.size:
                continue
            best = max(best, float(
                np.frombuffer(r["vec"], dtype=np.float32) @ q))
        return best

    # -- place graph (route knowledge) ------------------------------------

    def record_traversal(
        self,
        from_place: str,
        to_place: str,
        *,
        success: bool,
        duration_s=None,
        guide_path=None,
    ) -> tuple[PlaceEdge, bool]:
        """Record one A->B attempt.

        ``success`` means arrival was confirmed.  A guide path is stored only
        for successful traversals (a failed route is not a guide).  The median
        duration is over the last <=100 successful durations that carried one.
        Returns ``(edge, created)``.
        """
        src = self.get_place(from_place)
        if src is None:
            raise KeyError(from_place)
        dst = self.get_place(to_place)
        if dst is None:
            raise KeyError(to_place)
        if not isinstance(success, bool):
            raise ValueError("success must be a boolean")
        duration = _parse_duration(duration_s)
        guide = parse_guide_path(guide_path)
        ts = _now()
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM place_edges WHERE from_place = ? AND"
                " to_place = ?",
                (src.id, dst.id),
            ).fetchone()
            created = row is None
            if created:
                traversals = 1
                successes = 1 if success else 0
                durations: list[float] = []
                last_success_at = ts if success else None
                guide_stored = guide if (success and guide is not None) else None
            else:
                traversals = int(row["traversals"]) + 1
                successes = int(row["successes"]) + (1 if success else 0)
                durations = _load_durations(row["durations_json"])
                last_success_at = (ts if success
                                   else row["last_success_at"])
                guide_stored = (json.loads(row["guide_path_json"])
                                if row["guide_path_json"] else None)
            if success and duration is not None:
                durations = (durations + [duration])[-100:]
            if success and guide is not None:
                guide_stored = guide
            median = (float(statistics.median(durations))
                      if durations else None)
            guide_json = (json.dumps(guide_stored, separators=(",", ":"))
                          if guide_stored is not None else None)
            durations_json = json.dumps(durations, separators=(",", ":"))
            if created:
                self._conn.execute(
                    "INSERT INTO place_edges (from_place, to_place,"
                    " traversals, successes, last_attempt_at,"
                    " last_success_at, median_duration_s, guide_path_json,"
                    " durations_json, created_at, updated_at)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (src.id, dst.id, traversals, successes, ts,
                     last_success_at, median, guide_json, durations_json,
                     ts, ts),
                )
            else:
                self._conn.execute(
                    "UPDATE place_edges SET traversals = ?, successes = ?,"
                    " last_attempt_at = ?, last_success_at = ?,"
                    " median_duration_s = ?, guide_path_json = ?,"
                    " durations_json = ?, updated_at = ?"
                    " WHERE from_place = ? AND to_place = ?",
                    (traversals, successes, ts, last_success_at, median,
                     guide_json, durations_json, ts, src.id, dst.id),
                )
            self._conn.commit()
        edge = self.get_edge(src.id, dst.id)
        if edge is None:  # pragma: no cover - the upsert just wrote it
            raise RuntimeError("place edge vanished after record")
        return edge, created

    def get_edge(self, from_place: str, to_place: str) -> PlaceEdge | None:
        src = self.get_place(from_place)
        dst = self.get_place(to_place)
        if src is None or dst is None:
            return None
        with self._lock:
            row = self._conn.execute(
                "SELECT e.*, f.name AS from_name, t.name AS to_name"
                " FROM place_edges e"
                " JOIN places f ON f.id = e.from_place"
                " JOIN places t ON t.id = e.to_place"
                " WHERE e.from_place = ? AND e.to_place = ?",
                (src.id, dst.id),
            ).fetchone()
            if row is None:
                return None
            return self._row_to_edge(row)

    def list_edges(self) -> list[PlaceEdge]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT e.*, f.name AS from_name, t.name AS to_name"
                " FROM place_edges e"
                " JOIN places f ON f.id = e.from_place"
                " JOIN places t ON t.id = e.to_place"
                " ORDER BY f.name COLLATE NOCASE, t.name COLLATE NOCASE"
            ).fetchall()
            return [self._row_to_edge(r) for r in rows]

    def graph(self) -> dict:
        """JSON-ready place graph: nodes (places) and traversal edges."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, name, created_at, last_seen_at FROM places"
                " WHERE status = 'confirmed'"
                " ORDER BY name COLLATE NOCASE"
            ).fetchall()
            nodes = [
                {
                    "id": r["id"],
                    "name": r["name"],
                    "created_at": r["created_at"],
                    "last_seen_at": r["last_seen_at"],
                }
                for r in rows
            ]
        edges = [e.to_dict() for e in self.list_edges()]
        return {
            "nodes": nodes,
            "edges": edges,
            "node_count": len(nodes),
            "edge_count": len(edges),
        }

    # -- facts / keyframes ------------------------------------------------

    def add_fact(self, id_or_name: str, text: str, *, source: str = "operator") -> Fact:
        place = self.get_place(id_or_name)
        if place is None:
            raise KeyError(id_or_name)
        text = " ".join(str(text).split())
        if not text:
            raise ValueError("fact must be non-empty")
        ts = _now()
        with self._lock:
            self._conn.execute(
                "INSERT INTO facts (place_id, text, source, created_at)"
                " VALUES (?,?,?,?)",
                (place.id, text, source, ts),
            )
            self._conn.execute(
                "UPDATE places SET updated_at = ? WHERE id = ?", (ts, place.id)
            )
            self._conn.commit()
        return Fact(text, source, ts)

    def add_facts_unique(
        self,
        id_or_name: str,
        texts,
        *,
        source: str = "import",
        max_new: int = 8,
    ) -> int:
        """Add facts that are not already present (case-insensitive)."""
        place = self.get_place(id_or_name)
        if place is None:
            raise KeyError(id_or_name)
        existing = {f.text.strip().lower() for f in place.facts}
        added = 0
        for text in _clean_facts(texts):
            key = text.lower()
            if key in existing:
                continue
            self.add_fact(place.id, text, source=source)
            existing.add(key)
            added += 1
            if added >= int(max_new):
                break
        return added

    def merge_pending_into(
        self,
        source_id: str,
        target_id: str,
        *,
        max_keyframes: int = 32,
    ) -> dict:
        """Merge one pending place's evidence into a confirmed place.

        Descriptors are always copied (recognition benefits even when the
        target already holds its keyframe budget); keyframes and facts are
        copied up to the caps.  The source pending row is deleted afterwards.
        """
        src = self.get_place(source_id)
        dst = self.get_place(target_id)
        if src is None:
            raise KeyError(source_id)
        if dst is None:
            raise KeyError(target_id)
        if not src.pending:
            raise ValueError("source is not a pending place")
        if dst.pending:
            raise ValueError("target must be a confirmed place")
        if src.id == dst.id:
            raise ValueError("cannot merge a place into itself")

        with self._lock:
            rows = self._conn.execute(
                "SELECT dim, vec FROM descriptors WHERE place_id = ?",
                (src.id,),
            ).fetchall()
        added_desc = 0
        for row in rows:
            try:
                vec = np.frombuffer(row["vec"], dtype=np.float32)
                self.add_descriptor(
                    dst.id, vec, source="dinov3:pending-merge")
                added_desc += 1
            except Exception:  # noqa: BLE001 - keep the other evidence
                continue

        added_kf = 0
        room = max(0, int(max_keyframes) - len(dst.keyframes))
        for kf in src.keyframes:
            if added_kf >= room:
                break
            try:
                data = Path(kf.path).read_bytes()
                self.add_keyframe(
                    dst.id, data, role=kf.role, stamp_us=kf.stamp_us)
                added_kf += 1
            except Exception:  # noqa: BLE001 - keep the other evidence
                continue

        added_facts = self.add_facts_unique(
            dst.id, [f.text for f in src.facts], source="pending-merge")
        self.touch_seen(dst.id)
        self.delete_place(src.id)
        return {
            "source_id": src.id,
            "target_id": dst.id,
            "keyframes": added_kf,
            "descriptors": added_desc,
            "facts": added_facts,
        }

    def prune_pending(
        self,
        *,
        ttl_s: float = 14 * 86400,
        max_count: int = 50,
        now: float | None = None,
    ) -> int:
        """Delete stale pending places and cap the queue.

        Confirmed places are never touched.  Returns the number deleted.
        """
        now = time.time() if now is None else float(now)
        victims: list[str] = []
        with self._lock:
            if ttl_s > 0:
                rows = self._conn.execute(
                    "SELECT id FROM places WHERE status = 'pending'"
                    " AND COALESCE(last_seen_at, created_at) < ?",
                    (now - float(ttl_s),),
                ).fetchall()
                victims.extend(str(r["id"]) for r in rows)
            rows = self._conn.execute(
                "SELECT id FROM places WHERE status = 'pending'"
                " ORDER BY COALESCE(last_seen_at, created_at) DESC,"
                " created_at DESC"
            ).fetchall()
            keep = max(0, int(max_count))
            victims.extend(str(r["id"]) for r in rows[keep:])
        deleted = 0
        for pid in dict.fromkeys(victims):
            if self.delete_place(pid):
                deleted += 1
        return deleted

    def add_keyframe(
        self,
        id_or_name: str,
        jpeg: bytes,
        *,
        role: str = "near",
        stamp_us: int = 0,
    ) -> Keyframe:
        place = self.get_place(id_or_name)
        if place is None:
            raise KeyError(id_or_name)
        if not jpeg:
            raise ValueError("empty keyframe")
        kdir = self.keyframe_dir / place.id
        kdir.mkdir(parents=True, exist_ok=True)
        ts = _now()
        path = kdir / f"{int(stamp_us) or int(ts * 1e6)}_{role}.jpg"
        path.write_bytes(jpeg)
        with self._lock:
            self._conn.execute(
                "INSERT INTO keyframes (place_id, path, role, stamp_us, created_at)"
                " VALUES (?,?,?,?,?)",
                (place.id, str(path), role, int(stamp_us), ts),
            )
            self._conn.commit()
        return Keyframe(str(path), role, int(stamp_us), ts)

    # -- internals --------------------------------------------------------

    @staticmethod
    def _row_to_edge(row: sqlite3.Row) -> PlaceEdge:
        guide = (json.loads(row["guide_path_json"])
                 if row["guide_path_json"] else None)
        return PlaceEdge(
            from_place=row["from_place"],
            to_place=row["to_place"],
            from_name=row["from_name"],
            to_name=row["to_name"],
            traversals=int(row["traversals"]),
            successes=int(row["successes"]),
            last_attempt_at=row["last_attempt_at"],
            last_success_at=row["last_success_at"],
            median_duration_s=row["median_duration_s"],
            guide_path=guide,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def _row_to_place(self, row: sqlite3.Row) -> Place:
        try:
            suggested = json.loads(row["suggested_facts_json"] or "[]")
        except (TypeError, ValueError):
            suggested = []
        if not isinstance(suggested, list):
            suggested = []
        return Place(
            id=row["id"],
            name=row["name"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            last_seen_at=row["last_seen_at"],
            status=row["status"] or "confirmed",
            source=row["source"] or "operator",
            confirmed_at=row["confirmed_at"],
            seen_count=int(row["seen_count"] or 1),
            best_match_id=row["best_match_id"],
            best_match_score=row["best_match_score"],
            suggested_name=row["suggested_name"],
            suggested_facts=[str(v) for v in suggested if str(v).strip()],
            suggested_at=row["suggested_at"],
            name_source=row["name_source"],
        )


def uuid_like(name: str) -> str:
    """A stable, human-readable id derived from the name + a short random
    suffix, so ids survive renames without colliding."""
    import re
    import uuid

    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "place"
    return f"{slug[:40]}-{uuid.uuid4().hex[:6]}"
