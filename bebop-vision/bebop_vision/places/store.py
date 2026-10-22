"""SQLite-backed semantic place memory for the Places service.

A *place* is a named location the robot has been taught. It carries:

  - a name (operator-set, unique) and timestamps,
  - one or more keyframe JPEGs (the view at teach time),
  - DINOv3 descriptors (L2-normalised float32) for appearance matching,
  - free-text facts, each attributed to its source ("operator" or "vision"),
  - an optional odom pose recorded at teach time (legacy capture metadata;
    **never** used as a map coordinate system),
  - at most one accumulated local submap per place (gravity-aligned patches
    merged within one vision-session frame, anchored at the first capture),
  - route edges between places with traversal/success counters.

Deliberately dependency-light: stdlib sqlite3 plus numpy for the descriptor
vectors. This module knows nothing about the camera or the LLM, so it is
fully unit-testable offline.

Localisation here is *appearance-based place recognition*: a query
descriptor is matched against the stored ones by cosine similarity, and the
best place wins only if it clears a similarity margin over the runner-up.

The single global metric map is retired as a product: each place owns a
local patch, and places are linked by *route knowledge* (edges) rather than
by a global transform.  Submap coordinates come only from the structure
snapshot's gravity-aligned visual ``map`` frame; wheel odometry is not read.
"""
from __future__ import annotations

import json
import math
import sqlite3
import shutil
import statistics
import uuid
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .submap import (ParsedSubmap, merge_submaps, parse_guide_path,
                     parse_structure_snapshot, rle_true_count)

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
CREATE TABLE IF NOT EXISTS tags (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    place_id   TEXT NOT NULL REFERENCES places(id) ON DELETE CASCADE,
    label      TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 0.0,
    source     TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS place_maps (
    place_id        TEXT PRIMARY KEY REFERENCES places(id) ON DELETE CASCADE,
    anchor_x        REAL NOT NULL,
    anchor_y        REAL NOT NULL,
    anchor_yaw_deg  REAL NOT NULL,
    origin_x        REAL NOT NULL,
    origin_y        REAL NOT NULL,
    resolution_m    REAL NOT NULL,
    width           INTEGER NOT NULL,
    height          INTEGER NOT NULL,
    occupied_rle    TEXT NOT NULL,
    free_rle        TEXT NOT NULL,
    walls_json      TEXT NOT NULL,
    gravity_json    TEXT NOT NULL,
    camera_height_m REAL NOT NULL,
    revision        INTEGER NOT NULL,
    created_at      REAL NOT NULL,
    updated_at      REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS routes (
    id              TEXT PRIMARY KEY,
    from_place      TEXT NOT NULL REFERENCES places(id) ON DELETE CASCADE,
    to_place        TEXT NOT NULL REFERENCES places(id) ON DELETE CASCADE,
    label           TEXT,
    status          TEXT NOT NULL DEFAULT 'taught',
    length_m        REAL NOT NULL,
    duration_s      REAL NOT NULL,
    node_count      INTEGER NOT NULL,
    heading_source  TEXT,
    created_at      REAL NOT NULL,
    updated_at      REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS route_nodes (
    route_id        TEXT NOT NULL REFERENCES routes(id) ON DELETE CASCADE,
    seq             INTEGER NOT NULL,
    x               REAL NOT NULL,
    y               REAL NOT NULL,
    theta           REAL NOT NULL,
    dist_m          REAL NOT NULL,
    t_s             REAL NOT NULL,
    keyframes_json  TEXT NOT NULL,
    vec             BLOB,
    dim             INTEGER,
    vec_role        TEXT,
    created_at      REAL NOT NULL,
    PRIMARY KEY (route_id, seq)
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
CREATE INDEX IF NOT EXISTS idx_tags_place  ON tags(place_id);
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
class Tag:
    label: str
    confidence: float
    source: str
    created_at: float


@dataclass
class PlaceMap:
    """One place's accumulated local submap.

    The grid, walls and anchor all live in the first capture session's
    gravity-aligned ``map`` frame.  ``anchor_*`` is the capture pose pinned at
    the **first** capture; re-captures only add evidence to the raster, so the
    anchor never moves and the accumulated geometry stays coherent.
    Anchor-relative coordinates are derived, not stored.
    """

    place_id: str
    anchor_x: float
    anchor_y: float
    anchor_yaw_deg: float
    origin_x: float
    origin_y: float
    resolution_m: float
    width: int
    height: int
    occupied_rle: str
    free_rle: str
    walls: list[list[float]]
    gravity: dict
    camera_height_m: float
    revision: int
    created_at: float
    updated_at: float

    def to_dict(self) -> dict:
        return {
            "place_id": self.place_id,
            "revision": self.revision,
            "frame": "capture_session_map",
            "anchor": {
                "x": self.anchor_x,
                "y": self.anchor_y,
                "yaw_deg": self.anchor_yaw_deg,
            },
            "grid": {
                "origin": [self.origin_x, self.origin_y],
                "resolution_m": self.resolution_m,
                "width": self.width,
                "height": self.height,
                "encoding": "rle_base64",
                "occupied_rle": self.occupied_rle,
                "free_rle": self.free_rle,
                "occupied_count": rle_true_count(self.occupied_rle),
                "free_count": rle_true_count(self.free_rle),
            },
            "walls": self.walls,
            "gravity": self.gravity,
            "camera_height_m": self.camera_height_m,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass
class RouteNode:
    """One taught waypoint: pose in the route frame (origin + heading at
    the start of the teach drive), distance along the route, keyframes."""

    seq: int
    x: float
    y: float
    theta: float
    dist_m: float
    t_s: float
    keyframes: dict            # role -> jpeg path
    has_descriptor: bool
    vec_role: str | None

    def to_dict(self, *, root: Path | None = None) -> dict:
        return {
            "seq": self.seq,
            "x": round(self.x, 4), "y": round(self.y, 4),
            "theta": round(self.theta, 4),
            "dist_m": round(self.dist_m, 3), "t_s": round(self.t_s, 2),
            "keyframes": {role: (str(Path(p).relative_to(root))
                                 if root and str(p).startswith(str(root))
                                 else str(p))
                          for role, p in self.keyframes.items()},
            "has_descriptor": self.has_descriptor,
            "vec_role": self.vec_role,
        }


@dataclass
class Route:
    """A taught trail from one place to another."""

    id: str
    from_place: str
    to_place: str
    from_name: str
    to_name: str
    label: str | None
    status: str
    length_m: float
    duration_s: float
    node_count: int
    heading_source: str | None
    created_at: float
    updated_at: float
    nodes: list[RouteNode] = field(default_factory=list)

    def to_dict(self, *, root: Path | None = None, nodes: bool = True) -> dict:
        out = {
            "id": self.id,
            "from_place": self.from_place, "to_place": self.to_place,
            "from_name": self.from_name, "to_name": self.to_name,
            "label": self.label, "status": self.status,
            "length_m": round(self.length_m, 3),
            "duration_s": round(self.duration_s, 2),
            "node_count": self.node_count,
            "heading_source": self.heading_source,
            "created_at": self.created_at, "updated_at": self.updated_at,
        }
        if nodes:
            out["nodes"] = [n.to_dict(root=root) for n in self.nodes]
        return out


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
    odom: tuple[float, float, float] | None = None
    facts: list[Fact] = field(default_factory=list)
    keyframes: list[Keyframe] = field(default_factory=list)
    tags: list[Tag] = field(default_factory=list)
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
            "odom": list(self.odom) if self.odom is not None else None,
            "facts": [
                {"text": f.text, "source": f.source, "created_at": f.created_at}
                for f in self.facts
            ],
            "tags": [
                {"label": t.label, "confidence": t.confidence, "source": t.source}
                for t in self.tags
            ],
            "keyframes": kf,
        }


@dataclass
class Match:
    """One place's best score for a query descriptor."""

    place: Place
    score: float

    def to_dict(self, *, root: Path | None = None) -> dict:
        return {"place": self.place.to_dict(root=root), "score": self.score}


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
        odom: tuple[float, float, float] | None = None,
        status: str = "confirmed",
        source: str = "operator",
    ) -> Place:
        name = _clean_name(name)
        status = _clean_status(status)
        pid = uuid_like(name)
        ts = _now()
        ox, oy, ot = (odom if odom is not None else (None, None, None))
        confirmed_at = ts if status == "confirmed" else None
        with self._lock:
            try:
                self._conn.execute(
                    "INSERT INTO places (id, name, created_at, updated_at,"
                    " odom_x, odom_y, odom_theta, status, source,"
                    " confirmed_at, name_source)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (pid, name, ts, ts, ox, oy, ot, status, source,
                     confirmed_at,
                     "operator" if status == "confirmed" else None),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"a place named {name!r} already exists") from exc
            self._conn.commit()
        return self.get_place(pid)

    def create_pending_place(
        self,
        *,
        odom: tuple[float, float, float] | None = None,
        source: str = "scout",
    ) -> Place:
        """Create a hidden candidate place for operator review.

        The name is a unique placeholder; ``display_name`` hides it.  The
        place is not recognized until ``promote_place`` is called.
        """
        import uuid

        name = f"{PENDING_NAME_PREFIX}{uuid.uuid4().hex[:12]}"
        return self.create_place(name, odom=odom, status="pending",
                                 source=source)

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
            place.tags = [
                Tag(r["label"], r["confidence"], r["source"], r["created_at"])
                for r in self._conn.execute(
                    "SELECT label, confidence, source, created_at FROM tags"
                    " WHERE place_id = ? ORDER BY confidence DESC", (place.id,),
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
        self.add_facts_unique(place.id, facts or [], source="router")
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

    # -- local maps -------------------------------------------------------

    def upsert_place_map(
        self,
        id_or_name: str,
        snapshot: dict,
        *,
        pose=None,
        replace: bool = False,
    ) -> tuple[PlaceMap, bool]:
        """Accumulate a structure snapshot into a place's local submap.

        ``snapshot`` is the :9096 ``/structure`` payload (or a wrapper with
        ``structure`` + ``pose``).  The anchor is the capture pose from that
        payload, pinned at the **first** capture; wheel odometry is never
        consulted.

        A re-capture **merges** by default: the stored and incoming
        occupied/free cell sets are unioned, walls are unioned with
        near-duplicates dropped, and the result is rebinned into a union
        bounding box at the stored resolution.  Cells are placed by their
        capture-frame centre (``origin + (i + 0.5) * resolution_m``), so
        captures with different ``origin``/``width``/``height`` add evidence
        instead of overwriting each other and counts never shrink.

        Pass ``replace=True`` to deliberately start the place over: the row is
        reset to exactly this patch, anchor included.  ``revision`` increments
        on every capture either way; ``created`` only reports whether the row
        existed before.  Returns ``(map, created)``.
        """
        if not isinstance(replace, bool):
            raise ValueError("replace must be a boolean")
        place = self.get_place(id_or_name)
        if place is None:
            raise KeyError(id_or_name)
        parsed = parse_structure_snapshot(snapshot, pose=pose)
        ts = _now()
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM place_maps WHERE place_id = ?",
                (place.id,),
            ).fetchone()
            if row is None:
                created = True
                revision = 1
                created_at = ts
                geometry = parsed
                anchor_x = parsed.anchor_x
                anchor_y = parsed.anchor_y
                anchor_yaw_deg = parsed.anchor_yaw_deg
                gravity_json = json.dumps(parsed.gravity,
                                          separators=(",", ":"))
                camera_height_m = parsed.camera_height_m
            else:
                created = False
                revision = int(row["revision"]) + 1
                created_at = float(row["created_at"])
                if replace:
                    geometry = parsed
                    anchor_x = parsed.anchor_x
                    anchor_y = parsed.anchor_y
                    anchor_yaw_deg = parsed.anchor_yaw_deg
                    gravity_json = json.dumps(parsed.gravity,
                                              separators=(",", ":"))
                    camera_height_m = parsed.camera_height_m
                else:
                    # The anchor and orientation are pinned to the first
                    # capture; only cell/wall evidence accumulates.
                    anchor_x = float(row["anchor_x"])
                    anchor_y = float(row["anchor_y"])
                    anchor_yaw_deg = float(row["anchor_yaw_deg"])
                    gravity_json = row["gravity_json"]
                    camera_height_m = float(row["camera_height_m"])
                    stored = ParsedSubmap(
                        anchor_x=anchor_x,
                        anchor_y=anchor_y,
                        anchor_yaw_deg=anchor_yaw_deg,
                        origin_x=float(row["origin_x"]),
                        origin_y=float(row["origin_y"]),
                        resolution_m=float(row["resolution_m"]),
                        width=int(row["width"]),
                        height=int(row["height"]),
                        occupied_rle=row["occupied_rle"],
                        free_rle=row["free_rle"],
                        walls=json.loads(row["walls_json"] or "[]"),
                        gravity=json.loads(gravity_json or "{}"),
                        camera_height_m=camera_height_m,
                    )
                    geometry = merge_submaps(stored, parsed)
            walls_json = json.dumps(geometry.walls, separators=(",", ":"))
            self._conn.execute(
                "INSERT INTO place_maps (place_id, anchor_x, anchor_y,"
                " anchor_yaw_deg, origin_x, origin_y, resolution_m, width,"
                " height, occupied_rle, free_rle, walls_json, gravity_json,"
                " camera_height_m, revision, created_at, updated_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
                " ON CONFLICT(place_id) DO UPDATE SET"
                " anchor_x=excluded.anchor_x, anchor_y=excluded.anchor_y,"
                " anchor_yaw_deg=excluded.anchor_yaw_deg,"
                " origin_x=excluded.origin_x, origin_y=excluded.origin_y,"
                " resolution_m=excluded.resolution_m,"
                " width=excluded.width, height=excluded.height,"
                " occupied_rle=excluded.occupied_rle,"
                " free_rle=excluded.free_rle,"
                " walls_json=excluded.walls_json,"
                " gravity_json=excluded.gravity_json,"
                " camera_height_m=excluded.camera_height_m,"
                " revision=excluded.revision, updated_at=excluded.updated_at",
                (
                    place.id, anchor_x, anchor_y, anchor_yaw_deg,
                    geometry.origin_x, geometry.origin_y,
                    geometry.resolution_m, geometry.width, geometry.height,
                    geometry.occupied_rle, geometry.free_rle, walls_json,
                    gravity_json, camera_height_m, revision, created_at, ts,
                ),
            )
            self._conn.execute(
                "UPDATE places SET updated_at = ? WHERE id = ?", (ts, place.id)
            )
            self._conn.commit()
        result = self.get_place_map(place.id)
        if result is None:  # pragma: no cover - the upsert just wrote it
            raise RuntimeError("place map vanished after upsert")
        return result, created

    def get_place_map(self, id_or_name: str) -> PlaceMap | None:
        place = self.get_place(id_or_name)
        if place is None:
            return None
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM place_maps WHERE place_id = ?", (place.id,)
            ).fetchone()
            if row is None:
                return None
            return self._row_to_place_map(row)

    def delete_place_map(self, id_or_name: str) -> bool:
        place = self.get_place(id_or_name)
        if place is None:
            return False
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM place_maps WHERE place_id = ?", (place.id,)
            )
            self._conn.commit()
            return cur.rowcount > 0

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

    # -- routes (taught trails) ---------------------------------------------

    def create_route(
        self,
        from_place: str,
        to_place: str,
        nodes: list[dict],
        *,
        length_m: float,
        duration_s: float,
        label: str | None = None,
        heading_source: str | None = None,
        record_edge: bool = True,
    ) -> Route:
        """Persist a taught trail and (by default) record the A->B edge.

        Each node dict: ``{x, y, theta, dist_m, t_s, keyframes: {role:
        jpeg bytes}, vec: array | None, vec_role: str | None}``.  Keyframe
        JPEGs land under ``routes/<route_id>/``.  Raises KeyError for an
        unknown place and ValueError for fewer than two nodes.
        """
        src = self.get_place(from_place)
        if src is None:
            raise KeyError(from_place)
        dst = self.get_place(to_place)
        if dst is None:
            raise KeyError(to_place)
        if len(nodes) < 2:
            raise ValueError("a route needs at least two nodes")
        ts = _now()
        route_id = f"route-{uuid.uuid4().hex[:10]}"
        rdir = self.root / "routes" / route_id
        rdir.mkdir(parents=True, exist_ok=True)
        rows = []
        for seq, n in enumerate(nodes):
            paths = {}
            for role, jpeg in (n.get("keyframes") or {}).items():
                if not jpeg:
                    continue
                path = rdir / f"{seq:04d}_{role}.jpg"
                path.write_bytes(bytes(jpeg))
                paths[str(role)] = str(path)
            vec = n.get("vec")
            blob = dim = None
            if vec is not None:
                arr = np.asarray(vec, dtype=np.float32).reshape(-1)
                if arr.size:
                    arr = arr / (np.linalg.norm(arr) + 1e-12)
                    blob, dim = arr.tobytes(), int(arr.size)
            rows.append((route_id, seq, float(n["x"]), float(n["y"]),
                         float(n["theta"]), float(n.get("dist_m", 0.0)),
                         float(n.get("t_s", 0.0)),
                         json.dumps(paths, separators=(",", ":")),
                         blob, dim, n.get("vec_role"), ts))
        with self._lock:
            self._conn.execute(
                "INSERT INTO routes (id, from_place, to_place, label, status,"
                " length_m, duration_s, node_count, heading_source,"
                " created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (route_id, src.id, dst.id, label, "taught", float(length_m),
                 float(duration_s), len(rows), heading_source, ts, ts))
            self._conn.executemany(
                "INSERT INTO route_nodes (route_id, seq, x, y, theta, dist_m,"
                " t_s, keyframes_json, vec, dim, vec_role, created_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows)
            self._conn.commit()
        if record_edge:
            guide = [[float(n["x"]), float(n["y"])] for n in nodes]
            self.record_traversal(src.id, dst.id, success=True,
                                  duration_s=duration_s, guide_path=guide)
        route = self.get_route(route_id)
        if route is None:  # pragma: no cover
            raise RuntimeError("route vanished after insert")
        return route

    def list_routes(self, *, from_place: str | None = None,
                    to_place: str | None = None) -> list[Route]:
        clauses, args = [], []
        if from_place:
            src = self.get_place(from_place)
            if src is None:
                return []
            clauses.append("r.from_place = ?"); args.append(src.id)
        if to_place:
            dst = self.get_place(to_place)
            if dst is None:
                return []
            clauses.append("r.to_place = ?"); args.append(dst.id)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        with self._lock:
            rows = self._conn.execute(
                "SELECT r.*, a.name AS from_name, b.name AS to_name"
                " FROM routes r JOIN places a ON a.id = r.from_place"
                " JOIN places b ON b.id = r.to_place" + where +
                " ORDER BY r.created_at DESC", args).fetchall()
        return [self._row_to_route(r, []) for r in rows]

    def get_route(self, route_id: str) -> Route | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT r.*, a.name AS from_name, b.name AS to_name"
                " FROM routes r JOIN places a ON a.id = r.from_place"
                " JOIN places b ON b.id = r.to_place WHERE r.id = ?",
                (route_id,)).fetchone()
            if row is None:
                return None
            nrows = self._conn.execute(
                "SELECT seq, x, y, theta, dist_m, t_s, keyframes_json,"
                " dim, vec_role FROM route_nodes WHERE route_id = ?"
                " ORDER BY seq", (route_id,)).fetchall()
        nodes = [RouteNode(int(n["seq"]), float(n["x"]), float(n["y"]),
                           float(n["theta"]), float(n["dist_m"]),
                           float(n["t_s"]), json.loads(n["keyframes_json"]),
                           n["dim"] is not None, n["vec_role"])
                 for n in nrows]
        return self._row_to_route(row, nodes)

    def route_descriptors(self, route_id: str):
        """(seqs[int], vecs[N, dim] float32 L2-normalised) for the nodes
        that carry a descriptor; empty arrays when none."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT seq, vec, dim FROM route_nodes WHERE route_id = ?"
                " AND vec IS NOT NULL ORDER BY seq", (route_id,)).fetchall()
        if not rows:
            return np.zeros(0, dtype=np.int64), np.zeros((0, 0), np.float32)
        seqs = np.array([int(r["seq"]) for r in rows], dtype=np.int64)
        vecs = np.stack([np.frombuffer(r["vec"], dtype=np.float32)
                         for r in rows])
        return seqs, vecs

    def route_keyframe_path(self, route_id: str, seq: int,
                            role: str) -> Path | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT keyframes_json FROM route_nodes WHERE route_id = ?"
                " AND seq = ?", (route_id, int(seq))).fetchone()
        if row is None:
            return None
        path = json.loads(row["keyframes_json"]).get(role)
        return Path(path) if path else None

    def delete_route(self, route_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM routes WHERE id = ?",
                                     (route_id,))
            self._conn.commit()
            deleted = cur.rowcount > 0
        if deleted:
            shutil.rmtree(self.root / "routes" / route_id, ignore_errors=True)
        return deleted

    def plan_trip(self, from_place: str, to_place: str, *,
                  min_success_rate: float = 0.0) -> dict:
        """Legs to drive from one place to another over taught routes.

        Returns {"from", "to", "legs": [...], "cost_s", "reachable"}; raises
        KeyError for an unknown place.
        """
        from .planner import plan

        src = self.get_place(from_place)
        if src is None:
            raise KeyError(from_place)
        dst = self.get_place(to_place)
        if dst is None:
            raise KeyError(to_place)
        routes = {r.id: {"length_m": r.length_m, "created_at": r.created_at,
                         "from_name": r.from_name, "to_name": r.to_name}
                  for r in self.list_routes()}
        graph = self.graph()
        names = {n["id"]: n["name"] for n in graph["nodes"]}
        legs = plan(graph, routes, src.id, dst.id,
                    min_success_rate=min_success_rate)
        for leg in legs:
            leg["from_name"] = names.get(leg["from"], leg["from"])
            leg["to_name"] = names.get(leg["to"], leg["to"])
            leg["length_m"] = round(routes[leg["route_id"]]["length_m"], 3)
        return {
            "from": src.id, "from_name": src.name,
            "to": dst.id, "to_name": dst.name,
            "legs": legs,
            "cost_s": round(sum(l["cost_s"] for l in legs), 1),
            "reachable": bool(legs) or src.id == dst.id,
        }

    def _row_to_route(self, row: sqlite3.Row, nodes: list) -> Route:
        return Route(
            id=str(row["id"]), from_place=str(row["from_place"]),
            to_place=str(row["to_place"]), from_name=str(row["from_name"]),
            to_name=str(row["to_name"]), label=row["label"],
            status=str(row["status"]), length_m=float(row["length_m"]),
            duration_s=float(row["duration_s"]),
            node_count=int(row["node_count"]),
            heading_source=row["heading_source"],
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]), nodes=nodes)

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
        """JSON-ready place graph: nodes (places) and route edges."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT p.id, p.name, p.created_at, p.last_seen_at,"
                " m.revision AS map_revision"
                " FROM places p"
                " LEFT JOIN place_maps m ON m.place_id = p.id"
                " WHERE p.status = 'confirmed'"
                " ORDER BY p.name COLLATE NOCASE"
            ).fetchall()
            nodes = [
                {
                    "id": r["id"],
                    "name": r["name"],
                    "created_at": r["created_at"],
                    "last_seen_at": r["last_seen_at"],
                    "has_map": r["map_revision"] is not None,
                    "map_revision": r["map_revision"],
                }
                for r in rows
            ]
        edges = [e.to_dict() for e in self.list_edges()]
        with self._lock:
            rrows = self._conn.execute(
                "SELECT id, from_place, to_place FROM routes").fetchall()
        by_pair: dict = {}
        for r in rrows:
            by_pair.setdefault((r["from_place"], r["to_place"]), []).append(
                str(r["id"]))
        for e in edges:
            e["route_ids"] = by_pair.get((e["from"], e["to"]), [])
        return {
            "nodes": nodes,
            "edges": edges,
            "node_count": len(nodes),
            "edge_count": len(edges),
        }

    # -- facts / tags / keyframes ----------------------------------------

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
        source: str = "router",
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

    def add_tag(
        self,
        id_or_name: str,
        label: str,
        *,
        confidence: float = 0.0,
        source: str = "vision",
    ) -> Tag:
        place = self.get_place(id_or_name)
        if place is None:
            raise KeyError(id_or_name)
        label = " ".join(str(label).split())
        if not label:
            raise ValueError("tag label must be non-empty")
        ts = _now()
        with self._lock:
            self._conn.execute(
                "INSERT INTO tags (place_id, label, confidence, source, created_at)"
                " VALUES (?,?,?,?,?)",
                (place.id, label, float(confidence), source, ts),
            )
            self._conn.commit()
        return Tag(label, float(confidence), source, ts)

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
    def _row_to_place_map(row: sqlite3.Row) -> PlaceMap:
        return PlaceMap(
            place_id=row["place_id"],
            anchor_x=row["anchor_x"],
            anchor_y=row["anchor_y"],
            anchor_yaw_deg=row["anchor_yaw_deg"],
            origin_x=row["origin_x"],
            origin_y=row["origin_y"],
            resolution_m=row["resolution_m"],
            width=int(row["width"]),
            height=int(row["height"]),
            occupied_rle=row["occupied_rle"],
            free_rle=row["free_rle"],
            walls=json.loads(row["walls_json"] or "[]"),
            gravity=json.loads(row["gravity_json"] or "{}"),
            camera_height_m=row["camera_height_m"],
            revision=int(row["revision"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

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
        if row["odom_x"] is None:
            odom = None
        else:
            odom = (row["odom_x"], row["odom_y"], row["odom_theta"])
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
            odom=odom,
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
