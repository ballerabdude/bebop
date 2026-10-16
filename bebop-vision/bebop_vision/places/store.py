"""SQLite-backed semantic place memory for the Places service.

A *place* is a named location the robot has been taught. It carries:

  - a name (operator-set, unique) and timestamps,
  - one or more keyframe JPEGs (the view at teach time),
  - DINOv3 descriptors (L2-normalised float32) for appearance matching,
  - free-text facts, each attributed to its source ("operator" or "vision"),
  - an optional odom pose recorded at teach time (kept for a future
    navigation phase; unused today).

Deliberately dependency-light: stdlib sqlite3 plus numpy for the descriptor
vectors. This module knows nothing about the camera or the LLM, so it is
fully unit-testable offline.

Localisation here is *appearance-based place recognition*: a query
descriptor is matched against the stored ones by cosine similarity, and the
best place wins only if it clears a similarity margin over the runner-up.
"""
from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_SCHEMA = """
CREATE TABLE IF NOT EXISTS places (
    id           TEXT PRIMARY KEY,
    name         TEXT NOT NULL COLLATE NOCASE UNIQUE,
    created_at   REAL NOT NULL,
    updated_at   REAL NOT NULL,
    last_seen_at REAL,
    odom_x       REAL,
    odom_y       REAL,
    odom_theta   REAL
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
CREATE INDEX IF NOT EXISTS idx_desc_place  ON descriptors(place_id);
CREATE INDEX IF NOT EXISTS idx_facts_place ON facts(place_id);
CREATE INDEX IF NOT EXISTS idx_kf_place    ON keyframes(place_id);
CREATE INDEX IF NOT EXISTS idx_tags_place  ON tags(place_id);
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


def _clean_name(name: str) -> str:
    name = " ".join(str(name).split())
    if not name:
        raise ValueError("place name must be non-empty")
    if len(name) > 64:
        raise ValueError("place name must be <= 64 characters")
    return name


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
        self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- places -----------------------------------------------------------

    def create_place(
        self,
        name: str,
        *,
        odom: tuple[float, float, float] | None = None,
    ) -> Place:
        name = _clean_name(name)
        pid = uuid_like(name)
        ts = _now()
        ox, oy, ot = (odom if odom is not None else (None, None, None))
        with self._lock:
            try:
                self._conn.execute(
                    "INSERT INTO places (id, name, created_at, updated_at,"
                    " odom_x, odom_y, odom_theta) VALUES (?,?,?,?,?,?,?)",
                    (pid, name, ts, ts, ox, oy, ot),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"a place named {name!r} already exists") from exc
            self._conn.commit()
        return self.get_place(pid)

    def list_places(self) -> list[Place]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id FROM places ORDER BY name COLLATE NOCASE"
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
                "UPDATE places SET last_seen_at = ? WHERE id = ?", (_now(), place.id)
            )
            self._conn.commit()

    def count(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM places").fetchone()[0])

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

    def match(self, vec: np.ndarray, *, k: int = 3) -> list[Match]:
        """Rank places by best cosine similarity to `vec`.

        Descriptors are stored L2-normalised, so the score is a dot product.
        Returns up to `k` places, best first.
        """
        q = np.asarray(vec, dtype=np.float32).reshape(-1)
        if q.size == 0:
            return []
        q = q / (np.linalg.norm(q) + 1e-12)
        with self._lock:
            rows = self._conn.execute(
                "SELECT place_id, dim, vec FROM descriptors"
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

    def _row_to_place(self, row: sqlite3.Row) -> Place:
        if row["odom_x"] is None:
            odom = None
        else:
            odom = (row["odom_x"], row["odom_y"], row["odom_theta"])
        return Place(
            id=row["id"],
            name=row["name"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            last_seen_at=row["last_seen_at"],
            odom=odom,
        )


def uuid_like(name: str) -> str:
    """A stable, human-readable id derived from the name + a short random
    suffix, so ids survive renames without colliding."""
    import re
    import uuid

    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "place"
    return f"{slug[:40]}-{uuid.uuid4().hex[:6]}"
