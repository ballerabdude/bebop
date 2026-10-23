"""Pending-place capture and store lifecycle (no torch, camera, or VLM)."""

import sqlite3
from pathlib import Path

import cv2
import numpy as np
import pytest

from bebop_vision.places import (
    PlaceScout,
    PlaceStore,
    ScoutConfig,
    StaticCapture,
    View,
)


def _jpeg(value: int) -> bytes:
    ok, buf = cv2.imencode(".jpg", np.full((16, 16, 3), value, np.uint8))
    assert ok
    return buf.tobytes()


class FakeEncoder:
    dim = 4

    def __init__(self, table=None):
        self.table = dict(table or {})

    def encode_jpeg(self, jpeg):
        if jpeg in self.table:
            vec = np.asarray(self.table[jpeg], np.float32)
        else:
            vec = np.asarray([float(sum(jpeg) % 997), 1.0, 0.0, 0.0],
                             np.float32)
        return vec / (np.linalg.norm(vec) + 1e-12)


def _view(jpeg: bytes, stamp: int = 1) -> View:
    return View("far", jpeg, stamp)


def _descriptor(vec):
    vec = np.asarray(vec, np.float32)
    return vec / (np.linalg.norm(vec) + 1e-12)


@pytest.fixture()
def store(tmp_path):
    value = PlaceStore(tmp_path / "places")
    yield value
    value.close()


# --- store lifecycle -------------------------------------------------------


def test_pending_place_hides_placeholder_and_promotes(store):
    pending = store.create_pending_place()
    assert pending.status == "pending"
    assert pending.display_name == "Unnamed place"
    assert pending.name.startswith("__pending__")
    assert store.count(status="pending") == 1
    assert store.count(status="confirmed") == 0
    assert [p.id for p in store.list_pending()] == [pending.id]
    assert [p.id for p in store.list_places(status="confirmed")] == []

    store.set_pending_suggestion(pending.id, "Workshop",
                                 ["a workbench", "tools on the wall"])
    pending = store.get_place(pending.id)
    assert pending.display_name == "Workshop"
    assert pending.suggested_facts == ["a workbench", "tools on the wall"]

    promoted = store.promote_place(pending.id, "Workshop",
                                   facts=["a workbench"],
                                   name_source="llm")
    assert promoted.status == "confirmed"
    assert promoted.name == "Workshop"
    assert promoted.name_source == "llm"
    assert promoted.confirmed_at is not None
    assert store.count(status="pending") == 0
    assert store.count(status="confirmed") == 1
    assert [f.text for f in promoted.facts] == ["a workbench"]


def test_match_filters_pending_and_confirmed(store):
    confirmed = store.create_place("Office")
    store.add_descriptor(confirmed.id, _descriptor([1, 0, 0, 0]))
    pending = store.create_pending_place()
    store.add_descriptor(pending.id, _descriptor([0, 1, 0, 0]))

    only_confirmed = store.match(_descriptor([0, 1, 0, 0]),
                                 status="confirmed")
    assert [m.place.id for m in only_confirmed] == [confirmed.id]
    assert only_confirmed[0].score == pytest.approx(0.0)
    only_pending = store.match(_descriptor([0, 1, 0, 0]), status="pending")
    assert [m.place.id for m in only_pending] == [pending.id]
    assert store.match(_descriptor([1, 0, 0, 0]), status="confirmed")[0].place.id \
        == confirmed.id


def test_graph_excludes_pending(store):
    confirmed = store.create_place("Office")
    store.create_pending_place()
    graph = store.graph()
    assert graph["node_count"] == 1
    assert [node["id"] for node in graph["nodes"]] == [confirmed.id]


def test_merge_pending_into_copies_evidence_and_deletes_source(store):
    target = store.create_place("Office")
    store.add_keyframe(target.id, _jpeg(10), role="far", stamp_us=1)
    store.add_descriptor(target.id, _descriptor([1, 0, 0, 0]))

    source = store.create_pending_place()
    store.add_keyframe(source.id, _jpeg(20), role="far", stamp_us=2)
    store.add_descriptor(source.id, _descriptor([0, 1, 0, 0]))
    store.add_fact(source.id, "a red chair", source="operator")

    outcome = store.merge_pending_into(source.id, target.id)
    assert store.get_place(source.id) is None
    merged = store.get_place(target.id)
    assert len(merged.keyframes) == 2
    assert store.descriptor_count(target.id) == 2
    assert [f.text for f in merged.facts] == ["a red chair"]
    assert outcome["descriptors"] == 1


def test_prune_pending_honours_ttl_and_cap(store):
    old = store.create_pending_place()
    store.note_pending_seen(old.id, seen_at=1000.0)
    recent = store.create_pending_place()
    store.note_pending_seen(recent.id, seen_at=2000.0)
    # TTL removes the old one, cap keeps the newest one only.
    deleted = store.prune_pending(ttl_s=100.0, max_count=1, now=2000.0)
    assert deleted == 1
    assert [p.id for p in store.list_pending()] == [recent.id]


def test_migration_adds_pending_columns_to_old_database(tmp_path):
    root = tmp_path / "places"
    root.mkdir()
    db = sqlite3.connect(root / "places.db")
    db.executescript(
        """
        CREATE TABLE places (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL COLLATE NOCASE UNIQUE,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            last_seen_at REAL,
            odom_x REAL,
            odom_y REAL,
            odom_theta REAL
        );
        INSERT INTO places (id, name, created_at, updated_at)
        VALUES ('legacy-1', 'Legacy', 1.0, 1.0);
        """
    )
    db.commit()
    db.close()

    opened = PlaceStore(root)
    try:
        place = opened.get_place("legacy-1")
        assert place is not None
        assert place.status == "confirmed"
        assert place.name_source is None
        assert place.suggested_name is None
    finally:
        opened.close()


# --- scout -----------------------------------------------------------------


def _scout(store, encoder, *, now_views=None, config=None):
    return PlaceScout(
        store, encoder, StaticCapture(list(now_views or [])),
        config=config or ScoutConfig(min_hits=3, min_dwell_s=2.0,
                                     new_pending_cooldown_s=0.0),
    )


def test_novel_view_becomes_pending_after_stable_sightings(store):
    jpeg = _jpeg(80)
    encoder = FakeEncoder({jpeg: [0, 1, 0, 0]})
    scout = _scout(store, encoder)
    view = _view(jpeg, 1)

    assert scout.tick([view], now=0.0)["action"] == "candidate"
    assert scout.tick([view], now=1.1)["action"] == "candidate"
    result = scout.tick([view], now=2.2)
    assert result["action"] == "created"
    pending = store.list_pending()
    assert len(pending) == 1
    assert pending[0].status == "pending"
    assert len(pending[0].keyframes) == 1
    assert pending[0].seen_count >= 1


def test_transient_view_does_not_create_pending(store):
    jpeg = _jpeg(80)
    scout = _scout(store, FakeEncoder({jpeg: [0, 1, 0, 0]}))
    assert scout.tick([_view(jpeg)], now=0.0)["action"] == "candidate"
    assert store.count(status="pending") == 0


def test_confirmed_match_is_known_and_not_pending(store):
    jpeg = _jpeg(20)
    place = store.create_place("Office")
    store.add_descriptor(place.id, _descriptor([1, 0, 0, 0]))
    scout = _scout(store, FakeEncoder({jpeg: [1, 0, 0, 0]}))
    result = scout.tick([_view(jpeg)], now=0.0)
    assert result["action"] == "known"
    assert result["place_id"] == place.id
    assert store.count(status="pending") == 0


def test_pending_match_adds_diverse_evidence(store):
    jpeg = _jpeg(80)
    other = _jpeg(120)
    encoder = FakeEncoder({jpeg: [0, 1, 0, 0], other: [0.5, 0.87, 0, 0]})
    scout = _scout(store, encoder)
    # Create a pending place directly so this test is about evidence updates.
    place = store.create_pending_place()
    store.add_keyframe(place.id, jpeg, role="far", stamp_us=1)
    store.add_descriptor(place.id, _descriptor([0, 1, 0, 0]))

    result = scout.tick([_view(other, 2)], now=0.0)
    assert result["action"] == "pending"
    assert len(store.get_place(place.id).keyframes) == 2
    assert store.descriptor_count(place.id) == 2

    # The exact same view is not added twice.
    scout.tick([_view(other, 3)], now=1.0)
    assert len(store.get_place(place.id).keyframes) == 2


def test_known_match_calls_on_known(store):
    jpeg = _jpeg(20)
    place = store.create_place("Office")
    store.add_descriptor(place.id, _descriptor([1, 0, 0, 0]))
    seen = []
    scout = PlaceScout(
        store, FakeEncoder({jpeg: [1, 0, 0, 0]}), StaticCapture([]),
        config=ScoutConfig(min_hits=3, min_dwell_s=2.0,
                           new_pending_cooldown_s=0.0),
        on_known=lambda pid, name, score, margin: seen.append((pid, name)))
    assert scout.tick([_view(jpeg)], now=0.0)["action"] == "known"
    assert seen == [(place.id, "Office")]

    def boom(*a):
        raise RuntimeError("callback failed")

    scout.on_known = boom
    assert scout.tick([_view(jpeg)], now=1.0)["action"] == "known"
    assert "on_known" in scout._last_error
