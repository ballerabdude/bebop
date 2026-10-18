"""Operator import of pending places: suggestions, confirmation, merge."""

import json
import urllib.request

import cv2
import numpy as np
import pytest

from bebop_vision.places import (
    PlaceImporter,
    PlaceStore,
    PlacesService,
    StaticCapture,
    View,
)


def _jpeg(value: int) -> bytes:
    ok, buf = cv2.imencode(".jpg", np.full((16, 16, 3), value, np.uint8))
    assert ok
    return buf.tobytes()


def _descriptor(vec):
    vec = np.asarray(vec, np.float32)
    return vec / (np.linalg.norm(vec) + 1e-12)


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


class FakeVLM:
    describe = "fake-vlm"

    def __init__(self, reply=None, *, error=None):
        self.reply = reply
        self.error = error
        self.calls = 0

    def complete(self, messages, *, max_tokens=384, temperature=0.0):
        self.calls += 1
        if self.error:
            raise RuntimeError(self.error)
        if isinstance(self.reply, str):
            return self.reply
        return json.dumps(self.reply or {
            "name": "Workshop",
            "facts": ["a workbench", "tools on the wall"],
            "confidence": 0.9,
        })


@pytest.fixture()
def store(tmp_path):
    value = PlaceStore(tmp_path / "places")
    yield value
    value.close()


def _pending(store, *, jpeg_value=80, vec=(1, 0, 0, 0)):
    place = store.create_pending_place()
    store.add_keyframe(place.id, _jpeg(jpeg_value), role="far", stamp_us=1)
    store.add_descriptor(place.id, _descriptor(vec))
    return store.get_place(place.id)


# --- importer --------------------------------------------------------------


def test_suggest_is_cached_and_does_not_confirm(store):
    place = _pending(store)
    vlm = FakeVLM()
    importer = PlaceImporter(store, vlm)

    result = importer.suggest(place.id)
    assert result["suggested_name"] == "Workshop"
    assert result["facts"] == ["a workbench", "tools on the wall"]
    assert result["cached"] is False
    assert vlm.calls == 1

    cached = importer.suggest(place.id)
    assert cached["cached"] is True
    assert vlm.calls == 1
    assert store.get_place(place.id).status == "pending"
    assert store.get_place(place.id).suggested_name == "Workshop"


def test_import_uses_edited_name_without_another_model_call(store):
    place = _pending(store)
    vlm = FakeVLM()
    importer = PlaceImporter(store, vlm)
    importer.suggest(place.id)

    outcome = importer.import_pending(
        [place.id], names={place.id: "My Workshop"})
    assert outcome["imported"] == 1
    assert vlm.calls == 1
    promoted = store.get_place(place.id)
    assert promoted.status == "confirmed"
    assert promoted.name == "My Workshop"
    assert promoted.name_source == "operator"
    assert [f.text for f in promoted.facts] == [
        "a workbench", "tools on the wall"]


def test_import_accepts_suggested_name(store):
    place = _pending(store)
    importer = PlaceImporter(store, FakeVLM())
    importer.suggest(place.id)
    outcome = importer.import_pending([place.id])
    assert outcome["imported"] == 1
    promoted = store.get_place(place.id)
    assert promoted.name == "Workshop"
    assert promoted.name_source == "llm"
    assert store.count(status="pending") == 0


def test_import_without_a_name_stays_pending(store):
    place = _pending(store)
    importer = PlaceImporter(store, FakeVLM())
    outcome = importer.import_pending([place.id])
    assert outcome["failed"] == 1
    assert outcome["results"][0]["error"] == "needs_name"
    assert store.get_place(place.id).status == "pending"


def test_import_name_conflict_stays_pending(store):
    existing = store.create_place("Workshop")
    place = _pending(store)
    importer = PlaceImporter(store, FakeVLM())
    outcome = importer.import_pending(
        [place.id], names={place.id: "Workshop"})
    assert outcome["failed"] == 1
    assert outcome["results"][0]["error"] == "name_conflict"
    assert outcome["results"][0]["conflict"]["id"] == existing.id
    assert store.get_place(place.id).status == "pending"


def test_suggestion_failure_keeps_pending_and_import_can_be_manual(store):
    place = _pending(store)
    vlm = FakeVLM(error="brain offline")
    importer = PlaceImporter(store, vlm)
    result = importer.suggest(place.id)
    assert result["error"] == "vlm_error"
    assert store.get_place(place.id).status == "pending"

    outcome = importer.import_pending(
        [place.id], names={place.id: "Manual"})
    assert outcome["imported"] == 1


def test_no_vlm_configuration_is_reported(store):
    place = _pending(store)
    importer = PlaceImporter(store, None)
    result = importer.suggest(place.id)
    assert result["error"] == "vlm_not_configured"
    outcome = importer.import_pending(
        [place.id], names={place.id: "Manual"})
    assert outcome["imported"] == 1


def test_merge_is_an_explicit_secondary_action(store):
    target = store.create_place("Office")
    store.add_keyframe(target.id, _jpeg(10), role="far", stamp_us=1)
    store.add_descriptor(target.id, _descriptor([1, 0, 0, 0]))
    place = _pending(store, jpeg_value=20, vec=(0, 1, 0, 0))
    importer = PlaceImporter(store, FakeVLM())

    # Importing as new does not merge, even when a target exists.
    outcome = importer.import_pending(
        [place.id], names={place.id: "Corner"})
    assert outcome["imported"] == 1
    assert store.get_place(place.id).status == "confirmed"
    assert store.count(status="pending") == 0

    other = _pending(store, jpeg_value=30, vec=(0, 0, 1, 0))
    merged = importer.import_pending(
        [other.id], merge_into={other.id: target.id})
    assert merged["merged"] == 1
    assert store.get_place(other.id) is None
    assert len(store.get_place(target.id).keyframes) == 2


# --- HTTP contract ---------------------------------------------------------


def _get(port, path):
    with urllib.request.urlopen(
            f"http://127.0.0.1:{port}{path}", timeout=5) as response:
        return response.status, json.loads(response.read())


def _post(port, path, body=None):
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(body or {}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        return response.status, json.loads(response.read())


def test_http_pending_suggest_and_import(tmp_path):
    jpeg = _jpeg(80)
    store = PlaceStore(tmp_path / "places")
    encoder = FakeEncoder({jpeg: [1, 0, 0, 0]})
    service = PlacesService(
        store,
        capture=StaticCapture([View("far", jpeg, 1)]),
        encoder=encoder,
        importer=PlaceImporter(store, FakeVLM()),
        host="127.0.0.1",
        port=0,
        min_score=0.5,
        min_margin=0.1,
    )
    service.start()
    try:
        pending = store.create_pending_place()
        store.add_keyframe(pending.id, jpeg, role="far", stamp_us=2)
        store.add_descriptor(pending.id, _descriptor([1, 0, 0, 0]))

        status, body = _get(service.bound_port, "/places")
        assert body["places"] == []
        assert body["pending_count"] == 1

        status, body = _get(service.bound_port, "/places/pending")
        assert status == 200
        assert body["count"] == 1
        assert body["places"][0]["id"] == pending.id
        assert body["places"][0]["display_name"] == "Unnamed place"

        status, body = _post(
            service.bound_port,
            f"/places/pending/{pending.id}/suggest",
        )
        assert status == 200
        assert body["suggested_name"] == "Workshop"

        status, body = _post(
            service.bound_port,
            "/places/pending/import",
            {"ids": [pending.id], "names": {pending.id: "My Workshop"}},
        )
        assert status == 200
        assert body["imported"] == 1
        confirmed = store.get_place(pending.id)
        assert confirmed.status == "confirmed"
        assert confirmed.name == "My Workshop"

        status, body = _get(service.bound_port, "/places")
        assert [item["id"] for item in body["places"]] == [pending.id]
        assert body["pending_count"] == 0

        status, body = _get(service.bound_port, "/whereami")
        assert body["matched"] is True
        assert body["place"]["name"] == "My Workshop"
    finally:
        service.stop()
        store.close()


def test_whereami_ignores_pending_only_place(tmp_path):
    jpeg = _jpeg(80)
    store = PlaceStore(tmp_path / "places")
    encoder = FakeEncoder({jpeg: [1, 0, 0, 0]})
    service = PlacesService(
        store,
        capture=StaticCapture([View("far", jpeg, 1)]),
        encoder=encoder,
        host="127.0.0.1",
        port=0,
    )
    try:
        pending = store.create_pending_place()
        store.add_keyframe(pending.id, jpeg, role="far", stamp_us=2)
        store.add_descriptor(pending.id, _descriptor([1, 0, 0, 0]))
        result = service.whereami()
        assert result["matched"] is False
        assert result.get("place") is None
    finally:
        store.close()
