"""Places: store, service, and HTTP contract tests (no torch, no camera).

The store and service are exercised with a deterministic fake encoder and a
static capture source, so these run anywhere numpy/cv2 do — the real DINOv3
encoder is covered separately on the robot.
"""

import json
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pytest

from bebop_vision.places import (
    PlaceStore,
    PlacesService,
    StaticCapture,
    View,
)


def _jpeg(value: int) -> bytes:
    ok, buf = cv2.imencode(".jpg", np.full((32, 32, 3), value, np.uint8))
    assert ok
    return buf.tobytes()


class FakeEncoder:
    """jpeg -> a preset vector (or a deterministic fallback)."""

    dim = 4
    device = "fake"

    def __init__(self, table=None):
        self.table = dict(table or {})

    @property
    def loaded(self):
        return True

    def encode_bgr(self, bgr):
        raise NotImplementedError

    def encode_jpeg(self, jpeg):
        if jpeg in self.table:
            v = np.asarray(self.table[jpeg], np.float32)
        else:
            v = np.asarray([float(sum(jpeg) % 997), 1.0, 0.0, 0.0], np.float32)
        return v / (np.linalg.norm(v) + 1e-12)


@pytest.fixture()
def store(tmp_path):
    s = PlaceStore(tmp_path / "places")
    yield s
    s.close()


# --- store -----------------------------------------------------------------


def test_create_and_list(store):
    store.create_place("Kitchen")
    store.create_place("Lab")
    names = [p.name for p in store.list_places()]
    assert names == ["Kitchen", "Lab"]  # case-insensitive name sort
    assert store.count() == 2


def test_names_are_unique(store):
    store.create_place("Kitchen")
    with pytest.raises(ValueError):
        store.create_place("kitchen")
    with pytest.raises(ValueError):
        store.create_place("   ")


def test_get_by_id_or_name(store):
    p = store.create_place("Kitchen")
    assert store.get_place(p.id).name == "Kitchen"
    assert store.get_place("kitchen").id == p.id
    assert store.get_place("nope") is None


def test_rename_and_delete(store):
    p = store.create_place("Kitchen")
    store.create_place("Lab")
    renamed = store.rename_place(p.id, "Dining Room")
    assert renamed.name == "Dining Room"
    with pytest.raises(ValueError):
        store.rename_place(renamed.id, "Lab")
    assert store.delete_place("Dining Room") is True
    assert store.delete_place("Dining Room") is False
    assert store.count() == 1


def test_facts_tags_and_odom(store):
    p = store.create_place("Lab", odom=(1.5, -0.5, 0.2))
    store.add_fact(p.id, "  there is a red mug  ", source="operator")
    store.add_fact(p.id, "vision saw a desk", source="vision")
    store.add_tag(p.id, "desk", confidence=0.8, source="vision")
    got = store.get_place(p.id)
    assert got.odom == (1.5, -0.5, 0.2)
    assert [f.text for f in got.facts] == ["there is a red mug", "vision saw a desk"]
    assert got.facts[0].source == "operator"
    assert got.tags[0].label == "desk"


def test_keyframes_roundtrip(store):
    p = store.create_place("Kitchen")
    kf = store.add_keyframe(p.id, _jpeg(10), role="near", stamp_us=12345)
    assert kf.path.endswith(".jpg")
    assert (store.root / "keyframes" / p.id) in Path(kf.path).parents
    got = store.get_place(p.id)
    assert got.keyframes[0].role == "near"
    # to_dict exposes a path relative to the store root
    d = got.to_dict(root=store.root)
    assert d["keyframes"][0]["path"] == f"keyframes/{p.id}/{Path(kf.path).name}"


def test_descriptor_match_ranks_and_normalises(store):
    a = store.create_place("Kitchen")
    b = store.create_place("Lab")
    store.add_descriptor(a.id, np.array([2.0, 0.0, 0.0, 0.0]))  # unnormalised ok
    store.add_descriptor(b.id, np.array([0.0, 1.0, 0.0, 0.0]))
    matches = store.match(np.array([1.0, 0.0, 0.0, 0.0]))
    assert [m.place.name for m in matches] == ["Kitchen", "Lab"]
    assert matches[0].score == pytest.approx(1.0, abs=1e-6)
    assert matches[1].score == pytest.approx(0.0, abs=1e-6)


def test_match_best_per_place(store):
    a = store.create_place("Kitchen")
    store.add_descriptor(a.id, np.array([1.0, 0.0, 0.0, 0.0]))
    store.add_descriptor(a.id, np.array([0.0, 1.0, 0.0, 0.0]))
    matches = store.match(np.array([0.0, 1.0, 0.0, 0.0]))
    assert len(matches) == 1
    assert matches[0].score == pytest.approx(1.0, abs=1e-6)


def test_persists_across_reopen(tmp_path):
    root = tmp_path / "places"
    s1 = PlaceStore(root)
    p = s1.create_place("Kitchen")
    s1.add_fact(p.id, "coffee machine")
    s1.add_descriptor(p.id, np.array([1.0, 0.0, 0.0, 0.0]))
    s1.close()

    s2 = PlaceStore(root)
    assert s2.count() == 1
    got = s2.get_place("Kitchen")
    assert got is not None and got.facts[0].text == "coffee machine"
    assert s2.descriptor_count() == 1
    s2.close()


# --- service + whereami ----------------------------------------------------


def _service(tmp_path, views, encoder):
    store = PlaceStore(tmp_path / "places")
    svc = PlacesService(
        store, capture=StaticCapture(views), encoder=encoder,
        host="127.0.0.1", port=0, min_score=0.5, min_margin=0.1,
    )
    return store, svc


def test_teach_encodes_and_stores(tmp_path):
    a = _jpeg(20)
    enc = FakeEncoder({a: [1, 0, 0, 0]})
    store, svc = _service(tmp_path, [View("near", a, 111)], enc)
    try:
        res = svc.teach("Kitchen")
        assert res["descriptors"] == 1
        assert res["views"] == 1
        assert svc.store.descriptor_count() == 1
    finally:
        store.close()


def test_capture_views_appends_to_existing_place(tmp_path):
    a, b = _jpeg(20), _jpeg(80)
    enc = FakeEncoder({a: [1, 0, 0, 0], b: [0, 1, 0, 0]})
    store, svc = _service(tmp_path, [View("far", a, 1)], enc)
    try:
        svc.teach("Kitchen")
        assert svc.store.descriptor_count() == 1
        svc.capture = StaticCapture([View("far", b, 2)])
        out = svc.capture_views("Kitchen")
        assert out["keyframes"] == 1 and out["descriptors"] == 1
        assert out["views_total"] == 2
        assert svc.store.descriptor_count() == 2
    finally:
        store.close()


def test_capture_burst_dedups_a_static_frame(tmp_path):
    a = _jpeg(20)
    enc = FakeEncoder({a: [1, 0, 0, 0]})
    store, svc = _service(tmp_path, [View("far", a, 1)], enc)
    try:
        svc.teach("Kitchen")
        # The same camera stamp every tick -> only the first is captured.
        out = svc.capture_views("Kitchen", seconds=0.4, interval_s=0.1)
        assert out["keyframes"] == 1
        assert out["views_total"] == 2  # the taught view + this one
    finally:
        store.close()


def test_whereami_matches_known_room(tmp_path):
    a, b = _jpeg(20), _jpeg(80)
    enc = FakeEncoder({a: [1, 0, 0, 0], b: [0, 1, 0, 0]})
    # teach both, then query with A
    store, svc = _service(tmp_path, [View("near", a, 1)], enc)
    try:
        svc.teach("Kitchen")
        svc.capture = StaticCapture([View("near", b, 2)])
        svc.teach("Lab")
        svc.capture = StaticCapture([View("near", a, 3)])
        out = svc.whereami()
        assert out["matched"] is True
        assert out["place"]["name"] == "Kitchen"
        assert out["score"] == pytest.approx(1.0, abs=1e-6)
        assert out["margin"] == pytest.approx(1.0, abs=1e-6)
    finally:
        store.close()


def test_whereami_unknown_when_below_threshold(tmp_path):
    a, unknown = _jpeg(20), _jpeg(200)
    enc = FakeEncoder({a: [1, 0, 0, 0], unknown: [0, 0, 1, 0]})
    store, svc = _service(tmp_path, [View("near", a, 1)], enc)
    try:
        svc.teach("Kitchen")
        svc.capture = StaticCapture([View("near", unknown, 2)])
        out = svc.whereami()
        assert out["matched"] is False
        assert out["place"] is None
    finally:
        store.close()


def test_whereami_no_frames(tmp_path):
    store, svc = _service(tmp_path, [], FakeEncoder())
    try:
        out = svc.whereami()
        assert out["matched"] is False
        assert "frames" in out["reason"]
    finally:
        store.close()


# --- HTTP contract ---------------------------------------------------------


def _get(port, path):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=5) as r:
        return r.status, json.loads(r.read())


def _post(port, path, body):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=5) as r:
        return r.status, json.loads(r.read())


def test_http_smoke(tmp_path):
    a = _jpeg(20)
    enc = FakeEncoder({a: [1, 0, 0, 0]})
    store, svc = _service(tmp_path, [View("near", a, 1)], enc)
    svc.start()
    try:
        port = svc.bound_port
        status, health = _get(port, "/healthz")
        assert status == 200 and health["ok"] is True

        status, res = _post(port, "/places", {"name": "Kitchen",
                                              "facts": ["coffee machine"]})
        assert status == 201
        pid = res["place"]["id"]

        status, listing = _get(port, "/places")
        assert status == 200 and [p["name"] for p in listing["places"]] == ["Kitchen"]

        status, one = _get(port, f"/places/{pid}")
        assert one["place"]["facts"][0]["text"] == "coffee machine"

        status, fact = _post(port, f"/places/{pid}/facts", {"text": "has a sink"})
        assert status == 201 and fact["fact"]["source"] == "operator"

        status, wa = _get(port, "/whereami")
        assert wa["matched"] is True and wa["place"]["name"] == "Kitchen"

        # rename: the id keeps resolving, the old name does not
        status, _ = _post(port, f"/places/{pid}/rename", {"name": "Dining"})
        assert status == 200
        status, one = _get(port, f"/places/{pid}")
        assert one["place"]["name"] == "Dining"
        with pytest.raises(urllib.error.HTTPError) as exc:
            _get(port, "/places/Kitchen")
        assert exc.value.code == 404

        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/places/{pid}", method="DELETE")
        with urllib.request.urlopen(req, timeout=5) as r:
            assert r.status == 200
        status, listing = _get(port, "/places")
        assert listing["places"] == []
    finally:
        svc.stop()
        store.close()
