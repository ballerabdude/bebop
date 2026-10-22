"""Route store + trail teach mode tests (no torch, no camera, no robot)."""

import json
import math
import urllib.error
import urllib.request
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from bebop_vision.places import (PlaceStore, PlacesService, StaticCapture,
                                 TrailConfig, TrailTeacher, View)


def _jpeg(value: int) -> bytes:
    ok, buf = cv2.imencode(".jpg", np.full((16, 16, 3), value, np.uint8))
    assert ok
    return buf.tobytes()


class FakeEncoder:
    dim = 4
    device = "fake"
    loaded = True

    def __init__(self, table=None):
        self.table = dict(table or {})
        self.calls = 0

    def encode_jpeg(self, jpeg):
        self.calls += 1
        if jpeg in self.table:
            return np.asarray(self.table[jpeg], dtype=np.float32)
        h = (sum(jpeg) % 97) / 97.0
        return np.asarray([h, 1 - h, 0.1, 0.2], dtype=np.float32)


class FakePose:
    """A pose source the test moves by hand."""

    def __init__(self, x=0.0, y=0.0, theta=0.0, source="gyro"):
        self.x, self.y, self.theta, self.source = x, y, theta, source

    def __call__(self):
        return SimpleNamespace(
            as_tuple=lambda: (self.x, self.y, self.theta),
            heading_source=self.source)


def _store(tmp_path):
    store = PlaceStore(tmp_path / "places")
    a = store.create_place("office")
    b = store.create_place("bedroom")
    return store, a, b


def _node(x, y, th=0.0, d=0.0, t=0.0, vec=None, kf=True):
    return {"x": x, "y": y, "theta": th, "dist_m": d, "t_s": t,
            "keyframes": {"far": _jpeg(int(50 + x * 10) % 255)} if kf else {},
            "vec": vec, "vec_role": "far" if vec is not None else None}


# --- store -----------------------------------------------------------------

def test_create_get_list_delete_route(tmp_path):
    store, a, b = _store(tmp_path)
    nodes = [_node(0, 0, vec=[1, 0, 0, 0]), _node(1, 0, d=1.0, t=3.0),
             _node(2, 0.5, th=0.3, d=2.1, t=6.5, vec=[0, 1, 0, 0])]
    r = store.create_route(a.id, "bedroom", nodes, length_m=2.1,
                           duration_s=6.5, label="hallway",
                           heading_source="gyro")
    assert r.id.startswith("route-") and r.node_count == 3
    assert r.from_name == "office" and r.to_name == "bedroom"
    got = store.get_route(r.id)
    assert [n.seq for n in got.nodes] == [0, 1, 2]
    assert got.nodes[2].x == pytest.approx(2.0)
    assert got.nodes[1].has_descriptor is False
    assert got.nodes[0].has_descriptor and got.nodes[0].vec_role == "far"
    kf = store.route_keyframe_path(r.id, 1, "far")
    assert kf is not None and kf.is_file()
    assert store.route_keyframe_path(r.id, 1, "near") is None
    seqs, vecs = store.route_descriptors(r.id)
    assert seqs.tolist() == [0, 2] and vecs.shape == (2, 4)
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0)
    # The edge was recorded with the trail as its guide path.
    edge = store.get_edge(a.id, b.id)
    assert edge is not None and edge.successes == 1
    assert edge.guide_path == [[0.0, 0.0], [1.0, 0.0], [2.0, 0.5]]
    assert edge.median_duration_s == pytest.approx(6.5)
    g = store.graph()
    assert g["edges"][0]["route_ids"] == [r.id]
    assert [x.id for x in store.list_routes(from_place="office")] == [r.id]
    assert store.list_routes(to_place="office") == []
    d = r.to_dict(root=store.root)
    json.dumps(d)
    assert d["nodes"][1]["keyframes"]["far"].startswith("routes/")
    assert store.delete_route(r.id) and not store.delete_route(r.id)
    assert store.get_route(r.id) is None
    assert not kf.exists()


def test_route_rejects_unknown_place_and_short_trails(tmp_path):
    store, a, b = _store(tmp_path)
    with pytest.raises(KeyError):
        store.create_route(a.id, "attic", [_node(0, 0), _node(1, 0)],
                           length_m=1, duration_s=1)
    with pytest.raises(ValueError):
        store.create_route(a.id, b.id, [_node(0, 0)], length_m=0,
                           duration_s=1)


def test_deleting_a_place_cascades_its_routes(tmp_path):
    store, a, b = _store(tmp_path)
    r = store.create_route(a.id, b.id, [_node(0, 0), _node(1, 0)],
                           length_m=1, duration_s=2)
    assert store.delete_place(b.id)
    assert store.get_route(r.id) is None
    assert store.route_descriptors(r.id)[0].size == 0


# --- teacher ---------------------------------------------------------------

def _teacher(store, pose, views=None, cfg=None, recognize=None):
    enc = FakeEncoder()
    cap = StaticCapture(views if views is not None
                        else [View("far", _jpeg(10), 1), View("near", _jpeg(20), 1)])
    t = TrailTeacher(store, cap, enc, pose, cfg or TrailConfig(),
                     recognize=recognize, log=lambda *a: None)
    return t, enc


def test_teach_takes_nodes_on_distance_and_turns(tmp_path):
    store, a, b = _store(tmp_path)
    pose = FakePose(5.0, 2.0, math.pi / 2)      # route frame is not the odom frame
    t, enc = _teacher(store, pose, cfg=TrailConfig(step_m=0.6, turn_deg=25,
                                                  min_interval_s=0.0))
    now = 100.0
    st = t.start("office", label="to bed", now=now)
    assert st["active"] and st["nodes"] == 1
    # Drive 2.0 m "forward" in the robot's heading (odom +y) in 0.1 m steps.
    for i in range(20):
        pose.y += 0.1
        now += 0.2
        t.tick(now=now)
    st = t.status()
    assert st["nodes"] == 4                    # 0, 0.6, 1.2, 1.8
    assert st["length_m"] == pytest.approx(2.0, abs=1e-6)
    assert st["last_node"]["x"] == pytest.approx(1.8, abs=1e-6)
    assert abs(st["last_node"]["y"]) < 1e-6    # route frame: x along start heading
    # Turn 30 deg on the spot: a node from the heading change alone.
    pose.theta += math.radians(30)
    now += 0.5
    assert t.tick(now=now)
    assert t.status()["nodes"] == 5
    # Small jitter: no node.
    pose.y += 0.05
    now += 0.5
    assert not t.tick(now=now)
    res = t.stop("bedroom", now=now + 1.0)
    assert res["route"]["from_name"] == "office"
    assert res["route"]["to_name"] == "bedroom"
    assert res["nodes"] == 6                   # + final node
    assert not res["created_place"]
    route = store.get_route(res["route"]["id"])
    assert route.heading_source == "gyro"
    assert route.nodes[-1].dist_m == pytest.approx(2.05, abs=1e-6)
    assert route.nodes[-1].theta == pytest.approx(math.radians(30), abs=1e-6)
    assert all(n.has_descriptor for n in route.nodes)
    assert all("far" in n.keyframes and "near" in n.keyframes
               for n in route.nodes)
    assert t.status()["active"] is False
    assert store.get_edge(a.id, b.id).guide_path[0] == [0.0, 0.0]


def test_teach_refuses_without_gyro_or_pose(tmp_path):
    store, a, b = _store(tmp_path)
    t, _ = _teacher(store, FakePose(source="odom"))
    with pytest.raises(RuntimeError, match="gyro"):
        t.start("office")
    t2, _ = _teacher(store, lambda: None)
    with pytest.raises(RuntimeError, match="no pose"):
        t2.start("office")
    t3, _ = _teacher(store, FakePose(source="odom"),
                     cfg=TrailConfig(require_gyro=False))
    assert t3.start("office")["active"]
    with pytest.raises(RuntimeError, match="already active"):
        t3.start("office")


def test_teach_uses_recognition_for_endpoints_and_creates_new_place(tmp_path):
    store, a, b = _store(tmp_path)
    pose = FakePose()
    rec = {"place": {"id": a.id}}
    t, enc = _teacher(store, pose, recognize=lambda: rec)
    t.start(now=0.0)                           # from recognition
    pose.x += 1.0
    t.tick(now=10.0)
    res = t.stop("Guest room", facts=["has a bunk bed"], now=12.0)
    assert res["created_place"]
    guest = store.get_place("Guest room")
    assert guest is not None
    assert store.descriptor_count(guest.id) >= 1
    assert [f.text for f in store.get_place(guest.id).facts] == ["has a bunk bed"] \
        if hasattr(store.get_place(guest.id), "facts") else True
    assert store.get_edge(a.id, guest.id) is not None
    # Unknown start with nothing recognised is an error, not a guess.
    t2, _ = _teacher(store, FakePose(), recognize=lambda: {"place": None})
    with pytest.raises(ValueError, match="from_place"):
        t2.start()
    with pytest.raises(KeyError):
        t2.start("nowhere")


def test_cancel_discards_and_stop_without_session_errors(tmp_path):
    store, a, b = _store(tmp_path)
    t, _ = _teacher(store, FakePose())
    t.start("office")
    assert t.cancel() == {"cancelled": True}
    assert t.cancel() == {"cancelled": False}
    with pytest.raises(RuntimeError, match="no teach session"):
        t.stop("bedroom")
    assert store.list_routes() == []


def test_stop_needs_a_destination(tmp_path):
    store, a, b = _store(tmp_path)
    t, _ = _teacher(store, FakePose(), recognize=lambda: {"place": None})
    t.start("office")
    with pytest.raises(ValueError, match="to_place"):
        t.stop()
    assert t.status()["active"]                # still recording


# --- HTTP ------------------------------------------------------------------

def _call(port, method, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=data, method=method,
        headers={"Content-Type": "application/json"} if data else {})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            ct = r.headers.get("Content-Type", "")
            raw = r.read()
            return r.status, (json.loads(raw) if "json" in ct else raw)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_http_teach_and_routes(tmp_path):
    store = PlaceStore(tmp_path / "places")
    store.create_place("office")
    enc = FakeEncoder()
    views = [View("far", _jpeg(10), 1), View("near", _jpeg(20), 1)]
    svc = PlacesService(store, capture=StaticCapture(views), encoder=enc,
                        host="127.0.0.1", port=0)
    svc.start()
    port = svc.bound_port
    try:
        code, body = _call(port, "GET", "/routes/teach")
        assert code == 503                      # no teacher configured
        pose = FakePose()
        svc.teacher = TrailTeacher(store, svc.capture, enc, pose,
                                   TrailConfig(min_interval_s=0.0),
                                   log=lambda *a: None)
        code, body = _call(port, "GET", "/routes/teach")
        assert code == 200 and body["active"] is False
        assert body["pose"]["heading_source"] == "gyro"
        code, body = _call(port, "POST", "/routes/teach/start",
                           {"from": "nowhere"})
        assert code == 404
        code, body = _call(port, "POST", "/routes/teach/start",
                           {"from": "office", "label": "t"})
        assert code == 201 and body["active"] and body["nodes"] == 1
        svc.teacher._t0 = 0.0
        svc.teacher._last_node_ts = 0.0
        code, body = _call(port, "POST", "/routes/teach/start",
                           {"from": "office"})
        assert code == 409
        pose.x = 1.0
        svc.teacher.tick(now=5.0)
        code, body = _call(port, "POST", "/routes/teach/stop",
                           {"to": "bedroom"})
        assert code == 201 and body["created_place"]
        rid = body["route"]["id"]
        code, body = _call(port, "GET", "/routes")
        assert code == 200 and [r["id"] for r in body["routes"]] == [rid]
        code, body = _call(port, "GET", f"/routes/{rid}")
        assert code == 200 and len(body["route"]["nodes"]) == 3
        code, body = _call(port, "GET", f"/routes/{rid}/nodes/1/keyframe?role=far")
        assert code == 200 and body[:2] == b"\xff\xd8"
        code, body = _call(port, "GET", f"/routes/{rid}/nodes/9/keyframe")
        assert code == 404
        code, body = _call(port, "GET", "/places/graph")
        assert body["edges"][0]["route_ids"] == [rid]
        code, body = _call(port, "POST", "/routes/teach/stop", {"to": "x"})
        assert code == 409
        code, body = _call(port, "DELETE", f"/routes/{rid}")
        assert code == 200
        code, body = _call(port, "DELETE", f"/routes/{rid}")
        assert code == 404
    finally:
        svc.stop()


# --- capture: far camera drops color frames ---------------------------------

def test_rig_capture_uses_newest_color_frame_from_history():
    import time
    from bebop_vision.places import RigCapture

    class Fr:
        def __init__(self, jpeg, age, stamp):
            self.color_jpeg, self._age, self.stamp_us = jpeg, age, stamp

        def age_s(self, now=None):
            return self._age

    class Cam:
        def __init__(self, frames):
            self._frames = frames

        def read(self):
            return self._frames[-1]

        def recent(self):
            return list(self._frames)

    rig = SimpleNamespace(cameras={
        # latest slot is depth-only; an older complete frame is fresh enough
        "far": Cam([Fr(b"old", 0.9, 1), Fr(b"good", 0.2, 2), Fr(None, 0.01, 3)]),
        # only stale color frames: skipped
        "near": Cam([Fr(b"stale", 2.0, 4), Fr(None, 0.01, 5)]),
    })
    views = RigCapture(rig, roles=("far", "near"), max_age_s=0.5).grab()
    assert [(v.role, v.jpeg, v.stamp_us) for v in views] == [("far", b"good", 2)]
