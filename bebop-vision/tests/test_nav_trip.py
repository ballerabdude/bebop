"""Graph planner + trip runner tests (no hardware)."""

import json
import math
import urllib.request
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from bebop_vision.nav import NavServer, TripRunner
from bebop_vision.places import PlaceStore
from bebop_vision.places.planner import edge_cost, plan


def _jpeg(v):
    ok, buf = cv2.imencode(".jpg", np.full((8, 8, 3), v, np.uint8))
    return buf.tobytes()


def _edge(a, b, routes, trav=0, succ=0, median=None):
    return {"from": a, "to": b, "route_ids": list(routes), "traversals": trav,
            "successes": succ, "median_duration_s": median}


ROUTES = {"r1": {"length_m": 4.0, "created_at": 1},
          "r1b": {"length_m": 3.0, "created_at": 2},
          "r2": {"length_m": 2.0, "created_at": 1},
          "r3": {"length_m": 10.0, "created_at": 1},
          "r4": {"length_m": 1.0, "created_at": 1}}


def test_edge_cost_rules():
    assert edge_cost(_edge("a", "b", []), ROUTES) is None
    assert edge_cost(_edge("a", "b", ["ghost"]), ROUTES) is None
    # No duration yet: newest route length at 0.2 m/s.
    assert edge_cost(_edge("a", "b", ["r1", "r1b"]), ROUTES) == pytest.approx(15.0)
    # Measured duration wins; failures penalise.
    assert edge_cost(_edge("a", "b", ["r1"], 4, 4, 20.0), ROUTES) == pytest.approx(20.0)
    assert edge_cost(_edge("a", "b", ["r1"], 4, 2, 20.0), ROUTES) == pytest.approx(40.0)


def test_plan_prefers_cheap_reliable_chain_and_respects_direction():
    graph = {"edges": [
        _edge("a", "b", ["r1"], 2, 2, 10.0),
        _edge("b", "c", ["r2"], 2, 2, 5.0),
        _edge("a", "c", ["r3"], 4, 1, 12.0),      # direct but unreliable
        _edge("c", "d", ["r4"]),
        _edge("d", "a", []),                       # no route: unusable
    ]}
    legs = plan(graph, ROUTES, "a", "c")
    assert [(l["from"], l["to"], l["route_id"]) for l in legs] == [
        ("a", "b", "r1"), ("b", "c", "r2")]
    assert plan(graph, ROUTES, "a", "d")[-1]["route_id"] == "r4"
    assert plan(graph, ROUTES, "c", "a") == []     # directed
    assert plan(graph, ROUTES, "a", "a") == []
    # Newest route on an edge is the one to drive.
    graph2 = {"edges": [_edge("a", "b", ["r1", "r1b"])]}
    assert plan(graph2, ROUTES, "a", "b")[0]["route_id"] == "r1b"
    # Unreliable edges can be excluded outright.
    graph3 = {"edges": [_edge("a", "c", ["r3"], 4, 1, 12.0)]}
    assert plan(graph3, ROUTES, "a", "c") and \
        plan(graph3, ROUTES, "a", "c", min_success_rate=0.5) == []


def _store_with_chain(tmp_path):
    store = PlaceStore(tmp_path / "places")
    a = store.create_place("office")
    b = store.create_place("hall")
    c = store.create_place("bedroom")
    nodes = lambda n: [{"x": i, "y": 0, "theta": 0, "dist_m": i, "t_s": i,
                        "keyframes": {"far": _jpeg(1)}, "vec": None,
                        "vec_role": "far"} for i in range(n)]
    r1 = store.create_route(a.id, b.id, nodes(3), length_m=2.0, duration_s=8)
    r2 = store.create_route(b.id, c.id, nodes(4), length_m=3.0, duration_s=12)
    return store, a, b, c, r1, r2


def test_store_plan_trip(tmp_path):
    store, a, b, c, r1, r2 = _store_with_chain(tmp_path)
    p = store.plan_trip("office", "bedroom")
    assert p["reachable"] and [l["route_id"] for l in p["legs"]] == [r1.id, r2.id]
    assert p["legs"][0]["from_name"] == "office" and p["legs"][1]["to_name"] == "bedroom"
    assert p["legs"][1]["length_m"] == pytest.approx(3.0)
    assert p["cost_s"] == pytest.approx(20.0)        # taught durations
    assert store.plan_trip("bedroom", "office")["reachable"] is False
    assert store.plan_trip("office", "office")["reachable"] is True
    with pytest.raises(KeyError):
        store.plan_trip("office", "attic")


class FakeFollower:
    def __init__(self, outcomes=None):
        self.state = "idle"
        self.started = []
        self.outcomes = list(outcomes or [])
        self.stopped = 0

    def follow(self, route_id):
        self.started.append(route_id)
        self.state = "following"
        return self.status()

    def finish(self):
        self.state = self.outcomes.pop(0) if self.outcomes else "arrived"

    def stop(self):
        self.stopped += 1
        self.state = "stopped"

    def status(self):
        return {"state": self.state}


def _runner(tmp_path, outcomes=None, recognize=None):
    store, a, b, c, r1, r2 = _store_with_chain(tmp_path)
    f = FakeFollower(outcomes)
    t = TripRunner(f, store, recognize=recognize, log=lambda *x: None)
    return t, f, store, (a, b, c), (r1, r2)


def test_trip_drives_legs_in_order_and_arrives(tmp_path):
    t, f, store, (a, b, c), (r1, r2) = _runner(tmp_path)
    st = t.start("bedroom", from_place="office")
    assert st["state"] == "running" and st["legs"] == 2
    t.tick(); assert f.started == [r1.id]
    t.tick(); assert f.state == "following"          # waiting
    f.finish(); t.tick()                               # leg 1 arrived
    assert t.status()["leg"] == 1
    t.tick(); assert f.started == [r1.id, r2.id]
    f.finish(); t.tick()
    st = t.status()
    assert st["state"] == "arrived" and st["history"][-1]["legs"] == 2
    assert st["detail"]["leg1"] == "arrived"


def test_trip_fails_on_failed_leg_and_stop(tmp_path):
    t, f, store, places, routes = _runner(tmp_path, ["failed_blocked"])
    t.start("bedroom", from_place="office")
    t.tick(); f.finish(); t.tick()
    assert t.status()["state"] == "failed: leg 1 failed_blocked"
    t.start("bedroom", from_place="office")
    t.tick()
    with pytest.raises(RuntimeError):
        t.start("bedroom", from_place="office")
    t.stop()
    assert t.status()["state"] == "stopped" and f.stopped == 1


def test_trip_start_uses_recognition_and_rejects_unreachable(tmp_path):
    t, f, store, (a, b, c), routes = _runner(
        tmp_path, recognize=lambda: {"place": {"id": a.id, "name": "office"},
                                     "score": 0.9})
    st = t.start("bedroom")
    assert st["plan"]["from_name"] == "office"
    assert st["detail"]["start_recognized"] == "office"
    t.stop()
    with pytest.raises(LookupError):
        t.start("office", from_place="bedroom")
    with pytest.raises(KeyError):
        t.start("attic", from_place="office")
    t2 = TripRunner(f, store, recognize=lambda: {"place": None},
                    log=lambda *x: None)
    with pytest.raises(ValueError):
        t2.start("bedroom")
    assert t2.start("office", from_place="office")["state"] == "arrived"


def test_http_trip_endpoints(tmp_path):
    t, f, store, (a, b, c), routes = _runner(tmp_path)
    drive = SimpleNamespace(trip=t, follower=f, status=lambda: {"state": "idle"},
                            cancel=lambda: {"state": "idle"},
                            clearance=lambda: None, current_pose=lambda: None)
    server = NavServer(drive, host="127.0.0.1", port=0).start()
    base = f"http://127.0.0.1:{server.bound_port}"

    def call(path, body=None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(base + path, data=data,
                                     method="POST" if data is not None else "GET",
                                     headers={"Content-Type": "application/json"} if data else {})
        try:
            with urllib.request.urlopen(req, timeout=3) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())
    try:
        assert call("/trip")[1]["state"] == "idle"
        assert call("/trip", {})[0] == 400
        assert call("/trip", {"to": "attic", "from": "office"})[0] == 404
        assert call("/trip", {"to": "office", "from": "bedroom"})[0] == 404
        code, body = call("/trip", {"to": "bedroom", "from": "office"})
        assert code == 202 and body["status"]["legs"] == 2
        assert call("/trip", {"to": "bedroom", "from": "office"})[0] == 409
        code, body = call("/trip/stop", {})
        assert code == 200 and body["status"]["state"] == "stopped"
    finally:
        server.stop()
