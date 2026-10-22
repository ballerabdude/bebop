"""Patch matcher + route follower tests (fake drive, fake encoder)."""

import json
import math
import time
import urllib.request
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from bebop_vision.nav import (FollowConfig, MatchConfig, NavServer,
                              RouteFollower, match_patches, wrap_pi)
from bebop_vision.places import PlaceStore, StaticCapture, View


# --- visual --------------------------------------------------------------------

def grid(seed=0, gh=14, gw=14, d=16):
    rng = np.random.default_rng(seed)
    g = rng.normal(size=(gh, gw, d)).astype(np.float32)
    return g / np.linalg.norm(g, axis=2, keepdims=True)


def shifted(g, dcol):
    """The same scene seen shifted right by dcol columns (wrap-free)."""
    out = grid(seed=99, gh=g.shape[0], gw=g.shape[1], d=g.shape[2])
    if dcol >= 0:
        out[:, dcol:] = g[:, :g.shape[1] - dcol]
    else:
        out[:, :g.shape[1] + dcol] = g[:, -dcol:]
    return out


def test_match_identity_and_shift_sign():
    ref = grid(1)
    m = match_patches(ref, ref)
    assert m.confident and m.median_dcol == 0 and abs(m.bearing_rad) < 1e-9
    # Scene appears 3 columns to the RIGHT in the live view -> turn right.
    m = match_patches(shifted(ref, 3), ref)
    assert m.confident and m.median_dcol == pytest.approx(3.0)
    assert m.bearing_rad < 0
    expected = -math.atan2(3 * 1280 / 14, 610.5)
    assert m.bearing_rad == pytest.approx(expected, abs=1e-6)
    m = match_patches(shifted(ref, -2), ref)
    assert m.confident and m.bearing_rad > 0


def test_match_rejects_unrelated_and_ambiguous():
    ref = grid(1)
    m = match_patches(grid(2), ref)
    assert not m.confident
    assert m.n_matches < 12 or m.mad_dcol > 2.5
    with pytest.raises(ValueError):
        match_patches(grid(1, gh=7), ref)
    m = match_patches(shifted(ref, 9), ref, MatchConfig(max_bearing_deg=30))
    assert not m.confident           # beyond the bearing bound


# --- fakes ---------------------------------------------------------------------

def _jpeg(v):
    ok, buf = cv2.imencode(".jpg", np.full((8, 8, 3), v, np.uint8))
    return buf.tobytes()


class FakeDrive:
    """Records goals; teleports the pose toward each goal in steps."""

    def __init__(self, x=0.0, y=0.0, th=0.0, step=0.3):
        self.x, self.y, self.th = x, y, th
        self.step = step
        self.goals = []
        self.cancelled = 0
        self.state = "idle"
        self.estop = False
        self.follower = None

    def current_pose(self):
        return SimpleNamespace(as_tuple=lambda: (self.x, self.y, self.th),
                               heading_source="gyro")

    def status(self):
        return {"state": self.state,
                "robot": {"estop_latched": self.estop}}

    def set_goal(self, *, bearing_deg, distance_m, label=""):
        if self.estop:
            raise RuntimeError("E-STOP latched")
        self.goals.append((bearing_deg, distance_m, label))
        b = math.radians(bearing_deg)
        self.th = wrap_pi(self.th + b)
        d = min(distance_m, self.step)
        self.x += d * math.cos(self.th)
        self.y += d * math.sin(self.th)
        self.state = "drive"
        return {"x": self.x, "y": self.y}

    def cancel(self):
        self.cancelled += 1
        self.state = "stopped"
        return self.status()


class FakePatchEncoder:
    """Keyframe JPEG value v -> grid(seed=v); live view -> the same grid
    shifted by `self.shift` columns (scene appears shifted right)."""

    dim = 16

    def __init__(self):
        self.shift = 0
        self.calls = 0

    def encode_patches_jpeg(self, jpeg, image_size=None):
        self.calls += 1
        bgr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        v = int(bgr[0, 0, 0])
        if v == 200:                      # live view
            return None, shifted(grid(self.ref_seed), self.shift)
        return None, grid(v)

    ref_seed = 1


def _route_store(tmp_path, nodes_xy):
    store = PlaceStore(tmp_path / "places")
    a = store.create_place("office")
    b = store.create_place("bedroom")
    nodes = []
    d = 0.0
    for i, (x, y) in enumerate(nodes_xy):
        if i:
            d += math.hypot(x - nodes_xy[i - 1][0], y - nodes_xy[i - 1][1])
        nodes.append({"x": x, "y": y, "theta": 0.0, "dist_m": d, "t_s": i,
                      "keyframes": {"far": _jpeg(1)}, "vec": None,
                      "vec_role": "far"})
    route = store.create_route(a.id, b.id, nodes, length_m=d, duration_s=10)
    return store, route, a, b


def _follower(tmp_path, nodes_xy, drive=None, enc=None, recognize=None,
              **cfg):
    store, route, a, b = _route_store(tmp_path, nodes_xy)
    drive = drive or FakeDrive()
    f = RouteFollower(drive, store, enc, StaticCapture([View("far", _jpeg(200), 1)]),
                      recognize=recognize, config=FollowConfig(**cfg),
                      log=lambda *a: None)
    drive.follower = f
    return f, store, route, drive, a, b


def run(f, max_ticks=200):
    for _ in range(max_ticks):
        f.tick(now=float(_))
        if f.status()["state"] != "following":
            return f.status()
    raise AssertionError(f"still following: {f.status()}")


# --- follower --------------------------------------------------------------------

def test_follows_dead_reckoning_and_arrives(tmp_path):
    f, store, route, drive, a, b = _follower(
        tmp_path, [(0, 0), (1.0, 0), (2.0, 0.5), (2.5, 1.5)], visual=False,
        recognize=lambda: {"place": {"id": b.id, "name": "bedroom"}})
    f.follow(route.id)
    st = run(f)
    assert st["state"] == "arrived"
    assert math.hypot(drive.x - 2.5, drive.y - 1.5) < 0.45
    assert drive.cancelled >= 1
    assert all(0 < d <= 0.8 for _, d, _ in drive.goals)
    edge = store.get_edge(a.id, b.id)
    assert edge.traversals == 2 and edge.successes == 2   # teach + follow
    assert st["history"][-1]["node_reached"] == 3


def test_route_frame_is_anchored_to_the_live_pose(tmp_path):
    # Robot starts at (5, 5) facing +y: the straight route must run along +y.
    drive = FakeDrive(5.0, 5.0, math.pi / 2)
    f, store, route, drive, a, b = _follower(
        tmp_path, [(0, 0), (1.0, 0), (2.0, 0)], drive=drive, visual=False,
        confirm_arrival=False)
    f.follow(route.id)
    st = run(f)
    assert st["state"] == "arrived"
    assert drive.x == pytest.approx(5.0, abs=0.3)
    assert drive.y == pytest.approx(7.0, abs=0.45)


def test_visual_shift_steers_and_corrects_yaw(tmp_path):
    enc = FakePatchEncoder()
    enc.shift = -2            # scene appears LEFT in the live view -> turn left
    f, store, route, drive, a, b = _follower(
        tmp_path, [(0, 0), (3.0, 0)], enc=enc, confirm_arrival=False,
        yaw_correction_alpha=0.5)
    f.follow(route.id)
    f.tick(now=0.0)
    st = f.status()
    assert st["detail"]["source"] == "visual"
    assert st["match"]["confident"]
    assert drive.goals[-1][0] > 0                      # turned left
    assert st["yaw_correction_deg"] > 0
    enc.shift = 0
    f.tick(now=1.0)
    assert f.status()["detail"]["source"] == "visual"
    assert abs(drive.goals[-1][0]) < 1e-6


def test_unconfident_match_falls_back_to_dead_reckoning(tmp_path):
    enc = FakePatchEncoder()
    enc.ref_seed = 7                                   # live never matches
    f, store, route, drive, a, b = _follower(
        tmp_path, [(0, 0), (2.0, 0)], enc=enc, confirm_arrival=False)
    f.follow(route.id)
    f.tick(now=0.0)
    assert f.status()["detail"]["source"] == "dead_reckoning"
    assert f.status()["match"]["confident"] is False


def test_blocked_twice_fails_and_records_failure(tmp_path):
    f, store, route, drive, a, b = _follower(
        tmp_path, [(0, 0), (5.0, 0)], visual=False, max_blocked_aborts=2)
    f.follow(route.id)
    f.tick(now=0.0)
    drive.state = "aborted_blocked"
    f.tick(now=1.0)
    assert f.status()["state"] == "following"
    drive.state = "drive"
    f.tick(now=2.0)
    drive.state = "aborted_blocked"
    f.tick(now=3.0)
    assert f.status()["state"] == "failed_blocked"
    edge = store.get_edge(a.id, b.id)
    assert edge.traversals == 2 and edge.successes == 1


def test_estop_and_stop(tmp_path):
    f, store, route, drive, a, b = _follower(
        tmp_path, [(0, 0), (5.0, 0)], visual=False)
    f.follow(route.id)
    with pytest.raises(RuntimeError, match="already"):
        f.follow(route.id)
    f.stop()
    assert f.status()["state"] == "stopped"
    assert store.get_edge(a.id, b.id).traversals == 1   # stop is not recorded
    f.follow(route.id)
    drive.estop = True
    f.tick(now=0.0)
    assert f.status()["state"] == "failed_estop"


def test_follow_by_destination_uses_recognition(tmp_path):
    f, store, route, drive, a, b = _follower(
        tmp_path, [(0, 0), (1.0, 0)], visual=False,
        recognize=lambda: {"place": {"id": a.id, "name": "office"}})
    st = f.follow(to_place="bedroom")
    assert st["route"]["id"] == route.id
    f.stop()
    with pytest.raises(KeyError):
        f.follow(to_place="kitchen")
    with pytest.raises(KeyError):
        f.follow("route-nope")
    with pytest.raises(ValueError):
        f.follow()


def test_arrival_unconfirmed_when_recognition_disagrees(tmp_path):
    f, store, route, drive, a, b = _follower(
        tmp_path, [(0, 0), (1.0, 0)], visual=False,
        recognize=lambda: {"place": {"id": a.id, "name": "office"}})
    f.follow(route.id)
    st = run(f)
    assert st["state"] == "arrived_unconfirmed"
    assert st["detail"]["recognized"] == "office"


def test_http_route_endpoints(tmp_path):
    f, store, route, drive, a, b = _follower(
        tmp_path, [(0, 0), (1.0, 0)], visual=False)
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
        assert call("/route")[1]["state"] == "idle"
        code, body = call("/route/follow", {"route_id": "nope"})
        assert code == 404
        code, body = call("/route/follow", {"route_id": route.id})
        assert code == 202 and body["status"]["state"] == "following"
        code, body = call("/route/follow", {"route_id": route.id})
        assert code == 409
        code, body = call("/route/stop", {})
        assert code == 200 and body["status"]["state"] == "stopped"
    finally:
        server.stop()


def test_blocked_driver_is_not_refreshed(tmp_path):
    f, store, route, drive, a, b = _follower(
        tmp_path, [(0, 0), (5.0, 0)], visual=False)
    f.follow(route.id)
    f.tick(now=0.0)
    n = len(drive.goals)
    drive.state = "blocked"
    f.tick(now=1.0)
    f.tick(now=2.0)
    assert len(drive.goals) == n                 # no goal refresh while blocked
    assert f.status()["detail"]["hold"] == "blocked"
    drive.state = "drive"
    f.tick(now=3.0)
    assert len(drive.goals) == n + 1
