"""Navigator: command parsing and mission lifecycle (fakes, no hardware)."""

import json
import urllib.request
from types import SimpleNamespace

import pytest

from bebop_vision.nav import NavServer, Navigator, parse_command, split_steps
from bebop_vision.places import PlaceStore

NAMES = ["Kitchen", "office room", "Master Bedroom", "front-entryway"]


def test_parse_places_objects_and_moves():
    assert parse_command("go to the kitchen", NAMES)["kind"] == "place"
    assert parse_command("Take me to the master bedroom please", NAMES)["target"] == "Master Bedroom"
    assert parse_command("navigate to office", NAMES)["target"] == "office room"
    assert parse_command("go to the front entryway", NAMES)["target"] == "front-entryway"
    o = parse_command("go to the dresser", NAMES)
    assert o["kind"] == "object" and o["target"] == "dresser"
    assert parse_command("find the basketball", NAMES)["target"] == "basketball"
    assert parse_command("the red chair by the window", NAMES)["target"] == "red chair by the window"
    r = parse_command("forward 1.5 m", NAMES)
    assert r["kind"] == "relative" and r["distance_m"] == 1.5 and r["bearing_deg"] == 0
    assert parse_command("go forward", NAMES)["distance_m"] == 1.0
    assert parse_command("move ahead 50 cm", NAMES)["distance_m"] == pytest.approx(0.5)
    assert parse_command("back up 2 metres", NAMES)["kind"] == "reverse"
    assert parse_command("reverse half a metre", NAMES)["distance_m"] == 0.5
    t = parse_command("turn left", NAMES)
    assert t["kind"] == "turn" and t["turn_deg"] == 90
    assert parse_command("turn around", NAMES)["turn_deg"] == 180
    assert parse_command("rotate right 30 degrees", NAMES)["turn_deg"] == -30
    assert parse_command("Stop!", NAMES)["kind"] == "stop"
    with pytest.raises(ValueError):
        parse_command("   ", NAMES)


class FakeDrive:
    def __init__(self):
        self.goals = []
        self.active = False
        self.state = "idle"
        self.cancelled = 0
        self.navigator = None

    def set_goal(self, **kw):
        self.goals.append(kw)
        self.active = True
        self.state = "drive"

    def cancel(self):
        self.cancelled += 1
        self.active = False
        self.state = "stopped"

    def status(self):
        return {"state": self.state, "active": self.active, "arming": False,
                "command": {"detail": {}}, "robot": {}}


class FakeTrip:
    def __init__(self):
        self.state = "idle"
        self.starts = []
        self.stops = 0

    def start(self, to, from_place=None):
        self.starts.append((to, from_place))
        self.state = "running"

    def stop(self):
        self.stops += 1
        self.state = "stopped"

    def status(self):
        return {"state": self.state, "leg": 0, "legs": 2}


class FakeSeeker:
    def __init__(self):
        self.state = "idle"
        self.seeks = []
        self.stops = 0

    def seek(self, what):
        self.seeks.append(what)
        self.state = "seeking"

    def stop(self):
        self.stops += 1
        self.state = "stopped"

    def status(self):
        return {"state": self.state, "last": {"range_m": 2.5}}


def _nav(tmp_path, recognize=None, with_route=True):
    store = PlaceStore(tmp_path / "places")
    a = store.create_place("office room")
    b = store.create_place("Kitchen")
    if with_route:
        nodes = [{"x": i, "y": 0, "theta": 0, "dist_m": i, "t_s": i,
                  "keyframes": {}, "vec": None, "vec_role": None} for i in range(2)]
        store.create_route(a.id, b.id, nodes, length_m=1, duration_s=4)
    drive, trip, seeker = FakeDrive(), FakeTrip(), FakeSeeker()
    nav = Navigator(drive, store, trip=trip, seeker=seeker, recognize=recognize,
                    log=lambda *x: None)
    drive.navigator = nav
    return nav, drive, trip, seeker, a, b


def test_place_with_route_becomes_a_trip(tmp_path):
    nav, drive, trip, seeker, a, b = _nav(
        tmp_path, recognize=lambda: {"place": {"id": a.id, "name": "office room"}})
    st = nav.command("go to the kitchen")
    assert st["state"] == "running" and st["mission"]["runner"] == "trip"
    assert trip.starts == [("Kitchen", a.id)] and st["mission"]["legs"] == 1
    assert "leg 1 of 2" in st["summary"]
    trip.state = "arrived"
    assert nav.status()["state"] == "arrived"
    assert nav.status()["summary"] == "Arrived at Kitchen."


def test_place_without_route_falls_back_to_seeking(tmp_path):
    nav, drive, trip, seeker, a, b = _nav(
        tmp_path, with_route=False,
        recognize=lambda: {"place": {"id": a.id, "name": "office room"}})
    st = nav.command("go to the kitchen")
    assert st["mission"]["runner"] == "seek" and seeker.seeks == ["the Kitchen"]
    assert "no taught route" in st["mission"]["note"]
    # Unknown current place: also seek.
    nav2, drive2, trip2, seeker2, a2, b2 = _nav(tmp_path / "x", recognize=lambda: {"place": None})
    st = nav2.command("go to the kitchen")
    assert st["mission"]["runner"] == "seek"
    assert "not recognised" in st["mission"]["note"]


def test_already_there(tmp_path):
    nav, drive, trip, seeker, a, b = _nav(
        tmp_path, recognize=lambda: {"place": {"id": b.id, "name": "Kitchen"}})
    st = nav.command("go to the kitchen")
    assert st["state"] == "arrived" and st["summary"] == "Already at Kitchen."
    assert not trip.starts and not seeker.seeks


def test_objects_moves_and_stop(tmp_path):
    nav, drive, trip, seeker, a, b = _nav(tmp_path)
    st = nav.command("go to the dresser")
    assert seeker.seeks == ["the dresser"] and st["state"] == "running"
    assert st["summary"] == "Heading for dresser, about 2.5 metres away."
    # A new command replaces the active one.
    st = nav.command("forward 1 m")
    assert seeker.stops == 1 and drive.goals[-1]["distance_m"] == 1.0
    assert st["mission"]["runner"] == "goal" and st["state"] == "running"
    drive.active = False
    drive.state = "reached"
    assert nav.status()["state"] == "arrived"
    assert nav.status()["summary"] == "Done: forward 1 m."
    nav.command("back 0.5 m")
    assert drive.goals[-1]["reverse_m"] == 0.5
    nav.command("turn left 45")
    assert drive.goals[-1]["turn_deg"] == 45
    st = nav.command("stop")
    assert st["state"] == "stopped" and drive.cancelled >= 1
    assert "Stopped" in st["summary"]
    seeker.state = "failed_not_found"
    nav.command("go to the unicorn")
    seeker.state = "failed_not_found"
    assert nav.status()["summary"] == "Could not get to unicorn: failed not found."


def test_http_navigate(tmp_path):
    nav, drive, trip, seeker, a, b = _nav(tmp_path)
    drive.trip, drive.follower, drive.seeker = trip, None, seeker
    drive.clearance = lambda: None
    drive.current_pose = lambda: None
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
        assert call("/navigate")[1]["state"] == "idle"
        assert call("/navigate", {})[0] == 400
        code, body = call("/navigate", {"command": "go to the dresser"})
        assert code == 202 and body["status"]["mission"]["runner"] == "seek"
        code, body = call("/navigate/stop", {})
        assert code == 200 and body["status"]["state"] == "stopped"
        code, body = call("/stop", {})
        assert code == 200
    finally:
        server.stop()


def test_split_steps_and_filler():
    assert split_steps("turn around and navigate to the window") == [
        "turn around", "navigate to the window"]
    assert split_steps("back up 1 m then turn left then go to the dresser") == [
        "back up 1 m", "turn left", "go to the dresser"]
    assert split_steps("go to the red chair and the table") == [
        "go to the red chair and the table"]
    steps = split_steps("turn around, and way into the distance, there is a "
                        "black window curtain. Navigate to it.")
    assert steps[0] == "turn around"
    assert parse_command(steps[1], [])["target"] == "black window curtain"


def test_multi_step_command_runs_steps_in_order(tmp_path):
    nav, drive, trip, seeker, a, b = _nav(tmp_path)
    st = nav.command("turn around and go to the dresser")
    assert st["state"] == "running" and st["steps_left"] == 1
    assert drive.goals[-1]["turn_deg"] == 180
    assert st["summary"].startswith("Step 1 of 2")
    nav.tick()                                    # still turning
    assert not seeker.seeks
    drive.active = False
    drive.state = "reached"
    assert nav.status()["state"] == "running"     # a step remains
    nav.tick()                                    # step 2 starts
    assert seeker.seeks == ["the dresser"]
    st = nav.status()
    assert st["summary"].startswith("Step 2 of 2") and st["steps_left"] == 0
    seeker.state = "arrived"
    assert nav.status()["state"] == "arrived"
    assert nav.status()["summary"] == "Step 2 of 2: Arrived at dresser."


def test_failed_step_drops_the_rest(tmp_path):
    nav, drive, trip, seeker, a, b = _nav(tmp_path)
    nav.command("forward 1 m then go to the dresser")
    drive.active = False
    drive.state = "aborted_blocked"
    nav.tick()
    assert nav.status()["state"] == "aborted_blocked"
    assert not seeker.seeks and nav.status()["steps_left"] == 0


def test_blocked_turn_backs_up_and_retries_once(tmp_path):
    nav, drive, trip, seeker, a, b = _nav(tmp_path)
    nav.command("turn around then go to the dresser")
    drive.active = False
    drive.state = "aborted_blocked"
    assert nav.status()["state"] == "running"       # recovery pending
    nav.tick()
    assert drive.goals[-1]["reverse_m"] == 0.4
    drive.active = False
    drive.state = "reached"
    nav.tick()                                      # retry the turn
    assert drive.goals[-1]["turn_deg"] == 180
    drive.active = False
    drive.state = "reached"
    nav.tick()                                      # turn done -> step 2
    assert seeker.seeks == ["the dresser"]
    # Blocked again after the retry: the command fails.
    nav.command("turn left")
    drive.active = False; drive.state = "aborted_blocked"; nav.tick()
    drive.active = False; drive.state = "reached"; nav.tick()
    drive.active = False; drive.state = "aborted_blocked"; nav.tick()
    assert nav.status()["state"] == "aborted_blocked"
