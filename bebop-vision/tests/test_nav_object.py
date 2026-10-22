"""Object-goal seeker tests (scripted VLM, fake drive, synthetic depth)."""

import json
import math
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from bebop_vision.nav import ObjectGoalConfig, ObjectSeeker, parse_locate_reply
from bebop_vision.nav.object_goal import strip_hints
from bebop_vision.nav.synth import small_geometry
from bebop_vision.places import StaticCapture, View

COLOR = {"color_fx": 610.5, "color_fy": 610.4, "color_cx": 641.3,
         "color_cy": 406.0, "color_width": 1280, "color_height": 800}


def test_parse_locate_reply():
    d = parse_locate_reply('Sure: {"found": true, "box": [0.1, 0.2, 0.3, 0.5], '
                           '"confidence": 0.8, "note": "dark dresser"}')
    assert d["found"] and d["box"] == [0.1, 0.2, 0.3, 0.5]
    assert d["confidence"] == 0.8 and d["note"] == "dark dresser"
    d = parse_locate_reply('{"found": false, "box": null}')
    assert not d["found"] and d["box"] is None
    with pytest.raises(ValueError):
        parse_locate_reply("no json here")
    with pytest.raises(ValueError):
        parse_locate_reply('{"found": true, "box": [0.5, 0.5, 0.2, 0.9]}')
    # Pixel boxes are converted when the sent image size is known.
    d = parse_locate_reply('{"found": true, "box": [64, 40, 128, 100]}', (640, 400))
    assert d["box"] == [0.1, 0.1, 0.2, 0.25]
    with pytest.raises(ValueError):
        parse_locate_reply('{"found": true, "box": [64, 40, 128, 100]}')
    # Mixed: fractional x, pixel y (seen live), in sent-image pixels...
    d = parse_locate_reply('{"found": true, "box": [0.4, 200, 0.5, 300]}', (640, 400))
    assert d["box"] == [0.4, 0.5, 0.5, 0.75]
    # ...or in original-frame pixels (also seen live: 407..528 of 800 rows).
    d = parse_locate_reply('{"found": true, "box": [0.42, 407, 0.49, 528]}',
                           (640, 400), (1280, 800))
    assert d["box"][1] == pytest.approx(407 / 800) and d["box"][3] == pytest.approx(528 / 800)
    # Out-of-range coordinates are clamped.
    d = parse_locate_reply('{"found": true, "box": [-0.1, 0.0, 1.4, 0.5]}')
    assert d["box"] == [0.0, 0.0, 1.0, 0.5]


class ScriptedVLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def complete(self, messages, *, max_tokens=120, temperature=0.0):
        self.calls += 1
        assert messages[0]["content"][0]["type"] == "image_url"
        return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]


class FakeDrive:
    def __init__(self):
        self.goals = []
        self.state = "idle"
        self.active = False
        self.estop = False
        self.cancelled = 0
        self.cmap = None

    def clearance(self):
        return self.cmap

    def status(self):
        return {"state": self.state, "active": self.active, "arming": False,
                "robot": {"estop_latched": self.estop}}

    def set_goal(self, *, bearing_deg=None, distance_m=None, turn_deg=None,
                 reverse_m=None, label=""):
        if turn_deg is not None:
            self.goals.append(("turn", round(turn_deg, 1)))
        elif reverse_m is not None:
            self.goals.append(("reverse", round(reverse_m, 2)))
        else:
            self.goals.append((round(bearing_deg, 1), round(distance_m, 2)))
        self.state, self.active = "drive", True

    def finish(self, state="reached"):
        self.state, self.active = state, False

    def cancel(self):
        self.cancelled += 1
        self.state, self.active = "stopped", False


def _jpeg():
    ok, buf = cv2.imencode(".jpg", np.full((800, 1280, 3), 90, np.uint8))
    return buf.tobytes()


class Cap(StaticCapture):
    """Static far view plus a rig whose far depth is a flat `depth_m` wall."""

    def __init__(self, depth_m, geom):
        super().__init__([View("far", _jpeg(), 1)])
        d = np.full((geom.height, geom.width), int(depth_m * 1000), np.uint16)
        self.rig = SimpleNamespace(cameras={"far": SimpleNamespace(
            read=lambda: SimpleNamespace(depth=d, depth_scale_mm=1.0))})


def box_reply(cx, cy=0.5, w=0.1, conf=0.9):
    return json.dumps({"found": True, "box": [cx - w / 2, cy - 0.05, cx + w / 2, cy + 0.05],
                       "confidence": conf, "note": "ok"})


def _seeker(replies, depth_m=3.0, **cfg):
    geom = small_geometry("far", 1.32, -17.4, width=848, height=480)
    # Real-sized intrinsics for the depth re-projection.
    geom = geom.__class__(**{**geom.__dict__, "fx": 414.0, "fy": 414.0,
                             "cx": 419.7, "cy": 242.7})
    drive = FakeDrive()
    vlm = ScriptedVLM(replies)
    s = ObjectSeeker(drive, vlm, Cap(depth_m, geom), geom, COLOR,
                     config=ObjectGoalConfig(**cfg), log=lambda *a: None)
    return s, drive, vlm


def test_locate_gives_bearing_and_range():
    s, drive, vlm = _seeker([box_reply(0.25)], depth_m=3.0)
    loc = s.locate("the dresser")
    assert loc["found"]
    # Box centre at 25 % of the width is left of centre -> positive bearing.
    expected = math.degrees(-math.atan2(0.25 * 1280 - 641.3, 610.5))
    assert loc["bearing_deg"] == pytest.approx(expected, abs=0.1)
    assert loc["range_m"] == pytest.approx(3.0, abs=0.01)
    assert loc["vlm_ms"] >= 0
    assert not drive.goals


def test_seek_steps_toward_object_and_arrives():
    # Range 2.6 -> 1.6 -> 1.0 as the fake depth shrinks per look.
    s, drive, vlm = _seeker([box_reply(0.5)], depth_m=2.6, standoff_m=0.9,
                            step_m=1.0)
    s.seek("the dresser")
    s.tick()
    assert drive.goals[0][1] == 1.0 and abs(drive.goals[0][0]) < 0.2
    assert s.status()["legs"] == 1
    s.tick()                                   # goal still active: wait
    assert len(drive.goals) == 1
    drive.finish("reached")
    s.capture.rig.cameras["far"].read().depth[:] = 1600
    s.tick()
    assert drive.goals[-1][1] == 0.7           # remaining 1.6 - 0.9
    drive.finish("reached")
    s.capture.rig.cameras["far"].read().depth[:] = 1000
    s.tick()
    st = s.status()
    assert st["state"] == "arrived" and st["legs"] == 2
    assert st["history"][-1]["last"]["range_m"] == pytest.approx(1.0)


def test_seek_without_depth_uses_default_step_and_not_found_fails():
    s, drive, vlm = _seeker([box_reply(0.5)], depth_m=0.0, default_range_m=1.5,
                            standoff_m=0.9)
    s.seek("the lamp")
    s.tick()
    assert drive.goals[0][1] == 0.6            # 1.5 - 0.9, no depth
    s2, drive2, _ = _seeker(['{"found": false, "box": null}'], max_misses=2,
                            scan=False)
    s2.seek("the unicorn")
    s2.tick(); assert s2.status()["state"] == "seeking"
    s2.tick(); assert s2.status()["state"] == "failed_not_found"
    assert not drive2.goals


def test_miss_scans_hinted_side_first_then_finds():
    replies = ['{"found": false, "box": null}', '{"found": false, "box": null}',
               box_reply(0.5), box_reply(0.5)]
    s, drive, vlm = _seeker(replies, depth_m=2.0)
    s.seek("the mirror on the left")
    s.tick()                                     # miss -> turn +45 (left first)
    assert drive.goals == [("turn", 45.0)] and s.status()["scanning"]
    s.tick()                                     # goal active: wait
    drive.finish("reached")
    s.tick()                                     # miss -> next left heading +90
    assert drive.goals[-1] == ("turn", 45.0)     # delta from 45 to 90
    drive.finish("reached")
    s.tick()                                     # found: drive toward it
    assert drive.goals[-1][0] != "turn" and not s.status()["scanning"]
    assert s.status()["state"] == "seeking"


def test_scan_order_by_hint():
    s, drive, vlm = _seeker([box_reply(0.5)])
    assert s._plan_scan("the mirror on the left")[:2] == [45, 90]
    assert s._plan_scan("the door on the right")[:2] == [-45, -90]
    assert s._plan_scan("the chair behind you")[0] == 180
    assert s._plan_scan("the lamp")[:2] == [45, -45]


def test_scan_exhausted_fails():
    s, drive, vlm = _seeker(['{"found": false, "box": null}'], max_misses=2,
                            scan_headings_deg=(45,))
    s.seek("the yeti")
    s.tick(); drive.finish("reached")            # one scan turn
    s.tick()                                     # miss again, no scans left
    assert s.status()["state"] == "failed_not_found"



def test_seek_handles_blocked_estop_and_stop():
    s, drive, vlm = _seeker([box_reply(0.5)], depth_m=4.0, max_blocked=2,
                            max_detours=0)
    s.seek("the chair")
    s.tick(); drive.finish("aborted_blocked")
    s.tick()
    assert s.status()["state"] == "failed_blocked"
    s.seek("the chair")
    drive.estop = True
    s.tick()
    assert s.status()["state"] == "failed_estop"
    drive.estop = False
    s.seek("the chair")
    with pytest.raises(RuntimeError):
        s.seek("the chair")
    s.stop()
    assert s.status()["state"] == "stopped" and drive.cancelled == 1


def test_hints_are_stripped_from_what_the_model_is_asked():
    assert strip_hints("the mirror on the left") == "the mirror"
    assert strip_hints("the door to the right side") == "the door"
    assert strip_hints("the chair behind you") == "the chair"
    assert strip_hints("the lamp") == "the lamp"
    assert strip_hints("on the left") == "on the left"      # nothing else to keep


def test_seen_object_gets_a_relook_before_scanning():
    replies = [box_reply(0.5), '{"found": false, "box": null}',
               box_reply(0.5), box_reply(0.5)]
    s, drive, vlm = _seeker(replies, depth_m=3.0)
    s.seek("the mirror on the left")
    assert s.status()["asked"] == "the mirror"
    s.tick(); drive.finish("reached")            # seen, leg 1
    s.tick()                                     # miss: re-look, no turn
    assert s.status()["misses"] == 1 and not s.status()["scanning"]
    assert all(g[0] != "turn" for g in drive.goals)
    s.tick()                                     # seen again, leg 2
    assert s.status()["misses"] == 0 and s.status()["legs"] == 2


def test_blocked_leg_detours_into_open_sector_then_looks_again():
    s, drive, vlm = _seeker([box_reply(0.5)], depth_m=4.0)
    drive.cmap = {"bearings_deg": [60, 30, 0, -30, -60],
                  "clearance_m": [0.5, 2.2, 0.5, 1.4, 0.5]}
    s.seek("the window")
    s.tick()                                    # leg 1 toward the object
    drive.finish("aborted_blocked")
    s.tick()                                    # detour: most open sector (+30)
    assert drive.goals[-1] == (30.0, 1.0) and s.status()["detouring"]
    assert s.status()["detours"] == 1
    drive.finish("reached")
    s.tick()                                    # looks again, drives on
    assert drive.goals[-1][0] != "turn" and not s.status()["detouring"]
    assert s.status()["state"] == "seeking"


def test_blocked_with_nothing_open_backs_up_then_fails_after_max_detours():
    s, drive, vlm = _seeker([box_reply(0.5)], depth_m=4.0, max_detours=2)
    drive.cmap = {"bearings_deg": [30, 0, -30], "clearance_m": [0.4, 0.3, 0.4]}
    s.seek("the window")
    s.tick(); drive.finish("aborted_blocked")
    s.tick()
    assert drive.goals[-1] == ("reverse", 0.4)
    drive.finish("reached")
    s.tick(); drive.finish("aborted_blocked")  # leg again, blocked again
    s.tick()
    assert drive.goals[-1] == ("reverse", 0.4) and s.status()["detours"] == 2
    drive.finish("reached")
    s.tick(); drive.finish("aborted_blocked")
    s.tick()
    assert s.status()["state"] == "failed_blocked"


def test_blocked_scan_turn_backs_up_and_retries_the_turn():
    replies = ['{"found": false, "box": null}']
    s, drive, vlm = _seeker(replies, depth_m=4.0)
    s.seek("the lamp")
    s.tick()                                    # miss -> scan turn +45
    assert drive.goals[-1] == ("turn", 45.0)
    drive.finish("aborted_blocked")
    s.tick()                                    # unblock: reverse, scan kept
    assert drive.goals[-1] == ("reverse", 0.4)
    drive.finish("reached")
    s.tick()                                    # miss -> the same +45 retried
    assert drive.goals[-1] == ("turn", 45.0)
