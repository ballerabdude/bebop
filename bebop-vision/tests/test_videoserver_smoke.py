"""Videoserver HTTP smoke tests.

The 2026-09-09 trim orphaned _body inside _frame and dropped render_depth
while the depth branch still called it — every request returned empty
replies. These tests exercise the real handler over HTTP so a handler
break fails loudly.
"""

import json
import time
import urllib.error
import urllib.request

import numpy as np
import pytest

from bebop_vision.orbbec import StampedFrame
from bebop_vision.videoserver import STREAMS, VideoServer


def _frame(role, stamp_us):
    depth = np.full((480, 848), 1500, np.uint16)
    import cv2
    ok, jpg = cv2.imencode(".jpg", np.zeros((800, 1280, 3), np.uint8),
                           [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return StampedFrame(depth=depth, stamp_us=stamp_us,
                        recv_ts=time.monotonic(), width=848, height=480,
                        fps=15.0, color=None, color_jpeg=jpg.tobytes(),
                        serial="S", role=role)


class FakeCam:
    def __init__(self, role):
        self.role = role
        self._n = 0

    def read(self):
        self._n += 1
        return _frame(self.role, self._n)


class FakeRig:
    def __init__(self):
        self.cameras = {r: FakeCam(r) for r in ("near", "far")}


@pytest.fixture
def server():
    v = VideoServer(FakeRig(), port=0)
    v.start()
    yield v
    v.stop()


def _get(port, path, timeout=5):
    return urllib.request.urlopen(f"http://127.0.0.1:{port}{path}",
                                  timeout=timeout)


def test_all_streams_serve_jpegs(server):
    assert STREAMS == ("color_near", "color_far", "depth_near", "depth_far")
    for stream in STREAMS:
        with _get(server._httpd.server_address[1],
                  f"/snapshot?stream={stream}") as r:
            assert r.status == 200
            body = r.read()
            assert body[:2] == b"\xff\xd8"      # JPEG SOI
            assert len(body) > 100


def _post(port, path, payload):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    return urllib.request.urlopen(req, timeout=5)


def test_notes_routes_roundtrip():
    from bebop_vision.session_notes import SessionNotes
    notes = SessionNotes()
    v = VideoServer(FakeRig(), port=0, notes=notes)
    v.start()
    try:
        port = v._httpd.server_address[1]
        # a marker with no segment recording is refused, not silently lost
        with pytest.raises(urllib.error.HTTPError) as ei:
            _post(port, "/event", {"kind": "exclude"})
        assert ei.value.code == 409
        notes.begin_segment("segment")
        with _post(port, "/intent", {"text": "to the kitchen", "from": "office",
                                     "to": "kitchen"}) as r:
            assert r.status == 200
            assert json.loads(r.read())["intent"]["to"] == "kitchen"
        with _post(port, "/event", {"kind": "obstacle", "note": "sock at 1 m"}) as r:
            assert r.status == 200
        with _get(port, "/notes") as r:
            snap = json.loads(r.read())
        assert snap["intent"]["text"] == "to the kitchen"
        assert snap["events_pending"] == 1
        with pytest.raises(urllib.error.HTTPError) as ei:
            _post(port, "/event", {"kind": "not a token!"})
        assert ei.value.code == 400
        with _post(port, "/intent", {"clear": True}) as r:
            assert json.loads(r.read())["intent"]["text"] == ""
    finally:
        v.stop()


def test_notes_routes_without_notes_are_503(server):
    port = server._httpd.server_address[1]
    with pytest.raises(urllib.error.HTTPError) as ei:
        _post(port, "/intent", {"text": "x"})
    assert ei.value.code == 503


def test_notes_routes_reject_missing_and_chunked_bodies():
    import http.client
    from bebop_vision.session_notes import SessionNotes
    notes = SessionNotes()
    notes.set_intent("keep me")
    v = VideoServer(FakeRig(), port=0, notes=notes)
    v.start()
    try:
        port = v._httpd.server_address[1]
        for headers, body in (({"Content-Length": "0"}, b""),
                              ({"Transfer-Encoding": "chunked"}, b"0\r\n\r\n"),
                              ({"Content-Length": "5"}, b"[1,2]"),
                              ({"Content-Length": "4"}, b"{oop")):
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            conn.request("POST", "/intent", body=body, headers=headers)
            resp = conn.getresponse()
            assert resp.status == 400, (headers, body, resp.status)
            resp.read(); conn.close()
        assert notes.intent().text == "keep me"     # nothing was cleared
    finally:
        v.stop()


def test_runs_route_serves_the_run_log(tmp_path):
    from bebop_vision.session_notes import SessionNotes
    notes = SessionNotes(state_path=tmp_path / "intent.json")
    v = VideoServer(FakeRig(), port=0, notes=notes)
    v.start()
    try:
        port = v._httpd.server_address[1]
        with _get(port, "/runs") as r:
            assert json.loads(r.read()) == {"runs": []}
        notes.set_intent("office to bedroom", task="go_to_place", target="master bedroom")
        notes.end_run("arrived", note="by the dresser")
        with _get(port, "/runs") as r:
            runs = json.loads(r.read())["runs"]
        assert len(runs) == 1 and runs[0]["outcome"] == "arrived"
        assert runs[0]["end_note"] == "by the dresser"
        with _get(port, f"/runs?since={runs[0]['ended_ns']}") as r:
            assert json.loads(r.read())["runs"] == []
        with _get(port, "/runs?since=junk") as r:          # bad since: everything
            assert len(json.loads(r.read())["runs"]) == 1
    finally:
        v.stop()


def test_intent_run_routes():
    from bebop_vision.session_notes import SessionNotes
    notes = SessionNotes()
    v = VideoServer(FakeRig(), port=0, notes=notes)
    v.start()
    try:
        port = v._httpd.server_address[1]

        def post(payload):
            with _post(port, "/intent", payload) as r:
                return json.loads(r.read())

        def refused(payload):
            with pytest.raises(urllib.error.HTTPError) as ei:
                _post(port, "/intent", payload)
            return ei.value.code

        assert refused({"text": "x", "mode": "continue"}) == 409   # no run yet
        for bad in ({"text": "x", "task": "dance"}, {"text": "x", "mode": "again"},
                    {"end": "teleported"}, {"repeat": "r_unknown"}):
            assert refused(bad) == 400, bad
        a = post({"text": "to the kitchen", "from": "office", "to": "kitchen",
                  "task": "go_to_place", "target": "Kitchen"})["intent"]
        assert a["run_id"].startswith("r") and a["task"] == "go_to_place"
        # continue changes only the keys sent
        b = post({"to": "pantry", "mode": "continue"})["intent"]
        assert (b["run_id"], b["text"], b["to"]) == (a["run_id"], "to the kitchen", "pantry")
        # ending works with nothing recording
        out = post({"end": "interrupted", "note": "estop"})
        assert out["event_recorded"] is False and out["run"]["interruptions"] == 1
        assert out["intent"]["run_id"] == a["run_id"]
        notes.begin_segment("seg")
        out = post({"end": "arrived"})
        assert out["ok"] and out["event_recorded"] is True
        assert out["run"]["outcome"] == "arrived" and out["intent"]["run_id"] == ""
        c = post({"repeat": a["run_id"]})["intent"]
        assert c["run_id"] != a["run_id"] and c["to"] == "pantry"
        with _get(port, "/notes") as r:
            snap = json.loads(r.read())
        assert [x["run_id"] for x in snap["recent"]] == [c["run_id"], a["run_id"]]
        assert snap["intent"]["run_id"] == c["run_id"] and snap["events_pending"] == 1
        assert post({"clear": True})["intent"]["run_id"] == ""
        # POST /event keeps its 409 once the segment closes
        notes.end_segment("seg")
        with pytest.raises(urllib.error.HTTPError) as ei:
            _post(port, "/event", {"kind": "note"})
        assert ei.value.code == 409
    finally:
        v.stop()


def test_intent_resume_unfinished_and_stops_routes():
    from bebop_vision.session_notes import SessionNotes
    notes = SessionNotes()
    v = VideoServer(FakeRig(), port=0, notes=notes)
    v.start()
    try:
        port = v._httpd.server_address[1]

        def post(payload):
            with _post(port, "/intent", payload) as r:
                return json.loads(r.read())

        def notes_snap():
            with _get(port, "/notes") as r:
                return json.loads(r.read())

        with pytest.raises(urllib.error.HTTPError) as ei:      # no run open
            _post(port, "/intent", {"resume": True, "note": "x"})
        assert ei.value.code == 409
        a = post({"text": "to the kitchen", "task": "go_to_place",
                  "target": "Kitchen"})["intent"]
        assert (a["stops"], a["resumes"], a["last_stop"]) == (0, 0, None)
        notes.begin_segment("seg")
        notes.note_stop("estop_operator", "operator: app button")
        notes.end_segment("seg")
        snap = notes_snap()
        assert snap["intent"]["stops"] == 1
        assert snap["intent"]["last_stop"]["cause"] == "estop_operator"
        assert snap["intent"]["last_stop"]["detail"] == "operator: app button"
        assert snap["recent"][0]["run_id"] == a["run_id"]
        assert snap["recent"][0]["last_stop"]["cause"] == "estop_operator"
        # resume while idle: held for the next segment
        out = post({"resume": True, "note": "all clear"})
        assert out["ok"] and out["run"]["resumes"] == 1 and out["held"] is True
        assert out["intent"]["run_id"] == a["run_id"] and out["intent"]["resumes"] == 1
        assert notes_snap()["events_held"] == 1
        # the operator declines: the run closes as unfinished with the note
        out = post({"end": "unfinished", "note": "ran out of time"})
        assert out["run"]["outcome"] == "unfinished" and out["intent"]["run_id"] == ""
        assert out["run"]["end_note"] == "ran out of time"
        snap = notes_snap()
        assert snap["recent"][0]["outcome"] == "unfinished"
        assert snap["recent"][0]["end_note"] == "ran out of time"
        assert snap["recent"][0]["stops"] == 1 and snap["recent"][0]["resumes"] == 1
        assert snap["events_held"] == 0
        # a clear closes as unfinished too
        b = post({"text": "loop", "task": "explore"})["intent"]
        post({"clear": True})
        row = notes_snap()["recent"][0]
        assert (row["run_id"], row["outcome"]) == (b["run_id"], "unfinished")
    finally:
        v.stop()
