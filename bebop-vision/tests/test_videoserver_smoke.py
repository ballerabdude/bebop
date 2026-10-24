"""Videoserver HTTP smoke tests.

The 2026-09-09 trim orphaned _body inside _frame and dropped render_depth
while the depth branch still called it — every request returned empty
replies. These tests exercise the real handler over HTTP so a handler
break fails loudly.
"""

import json
import threading
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
