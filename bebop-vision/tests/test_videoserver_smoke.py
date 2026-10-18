"""Videoserver HTTP smoke tests (post-BEV-excision guard).

The 2026-09-09 trim orphaned _body inside _frame and dropped render_depth
while the depth branch still called it — every request returned empty
replies. These tests exercise the real handler over HTTP so a handler
break fails loudly.
"""

import json
import threading
import time
import urllib.request

import cv2
import numpy as np
import pytest

from bebop_vision.orbbec import StampedFrame
from bebop_vision.videoserver import STREAMS, VideoServer

from test_master_bev import make_builder, scene


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


def _publish_bev(server):
    b = make_builder()
    frames = scene()
    grid = b.fuse({role: b.prepare(f) for role, f in frames.items()},
                  {"near": 0.01, "far": 0.01})
    ok, jpg = cv2.imencode(".jpg", np.zeros((48, 24, 3), np.uint8))
    assert ok
    server.publish_bev(grid, jpg.tobytes())
    return grid


def test_all_streams_serve_jpegs(server):
    assert "bev" in STREAMS
    _publish_bev(server)
    for stream in STREAMS:
        with _get(server._httpd.server_address[1],
                  f"/snapshot?stream={stream}") as r:
            assert r.status == 200
            body = r.read()
            assert body[:2] == b"\xff\xd8"      # JPEG SOI
            assert len(body) > 100


def test_bev_json_endpoint(server):
    grid = _publish_bev(server)
    with _get(server._httpd.server_address[1], "/bev") as r:
        assert r.status == 200
        assert r.headers.get_content_type() == "application/json"
        payload = json.loads(r.read())
    assert payload["rows"] == grid.rows
    assert payload["cols"] == grid.cols
    assert payload["encoding"] == "rle_pairs"
    assert payload["counts"]["unknown"] >= 0
    assert payload["contributions"]["far"]["points"] > 0
    assert payload["published_age_s"] >= 0.0


def test_register_replaces_and_closes_predecessor():
    """A reconnecting client must not leave the old session (and its
    NVENC pipeline + GPU fds) alive."""
    v = VideoServer(FakeRig(), port=0)
    closed = []

    class S:
        def __init__(self, name):
            self.name = name

        def close(self):
            closed.append(self.name)

    a, b = S("a"), S("b")
    v._register("color_near", a)
    v._register("color_near", b)
    assert closed == ["a"]
    assert v._sessions["color_near"] is b

    v._unregister("color_near", a)       # stale unregister is ignored
    assert v._sessions["color_near"] is b

    v._unregister("color_near", b)
    assert "color_near" not in v._sessions
