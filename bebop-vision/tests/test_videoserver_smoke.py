"""Videoserver HTTP smoke tests (post-BEV-excision guard).

The 2026-09-09 trim orphaned _body inside _frame and dropped render_depth
while the depth branch still called it — every request returned empty
replies. These tests exercise the real handler over HTTP so a handler
break fails loudly.
"""

import threading
import time
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
    assert "bev" not in STREAMS
    for stream in STREAMS:
        with _get(server._httpd.server_address[1],
                  f"/snapshot?stream={stream}") as r:
            assert r.status == 200
            body = r.read()
            assert body[:2] == b"\xff\xd8"      # JPEG SOI
            assert len(body) > 1000
