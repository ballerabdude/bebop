"""VideoServer BEV feed tests: renderer, publisher slot, HTTP routes."""

import math
import urllib.error
from types import SimpleNamespace
from urllib.request import urlopen

import numpy as np
import pytest

from bebop_vision.videoserver import (BEV_COLORS, STREAMS, VideoServer,
                                      _goal_bearing, render_bev)


def make_grid(occ=None, stamp_us=42, cell_m=0.05):
    if occ is None:
        occ = np.zeros((60, 60), np.uint8)
        occ[10:12, 20:22] = 1   # occupied blob ahead
        occ[30:32, 40:42] = 2   # hazard
        occ[45:47, 5:7] = 3     # inflated
    return SimpleNamespace(occ=occ, cell_m=cell_m, stamp_us=stamp_us)


def test_bev_is_a_stream():
    assert "bev" in STREAMS


def test_goal_bearing_extraction():
    assert _goal_bearing(None) is None
    assert _goal_bearing(SimpleNamespace(heading_rad=0.5)) == pytest.approx(0.5)
    assert _goal_bearing(SimpleNamespace(x=1.0, y=1.0)) == pytest.approx(
        math.pi / 4)
    assert _goal_bearing(object()) is None


def test_render_bev_shape_and_colors():
    img = render_bev(make_grid())
    assert img.shape == (480, 480, 3)
    # free background at the top-left corner
    assert (img[0:8, 0:8] == BEV_COLORS[0]).all()
    # col c of the grid renders at image col (59 - c): the horizontal
    # mirror keeps body +y (left) on the image left, camera-aligned
    # occupied blob at grid rows 10-11, cols 20-21
    assert (img[80:96, 304:320] == BEV_COLORS[1]).all()
    # hazard at grid rows 30-31, cols 40-41
    assert (img[240:256, 144:160] == BEV_COLORS[2]).all()
    # inflated at grid rows 45-46, cols 5-6
    assert (img[360:376, 424:440] == BEV_COLORS[3]).all()


def test_render_bev_goal_arrow():
    # straight ahead: the arrow runs up the center column from the robot
    # origin at the bottom edge (mirror-invariant bearing)
    img = render_bev(make_grid(), bearing_rad=0.0)
    white = np.all(img == 255, axis=-1)
    assert white.any()
    rows = np.where(white.any(axis=1))[0]
    assert rows.max() > 440          # anchored near the origin edge
    assert rows.min() < 440 - 8 * 12  # reaches ~1 m up the grid
    cols = np.where(white.any(axis=0))[0]
    assert abs(cols.mean() - 240) < 16
    # no goal -> no arrow anywhere
    bare = render_bev(make_grid())
    assert not np.all(bare == 255, axis=-1).any()


def test_render_bev_goal_bearing_left_is_left():
    # body +y (left) must render on the image LEFT (camera-aligned mirror)
    img = render_bev(make_grid(), bearing_rad=math.pi / 2)
    ys, xs = np.where(np.all(img == 255, axis=-1))
    assert xs.mean() < 240


def test_publish_bev_stores_encoded_frame():
    vserver = VideoServer(rig=None)   # the BEV feed never touches the rig
    assert vserver._bev is None
    vserver.publish_bev(make_grid(stamp_us=1234))
    slot = vserver._bev
    assert slot is not None
    assert slot.stamp_us == 1234
    assert slot.jpeg[:2] == b"\xff\xd8"   # JPEG SOI
    # None grid (fuse failure) keeps the previous frame
    vserver.publish_bev(None, goal=None)
    assert vserver._bev is slot
    # each publish replaces the slot object — the /video loop's frame
    # identity check relies on that
    vserver.publish_bev(make_grid(stamp_us=2345),
                        goal=SimpleNamespace(heading_rad=0.3))
    assert vserver._bev is not slot
    assert vserver._bev.stamp_us == 2345


def test_snapshot_503_before_first_grid():
    vserver = VideoServer(rig=None, port=0)
    vserver.start()
    port = vserver._httpd.server_address[1]
    try:
        with pytest.raises(urllib.error.HTTPError) as exc:
            urlopen(f"http://127.0.0.1:{port}/snapshot?stream=bev",
                    timeout=5)
        assert exc.value.code == 503
    finally:
        vserver.stop()


def test_video_route_serves_bev_multipart():
    vserver = VideoServer(rig=None, port=0)
    vserver.publish_bev(make_grid(stamp_us=777))
    vserver.start()
    port = vserver._httpd.server_address[1]
    try:
        with urlopen(f"http://127.0.0.1:{port}/video?stream=bev",
                     timeout=5) as resp:
            assert resp.headers.get_content_type() == \
                "multipart/x-mixed-replace"
            # first part: boundary, headers, blank line, start of the JPEG
            head = resp.read(128)
            header_blob, _, body_start = head.partition(b"\r\n\r\n")
            assert b"X-Timestamp-Us: 777" in header_blob
            assert body_start[:2] == b"\xff\xd8"
    finally:
        vserver.stop()
