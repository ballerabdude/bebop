"""Dashboard core unit tests on synthetic data.

Builds a tmp session tree (tiny label/depth npz + JPEGs + manifest), then
exercises the DatasetDashboard methods directly. No hardware, no real
dataset dependency. The ray-LUT loader is injected as a fake so the
overlay path is tested without config/raylut_near.npz.
"""

import base64
import io
import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from dashboard_core import DatasetDashboard, build_overlay  # noqa: E402

S20 = 1788716672926769002          # one tick stamp; second tick = +100e6
STRIDE = 4
COLOR_W, COLOR_H = 64, 48          # tiny color size, divisible by STRIDE


def _tiny_grid(cls_value):
    """60x60 uint8 grid filled with one class value (0/1/2)."""
    return np.full((60, 60), cls_value, np.uint8)


def _fake_lut():
    """Minimal ray LUT for the tiny color size: (H/4)x(W/4)=192 pixels.

    land_cells map each stride-4 color pixel to flat BEV cell 0..3599;
    -1 marks pixels with no landing (sky) — the overlay must skip those.
    """
    n_px = (COLOR_H // STRIDE) * (COLOR_W // STRIDE)
    lc = np.full(n_px, -1, np.int32)
    lc[n_px // 4: n_px // 2] = 0             # quarter land in cell 0
    lc[n_px // 2:] = 61                      # half land in cell 61
    # the first quarter stays -1 (sky): overlay must be transparent there
    return dict(land_cells=lc, n_px=n_px)


def _tiny_jpeg(value):
    ok, jpg = cv2.imencode(".jpg", np.full((COLOR_H, COLOR_W, 3), value,
                                           np.uint8))
    assert ok
    return jpg.tobytes()


@pytest.fixture
def dash(tmp_path):
    """Dashboard against a synthetic session with 3 ticks.

    Tick stamps S20, S20+100e6, S20+200e6. Labels npz carry the teacher
    keys (fused + bookkeeping); depth npz has near/far uint16 (480, 848)
    with a few invalid (0) pixels; color JPEGs are tiny solid-value
    images; the manifest has one row per tick.
    """
    sess = tmp_path / "navd_session_test"
    for sub in ("labels", "color", "color_far", "depth"):
        (sess / sub).mkdir(parents=True)
    rows = []
    for k, off in enumerate((0, 100_000_000, 200_000_000)):
        stamp = S20 + off
        s20 = f"{stamp:020d}"
        np.savez_compressed(
            sess / "labels" / f"{s20}.npz",
            fused=_tiny_grid(k % 3),
            teacher=_tiny_grid(1),
            sem_near=_tiny_grid(0), sem_far=_tiny_grid(0),
            floor_near=_tiny_grid(1), floor_far=_tiny_grid(1),
            disagree=_tiny_grid(1 if k == 1 else 0),
            unconfirmed=_tiny_grid(2 if k == 2 else 0))
        depth = np.full((480, 848), 1500, np.uint16)
        depth[:10, :10] = 0                        # invalid mm -> black
        np.savez_compressed(sess / "depth" / f"{s20}.npz",
                            near=depth, far=depth)
        (sess / "color" / f"{s20}.jpg").write_bytes(_tiny_jpeg(40 + k))
        (sess / "color_far" / f"{s20}.jpg").write_bytes(_tiny_jpeg(90))
        rows.append({"stamp_ns": stamp,
                     "cmd_vel": {"vx": 0.1 * k, "wz": 0.0},
                     "odom": {"x": 0.0, "y": 0.0, "theta": 0.0},
                     "goal": {"type": "none"}})
    with open(sess / "manifest.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    (tmp_path / "stray_dir").mkdir()               # no labels -> ignored
    return DatasetDashboard(tmp_path, load_lut=lambda role: _fake_lut())


def test_sessions_listing(dash):
    out = dash.list_sessions()
    assert len(out) == 1
    s = out[0]
    assert s["name"] == "navd_session_test"
    assert s["ticks"] == 3
    assert s["hand"] == 0


def test_sessions_hand_count_and_tick_stats(dash):
    rows = dash.list_ticks("navd_session_test")
    assert [r["stamp_ns"] for r in rows] == [S20 + o for o in
                                             (0, 100_000_000, 200_000_000)]
    # "stamp" must round-trip EXACTLY as a string: 19-digit stamp_ns
    # rounds in the browser when passed as a JSON number (…002 -> …000),
    # which made every tick fetch 404 (fixed 2026-09-07).
    assert [r["stamp"] for r in rows] == \
        [f"{S20 + o:020d}" for o in (0, 100_000_000, 200_000_000)]
    assert all(r["hand"] is False for r in rows)
    # fused grids were filled with class k%3 -> that class fraction is 1.0
    assert rows[0]["f0"] == 1.0 and rows[1]["f1"] == 1.0 \
        and rows[2]["f2"] == 1.0
    # tick 1 has disagree=1 everywhere, tick 2 unconfirmed=2 everywhere
    assert rows[1]["disagree_cells"] == 3600
    assert rows[2]["unconfirmed_cells"] == 3600
    assert rows[0]["disagree_cells"] == 0


def test_tick_payload_grids_and_images(dash):
    p = dash.tick_payload("navd_session_test", S20)
    g = p["grids"]
    # fixture fills fused with class 0 on tick 0 and every bookkeeping
    # grid with the value noted in the fixture (sem 0, floor/teacher 1,
    # disagree 0, unconfirmed 0)
    expect = {"fused": 0, "teacher": 1, "sem_near": 0, "sem_far": 0,
              "floor_near": 1, "floor_far": 1, "disagree": 0,
              "unconfirmed": 0}
    for k, v in expect.items():
        arr = np.asarray(g[k], np.uint8)
        assert arr.shape == (60, 60)
        assert np.unique(arr).tolist() == [v], k
    assert "hand" not in g
    # manifest row round-trips (cmd_vel from the fixture)
    assert p["manifest"]["cmd_vel"]["vx"] == 0.0
    # color is base64 JPEG passthrough -> decodes back to the tiny image
    jpg = base64.b64decode(p["color_near"])
    img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    assert img.shape == (COLOR_H, COLOR_W, 3)
    # depth renders to a decodable PNG at the repo's half-res view
    png = base64.b64decode(p["depth_near"])
    dep = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
    assert dep.shape == (240, 424, 3)
    assert p["depth_far"] is not None


def test_tick_payload_missing_files(dash, tmp_path):
    """A tick without depth/manifest still returns a valid payload."""
    stamp = S20 + 300_000_000
    s20 = f"{stamp:020d}"
    sess = tmp_path / "navd_session_test"
    np.savez_compressed(sess / "labels" / f"{s20}.npz", fused=_tiny_grid(1))
    p = dash.tick_payload("navd_session_test", stamp)
    assert np.asarray(p["grids"]["fused"]).shape == (60, 60)
    assert p["depth_near"] is None
    assert p["manifest"] is None


def test_save_hand_preserves_keys_and_audits(dash, tmp_path):
    stamp = S20
    s20 = f"{stamp:020d}"
    sess = tmp_path / "navd_session_test"
    lab = sess / "labels" / f"{s20}.npz"
    with np.load(lab) as z:
        keys_before = set(z.files)
    hand = np.where((np.mgrid[0:60, 0:60][0] < 30), 1, 0).astype(np.uint8)
    ok, msg = dash.save_hand("navd_session_test", stamp, hand.tolist())
    assert ok, msg
    with np.load(lab) as z:
        keys_after = set(z.files)
        stored = z["hand"]
    # every teacher key preserved untouched, hand added
    assert keys_before <= keys_after
    assert "hand" in keys_after
    assert np.array_equal(stored, hand)
    assert stored.dtype == np.uint8
    # audit line appended with counts before (fused all 0) / after
    lines = open(sess / "hand_edits.jsonl").read().strip().splitlines()
    assert len(lines) == 1
    a = json.loads(lines[0])
    assert a["stamp_ns"] == stamp
    assert a["class_counts_before"] == {"0": 3600, "1": 0, "2": 0}
    assert a["class_counts_after"] == {"0": 1800, "1": 1800, "2": 0}
    assert a["changed_cells"] == 1800
    assert "ts" in a


def test_save_hand_flat_and_invalid(dash):
    ok, _ = dash.save_hand("navd_session_test", S20, [1] * 3600)
    assert ok
    ok, msg = dash.save_hand("navd_session_test", S20, [1] * 59)
    assert not ok
    ok, msg = dash.save_hand("navd_session_test", S20,
                             np.full((60, 60), 5, np.uint8).tolist())
    assert not ok
    ok, msg = dash.save_hand("no_such_session", S20, [1] * 3600)
    assert not ok


def test_clear_hand_deletes_key(dash, tmp_path):
    stamp = S20 + 100_000_000
    s20 = f"{stamp:020d}"
    sess = tmp_path / "navd_session_test"
    lab = sess / "labels" / f"{s20}.npz"
    ok, _ = dash.save_hand("navd_session_test", stamp,
                           _tiny_grid(1).tolist())
    assert ok
    ok, _ = dash.clear_hand("navd_session_test", stamp)
    assert ok
    with np.load(lab) as z:
        assert "hand" not in z.files            # key deleted
        assert set(z.files) >= {"fused", "teacher"}   # teacher intact
        assert np.array_equal(z["fused"], _tiny_grid(1))
    lines = open(sess / "hand_edits.jsonl").read().strip().splitlines()
    assert len(lines) == 2                      # save + clear audited
    a = json.loads(lines[1])
    assert a["class_counts_after"] == {"0": 0, "1": 3600, "2": 0}
    # clearing twice is a no-op success
    ok, msg = dash.clear_hand("navd_session_test", stamp)
    assert ok and msg == "no hand to clear"


def test_overlay_with_injected_lut(dash):
    """Fake LUT injected via the constructor — no config dependency."""
    out = dash.overlay_payload("navd_session_test", S20, src="fused",
                               alpha=0.5)
    assert "error" not in out, out.get("error")
    assert out["role"] == "near"
    # RGBA overlay decodes and has opaque pixels where the LUT landed
    bgra = cv2.imdecode(
        np.frombuffer(base64.b64decode(out["overlay"]), np.uint8),
        cv2.IMREAD_UNCHANGED)
    assert bgra.shape == (COLOR_H, COLOR_W, 4)
    assert bgra[..., 3].max() == 255 and bgra[..., 3].min() == 0
    # blend decodes to the same-size color image
    blend = cv2.imdecode(
        np.frombuffer(base64.b64decode(out["blend"]), np.uint8),
        cv2.IMREAD_COLOR)
    assert blend.shape == (COLOR_H, COLOR_W, 3)


def test_overlay_uses_hand_when_present(dash):
    dash.save_hand("navd_session_test", S20, _tiny_grid(1).tolist())
    out = dash.overlay_payload("navd_session_test", S20)   # src=auto default
    assert "error" not in out
    assert out["src"] == "hand"
    # explicit fused request still works alongside a stored hand
    out = dash.overlay_payload("navd_session_test", S20, src="fused")
    assert "error" not in out


def test_overlay_errors(dash):
    out = dash.overlay_payload("navd_session_test", S20, src="hand")
    assert out["error"] == f"no hand grid for stamp {S20}"
    # no LUT available -> error, not crash
    plain = DatasetDashboard(dash.root, load_lut=lambda role: None)
    out = plain.overlay_payload("navd_session_test", S20)
    assert "error" in out


def test_build_overlay_invalid_shape():
    lut = dict(land_cells=np.full(37, -1, np.int32), n_px=37)
    bgra, vmask = build_overlay(lut, _tiny_grid(1), COLOR_H, COLOR_W)
    assert bgra is None and vmask is None


def test_session_traversal_refused(dash):
    with pytest.raises(FileNotFoundError):
        dash.list_ticks("../elsewhere")
    dash.tick_payload("navd_session_test", S20)  # sanity: real session ok


def test_stray_directory_ignored(dash):
    names = [s["name"] for s in dash.list_sessions()]
    assert names == ["navd_session_test"]


def test_real_palette_module_shapes():
    """Module-level invariants the UI relies on (no dataset needed)."""
    from dashboard_core import (CLASS_BGR, CLASS_RGB_HEX, DEPTH_VIEW,
                                   GRID, N_CELLS, class_palette_bgr)
    assert GRID == 60 and N_CELLS == 3600
    assert DEPTH_VIEW == (424, 240)
    assert sorted(CLASS_BGR) == [0, 1, 2] and sorted(CLASS_RGB_HEX) == [0, 1, 2]
    pal = class_palette_bgr()
    assert pal.shape == (N_CELLS + 1, 3)
    for cls, bgr in CLASS_BGR.items():
        assert pal[cls].tolist() == list(bgr)
