"""Places: store, service, and HTTP contract tests (no torch, no camera).

The store and service are exercised with a deterministic fake encoder and a
static capture source, so these run anywhere numpy/cv2 do — the real DINOv3
encoder is covered separately on the robot.
"""

import copy
import json
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pytest

from bebop_vision.places import (
    PlaceStore,
    PlacesService,
    StaticCapture,
    View,
)
from bebop_vision.places.submap import decode_rle, encode_rle, rle_true_count


def _jpeg(value: int) -> bytes:
    ok, buf = cv2.imencode(".jpg", np.full((32, 32, 3), value, np.uint8))
    assert ok
    return buf.tobytes()


class FakeEncoder:
    """jpeg -> a preset vector (or a deterministic fallback)."""

    dim = 4
    device = "fake"

    def __init__(self, table=None):
        self.table = dict(table or {})

    @property
    def loaded(self):
        return True

    def encode_bgr(self, bgr):
        raise NotImplementedError

    def encode_jpeg(self, jpeg):
        if jpeg in self.table:
            v = np.asarray(self.table[jpeg], np.float32)
        else:
            v = np.asarray([float(sum(jpeg) % 997), 1.0, 0.0, 0.0], np.float32)
        return v / (np.linalg.norm(v) + 1e-12)


@pytest.fixture()
def store(tmp_path):
    s = PlaceStore(tmp_path / "places")
    yield s
    s.close()


def _snapshot(*, width=4, height=3, occupied=None, free=None,
              pose=(1.0, 2.0, 15.0), camera_height=1.3,
              walls=None, origin=(-0.5, -0.4), resolution=0.05):
    """A minimal but valid :9096 /structure payload (visual map frame)."""
    occ = (np.zeros((height, width), dtype=bool) if occupied is None
           else np.asarray(occupied, dtype=bool))
    fr = (np.zeros((height, width), dtype=bool) if free is None
          else np.asarray(free, dtype=bool))
    assert occ.shape == (height, width) and fr.shape == (height, width)
    return {
        "frame_id": "map",
        "pose_frame": "map_from_camera_optical",
        "gravity_source": "estimated_from_floor",
        "gravity": {
            "valid": True,
            "up": [0.0, -1.0, 0.0],
            "roll_deg": 0.0,
            "pitch_deg": 2.5,
            "inlier_ratio": 0.92,
        },
        "floor": {
            "valid": True,
            "normal": [0.0, -1.0, 0.0],
            "offset_m": camera_height,
            "camera_height_m": camera_height,
            "inliers": 800,
            "rmse_m": 0.004,
        },
        "grid": {
            "origin": [float(origin[0]), float(origin[1])],
            "resolution_m": float(resolution),
            "width": width,
            "height": height,
            "occupied_count": int(occ.sum()),
            "free_count": int(fr.sum()),
            "encoding": "rle_base64",
            "occupied_rle": encode_rle(occ),
            "free_rle": encode_rle(fr),
        },
        "walls": ([[-0.5, 0.0, 0.5, 0.0]] if walls is None else walls),
        "wall_cells": [],
        "path": [[0.0, 0.0]],
        "pose": {
            "valid": True,
            "x": pose[0],
            "y": pose[1],
            "yaw_deg": pose[2],
            "t_ns": 123456,
            "segment": 0,
        },
        "revision": 7,
    }


# --- store -----------------------------------------------------------------


def test_create_and_list(store):
    store.create_place("Kitchen")
    store.create_place("Lab")
    names = [p.name for p in store.list_places()]
    assert names == ["Kitchen", "Lab"]  # case-insensitive name sort
    assert store.count() == 2


def test_names_are_unique(store):
    store.create_place("Kitchen")
    with pytest.raises(ValueError):
        store.create_place("kitchen")
    with pytest.raises(ValueError):
        store.create_place("   ")


def test_get_by_id_or_name(store):
    p = store.create_place("Kitchen")
    assert store.get_place(p.id).name == "Kitchen"
    assert store.get_place("kitchen").id == p.id
    assert store.get_place("nope") is None


def test_rename_and_delete(store):
    p = store.create_place("Kitchen")
    store.create_place("Lab")
    renamed = store.rename_place(p.id, "Dining Room")
    assert renamed.name == "Dining Room"
    with pytest.raises(ValueError):
        store.rename_place(renamed.id, "Lab")
    assert store.delete_place("Dining Room") is True
    assert store.delete_place("Dining Room") is False
    assert store.count() == 1


def test_facts_tags_and_odom(store):
    p = store.create_place("Lab", odom=(1.5, -0.5, 0.2))
    store.add_fact(p.id, "  there is a red mug  ", source="operator")
    store.add_fact(p.id, "vision saw a desk", source="vision")
    store.add_tag(p.id, "desk", confidence=0.8, source="vision")
    got = store.get_place(p.id)
    assert got.odom == (1.5, -0.5, 0.2)
    assert [f.text for f in got.facts] == ["there is a red mug", "vision saw a desk"]
    assert got.facts[0].source == "operator"
    assert got.tags[0].label == "desk"


def test_keyframes_roundtrip(store):
    p = store.create_place("Kitchen")
    kf = store.add_keyframe(p.id, _jpeg(10), role="near", stamp_us=12345)
    assert kf.path.endswith(".jpg")
    assert (store.root / "keyframes" / p.id) in Path(kf.path).parents
    got = store.get_place(p.id)
    assert got.keyframes[0].role == "near"
    # to_dict exposes a path relative to the store root
    d = got.to_dict(root=store.root)
    assert d["keyframes"][0]["path"] == f"keyframes/{p.id}/{Path(kf.path).name}"


def test_descriptor_match_ranks_and_normalises(store):
    a = store.create_place("Kitchen")
    b = store.create_place("Lab")
    store.add_descriptor(a.id, np.array([2.0, 0.0, 0.0, 0.0]))  # unnormalised ok
    store.add_descriptor(b.id, np.array([0.0, 1.0, 0.0, 0.0]))
    matches = store.match(np.array([1.0, 0.0, 0.0, 0.0]))
    assert [m.place.name for m in matches] == ["Kitchen", "Lab"]
    assert matches[0].score == pytest.approx(1.0, abs=1e-6)
    assert matches[1].score == pytest.approx(0.0, abs=1e-6)


def test_match_best_per_place(store):
    a = store.create_place("Kitchen")
    store.add_descriptor(a.id, np.array([1.0, 0.0, 0.0, 0.0]))
    store.add_descriptor(a.id, np.array([0.0, 1.0, 0.0, 0.0]))
    matches = store.match(np.array([0.0, 1.0, 0.0, 0.0]))
    assert len(matches) == 1
    assert matches[0].score == pytest.approx(1.0, abs=1e-6)


def test_persists_across_reopen(tmp_path):
    root = tmp_path / "places"
    s1 = PlaceStore(root)
    p = s1.create_place("Kitchen")
    s1.add_fact(p.id, "coffee machine")
    s1.add_descriptor(p.id, np.array([1.0, 0.0, 0.0, 0.0]))
    s1.close()

    s2 = PlaceStore(root)
    assert s2.count() == 1
    got = s2.get_place("Kitchen")
    assert got is not None and got.facts[0].text == "coffee machine"
    assert s2.descriptor_count() == 1
    s2.close()


# --- per-place local maps --------------------------------------------------


def test_place_map_upsert_bumps_revision_and_preserves_created_at(store):
    p = store.create_place("Lab")
    snap = _snapshot(pose=(1.0, 2.0, 15.0))
    first, created = store.upsert_place_map(p.id, snap)
    assert created is True
    assert first.revision == 1
    assert first.anchor_x == pytest.approx(1.0)
    assert first.anchor_y == pytest.approx(2.0)
    assert first.anchor_yaw_deg == pytest.approx(15.0)
    created_at = first.created_at

    refined, created = store.upsert_place_map(
        "lab", _snapshot(pose=(3.0, 4.0, 30.0)))
    assert created is False
    assert refined.revision == 2
    assert refined.created_at == pytest.approx(created_at)
    # The anchor is pinned to the first capture, so accumulation stays coherent.
    assert refined.anchor_x == pytest.approx(1.0)
    assert refined.anchor_y == pytest.approx(2.0)
    assert refined.anchor_yaw_deg == pytest.approx(15.0)
    assert refined.updated_at >= refined.created_at
    assert store.get_place_map(p.id).revision == 2


def test_second_capture_does_not_leak_into_another_place(store):
    a = store.create_place("Kitchen")
    b = store.create_place("Lab")
    occ_a = np.zeros((3, 4), dtype=bool)
    occ_a[0, 0] = True
    occ_b = np.zeros((3, 4), dtype=bool)
    occ_b[2, 3] = True
    map_a, _ = store.upsert_place_map(
        a.id, _snapshot(occupied=occ_a, pose=(1.0, 1.0, 0.0)))
    map_b, _ = store.upsert_place_map(
        b.id, _snapshot(occupied=occ_b, pose=(9.0, 9.0, 90.0)))
    assert map_a.occupied_rle != map_b.occupied_rle
    assert store.get_place_map(a.id).revision == 1
    assert store.get_place_map(b.id).revision == 1

    # A second capture for B must not touch A's row at all.
    store.upsert_place_map(
        b.id, _snapshot(occupied=occ_b, pose=(8.0, 7.0, 45.0)))
    got_a = store.get_place_map(a.id)
    got_b = store.get_place_map(b.id)
    assert got_a.revision == 1
    assert got_a.anchor_x == pytest.approx(1.0)
    assert got_a.occupied_rle == map_a.occupied_rle
    assert got_b.revision == 2
    assert got_b.anchor_x == pytest.approx(9.0)


def test_place_map_rle_round_trip(store):
    p = store.create_place("Lab")
    occ = np.zeros((3, 4), dtype=bool)
    occ[0, 0] = True
    occ[1, 2] = True
    occ[2, 3] = True
    free = ~occ
    place_map, _ = store.upsert_place_map(
        p.id, _snapshot(occupied=occ, free=free))
    assert np.array_equal(decode_rle(place_map.occupied_rle, 4, 3), occ)
    assert np.array_equal(decode_rle(place_map.free_rle, 4, 3), free)
    payload = place_map.to_dict()
    assert payload["grid"]["occupied_count"] == 3
    assert payload["grid"]["free_count"] == 9
    assert payload["grid"]["encoding"] == "rle_base64"
    assert payload["frame"] == "capture_session_map"
    assert payload["anchor"] == {"x": 1.0, "y": 2.0, "yaw_deg": 15.0}


def test_anchor_is_pinned_to_the_first_capture_across_recaptures(store):
    p = store.create_place("Lab")
    poses = [(1.0, 2.0, 15.0), (2.5, -0.5, -40.0), (0.0, 0.0, 180.0)]
    for i, pose in enumerate(poses):
        place_map, created = store.upsert_place_map(
            p.id, _snapshot(pose=pose))
        assert created is (i == 0)
        assert place_map.revision == i + 1
        # The first capture owns the anchor; later poses only add evidence.
        assert (place_map.anchor_x, place_map.anchor_y,
                place_map.anchor_yaw_deg) == pytest.approx(poses[0])


def test_recapture_accumulates_and_never_shrinks(store):
    """Counts never shrink under merge.

    The precise invariant is known cells (occupied ∪ free); free alone may
    fall when an observed obstruction reclassifies it (see
    `test_merge_reclassifies_free_as_occupied_without_losing_known_cells`).
    """
    p = store.create_place("Lab")
    first_occ = np.zeros((3, 4), dtype=bool)
    first_occ[1, 1] = True
    first_free = np.zeros((3, 4), dtype=bool)
    first_free[0, 0] = True
    first_free[2, 3] = True
    walls = [[-0.5, 0.0, 0.5, 0.0], [0.5, -0.4, 0.5, 0.1]]
    first, _ = store.upsert_place_map(
        p.id, _snapshot(occupied=first_occ, free=first_free, walls=walls))
    assert rle_true_count(first.occupied_rle) == 1
    assert rle_true_count(first.free_rle) == 2
    assert first.walls == walls

    # A new occupied cell accumulates; the capture has zero walls, which must
    # not erase the walls already known.
    second_occ = np.zeros((3, 4), dtype=bool)
    second_occ[0, 3] = True
    second, _ = store.upsert_place_map(
        p.id, _snapshot(occupied=second_occ, walls=[]))
    assert rle_true_count(second.occupied_rle) == 2
    assert rle_true_count(second.free_rle) == 2
    assert second.walls == walls
    occ = decode_rle(second.occupied_rle, second.width, second.height)
    assert occ.sum() == 2

    # A completely empty snapshot (no cells, no walls) changes nothing.
    empty = np.zeros((3, 4), dtype=bool)
    third, _ = store.upsert_place_map(
        p.id, _snapshot(occupied=empty, free=empty, walls=[]))
    assert rle_true_count(third.occupied_rle) == 2
    assert rle_true_count(third.free_rle) == 2
    assert third.walls == walls


def test_merge_reclassifies_free_as_occupied_without_losing_known_cells(store):
    """The merge invariant is known cells, not raw free_count.

    A live obstruction observed in a later capture legitimately reclassifies
    previously free space as occupied (measured: known |3312| -> |3312|,
    lost=0, occupied +6, free -6).  So free_count may fall; no cell that was
    known (occupied ∪ free) may become unknown.
    """
    p = store.create_place("Lab")
    free_a = np.zeros((3, 4), dtype=bool)
    free_a[0, :] = True
    free_a[1, :] = True
    first, _ = store.upsert_place_map(
        p.id, _snapshot(occupied=np.zeros((3, 4), bool), free=free_a))
    known_before = (rle_true_count(first.occupied_rle)
                    + rle_true_count(first.free_rle))
    assert known_before == 8
    assert rle_true_count(first.occupied_rle) == 0

    # The incoming capture sees obstructions in cells that were free before.
    occ_b = np.zeros((3, 4), dtype=bool)
    occ_b[0, 1] = True
    occ_b[1, 2] = True
    merged, _ = store.upsert_place_map(
        p.id, _snapshot(occupied=occ_b))
    occupied = rle_true_count(merged.occupied_rle)
    free = rle_true_count(merged.free_rle)
    assert occupied == 2
    assert free == known_before - 2   # free shrank, and that is legitimate
    assert occupied + free == known_before  # nothing known was lost


def test_merge_rebins_grids_with_different_origins_and_sizes(store):
    p = store.create_place("Lab")
    # Stored: 3x3 at (0, 0), 0.1 m cells; occupied at its near corner.
    occ_a = np.zeros((3, 3), dtype=bool)
    occ_a[0, 0] = True
    free_a = ~occ_a
    first, _ = store.upsert_place_map(
        p.id, _snapshot(width=3, height=3, occupied=occ_a, free=free_a,
                        origin=(0.0, 0.0), resolution=0.1, walls=[]))
    assert (first.width, first.height) == (3, 3)
    assert first.origin_x == pytest.approx(0.0)

    # Incoming: 2x2 at (-0.2, -0.2); occupied at (row 1, col 1), so its
    # centre is (-0.05, -0.05) -> lattice cell -1 relative to the stored grid.
    occ_b = np.zeros((2, 2), dtype=bool)
    occ_b[1, 1] = True
    merged, _ = store.upsert_place_map(
        p.id, _snapshot(width=2, height=2, occupied=occ_b,
                        origin=(-0.2, -0.2), resolution=0.1, walls=[]))

    # Union bbox spans lattice -1..2 -> 4x4 anchored at (-0.1, -0.1).
    assert (merged.width, merged.height) == (4, 4)
    assert merged.origin_x == pytest.approx(-0.1)
    assert merged.origin_y == pytest.approx(-0.1)
    occ = decode_rle(merged.occupied_rle, merged.width, merged.height)
    # Stored cell (0, 0) shifts by one; incoming cell (-1, -1) lands at (0, 0).
    assert bool(occ[1, 1]) is True
    assert bool(occ[0, 0]) is True
    assert int(occ.sum()) == 2
    # The cells sit at their true capture-frame centres (anchor-relative):
    # stored (0.05, 0.05) and incoming (-0.05, -0.05).
    ys, xs = np.nonzero(occ)
    centres = {(round(merged.origin_x + (x + 0.5) * 0.1, 6),
                round(merged.origin_y + (y + 0.5) * 0.1, 6))
               for y, x in zip(ys, xs)}
    assert centres == {(0.05, 0.05), (-0.05, -0.05)}
    # The pinned anchor is untouched by the rebase.
    assert (merged.anchor_x, merged.anchor_y,
            merged.anchor_yaw_deg) == pytest.approx((1.0, 2.0, 15.0))


def test_replace_true_resets_to_exactly_the_new_patch(store):
    p = store.create_place("Lab")
    occ_a = np.zeros((3, 4), dtype=bool)
    occ_a[0, 0] = True
    first, _ = store.upsert_place_map(
        p.id, _snapshot(occupied=occ_a, pose=(1.0, 2.0, 15.0)))
    assert rle_true_count(first.occupied_rle) == 1

    occ_b = np.zeros((2, 2), dtype=bool)
    occ_b[0, 1] = True
    free_b = ~occ_b
    walls_b = [[1.0, 1.0, 1.2, 1.0]]
    replaced, created = store.upsert_place_map(
        p.id,
        _snapshot(width=2, height=2, occupied=occ_b, free=free_b,
                  pose=(3.0, 4.0, 30.0), walls=walls_b,
                  origin=(1.0, 1.0), resolution=0.05),
        replace=True,
    )
    assert created is False
    assert replaced.revision == 2
    # Exactly the new patch: anchor, geometry and walls all come from it.
    assert (replaced.anchor_x, replaced.anchor_y,
            replaced.anchor_yaw_deg) == pytest.approx((3.0, 4.0, 30.0))
    assert (replaced.origin_x, replaced.origin_y) == pytest.approx((1.0, 1.0))
    assert (replaced.width, replaced.height) == (2, 2)
    assert np.array_equal(
        decode_rle(replaced.occupied_rle, 2, 2), occ_b)
    assert np.array_equal(decode_rle(replaced.free_rle, 2, 2), free_b)
    assert replaced.walls == walls_b
    assert rle_true_count(replaced.occupied_rle) == 1
    assert rle_true_count(replaced.free_rle) == 3

    with pytest.raises(ValueError):
        store.upsert_place_map(p.id, _snapshot(), replace="yes")


def test_map_uses_visual_pose_never_wheel_odometry(store):
    # The legacy capture odom is deliberately nonsense; a submap anchor must
    # come from the structure payload's gravity-aligned visual pose.
    p = store.create_place("Lab", odom=(50.0, -60.0, 1.23))
    # An explicit pose (e.g. from the live service) overrides the snapshot's
    # on the first capture, which is where the anchor is pinned.
    place_map, _ = store.upsert_place_map(
        p.id, _snapshot(pose=(0.0, 0.0, 0.0)),
        pose={"valid": True, "x": 3.0, "y": 4.0, "yaw_deg": 90.0})
    assert (place_map.anchor_x, place_map.anchor_y,
            place_map.anchor_yaw_deg) == pytest.approx((3.0, 4.0, 90.0))
    assert 50.0 not in (
        place_map.anchor_x, place_map.anchor_y, place_map.anchor_yaw_deg)

    # A re-capture never moves the pinned anchor, even with a fresh pose.
    refined, _ = store.upsert_place_map(
        p.id, _snapshot(pose=(1.5, 2.5, 30.0)),
        pose={"valid": True, "x": 1.5, "y": 2.5, "yaw_deg": 30.0})
    assert (refined.anchor_x, refined.anchor_y,
            refined.anchor_yaw_deg) == pytest.approx((3.0, 4.0, 90.0))
    # get_place still exposes legacy odom for compatibility, but it played no
    # part in the map coordinate system.
    assert store.get_place(p.id).odom == (50.0, -60.0, 1.23)


def test_malformed_map_payloads_are_rejected_not_stored(store):
    p = store.create_place("Lab")
    occ = np.zeros((3, 4), dtype=bool)
    occ[1, 1] = True
    valid = _snapshot(occupied=occ)
    cases = []

    cases.append({})  # no grid at all
    bad = copy.deepcopy(valid)
    bad["grid"]["encoding"] = "png"
    cases.append(bad)
    bad = copy.deepcopy(valid)
    bad["grid"]["width"] = 5  # RLE covers 12 cells, expected 15
    cases.append(bad)
    bad = copy.deepcopy(valid)
    bad["grid"]["occupied_rle"] = "not-base64!!"
    cases.append(bad)
    bad = copy.deepcopy(valid)
    bad["gravity"]["valid"] = False
    cases.append(bad)
    bad = copy.deepcopy(valid)
    bad["floor"]["valid"] = False
    cases.append(bad)
    bad = copy.deepcopy(valid)
    bad["pose"]["valid"] = False
    cases.append(bad)
    bad = copy.deepcopy(valid)
    bad["pose"]["x"] = "left"
    cases.append(bad)
    bad = copy.deepcopy(valid)
    bad["grid"]["resolution_m"] = 0
    cases.append(bad)
    bad = copy.deepcopy(valid)
    bad["walls"] = [[0.0, 0.0, 1.0]]  # not [x0,y0,x1,y1]
    cases.append(bad)

    for payload in cases:
        with pytest.raises(ValueError):
            store.upsert_place_map(p.id, payload)
        assert store.get_place_map(p.id) is None
    with pytest.raises(KeyError):
        store.upsert_place_map("no-such-place", valid)


def test_deleting_place_cascades_its_map_and_edges(store):
    a = store.create_place("Kitchen")
    b = store.create_place("Lab")
    store.upsert_place_map(a.id, _snapshot())
    store.record_traversal(a.id, b.id, success=True)
    assert store.get_place_map(a.id) is not None
    assert len(store.list_edges()) == 1
    store.delete_place(a.id)
    assert store.get_place_map(a.id) is None
    assert store.list_edges() == []


# --- place graph (route knowledge) ----------------------------------------


def test_graph_serializes_zero_one_and_many_edges(store):
    empty = store.graph()
    assert empty == {"nodes": [], "edges": [],
                     "node_count": 0, "edge_count": 0}

    a = store.create_place("A")
    b = store.create_place("B")
    c = store.create_place("C")
    one = store.graph()
    assert one["node_count"] == 3
    assert [n["name"] for n in one["nodes"]] == ["A", "B", "C"]
    assert all(n["has_map"] is False for n in one["nodes"])
    assert one["edges"] == [] and one["edge_count"] == 0

    store.upsert_place_map(a.id, _snapshot())
    edge, created = store.record_traversal(a.id, b.id, success=True)
    assert created is True
    graph = store.graph()
    assert graph["edge_count"] == 1
    assert graph["edges"][0]["from_name"] == "A"
    assert graph["edges"][0]["to_name"] == "B"
    assert graph["edges"][0]["traversals"] == 1
    assert graph["edges"][0]["successes"] == 1
    assert [n for n in graph["nodes"] if n["name"] == "A"][0]["has_map"]

    store.record_traversal(a.id, c.id, success=False)
    store.record_traversal(b.id, a.id, success=True)
    many = store.graph()
    assert many["edge_count"] == 3
    pairs = {(e["from"], e["to"]) for e in many["edges"]}
    assert pairs == {(a.id, b.id), (a.id, c.id), (b.id, a.id)}
    assert {e["from_name"] for e in many["edges"]} == {"A", "B"}


def test_edge_counters_success_semantics_and_median(store):
    a = store.create_place("Kitchen")
    b = store.create_place("Lab")
    guide = [[0.0, 0.0], [1.0, 0.5], [2.0, 0.5]]
    edge, created = store.record_traversal(
        a.id, b.id, success=True, duration_s=10.0, guide_path=guide)
    assert created is True
    assert (edge.traversals, edge.successes) == (1, 1)
    assert edge.median_duration_s == pytest.approx(10.0)
    assert edge.guide_path == guide
    first_success = edge.last_success_at

    # A failure counts as an attempt, not a success, and keeps the median.
    edge, _ = store.record_traversal(
        a.id, b.id, success=False, duration_s=99.0,
        guide_path=[[9.0, 9.0]])
    assert (edge.traversals, edge.successes) == (2, 1)
    assert edge.last_success_at == pytest.approx(first_success)
    assert edge.median_duration_s == pytest.approx(10.0)
    assert edge.guide_path == guide  # failed routes are not guides

    edge, _ = store.record_traversal(
        a.id, b.id, success=True, duration_s=20.0)
    edge, _ = store.record_traversal(
        a.id, b.id, success=True, duration_s=30.0)
    assert (edge.traversals, edge.successes) == (4, 3)
    assert edge.success_rate == pytest.approx(0.75)
    assert edge.median_duration_s == pytest.approx(20.0)

    # A success without a duration still counts, but is not a timing sample.
    edge, _ = store.record_traversal(a.id, b.id, success=True)
    assert (edge.traversals, edge.successes) == (5, 4)
    assert edge.median_duration_s == pytest.approx(20.0)
    assert edge.last_success_at >= first_success

    with pytest.raises(ValueError):
        store.record_traversal(a.id, b.id, success="yes")
    with pytest.raises(ValueError):
        store.record_traversal(a.id, b.id, success=True, duration_s=-1)
    with pytest.raises(ValueError):
        store.record_traversal(
            a.id, b.id, success=True, guide_path=[[0.0]])
    with pytest.raises(KeyError):
        store.record_traversal(a.id, "no-such-place", success=True)


# --- service + whereami ----------------------------------------------------


def _service(tmp_path, views, encoder):
    store = PlaceStore(tmp_path / "places")
    svc = PlacesService(
        store, capture=StaticCapture(views), encoder=encoder,
        host="127.0.0.1", port=0, min_score=0.5, min_margin=0.1,
    )
    return store, svc


def test_teach_encodes_and_stores(tmp_path):
    a = _jpeg(20)
    enc = FakeEncoder({a: [1, 0, 0, 0]})
    store, svc = _service(tmp_path, [View("near", a, 111)], enc)
    try:
        res = svc.teach("Kitchen")
        assert res["descriptors"] == 1
        assert res["views"] == 1
        assert svc.store.descriptor_count() == 1
    finally:
        store.close()


def test_capture_views_appends_to_existing_place(tmp_path):
    a, b = _jpeg(20), _jpeg(80)
    enc = FakeEncoder({a: [1, 0, 0, 0], b: [0, 1, 0, 0]})
    store, svc = _service(tmp_path, [View("far", a, 1)], enc)
    try:
        svc.teach("Kitchen")
        assert svc.store.descriptor_count() == 1
        svc.capture = StaticCapture([View("far", b, 2)])
        out = svc.capture_views("Kitchen")
        assert out["keyframes"] == 1 and out["descriptors"] == 1
        assert out["views_total"] == 2
        assert svc.store.descriptor_count() == 2
    finally:
        store.close()


def test_capture_burst_dedups_a_static_frame(tmp_path):
    a = _jpeg(20)
    enc = FakeEncoder({a: [1, 0, 0, 0]})
    store, svc = _service(tmp_path, [View("far", a, 1)], enc)
    try:
        svc.teach("Kitchen")
        # The same camera stamp every tick -> only the first is captured.
        out = svc.capture_views("Kitchen", seconds=0.4, interval_s=0.1)
        assert out["keyframes"] == 1
        assert out["views_total"] == 2  # the taught view + this one
    finally:
        store.close()


def test_whereami_matches_known_room(tmp_path):
    a, b = _jpeg(20), _jpeg(80)
    enc = FakeEncoder({a: [1, 0, 0, 0], b: [0, 1, 0, 0]})
    # teach both, then query with A
    store, svc = _service(tmp_path, [View("near", a, 1)], enc)
    try:
        svc.teach("Kitchen")
        svc.capture = StaticCapture([View("near", b, 2)])
        svc.teach("Lab")
        svc.capture = StaticCapture([View("near", a, 3)])
        out = svc.whereami()
        assert out["matched"] is True
        assert out["place"]["name"] == "Kitchen"
        assert out["score"] == pytest.approx(1.0, abs=1e-6)
        assert out["margin"] == pytest.approx(1.0, abs=1e-6)
    finally:
        store.close()


def test_whereami_unknown_when_below_threshold(tmp_path):
    a, unknown = _jpeg(20), _jpeg(200)
    enc = FakeEncoder({a: [1, 0, 0, 0], unknown: [0, 0, 1, 0]})
    store, svc = _service(tmp_path, [View("near", a, 1)], enc)
    try:
        svc.teach("Kitchen")
        svc.capture = StaticCapture([View("near", unknown, 2)])
        out = svc.whereami()
        assert out["matched"] is False
        assert out["place"] is None
    finally:
        store.close()


def test_whereami_no_frames(tmp_path):
    store, svc = _service(tmp_path, [], FakeEncoder())
    try:
        out = svc.whereami()
        assert out["matched"] is False
        assert "frames" in out["reason"]
    finally:
        store.close()


# --- HTTP contract ---------------------------------------------------------


def _get(port, path):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=5) as r:
        return r.status, json.loads(r.read())


def _post(port, path, body):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=5) as r:
        return r.status, json.loads(r.read())


def test_http_place_map_and_graph(tmp_path):
    store, svc = _service(tmp_path, [], FakeEncoder())
    svc.start()
    try:
        port = svc.bound_port
        pid = store.create_place("Kitchen").id
        snap = _snapshot(pose=(2.0, 3.0, 12.0))

        status, res = _post(port, f"/places/{pid}/map", snap)
        assert status == 201
        assert res["created"] is True
        assert res["map"]["revision"] == 1
        assert res["map"]["anchor"] == {"x": 2.0, "y": 3.0,
                                        "yaw_deg": 12.0}

        status, res = _post(port, f"/places/{pid}/map", snap)
        assert status == 200
        assert res["created"] is False
        assert res["map"]["revision"] == 2

        status, got = _get(port, f"/places/{pid}/map")
        assert status == 200
        assert got["map"]["grid"]["width"] == 4
        assert got["map"]["grid"]["occupied_count"] == 0

        status, graph = _get(port, "/places/graph")
        assert graph["node_count"] == 1
        assert graph["nodes"][0]["has_map"] is True
        assert graph["nodes"][0]["map_revision"] == 2
        assert graph["edges"] == []

        lab = store.create_place("Lab")
        status, res = _post(
            port, f"/places/{pid}/edges/{lab.id}",
            {"success": True, "duration_s": 12.5,
             "guide_path": [[0.0, 0.0], [1.0, 0.0]]})
        assert status == 201
        assert res["edge"]["traversals"] == 1
        assert res["edge"]["successes"] == 1
        assert res["edge"]["median_duration_s"] == 12.5

        status, res = _post(
            port, f"/places/{pid}/edges/{lab.id}", {"success": False})
        assert status == 200
        assert res["edge"]["traversals"] == 2
        assert res["edge"]["successes"] == 1

        status, graph = _get(port, "/places/graph")
        assert graph["edge_count"] == 1
        assert graph["edges"][0]["from_name"] == "Kitchen"
        assert graph["edges"][0]["to_name"] == "Lab"

        with pytest.raises(urllib.error.HTTPError) as exc:
            _post(port, "/places/nope/map", snap)
        assert exc.value.code == 404
        with pytest.raises(urllib.error.HTTPError) as exc:
            _post(port, f"/places/{pid}/map", {})
        assert exc.value.code == 400
        with pytest.raises(urllib.error.HTTPError) as exc:
            _get(port, f"/places/{lab.id}/map")
        assert exc.value.code == 404
        with pytest.raises(urllib.error.HTTPError) as exc:
            _post(port, f"/places/{pid}/edges/{lab.id}", {})
        assert exc.value.code == 400
    finally:
        svc.stop()
        store.close()


def test_http_map_merge_is_default_and_replace_is_explicit(tmp_path):
    store, svc = _service(tmp_path, [], FakeEncoder())
    svc.start()
    try:
        port = svc.bound_port
        pid = store.create_place("Kitchen").id
        occ_a = np.zeros((3, 4), dtype=bool)
        occ_a[1, 1] = True
        first = _snapshot(occupied=occ_a, pose=(2.0, 3.0, 12.0))
        status, res = _post(port, f"/places/{pid}/map", first)
        assert status == 201 and res["replaced"] is False
        assert res["map"]["grid"]["occupied_count"] == 1

        # Default re-capture merges: a zero-cell snapshot cannot erase it.
        status, res = _post(port, f"/places/{pid}/map", _snapshot())
        assert status == 200
        assert res["map"]["revision"] == 2
        assert res["map"]["grid"]["occupied_count"] == 1
        assert res["map"]["anchor"] == {"x": 2.0, "y": 3.0, "yaw_deg": 12.0}

        # Explicit replace resets the patch, anchor included.
        occ_b = np.zeros((2, 2), dtype=bool)
        occ_b[0, 0] = True
        replacement = _snapshot(width=2, height=2, occupied=occ_b,
                                pose=(5.0, 6.0, 45.0), walls=[])
        replacement["replace"] = True
        status, res = _post(port, f"/places/{pid}/map", replacement)
        assert status == 200 and res["replaced"] is True
        assert res["map"]["revision"] == 3
        assert res["map"]["grid"]["occupied_count"] == 1
        assert res["map"]["grid"]["width"] == 2
        assert res["map"]["anchor"] == {"x": 5.0, "y": 6.0, "yaw_deg": 45.0}
        assert res["map"]["walls"] == []

        replacement["replace"] = "yes"
        with pytest.raises(urllib.error.HTTPError) as exc:
            _post(port, f"/places/{pid}/map", replacement)
        assert exc.value.code == 400
    finally:
        svc.stop()
        store.close()


def test_http_smoke(tmp_path):
    a = _jpeg(20)
    enc = FakeEncoder({a: [1, 0, 0, 0]})
    store, svc = _service(tmp_path, [View("near", a, 1)], enc)
    svc.start()
    try:
        port = svc.bound_port
        status, health = _get(port, "/healthz")
        assert status == 200 and health["ok"] is True

        status, res = _post(port, "/places", {"name": "Kitchen",
                                              "facts": ["coffee machine"]})
        assert status == 201
        pid = res["place"]["id"]

        status, listing = _get(port, "/places")
        assert status == 200 and [p["name"] for p in listing["places"]] == ["Kitchen"]

        status, one = _get(port, f"/places/{pid}")
        assert one["place"]["facts"][0]["text"] == "coffee machine"

        status, fact = _post(port, f"/places/{pid}/facts", {"text": "has a sink"})
        assert status == 201 and fact["fact"]["source"] == "operator"

        status, wa = _get(port, "/whereami")
        assert wa["matched"] is True and wa["place"]["name"] == "Kitchen"

        # rename: the id keeps resolving, the old name does not
        status, _ = _post(port, f"/places/{pid}/rename", {"name": "Dining"})
        assert status == 200
        status, one = _get(port, f"/places/{pid}")
        assert one["place"]["name"] == "Dining"
        with pytest.raises(urllib.error.HTTPError) as exc:
            _get(port, "/places/Kitchen")
        assert exc.value.code == 404

        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/places/{pid}", method="DELETE")
        with urllib.request.urlopen(req, timeout=5) as r:
            assert r.status == 200
        status, listing = _get(port, "/places")
        assert listing["places"] == []
    finally:
        svc.stop()
        store.close()
