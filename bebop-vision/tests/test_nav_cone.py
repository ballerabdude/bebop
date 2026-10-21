"""Depth stop-cone tests on synthetic scenes (no hardware)."""

import json
import math
import time
from types import SimpleNamespace

import numpy as np
import pytest

from bebop_vision.nav import CameraGeometry, ConeConfig, DepthCone
from bebop_vision.nav.synth import render_depth, small_geometry

FOOTPRINT = (-0.13, 0.32, -0.34, 0.19)


def frame(depth, recv_ts=None, stamp_us=1000):
    return SimpleNamespace(depth=depth, depth_scale_mm=1.0, stamp_us=stamp_us,
                           recv_ts=recv_ts or time.monotonic())


def cone_for(geom, **kw):
    cfg = ConeConfig(footprint_m=FOOTPRINT, stride=2, **kw)
    return DepthCone([geom], cfg)


def test_rotation_maps_optical_axis_down_for_negative_pitch():
    g = small_geometry(pitch_deg=-66.0)
    fwd = g.rotation() @ np.array([0.0, 0.0, 1.0])
    assert fwd[0] == pytest.approx(math.cos(math.radians(-66)), abs=1e-6)
    assert fwd[2] == pytest.approx(math.sin(math.radians(-66)), abs=1e-6)
    # Image "down" points toward the robot and the floor.
    down = g.rotation() @ np.array([0.0, 1.0, 0.0])
    assert down[0] < 0 and down[2] < 0
    # Image "right" is body -y (right).
    right = g.rotation() @ np.array([1.0, 0.0, 0.0])
    assert right[1] == pytest.approx(-1.0, abs=1e-6)


def test_flat_floor_is_free_and_observed():
    g = small_geometry()
    cone = cone_for(g)
    cm = cone.evaluate({"near": frame(render_depth(g))})
    assert cm is not None and cm.trusted
    assert cm.n_obstacle == 0 and cm.n_drop == 0 and not cm.drop
    assert cm.n_floor > 1000
    assert not math.isfinite(cm.corridor_min_m)
    # The near camera (-66 deg) sees the floor out to ~1.7 m ahead.
    centre = cm.clearance_at(0.0)
    assert 1.4 < centre <= cone.config.lookahead_m
    assert cm.floor_fit["near"]["source"] == "fit"
    assert abs(cm.floor_fit["near"]["tilt_deg"]) < 0.5


def test_unobserved_space_is_not_free():
    g = small_geometry()
    cone = cone_for(g)
    depth = render_depth(g)
    # Blank the top of the image: floor beyond ~1 m is now unobserved.
    cut = int(g.height * 0.45)
    depth[:cut, :] = 0
    cm = cone.evaluate({"near": frame(depth)})
    assert cm.n_obstacle == 0
    assert cm.clearance_at(0.0) < 1.3
    centre = len(cm.observed_m) // 2
    assert 0 < cm.observed_m[centre] < 1.3


def test_wall_ahead_sets_obstacle_and_corridor():
    g = small_geometry()
    cone = cone_for(g)
    depth = render_depth(g, walls=[(1.0, -1.5, 1.5, 0.6)])
    cm = cone.evaluate({"near": frame(depth)})
    assert cm.n_obstacle > 50
    assert cm.clearance_at(0.0) == pytest.approx(1.0, abs=0.06)
    assert cm.corridor_min_m == pytest.approx(1.0, abs=0.06)
    assert not cm.drop


def test_box_on_one_side_leaves_other_side_clear():
    g = small_geometry()
    cone = cone_for(g)
    # Box to the left, 0.8 m ahead.
    depth = render_depth(g, boxes=[(0.8, 1.1, 0.3, 0.9, 0.4)])
    cm = cone.evaluate({"near": frame(depth)})
    left = cm.clearance_at(math.radians(35))
    right = cm.clearance_at(math.radians(-35))
    assert left < 1.0
    assert right > 1.3
    # Straight ahead the corridor (|y| <= ~0.42) is untouched.
    assert not math.isfinite(cm.corridor_min_m)


def test_chassis_footprint_is_ignored():
    g = small_geometry()
    cone = cone_for(g)
    # A block exactly where the chassis is.
    depth = render_depth(g, boxes=[(-0.1, 0.3, -0.3, 0.15, 0.55)])
    cm = cone.evaluate({"near": frame(depth)})
    assert cm.n_obstacle == 0
    assert not math.isfinite(cm.corridor_min_m)


def test_self_mask_polygon_removes_pixels():
    g_plain = small_geometry()
    # Mask the whole bottom half of the image.
    poly = ((0, 60), (212, 60), (212, 120), (0, 120))
    g_masked = small_geometry(self_polys=(poly,))
    depth = render_depth(g_plain)
    n_plain = cone_for(g_plain).points("near", depth, 1.0)[0].shape[0]
    n_masked = cone_for(g_masked).points("near", depth, 1.0)[0].shape[0]
    assert n_masked < 0.6 * n_plain


def test_drop_in_corridor_is_flagged():
    g = small_geometry()
    cone = cone_for(g)
    depth = render_depth(g, holes=[(0.9, 1.6, -0.6, 0.6, 0.30)])
    cm = cone.evaluate({"near": frame(depth)})
    assert cm.n_drop > 20
    assert cm.drop
    # The hole starts at 0.9 m: observed floor stops there even though
    # the far rim (1.6 m+) is floor again.
    assert 0.8 < cm.clearance_at(0.0) < 1.2
    assert 0.8 < cm.observed_m[len(cm.observed_m) // 2] < 1.2


def test_floor_beyond_an_occluder_does_not_count():
    g = small_geometry()
    cone = cone_for(g)
    # A low, thin bar across the view at 0.9 m: floor behind it is seen
    # (the camera looks over it) but is not contiguous with the near floor.
    depth = render_depth(g, boxes=[(0.9, 0.95, -1.5, 1.5, 0.3)])
    cm = cone.evaluate({"near": frame(depth)})
    assert cm.clearance_at(0.0) == pytest.approx(0.9, abs=0.08)


def test_floor_fit_corrects_a_mis_pitched_prior():
    # The real camera is pitched 7 deg off the YAML value.
    g_true = small_geometry(pitch_deg=-59.0)
    g_prior = small_geometry(pitch_deg=-66.0)
    depth = render_depth(g_true)
    raw = DepthCone([g_prior], ConeConfig(footprint_m=FOOTPRINT, stride=2,
                                          floor_fit=False))
    cm_raw = raw.evaluate({"near": frame(depth)})
    fitted = DepthCone([g_prior], ConeConfig(footprint_m=FOOTPRINT, stride=2))
    cm_fit = None
    for _ in range(6):   # the plane low-pass converges over a few frames
        cm_fit = fitted.evaluate({"near": frame(depth)})
    # Without the fit the tilted floor reads as obstacles/drops.
    assert cm_raw.n_obstacle + cm_raw.n_drop > 200
    assert cm_fit.n_obstacle + cm_fit.n_drop < 30
    assert cm_fit.floor_fit["near"]["source"] == "fit"
    assert abs(cm_fit.floor_fit["near"]["tilt_deg"] - 7.0) < 1.0


def test_two_cameras_merge():
    near = small_geometry("near", 1.27, -66.0)
    far = small_geometry("far", 1.32, -17.4)
    cone = DepthCone([near, far], ConeConfig(footprint_m=FOOTPRINT, stride=2))
    wall = [(2.5, -2.0, 2.0, 1.0)]
    cm = cone.evaluate({"near": frame(render_depth(near, walls=wall)),
                        "far": frame(render_depth(far, walls=wall))})
    assert cm.roles == ("near", "far")
    assert cm.clearance_at(0.0) == pytest.approx(2.5, abs=0.08)
    # Near only cannot see that far: the wall is beyond its floor view.
    cm_near = cone.evaluate({"near": frame(render_depth(near, walls=wall)),
                             "far": None})
    assert cm_near.roles == ("near",)
    assert cm_near.clearance_at(0.0) < 2.0


def test_missing_frames_and_untrusted_depth():
    g = small_geometry()
    cone = cone_for(g)
    assert cone.evaluate({"near": None}) is None
    depth = render_depth(g)
    depth[::1, ::1] = 0
    depth[::7, ::7] = 1500
    cm = cone.evaluate({"near": frame(depth)})
    assert cm is not None and not cm.trusted
    assert cm.valid_frac < 0.1


def test_wrong_depth_shape_is_rejected():
    g = small_geometry()
    cone = cone_for(g)
    with pytest.raises(ValueError):
        cone.points("near", np.zeros((10, 10), np.uint16), 1.0)


def test_depth_scale_is_applied():
    g = small_geometry()
    cone = cone_for(g)
    depth = render_depth(g, walls=[(1.0, -1.5, 1.5, 0.6)])
    # Same scene encoded at 0.5 mm per unit.
    cm_half = cone.evaluate({"near": SimpleNamespace(
        depth=(depth.astype(np.uint32) * 2).astype(np.uint16),
        depth_scale_mm=0.5, stamp_us=1, recv_ts=time.monotonic())})
    assert cm_half.clearance_at(0.0) == pytest.approx(1.0, abs=0.06)


def test_to_dict_is_json_and_age_tracks_recv_ts():
    g = small_geometry()
    cone = cone_for(g)
    t = time.monotonic() - 0.25
    cm = cone.evaluate({"near": frame(render_depth(g), recv_ts=t)})
    d = cm.to_dict()
    json.dumps(d)
    assert len(d["clearance_m"]) == cone.config.n_sectors
    assert 0.2 <= cm.age_s() < 1.0


def test_config_validation():
    with pytest.raises(ValueError):
        ConeConfig(n_sectors=4).validate()
    with pytest.raises(ValueError):
        ConeConfig(obstacle_min_z_m=0.01).validate()
    with pytest.raises(ValueError):
        ConeConfig(footprint_m=(0.5, 0.1, -0.3, 0.2)).validate()


def test_geometry_from_rig_reads_the_checked_in_config():
    g = CameraGeometry.from_rig("near")
    assert g.width == 848 and g.height == 480
    assert g.height_m == pytest.approx(1.27)
    assert g.pitch_deg == pytest.approx(-66.0)
    assert len(g.self_polys) >= 1
    with pytest.raises(KeyError):
        CameraGeometry.from_rig("sideways")


def test_far_camera_follows_near_plane_when_it_cannot_fit():
    near = small_geometry("near", 1.27, -66.0)
    # Far camera's real pitch is 1 deg off the YAML value; near is exact.
    far_true = small_geometry("far", 1.32, -16.4)
    far_prior = small_geometry("far", 1.32, -17.4)
    cone = DepthCone([near, far_prior],
                     ConeConfig(footprint_m=FOOTPRINT, stride=2))
    # 1. Open floor: both cameras fit; the far camera learns its delta.
    for _ in range(8):
        cm = cone.evaluate({"near": frame(render_depth(near)),
                            "far": frame(render_depth(far_true))})
    assert cm.floor_fit["far"]["source"] == "fit"
    assert cm.n_drop == 0
    learned_pitch = cm.floor_fit["far"]["plane"]["pitch_deg"]
    assert abs(learned_pitch - 1.0) < 0.5
    # 2. Facing a wall: the far camera sees no floor and cannot fit, but
    # keeps the learned plane instead of snapping back to the prior.
    wall = render_depth(far_true, walls=[(1.3, -3, 3, 2.5)], floor=False)
    cm = cone.evaluate({"near": frame(render_depth(near)),
                        "far": frame(wall)})
    assert cm.floor_fit["far"]["source"] == "prior"
    assert abs(cm.floor_fit["far"]["plane"]["pitch_deg"] - 1.0) < 0.6


def test_drops_only_from_drop_roles():
    near = small_geometry("near", 1.27, -66.0)
    far = small_geometry("far", 1.32, -17.4)
    hole = [(0.9, 1.6, -0.6, 0.6, 0.30)]
    # The near camera sees the hole, but only "far" may report drops.
    cone = DepthCone([near, far], ConeConfig(footprint_m=FOOTPRINT, stride=2,
                                             drop_roles=("far",)))
    cm = cone.evaluate({"near": frame(render_depth(near, holes=hole)),
                        "far": frame(render_depth(far))})
    assert not cm.drop and cm.n_drop == 0
    cone_all = DepthCone([near, far], ConeConfig(footprint_m=FOOTPRINT,
                                                 stride=2, drop_roles=()))
    cm = cone_all.evaluate({"near": frame(render_depth(near, holes=hole)),
                            "far": frame(render_depth(far))})
    assert cm.drop and cm.n_drop > 20
