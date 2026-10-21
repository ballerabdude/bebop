"""Gyro + odometry pose estimator tests (synthetic samples)."""

import collections
import math

import numpy as np
import pytest

from bebop_vision.nav import PoseConfig, PoseEstimator, wrap_pi
from bebop_vision.nav.synth import small_geometry


def make(roles=("far",), pitch=-17.4):
    geoms = [small_geometry(r, 1.32, pitch) for r in roles]
    taps = {r: collections.deque() for r in roles}
    est = PoseEstimator(geoms, taps, PoseConfig(bias_min_samples=50))
    return est, geoms, taps


def feed(taps, geoms, body_yaw_rate, t0, n, dt=0.005, bias=0.0):
    """Push n gyro samples that correspond to a body yaw rate (plus a
    constant bias), expressed in each camera's optical frame."""
    for g in geoms:
        R = g.rotation()
        # Body angular velocity (0, 0, w + bias) -> optical frame = R^T w.
        w_cam = R.T @ np.array([0.0, 0.0, body_yaw_rate + bias])
        for i in range(n):
            t = t0 + i * dt
            taps[g.role].append(("gyro", float(w_cam[0]), float(w_cam[1]),
                                 float(w_cam[2]), int(t * 1e6), t))


def test_bias_learned_while_still_then_gyro_heading():
    est, geoms, taps = make()
    t = 0.0
    # Still for 1 s with a 0.02 rad/s bias: 200 samples.
    for _ in range(10):
        feed(taps, geoms, 0.0, t, 20, bias=0.02)
        p = est.update((0, 0, 0), (0, 0), now=t + 0.1)
        t += 0.1
    assert p.calibrated and p.heading_source == "gyro"
    assert p.bias_rad_s == pytest.approx(0.02, abs=2e-3)
    assert abs(p.theta) < 0.01
    # Now rotate at 1 rad/s for 1 s (odom claims 2 rad: the slip case).
    odom_th = 0.0
    for _ in range(10):
        feed(taps, geoms, 1.0, t, 20, bias=0.02)
        odom_th += 0.2
        p = est.update((0, 0, odom_th), (0, 1.0), now=t + 0.1)
        t += 0.1
    assert p.heading_source == "gyro"
    assert p.theta == pytest.approx(1.0, abs=0.03)


def test_pitched_near_camera_gives_same_yaw():
    est, geoms, taps = make(roles=("near",), pitch=-66.0)
    t = 0.0
    for _ in range(10):
        feed(taps, geoms, 0.0, t, 20)
        est.update((0, 0, 0), (0, 0), now=t + 0.1)
        t += 0.1
    for _ in range(10):
        feed(taps, geoms, -0.5, t, 20)
        p = est.update((0, 0, 0), (0, -0.5), now=t + 0.1)
        t += 0.1
    assert p.theta == pytest.approx(-0.5, abs=0.03)


def test_distance_from_odom_along_estimated_heading():
    est, geoms, taps = make()
    t = 0.0
    for _ in range(10):
        feed(taps, geoms, 0.0, t, 20)
        est.update((0, 0, 0), (0, 0), now=t + 0.1)
        t += 0.1
    # Gyro says we turned +90 deg while odom says +180 deg.
    odom_th = 0.0
    for _ in range(10):
        feed(taps, geoms, math.pi / 2, t, 20)
        odom_th += math.pi / 10
        p = est.update((0, 0, odom_th), (0, 1), now=t + 0.1)
        t += 0.1
    assert p.theta == pytest.approx(math.pi / 2, abs=0.05)
    # Drive 1 m "forward" by odometry (odom thinks it is heading pi).
    ox = oy = 0.0
    for _ in range(10):
        feed(taps, geoms, 0.0, t, 20)
        ox += 0.1 * math.cos(odom_th)
        oy += 0.1 * math.sin(odom_th)
        p = est.update((ox, oy, odom_th), (0.3, 0), now=t + 0.1)
        t += 0.1
    # The estimate moves 1 m along its own +90 deg heading.
    assert p.x == pytest.approx(0.0, abs=0.05)
    assert p.y == pytest.approx(1.0, abs=0.05)
    # Reverse 0.5 m: negative projection.
    for _ in range(5):
        feed(taps, geoms, 0.0, t, 20)
        ox -= 0.1 * math.cos(odom_th)
        oy -= 0.1 * math.sin(odom_th)
        p = est.update((ox, oy, odom_th), (-0.15, 0), now=t + 0.1)
        t += 0.1
    assert p.y == pytest.approx(0.5, abs=0.05)


def test_falls_back_to_odom_without_gyro():
    est = PoseEstimator([small_geometry("far", 1.32, -17.4)], {})
    assert not est.has_gyro
    p = est.update((0, 0, 0), (0, 0), now=0.0)
    p = est.update((0, 0, 0.3), (0, 1), now=0.1)
    assert p.heading_source == "odom" and p.theta == pytest.approx(0.3)


def test_stale_gyro_falls_back_and_reports():
    est, geoms, taps = make()
    t = 0.0
    for _ in range(10):
        feed(taps, geoms, 0.0, t, 20)
        est.update((0, 0, 0), (0, 0), now=t + 0.1)
        t += 0.1
    p = est.update((0, 0, 0.1), (0, 1), now=t + 2.0)
    assert p.heading_source == "odom" and p.gyro_age_s > 1.0
    assert p.theta == pytest.approx(0.1)


def test_two_cameras_with_different_biases_agree():
    est, geoms, taps = make(roles=("near", "far"))
    # Give each camera a different pitch and bias.
    geoms[0] = small_geometry("near", 1.27, -66.0)
    est = PoseEstimator(geoms, taps, PoseConfig(bias_min_samples=50))
    t = 0.0
    for _ in range(10):
        feed(taps, [geoms[0]], 0.0, t, 20, bias=0.05)
        feed(taps, [geoms[1]], 0.0, t, 20, bias=-0.03)
        p = est.update((0, 0, 0), (0, 0), now=t + 0.1)
        t += 0.1
    assert p.calibrated
    for _ in range(10):
        feed(taps, [geoms[0]], 0.8, t, 20, bias=0.05)
        feed(taps, [geoms[1]], 0.8, t, 20, bias=-0.03)
        p = est.update((0, 0, 0), (0, 0.8), now=t + 0.1)
        t += 0.1
    assert p.theta == pytest.approx(0.8, abs=0.03)
