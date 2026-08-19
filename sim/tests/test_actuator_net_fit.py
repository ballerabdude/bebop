"""Tests for the actuator-net fit tool's data pipeline (numpy-only parts)."""

import numpy as np

from bebop_training.tools.actuator_net_fit import (
    HISTORY_LENGTH,
    NOLOAD_VEL,
    STALL_TORQUE,
    Maneuver,
    analytic_baseline,
    build_samples,
    load_logs,
    parse_log_filename,
)


def test_parse_log_filename_splits_joint_and_maneuver():
    from pathlib import Path

    assert parse_log_filename(Path("sysid_foot_right_joint_torque_chirp.csv")) == (
        "foot_right_joint",
        "torque_chirp",
    )
    assert parse_log_filename(Path("sysid_hip_abduction_left_joint_stall_torque.csv")) == (
        "hip_abduction_left_joint",
        "stall_torque",
    )
    assert parse_log_filename(Path("other_file.csv")) is None
    assert parse_log_filename(Path("sysid_noman.euver.csv")) is None


def _maneuver(cmd, vel, tau, model="RS02", joint="foot_right_joint", kind="torque_step"):
    n = len(cmd)
    return Maneuver(
        joint=joint,
        model=model,
        kind=kind,
        t=np.arange(n) * 0.005,
        cmd_tau=np.asarray(cmd, dtype=float),
        fb_vel=np.asarray(vel, dtype=float),
        fb_tau=np.asarray(tau, dtype=float),
    )


def test_build_samples_windowing_and_normalization():
    # 7 samples -> 7 - (H-1) = 5 windows. Distinct values make order checkable.
    cmd = np.arange(7, dtype=float) + 1.0  # 1..7
    vel = np.arange(7, dtype=float) + 11.0  # 11..17
    tau = np.arange(7, dtype=float) + 21.0  # 21..27
    m = _maneuver(cmd, vel, tau)
    X, y, mids = build_samples([m], "RS02", window=3)

    stall = STALL_TORQUE["RS02"]
    noload = NOLOAD_VEL["RS02"]
    assert X.shape == (5, 6)
    # First window covers samples 0..2, newest first.
    np.testing.assert_allclose(X[0, 0:3], np.array([3.0, 2.0, 1.0]) / stall)
    np.testing.assert_allclose(X[0, 3:6], np.array([13.0, 12.0, 11.0]) / noload)
    assert y[0] == 23.0 / stall  # target is the newest sample's fb_tau
    assert (mids == 0).all()


def test_build_samples_rejects_unknown_model():
    m = _maneuver([1] * 10, [1] * 10, [1] * 10)
    try:
        build_samples([m], "RS99")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown model")


def test_analytic_baseline_clips_like_dc_envelope():
    # Zero velocity: passes small commands, clips at effort_limit_sim (17 for RS02).
    pred = analytic_baseline(np.array([5.0, 30.0, -30.0]), np.zeros(3), "RS02")
    np.testing.assert_allclose(pred, [5.0, 17.0, -17.0])
    # At +no-load speed the positive rail collapses to ~0.
    pred = analytic_baseline(np.array([10.0]), np.array([NOLOAD_VEL["RS02"]]), "RS02")
    assert pred[0] <= 1e-6


def test_load_logs_resamples_and_groups(tmp_path):
    # 25 rows at 125 Hz (8 ms) -> ~0.2 s -> ~40 samples at 200 Hz.
    rows = ["t_s,maneuver,joint,model,cmd_pos,cmd_vel,cmd_tau,cmd_kp,cmd_kd,fb_pos,fb_vel,fb_tau,fb_temp"]
    for i in range(25):
        rows.append(f"{i * 0.008},torque-chirp,foot_right_joint,RS02,0,0,1.0,0,0,0,2.0,3.0,30")
    (tmp_path / "sysid_foot_right_joint_torque_chirp.csv").write_text("\n".join(rows))

    maneuvers = load_logs(tmp_path)
    assert len(maneuvers) == 1
    m = maneuvers[0]
    assert m.model == "RS02"
    assert m.kind == "torque_chirp"
    dt = np.diff(m.t)
    np.testing.assert_allclose(dt, 1.0 / 200.0, rtol=1e-6)
    np.testing.assert_allclose(m.cmd_tau, 1.0)
    np.testing.assert_allclose(m.fb_vel, 2.0)
    np.testing.assert_allclose(m.fb_tau, 3.0)

    # And the samples builder accepts the loaded maneuver.
    X, y, _ = build_samples(maneuvers, "RS02")
    assert X.shape[1] == 2 * HISTORY_LENGTH
    assert len(y) == X.shape[0]
