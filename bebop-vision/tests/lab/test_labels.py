"""labels.py against the synthetic sessions' known trajectory.

The Session here is a minimal stand-in built straight from the MCAP with
``mcap_tools`` (the real ``bebop_vision.lab.session`` is written separately);
it offers exactly what labels.py reads: ``series(name)``, ``ticks()``,
``calib``, ``name`` and ``path``. A second stand-in takes arrays directly so
still periods, recovery starts and stalls can be staged without a video
encoder.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

import synth
from bebop_vision.lab import labels
from bebop_vision.lab.types import SERIES_DTYPES, Calib, Path2D, Sample, Series
from bebop_vision.mcap_tools import ProtoDecoder, iter_messages, read_info

KW = dict(duration_s=4.0, hz=20.0, v=0.3, w=0.4)


# --------------------------------------------------------------------------
# Session stand-ins
# --------------------------------------------------------------------------

class _Ticks:
    def __init__(self, t_ns):
        self.t_ns = np.asarray(t_ns, dtype=np.int64)

    def __len__(self):
        return len(self.t_ns)


class FakeSession:
    """Series + calib + raw ticks from arrays."""

    def __init__(self, name, series, calib, tick_t_ns, path=None):
        self.name = name
        self.path = Path(path) if path is not None else Path(f"/nonexistent/{name}.mcap")
        self._series = series
        self.calib = calib
        self._ticks = np.asarray(tick_t_ns, dtype=np.int64)

    def series(self, name):
        return self._series.get(name) or Series.empty(name)

    def ticks(self, hz=None):
        if hz is None:
            return _Ticks(self._ticks)
        stride = max(1, int(round(20.0 / hz)))
        return _Ticks(self._ticks[::stride])

    @classmethod
    def from_mcap(cls, path):
        path = Path(path)
        info = read_info(path)
        dec = ProtoDecoder()
        rows = {k: [] for k in ("odom", "cmd_vel", "imu_gyro_near", "imu_gyro_far")}
        depth_t, calib_json = [], None
        topics = ["/odom", "/cmd_vel", "/imu_gyro_near", "/imu_gyro_far", "/depth_near", "/calib"]
        for schema, channel, msg in iter_messages(path, topics, footerless=info.footerless):
            t = int(msg.log_time)
            if channel.topic == "/depth_near":
                depth_t.append(t)
            elif channel.topic == "/calib":
                calib_json = json.loads(msg.data)
            else:
                m = dec.decode(schema, msg.data)
                if channel.topic == "/odom":
                    rows["odom"].append((t, m.x, m.y, m.theta))
                elif channel.topic == "/cmd_vel":
                    rows["cmd_vel"].append((t, m.vx, m.wz))
                else:
                    rows[channel.topic[1:]].append((t, m.device_stamp_us, m.gx, m.gy, m.gz))
        series = {k: Series(np.array(v, dtype=SERIES_DTYPES[k]), k) for k, v in rows.items()}
        mounts = {role: dict(calib_json["mounts"][serial]) for role, serial in synth.SERIALS.items()}
        calib = Calib(roles=dict(synth.SERIALS), mounts=mounts, recorded=calib_json)
        return cls(path.stem, series, calib, sorted(depth_t), path=path)


def _arc_session(name, *, segments, hz=20.0, w=0.4, imu_hz=200.0, gyro_bias=0.0,
                 odom_hole=None, gyro_hole=None, with_gyro=True):
    """Arrays for piecewise-constant forward speed ``segments`` [(duration_s, vx)].

    Odom and cmd at ``hz``; the near gyro at ``imu_hz`` in the optical frame
    with an additive body-yaw bias; ``odom_hole=(t0, t1)`` drops odom and cmd
    samples inside (a recorder stall), ``gyro_hole`` likewise for the gyro.
    Returns (session, truth dict).
    """
    n = int(round(sum(d for d, _ in segments) * hz))
    t = np.arange(n) / hz
    vx = np.zeros(n)
    acc = 0.0
    for d, v in segments:
        vx[(t >= acc - 1e-9) & (t < acc + d - 1e-9)] = v
        acc += d
    wz = np.where(vx != 0.0, w, 0.0)                 # still segments do not turn
    theta = np.concatenate([[0.0], np.cumsum(wz[:-1] / hz)])
    x = np.concatenate([[0.0], np.cumsum(vx[:-1] / hz * np.cos(theta[:-1]))])
    y = np.concatenate([[0.0], np.cumsum(vx[:-1] / hz * np.sin(theta[:-1]))])
    t_ns = synth.T0_NS + np.round(t * 1e9).astype(np.int64)
    keep = np.ones(n, bool)
    if odom_hole is not None:
        keep &= ~((t > odom_hole[0]) & (t < odom_hole[1]))
    odom = np.zeros(int(keep.sum()), SERIES_DTYPES["odom"])
    odom["t_ns"], odom["x"], odom["y"], odom["theta"] = t_ns[keep], x[keep], y[keep], theta[keep]
    cmd = np.zeros(int(keep.sum()), SERIES_DTYPES["cmd_vel"])
    cmd["t_ns"], cmd["vx"], cmd["wz"] = t_ns[keep], vx[keep], wz[keep]
    series = {"odom": Series(odom, "odom"), "cmd_vel": Series(cmd, "cmd_vel")}
    if with_gyro:
        R = synth.mount_rotation(synth.MOUNTS["near"]["pitch_deg"], 0.0)
        n_imu = int(round(sum(d for d, _ in segments) * imu_hz))
        tg = np.arange(n_imu) / imu_hz
        # piecewise constant like the odom integration (a linear ramp across
        # a speed step would integrate half a tick of extra heading); a sample
        # at tg is the rate over the interval ending at tg, which is how the
        # integrator attributes it
        wz_g = wz[np.clip(np.searchsorted(t, tg - 1e-9, side="right") - 1, 0, n - 1)]
        # body (0, 0, wz + bias) -> optical = R.T @ body
        g_opt = (R.T @ np.stack([np.zeros_like(wz_g), np.zeros_like(wz_g), wz_g + gyro_bias]))
        gk = np.ones(n_imu, bool)
        if gyro_hole is not None:
            gk &= ~((tg > gyro_hole[0]) & (tg < gyro_hole[1]))
        gyro = np.zeros(int(gk.sum()), SERIES_DTYPES["imu_gyro_near"])
        gyro["t_ns"] = synth.T0_NS + np.round(tg[gk] * 1e9).astype(np.int64)
        gyro["device_us"] = np.round(tg[gk] * 1e6).astype(np.int64)
        gyro["x"], gyro["y"], gyro["z"] = g_opt[0][gk], g_opt[1][gk], g_opt[2][gk]
        series["imu_gyro_near"] = Series(gyro, "imu_gyro_near")
    calib = Calib(roles=dict(synth.SERIALS), mounts={r: dict(m) for r, m in synth.MOUNTS.items()})
    sess = FakeSession(name, series, calib, t_ns[keep])
    return sess, {"t": t, "x": x, "y": y, "theta": theta, "vx": vx, "wz": wz}


def _truth_body_path(truth, ti, rel_t):
    """Ground-truth body-frame path of the tick at time ti (linear interp)."""
    t = truth["t"]
    xi, yi, thi = (np.interp(ti, t, truth["x"]), np.interp(ti, t, truth["y"]),
                   np.interp(ti, t, truth["theta"]))
    tq = ti + rel_t
    xs, ys = np.interp(tq, t, truth["x"]), np.interp(tq, t, truth["y"])
    bx, by = labels.to_body(xi, yi, thi, xs, ys)
    return np.stack([bx, by], -1), np.interp(tq, t, truth["theta"]) - thi


@pytest.fixture(scope="module")
def jpeg_sess(jpeg_session):
    return FakeSession.from_mcap(jpeg_session)


@pytest.fixture(scope="module")
def reverse_sess(reverse_session):
    return FakeSession.from_mcap(reverse_session)


# --------------------------------------------------------------------------
# fused_pose
# --------------------------------------------------------------------------

def test_fused_pose_matches_truth(jpeg_sess):
    truth = synth.expected_trajectory(**KW)
    pose = labels.fused_pose(jpeg_sess, labels.HindsightConfig())
    assert pose.fields == ("t_ns", "x", "y", "theta_fused", "theta_odom", "still", "gap")
    assert len(pose) == len(truth["t"])
    err_deg = np.degrees(np.abs(pose["theta_fused"] - truth["theta"]))
    assert err_deg.max() < 0.5
    assert np.abs(pose["x"] - truth["x"]).max() < 1e-3
    assert np.abs(pose["y"] - truth["y"]).max() < 1e-3
    assert np.allclose(pose["theta_odom"], truth["theta"])
    assert not pose["still"].any()      # always moving
    assert not pose["gap"].any()        # no stalls, continuous gyro


def test_fused_pose_odom_heading(jpeg_sess):
    pose = labels.fused_pose(jpeg_sess, labels.HindsightConfig(heading="odom"))
    assert np.array_equal(pose["theta_fused"], pose["theta_odom"])


def test_fused_pose_falls_back_without_gyro():
    sess, truth = _arc_session("nogyro", segments=[(4.0, 0.3)], with_gyro=False)
    pose, meta = labels._fuse(sess, labels.HindsightConfig())
    assert meta["heading_source"] == "odom"
    assert np.allclose(pose["theta_fused"], truth["theta"])
    assert np.abs(pose["x"] - truth["x"]).max() < 1e-9


def test_fused_pose_uses_far_gyro_when_near_missing(jpeg_sess):
    series = dict(jpeg_sess._series)
    series.pop("imu_gyro_near")
    sess = FakeSession("farOnly", series, jpeg_sess.calib, jpeg_sess._ticks)
    pose, meta = labels._fuse(sess, labels.HindsightConfig())
    assert meta["gyro_role"] == "far"
    truth = synth.expected_trajectory(**KW)
    assert np.degrees(np.abs(pose["theta_fused"] - truth["theta"])).max() < 0.5


def test_gyro_bias_estimated_from_still_period():
    # 2 s still (cmd zero, odom frozen) then 3 s of arc; the gyro carries a
    # 3 deg/s bias that would put the heading 9 deg off at the end.
    bias = math.radians(3.0)
    sess, truth = _arc_session("bias", segments=[(2.0, 0.0), (3.0, 0.3)], gyro_bias=bias)
    pose, meta = labels._fuse(sess, labels.HindsightConfig())
    assert meta["heading_source"] == "gyro"
    assert meta["still_gyro_samples"] >= 100
    assert abs(meta["bias_rad_s"] - bias) < 1e-6
    assert np.degrees(np.abs(pose["theta_fused"] - truth["theta"])).max() < 0.05
    still = pose["still"]
    t = (pose.t_ns - pose.t_ns[0]) * 1e-9
    assert still[t < 1.4].all()              # well inside the still stretch
    assert not still[t > 2.6].any()          # moving
    # without the correction the heading drifts
    pose_raw, _ = labels._fuse(sess, labels.HindsightConfig(min_still_samples=10 ** 6))
    assert np.degrees(abs(pose_raw["theta_fused"][-1] - truth["theta"][-1])) > 5.0


def test_fused_pose_flags_gaps():
    sess, truth = _arc_session("gaps", segments=[(5.0, 0.3)], odom_hole=(1.0, 1.7), gyro_hole=(3.0, 3.1))
    pose = labels.fused_pose(sess)
    t = (pose.t_ns - pose.t_ns[0]) * 1e-9
    gap = pose["gap"]
    # the odom sample right after the stall carries the flag
    k = int(np.searchsorted(t, 1.7 - 1e-6))
    assert gap[k]
    # every odom step overlapping the gyro gap (3.0, 3.1) carries the flag:
    # the sample at 3.05 lies inside it (its heading is np.interp's straight
    # line across the hole, 0.57 deg off here) and the step 3.05 -> 3.1 ends it
    flagged = sorted(round(float(v), 2) for v in t[gap])
    assert flagged == [1.7, 3.05, 3.1]
    err = np.degrees(pose["theta_fused"] - np.interp(t, truth["t"], truth["theta"]))
    assert abs(err[int(np.searchsorted(t, 3.05 - 1e-6))]) > 0.3
    # the sample on the gap's start is not inside it
    assert not gap[int(np.searchsorted(t, 3.0 - 1e-6))]


# --------------------------------------------------------------------------
# gyro clock handling
# --------------------------------------------------------------------------

def _with_gyro(sess, **fields):
    """Copy of ``sess`` with the near gyro's fields replaced."""
    g = sess._series["imu_gyro_near"].data.copy()
    for k, v in fields.items():
        g[k] = v
    series = dict(sess._series)
    series["imu_gyro_near"] = Series(g, "imu_gyro_near")
    return FakeSession(sess.name, series, sess.calib, sess._ticks)


def _bursty_log_times(device_us, floor_s=0.002, burst=7, host_of=None):
    """Host arrival stamps the SDK's batching produces: every ``burst`` samples
    arrive together, ``floor_s`` after the last one's device stamp (the
    irreducible delivery latency). ``host_of`` maps device seconds to host
    seconds (identity: the two clocks agree). 7 samples (35 ms) is not a
    multiple of the 50 ms odom period, so the odom samples see every phase
    of a burst, as on the robot."""
    t_dev = device_us * 1e-6
    last = (np.arange(len(t_dev)) // burst + 1) * burst - 1
    last = np.minimum(last, len(t_dev) - 1)
    host = host_of(t_dev[last]) if host_of is not None else t_dev[last]
    return synth.T0_NS + np.round((host + floor_s) * 1e9).astype(np.int64)


def test_gyro_placed_on_host_axis_by_device_time():
    # bursty, late arrivals: integrating on the device dt is not enough, the
    # samples must also sit at device time + offset on the odom axis, or the
    # heading lags by the arrival excess (0.4 rad/s * 0..30 ms = up to 0.7 deg).
    # Only the 2 ms floor stays (it is indistinguishable from the clock
    # offset): 0.4 rad/s * 2 ms = 0.05 deg.
    sess, truth = _arc_session("burst", segments=[(5.0, 0.3)])
    g = sess._series["imu_gyro_near"].data
    sess_b = _with_gyro(sess, t_ns=_bursty_log_times(g["device_us"]))
    pose, meta = labels._fuse(sess_b, labels.HindsightConfig())
    assert meta["gyro_clock"] == "device" and meta["gyro_placement"] == "device"
    assert meta["gyro_gaps"] == 0                      # device dt is clean
    err = np.degrees(pose["theta_fused"] - truth["theta"])
    assert np.abs(err).max() < 0.1
    # the fit sits on the latency floor and reports the batching excess
    assert abs(meta["gyro_host_offset_s"] - 0.002) < 0.001
    assert 0.01 < meta["gyro_arrival_excess_p50_s"] < 0.05
    # the truncated last burst biases the slope over a 5 s file (0.5 ms at
    # the end); a real multi-minute recording dilutes it
    assert abs(meta["gyro_drift_ms_per_min"]) < 10.0
    assert not pose["gap"].any()
    # the same arrivals placed at log time (what a device clock reset forces)
    # carry the batching into the heading
    dev = g["device_us"].copy()
    dev[1] = dev[0] - 1
    pose_l, meta_l = labels._fuse(_with_gyro(sess_b, device_us=dev), labels.HindsightConfig())
    assert meta_l["gyro_placement"] == "log"
    err_l = np.degrees(pose_l["theta_fused"] - truth["theta"])
    assert np.abs(err_l).max() > 0.3


def test_gyro_clock_drift_is_fitted():
    # the host clock runs 2 % fast against the camera (the 2026-10-18
    # sessions drift -7.5..+10.5 ms/min; stretched so 5 s shows it): odom,
    # cmd and the ticks carry host stamps, the gyro its device stamps plus a
    # bursty host arrival. The heading at odom sample i must still be the
    # truth at sample i; without the drift fit the gyro would land up to
    # 0.1 s off at the end (2.3 deg).
    sess, truth = _arc_session("drift", segments=[(5.0, 0.3)])

    def host_of(t_dev):
        return t_dev * 1.02 + 0.03

    series = {}
    for name in ("odom", "cmd_vel"):
        a = sess._series[name].data.copy()
        a["t_ns"] = synth.T0_NS + np.round(host_of((a["t_ns"] - synth.T0_NS) * 1e-9) * 1e9).astype(np.int64)
        series[name] = Series(a, name)
    g = sess._series["imu_gyro_near"].data.copy()
    g["t_ns"] = _bursty_log_times(g["device_us"], host_of=host_of)
    series["imu_gyro_near"] = Series(g, "imu_gyro_near")
    sess_d = FakeSession("drift", series, sess.calib, series["odom"].t_ns)
    pose, meta = labels._fuse(sess_d, labels.HindsightConfig())
    assert meta["gyro_placement"] == "device"
    assert abs(meta["gyro_drift_ms_per_min"] - 0.02 * 60e3) < 12.0     # within 1 %
    err = np.degrees(pose["theta_fused"] - truth["theta"])
    assert np.abs(err).max() < 0.1


def test_gyro_duplicate_stamp_is_not_a_fallback():
    # one repeated device stamp (the reviewer's reproduction) must not flip
    # the file to bursty log time: no new gaps, device placement kept
    sess, truth = _arc_session("dup", segments=[(5.0, 0.3)])
    g = sess._series["imu_gyro_near"].data
    dev = g["device_us"].copy()
    dev[500] = dev[499]
    sess_dup = _with_gyro(sess, device_us=dev, t_ns=_bursty_log_times(dev))
    pose, meta = labels._fuse(sess_dup, labels.HindsightConfig())
    assert meta["gyro_clock"] == "device" and meta["gyro_placement"] == "device"
    assert meta["gyro_clock_duplicates"] == 1 and meta["gyro_clock_resets"] == 0
    assert meta["gyro_gaps"] == 0
    err = np.degrees(pose["theta_fused"] - truth["theta"])
    assert np.abs(err).max() < 0.2          # one 5 ms sample's rotation at most


def test_gyro_device_clock_reset_falls_back_per_step_and_is_recorded():
    sess, truth = _arc_session("reset", segments=[(5.0, 0.3)])
    g = sess._series["imu_gyro_near"].data
    dev = g["device_us"].copy()
    dev[600:] -= 10 ** 9                      # camera reconnect: clock restarts
    sess_r = _with_gyro(sess, device_us=dev)  # log times stay clean here
    pose, meta = labels._fuse(sess_r, labels.HindsightConfig())
    assert meta["gyro_clock"] == "mixed" and meta["gyro_placement"] == "log"
    assert meta["gyro_clock_resets"] == 1
    assert meta["gyro_gaps"] == 0             # the reset step took the 5 ms log dt
    err = np.degrees(pose["theta_fused"] - truth["theta"])
    assert np.abs(err).max() < 0.1
    assert not pose["gap"].any()


def test_gyro_absent_device_clock_uses_log_time():
    sess, truth = _arc_session("nodev", segments=[(5.0, 0.3)])
    sess_0 = _with_gyro(sess, device_us=0)
    pose, meta = labels._fuse(sess_0, labels.HindsightConfig())
    assert meta["gyro_clock"] == "log" and meta["gyro_placement"] == "log"
    assert meta["gyro_clock_resets"] == 0
    assert "gyro_host_offset_s" not in meta
    err = np.degrees(pose["theta_fused"] - truth["theta"])
    assert np.abs(err).max() < 0.1            # synthetic log times are exact


# --------------------------------------------------------------------------
# hindsight_paths
# --------------------------------------------------------------------------

def test_hindsight_paths_match_true_arc(jpeg_sess):
    truth = synth.expected_trajectory(**KW)
    cfg = labels.HindsightConfig(horizon_s=3.0, dt_s=0.2)
    ls = labels.hindsight_paths(jpeg_sess, jpeg_sess.ticks(), cfg)
    assert ls.variant == "hindsight" and ls.version == labels.LABELS_VERSION
    assert len(ls) == 80 and ls.pts.shape == (80, 15, 2)
    assert np.allclose(ls.rel_t, np.arange(1, 16) * 0.2)
    # the recording ends at 3.95 s: ticks up to 0.95 s see the whole horizon
    assert ls.valid.sum() == 20
    assert ls.valid[:20].all() and not ls.valid[20:].any()
    assert np.isnan(ls.pts[20:]).all()
    for i in range(20):
        ti = i / 20.0
        pts_true, psi_true = _truth_body_path(truth, ti, ls.rel_t)
        assert np.abs(ls.pts[i] - pts_true).max() < 0.01
        assert np.abs(ls.psi[i] - psi_true).max() < 1e-3
        assert np.allclose(ls.v[i], 0.3, atol=1e-3)
    # a left turn (w > 0) bends to +y
    assert ls.pts[0, -1, 1] > 0.3
    assert not ls.flags["reverse_window"].any()
    assert not ls.flags["reversing"].any()
    assert np.isnan(ls.flags["truncated_at_s"]).all()
    assert not ls.flags["stall_gap"].any()
    assert not ls.flags["recovery_start"].any()
    p = ls.path(0)
    assert isinstance(p, Path2D) and p.pts.shape == (1, 15, 2) and p.psi.shape == (1, 15)
    assert ls.path(30) is None
    g = ls.get(int(ls.t_ns[3]))
    assert g["index"] == 3 and g["valid"] and isinstance(g["path"], Path2D)
    assert ls.get(int(ls.t_ns[3]) + 1) is None


def test_hindsight_accepts_arrays_and_subsampled_ticks(jpeg_sess):
    ls = labels.hindsight_paths(jpeg_sess, jpeg_sess.ticks(hz=5.0))
    assert len(ls) == 20
    ls2 = labels.hindsight_paths(jpeg_sess, np.asarray(jpeg_sess.ticks(hz=5.0).t_ns))
    assert np.array_equal(ls.t_ns, ls2.t_ns)
    assert np.array_equal(ls.valid, ls2.valid)


def test_reverse_truncation(reverse_sess):
    truth = synth.expected_trajectory(reverse_from_s=2.5, **KW)
    cfg = labels.HindsightConfig(horizon_s=3.0, dt_s=0.2)
    ls = labels.hindsight_paths(reverse_sess, reverse_sess.ticks(), cfg)
    f = ls.flags
    t = (ls.t_ns - ls.t_ns[0]) * 1e-9
    # ticks from 2.5 s on are reversing: not inputs
    rev = t >= 2.5 - 1e-9
    assert f["reversing"][rev].all() and not f["reversing"][~rev].any()
    assert not ls.valid[rev].any()
    # every earlier tick whose 3 s window reaches 2.5 s is flagged and truncated
    fwd_valid = (~rev) & (t + 3.0 <= 3.95 + 1e-9)
    assert ls.valid[fwd_valid].all()
    assert f["reverse_window"][~rev].all()          # all windows reach 2.5 s here
    trunc = f["truncated_at_s"]
    assert np.allclose(trunc[fwd_valid], 2.5 - t[fwd_valid], atol=1e-6)
    assert np.isnan(trunc[rev]).all()
    # tick at 0.5 s: rel 2.0 is the last kept sample; the rest hold the stop pose
    i = 10
    assert abs(trunc[i] - 2.0) < 1e-6
    kept = ls.rel_t <= 2.0 + 1e-9
    pts_true, _ = _truth_body_path(truth, 0.5, ls.rel_t)
    assert np.abs(ls.pts[i][kept] - pts_true[kept]).max() < 0.01
    stop = ls.pts[i][kept][-1]
    assert np.allclose(ls.pts[i][~kept], stop, atol=1e-6)
    assert np.allclose(ls.v[i][kept], 0.3, atol=1e-3)
    assert np.all(ls.v[i][~kept] == 0.0)
    # the held heading is the heading at the reverse time
    assert np.allclose(ls.psi[i][~kept], ls.psi[i][kept][-1], atol=1e-6)
    assert ls.census()["truncated"] == int(fwd_valid.sum())


def test_reverse_flag_only_when_truncation_disabled(reverse_sess):
    cfg = labels.HindsightConfig(reverse_truncate=False)
    ls = labels.hindsight_paths(reverse_sess, reverse_sess.ticks(), cfg)
    i = 10
    assert ls.valid[i] and ls.flags["reverse_window"][i]
    assert np.isnan(ls.flags["truncated_at_s"][i])
    # the full path comes back: the end point is closer than the farthest point
    d = np.hypot(ls.pts[i, :, 0], ls.pts[i, :, 1])
    assert d[-1] < d.max() - 0.05
    assert (ls.v[i][ls.rel_t > 2.0] < 0).all()


def test_recovery_start_after_a_reverse():
    # forward 2 s, reverse 1 s, forward 4 s (the tick at 3 s needs 3 s of coverage)
    sess, truth = _arc_session("recover", segments=[(2.0, 0.3), (1.0, -0.3), (4.0, 0.3)])
    ls = labels.hindsight_paths(sess, sess.ticks(), labels.HindsightConfig(recovery_s=1.0))
    f = ls.flags
    t = (ls.t_ns - ls.t_ns[0]) * 1e-9
    reversing = (t >= 2.0 - 1e-9) & (t < 3.0 - 1e-9)
    assert f["reversing"][reversing].all() and not f["reversing"][~reversing].any()
    assert not ls.valid[reversing].any()
    # the first forward second after the reverse is a recovery start
    rec = (t >= 3.0 - 1e-9) & (t <= 3.95 + 1e-9)
    assert f["recovery_start"][rec].all()
    assert not f["recovery_start"][t < 2.0].any()
    assert not f["recovery_start"][t > 4.0 + 1e-9].any()
    # windows before the reverse are truncated at it; the recovery windows
    # see only forward motion and keep the full arc
    assert f["reverse_window"][t < 2.0].all()
    assert not f["reverse_window"][t >= 3.0].any()
    j = int(np.searchsorted(t, 3.0 - 1e-9))
    assert ls.valid[j] and np.isnan(f["truncated_at_s"][j])
    pts_true, _ = _truth_body_path(truth, 3.0, ls.rel_t)
    assert np.abs(ls.pts[j] - pts_true).max() < 0.01


def test_stall_gap_flag_covers_windows_spanning_the_gap():
    sess, _ = _arc_session("stall", segments=[(8.0, 0.3)], odom_hole=(4.0, 4.7))
    # keep the stall windows valid so the flag's coverage can be checked
    ls = labels.hindsight_paths(sess, sess.ticks(),
                                labels.HindsightConfig(horizon_s=3.0, drop_stall_windows=False))
    t = (ls.t_ns - ls.t_ns[0]) * 1e-9
    spans = (t < 4.7) & (t + 3.0 >= 4.7 - 1e-9) & ls.valid
    assert spans.any()
    assert ls.flags["stall_gap"][spans].all()
    # windows that END inside the stall are flagged too: their tail is
    # interpolated across the hole although no sample inside the window
    # carries the pose gap flag (13 ticks here, t in 1.05..1.65)
    ends_inside = (t + 3.0 > 4.0 + 1e-6) & (t + 3.0 < 4.7 - 1e-6) & ls.valid
    assert ends_inside.sum() == 13
    assert ls.flags["stall_gap"][ends_inside].all()
    # a window ending exactly on the last sample before the hole is clean
    assert not ls.flags["stall_gap"][np.abs(t + 3.0 - 4.0) < 1e-6].any()
    assert not ls.flags["stall_gap"][t + 3.0 < 4.0 - 1e-6].any()
    assert not ls.flags["stall_gap"][(t >= 4.7) & ls.valid].any()
    # ticks inside the hole have no pose sample; the tick on its far edge
    # starts a clean window
    assert ls.flags["stall_gap"][np.abs(t - 4.0) < 1e-6].all()


def test_stall_gap_flag_covers_windows_overlapping_a_gyro_gap():
    sess, _ = _arc_session("ggap", segments=[(5.0, 0.3)], gyro_hole=(3.0, 3.1))
    ls = labels.hindsight_paths(sess, sess.ticks(), labels.HindsightConfig(drop_stall_windows=False))
    t = (ls.t_ns - ls.t_ns[0]) * 1e-9
    overlaps = (t < 3.1 - 1e-6) & (t + 3.0 > 3.0 + 1e-6)
    assert ls.flags["stall_gap"][overlaps].all()
    assert not ls.flags["stall_gap"][~overlaps].any()
    # the window (0.05, 3.05] is the reviewer's case: it ends inside the gap
    assert ls.flags["stall_gap"][1] and ls.valid[1]
    assert not ls.flags["stall_gap"][0]      # (0, 3.0]: the gap starts on its edge
    # by default those windows are not labels: interpolated, not measured
    default = labels.hindsight_paths(sess, sess.ticks())
    assert not default.valid[overlaps].any()
    assert (default.valid == (ls.valid & ~ls.flags["stall_gap"])).all()


def test_cmd_reverse_threshold():
    # a few mm/s of backward command (joystick deadzone) while odom drives
    # forward is not a reverse; the literal rule (threshold 0) makes it one
    sess, _ = _arc_session("deadzone", segments=[(5.0, 0.3)])
    cmd = sess._series["cmd_vel"].data.copy()
    cmd["vx"][40:43] = -0.004                 # t = 2.0 .. 2.1
    sess._series["cmd_vel"] = Series(cmd, "cmd_vel")
    ls = labels.hindsight_paths(sess, sess.ticks())
    assert not ls.flags["reversing"].any()
    assert not ls.flags["reverse_window"].any()
    assert ls.valid[:40].all()
    ls0 = labels.hindsight_paths(sess, sess.ticks(), labels.HindsightConfig(reverse_cmd_m_s=0.0))
    t = (ls0.t_ns - ls0.t_ns[0]) * 1e-9
    assert ls0.flags["reversing"][(t >= 2.0 - 1e-9) & (t < 2.15 - 1e-9)].all()
    assert ls0.flags["reverse_window"][t < 2.0].all()
    assert labels.HindsightConfig().reverse_cmd_m_s == 0.01


def test_path_accessor_copies_and_indexes():
    sess, _ = _arc_session("acc", segments=[(4.0, 0.3)])
    ls = labels.hindsight_paths(sess, sess.ticks())
    # both spellings of the contract work, slices and iteration too
    p_call, p_idx = ls.path(0), ls.path[0]
    assert isinstance(p_call, Path2D) and isinstance(p_idx, Path2D)
    assert np.array_equal(p_call.pts, p_idx.pts)
    assert len(ls.path) == len(ls) and len(ls.path[0:3]) == 3
    assert sum(p is None for p in ls.path) == int((~ls.valid).sum())
    assert ls.path[-1] is None                 # the last tick has no horizon
    # editing a returned path leaves the label set untouched
    before = ls.pts[0].copy(), ls.psi[0].copy(), ls.v[0].copy(), ls.rel_t.copy()
    p_idx.pts[0, 0, 0] = 999.0
    p_idx.psi[0, 0] = 999.0
    p_idx.v[0, 0] = 999.0
    p_idx.rel_t[0] = 999.0
    assert np.array_equal(ls.pts[0], before[0]) and np.array_equal(ls.psi[0], before[1])
    assert np.array_equal(ls.v[0], before[2]) and np.array_equal(ls.rel_t, before[3])
    assert ls.get(int(ls.t_ns[0]))["path"].pts[0, 0, 0] != 999.0


def test_still_flag_at_tick():
    sess, _ = _arc_session("still", segments=[(2.0, 0.0), (3.0, 0.3)])
    ls = labels.hindsight_paths(sess, sess.ticks())
    t = (ls.t_ns - ls.t_ns[0]) * 1e-9
    assert ls.flags["still"][t < 1.4].all()
    assert not ls.flags["still"][t > 2.6].any()
    # standing still then driving off is a legal label: the first points sit at the origin
    i = 0
    assert ls.valid[i]
    assert np.hypot(*ls.pts[i, 0]) < 1e-6
    assert np.hypot(*ls.pts[i, -1]) > 0.2


# --------------------------------------------------------------------------
# persistence
# --------------------------------------------------------------------------

def test_labelset_roundtrip(jpeg_sess, tmp_path):
    ls = labels.hindsight_paths(jpeg_sess, jpeg_sess.ticks())
    d = ls.save(tmp_path / "hs")
    assert (d / "labels.npz").is_file() and (d / "meta.json").is_file()
    meta = json.loads((d / "meta.json").read_text())
    assert meta["variant"] == "hindsight" and meta["version"] == labels.LABELS_VERSION
    assert meta["producer"] == labels.PRODUCER == f"labels@{labels.LABELS_VERSION}"
    for key in ("gyro_clock", "gyro_placement", "gyro_clock_resets", "gyro_gaps"):
        assert key in meta["fused"]
    assert meta["identity"] and meta["identity"] == labels._identity(jpeg_sess)
    assert meta["source"] == str(jpeg_sess.path)
    assert meta["census"]["valid"] == 20 and meta["config"]["horizon_s"] == 3.0
    back = labels.LabelSet.load(d)
    assert back.variant == ls.variant and back.version == ls.version
    assert np.array_equal(back.t_ns, ls.t_ns)
    assert np.array_equal(back.rel_t, ls.rel_t)
    assert np.array_equal(back.pts, ls.pts, equal_nan=True)
    assert np.array_equal(back.psi, ls.psi, equal_nan=True)
    assert np.array_equal(back.v, ls.v, equal_nan=True)
    assert np.array_equal(back.valid, ls.valid)
    assert back.flags.dtype == ls.flags.dtype
    for name in ls.flags.dtype.names:
        assert np.array_equal(back.flags[name], ls.flags[name], equal_nan=True)
    assert back.meta["producer"] == labels.PRODUCER


def test_label_session_writes_cache_layout_and_reuses(jpeg_sess, lab_cache, monkeypatch):
    monkeypatch.setenv("BEBOP_LAB_CACHE", str(lab_cache))
    ls = labels.label_session(jpeg_sess)
    d = lab_cache / jpeg_sess.name / "labels" / f"hindsight_v{labels.LABELS_VERSION}"
    assert d.is_dir() and (d / "meta.json").is_file()
    assert ls.meta["dir"] == str(d)
    stamp = (d / "labels.npz").stat().st_mtime_ns
    again = labels.label_session(jpeg_sess)
    assert (d / "labels.npz").stat().st_mtime_ns == stamp      # loaded, not rebuilt
    assert np.array_equal(again.t_ns, ls.t_ns)
    # a different config or tick set rebuilds
    labels.label_session(jpeg_sess, cfg=labels.HindsightConfig(horizon_s=2.0))
    assert (d / "labels.npz").stat().st_mtime_ns != stamp
    sub = labels.label_session(jpeg_sess, ticks=jpeg_sess.ticks(hz=5.0))
    assert len(sub) == 20
    # a stale producer rebuilds
    meta = json.loads((d / "meta.json").read_text())
    meta["producer"] = "labels@0"
    (d / "meta.json").write_text(json.dumps(meta))
    stamp = (d / "labels.npz").stat().st_mtime_ns
    labels.label_session(jpeg_sess, ticks=jpeg_sess.ticks(hz=5.0))
    assert json.loads((d / "meta.json").read_text())["producer"] == labels.PRODUCER
    # explicit out_dir is used as the label directory itself
    out = labels.label_session(jpeg_sess, out_dir=lab_cache / "explicit")
    assert (lab_cache / "explicit" / "labels.npz").is_file()
    assert len(out) == 80
    with pytest.raises(ValueError):
        labels.label_session(jpeg_sess, variant="footprint")


def test_label_session_honours_the_sessions_cache_dir(jpeg_sess, lab_cache, monkeypatch):
    # a Session indexed into its own cache_dir gets its labels next to its
    # index, not under the default root
    monkeypatch.setenv("BEBOP_LAB_CACHE", str(lab_cache / "default-root"))
    own = lab_cache / "elsewhere" / jpeg_sess.name
    sess = FakeSession(jpeg_sess.name, jpeg_sess._series, jpeg_sess.calib, jpeg_sess._ticks,
                       path=jpeg_sess.path)
    sess.cache_dir = own
    sub = Path("labels") / f"hindsight_v{labels.LABELS_VERSION}"
    assert labels.label_dir(sess) == own / sub
    ls = labels.label_session(sess)
    assert (own / sub / "labels.npz").is_file()
    assert ls.meta["dir"] == str(own / sub)
    assert not (lab_cache / "default-root").exists()
    # an explicit cache root still wins over the session's own directory
    assert labels.label_dir(sess, cache_root=lab_cache / "r") == lab_cache / "r" / sess.name / sub


# --------------------------------------------------------------------------
# constant-velocity baseline
# --------------------------------------------------------------------------

def test_constant_velocity_path_is_the_exact_arc():
    vx, wz = 0.3, 0.4
    p = labels.constant_velocity_path((vx, wz), horizon_s=3.0, dt_s=0.2)
    assert isinstance(p, Path2D)
    assert p.pts.shape == (1, 15, 2) and np.allclose(p.rel_t, np.arange(1, 16) * 0.2)
    r = vx / wz
    xs = r * np.sin(wz * p.rel_t)
    ys = r * (1 - np.cos(wz * p.rel_t))
    assert np.allclose(p.pts[0, :, 0], xs, atol=1e-6)
    assert np.allclose(p.pts[0, :, 1], ys, atol=1e-6)
    assert np.allclose(p.psi[0], wz * p.rel_t, atol=1e-6)
    assert np.allclose(p.v[0], vx)
    # against a fine Euler integration of the same twist
    dt = 1e-4
    th = np.arange(0, 3.0 + dt / 2, dt) * wz
    x = np.cumsum(vx * dt * np.cos(th))[::2000]
    y = np.cumsum(vx * dt * np.sin(th))[::2000]
    assert np.abs(p.pts[0, :, 0] - x[1:16]).max() < 1e-3
    assert np.abs(p.pts[0, :, 1] - y[1:16]).max() < 1e-3


def test_constant_velocity_path_inputs():
    straight = labels.constant_velocity_path((0.5, 0.0), 2.0, 0.5)
    assert np.allclose(straight.pts[0, :, 0], [0.25, 0.5, 0.75, 1.0])
    assert np.all(straight.pts[0, :, 1] == 0.0)
    zero = labels.constant_velocity_path(None, 1.0, 0.5)
    assert np.all(zero.pts == 0.0)
    rec = np.array([(1, 0.3, 0.4)], dtype=SERIES_DTYPES["cmd_vel"])[0]
    from_rec = labels.constant_velocity_path(rec)
    from_tuple = labels.constant_velocity_path((0.3, 0.4))
    assert np.array_equal(from_rec.pts, from_tuple.pts)
    sample = Sample(t_ns=1, session="s", index=0, data={"cmd_vel": rec, "odom": None})
    assert np.array_equal(labels.constant_velocity_path(sample).pts, from_tuple.pts)
    empty = Sample(t_ns=1, session="s", index=0, data={"cmd_vel": None})
    assert np.all(labels.constant_velocity_path(empty).pts == 0.0)
    # an empty cmd history window (first tick of a session with history_s)
    no_rows = np.empty(0, SERIES_DTYPES["cmd_vel"])
    assert np.all(labels.constant_velocity_path(no_rows).pts == 0.0)
    first = Sample(t_ns=1, session="s", index=0, data={"cmd_vel": no_rows})
    assert np.all(labels.constant_velocity_path(first).pts == 0.0)
    window = np.array([(1, 0.1, 0.0), (2, 0.3, 0.4)], dtype=SERIES_DTYPES["cmd_vel"])
    assert np.array_equal(labels.constant_velocity_path(window).pts, from_tuple.pts)
    series = Series(np.array([(1, 0.1, 0.0), (2, 0.3, 0.4)], dtype=SERIES_DTYPES["cmd_vel"]))
    assert np.array_equal(labels.constant_velocity_path(series).pts, from_tuple.pts)
    assert np.array_equal(labels.constant_velocity_path({"vx": 0.3, "wz": 0.4}).pts, from_tuple.pts)
    # the hindsight label of a constant-twist drive equals the CV arc to a few mm
    sess, _ = _arc_session("cv", segments=[(4.0, 0.3)])
    ls = labels.hindsight_paths(sess, sess.ticks())
    cv = labels.constant_velocity_path((0.3, 0.4))
    assert np.abs(ls.pts[0] - cv.pts[0]).max() < 0.01


def test_module_imports_stay_light():
    import sys
    for mod in ("torch", "av", "rerun", "onnxruntime", "timm", "transformers", "pandas", "polars"):
        assert not any(k == mod or k.startswith(mod + ".") for k in
                       _imported_by_labels()), mod


def _imported_by_labels():
    import subprocess
    import sys
    code = ("import sys; import bebop_vision.lab.labels; "
            "print('\\n'.join(sorted(sys.modules)))")
    out = subprocess.check_output([sys.executable, "-c", code],
                                  cwd=str(Path(__file__).resolve().parents[2]), text=True)
    return out.split()


def test_label_cache_is_rebuilt_when_the_mount_override_changes(jpeg_session, lab_cache):
    from bebop_vision.lab.labels import label_session
    from bebop_vision.lab.session import Session
    plain = Session(jpeg_session, cache_dir=lab_cache)
    a = label_session(plain, ticks=plain.ticks(hz=10))
    again = label_session(Session(jpeg_session, cache_dir=lab_cache), ticks=plain.ticks(hz=10))
    assert again.meta["mounts"] == a.meta["mounts"]
    tilted = Session(jpeg_session, cache_dir=lab_cache, overrides={"near": {"pitch_deg": -45.0}})
    b = label_session(tilted, ticks=tilted.ticks(hz=10))
    assert b.meta["mounts"]["near"]["pitch_deg"] == -45.0
    assert a.meta["mounts"]["near"]["pitch_deg"] != -45.0


def test_forward_driving_stays_forward_when_odom_heading_drifts_past_90_deg():
    # Odom over-reads turns (2x on carpet, AGENTS.md): its heading drifts away
    # from the gyro's while its steps still lie along its own heading. Past
    # 90 deg of drift, signing steps by the fused heading turned forward
    # driving into "reverse" (193619: -117 deg, 353 forward ticks flagged).
    sess, truth = _arc_session("drift", segments=[(1.0, 0.0), (7.0, 0.3)], w=0.4)
    od = sess._series["odom"].data.copy()
    th_od = 2.0 * od["theta"]                           # odom reads twice the turn
    step = np.hypot(np.diff(od["x"]), np.diff(od["y"]))
    od["theta"] = th_od
    od["x"] = np.concatenate([[0.0], np.cumsum(step * np.cos(th_od[:-1]))])
    od["y"] = np.concatenate([[0.0], np.cumsum(step * np.sin(th_od[:-1]))])
    sess._series["odom"] = Series(od, "odom")
    assert np.degrees(th_od[-1] - truth["theta"][-1]) > 120

    ls = labels.hindsight_paths(sess, sess.ticks(), labels.HindsightConfig(horizon_s=3.0, dt_s=0.2))
    assert not ls.flags["reversing"].any()
    assert not ls.flags["reverse_window"].any()
    # late in the drive, past 90 deg of drift, the label is still the true arc
    t = sess.ticks().t_ns
    i = int(np.searchsorted(t, synth.T0_NS + int(4.5e9)))
    lab = ls.get(int(t[i]))
    assert lab["valid"]
    want, _ = _truth_body_path(truth, 4.5, ls.rel_t)
    assert np.abs(lab["path"].best - want).max() < 0.02
