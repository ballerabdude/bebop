"""session.py: index build, series alignment, Ticks sampling and samples
against the synthetic ground truth."""

import json
import time
import warnings

import numpy as np
import pytest

import synth
from bebop_vision.lab import session as sess
from bebop_vision.lab.session import PRODUCER, Session, Ticks, sample_stamps
from bebop_vision.lab.types import SERIES_DTYPES, InputSpec, Sample, Series

HZ = 20.0
N = 80
DT_NS = 50_000_000


@pytest.fixture(scope="module")
def h265(h265_session, tmp_path_factory):
    return Session(h265_session, cache_dir=tmp_path_factory.mktemp("c") / "h265")


@pytest.fixture(scope="module")
def jpeg(jpeg_session, tmp_path_factory):
    return Session(jpeg_session, cache_dir=tmp_path_factory.mktemp("c") / "jpeg")


# --------------------------------------------------------------------------
# sample_stamps
# --------------------------------------------------------------------------

def _reference_sample_stamps(stamps, sample_hz):
    """Verbatim copy of tools/mcap_extract._sample_stamps."""
    if not stamps:
        return []
    step_ns = int(round(1e9 / max(float(sample_hz), 1e-3)))
    out = []
    j = 0
    target = stamps[0]
    last = stamps[-1]
    while target <= last:
        while (j + 1 < len(stamps)
               and abs(stamps[j + 1] - target) <= abs(stamps[j] - target)):
            j += 1
        stamp = stamps[j]
        if not out or stamp != out[-1]:
            out.append(stamp)
        target += step_ns
    return out


class TestSampleStamps:
    @pytest.mark.parametrize("hz", [10.0, 7.0, 21.3, 3.0])
    def test_matches_mcap_extract_on_jittered_ticks(self, hz):
        rng = np.random.default_rng(1)
        # ~21 Hz camera ticks with jitter and a few dropped frame sets
        t = synth.T0_NS + np.cumsum(rng.integers(40_000_000, 56_000_000, size=300))
        t = np.delete(t, [17, 18, 90, 200, 201, 202]).astype(np.int64)
        ref = _reference_sample_stamps([int(x) for x in t], hz)
        got = sample_stamps(t, hz)
        assert got.tolist() == ref
        assert len(np.unique(got)) == len(got)
        assert set(got.tolist()) <= set(t.tolist())

    def test_degenerate(self):
        assert sample_stamps(np.empty(0, np.int64), 10.0).tolist() == []
        assert sample_stamps(np.array([5], np.int64), 10.0).tolist() == [5]
        assert sample_stamps(np.array([5, 6], np.int64), 10.0).tolist() == [5]


# --------------------------------------------------------------------------
# Session / index
# --------------------------------------------------------------------------

class TestIndex:
    def test_build_time(self, h265_session, tmp_path):
        t0 = time.perf_counter()
        s = Session(h265_session, cache_dir=tmp_path / "timed")
        s.index()
        dt = time.perf_counter() - t0
        print(f"\n[lab] indexed the 4 s synthetic H.265 session in {dt * 1e3:.0f} ms "
              f"(builder pass {s.meta['index_seconds'] * 1e3:.0f} ms, "
              f"{s.info.message_count} messages)")
        assert dt < 10.0
        for name in ("meta.json", "series.npz", "ticks.npy", "calib.json", "video_near.bin",
                     "video_near.npy", "video_far.bin", "depth_near.bin", "depth_far.npy"):
            assert (tmp_path / "timed" / name).exists(), name

    def test_basics(self, h265):
        assert h265.name == "navd_session_20260101_000000"
        assert h265.start_ns == synth.T0_NS
        # the last message is the final 200 Hz IMU sample at 3.995 s
        assert h265.end_ns == synth.T0_NS + 799 * 5_000_000
        assert 3.9 < h265.duration_s < 4.1
        assert "/color_near" in h265.topics and "/runtime" in h265.topics
        assert "Session(" in repr(h265)

    def test_meta(self, h265):
        h265.index()   # the index is lazy; this is the first cache access
        meta = json.loads((h265.cache_dir / "meta.json").read_text())
        assert meta["producer"] == PRODUCER
        assert meta["identity"] == sess.file_identity(h265.path)
        assert meta["source"] == str(h265.path)
        assert meta["ticks"] == N and meta["tick_source"] == "/depth_near"
        assert meta["color_fmt"] == {"near": "h265", "far": "h265"}
        assert meta["keyframes"] == {"near": 10, "far": 10}
        assert meta["unknown_topics"] == ["/runtime_events"]
        for key in ("video_near", "video_far", "depth_near", "depth_far"):
            assert meta["payload_bytes"][key] == (h265.cache_dir / f"{key}.bin").stat().st_size

    def test_cache_rebuilt_when_payload_missing_or_truncated(self, h265_session, tmp_path):
        d = tmp_path / "c"
        Session(h265_session, cache_dir=d).index()
        t0 = int(Session(h265_session, cache_dir=d).ticks().t_ns[0])
        (d / "video_near.bin").unlink()
        s = Session(h265_session, cache_dir=d)
        assert s._cache_valid() is None
        assert synth.tick_of_colour_frame(s.frames("near").at(t0)) == 0   # rebuilt, not ValueError
        with open(d / "depth_far.bin", "r+b") as f:
            f.truncate(100)
        assert Session(h265_session, cache_dir=d)._cache_valid() is None
        assert int(Session(h265_session, cache_dir=d).depth("far").at(t0)[0, 0]) == 0

    def test_cache_reused_then_rebuilt_on_identity_change(self, h265_session, tmp_path):
        d = tmp_path / "c"
        Session(h265_session, cache_dir=d).index()
        meta_path = d / "meta.json"
        built = json.loads(meta_path.read_text())["built_at"]
        marker = d / "marker"
        marker.write_text("x")
        # same identity: no rebuild
        s2 = Session(h265_session, cache_dir=d)
        s2.index()
        assert marker.exists() and json.loads(meta_path.read_text())["built_at"] == built
        # a different identity (another recording at the same path) rebuilds
        meta = json.loads(meta_path.read_text())
        meta["identity"] = "0-deadbeef"
        meta_path.write_text(json.dumps(meta))
        time.sleep(1.1)   # built_at has second resolution
        s3 = Session(h265_session, cache_dir=d)
        s3.index()
        fresh = json.loads(meta_path.read_text())
        assert fresh["identity"] == sess.file_identity(h265_session)
        assert fresh["built_at"] != built
        assert len(s3.ticks()) == N

    def test_producer_version_change_rebuilds(self, h265_session, tmp_path):
        d = tmp_path / "c"
        Session(h265_session, cache_dir=d).index()
        meta_path = d / "meta.json"
        meta = json.loads(meta_path.read_text())
        meta["producer"] = "session@0"
        meta_path.write_text(json.dumps(meta))
        Session(h265_session, cache_dir=d).index()
        assert json.loads(meta_path.read_text())["producer"] == PRODUCER

    def test_force_rebuild(self, h265):
        h265.index()
        before = json.loads((h265.cache_dir / "meta.json").read_text())["built_at"]
        time.sleep(1.1)
        h265.index(force=True)
        assert json.loads((h265.cache_dir / "meta.json").read_text())["built_at"] != before
        assert len(h265.ticks()) == N

    def test_cache_root_env(self, monkeypatch, tmp_path, h265_session):
        monkeypatch.setenv("BEBOP_LAB_CACHE", str(tmp_path / "root"))
        s = Session(h265_session)
        assert s.cache_dir == tmp_path / "root" / s.name

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            Session(tmp_path / "nope.mcap")


class TestSeries:
    def test_all_names_exist(self, h265, jpeg):
        for name in SERIES_DTYPES:
            s = h265.series(name)
            assert isinstance(s, Series) and s.data.dtype == SERIES_DTYPES[name]
        with pytest.raises(KeyError):
            h265.series("nope")
        # the JPEG fixture predates runtime and notes: empty, not missing
        for name in ("runtime", "intent", "event", "place"):
            assert len(jpeg.series(name)) == 0
            assert jpeg.series(name).data.dtype == SERIES_DTYPES[name]

    def test_odom_cmd_match_trajectory(self, h265):
        exp = synth.expected_trajectory(**{"duration_s": 4.0, "hz": HZ, "v": 0.3, "w": 0.4})
        odom = h265.series("odom")
        cmd = h265.series("cmd_vel")
        assert len(odom) == N and len(cmd) == N
        np.testing.assert_array_equal(odom.t_ns, synth.T0_NS + np.arange(N) * DT_NS)
        np.testing.assert_allclose(odom["x"], exp["x"], atol=1e-9)
        np.testing.assert_allclose(odom["y"], exp["y"], atol=1e-9)
        np.testing.assert_allclose(odom["theta"], exp["theta"], atol=1e-9)
        np.testing.assert_allclose(cmd["vx"], exp["vx"])
        np.testing.assert_allclose(cmd["wz"], 0.4)

    def test_imu(self, h265):
        g = h265.series("imu_gyro_near")
        a = h265.series("imu_accel_far")
        assert len(g) == 800 and len(a) == 800
        assert abs(g.rate_hz() - 200.0) < 1.0
        assert g["device_us"][1] - g["device_us"][0] == 5000
        # body yaw rate w through the mount rotation (third row)
        R = synth.mount_rotation(-60.0, 0.0)
        wz = (R @ np.stack([g["x"], g["y"], g["z"]]))[2]
        np.testing.assert_allclose(wz, 0.4, atol=1e-9)
        np.testing.assert_allclose(np.linalg.norm(np.stack([a["x"], a["y"], a["z"]]), axis=0), 9.81, atol=1e-9)

    def test_runtime_flattened(self, h265):
        rt = h265.series("runtime")
        assert len(rt) == N     # the mode_changed push on /runtime_events is not a row
        assert (rt["mode"] == 2).all()
        assert (rt["estop"] == 0).all()
        assert (rt["operator_stale"] == 0).all() and (rt["seat_held"] == 1).all()
        np.testing.assert_allclose(rt["wheel_l_target"], 0.3 * 10 - 0.4, rtol=1e-6)
        np.testing.assert_allclose(rt["wheel_r_target"], 0.3 * 10 + 0.4, rtol=1e-6)
        np.testing.assert_allclose(rt["wheel_l_vel"], rt["wheel_l_target"])
        np.testing.assert_allclose(rt["battery_v"], 48.0 - 0.01 * np.arange(N), rtol=1e-5)
        np.testing.assert_allclose(rt["motor_a"], 2.0)
        np.testing.assert_allclose(rt["soc_pct"], 80.0)
        # telemetry is logged 1 ms after the tick
        assert rt.t_ns[0] == synth.T0_NS + 1_000_000

    def test_notes(self, h265):
        intent = h265.series("intent")
        assert len(intent) == 1 and intent["text"][0] == "hall -> kitchen"
        assert intent["from_place"][0] == "hall" and intent["to_place"][0] == "kitchen"
        ev = h265.series("event")
        assert len(ev) == 1 and ev["kind"][0] == "obstacle" and ev["note"][0] == "sock"
        assert ev.t_ns[0] == synth.T0_NS + (N // 3) * DT_NS
        pl = h265.series("place")
        assert len(pl) == 4 and (pl["name"] == "hall").all()
        np.testing.assert_allclose(pl["score"], 0.91, rtol=1e-6)


class TestRuntimeFlattening:
    def test_missing_submessages_and_fields(self):
        from bebop_vision.proto.bebop.runtime.v1 import bebop_runtime_pb2 as pb
        tel = pb.TelemetryFrame(mode=pb.MODE_IDLE)       # no wheels, drive, power
        row = np.array([sess._runtime_row(7, tel)], dtype=SERIES_DTYPES["runtime"])[0]
        assert row["t_ns"] == 7 and row["mode"] == 1 and row["estop"] == 0
        assert row["operator_stale"] == -1 and row["seat_held"] == -1
        for f in ("wheel_l_vel", "wheel_l_target", "wheel_r_vel", "wheel_r_target",
                  "battery_v", "motor_a", "soc_pct"):
            assert np.isnan(row[f]), f
        one = pb.TelemetryFrame(mode=pb.MODE_DIAL_IN)
        w = one.wheels.add()
        w.velocity_rad_s = 1.5
        row = np.array([sess._runtime_row(8, one)], dtype=SERIES_DTYPES["runtime"])[0]
        assert row["wheel_l_vel"] == pytest.approx(1.5) and np.isnan(row["wheel_r_vel"])

    def test_older_schema_without_fields(self):
        """A message class lacking the drive/power/wheels fields entirely
        (older firmware) flattens to NaN / -1 instead of raising."""
        from bebop_vision.proto.navd import telemetry_pb2
        fake = telemetry_pb2.Twist(vx=1.0)   # has none of the telemetry fields
        row = np.array([sess._runtime_row(1, fake)], dtype=SERIES_DTYPES["runtime"])[0]
        assert row["mode"] == -1 and row["estop"] == -1 and np.isnan(row["battery_v"])


class TestCalib:
    def test_from_recorded(self, h265):
        c = h265.calib
        assert c.roles == {"near": "S-NEAR", "far": "S-FAR"}
        assert c.mounts["near"]["pitch_deg"] == -60.0 and c.mounts["far"]["height_m"] == 1.32
        assert c.depth["near"].width == 106 and c.depth["near"].fx == 51.7
        assert c.color == {}       # no orbbec_intrinsics_S-NEAR.json on this machine
        assert c.recorded["recorded_at"] == "2026-10-24T00:00:00"
        assert c.self_mask["near"] == {"rects": 0, "polygons": []}
        cam = c.camera("near", image="depth")
        assert cam.pitch_deg == -60.0
        with pytest.raises(KeyError):
            c.camera("near", image="color")

    def test_overrides_apply_last(self, h265_session, tmp_path):
        s = Session(h265_session, cache_dir=tmp_path / "c", overrides={"near": {"pitch_deg": -59.0}})
        assert s.calib.mount("near")["pitch_deg"] == -59.0
        assert s.calib.mounts["near"]["pitch_deg"] == -60.0
        assert s.calib.camera("near", image="depth").pitch_deg == -59.0

    def test_colour_intrinsics_from_config_file(self, tmp_path):
        recorded = {"intrinsics": {"X1": {"fx": 1, "fy": 1, "cx": 2, "cy": 2, "width": 848, "height": 480},
                                   "X2": {"fx": 1, "fy": 1, "cx": 2, "cy": 2, "width": 848, "height": 480}},
                    "mounts": {"X1": {"height_m": 1.0, "pitch_deg": -9.0, "yaw_deg": 0.0},
                               "X2": {"height_m": 1.2, "pitch_deg": -66.0, "yaw_deg": 0.0}}}
        (tmp_path / "orbbec_intrinsics_X2.json").write_text(json.dumps(
            {"color_fx": 608.6, "color_fy": 608.7, "color_cx": 638.2, "color_cy": 397.4,
             "color_width": 1280, "color_height": 800}))
        c = sess.build_calib(recorded, intrinsics_dir=tmp_path)
        # steeper pitch is the near camera
        assert c.roles == {"near": "X2", "far": "X1"}
        assert c.color["near"].width == 1280 and "far" not in c.color
        u, v, z = c.camera("near").project(np.array([[1.0, 0.0, 0.0]]))
        assert z[0] > 0

    def test_rig_yaml_wins_over_pitch_and_warns_when_broken(self, tmp_path):
        recorded = {"mounts": {"A": {"height_m": 1.0, "pitch_deg": -9.0},
                               "B": {"height_m": 1.2, "pitch_deg": -66.0}}}
        rig = tmp_path / "rig.yaml"
        # the YAML says the steep one is far: the YAML is the authority
        rig.write_text("robots:\n  default:\n    cameras:\n      A: {role: near}\n      B: {role: far}\n")
        assert sess.build_calib(recorded, rig_yaml=rig).roles == {"near": "A", "far": "B"}
        rig.write_text("robots: [unterminated\n")
        with pytest.warns(RuntimeWarning, match="rig config"):
            c = sess.build_calib(recorded, rig_yaml=rig)
        assert c.roles == {"near": "B", "far": "A"}       # the pitch fallback
        rig.write_text("robots:\n  default: {}\n")
        with pytest.warns(RuntimeWarning):
            sess.build_calib(recorded, rig_yaml=rig)
        # an absent YAML (any machine but the robot) is not worth a warning
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            sess.build_calib(recorded, rig_yaml=tmp_path / "none.yaml")

    def test_colour_intrinsics_added_after_index_are_used(self, h265_session, tmp_path, monkeypatch):
        cfg = tmp_path / "config"
        cfg.mkdir()
        monkeypatch.setattr(sess, "_INTRINSICS_DIR", cfg)
        s = Session(h265_session, cache_dir=tmp_path / "c")
        assert s.calib.color == {}
        (cfg / "orbbec_intrinsics_S-FAR.json").write_text(json.dumps(
            {"color_fx": 600.0, "color_fy": 600.0, "color_cx": 640.0, "color_cy": 400.0,
             "color_width": 1280, "color_height": 800}))
        # no rebuild: calib.json still has no colour, the lookup is at load
        s2 = Session(h265_session, cache_dir=tmp_path / "c")
        assert s2._cache_valid() is not None
        assert sorted(s2.calib.color) == ["far"] and s2.calib.color["far"].fx == 600.0
        (cfg / "orbbec_intrinsics_S-NEAR.json").write_text("{not json")
        with pytest.warns(RuntimeWarning, match="colour intrinsics"):
            c = Session(h265_session, cache_dir=tmp_path / "c").calib
        assert "near" not in c.color


class TestFlagsAndTicks:
    def test_flags(self, h265):
        fl = h265.flags()
        assert fl.dtype == sess.FLAGS_DTYPE and len(fl) == N
        assert fl["has_color_near"].all() and fl["has_color_far"].all()
        assert not fl["far_hold_last"].any()
        assert fl["gap_before_ms"][0] == 0.0
        np.testing.assert_allclose(fl["gap_before_ms"][1:], 50.0)

    def test_raw_and_snapped_ticks(self, h265):
        raw = h265.ticks()
        assert isinstance(raw, Ticks) and len(raw) == N and raw.dt_ns == DT_NS
        np.testing.assert_array_equal(raw.t_ns, synth.T0_NS + np.arange(N) * DT_NS)
        ten = h265.ticks(hz=10)
        assert len(ten) == 40 and ten.dt_ns == 100_000_000
        np.testing.assert_array_equal(ten.t_ns, raw.t_ns[::2])
        assert set(ten.t_ns.tolist()) <= set(raw.t_ns.tolist())
        # a non-divisor rate still picks real ticks, no duplicates
        seven = h265.ticks(hz=7)
        assert len(np.unique(seven.t_ns)) == len(seven) == 28
        assert set(seven.t_ns.tolist()) <= set(raw.t_ns.tolist())

    def test_window(self, h265):
        t0 = synth.T0_NS + 10 * DT_NS
        t1 = synth.T0_NS + 20 * DT_NS
        w = h265.ticks(start_ns=t0, end_ns=t1)
        assert len(w) == 11 and w.t_ns[0] == t0 and w.t_ns[-1] == t1

    def test_windowed_snapped_ticks_are_a_subset_of_the_full_grid(self, h265):
        t = h265.ticks().t_ns
        full = set(h265.ticks(hz=10).t_ns.tolist())
        for start in (1, 2, 3, 7):
            win = h265.ticks(hz=10, start_ns=int(t[start]), end_ns=int(t[41])).t_ns.tolist()
            assert set(win) <= full, start
            assert win[0] == int(t[start + start % 2])      # the grid is anchored at the session start
        seven = set(h265.ticks(hz=7).t_ns.tolist())
        assert set(h265.ticks(hz=7, start_ns=int(t[5])).t_ns.tolist()) <= seven
        np.testing.assert_array_equal(h265.grid(10), h265.ticks(hz=10).t_ns)

    def test_flags_off_grid_keep_their_keys(self, h265):
        t = h265.ticks().t_ns
        keys = {"index", "t_ns", "raw_t_ns", "has_color_near", "has_color_far", "far_hold_last", "gap_before_ms"}
        near = Ticks(h265, np.array([int(t[4]) + 7_000_000]))
        fl = near.flags(0)
        assert set(fl) == keys and fl["raw_t_ns"] == int(t[4]) and fl["has_color_near"] is True
        assert near[0].flags["has_color_far"] is True
        far_off = Ticks(h265, np.array([int(t[-1]) + 10_000_000_000]))
        fl = far_off.flags(0)
        assert set(fl) == keys and fl["raw_t_ns"] is None and fl["has_color_near"] is None
        assert far_off[0].flags["gap_before_ms"] is None
        tight = Ticks(h265, np.array([int(t[4]) + 7_000_000]), spec=InputSpec(("odom",), tol_ns=1_000_000))
        assert tight.flags(0)["raw_t_ns"] is None
        on = h265.ticks().flags(4)
        assert on["raw_t_ns"] == on["t_ns"] == int(t[4]) and on["gap_before_ms"] == 50.0

    def test_far_hold_last_on_identical_payloads(self, tmp_path, jpeg_session):
        """The JPEG recorder path re-writes the previous far JPEG bytes for a
        tick without far colour; identical consecutive payloads set the flag."""
        from mcap.reader import make_reader
        from mcap.writer import CompressionType, Writer
        # rewrite the fixture with far colour of ticks 3 and 4 holding tick 2's bytes
        src = jpeg_session
        dst = tmp_path / "hold.mcap"
        with open(src, "rb") as f, open(dst, "wb") as g:
            rd = make_reader(f)
            wr = Writer(g, compression=CompressionType.NONE)
            wr.start(profile="", library="test")
            sch_ids, ch_ids = {}, {}
            far_payloads = []
            for schema, channel, msg in rd.iter_messages():
                if schema is not None and schema.id not in sch_ids:
                    sch_ids[schema.id] = wr.register_schema(schema.name, schema.encoding, schema.data)
                if channel.id not in ch_ids:
                    ch_ids[channel.id] = wr.register_channel(channel.topic, channel.message_encoding,
                                                             sch_ids[schema.id] if schema else 0)
                data = msg.data
                if channel.topic == "/color_far":
                    far_payloads.append(data)
                    k = len(far_payloads) - 1
                    if k in (3, 4):
                        data = far_payloads[2]
                wr.add_message(ch_ids[channel.id], msg.log_time, data, msg.publish_time)
            wr.finish()
        s = Session(dst, cache_dir=tmp_path / "c")
        fl = s.flags()
        assert fl["far_hold_last"].tolist()[:7] == [False, False, False, True, True, False, False]
        assert fl["has_color_far"].all()
        assert s.ticks().flags(3)["far_hold_last"] is True
        assert synth.tick_of_colour_frame(s.frames("far").at(int(fl["t_ns"][4]))) == 2

    def test_odom_fallback_when_no_depth(self, tmp_path, jpeg_session):
        from mcap.reader import make_reader
        from mcap.writer import CompressionType, Writer
        dst = tmp_path / "nodepth.mcap"
        with open(jpeg_session, "rb") as f, open(dst, "wb") as g:
            rd = make_reader(f)
            wr = Writer(g, compression=CompressionType.NONE)
            wr.start(profile="", library="test")
            sch_ids, ch_ids = {}, {}
            for schema, channel, msg in rd.iter_messages():
                if channel.topic.startswith("/depth"):
                    continue
                if schema is not None and schema.id not in sch_ids:
                    sch_ids[schema.id] = wr.register_schema(schema.name, schema.encoding, schema.data)
                if channel.id not in ch_ids:
                    ch_ids[channel.id] = wr.register_channel(channel.topic, channel.message_encoding,
                                                             sch_ids[schema.id] if schema else 0)
                wr.add_message(ch_ids[channel.id], msg.log_time, msg.data, msg.publish_time)
            wr.finish()
        s = Session(dst, cache_dir=tmp_path / "c")
        assert len(s.ticks()) == N and s.meta["tick_source"] == "/odom"
        assert len(s.depth("near")) == 0 and s.depth("near").at(int(s.ticks().t_ns[0])) is None


# --------------------------------------------------------------------------
# Samples
# --------------------------------------------------------------------------

ALL = ("color_near", "color_far", "depth_near", "depth_far", "imu_accel_near", "imu_gyro_near",
       "imu_accel_far", "imu_gyro_far", "odom", "cmd_vel", "runtime", "intent", "place", "event")


class TestSamples:
    def test_no_spec_sample_has_flags_only(self, h265):
        smp = h265.ticks()[5]
        assert isinstance(smp, Sample) and smp.data == {}
        assert smp.flags["has_color_near"] is True and smp.index == 5
        assert smp.session == h265.name and smp.calib is h265.calib

    def test_select_builds_spec(self, h265):
        tk = h265.ticks().select("odom", "color_near", history_s={"color_near": 0.2}, tol_ns=10_000_000)
        assert tk.spec == InputSpec(("odom", "color_near"), {"color_near": 0.2}, 10_000_000)
        assert len(tk) == N and tk.session is h265
        with pytest.raises(ValueError):
            h265.ticks().select("nope")

    def test_full_sample_shapes_and_values(self, h265):
        tk = h265.ticks().select(*ALL)
        exp = synth.expected_trajectory(duration_s=4.0, hz=HZ, v=0.3, w=0.4)
        for i in (0, 1, 26, 27, 79):
            smp = tk[i]
            assert smp.t_ns == synth.T0_NS + i * DT_NS and smp.index == i
            for m in ("color_near", "color_far"):
                assert smp[m].shape == (80, 128, 3) and smp[m].dtype == np.uint8
                assert synth.tick_of_colour_frame(smp[m]) == i % 32
            for m in ("depth_near", "depth_far"):
                assert smp[m].dtype == np.uint16 and int(smp[m][0, 0]) == i
            assert smp["odom"]["x"] == pytest.approx(exp["x"][i]) and smp["odom"]["theta"] == pytest.approx(exp["theta"][i])
            assert smp["cmd_vel"]["vx"] == pytest.approx(0.3) and smp["cmd_vel"]["wz"] == pytest.approx(0.4)
            for m in ("imu_accel_near", "imu_gyro_near", "imu_accel_far", "imu_gyro_far"):
                w = smp[m]
                assert w.dtype == SERIES_DTYPES[m]
                # 0.5 s window ending at the tick, 200 Hz: 101 samples
                # (both ends inclusive) once the session is 0.5 s old
                assert len(w) == (101 if i * DT_NS >= 500_000_000 else i * 10 + 1), (m, i)
                assert w["t_ns"][-1] <= smp.t_ns and w["t_ns"][0] >= smp.t_ns - 500_000_000
            if i == 0:
                assert smp["runtime"] is None     # telemetry lands 1 ms after the tick
            else:
                rt = smp["runtime"]
                assert rt["mode"] == 2 and rt["t_ns"] == smp.t_ns - DT_NS + 1_000_000
                assert rt["wheel_l_target"] == pytest.approx(2.6, rel=1e-6)
                assert rt["wheel_r_target"] == pytest.approx(3.4, rel=1e-6)
                assert rt["seat_held"] == 1
            assert smp["intent"]["text"] == "hall -> kitchen"
            assert smp["place"]["name"] == "hall"
            ev = smp["event"]
            assert ev.dtype == SERIES_DTYPES["event"]
            t_ev = synth.T0_NS + (N // 3) * DT_NS
            inside = t_ev <= smp.t_ns and smp.t_ns - t_ev <= 3_000_000_000
            assert len(ev) == (1 if inside else 0), i
            if inside:
                assert ev["kind"][0] == "obstacle"

    def test_image_history_lists(self, h265):
        tk = h265.ticks(hz=10).select("color_near", "depth_far", history_s={"color_near": 0.3, "depth_far": 0.3})
        smp = tk[7]
        frames = smp["color_near"]
        assert isinstance(frames, list) and len(frames) == 4
        # 10 Hz ticks: 7 -> raw ticks 8, 10, 12, 14
        assert [synth.tick_of_colour_frame(f) for f in frames] == [8, 10, 12, 14]
        assert [int(d[0, 0]) for d in smp["depth_far"]] == [8, 10, 12, 14]
        early = tk[1]["color_near"]
        assert len(early) == 4 and early[:2] == [None, None]
        assert synth.tick_of_colour_frame(early[-1]) == 2

    def test_history_at_window_start_comes_from_the_session(self, h265):
        t = h265.ticks().t_ns
        # raw ticks, window from tick 10: the four earlier ticks exist and are used
        w = h265.ticks(start_ns=int(t[10])).select("color_near", "odom",
                                                   history_s={"color_near": 0.2, "odom": 0.2})
        smp = w[0]
        assert [synth.tick_of_colour_frame(f) for f in smp["color_near"]] == [6, 7, 8, 9, 10]
        assert len(smp["odom"]) == 5 and smp["odom"]["t_ns"][0] == int(t[6])
        # snapped: the history follows the session-wide 10 Hz grid
        w10 = h265.ticks(hz=10, start_ns=int(t[11])).select("color_near", history_s={"color_near": 0.3})
        assert w10.t_ns[0] == int(t[12])
        assert [synth.tick_of_colour_frame(f) for f in w10[0]["color_near"]] == [6, 8, 10, 12]
        # an off-grid stamp gets the grid ticks before it, itself last
        off = Ticks(h265, np.array([int(t[12]) + 1]), spec=InputSpec(("depth_near",), {"depth_near": 0.2}))
        hist = off[0]["depth_near"]
        assert [None if d is None else int(d[0, 0]) for d in hist] == [9, 10, 11, 12, None]

    def test_state_history_and_tolerance(self, h265):
        tk = h265.ticks().select("odom", "cmd_vel", history_s={"odom": 0.5})
        smp = tk[20]
        odom = smp["odom"]
        assert odom.dtype == SERIES_DTYPES["odom"] and len(odom) == 11
        assert odom["t_ns"][-1] == smp.t_ns
        # a tiny tolerance with an off-grid tick finds nothing
        off = Ticks(h265, np.array([synth.T0_NS + 25_000_000]), spec=InputSpec(("cmd_vel",), tol_ns=1_000_000))
        assert off[0]["cmd_vel"] is None
        on = Ticks(h265, np.array([synth.T0_NS + 25_000_000]), spec=InputSpec(("cmd_vel",), tol_ns=30_000_000))
        assert on[0]["cmd_vel"]["vx"] == pytest.approx(0.3)

    def test_iteration_and_negative_index(self, h265):
        tk = h265.ticks(hz=10).select("odom")
        assert [s.index for s in tk][:3] == [0, 1, 2]
        assert tk[-1].t_ns == tk.t_ns[-1]
        with pytest.raises(IndexError):
            tk[len(tk)]

    def test_jpeg_session_without_runtime_or_notes(self, jpeg):
        tk = jpeg.ticks().select("color_near", "color_far", "odom", "runtime", "intent", "place", "event")
        smp = tk[12]
        assert synth.tick_of_colour_frame(smp["color_near"]) == 12
        assert smp["runtime"] is None and smp["intent"] is None and smp["place"] is None
        assert len(smp["event"]) == 0
        assert jpeg.summary()["color_fmt"] == {"near": "jpeg", "far": "jpeg"}

    def test_reverse_session_cmd(self, reverse_session, tmp_path):
        s = Session(reverse_session, cache_dir=tmp_path / "rev")
        tk = s.ticks().select("cmd_vel")
        assert tk[49]["cmd_vel"]["vx"] == pytest.approx(0.3)
        assert tk[50]["cmd_vel"]["vx"] == pytest.approx(-0.3)


class TestFrameTable:
    def test_columns_and_values(self, h265):
        import polars as pl
        df = h265.ticks(hz=10).frame()
        assert isinstance(df, pl.DataFrame) and df.height == 40
        for col in ("t_ns", "odom_x", "odom_y", "odom_theta", "cmd_vx", "cmd_wz", "has_color_near",
                    "has_color_far", "far_hold_last", "gap_before_ms", "rt_mode", "rt_wheel_l_target",
                    "rt_battery_v", "intent", "place"):
            assert col in df.columns, col
        exp = synth.expected_trajectory(duration_s=4.0, hz=HZ, v=0.3, w=0.4)
        np.testing.assert_allclose(df["odom_theta"].to_numpy(), exp["theta"][::2], atol=1e-9)
        np.testing.assert_allclose(df["cmd_vx"].to_numpy(), 0.3)
        assert df["rt_mode"][0] == -1 and (df["rt_mode"][1:] == 2).all()
        assert df["intent"][0] == "hall -> kitchen" and df["place"][5] == "hall"
        assert df["has_color_far"].all()
        idx = np.load(h265.cache_dir / "video_far.npy")
        np.testing.assert_array_equal(df["au_size_far"].to_numpy(), idx["size"][::2])
        assert (df["au_size_near"].to_numpy() > 0).all()

    def test_au_size_is_minus_one_without_an_au(self, dropped_far_session, tmp_path):
        s = Session(dropped_far_session, cache_dir=tmp_path / "drop")
        df = s.ticks().frame()
        sizes = df["au_size_far"].to_numpy()
        assert (sizes[5::5] == -1).all() and (np.delete(sizes, np.arange(5, 80, 5)) > 0).all()

    def test_jpeg_table_nulls(self, jpeg):
        df = jpeg.ticks().frame()
        assert df.height == N
        assert df["intent"].null_count() == N and df["place"].null_count() == N
        assert (df["rt_mode"] == -1).all() and df["rt_battery_v"].is_nan().all()


class TestFooterless:
    def test_indexes_truncated_file(self, footerless_session, tmp_path):
        s = Session(footerless_session, cache_dir=tmp_path / "cut")
        assert s.info.footerless is True
        tk = s.ticks()
        assert 0 < len(tk) < N
        assert s.meta["footerless"] is True
        sel = tk.select("color_near", "depth_near", "odom")
        smp = sel[3]
        assert synth.tick_of_colour_frame(smp["color_near"]) == 3 and int(smp["depth_near"][0, 0]) == 3
        assert smp["odom"]["theta"] == pytest.approx(3 * 0.4 / HZ)
        last = sel[len(sel) - 1]
        assert last["color_near"] is not None
        summ = s.summary()
        assert summ["footerless"] is True and summ["ticks"] == len(tk)


class TestSummary:
    def test_keys(self, h265):
        summ = h265.summary()
        for k in ("name", "duration_s", "topics", "ticks", "tick_hz", "color_fmt", "keyframes",
                  "color_far_missing", "far_hold_last", "max_gap_ms", "series", "calib_roles", "cache_dir"):
            assert k in summ, k
        assert summ["ticks"] == N and summ["tick_hz"] == pytest.approx(20.0)
        assert summ["color_far_missing"] == 0 and summ["max_gap_ms"] == 50.0
        json.dumps(summ)     # `lab info` prints it
