"""Session ledger + health merge on synthetic MCAPs (no hardware, no network).

The session file is written with the real mcap Writer and the recorder's
schemas; the note channels (/intent, /event, /place) and the firmware's
bebop.system.Power / Host messages have no generated bindings here and are
built from hand-made FileDescriptorSets, which is exactly how the tools
must read them (dynamic decoding from the schema in the file)."""

import json
import sys
from pathlib import Path

import pytest
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
from mcap.writer import CompressionType, Writer

from bebop_vision import foxglove_proto, navd_proto
from bebop_vision.mcap_tools import ProtoDecoder, read_info

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import mcap_merge_health, session_ledger  # noqa: E402

T0 = 1_792_000_000_000_000_000   # ns since the epoch


def _fds(file_name, package, messages):
    """Serialized FileDescriptorSet for {msg: [(field, type, label?)]}.
    Returns (bytes, {full_name: message class}) built from a private pool."""
    fdp = descriptor_pb2.FileDescriptorProto(name=file_name, package=package, syntax="proto3")
    types = {"string": descriptor_pb2.FieldDescriptorProto.TYPE_STRING,
             "float": descriptor_pb2.FieldDescriptorProto.TYPE_FLOAT,
             "int64": descriptor_pb2.FieldDescriptorProto.TYPE_INT64,
             "uint32": descriptor_pb2.FieldDescriptorProto.TYPE_UINT32,
             "bool": descriptor_pb2.FieldDescriptorProto.TYPE_BOOL}
    for name, fields in messages.items():
        m = fdp.message_type.add(name=name)
        for i, spec in enumerate(fields, start=1):
            fname, ftype = spec[0], spec[1]
            f = m.field.add(name=fname, number=i)
            if ftype in types:
                f.type = types[ftype]
            else:
                f.type = descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE
                f.type_name = f".{package}.{ftype}"
            f.label = (descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED
                       if len(spec) > 2 and spec[2] == "repeated"
                       else descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL)
    fds = descriptor_pb2.FileDescriptorSet()
    fds.file.append(fdp)
    pool = descriptor_pool.DescriptorPool()
    pool.Add(fdp)
    classes = {f"{package}.{n}": message_factory.GetMessageClass(
        pool.FindMessageTypeByName(f"{package}.{n}")) for n in messages}
    return fds.SerializeToString(), classes


NOTES_FDS, NOTES = _fds("bebop/navd/notes.proto", "bebop.navd", {
    "Intent": [("text", "string"), ("from_place", "string"), ("to_place", "string"),
               ("stamp_ns", "int64")],
    "Event": [("kind", "string"), ("note", "string"), ("stamp_ns", "int64")],
    "PlaceSeen": [("place_id", "string"), ("name", "string"), ("score", "float"),
                  ("stamp_ns", "int64")],
})
SYSTEM_FDS, SYSTEM = _fds("system.proto", "bebop.system", {
    "ThermalZone": [("name", "string"), ("temp_c", "float")],
    "Power": [("stamp_ns", "int64"), ("battery_voltage_v", "float"),
              ("state_of_charge_pct", "float"), ("board_temperature_c", "float"),
              ("fault_bits", "uint32")],
    "Host": [("stamp_ns", "int64"), ("cpu_pct", "float"), ("gpu_pct", "float"),
             ("disk_used_pct", "float"), ("thermal", "ThermalZone", "repeated")],
})


def _writer(path):
    f = open(path, "wb")
    w = Writer(f, chunk_size=64 * 1024, compression=CompressionType.NONE)
    w.start(profile="", library="test")
    return f, w


def write_session(path, *, seconds=4.0, tick_hz=20.0, gyro_hz=200.0):
    """A 4 s drive: 2 s straight at 0.3 m/s, 1 s left turn, 0.5 s reverse,
    0.5 s stop; one 0.3 s frame gap, one 50 ms gyro gap; notes on the new
    channels. Returns the number of messages written."""
    f, w = _writer(path)
    sch = {name: w.register_schema(*navd_proto.schema(cls))
           for name, cls in (("twist", navd_proto.Twist), ("odom", navd_proto.Odom),
                             ("gyro", navd_proto.ImuGyro))}
    sch_img = w.register_schema(*foxglove_proto.schema(foxglove_proto.CompressedImage))
    sch_notes = {n: w.register_schema(n, "protobuf", NOTES_FDS) for n in NOTES}
    ch = {
        "/color_near": w.register_channel("/color_near", "protobuf", sch_img),
        "/cmd_vel": w.register_channel("/cmd_vel", "protobuf", sch["twist"]),
        "/odom": w.register_channel("/odom", "protobuf", sch["odom"]),
        "/imu_gyro_near": w.register_channel("/imu_gyro_near", "protobuf", sch["gyro"]),
        "/intent": w.register_channel("/intent", "protobuf", sch_notes["bebop.navd.Intent"]),
        "/event": w.register_channel("/event", "protobuf", sch_notes["bebop.navd.Event"]),
        "/place": w.register_channel("/place", "protobuf", sch_notes["bebop.navd.PlaceSeen"]),
    }
    n = 0
    x = 0.0
    dt = 1.0 / tick_hz
    for i in range(int(seconds * tick_hz)):
        t = i * dt
        if 1.0 <= t < 1.3:          # one 0.3 s hole in the frame sets
            continue
        log = T0 + int(t * 1e9)
        if t < 2.0:
            vx, wz = 0.3, 0.0
        elif t < 3.0:
            vx, wz = 0.1, 0.6
        elif t < 3.5:
            vx, wz = -0.2, 0.0
        else:
            vx, wz = 0.0, 0.0
        x += vx * dt
        img = foxglove_proto.CompressedImage(frame_id="near", format="jpeg", data=b"\xff\xd8")
        img.timestamp.CopyFrom(foxglove_proto.timestamp(log))
        w.add_message(ch["/color_near"], log, img.SerializeToString(), log); n += 1
        w.add_message(ch["/cmd_vel"], log, navd_proto.Twist(vx=vx, wz=wz, stamp_ns=log)
                      .SerializeToString(), log); n += 1
        w.add_message(ch["/odom"], log, navd_proto.Odom(x=x, y=0.0, theta=0.0, stamp_ns=log)
                      .SerializeToString(), log); n += 1
    for k in range(int(seconds * gyro_hz)):
        dev_us = 1_000_000 + int(k * 1e6 / gyro_hz)
        if 400 <= k < 410:          # 50 ms gyro dropout
            continue
        log = T0 + int(k * 1e9 / gyro_hz)
        w.add_message(ch["/imu_gyro_near"], log,
                      navd_proto.ImuGyro(gx=0.0, gy=0.0, gz=0.01, stamp_ns=log,
                                         device_stamp_us=dev_us).SerializeToString(), log); n += 1
    intent = NOTES["bebop.navd.Intent"](text="living room to kitchen", from_place="living",
                                         to_place="kitchen", stamp_ns=T0)
    w.add_message(ch["/intent"], T0, intent.SerializeToString(), T0); n += 1
    for t, kind, note in ((0.5, "planted", "sock at 1 m"), (3.0, "recovery", "off-lane start")):
        log = T0 + int(t * 1e9)
        w.add_message(ch["/event"], log, NOTES["bebop.navd.Event"](kind=kind, note=note, stamp_ns=log)
                      .SerializeToString(), log); n += 1
    for t, name in ((0.0, "living"), (1.0, "living"), (2.0, "hall"), (3.0, "kitchen")):
        log = T0 + int(t * 1e9)
        w.add_message(ch["/place"], log, NOTES["bebop.navd.PlaceSeen"](
            place_id=name[:3], name=name, score=0.9, stamp_ns=log).SerializeToString(), log); n += 1
    w.finish(); f.close()
    return n


def write_system(path, start_ns, seconds, *, volts=(53.0, 52.0), cpu=(20.0, 60.0)):
    """A 1 Hz health log over [start, start+seconds]: Power + Host per second."""
    f, w = _writer(path)
    sp = w.register_schema("bebop.system.Power", "protobuf", SYSTEM_FDS)
    sh = w.register_schema("bebop.system.Host", "protobuf", SYSTEM_FDS)
    cp = w.register_channel("/power", "protobuf", sp)
    chh = w.register_channel("/host", "protobuf", sh)
    n = 0
    for i in range(int(seconds) + 1):
        log = start_ns + int(i * 1e9)
        frac = i / max(1, int(seconds))
        p = SYSTEM["bebop.system.Power"](stamp_ns=log, battery_voltage_v=volts[0] + (volts[1] - volts[0]) * frac,
                                         state_of_charge_pct=80.0 - 10.0 * frac,
                                         board_temperature_c=30.0 + i, fault_bits=0)
        w.add_message(cp, log, p.SerializeToString(), log); n += 1
        h = SYSTEM["bebop.system.Host"](stamp_ns=log, cpu_pct=cpu[0] + (cpu[1] - cpu[0]) * frac,
                                        gpu_pct=5.0, disk_used_pct=70.0)
        h.thermal.add(name="cpu", temp_c=50.0 + i)
        h.thermal.add(name="gpu", temp_c=45.0)
        w.add_message(chh, log, h.SerializeToString(), log); n += 1
    w.finish(); f.close()
    return n


@pytest.fixture
def mirror(tmp_path):
    """A mirror directory: one session, one overlapping health log that
    started 60 s earlier, one health log from the day before."""
    session = tmp_path / "navd_session_20261023_120000.mcap"
    n_session = write_session(session)
    overlap = tmp_path / "system_20261023_115900.mcap"
    n_overlap = write_system(overlap, T0 - 60_000_000_000, 120)
    other = tmp_path / "system_20261022_000000.mcap"
    write_system(other, T0 - 86_400_000_000_000, 3600)
    return {"dir": tmp_path, "session": session, "n_session": n_session,
            "overlap": overlap, "n_overlap": n_overlap, "other": other}


def test_read_info_summary_and_footerless(mirror, tmp_path):
    info = read_info(mirror["session"])
    assert not info.footerless and info.message_count == mirror["n_session"]
    assert info.message_counts["/color_near"] == 74          # 80 ticks minus the 6 in the hole
    assert abs(info.duration_s - 3.995) < 0.02
    cut = tmp_path / "navd_session_20261023_130000.mcap"
    data = mirror["session"].read_bytes()
    cut.write_bytes(data[: len(data) - 4000])                 # footer and tail gone
    info2 = read_info(cut)
    assert info2.footerless and info2.message_count > 0
    assert "/cmd_vel" in info2.message_counts and info2.start_ns == T0


def test_cmd_stats_turns_and_fractions():
    t = [i * 0.05 for i in range(80)]
    vx = [0.3] * 40 + [0.1] * 20 + [-0.2] * 10 + [0.0] * 10
    wz = [0.0] * 40 + [0.6] * 20 + [0.0] * 20
    s = session_ledger.cmd_stats(t, vx, wz)
    assert s == {"moving_frac": 0.875, "reverse_frac": 0.125,
                 "turn_events": 1, "turns_left": 1, "turns_right": 0}
    # two left turns 0.3 s apart merge; a right turn stays separate; a 0.2 s
    # blip is not an event
    wz2 = ([0.6] * 12 + [0.0] * 6 + [0.6] * 12 + [0.0] * 20 + [-0.6] * 12
           + [0.0] * 10 + [0.6] * 4 + [0.0] * 4)
    s2 = session_ledger.cmd_stats(t, [0.1] * 80, wz2)
    assert (s2["turn_events"], s2["turns_left"], s2["turns_right"]) == (2, 1, 1)


def test_gap_and_rate_stats():
    stamps = [0.0, 0.05, 0.10, 0.40, 0.45]
    assert session_ledger.gap_stats(stamps, 0.15) == (1, pytest.approx(0.30))
    assert session_ledger.rate_hz([0.0, 0.5, 1.0]) == pytest.approx(2.0)
    assert session_ledger.path_length([0, 3, 3], [0, 4, 4]) == pytest.approx(5.0)


def test_ledger_rows_and_health_join(mirror):
    rows = session_ledger.build_ledger([str(mirror["dir"])], log=lambda *_: None)
    assert len(rows) == 1
    r = rows[0]
    assert r["name"] == mirror["session"].name and not r["footerless"]
    assert r["ticks"] == 74 and 17.0 < r["tick_hz"] < 20.5
    assert (r["frame_gaps"], r["frame_gap_max_s"]) == (1, pytest.approx(0.35, abs=0.01))
    assert r["gyro_hz"] == pytest.approx(200.0, abs=1.0)
    assert (r["gyro_gaps"], r["gyro_gap_max_ms"]) == (1, pytest.approx(55.0, abs=0.1))
    assert r["moving_frac"] == pytest.approx(0.865, abs=0.01)   # 64 of 74 recorded ticks
    assert r["reverse_frac"] == pytest.approx(10 / 74, abs=0.01)
    assert (r["turn_events"], r["turns_left"], r["turns_right"]) == (1, 1, 0)
    # 0.3 m/s x 2 s minus the 0.3 s hole, + 0.1 m/s x 1 s, + 0.2 m/s x 0.5 s reversed
    assert r["odom_path_m"] == pytest.approx(0.3 * 1.7 + 0.1 * 1 + 0.2 * 0.5, abs=0.03)
    assert r["intent"]["text"] == "living room to kitchen" and r["intent"]["to_place"] == "kitchen"
    assert [e["kind"] for e in r["events"]] == ["planted", "recovery"]
    assert [p["name"] for p in r["places"]] == ["living", "hall", "kitchen"]
    assert r["places"][0]["sightings"] == 2 and r["places"][0]["last_ns"] == T0 + 1_000_000_000
    assert r["runtime_messages"] == 0
    h = r["health"]
    assert h["files"] == [mirror["overlap"].name]           # the day-before log is not joined
    assert h["rows_power"] == 4 and h["rows_host"] == 4        # seconds 60..63 of that log (session ends at 3.995 s)
    assert h["battery_v_min"] < h["battery_v_max"] <= 53.0
    assert h["soc_first"] > h["soc_last"]
    assert h["fault_any"] is False
    assert h["hottest_zone"]["name"] == "cpu" and h["hottest_zone"]["temp_c"] == pytest.approx(113.0)
    assert h["cpu_pct_max"] > h["cpu_pct_mean"] and h["disk_used_pct_last"] == 70.0


def test_ledger_cli_writes_jsonl(mirror, capsys):
    out = mirror["dir"] / "ledger.jsonl"
    assert session_ledger.main([str(mirror["dir"]), "--out", str(out)]) == 0
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["health"]["files"] == [mirror["overlap"].name]
    printed = capsys.readouterr().out
    assert mirror["session"].name[:34] in printed and "1 session(s)" in printed


def test_ledger_single_file_target_and_explicit_system_dir(mirror, tmp_path):
    empty = tmp_path / "nohealth"
    empty.mkdir()
    rows = session_ledger.build_ledger([str(mirror["session"])], system_dir=str(empty),
                                       log=lambda *_: None)
    assert rows[0]["health"] == {"files": [], "rows_power": 0, "rows_host": 0}


def test_merge_adds_only_overlapping_rows_and_never_in_place(mirror):
    before = (mirror["session"].stat().st_size, mirror["session"].read_bytes()[:256])
    out, n_s, n_h, files = mcap_merge_health.merge(mirror["session"])
    assert out == (mirror["dir"] / "merged" / mirror["session"].name).resolve()
    assert out != mirror["session"].resolve()
    assert n_s == mirror["n_session"] and n_h == 8 and files == [mirror["overlap"].name]
    assert (mirror["session"].stat().st_size, mirror["session"].read_bytes()[:256]) == before
    info = read_info(out)
    assert not info.footerless
    assert info.message_count == mirror["n_session"] + 8
    assert info.message_counts["/power"] == 4 and info.message_counts["/host"] == 4
    assert info.message_counts["/cmd_vel"] == 74
    # the merged file decodes with the same dynamic path as the inputs
    dec = ProtoDecoder()
    rows = session_ledger.build_ledger([str(out)], system_dir=str(mirror["dir"] / "merged"),
                                       log=lambda *_: None)
    assert rows[0]["turn_events"] == 1 and rows[0]["intent"]["to_place"] == "kitchen"
    assert dec.decode(info.schema_for_topic("/power"), b"").battery_voltage_v == 0.0
    # health rows are interleaved by log time in FILE order, not appended at
    # the end (the stream reader keeps file order; the seeking reader would
    # sort by log time and hide an append)
    from bebop_vision.mcap_tools import iter_messages
    seq = [ch.topic for _, ch, _ in iter_messages(out, ["/power", "/cmd_vel"],
                                                  footerless=True)]
    last_power = max(i for i, s in enumerate(seq) if s == "/power")
    last_cmd = max(i for i, s in enumerate(seq) if s == "/cmd_vel")
    assert seq.count("/power") == 4 and last_power < last_cmd


def test_merge_refuses_in_place(mirror):
    with pytest.raises(SystemExit):
        mcap_merge_health.merge(mirror["session"], out_dir=mirror["dir"])


def test_merge_cli(mirror, capsys):
    assert mcap_merge_health.main([str(mirror["session"]), "--out-dir",
                                   str(mirror["dir"] / "m2")]) == 0
    assert "8 health rows" in capsys.readouterr().out
    assert (mirror["dir"] / "m2" / mirror["session"].name).exists()


def test_cleared_intent_keeps_the_mission(tmp_path):
    path = tmp_path / "navd_session_20261023_140000.mcap"
    f, w = _writer(path)
    sch = w.register_schema("bebop.navd.Intent", "protobuf", NOTES_FDS)
    ch = w.register_channel("/intent", "protobuf", sch)
    sc = w.register_schema(*navd_proto.schema(navd_proto.Twist))
    cc = w.register_channel("/cmd_vel", "protobuf", sc)
    Intent = NOTES["bebop.navd.Intent"]
    for i, (text, frm, to) in enumerate((("living room to kitchen", "living room", "kitchen"),
                                         ("kitchen to office", "kitchen", "office"),
                                         ("", "", ""))):
        log = T0 + i * 1_000_000_000
        w.add_message(ch, log, Intent(text=text, from_place=frm, to_place=to,
                                      stamp_ns=log).SerializeToString(), log)
        w.add_message(cc, log, navd_proto.Twist(vx=0.2).SerializeToString(), log)
    w.finish(); f.close()
    row = session_ledger.analyse_session(path)
    assert [i["text"] for i in row["intents"]] == ["living room to kitchen", "kitchen to office", ""]
    assert row["intent"]["text"] == "kitchen to office"       # the clear did not erase it


SYS2_FDS, SYS2 = _fds("system2.proto", "bebop.system", {
    "Power": [("stamp_ns", "int64"), ("present", "bool"), ("status_received", "bool"),
              ("status_stale", "bool"), ("battery_voltage_v", "float"),
              ("state_of_charge_pct", "float"), ("board_temperature_c", "float"),
              ("fault_bits", "uint32"), ("fault_description", "string")],
    "Host": [("stamp_ns", "int64"), ("cpu_pct", "float"), ("gpu_pct", "float"),
             ("gpu_present", "bool"), ("disk_used_pct", "float")],
})


def test_health_ignores_placeholders_and_rail_bits(tmp_path):
    path = tmp_path / "system_20261023_000000.mcap"
    f, w = _writer(path)
    cp = w.register_channel("/power", "protobuf",
                            w.register_schema("bebop.system.Power", "protobuf", SYS2_FDS))
    chh = w.register_channel("/host", "protobuf",
                             w.register_schema("bebop.system.Host", "protobuf", SYS2_FDS))
    P, H = SYS2["bebop.system.Power"], SYS2["bebop.system.Host"]
    rows = [P(present=False),                                            # no board yet
            P(present=True, status_received=False),                      # no status yet
            P(present=True, status_received=True, battery_voltage_v=52.0,
              state_of_charge_pct=-1.0, board_temperature_c=31.0,
              fault_bits=0xD00000, fault_description="normal"),          # rails on, SOC unknown
            P(present=True, status_received=True, battery_voltage_v=51.5,
              state_of_charge_pct=70.0, board_temperature_c=32.0,
              fault_bits=0xD00000, fault_description="normal")]
    for i, msg in enumerate(rows):
        log = T0 + i * 1_000_000_000
        w.add_message(cp, log, msg.SerializeToString(), log)
        w.add_message(chh, log, H(cpu_pct=30.0, gpu_pct=0.0, gpu_present=False,
                                  disk_used_pct=70.0).SerializeToString(), log)
    w.finish(); f.close()
    info = read_info(path)
    h = session_ledger.summarise_health([info], T0, T0 + 10_000_000_000)
    assert h["rows_power"] == 4
    assert (h["battery_v_min"], h["battery_v_max"]) == (51.5, 52.0)   # no 0.0 V placeholder
    assert (h["soc_first"], h["soc_last"]) == (70.0, 70.0)            # -1 = unknown
    assert h["fault_any"] is False and h["faults"] == []              # rail bits are not faults
    assert h["gpu_pct_mean"] is None                                  # no GPU sampler


def test_merge_refuses_a_hard_link_to_the_session(mirror):
    import os
    out_dir = mirror["dir"] / "merged"
    out_dir.mkdir()
    os.link(mirror["session"], out_dir / mirror["session"].name)
    before = mirror["session"].read_bytes()
    with pytest.raises(SystemExit):
        mcap_merge_health.merge(mirror["session"])
    assert mirror["session"].read_bytes() == before


def test_iter_messages_fallback_does_not_repeat(mirror, monkeypatch):
    """A seeking pass that dies part-way falls back to streaming without
    yielding the messages it already produced."""
    from mcap.exceptions import McapError
    from bebop_vision import mcap_tools

    real = mcap_tools.make_reader

    class Dying:
        def __init__(self, f):
            self._r = real(f)

        def __getattr__(self, name):          # summary etc. from the real reader
            return getattr(self._r, name)

        def iter_messages(self, **kw):
            for i, item in enumerate(self._r.iter_messages(**kw)):
                if i == 25:
                    raise McapError("partial last chunk")
                yield item

    full = [(c.topic, m.log_time, m.sequence)
            for _, c, m in mcap_tools.iter_messages(mirror["session"], ["/cmd_vel", "/odom"])]
    monkeypatch.setattr(mcap_tools, "make_reader", Dying)
    got = [(c.topic, m.log_time, m.sequence)
           for _, c, m in mcap_tools.iter_messages(mirror["session"], ["/cmd_vel", "/odom"])]
    assert len(got) == len(set(got)) == len(full)
    assert sorted(got) == sorted(full)


def test_intent_run_fields_are_read_when_present(tmp_path):
    """The recorder's Intent schema (with task / target / run_id) passes the
    run through; the older hand-made schema above reads them as ""."""
    path = tmp_path / "navd_session_20261025_104233.mcap"
    f, w = _writer(path)
    ch = w.register_channel("/intent", "protobuf",
                            w.register_schema(*navd_proto.schema(navd_proto.Intent)))
    for i, (text, run) in enumerate((("door", "r20261025_104233_9f3a"),
                                     ("door", "r20261025_104233_9f3a"),
                                     ("door", "r20261025_104301_00be"), ("", ""))):
        log = T0 + i * 1_000_000_000
        w.add_message(ch, log, navd_proto.Intent(
            text=text, task="go_through_door" if text else "",
            target="left" if text else "", run_id=run, stamp_ns=log).SerializeToString(), log)
    w.finish(); f.close()
    row = session_ledger.analyse_session(path)
    assert row["intent"]["run_id"] == "r20261025_104301_00be"
    assert (row["intent"]["task"], row["intent"]["target"]) == ("go_through_door", "left")
    assert row["run_ids"] == ["r20261025_104233_9f3a", "r20261025_104301_00be"]


def test_old_intent_schema_reads_empty_run_fields(tmp_path):
    path = tmp_path / "navd_session_20261023_150000.mcap"
    f, w = _writer(path)
    ch = w.register_channel("/intent", "protobuf",
                            w.register_schema("bebop.navd.Intent", "protobuf", NOTES_FDS))
    w.add_message(ch, T0, NOTES["bebop.navd.Intent"](text="old", stamp_ns=T0)
                  .SerializeToString(), T0)
    w.finish(); f.close()
    row = session_ledger.analyse_session(path)
    assert (row["intent"]["task"], row["intent"]["run_id"]) == ("", "")
    assert row["run_ids"] == []
