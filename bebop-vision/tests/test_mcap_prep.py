"""rerun_dashboards/mcap_prep.py: wire-format telemetry check, /runtime split,
footer repair (the converter's input preparation; no Rerun needed)."""

import sys
from pathlib import Path

from mcap.reader import make_reader
from mcap.writer import CompressionType, Writer

from bebop_vision import navd_proto
from bebop_vision.proto.bebop.runtime.v1 import bebop_runtime_pb2 as pb

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "rerun_dashboards"))
import mcap_prep  # noqa: E402


def _frames():
    tel = pb.ServerRuntimeMessage(); tel.telemetry.host_unix_ms = 5
    tel.telemetry.drive.cmd_linear_x = 0.2
    ack = pb.ServerRuntimeMessage(request_id=7); ack.ack.ok = True
    mode = pb.ServerRuntimeMessage(); mode.mode_changed.mode = pb.MODE_DIAL_IN
    estop = pb.ServerRuntimeMessage(); estop.estop_latched.reason = "operator"
    return [tel, ack, mode, tel, estop]


def test_is_telemetry_reads_the_oneof_from_the_wire():
    kinds = [mcap_prep.is_telemetry(m.SerializeToString()) for m in _frames()]
    assert kinds == [True, False, False, True, False]
    assert mcap_prep.is_telemetry(b"") is False
    assert mcap_prep.is_telemetry(b"\xff") is False          # truncated varint


def _old_session(path, footer=True):
    f = open(path, "wb")
    w = Writer(f, compression=CompressionType.NONE); w.start()
    rt = w.register_channel("/runtime", "protobuf",
                            w.register_schema(*navd_proto.schema(pb.ServerRuntimeMessage)))
    cmd = w.register_channel("/cmd_vel", "protobuf", w.register_schema(*navd_proto.schema(navd_proto.Twist)))
    for i, m in enumerate(_frames() * 40):
        w.add_message(rt, 1_000 + i, m.SerializeToString(), 1_000 + i)
        w.add_message(cmd, 1_000 + i, navd_proto.Twist(vx=0.1).SerializeToString(), 1_000 + i)
    w.finish(); f.close()
    if not footer:
        data = path.read_bytes(); path.write_bytes(data[: len(data) - 200])
    return path


def _topics(path):
    out = {}
    with open(path, "rb") as f:
        for s, c, m in make_reader(f).iter_messages():
            out.setdefault(c.topic, []).append(m.data)
    return out


def test_old_runtime_is_split_and_new_files_pass_through(tmp_path):
    old = _old_session(tmp_path / "old.mcap")
    assert mcap_prep.needs_runtime_split(str(old))
    load, tmp = mcap_prep.prepare(str(old), str(tmp_path / "prep.mcap"))
    assert tmp == load and Path(load).exists()
    t = _topics(load)
    assert len(t["/runtime"]) == 80 and len(t["/runtime_events"]) == 120
    assert all(mcap_prep.is_telemetry(d) for d in t["/runtime"])
    assert len(t["/cmd_vel"]) == 200
    # a file that already has /runtime_events loads as is
    assert mcap_prep.needs_runtime_split(load) is False
    assert mcap_prep.prepare(load, str(tmp_path / "again.mcap")) == (load, None)


def test_footerless_capture_is_repaired(tmp_path):
    cut = _old_session(tmp_path / "cut.mcap", footer=False)
    assert not mcap_prep.has_footer(str(cut))
    load, tmp = mcap_prep.prepare(str(cut), str(tmp_path / "fixed.mcap"))
    assert tmp is not None and mcap_prep.has_footer(load)
    assert "/cmd_vel" in _topics(load)
