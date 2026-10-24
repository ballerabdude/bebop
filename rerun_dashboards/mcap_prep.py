"""Make an MCAP ready for Rerun before it is loaded (no Rerun import here).

Two fixes, done in one re-stream when needed:

- a capture cut by power loss has no footer; the viewer rejects it, so the
  readable records are re-streamed into a terminated copy (the last partial
  chunk is lost);
- drive sessions recorded between 2026-10-23 695dcbb and the /runtime split
  stored every firmware WebSocket frame on /runtime. Rerun plots a
  non-telemetry frame (ack, mode change, E-STOP, nav goal) as 0 in every
  /runtime panel, so those frames are moved to /runtime_events, which is
  where the recorder writes them now.

Frames are classified from the protobuf wire format (TelemetryFrame is field
4 of bebop.runtime.v1.ServerRuntimeMessage), so the converter venv needs no
protobuf bindings. Output is written uncompressed: the zstd chunk path of
this mcap build has lost data before (bebop_vision/recorder_mcap.py).
"""
from __future__ import annotations

import os
import struct

MAGIC = b"\x89MCAP0\r\n"
TELEMETRY_FIELD = 4


def has_footer(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            f.seek(-len(MAGIC), os.SEEK_END)
            return f.read(len(MAGIC)) == MAGIC
    except OSError:
        return False


def _varint(buf: bytes, i: int):
    shift = value = 0
    while True:
        b = buf[i]
        i += 1
        value |= (b & 0x7F) << shift
        if not b & 0x80:
            return value, i
        shift += 7


def is_telemetry(data: bytes) -> bool:
    """True if a serialized ServerRuntimeMessage carries a TelemetryFrame."""
    i, n = 0, len(data)
    try:
        while i < n:
            tag, i = _varint(data, i)
            field, wire = tag >> 3, tag & 7
            if wire == 0:
                _, i = _varint(data, i)
            elif wire == 2:
                length, i = _varint(data, i)
                if field == TELEMETRY_FIELD:
                    return True
                i += length
            elif wire == 1:
                i += 8
            elif wire == 5:
                i += 4
            else:
                return False
    except IndexError:
        return False
    return False


def needs_runtime_split(path: str) -> bool:
    """A footered file with /runtime but no /runtime_events (old recorder)."""
    from mcap.reader import make_reader
    try:
        with open(path, "rb") as f:
            summary = make_reader(f).get_summary()
    except Exception:
        return False
    if summary is None:
        return False
    topics = {c.topic for c in summary.channels.values()}
    return "/runtime" in topics and "/runtime_events" not in topics


def restream(src: str, dst: str, split_runtime: bool = True) -> dict:
    """Copy src to dst record by record (tolerating a missing footer), moving
    non-telemetry /runtime frames to /runtime_events. Returns counts."""
    from mcap.exceptions import EndOfFile, McapError
    from mcap.reader import StreamReader
    from mcap.records import Channel, Message, Schema
    from mcap.writer import CompressionType, Writer

    schemas, channels, schema_ids, channel_ids = {}, {}, {}, {}
    counts = {"messages": 0, "moved_to_runtime_events": 0}
    events_channel = None
    with open(src, "rb") as fin, open(dst, "wb") as fout:
        writer = Writer(fout, compression=CompressionType.NONE)
        writer.start()
        try:
            for record in StreamReader(fin).records:
                if isinstance(record, Schema):
                    schemas[record.id] = record
                elif isinstance(record, Channel):
                    channels[record.id] = record
                elif isinstance(record, Message):
                    ch = channels.get(record.channel_id)
                    if ch is None:
                        continue
                    sc = schemas.get(ch.schema_id)
                    if sc is not None and sc.id not in schema_ids:
                        schema_ids[sc.id] = writer.register_schema(sc.name, sc.encoding, sc.data)
                    if ch.id not in channel_ids:
                        channel_ids[ch.id] = writer.register_channel(
                            ch.topic, ch.message_encoding,
                            schema_ids.get(ch.schema_id, 0), ch.metadata)
                    out_ch = channel_ids[ch.id]
                    if (split_runtime and ch.topic == "/runtime"
                            and not is_telemetry(record.data)):
                        if events_channel is None:
                            events_channel = writer.register_channel(
                                "/runtime_events", ch.message_encoding,
                                schema_ids.get(ch.schema_id, 0), ch.metadata)
                        out_ch = events_channel
                        counts["moved_to_runtime_events"] += 1
                    writer.add_message(out_ch, record.log_time, record.data,
                                       record.publish_time)
                    counts["messages"] += 1
        except (EndOfFile, McapError, ValueError, struct.error):
            pass                      # a truncated tail: keep what was readable
        writer.finish()
    return counts


def prepare(path: str, tmp_path: str):
    """(path to load, temp file to delete or None) for a capture."""
    footer = has_footer(path)
    if footer and not needs_runtime_split(path):
        return path, None
    restream(path, tmp_path, split_runtime=True)
    return tmp_path, tmp_path
