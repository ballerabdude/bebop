"""Convert an MCAP capture into a Rerun .rrd that opens with the dashboard.

Rerun picks the application id from the loading stream, not from the MCAP,
so a direct `.mcap` open never matches the saved blueprint. This loads the
MCAP under the right app id with the matching dashboard and writes a
self-contained `.rrd` (data + default blueprint). `system_*` captures get
the `bebop_system` dashboard (power + host); everything else gets the navd
dashboard.

A footer-less MCAP (a live segment, or one killed before the SIGTERM handler
ran) makes Rerun's loader silently produce an empty recording, so we repair
it first: re-stream the complete chunks into a terminated temp MCAP.

    python mcap_to_rrd.py <in.mcap> <out.rrd>

Runs under the robot's dedicated rerun venv (`scripts/setup-rerun.sh`); the
firmware invokes it from `GET /captures/rrd/<stem>.rrd`. The venv's
rerun-sdk version must match the viewer's, or the .rrd won't load.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rerun as rr  # noqa: E402

import navd_blueprint  # noqa: E402
import system_blueprint  # noqa: E402

_MAGIC = b"\x89MCAP0\r\n"


def _has_footer(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            f.seek(-len(_MAGIC), os.SEEK_END)
            return f.read(len(_MAGIC)) == _MAGIC
    except OSError:
        return False


def _repair_mcap(src: str, dst: str) -> int:
    """Re-stream a footer-less MCAP into a terminated one. Drops any
    trailing partial chunk. Returns the number of messages written."""
    from mcap.exceptions import EndOfFile, McapError
    from mcap.reader import StreamReader
    from mcap.records import Channel, Message, Schema
    from mcap.writer import Writer

    schema_ids: dict = {}
    channel_ids: dict = {}
    n = 0
    with open(src, "rb") as fin, open(dst, "wb") as fout:
        writer = Writer(fout)
        writer.start()
        schemas: dict = {}
        channels: dict = {}
        try:
            for record in StreamReader(fin).records:
                if isinstance(record, Schema):
                    schemas[record.id] = record
                elif isinstance(record, Channel):
                    channels[record.id] = record
                elif isinstance(record, Message):
                    channel = channels.get(record.channel_id)
                    if channel is None:
                        continue
                    schema = schemas.get(channel.schema_id)
                    if schema is not None:
                        key = (schema.name, schema.encoding, schema.data)
                        if key not in schema_ids:
                            schema_ids[key] = writer.register_schema(
                                schema.name, schema.encoding, schema.data)
                        schema_id = schema_ids[key]
                    else:
                        schema_id = 0
                    meta = tuple(sorted(channel.metadata.items()))
                    ckey = (channel.topic, channel.message_encoding,
                            schema_id, meta)
                    if ckey not in channel_ids:
                        channel_ids[ckey] = writer.register_channel(
                            channel.topic, channel.message_encoding,
                            schema_id, dict(meta))
                    writer.add_message(channel_ids[ckey], record.log_time,
                                       record.data, record.publish_time)
                    n += 1
        except (EndOfFile, McapError):
            pass  # truncated tail: keep everything read so far
        writer.finish()
    return n


def _blueprint_for(in_path: str):
    """(app id, blueprint) — the system dashboard for `system_*` logs, the
    navd dashboard otherwise."""
    name = os.path.basename(in_path)
    if name.startswith("system_"):
        return system_blueprint.NAME, system_blueprint.build()
    return navd_blueprint.NAME, navd_blueprint.build()


def convert(in_path: str, out_path: str) -> None:
    app_id, blueprint = _blueprint_for(in_path)
    load_path = in_path
    repaired = None
    if not _has_footer(in_path):
        repaired = out_path + ".src.mcap"
        _repair_mcap(in_path, repaired)
        load_path = repaired
    try:
        rec = rr.RecordingStream(app_id)
        rec.save(out_path, default_blueprint=blueprint)
        rec.log_file_from_path(load_path)
        rec.flush()
        rec.disconnect()
    finally:
        if repaired is not None and os.path.exists(repaired):
            os.remove(repaired)


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        sys.exit("usage: mcap_to_rrd.py <in.mcap> <out.rrd>")
    convert(argv[0], argv[1])


if __name__ == "__main__":
    main()
