"""Read or repair an unterminated MCAP (e.g. after a sudden power cut).

MCAP is append-only and chunked. A writer finalizes a chunk on ``flush()``
and only writes the trailing DataEnd / summary / footer on ``finish()``. A
hard power cut (or SIGKILL) leaves a file whose complete chunks are on disk
but which has no footer — the indexed reader (``mcap.reader.make_reader``)
and even ``NonSeekingReader`` raise on it.

``iter_messages()`` streams the records and stops cleanly at the
truncation, yielding every message in the complete chunks. ``repair()``
rewrites them into a properly-terminated MCAP so any tool (Rerun, Foxglove,
``mcap_extract.py``) can open it.

    python tools/mcap_recover.py <in.mcap> <out.mcap>
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import IO, Iterator, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from mcap.exceptions import EndOfFile, McapError
    from mcap.reader import StreamReader
    from mcap.records import Channel, Message, Schema
except ImportError as exc:  # pragma: no cover
    raise ImportError("pip install mcap") from exc


def iter_messages(
    f: IO[bytes],
) -> Iterator[Tuple[Optional[Schema], Channel, Message]]:
    """Yield ``(schema, channel, message)`` in file order, tolerating a
    missing footer. Stops at the first truncated/incomplete record."""
    reader = StreamReader(f)
    schemas: dict = {}
    channels: dict = {}
    try:
        for record in reader.records:
            if isinstance(record, Schema):
                schemas[record.id] = record
            elif isinstance(record, Channel):
                channels[record.id] = record
            elif isinstance(record, Message):
                channel = channels.get(record.channel_id)
                if channel is None:
                    continue
                schema = (schemas.get(channel.schema_id)
                          if channel.schema_id else None)
                yield schema, channel, record
    except (EndOfFile, McapError):
        # Truncated tail (or a partial trailing chunk): keep everything
        # yielded so far and stop, rather than raising.
        return


def repair(in_path, out_path) -> int:
    """Re-stream ``in_path`` into a terminated MCAP at ``out_path``.

    Returns the number of messages written. Any trailing partial chunk is
    dropped (records after the last complete chunk)."""
    from mcap.writer import Writer

    schema_ids: dict = {}
    channel_ids: dict = {}
    n = 0
    with open(in_path, "rb") as fin, open(out_path, "wb") as fout:
        writer = Writer(fout)
        writer.start()
        for schema, channel, message in iter_messages(fin):
            if schema is not None:
                key = (schema.name, schema.encoding, schema.data)
                if key not in schema_ids:
                    schema_ids[key] = writer.register_schema(
                        schema.name, schema.encoding, schema.data)
                schema_id = schema_ids[key]
            else:
                schema_id = 0
            meta = tuple(sorted(channel.metadata.items()))
            ckey = (channel.topic, channel.message_encoding, schema_id, meta)
            if ckey not in channel_ids:
                channel_ids[ckey] = writer.register_channel(
                    channel.topic, channel.message_encoding, schema_id,
                    dict(meta))
            writer.add_message(channel_ids[ckey], message.log_time,
                               message.data, message.publish_time)
            n += 1
        writer.finish()
    return n


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        sys.exit("usage: python tools/mcap_recover.py <in.mcap> <out.mcap>")
    in_path, out_path = argv
    n = repair(in_path, out_path)
    print(f"recovered {n} messages -> {out_path}")


if __name__ == "__main__":
    main()
