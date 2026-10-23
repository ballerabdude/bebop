"""Read helpers for the robot's MCAP logs (navd drive sessions, firmware
system logs).

Three things every offline tool needs and none should re-implement:

* :func:`read_info` -- channels, schemas, per-topic counts and the time span
  of a file, from the summary when the footer exists and from a record
  stream when it does not (a segment copied while still being written, or
  cut by a power loss, has no footer and the indexed reader refuses it).
* :func:`iter_messages` -- ``(schema, channel, message)`` for a set of topics
  over either reader, optionally restricted to a time window, stopping
  cleanly at a truncated tail.
* :class:`ProtoDecoder` -- decodes a protobuf message from the
  ``FileDescriptorSet`` embedded as the MCAP schema, so a tool can read
  message types it has no generated bindings for (the firmware's
  ``bebop.system.*`` messages, or channels newer than the checked-in pb2).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Optional

from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
from mcap.exceptions import EndOfFile, McapError
from mcap.reader import make_reader
from mcap.records import Channel, Message, Schema
from mcap.stream_reader import StreamReader


@dataclass
class McapInfo:
    """What the summary (or a full scan) says about one file."""

    path: Path
    size_bytes: int
    schemas: dict = field(default_factory=dict)      # id -> Schema
    channels: dict = field(default_factory=dict)     # id -> Channel
    message_counts: dict = field(default_factory=dict)  # topic -> count
    message_count: int = 0
    start_ns: Optional[int] = None
    end_ns: Optional[int] = None
    footerless: bool = False

    @property
    def duration_s(self) -> float:
        if self.start_ns is None or self.end_ns is None:
            return 0.0
        return max(0.0, (self.end_ns - self.start_ns) / 1e9)

    def schema_for_topic(self, topic: str) -> Optional[Schema]:
        for ch in self.channels.values():
            if ch.topic == topic:
                return self.schemas.get(ch.schema_id)
        return None

    def topics(self) -> list:
        return sorted({ch.topic for ch in self.channels.values()})


def _stream_records(f) -> Iterator:
    """Records in file order; stops at the first truncated record."""
    reader = StreamReader(f)
    try:
        for record in reader.records:
            yield record
    except (EndOfFile, McapError):
        return


def _scan(path: Path) -> McapInfo:
    """Fallback for a footerless file: one pass over every record."""
    info = McapInfo(path=path, size_bytes=path.stat().st_size, footerless=True)
    counts_by_channel: dict = {}
    with open(path, "rb") as f:
        for record in _stream_records(f):
            if isinstance(record, Schema):
                info.schemas[record.id] = record
            elif isinstance(record, Channel):
                info.channels[record.id] = record
            elif isinstance(record, Message):
                counts_by_channel[record.channel_id] = (
                    counts_by_channel.get(record.channel_id, 0) + 1)
                info.message_count += 1
                t = record.log_time
                info.start_ns = t if info.start_ns is None else min(info.start_ns, t)
                info.end_ns = t if info.end_ns is None else max(info.end_ns, t)
    for cid, n in counts_by_channel.items():
        ch = info.channels.get(cid)
        if ch is not None:
            info.message_counts[ch.topic] = info.message_counts.get(ch.topic, 0) + n
    return info


def read_info(path) -> McapInfo:
    """Summary-backed :class:`McapInfo`, scanning the records when the file
    has no usable footer."""
    path = Path(path)
    try:
        with open(path, "rb") as f:
            summary = make_reader(f).get_summary()
    except (McapError, EndOfFile, ValueError, OSError):
        summary = None
    if summary is None or summary.statistics is None:
        return _scan(path)
    stats = summary.statistics
    info = McapInfo(path=path, size_bytes=path.stat().st_size,
                    schemas=dict(summary.schemas),
                    channels=dict(summary.channels),
                    message_count=int(stats.message_count),
                    start_ns=int(stats.message_start_time) if stats.message_count else None,
                    end_ns=int(stats.message_end_time) if stats.message_count else None)
    for cid, n in stats.channel_message_counts.items():
        ch = info.channels.get(cid)
        if ch is not None:
            info.message_counts[ch.topic] = info.message_counts.get(ch.topic, 0) + int(n)
    return info


def iter_messages(path, topics: Iterable[str], *, start_ns: Optional[int] = None,
                  end_ns: Optional[int] = None,
                  footerless: Optional[bool] = None) -> Iterator:
    """Yield ``(schema, channel, message)`` for ``topics`` in log-time order
    (seeking reader) or file order (stream fallback)."""
    path = Path(path)
    topics = list(topics)
    if footerless is None:
        footerless = read_info(path).footerless
    # Where the log-time-ordered pass stopped, so a fallback after a partial
    # pass does not yield those messages a second time.  The seeking reader
    # yields in non-decreasing log_time, so everything before `last_t` was
    # already yielded; `at_last` settles ties at `last_t` itself.
    last_t, at_last = None, set()
    if not footerless:
        try:
            with open(path, "rb") as f:
                reader = make_reader(f)
                for item in reader.iter_messages(topics=topics, start_time=start_ns,
                                                 end_time=end_ns):
                    m = item[2]
                    if m.log_time != last_t:
                        last_t, at_last = m.log_time, set()
                    at_last.add((m.channel_id, m.sequence, m.publish_time, len(m.data)))
                    yield item
            return
        except (McapError, EndOfFile):
            # A footer that lies (partial last chunk): fall through and stream.
            pass
    wanted = set(topics)
    schemas: dict = {}
    channels: dict = {}
    with open(path, "rb") as f:
        for record in _stream_records(f):
            if isinstance(record, Schema):
                schemas[record.id] = record
            elif isinstance(record, Channel):
                channels[record.id] = record
            elif isinstance(record, Message):
                ch = channels.get(record.channel_id)
                if ch is None or ch.topic not in wanted:
                    continue
                if start_ns is not None and record.log_time < start_ns:
                    continue
                if end_ns is not None and record.log_time >= end_ns:
                    continue
                if last_t is not None and (record.log_time < last_t or (
                        record.log_time == last_t and
                        (record.channel_id, record.sequence, record.publish_time,
                         len(record.data)) in at_last)):
                    continue
                yield schemas.get(ch.schema_id), ch, record


class ProtoDecoder:
    """Protobuf decoding from the schema data embedded in the MCAP.

    Each distinct schema gets its own :class:`DescriptorPool`, so files with
    the same message name but a different definition (an older recorder, a
    different firmware) never clash with each other or with the generated
    bindings in the default pool.
    """

    def __init__(self):
        self._classes: dict = {}

    def message_class(self, schema: Schema):
        if schema is None or schema.encoding != "protobuf":
            raise ValueError("not a protobuf schema")
        key = (schema.name, hashlib.sha1(schema.data).hexdigest())
        cls = self._classes.get(key)
        if cls is None:
            fds = descriptor_pb2.FileDescriptorSet()
            fds.ParseFromString(schema.data)
            pool = descriptor_pool.DescriptorPool()
            _add_files(pool, list(fds.file))
            cls = message_factory.GetMessageClass(
                pool.FindMessageTypeByName(schema.name))
            self._classes[key] = cls
        return cls

    def decode(self, schema: Schema, data: bytes):
        return self.message_class(schema).FromString(data)


def _add_files(pool, files):
    """Add FileDescriptorProtos in dependency order; pull well-known
    dependencies the set omits (e.g. google/protobuf/timestamp.proto) from
    the default pool."""
    names = {f.name for f in files}
    default = descriptor_pool.Default()
    extra = []
    for f in files:
        for dep in f.dependency:
            if dep not in names:
                try:
                    fdp = descriptor_pb2.FileDescriptorProto()
                    default.FindFileByName(dep).CopyToProto(fdp)
                    extra.append(fdp)
                    names.add(dep)
                except KeyError:
                    pass
    pending = extra + list(files)
    while pending:
        progressed = False
        for f in list(pending):
            try:
                pool.Add(f)
            except (TypeError, KeyError):
                continue
            pending.remove(f)
            progressed = True
        if not progressed:
            raise ValueError("unresolvable protobuf dependencies: "
                             + ", ".join(f.name for f in pending))


def field_value(msg, name, default=None):
    """``msg.<name>`` if the message has that field, else ``default``."""
    if msg is None or name not in msg.DESCRIPTOR.fields_by_name:
        return default
    return getattr(msg, name)


def repeated_message_fields(msg):
    """Names of the repeated message-typed fields of ``msg``."""
    out = []
    for fd in msg.DESCRIPTOR.fields:
        repeated = getattr(fd, "is_repeated", None)
        if repeated is None:  # protobuf < 5.26 has no is_repeated
            repeated = fd.label == fd.LABEL_REPEATED
        if repeated and fd.message_type is not None:
            out.append(fd.name)
    return out
