"""Merge the firmware's health log rows into a copy of a drive session.

    python tools/mcap_merge_health.py SESSION.mcap [--system-dir DIR] [--out-dir DIR]

Writes ``<out-dir>/<session name>``: every record of the session plus the
``/power`` and ``/host`` rows of each ``system_*.mcap`` whose time span
overlaps the session, interleaved by log time. Schemas and channels are
re-registered with fresh ids and message bytes are copied verbatim, so the
result decodes exactly like the inputs. The writer uses no chunk
compression (the zstd chunk path is lossy with this mcap build; see
``recorder_mcap.py``), and the output always goes to a separate directory:
the raw mirror stays byte-identical so the next rsync does not re-download
it.

Footerless inputs (a segment copied while still being written) are read
with the stream reader and lose only their truncated tail.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcap.writer import CompressionType, Writer  # noqa: E402

from bebop_vision.mcap_tools import iter_messages, read_info  # noqa: E402

HEALTH_TOPICS = ("/power", "/host", "/session")


class _Remap:
    """Schema/channel ids of the inputs -> fresh ids in the output."""

    def __init__(self, writer):
        self.writer = writer
        self.schemas = {}
        self.channels = {}

    def channel(self, schema, channel):
        if schema is not None:
            skey = (schema.name, schema.encoding, schema.data)
            sid = self.schemas.get(skey)
            if sid is None:
                sid = self.writer.register_schema(schema.name, schema.encoding, schema.data)
                self.schemas[skey] = sid
        else:
            sid = 0
        meta = tuple(sorted(channel.metadata.items()))
        ckey = (channel.topic, channel.message_encoding, sid, meta)
        cid = self.channels.get(ckey)
        if cid is None:
            cid = self.writer.register_channel(channel.topic, channel.message_encoding,
                                               sid, dict(meta))
            self.channels[ckey] = cid
        return cid


def _overlapping(system_dir, start_ns, end_ns):
    out = []
    for p in sorted(Path(system_dir).glob("system_*.mcap")):
        info = read_info(p)
        if (info.start_ns is not None and info.end_ns is not None
                and info.start_ns <= end_ns and start_ns <= info.end_ns):
            out.append(info)
    return out


def _health_rows(infos, start_ns, end_ns):
    for info in infos:
        yield from iter_messages(info.path, HEALTH_TOPICS, start_ns=start_ns,
                                 end_ns=end_ns + 1, footerless=info.footerless)


def merge(session_path, system_dir=None, out_dir=None, chunk_size=256 * 1024):
    """Returns (output path, session messages, health rows)."""
    session_path = Path(session_path).resolve()
    system_dir = Path(system_dir) if system_dir else session_path.parent
    out_dir = Path(out_dir) if out_dir else session_path.parent / "merged"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = (out_dir / session_path.name).resolve()
    if out_path == session_path or (out_path.exists() and out_path.samefile(session_path)):
        # samefile also catches a hard link to the session (cp -al snapshots).
        raise SystemExit("refusing to merge in place; pick another --out-dir")
    info = read_info(session_path)
    if info.start_ns is None:
        raise SystemExit(f"{session_path.name}: no messages")
    infos = _overlapping(system_dir, info.start_ns, info.end_ns)
    topics = info.topics()
    n_session = n_health = 0
    # Write beside the target and rename into place: an existing file is
    # never emptied, and a failed run leaves no half-written merge behind.
    tmp_path = out_path.with_name(out_path.name + ".partial")
    with open(tmp_path, "wb") as fout:
        writer = Writer(fout, chunk_size=chunk_size, compression=CompressionType.NONE)
        writer.start(profile="", library="bebop mcap_merge_health")
        remap = _Remap(writer)
        health = _health_rows(infos, info.start_ns, info.end_ns)
        pending = next(health, None)
        for schema, channel, msg in iter_messages(session_path, topics,
                                                  footerless=info.footerless):
            while pending is not None and pending[2].log_time <= msg.log_time:
                hs, hc, hm = pending
                writer.add_message(remap.channel(hs, hc), hm.log_time, hm.data, hm.publish_time)
                n_health += 1
                pending = next(health, None)
            writer.add_message(remap.channel(schema, channel), msg.log_time, msg.data,
                               msg.publish_time)
            n_session += 1
        while pending is not None:
            hs, hc, hm = pending
            writer.add_message(remap.channel(hs, hc), hm.log_time, hm.data, hm.publish_time)
            n_health += 1
            pending = next(health, None)
        writer.finish()
    os.replace(tmp_path, out_path)
    return out_path, n_session, n_health, [i.path.name for i in infos]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("session", help="navd_session_*.mcap to merge")
    ap.add_argument("--system-dir", default=None,
                    help="directory holding system_*.mcap (default: the session's)")
    ap.add_argument("--out-dir", default=None,
                    help="output directory (default: <session dir>/merged)")
    args = ap.parse_args(argv)
    t0 = time.monotonic()
    out, n_s, n_h, files = merge(args.session, args.system_dir, args.out_dir)
    dt = time.monotonic() - t0
    print(f"{out}: {n_s} session messages + {n_h} health rows from "
          f"{len(files)} file(s) {files} in {dt:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
