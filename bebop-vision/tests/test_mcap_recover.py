"""Reading / repairing an unterminated (power-loss) MCAP.

A writer finalizes a chunk on flush but only writes the footer on finish();
a hard power cut leaves complete chunks with no footer. The indexed reader
raises on that, so `tools/mcap_recover.py` streams instead.
"""

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcap.reader import make_reader
from mcap.writer import CompressionType, Writer

from tools.mcap_recover import iter_messages, repair


def _unterminated_mcap(n_messages=300, chunk_size=1024):
    """An MCAP with finalized chunks but no DataEnd/summary/footer."""
    buf = io.BytesIO()
    w = Writer(buf, chunk_size=chunk_size,
               compression=CompressionType.NONE)
    w.start()
    sch = w.register_schema("test.M", "jsonschema", b"{}")
    ch = w.register_channel("/x", "json", sch)
    for i in range(n_messages):
        w.add_message(ch, i, b'{"v": %d}' % i, i)
    # deliberately no finish(): simulate power loss before the footer
    return buf.getvalue()


def test_iter_messages_recovers_unterminated(tmp_path):
    p = tmp_path / "cut.mcap"
    p.write_bytes(_unterminated_mcap())
    with open(p, "rb") as f:
        msgs = list(iter_messages(f))
    assert len(msgs) > 0
    for schema, channel, msg in msgs[:5]:
        assert channel.topic == "/x"
        assert schema.name == "test.M"


def test_repair_makes_it_indexable(tmp_path):
    p = tmp_path / "cut.mcap"
    p.write_bytes(_unterminated_mcap())
    out = tmp_path / "repaired.mcap"
    n = repair(str(p), str(out))
    assert n > 0 and out.exists()
    # the indexed reader (which needs the footer) now works
    with open(out, "rb") as f:
        got = list(make_reader(f).iter_messages())
    assert len(got) == n


def test_terminated_file_still_reads(tmp_path):
    buf = io.BytesIO()
    w = Writer(buf, chunk_size=1024, compression=CompressionType.NONE)
    w.start()
    sch = w.register_schema("test.M", "jsonschema", b"{}")
    ch = w.register_channel("/x", "json", sch)
    for i in range(50):
        w.add_message(ch, i, b'{"v": %d}' % i, i)
    w.finish()
    p = tmp_path / "ok.mcap"
    p.write_bytes(buf.getvalue())
    with open(p, "rb") as f:
        assert len(list(iter_messages(f))) == 50
