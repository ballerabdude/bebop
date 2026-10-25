"""bebop_vision.lab.catalog on synthetic sessions, a ledger written by
tools/session_ledger and a local stand-in for the firmware's /captures
routes (no network, no robot)."""

import json
import os
import sys
import threading
from email.utils import formatdate
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import polars as pl
import pytest

from bebop_vision.lab import catalog
from bebop_vision.lab.catalog import Catalog
from bebop_vision.lab.types import file_identity
from tools import session_ledger

NOW = 1_792_900_000.0     # the fake robot's clock, seconds


def _link(src: Path, dst_dir: Path, name=None) -> Path:
    """The fixture MCAP under a second name / root (a hard link keeps the
    size, mtime and content identical, which is what the catalog keys on)."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / (name or src.name)
    try:
        os.link(src, dst)
    except OSError:
        dst.write_bytes(src.read_bytes())
    return dst


@pytest.fixture
def roots(tmp_path, jpeg_session, reverse_session):
    """Two roots: ``a`` holds the jpeg session, ``b`` the reverse session
    plus a *copy* of the jpeg session (first root must win)."""
    a = tmp_path / "a"
    b = tmp_path / "b"
    _link(jpeg_session, a)
    _link(reverse_session, b)
    _link(jpeg_session, b)
    (b / "not_a_session.mcap").write_bytes(b"x")     # wrong prefix, ignored
    (a / "navd_session_20260102_000000.mcap.part").write_bytes(b"y")  # partial, ignored
    return a, b


@pytest.fixture
def cat(roots, lab_cache):
    return Catalog(roots=roots, cache_dir=lab_cache, robot="http://127.0.0.1:9")


# -- disk ----------------------------------------------------------------------

def test_default_roots_resolve_against_the_package(monkeypatch):
    monkeypatch.delenv(catalog.CACHE_ENV, raising=False)
    c = Catalog()
    assert c.roots == [catalog.VISION_DIR / "datasets" / "sessions",
                       catalog.VISION_DIR / "datasets" / "bev-student" / "raw" / "mcap"]
    assert c.cache_dir == catalog.VISION_DIR / "datasets" / "lab"
    assert catalog.VISION_DIR.name == "bebop-vision"
    # missing roots are skipped, not errors
    assert Catalog(roots=("/nonexistent/x",)).local().height == 0


def test_cache_root_env_and_explicit(monkeypatch, tmp_path):
    monkeypatch.setenv(catalog.CACHE_ENV, str(tmp_path / "env-cache"))
    assert Catalog().cache_dir == (tmp_path / "env-cache").resolve()
    assert Catalog(cache_dir=tmp_path / "x").cache_dir == (tmp_path / "x").resolve()


def test_local_lists_sessions_first_root_wins(cat, roots, jpeg_session, reverse_session, lab_cache):
    a, b = roots
    df = cat.local()
    assert isinstance(df, pl.DataFrame)
    assert df.columns == ["name", "path", "size_bytes", "mtime", "has_cache"]
    assert df["name"].to_list() == [jpeg_session.stem, reverse_session.stem]
    assert df["path"].to_list() == [str(a / jpeg_session.name), str(b / reverse_session.name)]
    assert df["size_bytes"].to_list() == [jpeg_session.stat().st_size, reverse_session.stat().st_size]
    assert df.schema["mtime"] == pl.Datetime("ms", "UTC")
    assert df["has_cache"].to_list() == [False, False]
    (lab_cache / reverse_session.stem).mkdir()
    assert cat.local()["has_cache"].to_list() == [False, True]
    assert cat.cache_dir_for(reverse_session.stem) == lab_cache / reverse_session.stem


def test_local_with_a_ledger_joins_scalar_columns(cat, roots, jpeg_session, reverse_session):
    a, _ = roots
    row = session_ledger.analyse_session(jpeg_session)
    bogus = {"name": "navd_session_20990101_000000.mcap", "duration_s": 1.0}
    with open(a / "sessions.jsonl", "w") as f:
        f.write(json.dumps(row) + "\n")
        f.write("not json\n")
        f.write(json.dumps(bogus) + "\n")
        f.write(json.dumps(dict(row, ticks=3)) + "\n")       # later row replaces earlier
    df = cat.local()
    assert set(catalog.LEDGER_SCHEMA) <= set(df.columns)
    assert df.height == 2                               # ledger rows for unknown files do not add sessions
    got = df.filter(pl.col("name") == jpeg_session.stem).row(0, named=True)
    assert got["ticks"] == 3
    assert got["duration_s"] == pytest.approx(row["duration_s"])
    assert got["odom_path_m"] == pytest.approx(row["odom_path_m"])
    assert got["moving_frac"] == pytest.approx(1.0)
    assert got["intent"] is None and got["places"] == []  # jpeg session has no notes
    other = df.filter(pl.col("name") == reverse_session.stem).row(0, named=True)
    assert other["ticks"] is None and other["duration_s"] is None
    assert cat.ledger_row(jpeg_session.stem)["ticks"] == 3
    assert cat.ledger_row(jpeg_session.name)["ticks"] == 3
    assert cat.ledger_row(reverse_session.stem) is None


def test_explicit_ledger_path(cat, tmp_path, reverse_session):
    row = session_ledger.analyse_session(reverse_session)
    ledger = tmp_path / "elsewhere.jsonl"
    ledger.write_text(json.dumps(row) + "\n")
    c = Catalog(roots=cat.roots, cache_dir=cat.cache_dir, ledger=ledger)
    df = c.local()
    got = df.filter(pl.col("name") == reverse_session.stem).row(0, named=True)
    assert got["reverse_frac"] == pytest.approx(row["reverse_frac"]) and got["reverse_frac"] > 0.3
    assert got["intent"] == "hall -> kitchen" and got["places"] == ["hall"]
    assert Catalog(roots=cat.roots, ledger=tmp_path / "missing.jsonl").ledger() is None


def test_ledger_frame_schema_on_empty_input():
    df = catalog.ledger_frame([])
    assert df.height == 0
    assert df.schema["name"] == pl.Utf8 and df.schema["places"] == pl.List(pl.Utf8)


# -- resolve / info ------------------------------------------------------------

def test_resolve_name_stem_glob_and_path(cat, roots, jpeg_session, reverse_session):
    a, b = roots
    want = (a / jpeg_session.name).resolve()
    assert cat.resolve(jpeg_session.stem) == want
    assert cat.resolve(jpeg_session.name) == want
    assert cat.resolve("navd_session_20260102_*") == want
    assert cat.resolve("*20260102*.mcap") == want
    assert cat.resolve(str(b / jpeg_session.name)) == (b / jpeg_session.name).resolve()
    assert cat.resolve(b / reverse_session.name) == (b / reverse_session.name).resolve()
    assert cat.resolve(str(b / "navd_session_20260104_*")) == (b / reverse_session.name).resolve()
    with pytest.raises(ValueError, match="matches 2 files"):
        cat.resolve("navd_session_*")
    with pytest.raises(FileNotFoundError):
        cat.resolve("navd_session_19700101_000000")
    with pytest.raises(FileNotFoundError):
        cat.resolve("*nothing*")
    with pytest.raises(FileNotFoundError):
        cat.resolve(str(b / "navd_session_1999*"))


def test_resolve_path_glob_skips_part_files_and_is_cwd_relative(cat, roots, jpeg_session, reverse_session,
                                                                 monkeypatch, tmp_path):
    a, b = roots
    # root a also holds navd_session_20260102_000000.mcap.part (an interrupted
    # pull): the pattern must still mean the one MCAP
    assert (a / "navd_session_20260102_000000.mcap.part").exists()
    assert cat.resolve(str(a / "navd_session_20260102_*")) == (a / jpeg_session.name).resolve()
    with pytest.raises(FileNotFoundError):
        cat.resolve(str(a / "*.part"))
    # a relative path pattern resolves against the cwd, like a plain relative path
    monkeypatch.chdir(tmp_path)
    assert cat.resolve("b/" + reverse_session.name) == (b / reverse_session.name).resolve()
    assert cat.resolve("b/navd_session_20260104_*") == (b / reverse_session.name).resolve()
    assert cat.resolve("./b/*20260104*") == (b / reverse_session.name).resolve()


def test_info_is_light_and_matches_read_info(cat, roots, jpeg_session, lab_cache):
    a, _ = roots
    info = cat.info(jpeg_session.stem)
    assert info["name"] == jpeg_session.stem
    assert info["path"] == str(a / jpeg_session.name)
    assert info["size_bytes"] == jpeg_session.stat().st_size
    assert info["identity"] == file_identity(jpeg_session)
    assert info["footerless"] is False
    # 4 s at 200 Hz IMU: the last sample lands at 3.995 s and sets the span
    assert info["duration_s"] == pytest.approx(3.995, abs=1e-3)
    assert info["topics"]["/odom"] == 80 and info["topics"]["/color_near"] == 80
    assert "/runtime" not in info["topics"]
    assert info["messages"] == sum(info["topics"].values())
    assert info["ledger"] is None
    assert info["has_cache"] is False
    assert info["cache_dir"] == str(lab_cache / jpeg_session.stem)


def test_info_analyse_runs_the_ledger_analysis(cat, jpeg_session):
    info = cat.info(jpeg_session.stem, analyse=True)
    led = info["ledger"]
    assert led["ticks"] == 80 and led["tick_hz"] == pytest.approx(20.0, abs=0.1)
    assert led["moving_frac"] == pytest.approx(1.0)
    assert led["reverse_frac"] == 0.0
    assert led["odom_path_m"] == pytest.approx(0.3 * 79 / 20, abs=1e-3)
    assert "analysis_ms" in led


def test_info_footerless_session(footerless_session, lab_cache):
    c = Catalog(roots=(footerless_session.parent,), cache_dir=lab_cache)
    info = c.info(footerless_session.stem)
    assert info["footerless"] is True
    assert info["topics"]["/odom"] > 0
    assert c.local()["name"].to_list() == [footerless_session.stem]


# -- robot stand-in ------------------------------------------------------------

class _Robot(BaseHTTPRequestHandler):
    files = {}
    finalized = 0
    ranges = []
    short = set()        # names served with half their bytes (a dropped transfer)

    def log_message(self, *a):
        pass

    def date_time_string(self, timestamp=None):
        return formatdate(NOW, usegmt=True)

    def do_POST(self):
        if self.path == "/captures/finalize":
            type(self).finalized += 1
            self.send_response(200); self.send_header("Content-Length", "0"); self.end_headers()
        else:
            self.send_error(404)

    def do_GET(self):
        if self.path == "/captures":
            body = json.dumps({"files": [
                {"name": n, "sizeBytes": len(d), "modifiedMs": m, "active": a}
                for n, (d, m, a) in self.files.items()]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body))); self.end_headers()
            self.wfile.write(body)
            return
        name = self.path.rsplit("/", 1)[-1]
        if not self.path.startswith("/captures/dl/") or name not in self.files:
            self.send_error(404)
            return
        data = self.files[name][0]
        rng = self.headers.get("Range")
        if rng:
            type(self).ranges.append((name, rng))
            start = int(rng.split("=")[1].split("-")[0])
            body, code = data[start:], 206
        else:
            body, code = data, 200
        if name in self.short:
            body = body[:len(body) // 2]
        self.send_response(code)
        self.send_header("Content-Length", str(len(body))); self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def robot(jpeg_session):
    old = int((NOW - 3600) * 1000)
    _Robot.files = {
        # the real synthetic session, so a pulled copy is readable
        jpeg_session.name: (jpeg_session.read_bytes(), old, False),
        "navd_session_20261023_100000.mcap": (b"A" * 5000, old, False),
        "system_20261023_000000.mcap": (b"S" * 3000, old, False),
        "system_20261023_120000.mcap": (b"L" * 100, old, True),             # active
        "navd_session_20261023_125959.mcap": (b"R" * 100, int((NOW - 5) * 1000), False),  # recording
        "policy_capture_20261023_090000.mcap": (b"P" * 100, old, False),    # not ours
        "../evil.mcap": (b"E", old, False),
    }
    _Robot.finalized, _Robot.ranges, _Robot.short = 0, [], set()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Robot)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_remote_lists_prefixed_files_with_eligibility_and_local(robot, roots, lab_cache, jpeg_session):
    c = Catalog(roots=roots, cache_dir=lab_cache, robot=robot + "/")
    df = c.remote()
    assert df.columns == ["name", "file", "size_bytes", "modified_ms", "active", "eligible", "local"]
    assert df["file"].to_list() == sorted([
        jpeg_session.name, "navd_session_20261023_100000.mcap", "navd_session_20261023_125959.mcap",
        "system_20261023_000000.mcap", "system_20261023_120000.mcap"])
    rows = {r["file"]: r for r in df.iter_rows(named=True)}
    assert rows[jpeg_session.name]["name"] == jpeg_session.stem
    assert rows[jpeg_session.name]["local"] is True
    assert rows[jpeg_session.name]["eligible"] is True
    assert rows["navd_session_20261023_100000.mcap"]["local"] is False
    assert rows["navd_session_20261023_100000.mcap"]["size_bytes"] == 5000
    assert rows["system_20261023_120000.mcap"]["active"] is True
    assert rows["system_20261023_120000.mcap"]["eligible"] is False
    assert rows["navd_session_20261023_125959.mcap"]["active"] is False
    assert rows["navd_session_20261023_125959.mcap"]["eligible"] is False   # 5 s old on the robot clock
    assert rows["navd_session_20261023_125959.mcap"]["modified_ms"] == int((NOW - 5) * 1000)


def test_remote_unreachable_robot_raises(cat):
    with pytest.raises(OSError):
        cat.remote(timeout=1.0)


def test_pull_by_name_glob_and_states(robot, roots, lab_cache, tmp_path, jpeg_session):
    c = Catalog(roots=roots, cache_dir=lab_cache, robot=robot)
    dst = tmp_path / "pulled"
    got = c.pull(["navd_session_20261023_100000", "system_*", "navd_session_20261023_125959.mcap",
                  "navd_session_19990101_000000", "nomatch_*"], dst=dst)
    assert got == [
        ("navd_session_19990101_000000.mcap", "absent"),
        ("navd_session_20261023_100000.mcap", "pulled"),
        ("system_20261023_000000.mcap", "pulled"),
        ("system_20261023_120000.mcap", "busy"),
        ("navd_session_20261023_125959.mcap", "busy"),
    ]
    assert (dst / "navd_session_20261023_100000.mcap").read_bytes() == b"A" * 5000
    assert (dst / "system_20261023_000000.mcap").read_bytes() == b"S" * 3000
    assert not (dst / "system_20261023_120000.mcap").exists()
    assert _Robot.finalized == 1
    # again: everything already here, no download, finalize skipped
    assert c.pull("navd_session_20261023_100000", dst=dst, finalize=False) == [
        ("navd_session_20261023_100000.mcap", "have")]
    assert _Robot.finalized == 1 and _Robot.ranges == []


def test_pull_defaults_to_first_root_and_resumes(robot, roots, lab_cache, jpeg_session):
    a, _ = roots
    c = Catalog(roots=roots, cache_dir=lab_cache, robot=robot)
    (a / "navd_session_20261023_100000.mcap.part").write_bytes(b"A" * 1200)
    assert c.pull("navd_session_20261023_*", finalize=False) == [
        ("navd_session_20261023_100000.mcap", "resumed"),
        ("navd_session_20261023_125959.mcap", "busy")]
    assert ("navd_session_20261023_100000.mcap", "bytes=1200-") in _Robot.ranges
    assert (a / "navd_session_20261023_100000.mcap").read_bytes() == b"A" * 5000
    assert not (a / "navd_session_20261023_100000.mcap.part").exists()
    # the new file is now a session the catalog lists and resolves
    assert "navd_session_20261023_100000" in c.names()
    assert c.resolve("navd_session_20261023_100000") == (a / "navd_session_20261023_100000.mcap").resolve()


def test_pulled_real_session_is_readable(robot, tmp_path, lab_cache, jpeg_session):
    root = tmp_path / "fresh"
    c = Catalog(roots=(root,), cache_dir=lab_cache, robot=robot)
    assert c.local().height == 0
    assert c.pull(jpeg_session.stem, finalize=False) == [(jpeg_session.name, "pulled")]
    info = c.info(jpeg_session.stem)
    assert info["identity"] == file_identity(jpeg_session)
    assert info["topics"]["/odom"] == 80
    assert c.remote().filter(pl.col("name") == jpeg_session.stem)["local"].to_list() == [True]


def test_pull_reports_have_for_a_copy_in_any_root(robot, roots, lab_cache, tmp_path, jpeg_session):
    """The 2026-10-18 drives live in the second default root; pulling them
    again must not duplicate 6 GB into the first. remote().local and pull()
    agree on what is already here."""
    a, _ = roots
    fresh = tmp_path / "fresh"
    c = Catalog(roots=(fresh, a), cache_dir=lab_cache, robot=robot)
    assert c.remote().filter(pl.col("name") == jpeg_session.stem)["local"].to_list() == [True]
    assert c.pull(jpeg_session.stem, finalize=False) == [(jpeg_session.name, "have")]
    assert not (fresh / jpeg_session.name).exists()
    assert _Robot.ranges == []
    # an explicit dst outside the roots still does not re-pull a file a root holds
    assert c.pull(jpeg_session.stem, dst=tmp_path / "elsewhere", finalize=False) == [
        (jpeg_session.name, "have")]
    assert not (tmp_path / "elsewhere" / jpeg_session.name).exists()
    # a copy of the wrong size does not count
    (fresh / jpeg_session.name).write_bytes(b"short")
    assert Catalog(roots=(fresh,), cache_dir=lab_cache, robot=robot).remote().filter(
        pl.col("name") == jpeg_session.stem)["local"].to_list() == [False]


def test_pull_only_sees_prefixed_names(robot, roots, lab_cache, tmp_path):
    """Files the mirror script would never take (other prefixes, path
    separators) are not 'busy': a glob over them matches nothing and an
    explicit name is 'absent', the same view remote() gives."""
    c = Catalog(roots=roots, cache_dir=lab_cache, robot=robot)
    dst = tmp_path / "pulled"
    assert c.pull("policy_*", dst=dst, finalize=False) == []
    assert c.pull("*evil*", dst=dst, finalize=False) == []
    assert c.pull("policy_capture_20261023_090000", dst=dst, finalize=False) == [
        ("policy_capture_20261023_090000.mcap", "absent")]
    assert c.pull("*", dst=dst, finalize=False) == [
        ("navd_session_20260102_000000.mcap", "have"),          # in root a
        ("navd_session_20261023_100000.mcap", "pulled"),
        ("navd_session_20261023_125959.mcap", "busy"),
        ("system_20261023_000000.mcap", "pulled"),
        ("system_20261023_120000.mcap", "busy"),
    ]
    assert sorted(p.name for p in dst.iterdir()) == [
        "navd_session_20261023_100000.mcap", "system_20261023_000000.mcap"]


def test_pull_without_roots_needs_a_dst(robot, lab_cache, tmp_path):
    c = Catalog(roots=(), cache_dir=lab_cache, robot=robot)
    with pytest.raises(ValueError, match="no roots"):
        c.pull("navd_session_20261023_100000", finalize=False)
    assert c.pull("navd_session_20261023_100000", dst=tmp_path / "d", finalize=False) == [
        ("navd_session_20261023_100000.mcap", "pulled")]


def test_pull_failed_keeps_the_error_text(robot, roots, lab_cache, tmp_path, caplog):
    """A truncated transfer is 'failed' in the state list (the contract's
    (name, state) shape), with fetch's diagnosis kept in pull_errors and the
    log, and the .part left for a resume."""
    c = Catalog(roots=roots, cache_dir=lab_cache, robot=robot)
    dst = tmp_path / "pulled"
    _Robot.short.add("navd_session_20261023_100000.mcap")
    with caplog.at_level("WARNING", logger="bebop_vision.lab.catalog"):
        got = c.pull("navd_session_20261023_100000", dst=dst, finalize=False)
    assert got == [("navd_session_20261023_100000.mcap", "failed")]
    msg = c.pull_errors["navd_session_20261023_100000.mcap"]
    assert "got 2500 bytes, expected 5000" in msg
    assert "got 2500 bytes, expected 5000" in caplog.text
    assert (dst / "navd_session_20261023_100000.mcap.part").stat().st_size == 2500
    assert not (dst / "navd_session_20261023_100000.mcap").exists()
    # the next pull resumes from the .part and clears the error
    _Robot.short.clear()
    assert c.pull("navd_session_20261023_100000", dst=dst, finalize=False) == [
        ("navd_session_20261023_100000.mcap", "resumed")]
    assert c.pull_errors == {}
    assert (dst / "navd_session_20261023_100000.mcap").read_bytes() == b"A" * 5000


def test_tool_loader_shares_the_package_import():
    """The tests import ``tools.pull_captures`` as a package module; the
    catalog must reuse that object, not load a second copy by path."""
    from tools import pull_captures
    assert catalog._tool("pull_captures") is pull_captures
    assert catalog._tool("session_ledger") is session_ledger


def test_tool_loader_does_not_cache_a_failed_import(tmp_path, monkeypatch):
    """A tool that fails mid-import must fail the same way next time, not
    come back as an empty module (which would surface as AttributeError on
    the second info(analyse=True) / remote() in a long-lived notebook)."""
    (tmp_path / "broken_lab_tool.py").write_text("raise RuntimeError('boom')\n")
    monkeypatch.setattr(catalog, "TOOLS_DIR", tmp_path)
    monkeypatch.delitem(sys.modules, "tools.broken_lab_tool", raising=False)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="boom"):
            catalog._tool("broken_lab_tool")
    assert "tools.broken_lab_tool" not in sys.modules
