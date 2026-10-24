"""tools/pull_captures.py against a local stand-in for the firmware's
capture routes (listing, finalize, Range downloads)."""

import json
import sys
import threading
from email.utils import formatdate
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import pull_captures  # noqa: E402

NOW = 1_792_800_000.0     # the fake robot's clock, seconds


def _files():
    old = int((NOW - 3600) * 1000)
    return {
        "navd_session_20261023_100000.mcap": (b"A" * 5000, old, False),
        "system_20261023_000000.mcap": (b"S" * 3000, old, False),
        "system_20261023_120000.mcap": (b"L" * 100, old, True),            # active
        "navd_session_20261023_125959.mcap": (b"R" * 100, int((NOW - 5) * 1000), False),  # recording
        "policy_capture_20261023_090000.mcap": (b"P" * 100, old, False),   # not ours
    }


class _Robot(BaseHTTPRequestHandler):
    files = {}
    finalized = 0
    ranges = []

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
        self.send_response(code)
        self.send_header("Content-Length", str(len(body))); self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def robot():
    _Robot.files, _Robot.finalized, _Robot.ranges = _files(), 0, []
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Robot)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_pulls_finished_captures_only(robot, tmp_path):
    assert pull_captures.main(["--robot", robot, "--dst", str(tmp_path)]) == 0
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "navd_session_20261023_100000.mcap", "system_20261023_000000.mcap"]
    assert (tmp_path / "navd_session_20261023_100000.mcap").read_bytes() == b"A" * 5000
    assert _Robot.finalized == 1
    # a second pull finds everything already here and downloads nothing
    assert pull_captures.main(["--robot", robot, "--dst", str(tmp_path)]) == 0
    assert _Robot.ranges == []


def test_resumes_a_partial_download(robot, tmp_path):
    (tmp_path / "navd_session_20261023_100000.mcap.part").write_bytes(b"A" * 1200)
    assert pull_captures.main(["--robot", robot, "--dst", str(tmp_path),
                               "--no-finalize"]) == 0
    assert ("navd_session_20261023_100000.mcap", "bytes=1200-") in _Robot.ranges
    assert (tmp_path / "navd_session_20261023_100000.mcap").read_bytes() == b"A" * 5000
    assert not (tmp_path / "navd_session_20261023_100000.mcap.part").exists()
    assert _Robot.finalized == 0


def test_select_uses_the_robot_clock():
    files = [{"name": "navd_session_x.mcap", "modifiedMs": 10_000, "active": False},
             {"name": "system_y.mcap", "modifiedMs": 0, "active": True},
             {"name": "../evil.mcap", "modifiedMs": 0, "active": False}]
    assert pull_captures.select(files, now_ms=20_000, min_age_s=30) == []
    assert [f["name"] for f in pull_captures.select(files, now_ms=50_000, min_age_s=30)] == [
        "navd_session_x.mcap"]


def test_unreachable_robot_is_an_error(tmp_path):
    assert pull_captures.main(["--robot", "http://127.0.0.1:9", "--dst", str(tmp_path),
                               "--no-finalize"]) == 1


def test_dry_run_touches_nothing(robot, tmp_path, capsys):
    assert pull_captures.main(["--robot", robot, "--dst", str(tmp_path), "--dry-run"]) == 0
    assert list(tmp_path.iterdir()) == [] and _Robot.finalized == 0
    assert "2 of 2 eligible" in capsys.readouterr().out
