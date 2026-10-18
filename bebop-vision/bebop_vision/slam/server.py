"""Read-only HTTP status for the live SLAM POC (default :9096).

Endpoints (GET only):

    /healthz                  liveness + tracking state
    /status                   full status dict
    /pose                     current pose (or valid=false)
    /map                      bounded depth-geometry snapshot
    /structure                Phase-1 gravity/floor/occupancy structure
    /trajectory?limit=N       bounded recent trajectory

The server only reads ``LiveSlam`` state; it never opens a camera or sends a
motion command.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

DEFAULT_PORT = 9096


def _json_bytes(value) -> bytes:
    return json.dumps(value, allow_nan=False, separators=(",", ":")).encode()


class SlamStatusServer:
    def __init__(self, slam, host: str = "0.0.0.0", port: int = DEFAULT_PORT):
        self.slam = slam
        self.host = host
        self.port = int(port)
        self.bound_port: int | None = None
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self):
        if self._httpd is not None:
            return self
        service = self

        class Handler(_SlamHandler):
            slam_service = service

        self._httpd = ThreadingHTTPServer((self.host, self.port), Handler)
        self.bound_port = self._httpd.server_address[1]
        self._thread = threading.Thread(
            target=self._httpd.serve_forever, name="live-slam-http",
            daemon=True)
        self._thread.start()
        print(f"[live-slam] status on :{self.bound_port} "
              f"(/healthz /status /pose /map /structure /trajectory)",
              flush=True)
        return self

    def stop(self):
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
            self._thread = None


class _SlamHandler(BaseHTTPRequestHandler):
    slam_service: SlamStatusServer

    def log_message(self, format, *args):  # silence per-request stderr noise
        return

    def _send_bytes(self, code: int, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send(self, code: int, payload) -> None:
        self._send_bytes(code, _json_bytes(payload))

    def do_GET(self):
        slam = self.slam_service.slam
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        if path == "/healthz":
            status = slam.status()
            self._send(200, {
                "ok": status["ok"],
                "service": status["service"],
                "tracking_state": status["tracking_state"],
                "pose_valid": status["pose_valid"],
                "segment": status["segment"],
                "uptime_s": status["uptime_s"],
            })
        elif path == "/status":
            self._send(200, slam.status())
        elif path == "/pose":
            self._send(200, slam.pose())
        elif path == "/map":
            self._send_bytes(200, slam.map_snapshot_bytes())
        elif path == "/structure":
            self._send_bytes(200, slam.structure_snapshot_bytes())
        elif path == "/trajectory":
            raw = parse_qs(parsed.query).get("limit", [None])[0]
            limit = None
            if raw is not None:
                try:
                    limit = int(raw)
                except (TypeError, ValueError):
                    self._send(400, {"error": "limit must be an integer"})
                    return
            self._send(200, slam.trajectory(limit))
        elif path == "/":
            self._send(200, {
                "service": "live-slam",
                "endpoints": ["/healthz", "/status", "/pose", "/map",
                              "/structure", "/trajectory"],
                "read_only": True,
            })
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        self._send(405, {"error": "read-only service"})

    do_PUT = do_POST
    do_DELETE = do_POST
    do_PATCH = do_POST
