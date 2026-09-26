"""Operator video for the Orbbec rig — WebRTC (WHEP) only.

Lives INSIDE the bebop-vision process that owns the cameras (camera
exclusivity, docs §2.7) — main.py --goal-drive / --record-navd start it
after the rig opens.

Routes:
  POST /whep?stream=<name>   WebRTC (WHEP): SDP offer in, answer out.
       streams: color_near (default) | color_far | depth_near | depth_far
       Every stream is H.264 over SRTP/UDP (~150 ms, loss-tolerant). Color
       hands the camera's hardware JPEG to `nvjpegdec` (zero CPU pixels);
       depth renders a turbo-colormapped 424x240 BGR view server-side.
  GET  /snapshot?stream=...  single JPEG of the latest frame (tools/tests)
  GET  /healthz              liveness

The old MJPEG (`/video`) and fragmented-MP4 (`codec=h264|h265`) operator
paths were removed once every client moved to WebRTC.
"""

import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

PACING_S = 1.0 / 20.0      # serve at most 20 fps; frames arrive at 15
STREAMS = ("color_near", "color_far", "depth_near", "depth_far")
DEPTH_VIEW = (424, 240)    # rendered depth size (matches render_depth)


def render_depth(depth_mm):
    """uint16 (480, 848) mm -> half-res BGR turbo view, 0-4 m, invalid=black."""
    m = depth_mm > 0
    v = np.clip(depth_mm.astype(np.float32) / 4000.0, 0, 1) * 255
    vis = cv2.applyColorMap(v.astype(np.uint8), cv2.COLORMAP_TURBO)
    vis[~m] = 0
    return cv2.resize(vis, DEPTH_VIEW, interpolation=cv2.INTER_AREA)


def _whep_feed(session, cam, kind):
    """Push frames into a WHEP session until it or the peer ends."""
    last = None
    while not session.closed:
        if session.failed():
            break
        fr = cam.read() if cam else None
        if fr is not None and fr is not last:
            last = fr
            if kind == "color":
                data = getattr(fr, "color_jpeg", None)
            else:
                data = render_depth(fr.depth) if fr.depth is not None else None
            if data is not None:
                session.push(data)
        time.sleep(PACING_S)
    session.close()


class VideoServer:
    def __init__(self, rig, port=9092):
        self.rig = rig
        self.port = port
        self._httpd = None
        self._thread = None

    def start(self):
        import threading
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, fmt, *args):
                pass

            def _pick(self):
                """(stream name, camera role, data kind) from ?stream=,
                defaulting to color_near. Camera streams are named
                {kind}_{role} (color_near, depth_far)."""
                q = parse_qs(urlparse(self.path).query)
                name = (q.get("stream") or ["color_near"])[0]
                if name not in STREAMS:
                    name = "color_near"
                kind, role = name.rsplit("_", 1)
                return name, role, kind

            def _frame(self, role, kind):
                cam = server.rig.cameras.get(role)
                return cam.read() if cam else None

            def _body(self, fr, kind):
                if kind == "color":
                    return fr.color_jpeg
                if fr.depth is None:
                    return None
                ok, jpg = cv2.imencode(
                    ".jpg", render_depth(fr.depth),
                    [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                return jpg.tobytes() if ok else None

            def do_GET(self):
                if self.path.startswith("/snapshot"):
                    name, role, kind = self._pick()
                    fr = self._frame(role, kind)
                    if fr is None:
                        self.send_error(503, "no frame")
                        return
                    body = self._body(fr, kind)
                    if not body:
                        self.send_error(503, "no frame data")
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif self.path.startswith("/healthz"):
                    self.send_response(200)
                    self.send_header("Content-Length", "2")
                    self.end_headers()
                    self.wfile.write(b"ok")
                else:
                    self.send_error(404)

            def do_OPTIONS(self):
                # WHEP POSTs carry `Content-Type: application/sdp`, which
                # makes them non-simple, so browsers preflight them.
                self.send_response(204)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods",
                                 "POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers",
                                 "Content-Type")
                self.send_header("Content-Length", "0")
                self.end_headers()

            def do_POST(self):
                if self.path.startswith("/whep"):
                    self._whep()
                else:
                    self.send_error(404)

            def _whep(self):
                """WebRTC color/depth stream: offer in, answer out, then the
                frame feeder runs until the peer disconnects."""
                from .whep import WhepSession, webrtc_available
                name, role, kind = self._pick()
                if not webrtc_available():
                    self.send_error(503, "webrtc unavailable")
                    return
                try:
                    length = int(self.headers.get("Content-Length") or 0)
                except ValueError:
                    length = 0
                offer = self.rfile.read(length).decode("utf-8", "replace")
                cam = server.rig.cameras.get(role)
                fr = None
                deadline = time.monotonic() + 5.0
                while fr is None and time.monotonic() < deadline:
                    fr = cam.read() if cam else None
                    if fr is None:
                        time.sleep(0.05)
                if fr is None:
                    self.send_error(503, "no frame")
                    return
                fps = float(getattr(fr, "fps", 0) or 0) or 15.0
                if kind == "color":
                    session_kwargs = dict(input_kind="jpeg", fps=fps)
                else:
                    session_kwargs = dict(input_kind="bgr", fps=fps,
                                          width=DEPTH_VIEW[0],
                                          height=DEPTH_VIEW[1])
                try:
                    session = WhepSession(bitrate=4_000_000, **session_kwargs)
                    answer = session.negotiate(offer)
                except Exception as exc:  # noqa: BLE001
                    print(f"[videoserver] whep failed: {exc}")
                    self.send_error(503, f"webrtc failed: {exc}")
                    return
                body = answer.encode()
                self.send_response(201)
                self.send_header("Content-Type", "application/sdp")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                threading.Thread(
                    target=_whep_feed, args=(session, cam, kind), daemon=True,
                    name="whep-feed").start()

        try:
            self._httpd = ThreadingHTTPServer(("0.0.0.0", self.port), Handler)
        except OSError as exc:
            print(f"[videoserver] port {self.port} unavailable ({exc}); "
                  f"operator video disabled")
            self._httpd = None
            return
        self._thread = threading.Thread(
            target=self._httpd.serve_forever, kwargs={"poll_interval": 0.25},
            daemon=True, name="videoserver")
        self._thread.start()
        print(f"[videoserver] serving WebRTC (WHEP) streams {STREAMS} "
              f"on :{self.port}/whep")

    def stop(self):
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        self._thread = None
