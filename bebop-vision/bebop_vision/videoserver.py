"""Stage-1 OBSBOT replacement (plan §9.2): serve the near camera's stream.

Lives INSIDE the bebop-vision process that owns the cameras (camera
exclusivity, docs §2.7) — main.py --goal-drive / --record-navd start it
after the rig opens.

Routes:
  /video?stream=<name>   multipart/x-mixed-replace MJPEG
       streams: color_near (default) | color_far | depth_near | depth_far
       color streams pass the camera hardware-encoded JPEG through untouched
       (zero CPU); depth streams render a turbo-colormapped 424x240 view
       (0-4 m, invalid = black) per frame (~3 ms).
  /snapshot?stream=...   single JPEG of the latest frame
  /healthz               liveness
"""

import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

PACING_S = 1.0 / 20.0      # serve at most 20 fps; frames arrive at 15
STREAMS = ("color_near", "color_far", "depth_near", "depth_far")


def render_depth(depth_mm):
    """uint16 (480, 848) mm -> half-res BGR turbo view, 0-4 m, invalid=black."""
    m = depth_mm > 0
    v = np.clip(depth_mm.astype(np.float32) / 4000.0, 0, 1) * 255
    vis = cv2.applyColorMap(v.astype(np.uint8), cv2.COLORMAP_TURBO)
    vis[~m] = 0
    return cv2.resize(vis, (424, 240), interpolation=cv2.INTER_AREA)


class VideoServer:
    def __init__(self, rig, port=9092):
        self.rig = rig
        self.port = port
        self._httpd = None
        self._thread = None

    def start(self):
        import threading
        server = self
        # video is operator-critical but must never take down the control
        # or recording process: bind failures degrade to a logged warning

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, fmt, *args):
                pass

            def _pick(self):
                """(camera role, data kind) from ?stream=, defaulting to
                color_near. Names are {kind}_{role}: color_near, depth_far."""
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
                elif self.path.startswith("/video"):
                    name, role, kind = self._pick()
                    self.send_response(200)
                    self.send_header("Content-Type",
                                     "multipart/x-mixed-replace; "
                                     "boundary=frame")
                    self.end_headers()
                    last = None
                    try:
                        while True:
                            fr = self._frame(role, kind)
                            if fr is not None and fr is not last:
                                last = fr
                                body = self._body(fr, kind)
                                if body:
                                    self.wfile.write(
                                        b"--frame\r\nContent-Type: "
                                        b"image/jpeg\r\nX-Timestamp-Us: "
                                        + str(fr.stamp_us).encode()
                                        + b"\r\nContent-Length: "
                                        + str(len(body)).encode()
                                        + b"\r\n\r\n" + body + b"\r\n")
                                    self.wfile.flush()
                            time.sleep(PACING_S)
                    except (BrokenPipeError, ConnectionResetError,
                            ConnectionAbortedError, OSError):
                        pass
                else:
                    self.send_error(404)

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
        print(f"[videoserver] serving streams {STREAMS} on :{self.port}/video")

    def stop(self):
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        self._thread = None
