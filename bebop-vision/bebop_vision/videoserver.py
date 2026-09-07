"""Stage-1 OBSBOT replacement (plan §9.2): serve the near camera's stream.

Lives INSIDE the bebop-vision process that owns the cameras (camera
exclusivity, docs §2.7) — main.py --goal-drive / --record-navd start it
after the rig opens.

Routes:
  /video?stream=<name>   multipart/x-mixed-replace MJPEG
       streams: color_near (default) | color_far | depth_near | depth_far
                | bev
       color streams pass the camera hardware-encoded JPEG through untouched
       (zero CPU); depth streams render a turbo-colormapped 424x240 view
       (0-4 m, invalid = black) per frame (~3 ms); bev is the fused
       occupancy grid colorized top-down — BEV workers call publish_bev()
       once per grid (~10 Hz) and every client replays the same encoded
       bytes, so the feed costs nothing extra per viewer.
  /snapshot?stream=...   single JPEG of the latest frame
  /healthz               liveness
"""

import math
import time
from collections import namedtuple
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

PACING_S = 1.0 / 20.0      # serve at most 20 fps; frames arrive at 15
STREAMS = ("color_near", "color_far", "depth_near", "depth_far", "bev")

# BEV feed: 8x upscale of the 60x60 grid (480x480 px) at JPEG quality 80 —
# ~0.1 ms render + ~1 ms encode per grid, well inside the 10 Hz BEV budget.
BEV_SCALE = 8
BEV_JPEG_QUALITY = 80

# Cell-class palette (BGR) — same colors as the --display BEV overlay
# (main.py _render_bev): free dark, occupied red, hazard orange, inflated
# blue-gray.
BEV_COLORS = {0: (40, 40, 40), 1: (0, 0, 220), 2: (0, 140, 255),
              3: (80, 80, 160)}

# Latest published BEV feed frame: pre-encoded JPEG + grid stamp. The
# namedtuple exposes .stamp_us so the MJPEG part headers work for every
# stream kind; each publish replaces the object, which is exactly the
# frame-identity check the /video loop uses to skip duplicates.
_BevSlot = namedtuple("_BevSlot", ("jpeg", "stamp_us"))


def render_depth(depth_mm):
    """uint16 (480, 848) mm -> half-res BGR turbo view, 0-4 m, invalid=black."""
    m = depth_mm > 0
    v = np.clip(depth_mm.astype(np.float32) / 4000.0, 0, 1) * 255
    vis = cv2.applyColorMap(v.astype(np.uint8), cv2.COLORMAP_TURBO)
    vis[~m] = 0
    return cv2.resize(vis, (424, 240), interpolation=cv2.INTER_AREA)


def _goal_bearing(goal):
    """Body-frame goal bearing for the feed's arrow (None = no arrow).

    GoalHeading carries heading_rad; GoalPoint (x, y) points at atan2.
    """
    if goal is None:
        return None
    if hasattr(goal, "heading_rad"):
        return goal.heading_rad
    if hasattr(goal, "x") and hasattr(goal, "y"):
        return math.atan2(goal.y, goal.x)
    return None


def render_bev(grid, bearing_rad=None, scale=BEV_SCALE):
    """Occupancy grid -> colorized top-down BGR view for the feed.

    Drawn in grid convention (row 0 = far edge, col 0 = robot right) and
    mirrored horizontally at the end so body +y (left) renders LEFT —
    camera-aligned, matching the recorder's /bev_map Foxglove channel and
    the color views. `bearing_rad` draws a white arrow along the goal
    bearing from the robot origin at the bottom edge.
    """
    occ = grid.occ
    rows, cols = occ.shape
    cell = grid.cell_m
    # Colorize at grid resolution (the class masks only match there),
    # then nearest-neighbor upscale to the feed size.
    small = np.zeros((rows, cols, 3), np.uint8)
    for cls, color in BEV_COLORS.items():
        small[occ == cls] = color
    img = np.repeat(np.repeat(small, scale, axis=0), scale, axis=1)
    if bearing_rad is not None:
        def to_px(x, y):
            return (int((rows * cell - x) / cell * scale),
                    int((y + cols * cell / 2.0) / cell * scale))
        orow, ocol = to_px(0.0, 0.0)
        reach = 0.4 * rows * cell      # 1.2 m on the default 3 m grid
        trow, tcol = to_px(reach * math.cos(bearing_rad),
                           reach * math.sin(bearing_rad))
        cv2.arrowedLine(img, (ocol, orow), (tcol, trow),
                        (255, 255, 255), 2, tipLength=0.15)
    return np.ascontiguousarray(img[:, ::-1])


class VideoServer:
    def __init__(self, rig, port=9092):
        self.rig = rig
        self.port = port
        self._httpd = None
        self._thread = None
        # Latest BEV feed frame (None until the first grid fuses).
        self._bev = None

    def publish_bev(self, grid, goal=None):
        """Render + encode the fused BEV grid into the `bev` stream slot.

        Called from the BEV workers (~10 Hz, once per grid — HTTP clients
        replay the same encoded bytes, so extra viewers are free). `goal`
        is the goal-slot entry (GoalHeading / GoalPoint) for the bearing
        arrow. `grid=None` (no floor fit, cameras down) keeps the previous
        frame, same as a frozen camera stream; encode failures do too.
        """
        if grid is None:
            return
        bearing = _goal_bearing(goal)
        try:
            ok, jpg = cv2.imencode(
                ".jpg", render_bev(grid, bearing),
                [int(cv2.IMWRITE_JPEG_QUALITY), BEV_JPEG_QUALITY])
        except Exception:
            return
        if ok:
            self._bev = _BevSlot(jpg.tobytes(), int(grid.stamp_us))

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
                """(stream name, camera role, data kind) from ?stream=,
                defaulting to color_near. Camera streams are named
                {kind}_{role} (color_near, depth_far); bev is the fused
                occupancy-grid feed published by the BEV workers."""
                q = parse_qs(urlparse(self.path).query)
                name = (q.get("stream") or ["color_near"])[0]
                if name not in STREAMS:
                    name = "color_near"
                if name == "bev":
                    return name, None, "bev"
                kind, role = name.rsplit("_", 1)
                return name, role, kind

            def _frame(self, role, kind):
                if kind == "bev":
                    return server._bev
                cam = server.rig.cameras.get(role)
                return cam.read() if cam else None

            def _body(self, fr, kind):
                if kind == "bev":
                    return fr.jpeg
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
