"""Stage-1 OBSBOT replacement (plan §9.2): serve the near camera's stream.

Lives INSIDE the bebop-vision process that owns the cameras (camera
exclusivity, docs §2.7) — main.py --goal-drive / --record-navd start it
after the rig opens. Serves the camera hardware-encoded MJPEG frames as
multipart MJPEG on :9092/video — the same wire format the app's <img>
already consumed from the firmware /video route, with zero CPU encode.

Routes:
  /video     multipart/x-mixed-replace MJPEG (X-Timestamp-Us per part)
  /snapshot  single JPEG (latest frame)
  /healthz   liveness
"""

import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PACING_S = 1.0 / 20.0      # serve at most 20 fps; frames arrive at 15


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

            def do_GET(self):
                if self.path.startswith("/snapshot"):
                    f = server.rig.cameras.get("near")
                    fr = f.read() if f else None
                    if fr is None or fr.color_jpeg is None:
                        self.send_error(503, "no frame")
                        return
                    body = fr.color_jpeg
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
                    self.send_response(200)
                    self.send_header("Content-Type",
                                     "multipart/x-mixed-replace; "
                                     "boundary=frame")
                    self.end_headers()
                    cam = server.rig.cameras.get("near")
                    last = None
                    try:
                        while True:
                            fr = cam.read() if cam else None
                            if fr is not None and fr is not last \
                                    and fr.color_jpeg:
                                last = fr
                                body = fr.color_jpeg
                                self.wfile.write(
                                    b"--frame\r\nContent-Type: image/jpeg"
                                    b"\r\nX-Timestamp-Us: "
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
        print(f"[videoserver] serving near-camera MJPEG on :{self.port}/video")

    def stop(self):
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        self._thread = None
