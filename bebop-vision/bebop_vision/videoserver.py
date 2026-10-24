"""Operator video for the Orbbec rig — WebRTC (WHEP) only.

Lives INSIDE the bebop-vision process that owns the cameras (camera
exclusivity, docs §2.7) — main.py --record-navd starts it
after the rig opens.

Routes:
  POST   /whep?stream=<name>  WHEP: SDP offer in, 201 + SDP answer out, with
         `Location: /whep/<session-id>` (the session's WHEP resource).
         streams: color_near (default) | color_far | depth_near | depth_far
         Optional `&replace=<session-id>` tears that session down first (a
         client reconnecting after a stall frees its own old slot).
         503 + JSON {"error": "viewer limit", ...} and `Retry-After` when
         the stream already has --video-max-viewers viewers.
  DELETE /whep/<session-id>   tear that viewer down now (200; 404 unknown)
  GET    /whep                JSON {streams: {name: {viewers, encoder,
                              encoder_stats: {frames, au_out, idr_out,
                              keyframe_requests, viewers} | null,
                              sessions: {<sid6>: {state, au_in, au_sent,
                              au_dropped, pli, age_s}}}}, max_viewers}
  GET    /snapshot?stream=... single JPEG of the latest frame (tools/tests)
  GET    /healthz             liveness
  POST   /intent {text, from, to}   operator intent (sticky; any time);
                                    {"clear": true} clears it
  POST   /event  {kind, note}       marker in the current drive (obstacle,
                                    recovery, exclude, note); 409 when no
                                    segment is recording
  GET    /notes                     current intent + queue counts
  The note routes write into the active MCAP segment through
  bebop_vision.session_notes (see the recorder's /intent, /event, /place).

Multi-viewer fan-out (whep.py): each stream has ONE shared encoder
(`_StreamHub`: feeder thread + H264Encoder) feeding up to `max_viewers`
WebRTC peers. Viewers never evict each other. A viewer leaves on DELETE,
on ICE failure, after 5 s ICE-disconnected, on RTCP BYE / 8 s RTCP
silence, or when its media stalls; the stream's encoder is released 5 s
after its last viewer leaves (a quick stream flip reuses it). A joining
viewer waits for an IDR, and asks the encoder for one as soon as ICE
connects; IDRs also recur every --video-idr-s (1 s). A viewer whose
link backs up (its appsrc queue near full) stops receiving and resyncs on
the next IDR instead of losing frames silently (`au_dropped`). A POST
stuck negotiating gives its slot back after 30 s; each webrtcbin step of
the negotiation is bounded at 5 s.

Bitrates (CBR): color --video-color-kbps (3500), depth --video-depth-kbps
(1000; depth is a 424x240 colormap).

Testing on the robot (two viewers of one stream):
  1. Open the app's Drive/Video view in two browsers (or a phone and a
     laptop) on the same stream. Both play; neither evicts the other.
  2. `curl -s http://bebop.local:9092/whep | jq` shows
     color_near with "viewers": 2, "encoder": true and two sessions whose
     au_sent keeps climbing (a frozen viewer: au_in rising but au_sent
     flat = waiting for an IDR; both flat = encoder stalled, check
     encoder_stats.au_out). keyframe_requests vs idr_out shows whether
     nvv4l2h264enc honours forced keyframes (idr_out should rise faster
     than 1/s while viewers join). The journal
     (`journalctl -u bebop-vision -f | grep whep`) shows ONE
     "encoder started" line for the stream and one "viewer ... open" line
     per browser.
  3. Close one tab: within ~8 s (immediately if the app sends DELETE) the
     count drops to 1 and the other viewer keeps playing. Close the last:
     "encoder released (no viewers)" ~5 s later. Close lines carry the
     viewer's and the encoder's counters.
  4. Cap: open a 5th viewer of one stream — its POST answers 503 with
     {"error": "viewer limit", "viewers": 4, "max_viewers": 4}.
  5. Lifecycle by hand (no media; the session never connects, so it is
     reaped after 15 s anyway):
       OFFER=<an SDP offer captured from a browser's createOffer()>
       curl -si -X POST -H 'Content-Type: application/sdp' \
            --data-binary "$OFFER" 'http://bebop.local:9092/whep?stream=depth_near'
       # -> HTTP/1.1 201, Location: /whep/<id>
       curl -si -X DELETE http://bebop.local:9092/whep/<id>   # -> 200
       curl -si -X DELETE http://bebop.local:9092/whep/<id>   # -> 404
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

from .session_notes import NotRecording
from .whep_registry import (DEFAULT_MAX_VIEWERS, DEFAULT_RELEASE_GRACE_S,
                            ViewerLimit, WhepRegistry)

STREAMS = ("color_near", "color_far", "depth_near", "depth_far")
DEPTH_VIEW = (424, 240)    # rendered depth size (matches render_depth)
FIRST_FRAME_TIMEOUT_S = 5.0
ENCODER_IDLE_TIMEOUT_S = 8.0   # viewers present but no AU for this long
REAP_INTERVAL_S = 0.5
DEFAULT_BITRATES = {"color": 3_500_000, "depth": 1_000_000}
DEFAULT_IDR_INTERVAL_S = 1.0
DEFAULT_MAX_FPS = 30.0


def render_depth(depth_mm):
    """uint16 (480, 848) mm -> half-res BGR turbo view, 0-4 m, invalid=black."""
    m = depth_mm > 0
    v = np.clip(depth_mm.astype(np.float32) / 4000.0, 0, 1) * 255
    vis = cv2.applyColorMap(v.astype(np.uint8), cv2.COLORMAP_TURBO)
    vis[~m] = 0
    return cv2.resize(vis, DEPTH_VIEW, interpolation=cv2.INTER_AREA)


class NoFrame(RuntimeError):
    """The camera produced no frame to size the encoder from."""


def _wait_frame(cam, timeout_s):
    deadline = time.monotonic() + timeout_s
    while True:
        fr = cam.read() if cam else None
        if fr is not None or time.monotonic() >= deadline:
            return fr
        time.sleep(0.05)


class _StreamHub:
    """One stream's shared encoder and the feeder thread driving it.

    The feeder reads the camera, renders depth once (not per viewer) and
    pushes into the encoder only while at least one viewer is attached; the
    encoder's access units fan out to every attached peer. Keyframe
    requests (a viewer joining, or a browser PLI) are queued here and sent
    from the feeder thread, never from a GStreamer streaming thread.
    """

    def __init__(self, name, cam, kind, bitrate, idr_interval_s=1.0,
                 max_fps=DEFAULT_MAX_FPS, make_encoder=None):
        self.name = name
        self._cam = cam
        self._kind = kind
        fr = _wait_frame(cam, FIRST_FRAME_TIMEOUT_S)
        if fr is None:
            raise NoFrame(f"no frame from {name}")
        fps = float(getattr(fr, "fps", 0) or 0) or 15.0
        fps = min(fps, float(max_fps)) if max_fps else fps
        self._min_interval = 0.9 / fps
        self._poll_s = min(0.03, max(0.005, 0.5 / fps))
        if make_encoder is None:
            from .whep import H264Encoder
            make_encoder = H264Encoder
        if kind == "color":
            enc_kw = dict(input_kind="jpeg", fps=fps)
        else:
            enc_kw = dict(input_kind="bgr", fps=fps, width=DEPTH_VIEW[0],
                          height=DEPTH_VIEW[1])
        self._lock = threading.Lock()
        self._peers = set()
        self._kf_wanted = False
        self._stop = threading.Event()
        self._error = None
        self._feeding_since = None
        self.frames = 0
        self.encoder = make_encoder(bitrate=int(bitrate),
                                    idr_interval_s=idr_interval_s,
                                    on_au=self._fanout, **enc_kw)
        self._thread = threading.Thread(target=self._feed, daemon=True,
                                        name=f"whep-feed-{name}")
        self._thread.start()

    # ------------------------------------------------------------ fan-out
    def _fanout(self, au, is_idr):
        with self._lock:
            peers = tuple(self._peers)
        for peer in peers:
            peer.push_au(au, is_idr)

    def add_peer(self, peer):
        # No keyframe here: the peer asks for one once its ICE connects
        # (an IDR sent before that is dropped by webrtcbin).
        with self._lock:
            self._peers.add(peer)

    def remove_peer(self, peer):
        with self._lock:
            self._peers.discard(peer)

    def keyframe(self):
        """Ask for an IDR on the next feeder tick. Safe to call from any
        thread, including a GStreamer streaming thread (it only sets a
        flag; the feeder sends the upstream event)."""
        self._kf_wanted = True

    def viewers(self):
        with self._lock:
            return len(self._peers)

    # ------------------------------------------------------------- feeder
    def _feed(self):
        last = None
        last_push = 0.0
        try:
            while not self._stop.is_set():
                if not self.viewers():
                    self._feeding_since = None
                    time.sleep(0.05)
                    continue
                now = time.monotonic()
                if self._feeding_since is None:
                    self._feeding_since = now
                if self._kf_wanted:
                    self._kf_wanted = False
                    self.encoder.request_keyframe()
                fr = self._cam.read() if self._cam else None
                if (fr is not None and fr is not last
                        and now - last_push >= self._min_interval):
                    last = fr
                    if self._kind == "color":
                        data = getattr(fr, "color_jpeg", None)
                    else:
                        data = (render_depth(fr.depth)
                                if fr.depth is not None else None)
                    if data is not None:
                        self.encoder.push(data)
                        self.frames += 1
                        last_push = now
                time.sleep(self._poll_s)
        except Exception as exc:  # noqa: BLE001 - surfaced through poll()
            self._error = f"feeder: {exc}"

    def poll(self, now):
        """Reason the shared encoder is broken, or None."""
        if self._error:
            return self._error
        if not self._thread.is_alive() and not self._stop.is_set():
            return "feeder stopped"
        err = self.encoder.take_error()
        if err is not None:
            return f"pipeline error: {err}"
        since = self._feeding_since
        if since is not None and self.viewers():
            last = self.encoder.last_au_at
            ref = since if last is None or last < since else last
            if now - ref > ENCODER_IDLE_TIMEOUT_S:
                return "encoder idle"
        return None

    def stats(self):
        e = self.encoder
        return {"frames": self.frames, "au_out": getattr(e, "au_out", 0),
                "idr_out": getattr(e, "idr_out", 0),
                "keyframe_requests": getattr(e, "keyframe_requests", 0),
                "viewers": self.viewers()}

    def close(self):
        self._stop.set()
        if self._thread is not threading.current_thread():
            self._thread.join(timeout=1.0)
        with self._lock:
            self._peers.clear()
        self.encoder.close()


class VideoServer:
    def __init__(self, rig, port=9092, notes=None, bitrates=None,
                 max_viewers=DEFAULT_MAX_VIEWERS,
                 release_grace_s=DEFAULT_RELEASE_GRACE_S,
                 idr_interval_s=DEFAULT_IDR_INTERVAL_S,
                 max_fps=DEFAULT_MAX_FPS):
        self.rig = rig
        self.port = port
        self.notes = notes          # SessionNotes or None
        self.bitrates = dict(DEFAULT_BITRATES)
        self.bitrates.update(bitrates or {})
        self.idr_interval_s = float(idr_interval_s)
        self.max_fps = float(max_fps)
        self.registry = WhepRegistry(self._make_hub, max_viewers=max_viewers,
                                     release_grace_s=release_grace_s)
        self._httpd = None
        self._thread = None
        self._reaper = None
        self._stop = threading.Event()

    def _make_hub(self, name):
        kind, role = name.rsplit("_", 1)
        cam = self.rig.cameras.get(role)
        if cam is None:
            raise NoFrame(f"no {role} camera")
        return _StreamHub(name, cam, kind, self.bitrates[kind],
                          idr_interval_s=self.idr_interval_s,
                          max_fps=self.max_fps)

    def _reap_loop(self):
        while not self._stop.wait(REAP_INTERVAL_S):
            try:
                self.registry.reap()
            except Exception as exc:  # noqa: BLE001 - keep reaping
                print(f"[videoserver] reap failed: {exc}")

    def start(self):
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

            def _route(self):
                return urlparse(self.path).path.rstrip("/") or "/"

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

            def _send_json(self, code, payload, headers=None):
                body = json.dumps(payload).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                # Expose-Headers only takes effect on the actual response
                # (it is ignored on a preflight), so the 503's Retry-After
                # is readable cross-origin.
                self.send_header("Access-Control-Expose-Headers",
                                 "Location, Retry-After")
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _read_json(self):
                """A JSON object body, or ValueError (answered as 400).  A
                missing or chunked body is an error, never an empty object:
                an empty /intent would otherwise silently clear the intent."""
                if self.headers.get("Transfer-Encoding"):
                    raise ValueError("send a Content-Length body, not chunked")
                try:
                    length = int(self.headers.get("Content-Length") or 0)
                except ValueError:
                    length = 0
                if length <= 0:
                    raise ValueError("JSON object body required")
                raw = self.rfile.read(min(length, 65536))
                if not raw.strip():
                    raise ValueError("JSON object body required")
                payload = json.loads(raw.decode("utf-8", "replace"))
                if not isinstance(payload, dict):
                    raise ValueError("JSON object expected")
                return payload

            def _note(self, route):
                notes = server.notes
                if notes is None:
                    self._send_json(503, {"error": "no session notes"})
                    return
                try:
                    payload = self._read_json()
                    if route == "/intent":
                        if payload.get("clear"):
                            intent = notes.clear_intent()
                        else:
                            intent = notes.set_intent(
                                payload.get("text", ""),
                                payload.get("from", payload.get("from_place", "")),
                                payload.get("to", payload.get("to_place", "")))
                        self._send_json(200, {"ok": True,
                                              "intent": intent.to_dict()})
                    else:
                        ev = notes.add_event(payload.get("kind", "note"),
                                             payload.get("note", ""))
                        self._send_json(200, {"ok": True, "event": ev})
                except NotRecording as exc:
                    self._send_json(409, {"error": str(exc), "recording": False})
                except (ValueError, TypeError, RecursionError) as exc:
                    self._send_json(400, {"error": str(exc)})

            def do_GET(self):
                # Don't hold keep-alive connections open: each WHEP POST
                # from a reconnecting client otherwise leaves a handler
                # thread + socket alive until timeout, leaking fds.
                self.close_connection = True
                route = self._route()
                if route == "/snapshot":
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
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif route == "/whep":
                    self._send_json(200, {
                        "streams": server.registry.stats(),
                        "max_viewers": server.registry.max_viewers})
                elif route == "/notes":
                    if server.notes is None:
                        self._send_json(503, {"error": "no session notes"})
                    else:
                        self._send_json(200, server.notes.snapshot())
                elif route == "/healthz":
                    self.send_response(200)
                    self.send_header("Content-Length", "2")
                    self.end_headers()
                    self.wfile.write(b"ok")
                else:
                    self.send_error(404)

            def do_OPTIONS(self):
                # WHEP POSTs carry `Content-Type: application/sdp`, which
                # makes them non-simple, so browsers preflight them; the
                # DELETE on the session resource is preflighted too.
                self.close_connection = True
                self.send_response(204)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods",
                                 "GET, POST, DELETE, OPTIONS")
                self.send_header("Access-Control-Allow-Headers",
                                 "Content-Type, If-Match")
                self.send_header("Access-Control-Max-Age", "600")
                self.send_header("Content-Length", "0")
                self.end_headers()

            def do_POST(self):
                self.close_connection = True
                route = self._route()
                if route == "/whep":
                    self._whep()
                elif route == "/intent":
                    self._note("/intent")
                elif route == "/event":
                    self._note("/event")
                else:
                    self.send_error(404)

            def do_DELETE(self):
                self.close_connection = True
                route = self._route()
                if route.startswith("/whep/"):
                    sid = route[len("/whep/"):]
                    if server.registry.remove(sid, "client DELETE"):
                        self._send_json(200, {"ok": True})
                    else:
                        self._send_json(404, {"error": "unknown session"})
                else:
                    self._send_json(405, {"error": "method not allowed"})

            def do_PATCH(self):
                # Non-trickle server: the answer already carries every
                # candidate, so ICE trickle/restart PATCHes are refused.
                self.close_connection = True
                self._send_json(405, {"error": "trickle ICE not supported"},
                                {"Allow": "DELETE"})

            def _whep(self):
                """WHEP offer in, answer out; the viewer then shares the
                stream's encoder until it leaves."""
                from .whep import WhepPeer, webrtc_available
                name, role, kind = self._pick()
                try:
                    length = int(self.headers.get("Content-Length") or 0)
                except ValueError:
                    length = 0
                offer = self.rfile.read(length).decode("utf-8", "replace")
                if not webrtc_available():
                    self._send_json(503, {"error": "webrtc unavailable"})
                    return
                if "m=video" not in offer:
                    self._send_json(400, {"error": "SDP offer with a video "
                                                   "m-line required"})
                    return
                reg = server.registry
                q = parse_qs(urlparse(self.path).query)
                replace = (q.get("replace") or [""])[0]
                if replace:
                    reg.remove(replace, "replaced by client")
                reg.reap()          # free slots held by dead viewers first
                try:
                    sid = reg.reserve(name)
                except ViewerLimit as exc:
                    self._send_json(503, {
                        "error": "viewer limit", "stream": name,
                        "viewers": exc.viewers,
                        "max_viewers": exc.max_viewers,
                        "detail": str(exc)}, {"Retry-After": "5"})
                    return
                peer = None
                try:
                    hub = reg.hub(name)
                    peer = WhepPeer(on_keyframe_request=hub.keyframe)
                    answer = peer.negotiate(offer)
                except NoFrame as exc:
                    reg.remove(sid, "no frame")
                    if peer is not None:
                        peer.close()
                    self._send_json(503, {"error": "no frame",
                                          "detail": str(exc)})
                    return
                except Exception as exc:  # noqa: BLE001
                    reg.remove(sid, f"negotiation failed: {exc}")
                    if peer is not None:
                        peer.close()
                    print(f"[videoserver] whep {name} failed: {exc}")
                    self._send_json(503, {"error": "webrtc failed",
                                          "detail": str(exc)})
                    return
                if not reg.attach(sid, peer):
                    self._send_json(409, {"error": "session closed while "
                                                   "negotiating"})
                    return
                body = answer.encode()
                self.send_response(201)
                self.send_header("Content-Type", "application/sdp")
                self.send_header("Location", f"/whep/{sid}")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Expose-Headers",
                                 "Location, Retry-After")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                print(f"[videoserver] whep {name} viewer {sid[:6]} open "
                      f"({reg.viewers(name)}/{reg.max_viewers})")

        try:
            self._httpd = ThreadingHTTPServer(("0.0.0.0", self.port), Handler)
        except OSError as exc:
            print(f"[videoserver] port {self.port} unavailable ({exc}); "
                  f"operator video disabled")
            self._httpd = None
            return
        self._httpd.daemon_threads = True
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._httpd.serve_forever, kwargs={"poll_interval": 0.25},
            daemon=True, name="videoserver")
        self._thread.start()
        self._reaper = threading.Thread(target=self._reap_loop, daemon=True,
                                        name="whep-reaper")
        self._reaper.start()
        print(f"[videoserver] serving WebRTC (WHEP) streams {STREAMS} "
              f"on :{self.port}/whep (max {self.registry.max_viewers} viewers "
              f"per stream; color {self.bitrates['color'] // 1000} kbps, "
              f"depth {self.bitrates['depth'] // 1000} kbps)")

    def stop(self):
        self._stop.set()
        if self._reaper is not None:
            self._reaper.join(timeout=2.0)
            self._reaper = None
        self.registry.close_all()
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        self._thread = None
