"""WHEP (WebRTC-HTTP Egress Protocol) operator stream: shared encoder fan-out.

One H.264 encoder per camera stream, shared by every viewer of it:

    feeder ─push─> H264Encoder                    (one per stream)
                   appsrc ! (nv)jpegdec|videoconvert ! nvvidconv
                   ! nvv4l2h264enc ! h264parse ! appsink
                          │ Annex-B access units (new-sample callback)
               ┌──────────┼──────────┐
               v          v          v
            WhepPeer   WhepPeer   WhepPeer        (one per viewer, cheap)
            appsrc ! rtph264pay ! webrtcbin

Fan-out happens on encoded access units, not inside one pipeline, so a
viewer joining or leaving never touches the encoder or another viewer's
pipeline, and each viewer gets its own negotiated RTP payload type / SSRC.
Adding a viewer adds no encoder work: per-viewer cost is RTP payloading +
SRTP.

A joining viewer must start on an IDR. The peer drops access units until
the first IDR, and the hub asks the encoder for one right away
(`GstForceKeyUnit` upstream), so the first picture arrives in a frame or
two instead of waiting out the IDR interval. Browser PLI/FIR feedback (sent
after loss) arrives at the peer's appsrc as the same upstream event and is
forwarded to the shared encoder, rate limited.

The browser POSTs an SDP offer to `/whep`; the answer is non-trickle (host
candidates gathered before replying). Media flows over SRTP/UDP, so latency
is ~150 ms and loss doesn't head-of-line-block.

Requires the GStreamer `nice` plugin (`gstreamer1.0-nice`); without it
`webrtcbin` refuses to leave NULL ("libnice elements are not available").

H.264 only: WebRTC H.265 is Safari-only. The encoder is the Jetson NVENC
chain the recorder uses (`nvv4l2h264enc`); on a box without NVENC it falls
back to `x264enc` (ultrafast/zerolatency, constrained baseline) so the
server still works on a dev machine.
"""

from __future__ import annotations

import sys
import threading
import time

from .whep_registry import PeerWatch

_SYSTEM_GI = "/usr/lib/python3/dist-packages"
_gst = None
_gst_lock = threading.Lock()

KEYFRAME_MIN_INTERVAL_S = 0.25   # rate limit for join/PLI keyframe requests


def _load():
    global _gst
    with _gst_lock:
        if _gst is not None:
            return _gst
        if _SYSTEM_GI not in sys.path:
            sys.path.append(_SYSTEM_GI)
        import gi

        gi.require_version("Gst", "1.0")
        gi.require_version("GstApp", "1.0")
        gi.require_version("GstWebRTC", "1.0")
        gi.require_version("GstSdp", "1.0")
        try:
            gi.require_version("GstVideo", "1.0")   # force-key-unit helper
        except ValueError:
            pass
        from gi.repository import Gst, GstApp, GstWebRTC, GstSdp

        if not Gst.is_initialized():
            Gst.init(None)
        _gst = (Gst, GstApp, GstWebRTC, GstSdp)
        return _gst


def _h264_payload_type(sdp_text: str):
    """Payload type the offer uses for H264/90000. Chrome puts H264 well
    above 96 (e.g. 96 is VP8), and the answer reuses the offer's PT, so
    the payloader must be told to send that PT."""
    for line in sdp_text.splitlines():
        if line.startswith("a=rtpmap:") and "H264/90000" in line:
            try:
                return int(line.split(":", 1)[1].split(" ", 1)[0])
            except (ValueError, IndexError):
                return None
    return None


def has_idr(data) -> bool:
    """True when an Annex-B H.264 access unit contains an IDR slice."""
    i = 0
    n = len(data)
    while True:
        j = data.find(b"\x00\x00\x01", i)
        if j < 0:
            return False
        k = j + 3
        while k < n and data[k] == 0:
            k += 1
        if k < n and (data[k] & 0x1F) == 5:
            return True
        i = j + 3


def _find(Gst, name) -> bool:
    return Gst.ElementFactory.find(name) is not None


def encoder_backend():
    """"nvenc" (Jetson), "x264" (software fallback) or None."""
    try:
        Gst, *_ = _load()
    except Exception:
        return None
    if _find(Gst, "nvv4l2h264enc") and _find(Gst, "nvvidconv"):
        return "nvenc"
    if _find(Gst, "x264enc"):
        return "x264"
    return None


def webrtc_available() -> bool:
    """True when the encoder + a webrtcbin send pipeline can be built here."""
    try:
        Gst, *_ = _load()
    except Exception:
        return False
    for el in ("webrtcbin", "nicesrc", "nicesink", "rtph264pay", "h264parse",
               "appsrc", "appsink", "videoconvert"):
        if not _find(Gst, el):
            return False
    return encoder_backend() is not None


def _force_key_unit_event(Gst):
    try:
        from gi.repository import GstVideo  # noqa: WPS433
        return GstVideo.video_event_new_upstream_force_key_unit(
            Gst.CLOCK_TIME_NONE, True, 0)
    except Exception:
        st = Gst.Structure.new_from_string(
            "GstForceKeyUnit, all-headers=(boolean)true")
        return Gst.Event.new_custom(Gst.EventType.CUSTOM_UPSTREAM, st)


def _is_force_key_unit(Gst, event) -> bool:
    try:
        if event.type != Gst.EventType.CUSTOM_UPSTREAM:
            return False
        st = event.get_structure()
        return st is not None and st.get_name() == "GstForceKeyUnit"
    except Exception:
        return False


def _drain_bus(Gst, pipeline):
    """Pop pending bus messages; return an error/EOS string or None."""
    if pipeline is None:
        return None
    try:
        bus = pipeline.get_bus()
        if bus is None:
            return None
        while True:
            msg = bus.try_pop()
            if msg is None:
                return None
            if msg.type == Gst.MessageType.ERROR:
                err, dbg = msg.parse_error()
                return f"{err.message} [{dbg or ''}]"
            if msg.type == Gst.MessageType.EOS:
                return "EOS"
    except Exception:
        return None


class H264Encoder:
    """One shared H.264 encoder: frames in, Annex-B access units out.

    `input_kind="jpeg"` takes camera MJPEG bytes; `input_kind="bgr"` takes
    a BGR ndarray of `width`x`height` (server-rendered depth).
    `on_au(au_bytes, is_idr)` runs on the GStreamer streaming thread for
    every access unit; keep it short and non-blocking.
    `push` is driven by a single feeder thread.
    """

    def __init__(self, input_kind="jpeg", width=0, height=0, fps=15.0,
                 bitrate=3_500_000, idr_interval_s=1.0, on_au=None):
        Gst, GstApp, _GstWebRTC, _GstSdp = _load()
        backend = encoder_backend()
        if backend is None:
            raise RuntimeError("no H.264 encoder (nvv4l2h264enc / x264enc)")
        self.backend = backend
        self._Gst = Gst
        self._input_kind = input_kind
        self._on_au = on_au
        self._closed = False
        self._fps = float(fps) if fps else 15.0
        self._dur_ns = int(round(Gst.SECOND / self._fps))
        self._pts = 0
        self._kf_lock = threading.Lock()
        self._last_kf_req = 0.0
        self.au_out = 0
        self.idr_out = 0
        self.keyframe_requests = 0
        self.last_au_at = None

        pipeline = Gst.Pipeline.new("whep-enc")

        def mk(factory, name, **props):
            el = Gst.ElementFactory.make(factory, name)
            if el is None:
                raise RuntimeError(f"missing GStreamer element {factory}")
            for k, v in props.items():
                if isinstance(v, str):
                    Gst.util_set_object_arg(el, k.replace("_", "-"), v)
                else:
                    el.set_property(k.replace("_", "-"), v)
            pipeline.add(el)
            return el

        def caps(name, text):
            return mk("capsfilter", name, caps=Gst.Caps.from_string(text))

        self._src = mk("appsrc", "src", is_live=True)
        self._src.set_property("format", Gst.Format.TIME)
        # Non-blocking producer: a stalled encoder must never wedge the
        # feeder thread; dropping stale frames is right for live video.
        self._src.set_property("block", False)
        self._src.set_property("max-buffers", 4)
        self._src.set_property("leaky-type", GstApp.AppLeakyType.DOWNSTREAM)
        rate = max(1, int(round(self._fps)))
        gop = max(1, int(round(self._fps * float(idr_interval_s))))

        chain = []
        if input_kind == "jpeg":
            self._src.set_property("caps", Gst.Caps.from_string(
                f"image/jpeg,framerate={rate}/1"))
            if backend == "nvenc" and _find(Gst, "nvjpegdec"):
                chain.append(mk("nvjpegdec", "dec"))
            else:
                chain.append(mk("jpegdec", "dec"))
                chain.append(mk("videoconvert", "vc"))
                chain.append(caps("cf_in", "video/x-raw,format=I420"))
        else:
            self._src.set_property("caps", Gst.Caps.from_string(
                f"video/x-raw,format=BGR,width={int(width)},"
                f"height={int(height)},framerate={rate}/1"))
            chain.append(mk("videoconvert", "vc"))
            chain.append(caps("cf_in", "video/x-raw,format=I420"))

        if backend == "nvenc":
            chain.append(mk("nvvidconv", "conv"))
            chain.append(caps("cf", "video/x-raw(memory:NVMM),format=NV12"))
            enc = mk("nvv4l2h264enc", "enc")
            enc.set_property("bitrate", int(bitrate))
            enc.set_property("control-rate", 1)  # constant_bitrate
            # idrinterval, not just iframeinterval: nvv4l2h264enc emits
            # non-IDR I-frames at iframeinterval, while IDR keyframes (the
            # only point a decoder can (re)start from) default to every
            # 256 frames (~17 s).
            enc.set_property("iframeinterval", gop)
            enc.set_property("idrinterval", gop)
            chain.append(enc)
        else:
            enc = mk("x264enc", "enc", tune="zerolatency",
                     speed_preset="ultrafast")
            enc.set_property("bitrate", max(1, int(bitrate) // 1000))  # kbit/s
            enc.set_property("key-int-max", gop)
            chain.append(enc)
            chain.append(caps("cf_prof",
                              "video/x-h264,profile=constrained-baseline"))
        self._enc = enc
        parse = mk("h264parse", "parse")
        # SPS/PPS in front of every IDR, so any IDR is a valid start point
        # for a viewer that joins mid-stream.
        parse.set_property("config-interval", -1)
        chain.append(parse)
        chain.append(caps("cf_out", "video/x-h264,stream-format=byte-stream,"
                                    "alignment=au"))
        self._sink = mk("appsink", "sink")
        self._sink.set_property("emit-signals", True)
        self._sink.set_property("sync", False)
        self._sink.set_property("max-buffers", 4)
        self._sink.set_property("drop", True)
        self._sink_handler = self._sink.connect("new-sample", self._on_sample)

        prev = self._src
        for el in chain + [self._sink]:
            # Element.link returns a Python bool; Pad.link returns the enum.
            if not prev.link(el):
                raise RuntimeError(f"failed to link {prev.name} -> {el.name}")
            prev = el

        self._pipeline = pipeline
        if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            pipeline.set_state(Gst.State.NULL)
            raise RuntimeError("whep encoder pipeline failed to start")

    def _on_sample(self, sink):
        Gst = self._Gst
        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.OK
        buf = sample.get_buffer()
        try:
            au = bytes(buf.extract_dup(0, buf.get_size()))
        except Exception:
            return Gst.FlowReturn.OK
        idr = has_idr(au)
        self.au_out += 1
        if idr:
            self.idr_out += 1
        self.last_au_at = time.monotonic()
        cb = self._on_au
        if cb is not None:
            try:
                cb(au, idr)
            except Exception as exc:  # noqa: BLE001 - never kill streaming
                print(f"[whep] fan-out error: {exc}")
        return Gst.FlowReturn.OK

    def push(self, data) -> None:
        """Push one frame: MJPEG bytes (jpeg) or a BGR ndarray (bgr)."""
        src = self._src
        if self._closed or src is None:
            return
        Gst = self._Gst
        if self._input_kind == "jpeg":
            blob = data if isinstance(data, (bytes, bytearray)) else bytes(data)
        else:
            import numpy as np

            blob = np.ascontiguousarray(data).tobytes()
        buf = Gst.Buffer.new_allocate(None, len(blob), None)
        buf.fill(0, blob)
        buf.pts = self._pts
        buf.duration = self._dur_ns
        self._pts += self._dur_ns
        src.emit("push-buffer", buf)

    def request_keyframe(self) -> bool:
        """Ask the encoder for an IDR now (rate limited). True if sent."""
        now = time.monotonic()
        with self._kf_lock:
            if now - self._last_kf_req < KEYFRAME_MIN_INTERVAL_S:
                return False
            self._last_kf_req = now
        enc = self._enc
        if self._closed or enc is None:
            return False
        try:
            # An upstream event is *sent to* a src pad: it lands in the
            # encoder's src-event handler (GstVideoEncoder force-key-unit).
            ok = enc.get_static_pad("src").send_event(
                _force_key_unit_event(self._Gst))
        except Exception:
            ok = False
        if ok:
            self.keyframe_requests += 1
        return bool(ok)

    def take_error(self):
        """Drain the encoder bus; an error/EOS string or None."""
        return _drain_bus(self._Gst, self._pipeline)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._on_au = None
        pipeline = self._pipeline
        try:
            self._src.emit("end-of-stream")
        except Exception:
            pass
        try:
            self._sink.disconnect(self._sink_handler)
        except Exception:
            pass
        try:
            pipeline.set_state(self._Gst.State.NULL)
        except Exception:
            pass
        # Drop references so the NVMM/dmabuf fds are released on finalize.
        self._pipeline = None
        self._src = None
        self._sink = None
        self._enc = None


# Per-viewer appsrc queue depth (encoded AUs). push_au drops to the next
# IDR itself before this fills, so appsrc's own leaky drop (which would
# silently cut a frame out of the reference chain with contiguous RTP
# sequence numbers, i.e. no loss for the browser to PLI on) never fires.
PEER_QUEUE_AUS = 8
PEER_QUEUE_HIGH = PEER_QUEUE_AUS - 2


def _await_promise(Gst, emit, timeout_s, what):
    """emit(promise) then wait for its reply with a deadline.

    Gst.Promise.wait() has no timeout; a webrtcbin that never answers
    would park the HTTP handler (and its reserved viewer slot) forever.
    On timeout the promise is interrupted and RuntimeError raised."""
    done = threading.Event()
    p = Gst.Promise.new_with_change_func(lambda *_a: done.set())
    emit(p)
    if not done.wait(timeout_s):
        try:
            p.interrupt()
        except Exception:
            pass
        raise RuntimeError(f"{what} timed out after {timeout_s:.0f} s")
    # Already resolved, so wait() returns at once. (GstPromise has no
    # get_result() in the robot's GStreamer 1.24 bindings.)
    result = p.wait()
    if result != Gst.PromiseResult.REPLIED:
        raise RuntimeError(f"{what} failed ({result.value_nick})")
    return p.get_reply()


_STATE_NAMES = ("new", "connecting", "connected", "disconnected", "failed",
                "closed")


class WhepPeer:
    """One viewer: encoded AUs -> rtph264pay -> webrtcbin.

    `push_au` is called from the shared encoder's streaming thread;
    `negotiate`, `poll` and `close` from HTTP / reaper threads. A lock
    keeps push and close from racing.
    """

    def __init__(self, on_keyframe_request=None, latency_ms=0, **watch_kw):
        Gst, GstApp, GstWebRTC, GstSdp = _load()
        self._Gst = Gst
        self._GstWebRTC = GstWebRTC
        self._GstSdp = GstSdp
        self._lock = threading.Lock()
        self._closed = False
        self._pad_linked = False
        self._synced = False          # an IDR has been delivered
        self._kf_asked = 0.0
        self._on_kf = on_keyframe_request
        self.watch = PeerWatch(time.monotonic(), **watch_kw)
        self.au_in = 0
        self.au_sent = 0
        self.au_dropped = 0
        self.pli = 0

        pipeline = Gst.Pipeline.new("whep-peer")

        def mk(factory, name):
            el = Gst.ElementFactory.make(factory, name)
            if el is None:
                raise RuntimeError(f"missing GStreamer element {factory}")
            pipeline.add(el)
            return el

        self._src = mk("appsrc", "src")
        self._src.set_property("is-live", True)
        self._src.set_property("format", Gst.Format.TIME)
        # Timestamp on arrival: AUs are pushed as soon as they leave the
        # shared encoder, so arrival time is a faithful capture clock and
        # gives each peer its own zero-based timeline.
        self._src.set_property("do-timestamp", True)
        self._src.set_property("block", False)
        self._src.set_property("max-buffers", PEER_QUEUE_AUS)
        self._src.set_property("leaky-type", GstApp.AppLeakyType.DOWNSTREAM)
        self._src.set_property("caps", Gst.Caps.from_string(
            "video/x-h264,stream-format=byte-stream,alignment=au"))
        self._pay = mk("rtph264pay", "pay")
        self._pay.set_property("config-interval", -1)
        self._pay.set_property("pt", 96)
        self._webrtc = mk("webrtcbin", "sendrecv")
        self._webrtc.set_property(
            "bundle-policy", GstWebRTC.WebRTCBundlePolicy.MAX_BUNDLE)
        self._webrtc.set_property("latency", int(latency_ms))
        if not self._src.link(self._pay):
            raise RuntimeError("failed to link appsrc -> rtph264pay")

        # Browser PLI/FIR arrives as an upstream GstForceKeyUnit event
        # from webrtcbin's RTP session; forward it to the shared encoder.
        def _upstream(pad, info):
            ev = info.get_event()
            if ev is not None and _is_force_key_unit(Gst, ev):
                self.pli += 1
                cb = self._on_kf
                if cb is not None:
                    try:
                        cb()
                    except Exception:
                        pass
            return Gst.PadProbeReturn.OK

        self._probe_cb = _upstream
        self._src.get_static_pad("src").add_probe(
            Gst.PadProbeType.EVENT_UPSTREAM, _upstream)
        self._watch_rtcp()

        self._pipeline = pipeline
        if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            pipeline.set_state(Gst.State.NULL)
            raise RuntimeError("webrtc pipeline failed to start")
        pipeline.get_state(Gst.CLOCK_TIME_NONE)

    def _watch_rtcp(self):
        """Hook webrtcbin's internal rtpbin: RTCP from the browser marks
        the viewer alive, an RTCP BYE marks it gone. Best effort; without
        it the peer still times out on ICE state."""
        rtpbin = None
        try:
            rtpbin = self._webrtc.get_by_name("rtpbin")
        except Exception:
            rtpbin = None
        if rtpbin is None:
            return
        watch = self.watch

        def _active(_bin, _session, _ssrc):
            watch.on_rtcp(time.monotonic())

        def _bye(_bin, _session, _ssrc):
            watch.on_bye()

        try:
            self._rtcp_handlers = (rtpbin, [
                rtpbin.connect("on-ssrc-active", _active),
                rtpbin.connect("on-bye-ssrc", _bye),
            ])
        except Exception:
            self._rtcp_handlers = None

    # ------------------------------------------------------------ signaling
    def negotiate(self, offer_sdp: str, timeout_s: float = 5.0,
                  promise_timeout_s: float = 5.0) -> str:
        """Apply a browser offer, return the answer SDP (non-trickle).
        Each webrtcbin step is bounded by `promise_timeout_s`."""
        Gst = self._Gst
        GstSdp = self._GstSdp
        GstWebRTC = self._GstWebRTC
        webrtc = self._webrtc

        res, sdp = GstSdp.SDPMessage.new()
        GstSdp.sdp_message_parse_buffer(offer_sdp.encode(), sdp)
        offer = GstWebRTC.WebRTCSessionDescription.new(
            GstWebRTC.WebRTCSDPType.OFFER, sdp)
        _await_promise(
            Gst, lambda p: webrtc.emit("set-remote-description", offer, p),
            promise_timeout_s, "set-remote-description")

        # Send H.264 on the payload type the offer negotiated for it
        # (not necessarily 96 — Chrome uses 96 for VP8).
        pt = _h264_payload_type(offer_sdp)
        if pt is not None:
            self._pay.set_property("pt", pt)

        # The offer's m-line created a transceiver; link to its sink pad.
        pad = self._webrtc.request_pad_simple("sink_%u")
        if pad is None:
            pad = self._webrtc.get_static_pad("sink_0")
        if pad is None:
            raise RuntimeError("webrtcbin produced no sink pad")
        # Pad.link returns a Gst.PadLinkReturn enum whose OK is 0 (falsy).
        if self._pay.get_static_pad("src").link(pad) != Gst.PadLinkReturn.OK:
            raise RuntimeError("failed to link payloader to webrtcbin")

        reply = _await_promise(
            Gst, lambda p: webrtc.emit("create-answer", None, p),
            promise_timeout_s, "create-answer")
        answer = reply.get_value("answer") if reply is not None else None
        if answer is None:
            err = reply.get_value("error") if reply is not None else None
            raise RuntimeError(
                f"create-answer failed: {err.message if err else 'unknown'}")

        _await_promise(
            Gst, lambda p: webrtc.emit("set-local-description", answer, p),
            promise_timeout_s, "set-local-description")

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if (self._webrtc.get_property("ice-gathering-state")
                    == GstWebRTC.WebRTCICEGatheringState.COMPLETE):
                break
            time.sleep(0.02)
        local = self._webrtc.get_property("local-description")
        with self._lock:
            self._pad_linked = True
        return (local or answer).sdp.as_text()

    # --------------------------------------------------------------- media
    def push_au(self, au: bytes, is_idr: bool) -> None:
        """Forward one encoded access unit; nothing until the first IDR."""
        with self._lock:
            if self._closed or not self._pad_linked:
                return
            self.au_in += 1
            if not self._synced:
                # Start on an IDR, and only once ICE/DTLS is up: media
                # pushed before that is dropped by webrtcbin, so an IDR
                # spent earlier would leave the viewer on undecodable
                # P-frames until the next periodic IDR.
                if self.state_name() != "connected":
                    return
                if not is_idr:
                    now = time.monotonic()
                    cb = self._on_kf
                    if cb is not None and now - self._kf_asked >= 0.5:
                        self._kf_asked = now
                        cb()
                    return
                self._synced = True
            src = self._src
            Gst = self._Gst
            try:
                level = src.get_property("current-level-buffers")
            except Exception:
                level = 0
            if level >= PEER_QUEUE_HIGH:
                # This viewer's link is backed up. Dropping one AU here
                # would break the reference chain invisibly (RTP seqnums
                # stay contiguous, so no PLI), so instead stop sending
                # and resync cleanly on the next IDR.
                self.au_dropped += 1
                self._synced = False
                now = time.monotonic()
                cb = self._on_kf
                if cb is not None and now - self._kf_asked >= 0.5:
                    self._kf_asked = now
                    cb()
                return
            buf = Gst.Buffer.new_allocate(None, len(au), None)
            buf.fill(0, au)
            if not is_idr:
                try:
                    buf.set_flags(Gst.BufferFlags.DELTA_UNIT)
                except Exception:
                    pass
            src.emit("push-buffer", buf)
            self.au_sent += 1
        self.watch.on_media(time.monotonic())

    @property
    def synced(self) -> bool:
        return self._synced

    def state_name(self):
        w = self._webrtc
        if w is None:
            return "closed"
        try:
            st = w.get_property("connection-state")
        except Exception:
            return None
        nick = getattr(st, "value_nick", None)
        if nick:
            return nick
        try:
            return _STATE_NAMES[int(st)]
        except Exception:
            return None

    def poll(self, now):
        """Reason to tear this viewer down, or None while it is healthy."""
        if self._closed:
            return "closed"
        err = _drain_bus(self._Gst, self._pipeline)
        if err is not None:
            return f"pipeline error: {err}"
        return self.watch.verdict(self.state_name(), now)

    def stats(self) -> dict:
        return {"state": self.state_name(), "au_in": self.au_in,
                "au_sent": self.au_sent, "au_dropped": self.au_dropped,
                "pli": self.pli}

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            pipeline = self._pipeline
            src = self._src
            handlers = getattr(self, "_rtcp_handlers", None)
            # Drop every reference so webrtcbin/libnice finalize now. The
            # ICE agent's sockets are only released on finalize; holding
            # them (plus probe closures that capture self) bled sockets
            # until fd numbers crossed FD_SETSIZE and select() aborted.
            self._pipeline = None
            self._webrtc = None
            self._src = None
            self._pay = None
            self._probe_cb = None
            self._on_kf = None
            self._rtcp_handlers = None
        if handlers:
            rtpbin, ids = handlers
            for hid in ids:
                try:
                    rtpbin.disconnect(hid)
                except Exception:
                    pass
        try:
            src.emit("end-of-stream")
        except Exception:
            pass
        try:
            pipeline.set_state(self._Gst.State.NULL)
        except Exception:
            pass
