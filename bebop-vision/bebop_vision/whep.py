"""WHEP (WebRTC-HTTP Egress Protocol) operator stream.

One NVENC H.264 pipeline + `webrtcbin` per client. The browser POSTs an
SDP offer to `/whep`; we answer and then push frames. Media flows over
SRTP/UDP, so latency is ~150 ms and packet loss doesn't head-of-line-block
the way the old TCP fMP4/MJPEG paths did.

Every operator stream rides this: color is the camera's hardware JPEG
(`input_kind="jpeg"`), depth is rendered server-side to BGR
(`input_kind="bgr"`).

Requires the GStreamer `nice` plugin (`gstreamer1.0-nice`); without it
`webrtcbin` refuses to leave NULL ("libnice elements are not available").

H.264 only: WebRTC H.265 is Safari-only. The encoder is the same NVENC
chain the recorder uses.
"""

from __future__ import annotations

import sys
import threading
import time

_SYSTEM_GI = "/usr/lib/python3/dist-packages"
_gst = None
_gst_lock = threading.Lock()


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


def webrtc_available() -> bool:
    """True when a webrtcbin send pipeline can be constructed here."""
    try:
        Gst, *_ = _load()
    except Exception:
        return False
    for el in ("webrtcbin", "nicesrc", "nicesink", "nvv4l2h264enc",
               "rtph264pay"):
        if not Gst.ElementFactory.find(el):
            return False
    return True


class WhepSession:
    """One client: NVENC H.264 -> RTP -> webrtcbin.

    `input_kind="jpeg"` takes camera MJPEG bytes; `input_kind="bgr"` takes
    a BGR ndarray of `width`x`height` (server-rendered depth/BEV).
    Not thread-safe for concurrent `push`; the videoserver drives it from
    a single feeder thread.
    """

    def __init__(self, input_kind="jpeg", width=0, height=0, fps=15.0,
                 bitrate=4_000_000, latency_ms=0):
        Gst, GstApp, GstWebRTC, GstSdp = _load()
        self._Gst = Gst
        self._GstWebRTC = GstWebRTC
        self._GstSdp = GstSdp
        self._input_kind = input_kind
        self._closed = False
        self._fps = float(fps) if fps else 15.0
        self._dur_ns = int(round(Gst.SECOND / self._fps))
        self._pts = 0

        pipeline = Gst.Pipeline.new("whep")

        def mk(factory, name):
            el = Gst.ElementFactory.make(factory, name)
            if el is None:
                raise RuntimeError(f"missing GStreamer element {factory}")
            pipeline.add(el)
            return el

        self._src = mk("appsrc", "src")
        self._src.set_property("is-live", True)
        self._src.set_property("format", Gst.Format.TIME)
        self._src.set_property("block", True)
        self._src.set_property("max-buffers", 4)
        self._src.set_property("leaky-type", GstApp.AppLeakyType.DOWNSTREAM)
        rate = max(1, int(round(self._fps)))

        chain = []
        if input_kind == "jpeg":
            self._src.set_property(
                "caps", Gst.Caps.from_string(f"image/jpeg,framerate={rate}/1"))
            chain.append(mk("nvjpegdec", "dec"))
        else:
            self._src.set_property(
                "caps",
                Gst.Caps.from_string(
                    f"video/x-raw,format=BGR,width={int(width)},"
                    f"height={int(height)},framerate={rate}/1"))
            chain.append(mk("videoconvert", "vc"))
            cf_in = mk("capsfilter", "cf_in")
            cf_in.set_property(
                "caps", Gst.Caps.from_string("video/x-raw,format=I420"))
            chain.append(cf_in)

        conv = mk("nvvidconv", "conv")
        cf = mk("capsfilter", "cf")
        cf.set_property(
            "caps",
            Gst.Caps.from_string("video/x-raw(memory:NVMM),format=NV12"))
        enc = mk("nvv4l2h264enc", "enc")
        enc.set_property("bitrate", int(bitrate))
        enc.set_property("control-rate", 1)  # constant_bitrate
        enc.set_property("iframeinterval", max(1, int(round(self._fps))))
        parse = mk("h264parse", "parse")
        parse.set_property("config-interval", -1)
        self._pay = mk("rtph264pay", "pay")
        self._pay.set_property("config-interval", -1)
        self._pay.set_property("pt", 96)
        self._webrtc = mk("webrtcbin", "sendrecv")
        self._webrtc.set_property(
            "bundle-policy", GstWebRTC.WebRTCBundlePolicy.MAX_BUNDLE)
        self._webrtc.set_property("latency", int(latency_ms))

        prev = self._src
        for el in chain + [conv, cf, enc, parse, self._pay]:
            # Element.link returns a Python bool; Pad.link returns the enum.
            if not prev.link(el):
                raise RuntimeError(f"failed to link {prev.name} -> {el.name}")
            prev = el

        self._pipeline = pipeline
        ret = pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("webrtc pipeline failed to start")
        # Wait for the pipeline to settle so webrtcbin is open.
        pipeline.get_state(Gst.CLOCK_TIME_NONE)
        self._pad_linked = False

    # ------------------------------------------------------------ signaling
    def negotiate(self, offer_sdp: str, timeout_s: float = 5.0) -> str:
        """Apply a browser offer, return the answer SDP (non-trickle)."""
        Gst = self._Gst
        GstSdp = self._GstSdp
        GstWebRTC = self._GstWebRTC

        res, sdp = GstSdp.SDPMessage.new()
        GstSdp.sdp_message_parse_buffer(offer_sdp.encode(), sdp)
        # GStreamer 1.20+ signals take a WebRTCSessionDescription, not a
        # bare SDPMessage.
        offer = GstWebRTC.WebRTCSessionDescription.new(
            GstWebRTC.WebRTCSDPType.OFFER, sdp)
        p = Gst.Promise.new()
        self._webrtc.emit("set-remote-description", offer, p)
        p.wait()

        # Send H.264 on the payload type the offer/answer negotiated for it
        # (not necessarily 96 — Chrome uses 96 for VP8).
        pt = _h264_payload_type(offer_sdp)
        if pt is not None:
            self._pay.set_property("pt", pt)

        # The offer's m-line created a transceiver; now its sink pad
        # exists, so link the encoder chain to it.
        pad = self._webrtc.request_pad_simple("sink_%u")
        if pad is None:
            pad = self._webrtc.get_static_pad("sink_0")
        if pad is None:
            raise RuntimeError("webrtcbin produced no sink pad")
        # Pad.link returns a Gst.PadLinkReturn enum whose OK is 0 (falsy),
        # so compare explicitly rather than testing truthiness.
        if self._pay.get_static_pad("src").link(pad) != Gst.PadLinkReturn.OK:
            raise RuntimeError("failed to link encoder to webrtcbin")
        self._pad_linked = True

        p = Gst.Promise.new()
        self._webrtc.emit("create-answer", None, p)
        p.wait()
        reply = p.get_reply()
        answer = reply.get_value("answer")
        if answer is None:
            err = reply.get_value("error")
            raise RuntimeError(
                f"create-answer failed: {err.message if err else 'unknown'}")

        p = Gst.Promise.new()
        self._webrtc.emit("set-local-description", answer, p)
        p.wait()

        # Non-trickle: wait for host candidates to be gathered so the
        # answer SDP is complete for the browser.
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if (self._webrtc.get_property("ice-gathering-state")
                    == GstWebRTC.WebRTCICEGatheringState.COMPLETE):
                break
            time.sleep(0.02)
        local = self._webrtc.get_property("local-description")
        return (local or answer).sdp.as_text()

    # --------------------------------------------------------------- media
    def push(self, data) -> None:
        """Push one frame: MJPEG bytes (jpeg) or a BGR ndarray (bgr)."""
        if self._closed or not self._pad_linked:
            return
        Gst = self._Gst
        if self._input_kind == "jpeg":
            blob = data if isinstance(data, (bytes, bytearray)) else bytes(data)
            nbytes = len(blob)
        else:
            import numpy as np

            arr = np.ascontiguousarray(data)
            blob = arr.tobytes()
            nbytes = arr.nbytes
        buf = Gst.Buffer.new_allocate(None, nbytes, None)
        buf.fill(0, blob)
        buf.pts = self._pts
        buf.duration = self._dur_ns
        self._pts += self._dur_ns
        self._src.emit("push-buffer", buf)

    def connection_state(self):
        try:
            return self._webrtc.get_property("connection-state")
        except Exception:
            return None

    def failed(self) -> bool:
        GstWebRTC = self._GstWebRTC
        state = self.connection_state()
        return state in (
            GstWebRTC.WebRTCPeerConnectionState.FAILED,
            GstWebRTC.WebRTCPeerConnectionState.CLOSED,
            GstWebRTC.WebRTCPeerConnectionState.DISCONNECTED,
        )

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._src.emit("end-of-stream")
        except Exception:
            pass
        try:
            self._pipeline.set_state(self._Gst.State.NULL)
        except Exception:
            pass
        self._pipeline = None
