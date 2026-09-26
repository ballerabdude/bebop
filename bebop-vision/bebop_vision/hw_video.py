"""Hardware video encoding (Jetson NVENC) via GStreamer `nvv4l2*enc`.

The Orin Nano had no NVENC, which is why the operator stream and the navd
recordings used software JPEG/MJPEG (see docs/navd.md §3.2). The Thor
(JetPack 7) ships a real NVENC block: `nvv4l2h264enc` / `nvv4l2h265enc`
are present and hardware-encode 1280x800 in a few ms. This module wraps a
persistent GStreamer pipeline (`appsrc` raw BGR -> `nvvidconv` -> NVMM
NV12 -> NVENC -> `appsink` Annex-B access units) so the recorder can store
H.265/H.264 instead of per-frame JPEG.

Why GStreamer and not PyAV: FFmpeg's `h264_v4l2m2m` / `hevc_v4l2m2m`
encoders need a V4L2 mem2mem node, and the Tegra encoder is not exposed
that way (it is `/dev/nvhost-msenc`, reachable only through the
`nvv4l2*` GStreamer elements). PyAV *can* decode the resulting H.265 in
software, so the training extractor stays on the existing `av` dependency.

The system GStreamer python bindings (`gi`) live in
`/usr/lib/python3/dist-packages`, which the venv does not include; we add
that path lazily on first use. If anything is missing (non-Jetson dev box,
no gi, no nvv4l2 element) the caller gets `HwEncodeUnavailable` and falls
back to JPEG.
"""

from __future__ import annotations

import collections
import sys
import threading
import time

import numpy as np

# System GStreamer bindings; appended (not prepended) so venv packages win.
_SYSTEM_GI = "/usr/lib/python3/dist-packages"

_gst_lock = threading.Lock()
_gst = None  # (Gst, GstApp)

# codec -> (encoder element, bitstream parser element)
_ENCODERS = {
    "h264": ("nvv4l2h264enc", "h264parse"),
    "h265": ("nvv4l2h265enc", "h265parse"),
}


def _nv12_caps(out_width, out_height):
    if out_width and out_height:
        return (f"video/x-raw(memory:NVMM),format=NV12,width={out_width},"
                f"height={out_height} ! ")
    return "video/x-raw(memory:NVMM),format=NV12 ! "


def _appsrc_prefix(kind, width, height, fps, out_width, out_height):
    """`appsrc` + pre-encode chain for either raw BGR or camera MJPEG.

    `kind="jpeg"` is the zero-CPU path the cameras actually use: the
    Orbbec already hands us hardware-encoded MJPEG, so `nvjpegdec` decodes
    it on the NVJPG block and `nvvidconv` scales/colour-converts on the
    GPU — Python never touches the pixels. `kind="bgr"` is the fallback
    for raw-RGB configs (software `videoconvert` to I420).
    """
    scale = _nv12_caps(out_width, out_height)
    if kind == "jpeg":
        return (
            "appsrc name=src is-live=true format=time block=true "
            "max-buffers=4 leaky-type=downstream "
            f"caps=image/jpeg,framerate={max(1, int(round(fps)))}/1 ! "
            "nvjpegdec ! nvvidconv ! " + scale
        )
    return (
        "appsrc name=src is-live=true format=time block=true "
        "max-buffers=2 leaky-type=downstream "
        f"caps=video/x-raw,format=BGR,width={width},height={height},"
        f"framerate={max(1, int(round(fps)))}/1 ! "
        "videoconvert ! video/x-raw,format=I420 ! nvvidconv ! " + scale
    )


class HwEncodeUnavailable(RuntimeError):
    """Raised when no hardware encoder can be constructed on this host."""


def _load_gst():
    """Import GStreamer bindings once; raise HwEncodeUnavailable if absent."""
    global _gst
    with _gst_lock:
        if _gst is not None:
            return _gst
        if _SYSTEM_GI not in sys.path:
            sys.path.append(_SYSTEM_GI)
        try:
            import gi
            gi.require_version("Gst", "1.0")
            gi.require_version("GstApp", "1.0")
            from gi.repository import Gst, GstApp
        except Exception as exc:  # pragma: no cover - host dependent
            raise HwEncodeUnavailable(
                f"GStreamer python bindings unavailable: {exc}") from exc
        if not Gst.is_initialized():
            Gst.init(None)
        _gst = (Gst, GstApp)
        return _gst


def encoder_available(codec: str = "h265") -> bool:
    """True when the NVDEC/NVENC element for `codec` exists on this host."""
    if codec not in _ENCODERS:
        return False
    try:
        Gst, _ = _load_gst()
    except HwEncodeUnavailable:
        return False
    return bool(Gst.ElementFactory.find(_ENCODERS[codec][0]))


class HwVideoEncoder:
    """One persistent hardware encoder for one camera stream.

    Frames are pushed in display order with an opaque per-frame tag
    (we use the recorder tick's log_ns). Access units are returned in the
    same order, each paired with the tag of the frame it belongs to. The
    encoder introduces ~1 frame of latency, so `push()` may return fewer
    AUs than frames pushed; the FIFO pairs every AU with the oldest
    un-emitted tag, which keeps tick alignment exact. `flush()` drains the
    tail at end of session.

    Not thread-safe: construct and use from a single thread (the recorder
    loop), like the MCAP writer.
    """

    def __init__(self, width, height, fps=10.0, codec="h265",
                 bitrate=2_000_000, iframe_interval=None, input_kind="bgr"):
        if codec not in _ENCODERS:
            raise HwEncodeUnavailable(f"unsupported codec {codec!r}")
        Gst, _ = _load_gst()
        enc, parse = _ENCODERS[codec]
        if not Gst.ElementFactory.find(enc):
            raise HwEncodeUnavailable(f"{enc} not present")
        self.width = int(width)
        self.height = int(height)
        self.input_kind = input_kind
        # JPEG input carries its own dimensions; pass through unscaled.
        if input_kind == "jpeg":
            ow = oh = None
        else:
            ow = self.width - (self.width % 2)
            oh = self.height - (self.height % 2)
        self.fps = float(fps)
        self.codec = codec
        self._dur_ns = int(round(Gst.SECOND / self.fps)) if self.fps else 0
        gop = int(iframe_interval if iframe_interval is not None
                  else max(1, round(self.fps)))
        desc = (
            _appsrc_prefix(input_kind, self.width, self.height, self.fps, ow, oh)
            + f"{enc} bitrate={int(bitrate)} control-rate=constant_bitrate "
            f"iframeinterval={gop} ! "
            f"{parse} ! appsink name=sink sync=false emit-signals=false "
            "max-buffers=8"
        )
        self._pipeline = Gst.parse_launch(desc)
        self._src = self._pipeline.get_by_name("src")
        self._sink = self._pipeline.get_by_name("sink")
        self._pending = collections.deque()
        self._pts = 0
        self._Gst = Gst
        self._pipeline.set_state(Gst.State.PLAYING)

    def push(self, data, tag=None):
        """Push one frame; return [(au_bytes, tag), ...] now available.

        `data` is camera MJPEG bytes when `input_kind == "jpeg"`, else a
        BGR ndarray.
        """
        Gst = self._Gst
        if self.input_kind == "jpeg":
            blob = data if isinstance(data, (bytes, bytearray)) else bytes(data)
            nbytes = len(blob)
        else:
            arr = np.ascontiguousarray(data)
            blob = arr.tobytes()
            nbytes = arr.nbytes
        buf = Gst.Buffer.new_allocate(None, nbytes, None)
        buf.fill(0, blob)
        buf.pts = self._pts
        buf.duration = self._dur_ns
        self._pts += self._dur_ns
        self._pending.append(tag)
        self._src.emit("push-buffer", buf)
        return self._drain()

    def _drain(self):
        out = []
        while True:
            s = self._sink.try_pull_sample(0)
            if s is None:
                break
            b = s.get_buffer()
            tag = self._pending.popleft() if self._pending else None
            out.append((bytes(b.extract_dup(0, b.get_size())), tag))
        return out

    def flush(self):
        """End-of-stream: return the remaining [(au_bytes, tag), ...]."""
        Gst = self._Gst
        self._src.emit("end-of-stream")
        out = []
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            out.extend(self._drain())
            if self._pipeline is None:
                break
            bus = self._pipeline.get_bus()
            if bus is not None and bus.have_pending():
                msg = bus.pop()
                if msg is not None and msg.type == Gst.MessageType.EOS:
                    break
            time.sleep(0.01)
        out.extend(self._drain())
        return out

    def close(self):
        try:
            self._pipeline.set_state(self._Gst.State.NULL)
        except Exception:
            pass
        self._pipeline = None
