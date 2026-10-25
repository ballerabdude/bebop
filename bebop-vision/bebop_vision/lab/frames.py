"""Random access and sequential decode of a session's colour and depth.

The recorder writes colour as one H.265 (or H.264) access unit per tick,
or one JPEG per tick on the no-encoder path, and depth as one 16-bit PNG
per tick. The session index concatenates each role's payloads into a
``.bin`` next to a structured ``.npy`` index, and these classes read that
pair through a memory map, so a 20-minute drive is never loaded into RAM.

Video random access decodes from the previous keyframe forward: the
recorder's IDR interval is short (the NVENC GOP) and PyAV's software
decoder does a 1280x800 HEVC frame in a few milliseconds, so a seek costs
at most one GOP of decode. Sequential access keeps the decoder open and
feeds the next AU, which is what a training pass or a viewer does.

Decoded frames are paired with access units by presentation stamp: each
packet is tagged ``pts = <index row>`` and the decoder hands the stamp
back on the frame (verified with PyAV 12 / libavcodec 60 on raw HEVC and
H.264), so an AU that yields no frame (the slices before the first IRAP of
a segment raise, a damaged AU raises under ``err_detect=explode``) cannot
shift later frames onto the wrong tick, which pairing by emission order
did. A frame whose reference is missing is a different failure: ffmpeg
conceals it silently (the missing reference is a flat grey picture, no
corrupt flag is set), so every row between a hole or a failed AU and the
next keyframe is marked *tainted* and returned as None instead of as a
grey smear. Holes come from the session's tick grid (a tick with depth but
no colour AU); failed AUs are learnt while decoding.

``av`` and ``cv2`` are imported inside the functions that need them so the
package imports without either.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence, Tuple

import numpy as np

from bebop_vision.lab.types import file_identity

#: Bump when the materialized layout changes; readers rebuild on mismatch.
PRODUCER = "frames@1"

#: Structured index of a colour ``.bin``: one row per access unit / JPEG.
VIDEO_INDEX_DTYPE = np.dtype([("t_ns", "i8"), ("offset", "i8"), ("size", "i8"),
                              ("keyframe", "?"), ("fmt", "U8")])
#: Structured index of a depth ``.bin``: one row per PNG.
DEPTH_INDEX_DTYPE = np.dtype([("t_ns", "i8"), ("offset", "i8"), ("size", "i8")])

#: Foxglove/FFmpeg naming: the recorder tags the bitstream ``h265`` but
#: PyAV's decoder is ``hevc``.
_DECODER_FOR = {"h265": "hevc", "hevc": "hevc", "h264": "h264"}

#: How many AUs past the requested one the decoder may be fed before the
#: stream is flushed to force the frame out. One AU per tick with no
#: B-frames and slice threading means a decoder normally emits a frame per
#: AU; the allowance covers a decoder holding a frame without flushing (a
#: flush ends the stream and forces a reopen).
_LOOKAHEAD_AUS = 8


def is_keyframe(au: bytes, fmt: str) -> bool:
    """Whether an Annex-B access unit starts a decodable segment.

    HEVC: the first VCL NAL has type 16..23 (IRAP: BLA/IDR/CRA). H.264: the
    first VCL NAL is type 5 (IDR). JPEG: every frame. Parameter sets (HEVC
    32..34, H.264 7/8) and SEI are skipped; a payload with no start code
    (length-prefixed hvcC) is reported as not a keyframe, since nothing in
    the recorder emits one.
    """
    if fmt == "jpeg":
        return True
    pos = 0
    n = len(au)
    while True:
        p = au.find(b"\x00\x00\x01", pos)
        if p < 0 or p + 3 >= n:
            return False
        hdr = au[p + 3]
        pos = p + 3
        if fmt in ("h265", "hevc"):
            nal = (hdr >> 1) & 0x3F
            if nal >= 32:        # VPS/SPS/PPS/AUD/SEI: not a slice
                continue
            return 16 <= nal <= 23
        nal = hdr & 0x1F
        if nal in (6, 7, 8, 9) or nal >= 10:  # SEI, SPS, PPS, AUD, ...
            continue
        return nal == 5


def _memmap(path: Path) -> Optional[np.memmap]:
    """Read-only byte view of ``path``; None for an empty file (``np.memmap``
    refuses a zero-length mapping)."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return None
    return np.memmap(path, dtype=np.uint8, mode="r")


class _BinSeries:
    """Shared ``t_ns`` lookup over a ``.bin`` + structured index pair."""

    def __init__(self, bin_path: Path, index_path: Path, dtype: np.dtype, name: str = ""):
        self.bin_path = Path(bin_path)
        self.index_path = Path(index_path)
        self.name = name
        if self.index_path.exists():
            idx = np.load(self.index_path)
            if idx.dtype != dtype:
                idx = idx.astype(dtype)
        else:
            idx = np.empty(0, dtype=dtype)
        order = np.argsort(idx["t_ns"], kind="stable")
        self.index = idx[order] if not np.all(order == np.arange(len(idx))) else idx
        self._mm = _memmap(self.bin_path)

    def __len__(self) -> int:
        return int(len(self.index))

    @property
    def t_ns(self) -> np.ndarray:
        return self.index["t_ns"]

    def position(self, t_ns: int) -> Optional[int]:
        """Row with exactly this ``t_ns``, else None."""
        i = int(np.searchsorted(self.t_ns, t_ns))
        if i < len(self.index) and int(self.t_ns[i]) == int(t_ns):
            return i
        return None

    def nearest_t(self, t_ns: int, tol_ns: Optional[int] = None) -> Optional[int]:
        """The stamp nearest ``t_ns`` (within ``tol_ns`` if given), else None."""
        if len(self.index) == 0:
            return None
        i = int(np.searchsorted(self.t_ns, t_ns))
        best, best_dt = None, None
        for j in (i - 1, i):
            if 0 <= j < len(self.index):
                dt = abs(int(self.t_ns[j]) - int(t_ns))
                if best_dt is None or dt < best_dt:
                    best, best_dt = int(self.t_ns[j]), dt
        if best is None or (tol_ns is not None and best_dt > tol_ns):
            return None
        return best

    def payload(self, i: int) -> bytes:
        row = self.index[i]
        if self._mm is None:
            raise ValueError(f"{self.name}: empty payload store {self.bin_path}")
        off, size = int(row["offset"]), int(row["size"])
        return bytes(self._mm[off:off + size])


class FrameSeries(_BinSeries):
    """One role's colour stream: BGR uint8 frames by tick ``t_ns``.

    ``at(t_ns)`` returns the frame recorded at exactly that stamp (per-tick
    channels share the tick's log_time) or None when the recorder wrote no
    colour for that tick (how a dropped far frame shows up) or when the AU
    cannot be decoded faithfully (``tainted``: see the module docstring).

    ``ticks`` is the session's raw tick grid; with it, a tick that has no
    AU marks the rest of its GOP tainted, since the following P-frames
    reference a picture the stream never carried. Without it only decode
    failures taint.
    """

    def __init__(self, bin_path: Path, index_path: Path, name: str = "", cache_frames: int = 64,
                 ticks: Optional[np.ndarray] = None):
        super().__init__(bin_path, index_path, VIDEO_INDEX_DTYPE, name)
        self.cache_frames = int(cache_frames)
        self._lru: "OrderedDict[int, np.ndarray]" = OrderedDict()
        self._cc = None            # open PyAV decoder, or None
        self._fed = -1             # index of the last AU fed to self._cc
        self._fmt = str(self.index["fmt"][0]) if len(self.index) else ""
        kf = np.flatnonzero(self.index["keyframe"]) if len(self.index) else np.empty(0, np.int64)
        self._kf_pos = kf.astype(np.int64)
        self._tainted = self._initial_taint(ticks)
        self.decode_calls = 0      # AUs fed, for tests and `lab info`

    # -- public --------------------------------------------------------------
    @property
    def fmt(self) -> str:
        """``"h265" | "h264" | "jpeg"`` (empty for an empty series)."""
        return self._fmt

    @property
    def keyframes(self) -> np.ndarray:
        """Stamps of the keyframes (every frame for JPEG)."""
        return self.t_ns[self._kf_pos]

    @property
    def tainted(self) -> np.ndarray:
        """Per index row: True where ``at`` returns None although an AU was
        recorded (a hole or a failed AU earlier in the same GOP, or no
        keyframe before the row). Grows as decode failures are met."""
        return self._tainted

    def at(self, t_ns: int) -> Optional[np.ndarray]:
        i = self.position(int(t_ns))
        if i is None:
            return None
        return self.frame_at_index(i)

    def iter(self, t_ns_list: Iterable[int]) -> Iterator[Tuple[int, Optional[np.ndarray]]]:
        """``(t_ns, frame)`` for each stamp, decoding sequentially."""
        for t in t_ns_list:
            yield int(t), self.at(int(t))

    def frame_at_index(self, i: int) -> Optional[np.ndarray]:
        """Row ``i`` of the index decoded to BGR uint8."""
        if i < 0 or i >= len(self.index):
            return None
        hit = self._lru.get(i)
        if hit is not None:
            self._lru.move_to_end(i)
            return hit
        if self._tainted[i]:
            return None
        fmt = str(self.index["fmt"][i])
        if fmt == "jpeg":
            frame = self._decode_jpeg(self.payload(i))
        else:
            frame = self._decode_video(i)
        if frame is not None:
            self._remember(i, frame)
        return frame

    def close(self) -> None:
        self._cc = None
        self._fed = -1

    # -- internals -----------------------------------------------------------
    def _initial_taint(self, ticks: Optional[np.ndarray]) -> np.ndarray:
        n = len(self.index)
        tainted = np.zeros(n, dtype=bool)
        if n == 0 or self._fmt == "jpeg" or len(self._kf_pos) == 0:
            # JPEG rows are independent; a stream with no detected IRAP at
            # all is decoded from its start (unknown framing, not a hole).
            return tainted
        first_kf = int(self._kf_pos[0])
        tainted[:first_kf] = True      # P-frames with no IRAP to start from
        if ticks is not None and len(ticks):
            t = self.t_ns
            holes = np.asarray(ticks, dtype=np.int64)
            holes = holes[(holes > int(t[0])) & (holes < int(t[-1]))]
            holes = holes[~np.isin(holes, t)]
            for h in np.unique(holes):
                row = int(np.searchsorted(t, h))
                # a hole right before a keyframe costs nothing: the IRAP
                # does not reference the missing picture
                if not self.index["keyframe"][row]:
                    self._taint_from(row, tainted)
        return tainted

    def _taint_from(self, row: int, tainted: Optional[np.ndarray] = None) -> None:
        """Mark ``row`` up to (excluding) the next keyframe as tainted."""
        tainted = self._tainted if tainted is None else tainted
        if row >= len(tainted):
            return
        k = int(np.searchsorted(self._kf_pos, row, side="right"))
        end = int(self._kf_pos[k]) if k < len(self._kf_pos) else len(tainted)
        tainted[row:end] = True

    def _remember(self, i: int, frame: np.ndarray) -> None:
        self._lru[i] = frame
        self._lru.move_to_end(i)
        while len(self._lru) > self.cache_frames:
            self._lru.popitem(last=False)

    @staticmethod
    def _decode_jpeg(data: bytes) -> Optional[np.ndarray]:
        import cv2
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        return img

    def _keyframe_before(self, i: int) -> int:
        """Index of the last keyframe at or before ``i`` (0 when none is
        marked, so a stream without detected IRAPs still decodes from its
        start)."""
        k = int(np.searchsorted(self._kf_pos, i, side="right")) - 1
        return int(self._kf_pos[k]) if k >= 0 else 0

    def _open(self, start: int) -> None:
        import av
        cc = av.CodecContext.create(_DECODER_FOR.get(self._fmt, self._fmt), "r")
        # Slice threading keeps the decoder delay at zero whatever PyAV's
        # default thread count is (frame threading holds up to N frames
        # back, which would exceed _LOOKAHEAD_AUS and force a flush+reopen
        # per frame). A damaged AU must raise rather than be concealed into
        # a grey picture, hence explode; a missing reference is not an
        # error to ffmpeg and is handled through `tainted` instead.
        cc.thread_type = "SLICE"
        cc.options = {"err_detect": "explode"}
        self._cc = cc
        self._fed = start - 1

    def _feed(self, i: Optional[int]) -> bool:
        """Feed AU ``i`` (None flushes) and cache whatever comes out, paired
        by ``pts``. Returns False when the AU failed to decode (the rest of
        its GOP is then tainted)."""
        import av
        if i is None:
            pkt = None
        else:
            pkt = av.Packet(self.payload(i))
            pkt.pts = i
            self._fed = i
            self.decode_calls += 1
        try:
            out = self._cc.decode(pkt)
        except av.error.FFmpegError:
            if i is not None:
                self._taint_from(i)
            return False
        for fr in out:
            # A decoder that drops the stamp (none seen) falls back to the
            # one-frame-per-AU assumption: the frame is the AU just fed.
            j = int(fr.pts) if fr.pts is not None else self._fed
            if 0 <= j < len(self.index) and not self._tainted[j]:
                self._remember(j, fr.to_ndarray(format="bgr24"))
        return True

    def _decode_video(self, i: int) -> Optional[np.ndarray]:
        kf = self._keyframe_before(i)
        # Reuse the open decoder only when the target lies ahead of it and
        # continuing costs no more AUs than restarting at the keyframe.
        reuse = (self._cc is not None and self._fed < i and (i - self._fed) <= (i - kf + 1))
        if not reuse:
            self._open(kf)
        while self._fed < i and not self._tainted[i]:
            self._feed(self._fed + 1)
        if self._tainted[i]:
            return None
        extra = 0
        while i not in self._lru and self._fed + 1 < len(self.index) and extra < _LOOKAHEAD_AUS:
            self._feed(self._fed + 1)
            extra += 1
        if i not in self._lru:
            # End of stream (or the decoder is holding frames): flush so the
            # tail comes out, then drop the decoder, since a flushed context
            # accepts no more packets.
            self._feed(None)
            self.close()
        frame = self._lru.get(i)
        if frame is not None:
            self._lru.move_to_end(i)
        return frame


class DepthSeries(_BinSeries):
    """One role's depth stream: uint16 millimetre images by tick ``t_ns``."""

    def __init__(self, bin_path: Path, index_path: Path, name: str = ""):
        super().__init__(bin_path, index_path, DEPTH_INDEX_DTYPE, name)

    def at(self, t_ns: int) -> Optional[np.ndarray]:
        i = self.position(int(t_ns))
        if i is None:
            return None
        return self.depth_at_index(i)

    def depth_at_index(self, i: int) -> Optional[np.ndarray]:
        import cv2
        if i < 0 or i >= len(self.index):
            return None
        img = cv2.imdecode(np.frombuffer(self.payload(i), np.uint8), cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        if img.ndim != 2:
            img = img[..., 0]
        if img.dtype != np.uint16:
            img = img.astype(np.uint16)
        return img

    def iter(self, t_ns_list: Iterable[int]) -> Iterator[Tuple[int, Optional[np.ndarray]]]:
        for t in t_ns_list:
            yield int(t), self.at(int(t))


def materialize_key(roles: Sequence[str], t_ns_list: Sequence[int],
                    size: Optional[Tuple[int, int]] = None) -> str:
    """Short stable key of a materialization request (roles, size and the
    exact tick list), so different requests land in different directories
    and a reader can tell which one it is looking at."""
    t = np.ascontiguousarray(np.asarray(list(t_ns_list), dtype=np.int64))
    h = hashlib.sha1()
    h.update(",".join(roles).encode())
    h.update(b"|" + (f"{int(size[0])}x{int(size[1])}" if size else "native").encode())
    h.update(b"|" + t.tobytes())
    return h.hexdigest()[:12]


def materialize(session, roles: Sequence[str], t_ns_list: Sequence[int],
                size: Optional[Tuple[int, int]] = None, out_dir: Optional[Path] = None,
                force: bool = False) -> Path:
    """Decode the colour frames of ``roles`` at ``t_ns_list`` into uint8
    memmaps ``frames_<role>.npy [N, H, W, 3]`` plus ``t_ns.npy`` and
    ``valid_<role>.npy`` (False where the tick had no decodable frame;
    those rows are zero). ``size`` is ``(W, H)`` to resize with cv2
    (INTER_AREA). Returns the directory. A training pass reads this instead
    of decoding video.

    The default directory is ``<cache>/materialized/<key>`` with
    :func:`materialize_key`; ``meta.json`` carries the MCAP identity, this
    module's producer and the key, and a directory whose meta matches is
    returned as is unless ``force``.
    """
    import cv2
    roles = tuple(roles)
    t_ns = np.asarray(list(t_ns_list), dtype=np.int64)
    key = materialize_key(roles, t_ns, size)
    out = Path(out_dir) if out_dir is not None else Path(session.index()) / "materialized" / key
    identity = file_identity(session.path)
    want = {"source": str(session.path), "identity": identity, "producer": PRODUCER, "key": key}
    meta_path = out / "meta.json"
    if not force and meta_path.exists():
        try:
            have = json.loads(meta_path.read_text())
        except (OSError, ValueError):
            have = {}
        if all(have.get(k) == v for k, v in want.items()) and \
                all((out / f"frames_{r}.npy").exists() and (out / f"valid_{r}.npy").exists() for r in roles):
            return out
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "t_ns.npy", t_ns)
    meta = dict(want, roles=list(roles), n=int(len(t_ns)), size=list(size) if size else None,
                built_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
    for role in roles:
        fs = session.frames(role)
        shape = None
        mm = None
        valid = np.zeros(len(t_ns), dtype=bool)
        for k, (t, frame) in enumerate(fs.iter(t_ns)):
            if frame is None:
                continue
            if size is not None and (frame.shape[1], frame.shape[0]) != tuple(size):
                frame = cv2.resize(frame, tuple(size), interpolation=cv2.INTER_AREA)
            if mm is None:
                shape = (len(t_ns),) + tuple(frame.shape)
                mm = np.lib.format.open_memmap(out / f"frames_{role}.npy", mode="w+",
                                               dtype=np.uint8, shape=shape)
            mm[k] = frame
            valid[k] = True
        if mm is None:
            # No frame at all: still leave a well-formed (empty) array.
            h, w = (size[1], size[0]) if size else (0, 0)
            mm = np.lib.format.open_memmap(out / f"frames_{role}.npy", mode="w+",
                                           dtype=np.uint8, shape=(len(t_ns), h, w, 3))
            shape = mm.shape
        mm.flush()
        del mm
        np.save(out / f"valid_{role}.npy", valid)
        meta[f"shape_{role}"] = list(shape)
        meta[f"valid_{role}"] = int(valid.sum())
    # meta.json is written last, so a crash mid-build leaves no "valid" dir
    meta_path.write_text(json.dumps(meta, indent=2))
    return out
