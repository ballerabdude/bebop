"""navd recorder v2: teleop sessions -> one MCAP file per session (plan §7.1).

MCAP is the single data artifact: copied off the robot (scp) it carries
everything the workstation needs — color, both depths, the operator's
teleop twist (the imitation label), odometry, and calibration. Indexed
and seekable, opens in Foxglove for review, and `tools/mcap_extract.py`
unpacks it into the `datasets/navd-v0/` training layout.

Channels:
  /color_near   foxglove.CompressedVideo (H.265 access units, hardware
                NVENC; falls back to CompressedImage JPEG when no encoder)
  /color_far    same encoding (both cameras stream MJPEG, re-encoded to
                H.265 for storage; the camera's own JPEG is only kept on
                the fallback path)
  /depth_near   raw PNG bytes (schemaless; uint16 mm, lossless — training data)
  /depth_far    raw PNG bytes (schemaless)
  /depth_near_preview  foxglove.RawImage (JSON; 106x60 16uc1 — dashboard only)
  /depth_far_preview   foxglove.RawImage (same encoding, far camera)
  /cmd_vel      JSON  {"vx", "wz", "stamp_ns"}   — operator twist (teleop label)
  /odom         JSON  {"x", "y", "theta", "stamp_ns"}
  /calib        JSON  intrinsics + rig extrinsics, written once at start

Training channels are the raw PNGs (tools/mcap_extract.py);
the Foxglove-schema channels exist so sessions open as a live dashboard in
Foxglove Studio (foxglove/bebop_navd_layout.json).

All messages of one tick share the same log_time (µs since epoch), so the
extractor can group them by exact match. Image messages additionally carry
`stamp_us` (SDK device stamp — per-camera clock domain, NOT comparable
across cameras) and `pair_ms` (arrival-time residual of this camera's frame
vs the paired near frame; hardware sync keeps it ~0-2 ms) so cross-camera
alignment is verifiable offline.
"""

import base64
import math
import json
import threading
import time

import numpy as np

try:
    from mcap.writer import Writer, CompressionType
except ImportError as exc:  # pragma: no cover
    raise ImportError("pip install mcap") from exc

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise ImportError("pip install opencv-python") from exc



# Official Foxglove JSON Schemas (foxglove/foxglove-sdk schemas/jsonschema/) — the recorder registers these verbatim so
# Foxglove recognizes the well-known message types in JSON-encoded MCAP.
_FOXGLOVE_COMPRESSED_IMAGE_SCHEMA = json.dumps({
  "title": "foxglove.CompressedImage",
  "description": "A compressed image",
  "type": "object",
  "properties": {
    "timestamp": {
      "type": "object",
      "title": "time",
      "properties": {
        "sec": {
          "type": "integer",
          "minimum": 0
        },
        "nsec": {
          "type": "integer",
          "minimum": 0,
          "maximum": 999999999
        }
      },
      "description": "Timestamp of image"
    },
    "frame_id": {
      "type": "string",
      "description": "Frame of reference for the image."
    },
    "data": {
      "type": "string",
      "contentEncoding": "base64",
      "description": "Compressed image data"
    },
    "format": {
      "type": "string",
      "description": "Image format. Supported: jpeg, png, webp, avif"
    }
  },
  "required": [
    "timestamp",
    "frame_id",
    "data",
    "format"
  ]
})
_FOXGLOVE_COMPRESSED_VIDEO_SCHEMA = json.dumps({
  "title": "foxglove.CompressedVideo",
  "description": "A compressed video frame",
  "type": "object",
  "properties": {
    "timestamp": {
      "type": "object",
      "title": "time",
      "properties": {
        "sec": {
          "type": "integer",
          "minimum": 0
        },
        "nsec": {
          "type": "integer",
          "minimum": 0,
          "maximum": 999999999
        }
      },
      "description": "Timestamp of video frame"
    },
    "frame_id": {
      "type": "string",
      "description": "Frame of reference for the image."
    },
    "data": {
      "type": "string",
      "contentEncoding": "base64",
      "description": "Compressed video frame data"
    },
    "format": {
      "type": "string",
      "description": "Video format. Supported: h264, h265"
    }
  },
  "required": [
    "timestamp",
    "frame_id",
    "data",
    "format"
  ]
})
_FOXGLOVE_RAW_IMAGE_SCHEMA = json.dumps({
  "title": "foxglove.RawImage",
  "description": "A raw image",
  "type": "object",
  "properties": {
    "timestamp": {
      "type": "object",
      "title": "time",
      "properties": {
        "sec": {
          "type": "integer",
          "minimum": 0
        },
        "nsec": {
          "type": "integer",
          "minimum": 0,
          "maximum": 999999999
        }
      },
      "description": "Timestamp of image"
    },
    "frame_id": {
      "type": "string",
      "description": "Frame of reference for the image."
    },
    "width": {
      "type": "integer",
      "minimum": 0,
      "description": "Image width in pixels"
    },
    "height": {
      "type": "integer",
      "minimum": 0,
      "description": "Image height in pixels"
    },
    "encoding": {
      "type": "string",
      "description": "Encoding of the raw image data (rgb8, rgba8, bgr8, 8UC3, mono8, 8UC1, mono16, 16UC1, 32FC1, yuv422, ...)."
    },
    "step": {
      "type": "integer",
      "minimum": 0,
      "description": "Byte length of a single row."
    },
    "data": {
      "type": "string",
      "contentEncoding": "base64",
      "description": "Raw image data."
    }
  },
  "required": [
    "timestamp",
    "frame_id",
    "width",
    "height",
    "encoding",
    "step",
    "data"
  ]
})


def _obj_schema(properties):
    return json.dumps({
        "type": "object",
        "properties": properties,
        "additionalProperties": True,
    }).encode()


class NavdRecorder:
    """Capture the navd teleop session to MCAP at a fixed rate."""

    def __init__(self, rig, robot, out_path,
                 rate_hz=10.0, jpeg_quality=85, workers=6,
                 max_frame_age_s=0.3, color_codec="h265"):
        self.rig = rig
        self.robot = robot
        self.rate_hz = rate_hz
        self.jpeg_quality = jpeg_quality
        self.max_frame_age_s = max_frame_age_s
        # Hardware (NVENC) color video: store H.265/H.264 access units in
        # the MCAP instead of per-tick JPEG (docs/navd.md §3.2 — the Thor
        # finally has an encoder; the Orin Nano did not). Falls back to
        # JPEG (camera MJPEG passthrough) when no encoder is available, so
        # workstation tests and non-Jetson hosts keep the old path.
        self._HwVideoEncoder = None
        self._video_codec = None
        self._encoders = {}
        try:
            from .hw_video import HwVideoEncoder, encoder_available
            if color_codec and encoder_available(color_codec):
                self._HwVideoEncoder = HwVideoEncoder
                self._video_codec = color_codec
        except Exception:
            pass
        self.bytes_written = 0
        self.frames = 0
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        # Per-camera BEV + encode jobs run here. Only numpy/cv2 work is
        # submitted (both release the GIL); every pyorbbecsdk call stays in
        # the recorder thread and MCAP writes stay serial (Writer is not
        # thread-safe). Same split as the excised BEV worker; the
        # corruption in §2.8 was from pooling capture, not processing.
        from concurrent.futures import ThreadPoolExecutor
        self._pool = ThreadPoolExecutor(max_workers=workers,
                                        thread_name_prefix="navd-rec")
        # role -> (stamp_us, {"png", "jpg"}) for the last encoded frame
        self._cache = {}
        self._file = open(out_path, "wb")
        # No chunk compression: payloads are already-compressed JPEG/PNG,
        # and the zstd chunk path has proven lossy with this mcap build.
        self._writer = Writer(self._file, compression=CompressionType.NONE)
        self._writer.start()
        state_schema = _obj_schema({"stamp_us": {"type": "integer"}})
        self._sch_state = self._writer.register_schema(
            "bebop.navd.State", "jsonschema", state_schema)
        self._sch_calib = self._writer.register_schema(
            "bebop.navd.Calib", "jsonschema", _obj_schema({}))
        # Foxglove well-known schemas (JSON-encoded): Foxglove Studio
        # resolves these by schema name and renders them in Image panels.
        # Schema names must be exactly the Foxglove well-known names
        # (foxglove.CompressedImage / foxglove.RawImage — per the schema
        # docs' JSON reference implementations) or the panels report the
        # topic "not available". Schema data is the official JSON Schema.
        self._sch_compressed_image = self._writer.register_schema(
            "foxglove.CompressedImage", "jsonschema",
            _FOXGLOVE_COMPRESSED_IMAGE_SCHEMA.encode())
        self._sch_compressed_video = self._writer.register_schema(
            "foxglove.CompressedVideo", "jsonschema",
            _FOXGLOVE_COMPRESSED_VIDEO_SCHEMA.encode())
        self._sch_raw_image = self._writer.register_schema(
            "foxglove.RawImage", "jsonschema",
            _FOXGLOVE_RAW_IMAGE_SCHEMA.encode())
        # Color is CompressedVideo (H.265/H.264) when a hardware encoder is
        # available, else the legacy per-tick CompressedImage JPEG.
        color_schema = (self._sch_compressed_video if self._video_codec
                        else self._sch_compressed_image)
        self._ch = {
            "cmd_vel": self._writer.register_channel(
                "/cmd_vel", "json", self._sch_state),
            "odom": self._writer.register_channel(
                "/odom", "json", self._sch_state),
            "calib": self._writer.register_channel(
                "/calib", "json", self._sch_calib),
            "color_near": self._writer.register_channel(
                "/color_near", "json", color_schema),
            "color_far": self._writer.register_channel(
                "/color_far", "json", color_schema),
            "depth_near_preview": self._writer.register_channel(
                "/depth_near_preview", "json", self._sch_raw_image),
            "depth_far_preview": self._writer.register_channel(
                "/depth_far_preview", "json", self._sch_raw_image),
            # Training-depth channels: CompressedImage-wrapped lossless PNG
            # (16-bit). Foxglove rejects schemaless "raw" channels, so the
            # bytes ride in base64 like the color channel; the extractor
            # unwraps them back to PNG bytes.
            self._depth_topic("near"): self._writer.register_channel(
                self._depth_topic("near"), "json", self._sch_compressed_image),
            self._depth_topic("far"): self._writer.register_channel(
                self._depth_topic("far"), "json", self._sch_compressed_image),
        }
        self._write_calib()

    # --- payloads ------------------------------------------------------------

    def _write_calib(self):
        from .orbbec import load_intrinsics, load_rig_config
        cfg = load_rig_config()["robots"]["default"]
        calib = {"intrinsics": {
                     serial: {**{k: float(intr[k])
                                 for k in ("fx", "fy", "cx", "cy")},
                              "width": int(intr.get("width", 0)),
                              "height": int(intr.get("height", 0))}
                     for serial in cfg["cameras"]
                     for intr in [load_intrinsics(serial)]},
                 "mounts": {s: {"height_m": float(c["height_m"]),
                                "pitch_deg": float(c["pitch_deg"]),
                                "yaw_deg": float(c.get("yaw_deg", 0.0))}
                            for s, c in cfg["cameras"].items()},
                 "camera_self_mask_pixels": {
                     role: [list(r) for r in cam.mask_rects]
                     for role, cam in self.rig.cameras.items()},
                 "camera_self_mask_polygons": {
                     role: [list(p) for p in getattr(cam, "mask_polys", [])]
                     for role, cam in self.rig.cameras.items()},
                 "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        self._add(self._ch["calib"], json.dumps(calib).encode())

    def _add(self, channel_id, data, log_ns=None):
        # MCAP log_time/publish_time are nanoseconds since the epoch.
        log_ns = log_ns if log_ns is not None else time.time_ns()
        self._writer.add_message(channel_id, log_ns, data, log_ns)
        self.bytes_written += len(data)

    @staticmethod
    def _b64(arr):
        return base64.b64encode(arr.tobytes()).decode()

    # --- loop ----------------------------------------------------------------

    def start(self):
        if self._running:
            return self
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="navd-recorder")
        self._thread.start()
        return self

    def _loop(self):
        interval = 1.0 / self.rate_hz
        seq = 0
        while self._running:
            t0 = time.monotonic()
            log_ns = time.time_ns()
            try:
                self._tick(log_ns, seq)
            except Exception as exc:
                print(f"[recorder] tick error: {type(exc).__name__}: {exc}")
            seq += 1
            self.frames += 1
            time.sleep(max(0.0, interval - (time.monotonic() - t0)))

    @staticmethod
    def _depth_topic(role):
        return f"/depth_{role}"

    @staticmethod
    def _stamp(log_ns):
        return {"sec": int(log_ns // 1_000_000_000),
                "nsec": int(log_ns % 1_000_000_000)}

    def _pair_frames(self, max_age_s):
        """Freshest same-instant frame pair across the camera histories.

        Returns ({role: StampedFrame}, {role: delta_ms vs the anchor}).
        Cameras without history support (test fakes) fall back to read().
        A single-camera rig just uses its latest frame.
        """
        hists = {}
        for role, cam in self.rig.cameras.items():
            h = []
            try:
                h = [f for f in cam.recent() if f.age_s() <= max_age_s]
            except AttributeError:
                pass
            if not h:  # no history support or capture hasn't started yet
                f = cam.read()
                h = [f] if f is not None and f.age_s() <= max_age_s else []
            if not h:
                continue
            hists[role] = h
        if not hists:
            return {}, {}
        if len(hists) == 1:
            role, h = next(iter(hists.items()))
            return {role: h[-1]}, {}
        anchor_role, anchor_hist = next(iter(hists.items()))
        best, best_pair, best_deltas = None, None, {}
        for nf in reversed(anchor_hist):  # freshest anchor first
            pair, deltas = {anchor_role: nf}, {}
            for role, h in hists.items():
                if role == anchor_role:
                    continue
                ff = min(h, key=lambda f: abs(f.recv_ts - nf.recv_ts))
                pair[role] = ff
                deltas[role] = abs(ff.recv_ts - nf.recv_ts)
            worst = max(deltas.values(), default=0.0)
            if best is None or worst < best:
                best, best_pair, best_deltas = worst, pair, deltas
            if best < 0.002:  # sub-2 ms: freshest such pair wins
                break
        pair_ms = {role: d * 1e3 for role, d in best_deltas.items()}
        return best_pair, pair_ms

    @staticmethod
    def _compressed_image_msg(frame_id, fmt, blob, log_ns,
                              stamp_us=None, pair_ms=None):
        msg = {
            "timestamp": NavdRecorder._stamp(log_ns),
            "frame_id": frame_id,
            "format": fmt,
            "data": base64.b64encode(blob).decode(),
        }
        if stamp_us is not None:
            msg["stamp_us"] = int(stamp_us)
        if pair_ms is not None:
            msg["pair_ms"] = round(float(pair_ms), 3)
        return json.dumps(msg).encode()

    def _tick(self, log_ns, seq):
        st = self.robot.state
        self._add(self._ch["cmd_vel"],
                  json.dumps({"vx": float(st.cmd[0]), "wz": float(st.cmd[1]),
                              "stamp_ns": log_ns}).encode(), log_ns)
        self._add(self._ch["odom"],
                  json.dumps({"x": float(st.odom[0]), "y": float(st.odom[1]),
                              "theta": float(st.odom[2]),
                              "stamp_ns": log_ns}).encode(), log_ns)
        # Camera reads stay serial in this thread (pyorbbecsdk must not be
        # pooled — §2.8). BEV + PNG/JPEG encodes are numpy/cv2 (GIL-releasing)
        # and run as per-camera jobs in the pool; results are written
        # serially. With both cameras (PNG16 + JPEG + BEV each) the tick
        # measures ~165 ms at 6 workers on the Orin Nano (vs ~330 ms fully
        # serial, 2026-09-06); 6 cores — more workers oversubscribes.
        #
        # Cross-camera pairing: hardware sync phase-locks the capture
        # instants (tools/orbbec_sync_test.py: host-clock phase spread
        # 0.00 ms), but each camera feeds its own latest-wins slot, so two
        # independent slot reads can land a frame period apart (measured
        # med 0.3 ms / max 76.6 ms, tools/orbbec_slot_probe.py). Pairing
        # takes the freshest same-instant pair from the per-camera
        # histories; the residual |Δrecv| rides in the image messages
        # (pair_ms) with the device stamp (stamp_us) so offline consumers
        # can verify alignment.
        frames, pair_ms = self._pair_frames(self.max_frame_age_s)
        jobs = {}
        for role, f in frames.items():
            cached = self._cache.get(role)
            if cached is not None and cached[0] == f.stamp_us:
                jobs[role] = cached[1]
                continue
            job = {"png": self._pool.submit(self._encode_png16, f.depth)}
            if self._video_codec:
                # Camera MJPEG goes straight to the hardware nvjpegdec ->
                # NVENC chain (zero CPU pixels). Only a raw-RGB rig needs
                # the pool decode to BGR.
                if f.color_jpeg is not None:
                    job["color"] = f.color_jpeg
                    job["color_kind"] = "jpeg"
                else:
                    job["color"] = self._pool.submit(self._decode_bgr, f)
                    job["color_kind"] = "bgr"
            else:
                # Camera-MJPEG frames arrive pre-encoded (rig color_format:
                # mjpg) — the bytes go into the MCAP verbatim, no CPU encode.
                job["jpg"] = (f.color_jpeg if f.color_jpeg is not None else
                              (self._pool.submit(self._encode_jpeg, f.color,
                                                 self.jpeg_quality)
                               if f.color is not None else None))
            jobs[role] = {"fut": job}
        for role, f in frames.items():
            job = jobs[role]
            if "fut" in job:
                fut = job["fut"]
                png = fut["png"].result()
                job = {"png": png}
                if self._video_codec:
                    color = fut["color"]
                    if hasattr(color, "result"):
                        color = color.result()
                    job["color"] = color
                    job["color_kind"] = fut["color_kind"]
                else:
                    jpg = fut["jpg"]
                    if jpg is not None and hasattr(jpg, "result"):
                        jpg = jpg.result()
                    job["jpg"] = jpg
                jobs[role] = job
                self._cache[role] = (f.stamp_us, job)
            meta = {"stamp_us": f.stamp_us, "pair_ms": pair_ms.get(role)}
            self._add(self._ch[self._depth_topic(role)],
                      self._compressed_image_msg(f"depth_{role}", "png",
                                                 job["png"], log_ns, **meta),
                      log_ns)
            if self._video_codec:
                self._write_video_color(role, job.get("color"),
                                        job.get("color_kind"), log_ns, meta)
            elif job.get("jpg") is not None:
                self._add(self._ch[f"color_{role}"],
                          self._compressed_image_msg(f"{role}_color", "jpeg",
                                                     job["jpg"], log_ns, **meta),
                          log_ns)
            if role in ("near", "far"):
                self._add(self._ch[f"depth_{role}_preview"],
                          self._depth_preview(f.depth, log_ns,
                                              frame_id=f"{role}_depth_preview"),
                          log_ns)

    def _write_video_color(self, role, payload, kind, log_ns, meta):
        """Push one color frame through this role's NVENC and write the AUs.

        The encoder runs ~1 frame behind, so each AU is tagged with the
        tick that produced it (passed through the encoder FIFO) and written
        at that tick's log_time — keeping it grouped with the same tick's
        depth/cmd in the extractor. A fresh encoder is built lazily from
        the first frame; `payload` is camera MJPEG bytes (`kind="jpeg"`) or
        a BGR ndarray (`kind="bgr"`).
        """
        if payload is None:
            return
        enc = self._encoders.get(role)
        if enc is None:
            try:
                if kind == "jpeg":
                    w = h = 0  # nvjpegdec supplies its own dimensions
                else:
                    h, w = payload.shape[0], payload.shape[1]
                enc = self._HwVideoEncoder(
                    w, h, fps=self.rate_hz, codec=self._video_codec,
                    input_kind=kind)
            except Exception as exc:
                print(f"[recorder] hw encoder {role} unavailable: {exc}; "
                      f"falling back to JPEG")
                self._video_codec = None
                return
            self._encoders[role] = enc
        tag = (log_ns, meta.get("stamp_us"), meta.get("pair_ms"))
        for au, out_tag in enc.push(payload, tag):
            t_log, t_stamp, t_pair = out_tag if out_tag else tag
            self._add(self._ch[f"color_{role}"],
                      self._compressed_image_msg(
                          f"{role}_color", self._video_codec, au, t_log,
                          stamp_us=t_stamp, pair_ms=t_pair),
                      t_log)

    @staticmethod
    def _decode_bgr(f):
        """BGR ndarray for the hardware encoder, from RGB or camera MJPEG."""
        if f.color is not None:
            return cv2.cvtColor(f.color, cv2.COLOR_RGB2BGR)
        if f.color_jpeg is not None:
            bgr = cv2.imdecode(np.frombuffer(f.color_jpeg, np.uint8),
                               cv2.IMREAD_COLOR)
            return bgr
        return None

    @staticmethod
    def _encode_jpeg(color, quality):
        ok, jpg = cv2.imencode(
            ".jpg", cv2.cvtColor(color, cv2.COLOR_RGB2BGR),
            [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        return jpg.tobytes() if ok else b""

    @staticmethod
    def _encode_png16(depth):
        ok, png = cv2.imencode(
            ".png", depth, [int(cv2.IMWRITE_PNG_COMPRESSION), 1,
                            int(cv2.IMWRITE_PNG_STRATEGY),
                            int(cv2.IMWRITE_PNG_STRATEGY_RLE)])
        return png.tobytes() if ok else b""

    @staticmethod
    def _depth_preview(depth, log_ns, stride=8, frame_id="near_depth_preview"):
        """Quarter-frame 16uc1 RawImage payload — dashboard only."""
        small = np.ascontiguousarray(depth[::stride, ::stride])
        h, w = small.shape
        return json.dumps({
            "timestamp": NavdRecorder._stamp(log_ns),
            "frame_id": frame_id,
            "width": int(w), "height": int(h),
            "encoding": "16UC1", "step": int(w * 2),
            "data": base64.b64encode(small.tobytes()).decode(),
        }).encode()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._pool.shutdown(wait=True)
        # Drain each hardware encoder's tail (the last GOP-buffered AUs)
        # and write them at their original tick log_time.
        for role, enc in self._encoders.items():
            try:
                for au, tag in enc.flush():
                    t_log, t_stamp, t_pair = tag if tag else (
                        time.time_ns(), None, None)
                    self._add(self._ch[f"color_{role}"],
                              self._compressed_image_msg(
                                  f"{role}_color", self._video_codec, au,
                                  t_log, stamp_us=t_stamp, pair_ms=t_pair),
                              t_log)
            except Exception as exc:
                print(f"[recorder] hw encoder {role} flush failed: {exc}")
            finally:
                enc.close()
        self._encoders = {}
        with self._lock:
            self._writer.finish()
            self._file.close()

