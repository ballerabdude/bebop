"""navd recorder v2: teleop sessions -> one MCAP file per session (plan §7.1).

MCAP is the single data artifact: copied off the robot (scp) it carries
everything the workstation needs — color, both depths, the operator's
teleop twist (the imitation label), odometry, the goal slot, the online
geometric teacher grid, and calibration. Indexed and seekable, opens in
Foxglove for review, and `tools/mcap_extract.py` unpacks it into the
`datasets/navd-v0/` training layout.

Channels:
  /color_near   foxglove.CompressedImage (JSON: {format: "jpeg", data: b64})
  /color_far    foxglove.CompressedImage (same encoding; both cameras stream
                RGB since the far camera's USB 3 cable swap)
  /depth_near   raw PNG bytes (schemaless; uint16 mm, lossless — training data)
  /depth_far    raw PNG bytes (schemaless)
  /depth_near_preview  foxglove.RawImage (JSON; 106x60 16uc1 — dashboard only)
  /depth_far_preview   foxglove.RawImage (same encoding, far camera)
  /bev_map      foxglove.RawImage (JSON; 60x60 rgb8 top-down teacher map)
  /cmd_vel      JSON  {"vx", "wz", "stamp_ns"}   — operator twist (teleop label)
  /odom         JSON  {"x", "y", "theta", "stamp_ns"}
  /goal         JSON  {"type": "heading"|"point"|"none", ...}
  /bev_teacher  JSON  {"raw": b64 60x60 uint8, "plane_ok": {...}, "stamp_ns"}
  /calib        JSON  intrinsics + rig extrinsics, written once at start

Training channels are the raw PNGs + /bev_teacher (tools/mcap_extract.py);
the Foxglove-schema channels exist so sessions open as a live dashboard in
Foxglove Studio (foxglove/bebop_navd_layout.json).

All messages of one tick share the same log_time (µs since epoch), so the
extractor can group them by exact match.
"""

import base64
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

from .bev import BevBuilder
from .goal_planner import GoalHeading, GoalPoint


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

    def __init__(self, rig, robot, goal_slot, out_path, builder=None,
                 rate_hz=10.0, jpeg_quality=85, workers=6):
        self.rig = rig
        self.robot = robot
        self.goal_slot = goal_slot
        self.rate_hz = rate_hz
        self.jpeg_quality = jpeg_quality
        self.builder = builder or BevBuilder()
        self.bytes_written = 0
        self.frames = 0
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        # Per-camera BEV + encode jobs run here. Only numpy/cv2 work is
        # submitted (both release the GIL); every pyorbbecsdk call stays in
        # the recorder thread and MCAP writes stay serial (Writer is not
        # thread-safe). Same split as the --goal-drive BEV worker; the
        # corruption in §2.8 was from pooling capture, not processing.
        from concurrent.futures import ThreadPoolExecutor
        self._pool = ThreadPoolExecutor(max_workers=workers,
                                        thread_name_prefix="navd-rec")
        # role -> (stamp_us, {"png", "jpg", "bev"}) for the last encoded frame
        self._cache = {}
        self._file = open(out_path, "wb")
        # No chunk compression: payloads are already-compressed JPEG/PNG,
        # and the zstd chunk path has proven lossy with this mcap build.
        self._writer = Writer(self._file, compression=CompressionType.NONE)
        self._writer.start()
        state_schema = _obj_schema({"stamp_us": {"type": "integer"}})
        self._sch_state = self._writer.register_schema(
            "bebop.navd.State", "jsonschema", state_schema)
        self._sch_bev = self._writer.register_schema(
            "bebop.navd.BevTeacher", "jsonschema", _obj_schema(
                {"raw": {"type": "string"}, "plane_ok": {"type": "object"},
                 "stamp_us": {"type": "integer"}}))
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
        self._sch_raw_image = self._writer.register_schema(
            "foxglove.RawImage", "jsonschema",
            _FOXGLOVE_RAW_IMAGE_SCHEMA.encode())
        self._ch = {
            "cmd_vel": self._writer.register_channel(
                "/cmd_vel", "json", self._sch_state),
            "odom": self._writer.register_channel(
                "/odom", "json", self._sch_state),
            "goal": self._writer.register_channel(
                "/goal", "json", self._sch_state),
            "bev": self._writer.register_channel(
                "/bev_teacher", "json", self._sch_bev),
            "calib": self._writer.register_channel(
                "/calib", "json", self._sch_calib),
            "color_near": self._writer.register_channel(
                "/color_near", "json", self._sch_compressed_image),
            "color_far": self._writer.register_channel(
                "/color_far", "json", self._sch_compressed_image),
            "depth_near_preview": self._writer.register_channel(
                "/depth_near_preview", "json", self._sch_raw_image),
            "depth_far_preview": self._writer.register_channel(
                "/depth_far_preview", "json", self._sch_raw_image),
            "bev_map": self._writer.register_channel(
                "/bev_map", "json", self._sch_raw_image),
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
        calib = {"intrinsics": self.builder.intrinsics,
                 "mounts": {s: {"height_m": m.height_m, "pitch_deg": m.pitch_deg,
                                "yaw_deg": m.yaw_deg}
                            for s, m in self.builder.mounts.items()},
                 "bev": {"range_m": self.builder.range_m,
                         "width_m": self.builder.width_m,
                         "cell_m": self.builder.cell_m,
                         "near_authority_m": self.builder.near_authority_m,
                         "min_range_m": self.builder.min_range_m},
                 "camera_self_mask_pixels": {
                     role: [list(r) for r in cam.mask_rects]
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

    @staticmethod
    def _compressed_image_msg(frame_id, fmt, blob, log_ns):
        return json.dumps({
            "timestamp": NavdRecorder._stamp(log_ns),
            "frame_id": frame_id,
            "format": fmt,
            "data": base64.b64encode(blob).decode(),
        }).encode()

    def _tick(self, log_ns, seq):
        st = self.robot.state
        self._add(self._ch["cmd_vel"],
                  json.dumps({"vx": float(st.cmd[0]), "wz": float(st.cmd[1]),
                              "stamp_ns": log_ns}).encode(), log_ns)
        self._add(self._ch["odom"],
                  json.dumps({"x": float(st.odom[0]), "y": float(st.odom[1]),
                              "theta": float(st.odom[2]),
                              "stamp_ns": log_ns}).encode(), log_ns)
        goal = self.goal_slot.get()
        if isinstance(goal, GoalHeading):
            g = {"type": "heading", "heading_rad": float(goal.heading_rad)}
        elif isinstance(goal, GoalPoint):
            g = {"type": "point", "x": float(goal.x), "y": float(goal.y)}
        else:
            g = {"type": "none"}
        self._add(self._ch["goal"], json.dumps(g).encode(), log_ns)

        # Camera reads stay serial in this thread (pyorbbecsdk must not be
        # pooled — §2.8). BEV + PNG/JPEG encodes are numpy/cv2 (GIL-releasing)
        # and run as per-camera jobs in the pool; results are written
        # serially. With both cameras (PNG16 + JPEG + BEV each) the tick
        # measures ~165 ms at 6 workers on the Orin Nano (vs ~330 ms fully
        # serial, 2026-09-06); 6 cores — more workers oversubscribes.
        #
        # A camera whose frame stamp is unchanged since the last tick (far
        # ticks at 10 fps, below the recorder rate) reuses the cached
        # encoded payloads + BEV result — the channel is still written every
        # tick so the extractor's per-tick alignment contract holds.
        frames = {}
        for role, cam in self.rig.cameras.items():
            f = cam.read()
            if f is None or f.age_s() > self.builder.max_frame_age_s:
                continue
            frames[role] = f
        jobs = {}
        for role, f in frames.items():
            cached = self._cache.get(role)
            if cached is not None and cached[0] == f.stamp_us:
                jobs[role] = cached[1]
                continue
            jobs[role] = {"fut": {
                "bev": self._pool.submit(self.builder.process, f),
                "png": self._pool.submit(self._encode_png16, f.depth),
                # Camera-MJPEG frames arrive pre-encoded (rig color_format:
                # mjpg) — the bytes go into the MCAP verbatim, no CPU encode.
                "jpg": (f.color_jpeg if f.color_jpeg is not None else
                        (self._pool.submit(self._encode_jpeg, f.color,
                                           self.jpeg_quality)
                         if f.color is not None else None)),
            }}
        per_cam, ages = {}, {}
        for role, f in frames.items():
            job = jobs[role]
            if "fut" in job:
                fut = job["fut"]
                try:
                    bev = fut["bev"].result()
                except Exception as exc:
                    print(f"[recorder] BEV error ({role}): "
                          f"{type(exc).__name__}: {exc}")
                    bev = None
                png = fut["png"].result()
                jpg = fut["jpg"]
                if jpg is not None and hasattr(jpg, "result"):
                    jpg = jpg.result()
                job = {"png": png, "jpg": jpg, "bev": bev}
                jobs[role] = job
                self._cache[role] = (f.stamp_us, job)
            png, jpg = job["png"], job["jpg"]
            per_cam[role] = job["bev"]
            self._add(self._ch[self._depth_topic(role)],
                      self._compressed_image_msg(f"depth_{role}", "png", png,
                                                 log_ns),
                      log_ns)
            if jpg is not None:
                self._add(self._ch[f"color_{role}"],
                          self._compressed_image_msg(f"{role}_color", "jpeg",
                                                     jpg, log_ns),
                          log_ns)
            if role in ("near", "far"):
                self._add(self._ch[f"depth_{role}_preview"],
                          self._depth_preview(f.depth, log_ns,
                                              frame_id=f"{role}_depth_preview"),
                          log_ns)
            ages[role] = f.age_s()
        grid = self.builder.fuse(per_cam, ages)
        bev = {"raw": self._b64(grid.raw) if grid is not None else None,
               "plane_ok": grid.plane_ok if grid is not None else {},
               "stamp_ns": log_ns}
        self._add(self._ch["bev"], json.dumps(bev).encode(), log_ns)
        if grid is not None:
            self._add(self._ch["bev_map"],
                      self._bev_map_image(grid, goal, log_ns), log_ns)

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

    @staticmethod
    def _bev_map_image(grid, goal=None, log_ns=0):
        """Top-down 60x60 rgb8 RawImage of the fused teacher grid.

        Row 0 = far edge (+range), row grows toward the robot; col 0 =
        right edge (y = -width/2). Free cells dark, occupied red, hazard
        orange, inflated blue-gray — same colors as the --display overlay.
        """
        colors = {
            0: (40, 40, 40),    # free
            1: (60, 40, 220),   # occupied (rgb)
            2: (40, 150, 255),  # hazard
            3: (150, 90, 90),   # inflated
        }
        h, w = grid.occ.shape
        img = np.zeros((h, w, 3), np.uint8)
        for cls, rgb in colors.items():
            img[grid.occ == cls] = rgb
        if goal is not None:
            bearing = (goal.heading_rad if hasattr(goal, "heading_rad")
                       else goal[2] if isinstance(goal, tuple) else None)
            if bearing is not None:
                import math
                cy, cx = h - 1, w // 2
                hy = int(round(cy - 20 * math.cos(bearing)))
                hx = int(round(cx + 20 * math.sin(bearing)))
                cv2.line(img, (cx, cy), (hx, hy), (255, 255, 255), 1)
        return json.dumps({
            "timestamp": NavdRecorder._stamp(log_ns),
            "frame_id": "navd_bev",
            "width": int(w), "height": int(h),
            "encoding": "rgb8", "step": int(w * 3),
            "data": base64.b64encode(img.tobytes()).decode(),
        }).encode()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._pool.shutdown(wait=True)
        with self._lock:
            self._writer.finish()
            self._file.close()
