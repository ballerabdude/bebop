"""navd recorder v2: teleop sessions -> one MCAP file per session (plan §7.1).

MCAP is the single data artifact: copied off the robot (scp) it carries
everything the workstation needs — color, both depths, the operator's
teleop twist (the imitation label), odometry, and calibration. Indexed
and seekable, opens in Foxglove and Rerun for review, and
`tools/mcap_extract.py` unpacks it into the `datasets/navd-v0/` training
layout.

Channels:
  /color_near   foxglove.CompressedVideo (H.265 access units, hardware
                NVENC; falls back to CompressedImage JPEG when no encoder)
  /color_far    same encoding (both cameras stream MJPEG, re-encoded to
                H.265 for storage; the camera's own JPEG is only kept on
                the fallback path)
  /depth_near   foxglove.CompressedImage (lossless 16-bit PNG — training data)
  /depth_far    same encoding
  /cmd_vel      bebop.navd.Twist (Protobuf) — operator twist (teleop label)
  /odom         bebop.navd.Odom (Protobuf)
  /imu_accel_<role>  bebop.navd.ImuAccel (Protobuf, full accel rate)
  /imu_gyro_<role>   bebop.navd.ImuGyro  (Protobuf, full gyro rate)
  /calib        JSON intrinsics + rig extrinsics, written once at start

All image/video/telemetry channels are Protobuf-encoded so Rerun's MCAP
decoder renders them directly (images/video via `foxglove.*`, telemetry via
`bebop.navd.*` struct fields the dashboard maps to plots); `/calib` stays
JSON.

All messages of one tick share the same log_time (ns since epoch), so the
extractor can group them by exact match. Cross-camera pairing (the
hardware-sync residual, kept ~0-2 ms) is used to choose which frame pair
to record; the winning device stamp / residual are no longer written to
the wire (the standard Foxglove protobuf schemas have no field for them).
"""

import json
import threading
import time

import numpy as np

from . import foxglove_proto
from . import navd_proto

try:
    from mcap.writer import Writer, CompressionType
except ImportError as exc:  # pragma: no cover
    raise ImportError("pip install mcap") from exc

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise ImportError("pip install opencv-python") from exc



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
        # Segment start (monotonic): the cameras run across segments and keep
        # buffering IMU, so the first tick would otherwise flush the idle-period
        # backlog into this segment (IMU predating the video/depth). Drop any
        # sample older than this.
        self._t0 = time.monotonic()
        self.rate_hz = rate_hz
        self.jpeg_quality = jpeg_quality
        self.max_frame_age_s = max_frame_age_s
        # Hardware (NVENC) color video: store H.265/H.264 access units in
        # the MCAP instead of per-tick JPEG (docs/navd.md §3.2 — the Thor
        # finally has an encoder; the Orin Nano did not). Falls back to
        # JPEG (camera MJPEG passthrough) when no encoder is available, so
        # workstation tests and non-Jetson hosts keep the old path.
        #
        # H.265 is the default for storage efficiency. Reviewing in Foxglove
        # needs a client whose WebCodecs can decode HEVC (there is no
        # software path — see the CompressedVideo docs' platform caveat).
        # The native Rerun viewer decodes HEVC via system FFmpeg with no
        # GPU, but its `foxglove` decoder maps Foxglove messages only when
        # they are Protobuf-encoded (this recorder currently writes JSON).
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
        self._sch_calib = self._writer.register_schema(
            "bebop.navd.Calib", "jsonschema", _obj_schema({}))
        # Telemetry as Protobuf structs: Rerun's generic Protobuf decoder
        # exposes their fields for the dashboard's component-mapped plots,
        # so a session opens directly with no export step.
        self._sch_twist = self._writer.register_schema(
            *navd_proto.schema(navd_proto.Twist))
        self._sch_odom = self._writer.register_schema(
            *navd_proto.schema(navd_proto.Odom))
        # Foxglove well-known schemas, Protobuf-encoded. Protobuf (not the
        # JSON schemas) is what Rerun's `foxglove` MCAP decoder understands,
        # so `rerun <session>.mcap` maps these topics to Rerun archetypes;
        # Foxglove Studio reads protobuf schemas too. Schema data is a
        # serialized FileDescriptorSet (see foxglove_proto).
        self._sch_compressed_image = self._writer.register_schema(
            *foxglove_proto.schema(foxglove_proto.CompressedImage))
        self._sch_compressed_video = self._writer.register_schema(
            *foxglove_proto.schema(foxglove_proto.CompressedVideo))
        # Color is CompressedVideo (H.265/H.264) when a hardware encoder is
        # available, else the legacy per-tick CompressedImage JPEG.
        color_schema = (self._sch_compressed_video if self._video_codec
                        else self._sch_compressed_image)
        self._ch = {
            "cmd_vel": self._writer.register_channel(
                "/cmd_vel", "protobuf", self._sch_twist),
            "odom": self._writer.register_channel(
                "/odom", "protobuf", self._sch_odom),
            "calib": self._writer.register_channel(
                "/calib", "json", self._sch_calib),
            "color_near": self._writer.register_channel(
                "/color_near", "protobuf", color_schema),
            "color_far": self._writer.register_channel(
                "/color_far", "protobuf", color_schema),
            # Training-depth channels: CompressedImage-wrapped lossless PNG
            # (16-bit); the extractor unwraps `data` back to PNG bytes.
            self._depth_topic("near"): self._writer.register_channel(
                self._depth_topic("near"), "protobuf",
                self._sch_compressed_image),
            self._depth_topic("far"): self._writer.register_channel(
                self._depth_topic("far"), "protobuf",
                self._sch_compressed_image),
        }
        # Camera IMU: full-rate accel/gyro samples drained from each camera's
        # capture-thread buffer (not one latest-sample-per-tick). Log_time is
        # host epoch reconstructed from the sample's monotonic recv_ts;
        # `device_stamp_us` is the camera clock VIO fuses on.
        self._sch_imu_accel = self._writer.register_schema(
            *navd_proto.schema(navd_proto.ImuAccel))
        self._sch_imu_gyro = self._writer.register_schema(
            *navd_proto.schema(navd_proto.ImuGyro))
        self._imu_ch = {
            role: {"accel": self._writer.register_channel(
                       f"/imu_accel_{role}", "protobuf", self._sch_imu_accel),
                   "gyro": self._writer.register_channel(
                       f"/imu_gyro_{role}", "protobuf", self._sch_imu_gyro)}
            for role in self.rig.cameras}
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
    def _compressed_msg(msg_cls, frame_id, fmt, blob, log_ns):
        """Protobuf foxglove CompressedImage/CompressedVideo bytes."""
        return msg_cls(timestamp=foxglove_proto.timestamp(log_ns),
                       frame_id=frame_id, data=blob,
                       format=fmt).SerializeToString()

    def _tick(self, log_ns, seq):
        st = self.robot.state
        self._add(self._ch["cmd_vel"],
                  navd_proto.Twist(vx=float(st.cmd[0]), wz=float(st.cmd[1]),
                                   stamp_ns=log_ns).SerializeToString(), log_ns)
        self._add(self._ch["odom"],
                  navd_proto.Odom(x=float(st.odom[0]), y=float(st.odom[1]),
                                  theta=float(st.odom[2]),
                                  stamp_ns=log_ns).SerializeToString(), log_ns)
        self._drain_imu(time.time_ns() - time.monotonic_ns())
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
                      self._compressed_msg(
                          foxglove_proto.CompressedImage, f"depth_{role}",
                          "png", job["png"], log_ns),
                      log_ns)
            if self._video_codec:
                self._write_video_color(role, job.get("color"),
                                        job.get("color_kind"), log_ns, meta)
            elif job.get("jpg") is not None:
                self._add(self._ch[f"color_{role}"],
                          self._compressed_msg(
                              foxglove_proto.CompressedImage,
                              f"{role}_color", "jpeg", job["jpg"], log_ns),
                          log_ns)

    def _drain_imu(self, wall_off):
        """Write all buffered camera IMU samples at their own log times.

        The buffer is drained every tick so nothing is dropped between ticks
        (unlike the image path's latest-wins slots). `wall_off` converts the
        sample's monotonic arrival time to host epoch ns; `device_stamp_us`
        rides along for device-clock fusion. Samples older than the segment
        start are discarded (the cameras buffer across segments).
        """
        for role, cam in self.rig.cameras.items():
            drain = getattr(cam, "drain_imu", None)
            chans = self._imu_ch.get(role)
            if drain is None or not chans:
                continue
            for kind, x, y, z, dev_us, recv_ts in drain():
                if recv_ts < self._t0:
                    continue
                log_ns = int(recv_ts * 1e9) + wall_off
                if kind == "accel":
                    msg = navd_proto.ImuAccel(ax=x, ay=y, az=z,
                                              stamp_ns=log_ns,
                                              device_stamp_us=int(dev_us))
                    cid = chans["accel"]
                else:
                    msg = navd_proto.ImuGyro(gx=x, gy=y, gz=z,
                                             stamp_ns=log_ns,
                                             device_stamp_us=int(dev_us))
                    cid = chans["gyro"]
                self._add(cid, msg.SerializeToString(), log_ns)

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
            t_log = (out_tag if out_tag else tag)[0]
            self._add(self._ch[f"color_{role}"],
                      self._compressed_msg(
                          foxglove_proto.CompressedVideo, f"{role}_color",
                          self._video_codec, au, t_log),
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
                    t_log = tag[0] if tag else time.time_ns()
                    self._add(self._ch[f"color_{role}"],
                              self._compressed_msg(
                                  foxglove_proto.CompressedVideo,
                                  f"{role}_color", self._video_codec, au,
                                  t_log),
                              t_log)
            except Exception as exc:
                print(f"[recorder] hw encoder {role} flush failed: {exc}")
            finally:
                enc.close()
        self._encoders = {}
        with self._lock:
            self._writer.finish()
            self._file.close()

