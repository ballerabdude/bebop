"""Unpack a navd MCAP session into the navd-v0 training layout.

On the workstation, after scp'ing the session file off the robot:

    python tools/mcap_extract.py session.mcap datasets/navd-v0/session01

Produces (per aligned tick — all channels of a tick share log_time):
    color/{stamp}.jpg          PE camera frame (as recorded, JPEG)
    color_far/{stamp}.jpg      ED camera frame (as recorded, JPEG)
    depth/{stamp}.npz          near, far  (uint16 mm)
    labels/{stamp}.npz         teacher (uint8 60x60) — pre-fill for hand labeling
    manifest.jsonl             stamp_ns, cmd_vel, odom, goal, plane_ok, paths
    imu_{accel,gyro}_{near,far}.npz  device_us, stamp_ns, x, y, z (full rate)

Hand labels overwrite `labels/{stamp}.npz`'s `hand` array (same 60x60
uint8 semantics: 0 navigable, 1 blocked, 2 caution); training prefers
`hand` when present and falls back to `teacher`.
"""

import argparse
import base64
import bisect
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from bebop_vision.foxglove_proto import (CompressedImage,
                                            CompressedVideo, RawImage)
    from bebop_vision import navd_proto
    _PROTO = {m.DESCRIPTOR.full_name: m
              for m in (CompressedImage, CompressedVideo, RawImage)}
    _NAVD = {navd_proto.Twist.DESCRIPTOR.full_name: navd_proto.Twist,
             navd_proto.Odom.DESCRIPTOR.full_name: navd_proto.Odom}
    _IMU = {navd_proto.ImuAccel.DESCRIPTOR.full_name: navd_proto.ImuAccel,
            navd_proto.ImuGyro.DESCRIPTOR.full_name: navd_proto.ImuGyro}
except ImportError:  # pragma: no cover - standalone use
    _PROTO = {}
    _NAVD = {}
    _IMU = {}

try:
    from tools.mcap_recover import iter_messages as iter_mcap_messages
except ImportError:  # pragma: no cover - standalone use
    from mcap_recover import iter_messages as iter_mcap_messages


def _foxglove_blob(schema_name, message_encoding, data):
    """(blob, format) from a Foxglove image/video message.

    Protobuf is what the recorder writes now; JSON is kept so sessions
    recorded before the protobuf migration still extract.
    """
    if message_encoding == "protobuf" and schema_name in _PROTO:
        msg = _PROTO[schema_name].FromString(data)
        return msg.data, getattr(msg, "format", None)
    payload = json.loads(data)
    return base64.b64decode(payload["data"]), payload.get("format")


def _decode_video(aus_by_role):
    """Decode hardware H.264/H.265 access units to per-log_time BGR frames.

    AUs are fed to PyAV (software decode) in log_time order; a decoder
    emits frames in display order, so the i-th decoded frame is paired with
    the i-th AU (the recorder emits one AU per recorded tick). Returns
    {role: {log_time: BGR ndarray}}.
    """
    import av
    decoded = {}
    for role, items in aus_by_role.items():
        if not items:
            continue
        items = sorted(items, key=lambda x: x[0])
        # Foxglove/FFmpeg naming: the H.265 bitstream is format "h265" but
        # PyAV's decoder is "hevc".
        codec = {"h265": "hevc", "h264": "h264"}[items[0][1]]
        cc = av.CodecContext.create(codec, "r")
        frames = []
        for _, _, au in items:
            for fr in cc.decode(av.Packet(au)):
                frames.append(fr.to_ndarray(format="bgr24"))
        for fr in cc.decode(None):
            frames.append(fr.to_ndarray(format="bgr24"))
        n = min(len(frames), len(items))
        decoded[role] = {items[i][0]: frames[i] for i in range(n)}
    return decoded


def extract(mcap_path, out_dir, tol_us=15_000):
    out = Path(out_dir)
    for sub in ("color", "color_far", "depth", "labels"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    IMAGE_TOPICS = ("/color_near", "/color_far", "/depth_near", "/depth_far")
    COLOR_ROLE = {"/color_near": "near", "/color_far": "far"}
    VIDEO_FORMATS = ("h264", "h265")
    ticks = {}  # log_us -> {topic: decoded payload}
    video_aus = {"near": [], "far": []}  # role -> [(log_time, codec, bytes)]
    imu = {}  # topic -> [ImuAccel/ImuGyro] (kept out of `ticks`: full rate)
    with open(mcap_path, "rb") as f:
        for schema, channel, message in iter_mcap_messages(f):
            topic = channel.topic
            if (channel.message_encoding == "protobuf"
                    and schema.name in _IMU):
                imu.setdefault(topic, []).append(
                    _IMU[schema.name].FromString(message.data))
                continue
            if channel.message_encoding == "raw":
                ticks.setdefault(message.log_time, {})[topic] = message.data
            elif (channel.message_encoding == "protobuf"
                  and schema.name in _NAVD):
                msg = _NAVD[schema.name].FromString(message.data)
                ticks.setdefault(message.log_time, {})[topic] = {
                    f.name: getattr(msg, f.name) for f in msg.DESCRIPTOR.fields}
            elif ((channel.message_encoding == "protobuf"
                   and schema.name in _PROTO)
                  or topic in IMAGE_TOPICS):
                raw, fmt = _foxglove_blob(
                    schema.name, channel.message_encoding, message.data)
                if fmt in VIDEO_FORMATS and topic in COLOR_ROLE:
                    video_aus[COLOR_ROLE[topic]].append(
                        (message.log_time, fmt, raw))
                ticks.setdefault(message.log_time, {})[topic] = raw
            else:
                ticks.setdefault(message.log_time, {})[topic] = \
                    json.loads(message.data)

    for topic, samples in sorted(imu.items()):
        kind = "accel" if "accel" in topic else "gyro"
        role = topic.rsplit("_", 1)[-1]
        fld = "a" if kind == "accel" else "g"
        np.savez_compressed(
            out / f"imu_{kind}_{role}.npz",
            device_us=np.array([s.device_stamp_us for s in samples], np.int64),
            stamp_ns=np.array([s.stamp_ns for s in samples], np.int64),
            x=np.array([getattr(s, f"{fld}x") for s in samples], np.float64),
            y=np.array([getattr(s, f"{fld}y") for s in samples], np.float64),
            z=np.array([getattr(s, f"{fld}z") for s in samples], np.float64))

    # Hardware-encoded color (H.265/H.264, CompressedVideo): decode the AU
    # stream in order and align each decoded frame with the log_time of the
    # AU that produced it. The recorder tags every AU with its source tick,
    # so this reproduces the exact per-tick color frame.
    decoded_color = _decode_video(aus_by_role=video_aus) \
        if any(video_aus.values()) else {}

    stamps = sorted(t for t, chans in ticks.items() if "/depth_near" in chans)
    manifest = []
    for stamp in stamps:
        chans = ticks[stamp]
        # attach the nearest state messages within tolerance
        def nearest(topic):
            best, best_dt = None, tol_us + 1
            for t, chans2 in ticks.items():
                if topic in chans2 and abs(t - stamp) < best_dt:
                    best, best_dt = chans2[topic], abs(t - stamp)
            return best
        cmd = nearest("/cmd_vel") or {"vx": 0.0, "wz": 0.0}
        odom = nearest("/odom") or {"x": 0.0, "y": 0.0, "theta": 0.0}
        goal = nearest("/goal") or {"type": "none"}
        bev = nearest("/bev_teacher") or {}
        calib = next((c["/calib"] for c in ticks.values() if "/calib" in c), {})
        stamp_s = f"{stamp:020d}"
        cv2 = __import__("cv2")
        for role, sub in (("near", "color"), ("far", "color_far")):
            bgr = decoded_color.get(role, {}).get(stamp)
            if bgr is not None:
                ok, jpg = cv2.imencode(
                    ".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                (out / sub / f"{stamp_s}.jpg").write_bytes(
                    jpg.tobytes() if ok else b"")
            elif not video_aus[role]:
                # legacy path: already JPEG bytes
                (out / sub / f"{stamp_s}.jpg").write_bytes(
                    chans.get(f"/color_{role}", b""))
        depth = {}
        for role in ("near", "far"):
            data = chans.get(f"/depth_{role}")
            if data:
                arr = np.frombuffer(data, np.uint8)
                depth[role] = np.asarray(
                    __import__("cv2").imdecode(arr, __import__("cv2").IMREAD_UNCHANGED))
        np.savez_compressed(out / "depth" / f"{stamp_s}.npz", **depth)
        # no BEV channel -> no recorded teacher; the npz is still written
        # (fuse_navd_labels fills it with `fused` from SAM x depth)
        payload = {}
        if bev.get("raw"):
            payload["teacher"] = np.frombuffer(
                base64.b64decode(bev["raw"]), np.uint8).reshape(60, 60)
        np.savez_compressed(out / "labels" / f"{stamp_s}.npz", **payload)
        row = {"stamp_ns": stamp, "dir": stamp_s,
               "cmd_vel": cmd, "odom": odom, "goal": goal,
               "plane_ok": bev.get("plane_ok", {}),
               "has_color": "/color_near" in chans,
               "has_color_far": "/color_far" in chans,
               "calib": calib}
        manifest.append(row)
    with open(out / "manifest.jsonl", "w") as f:
        for row in manifest:
            f.write(json.dumps(row) + "\n")
    print(f"{len(manifest)} ticks -> {out}")
    return manifest


# ---------------------------------------------------------------------------
# bev-student layout
# ---------------------------------------------------------------------------

def _decode_video_jpegs(aus_by_role, quality=90):
    """Decode H.264/H.265 access units to JPEG bytes, keyed by log_time.

    The recorder emits one AU per tick.  PyAV may buffer/reorder, so decoded
    frames are paired with the AU list by index (the same convention as the
    legacy extractor); any mismatch is reported and the tail is dropped.
    """
    import av
    import cv2

    out = {}
    for role, items in aus_by_role.items():
        if not items:
            continue
        items = sorted(items, key=lambda x: x[0])
        codec = {"h265": "hevc", "hevc": "hevc",
                 "h264": "h264"}[items[0][1]]
        cc = av.CodecContext.create(codec, "r")
        jpgs = []

        def encode(frame):
            bgr = frame.to_ndarray(format="bgr24")
            ok, jpg = cv2.imencode(
                ".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            if ok:
                jpgs.append(jpg.tobytes())

        for _, _, au in items:
            for fr in cc.decode(av.Packet(au)):
                encode(fr)
        for fr in cc.decode(None):
            encode(fr)
        n = min(len(jpgs), len(items))
        if n != len(items):
            print(f"[extract] {role}: decoded {len(jpgs)} frames for "
                  f"{len(items)} AUs; dropping {len(items) - n}",
                  flush=True)
        out[role] = {items[i][0]: jpgs[i] for i in range(n)}
    return out


def _sample_stamps(stamps, sample_hz):
    """Pick the nearest source tick to each target on a fixed Hz grid.

    The recorder writes at the camera/frame-set rate (~21 Hz on these
    sessions), not at exactly 10 Hz.  A minimum-interval filter undersamples
    to ~8 Hz; snapping to a 100 ms grid gives the intended 10 Hz and keeps
    the nearest real frame (never an interpolated one).
    """
    if not stamps:
        return []
    step_ns = int(round(1e9 / max(float(sample_hz), 1e-3)))
    out = []
    j = 0
    target = stamps[0]
    last = stamps[-1]
    while target <= last:
        while (j + 1 < len(stamps)
               and abs(stamps[j + 1] - target) <= abs(stamps[j] - target)):
            j += 1
        stamp = stamps[j]
        if not out or stamp != out[-1]:
            out.append(stamp)
        target += step_ns
    return out


def _nearest_series(series, target, tol_ns):
    """Nearest value in a sorted [(stamp, value)] list within tol_ns."""
    if not series:
        return None
    stamps = [s for s, _ in series]
    i = bisect.bisect_left(stamps, target)
    best = None
    best_dt = tol_ns + 1
    for j in (i - 1, i):
        if 0 <= j < len(series):
            dt = abs(series[j][0] - target)
            if dt < best_dt:
                best, best_dt = series[j][1], dt
    return best


def extract_bev_student(mcap_path, out_dir, *, sample_hz=10.0,
                        force=False, config_dir=None, tol_ns=15_000_000,
                        jpeg_quality=90):
    """Unpack one MCAP into the bev-student training layout.

    Per sampled tick::

        color_near/<stamp>.jpg   color_far/<stamp>.jpg
        depth_near/<stamp>.png   depth_far/<stamp>.png  (uint16 mm)
        index.jsonl
        calib.json
        meta.json

    ``labels/`` is written later by ``tools/bev_student_labels.py``.
    """
    import cv2
    from bebop_vision.orbbec import (CONFIG_DIR, load_intrinsics,
                                     load_rig_config)

    mcap_path = Path(mcap_path)
    out = Path(out_dir)
    index_path = out / "index.jsonl"
    if index_path.exists() and not force:
        print(f"[extract] {out} already has index.jsonl (use --force)")
        return None
    for sub in ("color_near", "color_far", "depth_near", "depth_far"):
        d = out / sub
        d.mkdir(parents=True, exist_ok=True)
        if force:
            # Drop frames from a previous sampling pass; labels are owned by
            # tools/bev_student_labels.py and are left alone.
            for p in d.glob("*"):
                if p.is_file():
                    p.unlink()

    IMAGE_TOPICS = ("/color_near", "/color_far", "/depth_near", "/depth_far")
    COLOR_ROLE = {"/color_near": "near", "/color_far": "far"}
    VIDEO_FORMATS = ("h264", "h265", "hevc")
    STATE_TOPICS = ("/cmd_vel", "/odom", "/goal", "/bev_teacher")

    ticks = {}
    video_aus = {"near": [], "far": []}
    direct_color = {"near": {}, "far": {}}
    states = {t: [] for t in STATE_TOPICS}
    calib = None
    with open(mcap_path, "rb") as f:
        for schema, channel, message in iter_mcap_messages(f):
            topic = channel.topic
            log_time = int(message.log_time)
            schema_name = getattr(schema, "name", None)
            if topic == "/calib":
                try:
                    calib = json.loads(message.data)
                except (TypeError, json.JSONDecodeError):
                    pass
                continue
            if (channel.message_encoding == "protobuf"
                    and schema_name in _NAVD):
                msg = _NAVD[schema_name].FromString(message.data)
                payload = {fld.name: getattr(msg, fld.name)
                           for fld in msg.DESCRIPTOR.fields}
                if topic in states:
                    states[topic].append((log_time, payload))
                ticks.setdefault(log_time, {})[topic] = payload
                continue
            if (channel.message_encoding == "protobuf"
                    and schema_name in _PROTO) or topic in IMAGE_TOPICS:
                raw, fmt = _foxglove_blob(
                    schema_name, channel.message_encoding, message.data)
                if fmt in VIDEO_FORMATS and topic in COLOR_ROLE:
                    role = COLOR_ROLE[topic]
                    video_aus[role].append((log_time, fmt, raw))
                    ticks.setdefault(log_time, {})[topic] = b""
                elif topic in COLOR_ROLE:
                    direct_color[COLOR_ROLE[topic]][log_time] = raw
                    ticks.setdefault(log_time, {})[topic] = raw
                else:
                    ticks.setdefault(log_time, {})[topic] = raw
                continue
            if channel.message_encoding == "raw":
                ticks.setdefault(log_time, {})[topic] = message.data
            else:
                try:
                    ticks.setdefault(log_time, {})[topic] = \
                        json.loads(message.data)
                except (TypeError, ValueError):
                    # Unknown protobuf/raw channels (e.g. IMU) are not
                    # needed for the BEV dataset; ignore undecodable ones.
                    pass

    decoded = _decode_video_jpegs(video_aus, quality=jpeg_quality)
    for role in ("near", "far"):
        decoded.setdefault(role, {}).update(direct_color[role])
    for t in states.values():
        t.sort(key=lambda x: x[0])

    stamps = sorted(t for t, ch in ticks.items()
                    if "/depth_near" in ch and "/depth_far" in ch)
    sampled = _sample_stamps(stamps, sample_hz)

    cfg = load_rig_config()["robots"]["default"]
    intr_by_serial = {}
    for serial in cfg.get("cameras", {}):
        try:
            intr_by_serial[serial] = load_intrinsics(
                serial, config_dir or CONFIG_DIR)
        except FileNotFoundError:
            pass
    calib_out = {
        "recorded": calib,
        "intrinsics": intr_by_serial,
        "mounts": (calib or {}).get("mounts", {}),
        "master_bev": cfg.get("master_bev"),
        "bev_student": cfg.get("bev_student"),
        "source_mcap": str(mcap_path),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(out / "calib.json", "w") as f:
        json.dump(calib_out, f, indent=2)

    rows = []
    skipped_color = skipped_depth = 0
    for stamp in sampled:
        chans = ticks[stamp]
        stamp_s = f"{stamp:020d}"
        jpg_near = decoded.get("near", {}).get(stamp)
        jpg_far = decoded.get("far", {}).get(stamp)
        if jpg_near is None or jpg_far is None:
            skipped_color += 1
            continue
        depths = {}
        bad_depth = False
        for role in ("near", "far"):
            png = chans.get(f"/depth_{role}")
            if not png:
                bad_depth = True
                break
            img = cv2.imdecode(np.frombuffer(png, np.uint8),
                               cv2.IMREAD_UNCHANGED)
            if img is None or img.ndim != 2:
                bad_depth = True
                break
            if img.dtype != np.uint16:
                img = img.astype(np.uint16)
            depths[role] = img
        if bad_depth:
            skipped_depth += 1
            continue

        (out / "color_near" / f"{stamp_s}.jpg").write_bytes(jpg_near)
        (out / "color_far" / f"{stamp_s}.jpg").write_bytes(jpg_far)
        for role, depth in depths.items():
            ok = cv2.imwrite(
                str(out / f"depth_{role}" / f"{stamp_s}.png"), depth,
                [int(cv2.IMWRITE_PNG_COMPRESSION), 1])
            if not ok:
                raise RuntimeError(f"failed to write depth_{role}/{stamp_s}.png")

        odom = _nearest_series(states["/odom"], stamp, tol_ns) or {}
        cmd = _nearest_series(states["/cmd_vel"], stamp, tol_ns) or {}
        goal = _nearest_series(states["/goal"], stamp, tol_ns) or {}
        bev = _nearest_series(states["/bev_teacher"], stamp, tol_ns) or {}
        rows.append({
            "stamp_ns": stamp,
            "stamp_us": stamp // 1000,
            "session": out.name,
            "frame": len(rows),
            "color_near": f"color_near/{stamp_s}.jpg",
            "color_far": f"color_far/{stamp_s}.jpg",
            "depth_near": f"depth_near/{stamp_s}.png",
            "depth_far": f"depth_far/{stamp_s}.png",
            "label": None,
            "pose": {"x": float(odom.get("x", 0.0)),
                     "y": float(odom.get("y", 0.0)),
                     "theta": float(odom.get("theta", 0.0))},
            "cmd_vel": {"vx": float(cmd.get("vx", 0.0)),
                        "wz": float(cmd.get("wz", 0.0))},
            "goal": goal or {"type": "none"},
            "plane_ok": bev.get("plane_ok", {}) if isinstance(bev, dict) else {},
            "split": None,
            "has_color": True,
            "camera_pair_delta_ms": None,
        })

    with open(index_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    span_s = ((stamps[-1] - stamps[0]) / 1e9
              if len(stamps) > 1 else 0.0)
    source_hz = ((len(stamps) - 1) / span_s) if span_s > 0 else 0.0
    extracted_hz = ((len(rows) - 1) / span_s) if span_s > 0 else 0.0
    meta = {
        "session": out.name,
        "source_mcap": str(mcap_path),
        "source_size": int(mcap_path.stat().st_size),
        "extracted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "sample_hz": float(sample_hz),
        "frames": len(rows),
        "ticks_total": len(stamps),
        "duration_s": round(span_s, 3),
        "source_tick_hz": round(source_hz, 3),
        "extracted_hz": round(extracted_hz, 3),
        "skipped_missing_color": skipped_color,
        "skipped_missing_depth": skipped_depth,
        "roles": ["near", "far"],
    }
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[extract] {out.name}: {len(rows)}/{len(stamps)} ticks "
          f"@ {sample_hz:.1f} Hz -> {out}", flush=True)
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mcap", nargs="?", help="input MCAP (legacy or new layout)")
    ap.add_argument("out", nargs="?", help="output directory")
    ap.add_argument("--layout", choices=("navd-v0", "bev-student"),
                    default="navd-v0",
                    help="output layout (default: legacy navd-v0)")
    ap.add_argument("--sample-hz", type=float, default=10.0,
                    help="bev-student sampling rate (default: 10)")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing bev-student index")
    args = ap.parse_args(argv)
    if not args.mcap or not args.out:
        ap.error("mcap and out are required")
    if args.layout == "bev-student":
        extract_bev_student(args.mcap, args.out, sample_hz=args.sample_hz,
                            force=args.force)
    else:
        extract(args.mcap, args.out)


if __name__ == "__main__":
    main()
