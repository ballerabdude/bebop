"""Unpack a navd MCAP session into the navd-v0 training layout.

On the workstation, after scp'ing the session file off the robot:

    python tools/mcap_extract.py session.mcap datasets/navd-v0/session01

Produces (per aligned tick — all channels of a tick share log_time):
    color/{stamp}.jpg          PE camera frame (as recorded, JPEG)
    color_far/{stamp}.jpg      ED camera frame (as recorded, JPEG)
    depth/{stamp}.npz          near, far  (uint16 mm)
    labels/{stamp}.npz         teacher (uint8 60x60) — pre-fill for hand labeling
    manifest.jsonl             stamp_ns, cmd_vel, odom, goal, plane_ok, paths

Hand labels overwrite `labels/{stamp}.npz`'s `hand` array (same 60x60
uint8 semantics: 0 navigable, 1 blocked, 2 caution); training prefers
`hand` when present and falls back to `teacher`.
"""

import base64
import json
import sys
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
except ImportError:  # pragma: no cover - standalone use
    _PROTO = {}
    _NAVD = {}

try:
    from mcap.reader import make_reader
except ImportError as exc:  # pragma: no cover
    raise ImportError("pip install mcap") from exc


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
    with open(mcap_path, "rb") as f:
        for schema, channel, message in make_reader(f).iter_messages():
            topic = channel.topic
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


if __name__ == "__main__":
    extract(sys.argv[1], sys.argv[2])
