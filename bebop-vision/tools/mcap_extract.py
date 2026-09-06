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

try:
    from mcap.reader import make_reader
except ImportError as exc:  # pragma: no cover
    raise ImportError("pip install mcap") from exc


def extract(mcap_path, out_dir, tol_us=15_000):
    out = Path(out_dir)
    for sub in ("color", "color_far", "depth", "labels"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    IMAGE_TOPICS = ("/color_near", "/color_far", "/depth_near", "/depth_far")
    ticks = {}  # log_us -> {topic: decoded payload}
    with open(mcap_path, "rb") as f:
        for schema, channel, message in make_reader(f).iter_messages():
            topic = channel.topic
            if channel.message_encoding == "raw":
                ticks.setdefault(message.log_time, {})[topic] = message.data
            else:
                payload = json.loads(message.data)
                # Foxglove CompressedImage -> raw codec bytes for storage
                if topic in IMAGE_TOPICS and isinstance(payload, dict):
                    payload = base64.b64decode(payload["data"])
                ticks.setdefault(message.log_time, {})[topic] = payload

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
        (out / "color" / f"{stamp_s}.jpg").write_bytes(
            chans.get("/color_near", b""))
        (out / "color_far" / f"{stamp_s}.jpg").write_bytes(
            chans.get("/color_far", b""))
        depth = {}
        for role in ("near", "far"):
            data = chans.get(f"/depth_{role}")
            if data:
                arr = np.frombuffer(data, np.uint8)
                depth[role] = np.asarray(
                    __import__("cv2").imdecode(arr, __import__("cv2").IMREAD_UNCHANGED))
        np.savez_compressed(out / "depth" / f"{stamp_s}.npz", **depth)
        teacher = None
        if bev.get("raw"):
            teacher = np.frombuffer(base64.b64decode(bev["raw"]), np.uint8).reshape(60, 60)
        np.savez_compressed(out / "labels" / f"{stamp_s}.npz",
                            teacher=teacher)
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
