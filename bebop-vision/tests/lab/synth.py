"""Synthetic drive sessions for the lab tests.

Writes an MCAP with the recorder's exact channel set and schemas
(``bebop_vision/recorder_mcap.py``) from a *known* trajectory, so tests can
check alignment, decoding and labels against ground truth instead of
against another reader.

    path = write_synthetic_session(tmp_path / "navd_session_20260101_000000.mcap",
                                   duration_s=4.0, hz=20.0, codec="h265")

Trajectory: constant forward speed ``v`` with a yaw rate ``w`` (an arc), so
``odom`` is an exact circle segment and the gyro z-axis in the *optical*
frame reproduces ``w`` after the mount rotation. Colour frames are a moving
gradient with the tick index painted into the top-left 8x8 block (grey level ``frame index * 8``), so a decoded frame can be matched to its
tick. Depth is a flat floor at 1500 mm with the tick index in pixel (0, 0).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

from bebop_vision import foxglove_proto, navd_proto
from bebop_vision.proto.bebop.runtime.v1 import bebop_runtime_pb2 as pb
from bebop_vision.lab.types import mount_rotation

SERIALS = {"near": "S-NEAR", "far": "S-FAR"}
MOUNTS = {"near": {"height_m": 1.27, "pitch_deg": -60.0, "yaw_deg": 0.0},
          "far": {"height_m": 1.32, "pitch_deg": -9.0, "yaw_deg": 0.0}}
COLOR_WH = (128, 80)      # small colour frames keep the encoder fast
DEPTH_WH = (106, 60)      # same 848:480 aspect, 1/8 scale
T0_NS = 1_792_800_000_000_000_000  # 2026-10-24-ish, ns since the epoch


def _obj_schema(properties):
    return json.dumps({"type": "object", "properties": properties}).encode()


def colour_frame(i: int, w: int = COLOR_WH[0], h: int = COLOR_WH[1]) -> np.ndarray:
    """BGR uint8 frame for tick ``i``: a gradient that slides with i, with
    the tick painted into the top-left 16x16 block as grey (i * 8) % 256."""
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.zeros((h, w, 3), np.uint8)
    img[..., 0] = ((xx * 2 + i * 3) % 256).astype(np.uint8)
    img[..., 1] = ((yy * 3) % 256).astype(np.uint8)
    img[..., 2] = ((xx + yy + i) % 256).astype(np.uint8)
    # grey marker block: luma survives 4:2:0 chroma subsampling and CRF
    img[0:16, 0:16, :] = (i * 8) % 256
    return img


def tick_of_colour_frame(img: np.ndarray) -> int:
    """Inverse of :func:`colour_frame`'s marker (mod 32 ticks); tolerant to
    lossy encoding by rounding to the nearest multiple of 4."""
    grey = float(np.asarray(img)[3:13, 3:13, :].mean())
    return int(round(grey / 8.0)) % 32


def depth_frame(i: int, w: int = DEPTH_WH[0], h: int = DEPTH_WH[1]) -> np.ndarray:
    d = np.full((h, w), 1500, np.uint16)
    d[0, 0] = i
    return d


def _encode_video(frames, codec: str):
    """Encode BGR frames with PyAV's software encoder; one AU per frame, in
    order (B-frames disabled so decode order == display order)."""
    import av
    enc_name = {"h265": "libx265", "h264": "libx264"}[codec]
    h, w = frames[0].shape[:2]
    cc = av.CodecContext.create(enc_name, "w")
    cc.width, cc.height = w, h
    cc.pix_fmt = "yuv420p"
    from fractions import Fraction
    cc.time_base = Fraction(1, 30)
    cc.framerate = Fraction(30, 1)
    cc.gop_size = 8
    opts = {"preset": "ultrafast", "tune": "zerolatency"}
    if codec == "h265":
        # repeat VPS/SPS/PPS before every keyframe (what the recorder's
        # ParameterSetInjector guarantees) and no B-frames
        opts["x265-params"] = "repeat-headers=1:bframes=0:keyint=8:min-keyint=8:scenecut=0:crf=16:log-level=none"
    else:
        opts["x264-params"] = "repeat-headers=1:bframes=0:keyint=8:min-keyint=8:scenecut=0:crf=16:log=0"
    cc.options = opts
    aus = []
    for k, bgr in enumerate(frames):
        frame = av.VideoFrame.from_ndarray(bgr, format="bgr24").reformat(format="yuv420p")
        frame.pts = k
        for pkt in cc.encode(frame):
            aus.append(bytes(pkt))
    for pkt in cc.encode(None):
        aus.append(bytes(pkt))
    if len(aus) != len(frames):
        raise RuntimeError(f"encoder emitted {len(aus)} AUs for {len(frames)} frames")
    return aus


def _intent(fields: Mapping[str, Any], stamp_ns: int) -> bytes:
    """A serialized ``bebop.navd.Intent`` with the fields this module's
    message actually has (``stamp_ns`` defaults to the log time)."""
    known = navd_proto.Intent.DESCRIPTOR.fields_by_name
    kw = {k: v for k, v in dict(fields).items() if k in known}
    kw.setdefault("stamp_ns", int(stamp_ns))
    return navd_proto.Intent(**kw).SerializeToString()


def stop_event(offset_s: float, cause: str, detail: str = "") -> Tuple[float, str, str]:
    """An ``extra_events`` entry for the ``stop`` event ``SessionNotes.note_stop``
    queues into a closing segment: note ``"<cause>"`` or ``"<cause>: <detail>"``."""
    return (float(offset_s), "stop", f"{cause}: {detail}" if detail else cause)


def resume_event(offset_s: float, note: str = "") -> Tuple[float, str, str]:
    """An ``extra_events`` entry for the ``resume`` event ``SessionNotes.resume_run``
    writes as the first event of the next segment, with the operator's note."""
    return (float(offset_s), "resume", note)


def write_synthetic_session(path, *, duration_s: float = 4.0, hz: float = 20.0,
                            codec: Optional[str] = "h265", v: float = 0.3,
                            w: float = 0.4, imu_hz: float = 200.0,
                            with_runtime: bool = True, with_notes: bool = True,
                            drop_far_every: int = 0, reverse_from_s: Optional[float] = None,
                            t0_ns: int = T0_NS, footer: bool = True,
                            intent_kw: Optional[Mapping[str, Any]] = None,
                            extra_intents: Sequence[Tuple[float, Mapping[str, Any]]] = (),
                            extra_events: Sequence[Tuple[float, str, str]] = ()) -> Path:
    """Write a synthetic session MCAP. Returns the path.

    ``codec`` is ``"h265"``, ``"h264"`` (CompressedVideo AUs) or ``None``
    (CompressedImage JPEG, the recorder's no-encoder path).
    ``drop_far_every``: every n-th tick has no far colour (``/color_far``
    message absent for that tick), mimicking the far camera dropping frames.
    ``reverse_from_s``: after this time the robot drives backwards (``vx<0``).
    ``footer=False`` leaves the file without a summary / footer (power cut);
    the last, still-buffered chunk is lost, as on the robot.
    ``intent_kw`` overrides / extends the fields of the first ``/intent``
    (written at the first tick when ``with_notes``), e.g. ``task``,
    ``target``, ``run_id``, ``run_started_ns``; fields the regenerated
    ``navd_proto.Intent`` does not have are dropped, so the helper also runs
    against an older module. ``extra_intents``: ``(seconds after the start,
    fields)`` pairs written as further ``/intent`` rows (a run change or a
    clear mid-segment); each one is the complete message, not a delta.
    ``extra_events``: ``(seconds after the start, kind, note)`` written as
    further ``/event`` rows (a run outcome such as ``arrived``), stamped like
    the recorder stamps them: log time = the event's own ``stamp_ns``.
    """
    import cv2
    from mcap.writer import CompressionType, Writer

    path = Path(path)
    n = int(round(duration_s * hz))
    dt_ns = int(round(1e9 / hz))
    stamps = [t0_ns + i * dt_ns for i in range(n)]

    # --- trajectory (exact arc) -------------------------------------------
    t = np.arange(n) / hz
    vx = np.full(n, v)
    if reverse_from_s is not None:
        vx[t >= reverse_from_s] = -v
    theta = np.cumsum(np.r_[0.0, w / hz * np.ones(n - 1)])
    x = np.cumsum(np.r_[0.0, (vx[:-1] / hz) * np.cos(theta[:-1])])
    y = np.cumsum(np.r_[0.0, (vx[:-1] / hz) * np.sin(theta[:-1])])

    frames = [colour_frame(i) for i in range(n)]
    aus = _encode_video(frames, codec) if codec else None

    with open(path, "wb") as f:
        wr = Writer(f, compression=CompressionType.NONE, chunk_size=256 * 1024)
        wr.start(profile="", library="bebop-lab-synth")
        sch_calib = wr.register_schema("bebop.navd.Calib", "jsonschema", _obj_schema({}))
        sch_twist = wr.register_schema(*navd_proto.schema(navd_proto.Twist))
        sch_odom = wr.register_schema(*navd_proto.schema(navd_proto.Odom))
        sch_img = wr.register_schema(*foxglove_proto.schema(foxglove_proto.CompressedImage))
        sch_vid = wr.register_schema(*foxglove_proto.schema(foxglove_proto.CompressedVideo))
        sch_acc = wr.register_schema(*navd_proto.schema(navd_proto.ImuAccel))
        sch_gyr = wr.register_schema(*navd_proto.schema(navd_proto.ImuGyro))
        ch = {
            "calib": wr.register_channel("/calib", "json", sch_calib),
            "cmd_vel": wr.register_channel("/cmd_vel", "protobuf", sch_twist),
            "odom": wr.register_channel("/odom", "protobuf", sch_odom),
            "color_near": wr.register_channel("/color_near", "protobuf", sch_vid if codec else sch_img),
            "color_far": wr.register_channel("/color_far", "protobuf", sch_vid if codec else sch_img),
            "depth_near": wr.register_channel("/depth_near", "protobuf", sch_img),
            "depth_far": wr.register_channel("/depth_far", "protobuf", sch_img),
        }
        for role in ("near", "far"):
            ch[f"imu_accel_{role}"] = wr.register_channel(f"/imu_accel_{role}", "protobuf", sch_acc)
            ch[f"imu_gyro_{role}"] = wr.register_channel(f"/imu_gyro_{role}", "protobuf", sch_gyr)
        if with_runtime:
            sch_rt = wr.register_schema(*navd_proto.schema(pb.ServerRuntimeMessage))
            ch["runtime"] = wr.register_channel("/runtime", "protobuf", sch_rt)
            ch["runtime_events"] = wr.register_channel("/runtime_events", "protobuf", sch_rt)
        if with_notes:
            for key, topic, cls in (("intent", "/intent", navd_proto.Intent),
                                    ("event", "/event", navd_proto.Event),
                                    ("place", "/place", navd_proto.PlaceSeen)):
                ch[key] = wr.register_channel(topic, "protobuf",
                                              wr.register_schema(*navd_proto.schema(cls)))

        def add(key, data, log_ns):
            wr.add_message(ch[key], log_ns, data, log_ns)

        # calib once, like the recorder
        dw, dh = DEPTH_WH
        calib = {
            "intrinsics": {SERIALS[r]: {"fx": 51.7, "fy": 51.7, "cx": dw / 2, "cy": dh / 2,
                                        "width": dw, "height": dh} for r in SERIALS},
            "mounts": {SERIALS[r]: MOUNTS[r] for r in SERIALS},
            "camera_self_mask_pixels": {r: 0 for r in SERIALS},
            "camera_self_mask_polygons": {r: [] for r in SERIALS},
            "recorded_at": "2026-10-24T00:00:00",
        }
        add("calib", json.dumps(calib).encode(), stamps[0])

        if with_notes:
            first = {"text": "hall -> kitchen", "from_place": "hall", "to_place": "kitchen"}
            first.update(intent_kw or {})
            add("intent", _intent(first, stamps[0]), stamps[0])
            for offset_s, fields in extra_intents:
                ts = t0_ns + int(round(float(offset_s) * 1e9))
                add("intent", _intent(fields, ts), ts)
            for offset_s, kind, note in extra_events:
                ts = t0_ns + int(round(float(offset_s) * 1e9))
                add("event", navd_proto.Event(kind=kind, note=note, stamp_ns=ts).SerializeToString(), ts)

        # IMU: gyro at imu_hz in the optical frame such that the body yaw rate is w
        R = {r: mount_rotation(MOUNTS[r]["pitch_deg"], MOUNTS[r]["yaw_deg"]) for r in SERIALS}
        n_imu = int(duration_s * imu_hz)
        imu_dt_ns = int(round(1e9 / imu_hz))
        for k in range(n_imu):
            ts = t0_ns + k * imu_dt_ns
            tk = k / imu_hz
            wz_body = w if (reverse_from_s is None or True) else w
            for r in SERIALS:
                # body angular velocity (0, 0, w) -> optical = R.T @ body
                g_opt = R[r].T @ np.array([0.0, 0.0, wz_body])
                a_opt = R[r].T @ np.array([0.0, 0.0, 9.81])
                add(f"imu_gyro_{r}", navd_proto.ImuGyro(gx=float(g_opt[0]), gy=float(g_opt[1]),
                                                        gz=float(g_opt[2]), stamp_ns=ts,
                                                        device_stamp_us=int(tk * 1e6)).SerializeToString(), ts)
                add(f"imu_accel_{r}", navd_proto.ImuAccel(ax=float(a_opt[0]), ay=float(a_opt[1]),
                                                          az=float(a_opt[2]), stamp_ns=ts,
                                                          device_stamp_us=int(tk * 1e6)).SerializeToString(), ts)

        for i, ts in enumerate(stamps):
            add("cmd_vel", navd_proto.Twist(vx=float(vx[i]), wz=float(w), stamp_ns=ts).SerializeToString(), ts)
            add("odom", navd_proto.Odom(x=float(x[i]), y=float(y[i]), theta=float(theta[i]),
                                        stamp_ns=ts).SerializeToString(), ts)
            for role in ("near", "far"):
                if role == "far" and drop_far_every and i % drop_far_every == 0 and i > 0:
                    pass
                elif codec:
                    msg = foxglove_proto.CompressedVideo(timestamp=foxglove_proto.timestamp(ts),
                                                         frame_id=f"camera_{role}", data=aus[i],
                                                         format=codec)
                    add(f"color_{role}", msg.SerializeToString(), ts)
                else:
                    ok, jpg = cv2.imencode(".jpg", frames[i], [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                    msg = foxglove_proto.CompressedImage(timestamp=foxglove_proto.timestamp(ts),
                                                         frame_id=f"camera_{role}", data=jpg.tobytes(),
                                                         format="jpeg")
                    add(f"color_{role}", msg.SerializeToString(), ts)
                ok, png = cv2.imencode(".png", depth_frame(i), [int(cv2.IMWRITE_PNG_COMPRESSION), 1])
                msg = foxglove_proto.CompressedImage(timestamp=foxglove_proto.timestamp(ts),
                                                     frame_id=f"camera_{role}", data=png.tobytes(),
                                                     format="png")
                add(f"depth_{role}", msg.SerializeToString(), ts)
            if with_runtime:
                tel = pb.TelemetryFrame(mode=pb.MODE_DIAL_IN, estop_latched=False, host_unix_ms=ts // 1_000_000)
                for name, sign in (("left", -1.0), ("right", 1.0)):
                    wh = tel.wheels.add()
                    wh.name = name
                    wh.armed = True
                    wh.velocity_rad_s = float(vx[i] * 10.0 + sign * w)
                    wh.target_velocity_rad_s = float(vx[i] * 10.0 + sign * w)
                tel.drive.present = True
                tel.drive.operator_stale = False
                tel.drive.has_active_operator = True
                tel.power.present = True
                tel.power.battery_voltage_v = 48.0 - 0.01 * i
                tel.power.total_motor_current_a = 2.0
                tel.power.state_of_charge_pct = 80.0
                add("runtime", pb.ServerRuntimeMessage(telemetry=tel).SerializeToString(), ts + 1_000_000)
                if i == n // 2:
                    add("runtime_events", pb.ServerRuntimeMessage(
                        mode_changed=pb.ModeChanged(mode=pb.MODE_DIAL_IN)).SerializeToString(), ts + 2_000_000)
            if with_notes:
                if i == n // 3:
                    add("event", navd_proto.Event(kind="obstacle", note="sock", stamp_ns=ts).SerializeToString(), ts)
                if i % int(hz) == 0:
                    add("place", navd_proto.PlaceSeen(place_id="p1", name="hall", score=0.91, margin=0.2,
                                                      stamp_ns=ts).SerializeToString(), ts)
        if footer:
            wr.finish()
        else:
            # No finish(): the file ends after the last completed chunk and
            # its message indexes, with no summary / footer, exactly like a
            # capture cut by a power loss (the buffered partial chunk is lost).
            f.flush()
    return path


def expected_trajectory(duration_s=4.0, hz=20.0, v=0.3, w=0.4, reverse_from_s=None):
    """The same arrays :func:`write_synthetic_session` wrote (x, y, theta, vx)."""
    n = int(round(duration_s * hz))
    t = np.arange(n) / hz
    vx = np.full(n, v)
    if reverse_from_s is not None:
        vx[t >= reverse_from_s] = -v
    theta = np.cumsum(np.r_[0.0, w / hz * np.ones(n - 1)])
    x = np.cumsum(np.r_[0.0, (vx[:-1] / hz) * np.cos(theta[:-1])])
    y = np.cumsum(np.r_[0.0, (vx[:-1] / hz) * np.sin(theta[:-1])])
    return {"t": t, "x": x, "y": y, "theta": theta, "vx": vx, "w": w}
