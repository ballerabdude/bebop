"""Probe: enumerate + validate the Orbbec camera IMU (accel/gyro) on the navd rig.

The Gemini 335Lg carries an IMU that pyorbbecsdk exposes as Accel/Gyro
streams. This tool answers the two questions that decide whether it is
usable as a VIO input — and makes explicit that it is NOT pose ground
truth (an IMU drifts; validate trajectories against tags/mocap instead):

  * what the device advertises: sample rates, full-scale ranges, noise/bias
    intrinsics, and whether the timestamp domain is global;
  * how the IMU timestamps line up with the depth frames: the IMU<->camera
    time offset VIO needs, reported in the shared device-clock domain.

Idle-safe: it refuses to run while the recorder / goal-drive holds the
cameras (one process per camera). Stop that first, or pass --force.

Run on the Jetson with no other camera user:
  .venv/bin/python tools/orbbec_imu_probe.py                 # both cameras, 10 s
  .venv/bin/python tools/orbbec_imu_probe.py --no-stream     # enumerate only
  .venv/bin/python tools/orbbec_imu_probe.py --seconds 20 --role near
"""

import argparse
import bisect
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from bebop_vision.orbbec import (_negotiate_depth_profile, _sdk,  # noqa: E402
                                 load_rig_config)

BUSY_RE = "record-nav[d]|main.p[y]"
DEFAULT_DEPTH_PROFILE = (848, 480, 30)


def _busy():
    p = subprocess.run(["pgrep", "-af", BUSY_RE], capture_output=True, text=True)
    return p.returncode == 0, p.stdout.strip()


def _rate_hz(ob, rate):
    try:
        return float(ob.convert_imu_sample_rate_to_value(rate))
    except Exception:
        return None


def _rate_enum(ob, rate):
    try:
        return ob.convert_imu_sample_rate_value_to_type(
            ob.convert_imu_sample_rate_to_value(rate))
    except Exception:
        return rate


def _ts(frame, attr):
    try:
        v = int(getattr(frame, attr)())
        return v if v > 0 else None
    except Exception:
        return None


def _fmt(v):
    if v is None:
        return "?"
    if hasattr(v, "x") and hasattr(v, "y") and hasattr(v, "z"):
        return f"({v.x:+.5g},{v.y:+.5g},{v.z:+.5g})"
    if isinstance(v, (int, float)):
        return f"{v:.5g}"
    if isinstance(v, (list, tuple)):
        return "[" + ",".join(_fmt(x) for x in v) + "]"
    return str(v)


def _profiles(ob, sensor, kind):
    rows = []
    plist = sensor.get_stream_profile_list()
    for i in range(plist.get_count()):
        p = plist.get_stream_profile_by_index(i)
        if kind == "accel":
            if not p.is_accel_stream_profile():
                continue
            p = p.as_accel_stream_profile()
            fs = ob.convert_accel_full_scale_range_to_string(p.get_full_scale_range())
        else:
            if not p.is_gyro_stream_profile():
                continue
            p = p.as_gyro_stream_profile()
            fs = ob.convert_gyro_full_scale_range_to_string(p.get_full_scale_range())
        rows.append({"profile": p, "hz": _rate_hz(ob, p.get_sample_rate()),
                     "fs": fs, "fmt": str(p.get_format()).split(".")[-1],
                     "fs_enum": p.get_full_scale_range(),
                     "rate_enum": _rate_enum(ob, p.get_sample_rate())})
    return rows


def _pick(rows, want_hz):
    if not rows:
        return None
    usable = [r for r in rows if r["hz"]]
    if not usable:
        return rows[0]
    if want_hz:
        le = [r for r in usable if r["hz"] <= want_hz]
        if le:
            return max(le, key=lambda r: r["hz"])
    return max(usable, key=lambda r: r["hz"])


def _find_sensor(ob, dev, kind):
    target = {"accel": ob.OBSensorType.ACCEL_SENSOR,
              "gyro": ob.OBSensorType.GYRO_SENSOR,
              "depth": ob.OBSensorType.DEPTH_SENSOR}[kind]
    for i in range(dev.get_sensor_list().get_count()):
        s = dev.get_sensor_list().get_sensor_by_index(i)
        if s.get_type() == target:
            return s
    return None


def _enumerate(ob, dev, role, serial):
    info = dev.get_device_info()
    print(f"=== {role} ({serial}) ===")
    print(f"  name={info.get_name()} pid=0x{info.get_pid():04x} "
          f"fw={info.get_firmware_version()}")
    kinds = [str(s.get_type()).split(".")[-1]
             for s in (dev.get_sensor_list().get_sensor_by_index(i)
                       for i in range(dev.get_sensor_list().get_count()))]
    print(f"  sensors: {kinds}")
    try:
        print(f"  global_timestamp_supported: {dev.is_global_timestamp_supported()}")
    except Exception as exc:
        print(f"  global_timestamp_supported: ? ({exc})")

    picked = {}
    for kind in ("accel", "gyro"):
        sensor = _find_sensor(ob, dev, kind)
        if sensor is None:
            print(f"  [{kind}] no sensor")
            continue
        rows = _profiles(ob, sensor, kind)
        print(f"  [{kind}] {len(rows)} profile(s):")
        for r in rows:
            print(f"      {r['hz']} Hz  fs={r['fs']}  fmt={r['fmt']}")
        best = _pick(rows, None)
        picked[kind] = best
        if best is not None:
            try:
                intr = best["profile"].get_intrinsic()
                print(f"      intrinsic@{best['hz']} Hz: bias={_fmt(intr.bias)} "
                      f"noise_density={_fmt(intr.noise_density)} "
                      f"random_walk={_fmt(intr.random_walk)} "
                      f"ref_temp={_fmt(intr.reference_temp)}")
            except Exception as exc:
                print(f"      intrinsic unavailable: {exc}")
    return picked


def _stream(ob, dev, role, accel_row, gyro_row, seconds, depth_profile):
    config = ob.Config()
    if accel_row is not None:
        config.enable_accel_stream(accel_row["fs_enum"], accel_row["rate_enum"])
    if gyro_row is not None:
        config.enable_gyro_stream(gyro_row["fs_enum"], gyro_row["rate_enum"])
    depth_sensor = _find_sensor(ob, dev, "depth")
    if depth_sensor is not None:
        w, h, fps = _negotiate_depth_profile(depth_sensor, ob, depth_profile)
        config.enable_video_stream(ob.OBStreamType.DEPTH_STREAM, w, h, fps,
                                   ob.OBFormat.Y16)

    pipe = ob.Pipeline(dev)
    pipe.start(config)
    accel, gyro, depth = [], [], []
    t_end = time.monotonic() + seconds
    try:
        while time.monotonic() < t_end:
            fs = pipe.wait_for_frames(100)
            if fs is None:
                continue
            af = fs.get_accel_frame()
            if af is not None:
                v = af.get_value()
                accel.append((_ts(af, "get_timestamp_us"),
                              _ts(af, "get_global_timestamp_us"),
                              _ts(af, "get_system_timestamp_us"),
                              time.monotonic(), float(v.x), float(v.y), float(v.z)))
            gf = fs.get_gyro_frame()
            if gf is not None:
                v = gf.get_value()
                gyro.append((_ts(gf, "get_timestamp_us"),
                             _ts(gf, "get_global_timestamp_us"),
                             _ts(gf, "get_system_timestamp_us"),
                             time.monotonic(), float(v.x), float(v.y), float(v.z)))
            df = fs.get_depth_frame()
            if df is not None:
                depth.append(_ts(df, "get_timestamp_us"))
    finally:
        pipe.stop()
    _report(role, accel_row, gyro_row, accel, gyro, depth)


def _rate(rows):
    ts = [r[0] for r in rows if r[0] is not None]
    if len(ts) < 2:
        return None
    span = ts[-1] - ts[0]
    return (len(ts) - 1) * 1e6 / span if span > 0 else None


def _norm(row):
    return (row[4] ** 2 + row[5] ** 2 + row[6] ** 2) ** 0.5


def _offset_to_depth(rows, depth):
    if not rows or not depth:
        return None
    offs = []
    for r in rows:
        if r[0] is None:
            continue
        j = bisect.bisect_left(depth, r[0])
        cand = [depth[k] for k in (j - 1, j) if 0 <= k < len(depth)]
        if cand:
            offs.append(r[0] - min(cand, key=lambda t: abs(t - r[0])))
    return offs or None


def _report(role, accel_row, gyro_row, accel, gyro, depth):
    def label(row, kind):
        return f"{kind}@{row['hz']:.0f}Hz" if row else kind
    print(f"--- stream [{role}] {len(accel)} accel, {len(gyro)} gyro, "
          f"{len(depth)} depth frames ---")
    for rows, kind, row in ((accel, "accel", accel_row), (gyro, "gyro", gyro_row)):
        if not rows:
            print(f"  [{label(row, kind)}] no frames")
            continue
        hz = _rate(rows)
        mag = statistics.median(_norm(r) for r in rows)
        unit = "m/s^2" if kind == "accel" else "rad/s"
        print(f"  [{label(row, kind)}] observed={hz:.2f} Hz  "
              f"|{kind}| median={mag:.4g} {unit}")
        gts = [r[1] for r in rows if r[1] is not None]
        sts = [r[2] for r in rows if r[2] is not None]
        print(f"      global_ts={'yes' if gts else 'no'} "
              f"system_ts={'yes' if sts else 'no'}")
        ts = [r[0] for r in rows if r[0] is not None]
        if len(ts) >= 2 and rows[-1][3] > rows[0][3] and ts[-1] != ts[0]:
            host = (rows[-1][3] - rows[0][3]) * 1e6
            dev = ts[-1] - ts[0]
            print(f"      host/device clock ratio={host / dev:.6f} "
                  f"(1.0 = device clock tracks host)")
        offs = _offset_to_depth(rows, depth)
        if offs:
            s = sorted(offs)
            print(f"      offset to depth (device-clock): median={s[len(s)//2]/1e3:+.2f} ms "
                  f"p5={s[int(0.05*len(s))]/1e3:+.2f} "
                  f"p95={s[int(0.95*len(s))]/1e3:+.2f} ms")
    if accel:
        print("  NOTE: ~9.81 m/s^2 |accel| at rest confirms a valid accel frame; "
              "the IMU is a VIO input, not trajectory ground truth.")


def probe_device(ob, dev, role, serial, args):
    picked = _enumerate(ob, dev, role, serial)
    if args.no_stream:
        return
    accel = picked.get("accel")
    gyro = picked.get("gyro")
    if accel is None and gyro is None:
        print(f"  [{role}] no IMU streams to sample")
        return
    if accel is not None and args.hz:
        rows = _profiles(ob, _find_sensor(ob, dev, "accel"), "accel")
        accel = _pick(rows, args.hz) or accel
    if gyro is not None and args.hz:
        rows = _profiles(ob, _find_sensor(ob, dev, "gyro"), "gyro")
        gyro = _pick(rows, args.hz) or gyro
    _stream(ob, dev, role, accel, gyro, args.seconds, args.depth_profile)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=10.0,
                    help="stream duration per camera (default 10)")
    ap.add_argument("--role", choices=["near", "far"], help="probe one camera")
    ap.add_argument("--hz", type=float, default=200.0,
                    help="preferred IMU sample rate (default 200)")
    ap.add_argument("--depth-profile", type=lambda s: tuple(int(x) for x in s.split("x")),
                    default=DEFAULT_DEPTH_PROFILE, metavar="WxHxFPS",
                    help="depth profile for the IMU<->depth offset check")
    ap.add_argument("--no-stream", action="store_true",
                    help="enumerate profiles/intrinsics only")
    ap.add_argument("--force", action="store_true",
                    help="skip the camera-busy guard (accepts a conflict)")
    args = ap.parse_args()

    if not args.force:
        busy, who = _busy()
        if busy:
            raise SystemExit(
                "camera busy — recorder/goal-drive is running; stop it first "
                "(sudo pkill -TERM -f 'record-nav[d]'):\n" + who)

    ob = _sdk()
    ob.Context.set_logger_level(ob.OBLogLevel.ERROR)
    cfg = load_rig_config()
    cams = cfg["robots"]["default"]["cameras"]
    ctx = ob.Context()
    for serial, c in cams.items():
        role = c["role"]
        if args.role and role != args.role:
            continue
        dev = ctx.query_devices().get_device_by_serial_number(serial)
        if dev is None:
            print(f"=== {role} ({serial}) === NOT FOUND "
                  f"(close OrbbecViewer / other users)")
            continue
        probe_device(ob, dev, role, serial, args)


if __name__ == "__main__":
    main()