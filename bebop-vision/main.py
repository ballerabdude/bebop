"""bebop-vision data recorder (the data-collection pipeline entry point).

Owns the Orbbec camera rig (exclusivity, docs §2.7), the operator video
server, and the navd MCAP recorder: teleop sessions -> color (both
cameras) + lossless depth + cmd_vel/odom, the dataset every
labeling/modeling approach trains from.

    python main.py --record-navd /var/lib/bebop-captures --auto

Drive the robot from the app (manual teleop only — this file carries no
driving; the model/drive stack lives on the exp/navd-ai branch).
"""

import argparse
import os
import sys
import threading
import time
from pathlib import Path

from bebop_vision.proto.bebop.runtime.v1 import bebop_runtime_pb2 as pb
from bebop_vision.robot import DEFAULT_URL


def _dir_bytes(path):
    return sum(f.stat().st_size for f in Path(path).glob("*.mcap") if f.is_file())


def _prune_sessions(out_dir, budget_bytes):
    """Delete oldest navd session files until they fit the budget.

    Only our own `navd_session_*.mcap` files are ever removed — the
    capture dir may be shared with the firmware's policy captures.
    """
    files = sorted(Path(out_dir).glob("navd_session_*.mcap"),
                   key=lambda f: f.stat().st_mtime)
    total = sum(f.stat().st_size for f in files)
    for f in files:
        if total <= budget_bytes:
            break
        total -= f.stat().st_size
        f.unlink()
        print(f"\n[record-navd] pruned {f.name} (disk budget)")


RECORDER_LOCK = Path("/tmp/navd_recorder.lock")


def _acquire_recorder_lock():
    """Single-instance guard: the Orbbec camera is exclusive, and a stale
    second recorder otherwise dies with a cryptic uvc_open error."""
    if RECORDER_LOCK.exists():
        try:
            pid = int(RECORDER_LOCK.read_text().strip())
            if Path(f"/proc/{pid}").exists():
                raise SystemExit(f"another navd recorder is running (pid {pid}); "
                                 f"stop it first or: kill {pid}")
        except ValueError:
            pass
        RECORDER_LOCK.unlink()
    RECORDER_LOCK.write_text(str(os.getpid()))


def _release_recorder_lock():
    RECORDER_LOCK.unlink(missing_ok=True)


def _drive_active(robot):
    """True while the robot is in a driveable, armed, non-estop state.

    DIAL_IN counts: the app's manual teleop drives in DialIn (armed wheels
    + cmd_vel), which is the data-collection mode.
    """
    st = robot.state
    return (st.connected
            and st.mode in (pb.MODE_DIAL_IN, pb.MODE_RUN_POLICY)
            and not st.estop_latched
            # differential drive: a single armed wheel is a fault (failed
            # enable), not a drivable state — don't record half-armed runs
            and all(st.wheel_armed.values()))


def run_record_navd(args):
    """Recorder v2 (plan §7.1): teleop session(s) -> MCAP file(s).

    Default: one session for --seconds (or until Ctrl-C). With --auto the
    recorder follows the drive state instead: a segment opens when the
    firmware is in a driveable, armed state (you start driving) and closes
    when that ends, rolling over on size/time and pruning the oldest files
    under a disk budget — a mirror of the firmware's own policy-capture
    design. Manual drive = captured data, no SSH per run.

    """
    from bebop_vision.orbbec import OrbbecRig
    from bebop_vision.recorder_mcap import NavdRecorder
    from bebop_vision.videoserver import VideoServer
    from bebop_vision.robot import RobotClient
    import time as _time
    import math

    _acquire_recorder_lock()
    robot = RobotClient(args.robot_url).start()
    if not robot.await_connection(5.0):
        _release_recorder_lock()
        raise SystemExit(f"cannot reach robot runtime at {robot.url}")
    print(f"[record-navd] robot: {robot.describe()}")

    roles = tuple(r.strip() for r in args.roles.split(",") if r.strip())
    rig = OrbbecRig(rig_path=args.rig, color=True, roles=roles or None)
    if not rig.wait_for_pair(timeout=10.0):
        rig.stop()
        raise SystemExit("cameras did not produce fresh frames within 10 s")
    vserver = None
    if not args.no_video_server:
        vserver = VideoServer(rig, port=args.video_port)
        vserver.start()


    out_dir = Path(args.record_navd).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    # Fail fast with a readable message: the root-run firmware recreates the
    # capture dir as root if it ever goes missing, which lands here as a
    # PermissionError mid-run (seen 2026-09-06) unless checked up front.
    if not os.access(out_dir, os.W_OK):
        _release_recorder_lock()
        robot.stop()
        raise SystemExit(
            f"cannot write to {out_dir} "
            f"(owned by {out_dir.owner()}:{out_dir.group()}); fix with "
            f"'sudo chown bebop:bebop {out_dir}' or re-run install-jetson.sh")
    from bebop_vision.orbbec import load_rig_config
    safety = load_rig_config()["robots"]["default"].get("safety", {})
    max_frame_age_s = float(safety.get("max_frame_age_s", 0.3))
    budget = args.disk_budget_gb * 1e9
    rate = args.record_rate if args.record_rate is not None else 10.0
    rec = None
    rec_holder = {"rec": None}   # active segment recorder (set by new_segment)

    def new_segment():
        path = out_dir / f"navd_session_{_time.strftime('%Y%m%d_%H%M%S')}.mcap"
        rec = NavdRecorder(
            rig, robot, path, rate_hz=rate,
            max_frame_age_s=max_frame_age_s)
        rec.start()
        rec_holder["rec"] = rec
        print(f"\n[record-navd] recording -> {path}")
        return rec, path, _time.monotonic()

    def close_segment(rec, path):
        rec.stop()
        if rec_holder["rec"] is rec:
            rec_holder["rec"] = None
        print(f"\n[record-navd] closed {path.name} "
              f"({rec.frames} frames, {rec.bytes_written/1e6:.1f} MB)")
        _prune_sessions(out_dir, budget)

    if not args.auto:
        rec, path, _ = new_segment()
        t0 = _time.monotonic()
        try:
            while True:
                if args.seconds and _time.monotonic() - t0 > args.seconds:
                    break
                _time.sleep(0.2)
                print(f"\r[record-navd] {rec.frames} frames, "
                      f"{rec.bytes_written/1e6:.1f} MB", end="", flush=True)
        except KeyboardInterrupt:
            pass
        finally:
            close_segment(rec, path)
            if vserver is not None:
                vserver.stop()
            rig.stop()
            robot.stop()
            _release_recorder_lock()
        return

    # --auto: follow the drive state; roll segments on size/time.
    max_bytes = args.max_segment_mb * 1e6
    max_s = args.max_segment_min * 60.0
    seg = seg_path = seg_t0 = None
    t_start = _time.monotonic()
    try:
        while True:
            if args.seconds and _time.monotonic() - t_start > args.seconds:
                break
            active = _drive_active(robot)
            rolled = False
            if seg is not None:
                rolled = (seg.bytes_written >= max_bytes
                          or _time.monotonic() - seg_t0 >= max_s)
            if active and seg is None:
                try:
                    seg, seg_path, seg_t0 = new_segment()
                except OSError as exc:
                    print(f"\n[record-navd] cannot open segment: {exc}; "
                          f"waiting (fix the cause, drive state keeps "
                          f"retriggering)")
                    _time.sleep(2.0)
                    active = False
            elif seg is not None and (not active or rolled):
                try:
                    close_segment(seg, seg_path)
                except OSError as exc:
                    print(f"\n[record-navd] error closing segment: {exc}")
                seg = None
                if active and rolled:
                    try:
                        seg, seg_path, seg_t0 = new_segment()
                    except OSError as exc:
                        print(f"\n[record-navd] cannot roll to next segment: "
                              f"{exc}; waiting")
                        _time.sleep(2.0)
                        active = False
            if seg is not None:
                print(f"\r[record-navd] {seg.frames} frames, "
                      f"{seg.bytes_written/1e6:.1f} MB", end="", flush=True)
            else:
                print(f"\r[record-navd] waiting for drive state "
                      f"(DialIn/RunPolicy + armed)...", end="", flush=True)
            _time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        if seg is not None:
            try:
                close_segment(seg, seg_path)
            except OSError as exc:
                print(f"\n[record-navd] error closing final segment: {exc}")
        if vserver is not None:
            vserver.stop()
        rig.stop()
        robot.stop()
        _release_recorder_lock()
        print("\n[record-navd] done (auto mode)")


def main():
    parser = argparse.ArgumentParser(
        description="bebop-vision data recorder (navd MCAP sessions)")
    parser.add_argument("--record-navd", metavar="DIR",
                        help="navd recorder v2: teleop session -> MCAP in DIR")
    parser.add_argument("--auto", action="store_true",
                        help="record-navd: start/stop segments with the drive "
                             "state (wheels armed, no estop), roll on "
                             "size/time, prune under --disk-budget-gb")
    parser.add_argument("--seconds", type=float,
                        help="stop after this many seconds")
    parser.add_argument("--record-rate", type=float, default=None,
                        help="recording rate in Hz (default: 10)")
    parser.add_argument("--max-segment-mb", type=float, default=400.0,
                        help="record-navd --auto: roll segment above this size")
    parser.add_argument("--max-segment-min", type=float, default=10.0,
                        help="record-navd --auto: roll segment above this many minutes")
    parser.add_argument("--disk-budget-gb", type=float, default=20.0,
                        help="record-navd --auto: prune oldest sessions below this total")
    parser.add_argument("--rig", metavar="YAML",
                        help="rig config path (default: config/orbbec_rig.yaml)")
    parser.add_argument("--no-video-server", action="store_true",
                        help="do not serve the operator MJPEG stream on "
                             "--video-port")
    parser.add_argument("--video-port", type=int, default=9092,
                        help="port for the operator video stream "
                             "(default: 9092)")
    parser.add_argument("--roles", default="",
                        help="comma-separated camera roles to open "
                             "(default: all configured, e.g. 'near' while "
                             "the far camera's USB cable is unfixed)")
    parser.add_argument("--robot-url", default=DEFAULT_URL)
    args = parser.parse_args()

    if not args.record_navd:
        parser.error(
            "nothing to do — bebop-vision is a data recorder. Use "
            "--record-navd <dir> [--auto]; drive the robot from the app. "
            "(Model/drive code lives on the exp/navd-ai branch.)")

    # Graceful SIGTERM: sessions started in the background (nohup ... &
    # inside a non-interactive shell) inherit SIGINT=SIG_IGN — CPython
    # keeps an inherited ignore — so a Ctrl-C-style shutdown never
    # arrives and pkill's default SIGTERM would hard-kill the process,
    # leaving the recorder lock behind and any open MCAP segment
    # unflushed (seen 2026-09-06). Translate SIGTERM into the
    # KeyboardInterrupt path the recorder loop already handles.
    import signal

    def _sigterm_to_int(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _sigterm_to_int)

    run_record_navd(args)


if __name__ == "__main__":
    main()
