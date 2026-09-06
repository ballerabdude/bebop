"""Entry point: run the bebop-vision navigable-path pipeline.

The default source is the firmware's MJPEG endpoint (the firmware owns
the camera; bebop-vision never opens a capture device directly).

Examples:
    python main.py --display
    python main.py --record run.mp4 --seconds 10
    python main.py --record-dataset datasets/indoor-v1 --concepts "floor,wall" --seconds 60
    python main.py --goal-drive --goal-heading-deg 25      # navd Phase A
    python main.py --goal-drive --goal-xy 1.5 0.5 --display
"""

import argparse
import math
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

from bebop_vision import config
from bebop_vision.robot import DEFAULT_URL
from bebop_vision.proto.bebop.runtime.v1 import bebop_runtime_pb2 as pb


def _render_bev(grid, goal):
    """BEV debug overlay (bench, --display): cell classes + goal arrow.

    Row 0 of the image = far edge (+range), matching the grid convention;
    the white arrow points along the goal bearing from the robot origin.
    """
    import cv2

    colors = {0: (40, 40, 40), 1: (0, 0, 220), 2: (0, 140, 255), 3: (80, 80, 160)}
    rows, cols = grid.occ.shape
    scale = 8
    img = np.zeros((rows * scale, cols * scale, 3), np.uint8)
    for cls, color in colors.items():
        img[grid.occ == cls] = color

    cell = grid.cell_m

    def to_px(x, y):
        return (int((y + cols * cell / 2.0) / cell * scale),
                int((rows * cell - x) / cell * scale))

    if goal is not None:
        bearing = goal.heading_rad if hasattr(goal, "heading_rad") else 0.0
        ox, oy = to_px(0.0, 0.0)
        px, py = to_px(1.2 * math.cos(bearing), 1.2 * math.sin(bearing))
        cv2.arrowedLine(img, (oy, ox), (py, px), (255, 255, 255), 2, tipLength=0.15)
    return img


def run_goal_drive(args):
    from bebop_vision.bev import BevBuilder
    from bebop_vision.goal_planner import (GoalDriveNode, GoalHeading,
                                           GoalPlanner, GoalPoint, GoalSlot,
                                           parse_goal)
    from bebop_vision.orbbec import OrbbecRig
    from bebop_vision.robot import RobotClient

    robot = RobotClient(args.robot_url).start()
    if not robot.await_connection(5.0):
        raise SystemExit(f"cannot reach robot runtime at {robot.url}")
    print(f"[goal-drive] robot: {robot.describe()}")

    rig = OrbbecRig(rig_path=args.rig, color=args.color)
    if not rig.wait_for_pair(timeout=10.0):
        rig.stop()
        raise SystemExit("cameras did not produce fresh frames within 10 s")

    builder = BevBuilder()
    planner = GoalPlanner(v_max=args.v_max, wz_max=args.wz_max,
                          wz_turn=args.wz_turn)
    goal_slot = GoalSlot()
    if args.goal_heading_deg is not None:
        goal_slot.set(GoalHeading(math.radians(args.goal_heading_deg)))
    elif args.goal_xy is not None:
        goal_slot.set(GoalPoint(*args.goal_xy))

    state = {"grid": None, "last_stamp": {}, "cached": {}, "stats_ts": time.monotonic(),
             "grids": 0}
    stop_evt = threading.Event()
    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="bev-proc")

    def bev_worker():
        while not stop_evt.is_set():
            t0 = time.monotonic()
            per_cam, ages = {}, {}
            jobs = {}
            for role, cam in rig.cameras.items():
                f = cam.read()
                if f is None or f.age_s() > builder.max_frame_age_s:
                    per_cam[role], ages[role] = None, None
                    continue
                ages[role] = f.age_s()
                if state["last_stamp"].get(role) != f.stamp_us:
                    jobs[role] = (pool.submit(builder.process, f), f.stamp_us)
                per_cam[role] = state["cached"].get(role)
            for role, (job, stamp) in jobs.items():
                try:
                    state["cached"][role] = job.result()
                except Exception as exc:
                    print(f"[goal-drive] BEV error ({role}): "
                          f"{type(exc).__name__}: {exc}")
                    state["cached"][role] = None
                state["last_stamp"][role] = stamp
                per_cam[role] = state["cached"].get(role)
            try:
                grid = builder.fuse(per_cam, ages)
            except Exception as exc:
                print(f"[goal-drive] fuse error: {type(exc).__name__}: {exc}")
                grid = None
            state["grid"] = grid
            state["grids"] += 1
            now = time.monotonic()
            if now - state["stats_ts"] >= 5.0:
                state["stats_ts"] = now
                info = " ".join(f"{r}:{cam.read_fps:.0f}fps" for r, cam in rig.cameras.items())
                print(f"[goal-drive] {state['grids']} grids | cams {info}")
                state["grids"] = 0
            time.sleep(max(0.0, 0.1 - (time.monotonic() - t0)))

    def stdin_loop():
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                g = parse_goal(line)
            except ValueError as exc:
                print(f"[goal] {exc}")
                continue
            if g == "stop":
                goal_slot.clear()
                print("[goal] cleared (stop)")
            else:
                goal_slot.set(g)
                print(f"[goal] set {g}")

    worker = threading.Thread(target=bev_worker, daemon=True, name="bev-worker")
    worker.start()
    if sys.stdin is not None and sys.stdin.isatty():
        threading.Thread(target=stdin_loop, daemon=True, name="goal-stdin").start()
        print("[goal] type 'heading <deg>' | 'xy <x> <y>' | 'stop' + Enter")

    node = GoalDriveNode(robot, planner, lambda: state["grid"], goal_slot,
                         command_hz=args.command_hz,
                         require_mode=None if args.drive_any_mode else pb.MODE_RUN_POLICY)
    robot.on_estop.append(lambda reason: node.stop())
    print(f"[goal-drive] running (v_max={args.v_max} wz_max={args.wz_max} "
          f"hz={args.command_hz}); Ctrl+C to stop")
    t_start = time.monotonic()
    t_status = 0.0
    try:
        while True:
            node.on_grid()
            now = time.monotonic()
            if now - t_status >= 2.0:
                t_status = now
                st = robot.state
                print(f"[status] {node.last_info} | "
                      f"odom=({st.odom[0]:.2f}, {st.odom[1]:.2f}, "
                      f"{math.degrees(st.odom[2]):.0f}deg) | "
                      f"mode={pb.Mode.Name(st.mode)} estop={st.estop_latched}")
            if args.display:
                grid = state["grid"]
                if grid is not None:
                    import cv2
                    cv2.imshow("navd BEV", _render_bev(grid, goal_slot.get()))
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            if args.seconds and time.monotonic() - t_start > args.seconds:
                break
            time.sleep(0.005)
    except KeyboardInterrupt:
        pass
    finally:
        stop_evt.set()
        node.stop()
        pool.shutdown(wait=False)
        rig.stop()
        robot.stop()
        try:
            import cv2
            cv2.destroyAllWindows()
        except Exception:
            pass
        print(f"[goal-drive] final: {robot.describe()} last={node.last_info}")


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
    + cmd_vel), which is the no-policy data-collection mode. RUN_POLICY
    stays valid for goal-drive sessions.
    """
    st = robot.state
    from bebop_vision.proto.bebop.runtime.v1 import bebop_runtime_pb2 as pb
    return (st.connected
            and st.mode in (pb.MODE_DIAL_IN, pb.MODE_RUN_POLICY)
            and not st.estop_latched
            and any(st.wheel_armed.values()))


def run_record_navd(args):
    """Recorder v2 (plan §7.1): teleop session(s) -> MCAP file(s).

    Default: one session for --seconds (or until Ctrl-C). With --auto the
    recorder follows the drive state instead: a segment opens when the
    firmware is in MODE_RUN_POLICY with wheels armed (you start driving)
    and closes when that ends, rolling over on size/time and pruning the
    oldest files under a disk budget — a mirror of the firmware's own
    policy-capture design. Manual drive = captured data, no SSH per run.
    """
    from bebop_vision.bev import BevBuilder
    from bebop_vision.goal_planner import (GoalHeading, GoalPlanner, GoalPoint,
                                           GoalSlot, parse_goal)
    from bebop_vision.orbbec import OrbbecRig
    from bebop_vision.recorder_mcap import NavdRecorder
    from bebop_vision.robot import RobotClient
    import time as _time

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

    builder = BevBuilder()
    goal_slot = GoalSlot()
    if args.goal_heading_deg is not None:
        goal_slot.set(GoalHeading(math.radians(args.goal_heading_deg)))
    elif args.goal_xy is not None:
        goal_slot.set(GoalPoint(*args.goal_xy))

    out_dir = Path(args.record_navd).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    budget = args.disk_budget_gb * 1e9
    rate = args.record_rate if args.record_rate is not None else 10.0

    if sys.stdin is not None and sys.stdin.isatty():
        def stdin_loop():
            for line in sys.stdin:
                try:
                    g = parse_goal(line)
                except ValueError as exc:
                    print(f"[goal] {exc}")
                    continue
                if g == "stop":
                    goal_slot.clear()
                else:
                    goal_slot.set(g)
        threading.Thread(target=stdin_loop, daemon=True).start()
        print("[goal] type 'heading <deg>' | 'xy <x> <y>' | 'stop' + Enter")

    def new_segment():
        path = out_dir / f"navd_session_{_time.strftime('%Y%m%d_%H%M%S')}.mcap"
        rec = NavdRecorder(rig, robot, goal_slot, path, builder=builder,
                           rate_hz=rate)
        rec.start()
        print(f"\n[record-navd] recording -> {path}")
        return rec, path, _time.monotonic()

    def close_segment(rec, path):
        rec.stop()
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
                seg, seg_path, seg_t0 = new_segment()
            elif seg is not None and (not active or rolled):
                close_segment(seg, seg_path)
                seg = None
                if active and rolled:
                    seg, seg_path, seg_t0 = new_segment()
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
            close_segment(seg, seg_path)
        rig.stop()
        robot.stop()
        _release_recorder_lock()
        print("\n[record-navd] done (auto mode)")


def main():
    parser = argparse.ArgumentParser(description="bebop-vision navigable-path pipeline")
    parser.add_argument("--source", default=config.DEFAULT_SOURCE,
                        help="video source URL or file path (default: the robot's "
                             "firmware MJPEG endpoint)")
    parser.add_argument("--nav-model", default=config.DEFAULT_NAV_MODEL)
    parser.add_argument("--display", action="store_true", help="show live window (needs a display)")
    parser.add_argument("--no-hud", action="store_true", help="hide the stats overlay")
    parser.add_argument("--record", metavar="PATH", help="write annotated video to PATH")
    parser.add_argument("--seconds", type=float, help="stop after this many seconds")
    parser.add_argument("--record-dataset", metavar="DIR",
                        help="record a teacher dataset (frames + SAM 3.1 masks) instead of the pipeline")
    parser.add_argument("--record-rate", type=float, default=None,
                    help="recording rate in Hz (default: 2 for --record-dataset, 10 for --record-navd)")
    parser.add_argument("--concepts",
                        help="comma-separated teacher concepts for --record-dataset")
    parser.add_argument("--conf", type=float, default=config.DEFAULT_CONFIDENCE)
    parser.add_argument("--sam-model", default="sam3.1", choices=["sam3", "sam3.1"])
    parser.add_argument("--sam-trt", metavar="ENGINE",
                        help="TensorRT engine for the SAM 3 vision encoder")
    parser.add_argument("--drive", action="store_true",
                        help="drive the robot: nav mask -> planner -> SetVelocityCommand")
    parser.add_argument("--goal-drive", action="store_true",
                        help="navd: Orbbec depth -> BEV grid -> goal planner -> twist")
    goal_group = parser.add_mutually_exclusive_group()
    goal_group.add_argument("--goal-heading-deg", type=float, metavar="DEG",
                            help="initial goal: body-frame heading offset (deg, + left)")
    goal_group.add_argument("--goal-xy", type=float, nargs=2, metavar=("X", "Y"),
                            help="initial goal: odom waypoint (m)")
    parser.add_argument("--rig", metavar="YAML",
                        help="rig config path (default: config/orbbec_rig.yaml)")
    parser.add_argument("--color", action="store_true",
                        help="also stream camera color (recording/debug; costs USB+CPU)")
    parser.add_argument("--record-navd", metavar="DIR",
                        help="navd recorder v2: teleop session -> MCAP in DIR")
    parser.add_argument("--auto", action="store_true",
                        help="record-navd: start/stop segments with the drive "
                             "state (RunPolicy + armed), roll on size/time, "
                             "prune under --disk-budget-gb")
    parser.add_argument("--max-segment-mb", type=float, default=400.0,
                        help="record-navd --auto: roll segment above this size")
    parser.add_argument("--max-segment-min", type=float, default=10.0,
                        help="record-navd --auto: roll segment above this many minutes")
    parser.add_argument("--disk-budget-gb", type=float, default=20.0,
                        help="record-navd --auto: prune oldest sessions below this total")
    parser.add_argument("--roles", default="",
                        help="comma-separated camera roles to open "
                             "(default: all configured, e.g. 'near' while "
                             "the far camera's USB cable is unfixed)")
    parser.add_argument("--robot-url", default=DEFAULT_URL)
    parser.add_argument("--v-max", type=float, default=0.4)
    parser.add_argument("--wz-max", type=float, default=1.2)
    parser.add_argument("--wz-turn", type=float, default=1.8,
                        help="search/rotate-in-place turn rate (rad/s)")
    parser.add_argument("--command-hz", type=float, default=10.0)
    parser.add_argument("--drive-any-mode", action="store_true",
                        help="command velocity regardless of firmware mode (bench only)")
    args = parser.parse_args()

    if args.goal_drive:
        run_goal_drive(args)
        return

    if args.record_navd:
        run_record_navd(args)
        return

    if args.record_dataset:
        from bebop_vision.recorder import DatasetRecorder
        concepts = (
            [c.strip() for c in args.concepts.split(",") if c.strip()]
            if args.concepts
            else list(config.RECORD_CONCEPTS)
        )
        recorder = DatasetRecorder(
            source=args.source,
            out_dir=args.record_dataset,
            concepts=concepts,
            conf=args.conf,
            version=args.sam_model,
            trt_engine=args.sam_trt,
            rate_hz=args.record_rate if args.record_rate is not None else 2.0,
            display=args.display,
        )
        recorder.run(duration=args.seconds)
        return

    from bebop_vision.pipeline import NavPipeline

    pipeline = NavPipeline(
        source=args.source,
        nav_model=args.nav_model,
        display=args.display,
        record_path=args.record,
        duration=args.seconds,
        show_hud=not args.no_hud,
    )

    if not args.drive:
        pipeline.run()
        return

    from bebop_vision.planner import DriveNode, SectorPlanner
    from bebop_vision.robot import RobotClient

    robot = RobotClient(args.robot_url).start()
    if not robot.await_connection(5.0):
        raise SystemExit(f"cannot reach robot runtime at {robot.url}")
    print(f"[drive] robot: {robot.describe()}")

    planner = SectorPlanner(v_max=args.v_max, wz_max=args.wz_max)
    drive = DriveNode(robot, planner, command_hz=args.command_hz,
                      require_mode=None if args.drive_any_mode else pb.MODE_RUN_POLICY)
    robot.on_estop.append(lambda reason: drive.stop())
    try:
        pipeline.run(frame_sink=drive.on_frame)
    finally:
        drive.stop()
        robot.stop()
        print(f"[drive] final: {robot.describe()}")


if __name__ == "__main__":
    main()