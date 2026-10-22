"""bebop-vision data recorder (the data-collection pipeline entry point).

Owns the Orbbec camera rig (exclusivity, docs §2.7), the operator video
server, and the navd MCAP recorder: teleop sessions -> color (both
cameras) + lossless depth + cmd_vel/odom, the dataset every
labeling/modeling approach trains from.

    python main.py --record-navd /var/lib/bebop-captures --auto

Drive the robot from the app (manual teleop only — this file carries no
driving; the model/drive stack lives on the exp/navd-ai branch).
"""

import os

# The Thor's OpenBLAS build keeps a spinning worker pool alive after even a
# tiny matmul.  The BEV/recorder use small matrices where one thread is
# faster, and the idle pool otherwise burns several cores (measured: 5 BLAS
# threads at 70-90% each in a standalone --bev run).  Set these before numpy
# is imported anywhere in this process.
for _thread_var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_thread_var, "1")

import argparse
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

    # Live master BEV: fuses both depth cameras into one non-square grid
    # and publishes /bev + /snapshot?stream=bev on the video server.  It is
    # independent of recording/armed state, so the map is live while the
    # robot is idle too.
    bev_service = None
    if vserver is not None and not args.no_bev:
        if not args.bev_model:
            print("[bev] no --bev-model; BEV disabled (model-only)",
                  flush=True)
        else:
            from bebop_vision.bev_service import BevService
            bev_service = BevService(
                rig, hz=args.bev_hz, video_server=vserver,
                model_path=args.bev_model,
                model_meta=args.bev_model_meta,
                hazard_source=args.bev_hazard_source,
                hazard_threshold=args.bev_hazard_threshold)
            bev_service.start()

    # Semantic place memory (bebop_vision/places): teaches named rooms from
    # the live view and recognises them by DINOv3 appearance. Shares the rig
    # read-only; the encoder warms on a background thread so a slow first
    # load never stalls recording or the drive loop.
    places = None
    if not args.no_places:
        from bebop_vision.places import (DinoV3Encoder, PlaceImporter,
                                         PlaceScout, PlaceStore,
                                         PlacesService, RigCapture,
                                         ScoutConfig)
        from bebop_vision.router import ChatVLMClient
        pstore = PlaceStore(args.places_root)
        penc = DinoV3Encoder()
        pcapture = RigCapture(rig, roles=tuple(rig.cameras.keys()))
        # The VLM client is cheap to construct and is only called by an
        # operator-requested import suggestion.  The scout never calls it.
        vlm = ChatVLMClient(args.router_url, args.router_model,
                            timeout=args.router_timeout)
        scout = PlaceScout(
            pstore, penc, pcapture,
            config=ScoutConfig(
                interval_s=args.place_scout_interval,
                max_pending=args.place_pending_max,
                pending_ttl_s=args.place_pending_ttl_days * 86400.0,
            ),
        )
        importer = PlaceImporter(pstore, vlm)
        places = PlacesService(
            pstore,
            capture=pcapture,
            encoder=penc,
            port=args.places_port,
            scout=scout,
            importer=importer,
        )
        threading.Thread(target=penc.warmup, name="places-warmup",
                         daemon=True).start()
        places.start()

        # Legacy System 1 router: fast VLM skills over one shared tick.  It
        # is off by default now that the lofi scout + operator import is the
        # normal path; enabling it still costs an LLM call per tick.
        if args.place_router:
            from bebop_vision.router import (
                CaptureMapExecutor, CaptureMapSkill,
                PlaceMemoryExecutor, PlaceMemorySkill, RouterService,
                VoiceBusyProbe,
            )
            print("[router] legacy periodic LLM router enabled; the scout "
                  "is also running", flush=True)
            rskill = PlaceMemorySkill(pstore, penc, vlm)
            rexec = PlaceMemoryExecutor(pstore, penc,
                                        cooldown_s=args.router_cooldown)
            extra_skills = []
            if args.live_slam:
                structure_url = (f"http://127.0.0.1:{args.live_slam_port}"
                                 "/structure")
                extra_skills.append((
                    CaptureMapSkill(pstore, penc, vlm,
                                    structure_url=structure_url,
                                    cooldown_s=args.router_cooldown),
                    CaptureMapExecutor(pstore,
                                       structure_url=structure_url,
                                       cooldown_s=args.router_cooldown),
                ))
            places.router = RouterService(
                rskill, rexec, capture=places.capture,
                skills=extra_skills,
                voice_busy=VoiceBusyProbe(args.router_voice_url),
                interval_s=args.router_interval,
            )
            places.router.start()


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

    # Live visual localization (observe-only): consumes the already-open
    # rig slots, starts/stops with the recorder, and is independent of the
    # recording armed state. No camera is opened here.
    slam = None
    slam_server = None

    def stop_live_slam():
        nonlocal slam, slam_server
        if slam_server is not None:
            try:
                slam_server.stop()
            finally:
                slam_server = None
        if slam is not None:
            try:
                slam.stop()
            finally:
                slam = None

    if args.live_slam:
        from bebop_vision.slam import (LiveSlam, LiveSlamConfig,
                                       SlamStatusServer)
        try:
            log_path = (Path(args.live_slam_log).expanduser()
                        if args.live_slam_log else None)
            slam = LiveSlam(rig, LiveSlamConfig(
                role=args.live_slam_camera,
                rig_path=(Path(args.rig).expanduser() if args.rig else None),
                enable_slam=bool(args.live_slam_map),
                max_pair_delta_us=int(round(
                    args.live_slam_max_pair_skew_ms * 1000)),
                log_path=log_path))
            slam.start()
            slam_server = SlamStatusServer(slam, port=args.live_slam_port)
            slam_server.start()
        except Exception as exc:
            stop_live_slam()
            if places is not None:
                if places.router is not None:
                    places.router.stop()
                places.stop()
            if bev_service is not None:
                bev_service.stop()
            if vserver is not None:
                vserver.stop()
            rig.stop()
            robot.stop()
            _release_recorder_lock()
            raise SystemExit(f"live-slam failed to start: {exc}")

    from bebop_vision.orbbec import load_rig_config
    safety = load_rig_config()["robots"]["default"].get("safety", {})
    max_frame_age_s = float(safety.get("max_frame_age_s", 0.3))

    # Waypoint driver: depth stop-cone + bearing controller over the same
    # rig slots.  Idle until a goal is POSTed; arms only for the goal.
    nav = None
    nav_server = None

    def stop_nav():
        nonlocal nav, nav_server
        if nav_server is not None:
            try:
                nav_server.stop()
            finally:
                nav_server = None
        if nav is not None:
            try:
                nav.stop()
            finally:
                nav = None

    if args.nav:
        from bebop_vision.nav import (BearingController, CameraGeometry,
                                      ConeConfig, ControllerConfig, DepthCone,
                                      NavDrive, NavServer, footprint_from_rig)
        try:
            nav_roles = tuple(r.strip() for r in args.nav_roles.split(",")
                              if r.strip() and r.strip() in rig.cameras)
            if not nav_roles:
                raise RuntimeError(
                    f"--nav-roles {args.nav_roles!r} matches no open camera")
            geoms = [CameraGeometry.from_rig(r, rig_path=args.rig)
                     for r in nav_roles]
            cone = DepthCone(geoms, ConeConfig(
                footprint_m=tuple(footprint_from_rig(rig_path=args.rig))))
            ctrl = BearingController(ControllerConfig(
                v_max=args.nav_v_max, w_max=args.nav_w_max,
                stop_m=args.nav_stop_m))
            nav = NavDrive(rig, robot, cone, ctrl, hz=args.nav_hz,
                           max_frame_age_s=max_frame_age_s,
                           roles=nav_roles).start()
            nav_server = NavServer(nav, port=args.nav_port).start()
            if places is not None:
                from bebop_vision.places import TrailConfig, TrailTeacher
                places.teacher = TrailTeacher(
                    pstore, pcapture, penc, nav.current_pose,
                    TrailConfig(step_m=args.teach_step_m,
                                turn_deg=args.teach_turn_deg),
                    recognize=lambda: places.whereami(),
                ).start_thread()
                print("[trail] route teach mode on the places service "
                      "(POST /routes/teach/start|stop)", flush=True)
                from bebop_vision.nav import FollowConfig, RouteFollower
                far_capture = RigCapture(rig, roles=("far",), max_age_s=0.8)
                nav.follower = RouteFollower(
                    nav, pstore, penc, far_capture,
                    recognize=lambda: places.whereami(),
                    config=FollowConfig(
                        patch_image_size=args.follow_patch_size),
                ).start_thread()
                print("[follow] route following on :%d "
                      "(POST /route/follow)" % args.nav_port, flush=True)
                from bebop_vision.nav import TripRunner
                nav.trip = TripRunner(
                    nav.follower, pstore,
                    recognize=lambda: places.whereami()).start_thread()
                print("[trip] multi-leg trips on :%d (POST /trip)"
                      % args.nav_port, flush=True)
        except Exception as exc:
            stop_nav()
            stop_live_slam()
            if places is not None:
                if places.router is not None:
                    places.router.stop()
                places.stop()
            if bev_service is not None:
                bev_service.stop()
            if vserver is not None:
                vserver.stop()
            rig.stop()
            robot.stop()
            _release_recorder_lock()
            raise SystemExit(f"nav failed to start: {exc}")
    budget = args.disk_budget_gb * 1e9
    rate = args.record_rate if args.record_rate is not None else 30.0
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
            stop_nav()
            stop_live_slam()
            if places is not None:
                if places.router is not None:
                    places.router.stop()
                places.stop()
            if bev_service is not None:
                bev_service.stop()
            if vserver is not None:
                vserver.stop()
            rig.stop()
            robot.stop()
            _release_recorder_lock()
        return

    # --auto: follow the drive state; one MCAP per drive session (rolling by
    # size/time is opt-in via --max-segment-*, 0 = off).
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
                rolled = ((max_bytes > 0 and seg.bytes_written >= max_bytes)
                          or (max_s > 0
                              and _time.monotonic() - seg_t0 >= max_s))
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
        stop_nav()
        stop_live_slam()
        if places is not None:
            if places.router is not None:
                places.router.stop()
            places.stop()
        if bev_service is not None:
            bev_service.stop()
        if vserver is not None:
            vserver.stop()
        rig.stop()
        robot.stop()
        _release_recorder_lock()
        print("\n[record-navd] done (auto mode)")


def run_bev(args):
    """Standalone live master-BEV view (no recording, no robot connection).

    Opens the depth cameras, starts the operator video server and fuses both
    cameras into the non-square master grid.  ``GET /bev`` returns the grid
    as JSON; ``GET /snapshot?stream=bev`` returns the rendered JPEG.
    """
    if not args.bev_model:
        raise SystemExit(
            "--bev is model-only; pass --bev-model <engine>")
    from bebop_vision.bev_service import BevService
    from bebop_vision.orbbec import OrbbecRig
    from bebop_vision.videoserver import VideoServer
    import time as _time

    _acquire_recorder_lock()
    roles = tuple(r.strip() for r in args.roles.split(",") if r.strip())
    rig = OrbbecRig(rig_path=args.rig, color=bool(args.bev_model),
                    roles=roles or None)
    if not rig.wait_for_pair(timeout=10.0):
        rig.stop()
        _release_recorder_lock()
        raise SystemExit("cameras did not produce fresh frames within 10 s")

    vserver = None
    if not args.no_video_server:
        vserver = VideoServer(rig, port=args.video_port)
        vserver.start()

    service = BevService(
        rig, hz=args.bev_hz, video_server=vserver,
        model_path=args.bev_model, model_meta=args.bev_model_meta,
        hazard_source=args.bev_hazard_source)
    service.start()
    t0 = _time.monotonic()
    try:
        while True:
            if args.seconds and _time.monotonic() - t0 > args.seconds:
                break
            _time.sleep(0.5)
            grid = service.latest()
            if grid is None:
                print("\r[bev] waiting for a fused grid...", end="",
                      flush=True)
                continue
            c = grid.counts()
            print(f"\r[bev] {service.status()['fps']:4.1f} Hz  "
                  f"{grid.rows}x{grid.cols}  occ={c['occupied']} "
                  f"haz={c['hazard']} unk={c['unknown']}", end="",
                  flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        service.stop()
        if vserver is not None:
            vserver.stop()
        rig.stop()
        _release_recorder_lock()
        print("\n[bev] done")


def main():
    parser = argparse.ArgumentParser(
        description="bebop-vision data recorder (navd MCAP sessions)")
    parser.add_argument("--record-navd", metavar="DIR",
                        help="navd recorder v2: teleop session -> MCAP in DIR")
    parser.add_argument("--bev-model", default=None,
                        help="learned BEV TensorRT engine; "
                             "default: geometric master BEV")
    parser.add_argument("--bev-model-meta", default=None,
                        help="metadata JSON next to the engine")
    parser.add_argument("--bev-hazard-source",
                        choices=("geometric", "union", "model"),
                        default="model",
                        help="hazard source for the learned BEV "
                             "(default: model-only)")
    parser.add_argument("--bev-hazard-threshold", type=float, default=0.5,
                        help="student hazard probability threshold "
                             "(0.5 = plain argmax; lower trades precision "
                             "for stair/drop recall)")
    parser.add_argument("--bev", action="store_true",
                        help="run the live learned-BEV view only (no "
                             "recording, no robot connection)")
    parser.add_argument("--bev-hz", type=float, default=10.0,
                        help="master-BEV update rate (default: 10)")
    parser.add_argument("--no-bev", action="store_true",
                        help="record-navd: do not run the live master-BEV "
                             "worker")
    parser.add_argument("--auto", action="store_true",
                        help="record-navd: start/stop segments with the drive "
                             "state (wheels armed, no estop), roll on "
                             "size/time, prune under --disk-budget-gb")
    parser.add_argument("--seconds", type=float,
                        help="stop after this many seconds")
    parser.add_argument("--record-rate", type=float, default=None,
                        help="recording rate in Hz (default: 30; the dual "
                             "cameras are hardware-synced at 30 fps, so 30 "
                             "captures every frame — higher only duplicates)")
    parser.add_argument("--max-segment-mb", type=float, default=0.0,
                        help="record-navd --auto: roll the MCAP above this "
                             "size (0 = off, one file per drive session)")
    parser.add_argument("--max-segment-min", type=float, default=0.0,
                        help="record-navd --auto: roll the MCAP above this "
                             "many minutes (0 = off)")
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
    parser.add_argument("--live-slam", action="store_true",
                        help="observe-only live RGB-D visual odometry/SLAM "
                             "over the recorder's camera (no movement "
                             "commands; status on --live-slam-port)")
    parser.add_argument("--live-slam-camera", default="far",
                        choices=["far", "near"],
                        help="camera role used by --live-slam (default: far)")
    parser.add_argument("--live-slam-max-pair-skew-ms", type=float,
                        default=17.0,
                        help="max color/depth device-timestamp skew for a "
                             "valid RGB-D pair (default: 17 ms; measured Thor "
                             "synchronized phase up to 16.6 ms = 30 Hz half "
                             "period + rounding; 33 ms mismatches still "
                             "rejected)")
    parser.add_argument("--live-slam-port", type=int, default=9096,
                        help="read-only live-SLAM status HTTP port "
                             "(default: 9096)")
    parser.add_argument("--live-slam-map", action="store_true",
                        help="enable cuVSLAM SLAM and expose the separate "
                             "corrected/map pose (needs --live-slam)")
    parser.add_argument("--live-slam-log", default=None,
                        help="optional JSONL path for per-frame live-SLAM "
                             "records (needs --live-slam)")
    parser.add_argument("--live-slam-imu", action="store_true",
                        help="request camera IMU (rejected in this POC: no "
                             "validated camera-to-IMU calibration)")
    parser.add_argument("--nav", action="store_true",
                        help="start the depth stop-cone + bearing waypoint "
                             "driver (HTTP goals on --nav-port)")
    parser.add_argument("--nav-port", type=int, default=9097,
                        help="nav HTTP port (default 9097)")
    parser.add_argument("--nav-roles", default="near,far",
                        help="depth cameras feeding the stop-cone "
                             "(default near,far)")
    parser.add_argument("--nav-hz", type=float, default=15.0,
                        help="drive loop rate (default 15)")
    parser.add_argument("--nav-v-max", type=float, default=0.30,
                        help="forward speed cap m/s (default 0.30)")
    parser.add_argument("--nav-w-max", type=float, default=0.9,
                        help="yaw rate cap rad/s (default 0.9)")
    parser.add_argument("--nav-stop-m", type=float, default=0.60,
                        help="clearance from the mast below which forward "
                             "motion stops (default 0.60)")
    parser.add_argument("--teach-step-m", type=float, default=0.6,
                        help="route teach: new node every this many metres")
    parser.add_argument("--teach-turn-deg", type=float, default=25.0,
                        help="route teach: new node after this heading change")
    parser.add_argument("--follow-patch-size", type=int, default=224,
                        help="route following: DINOv3 input size for patch "
                             "matching (224 -> 14x14 grid, 336 -> 21x21)")
    parser.add_argument("--no-places", action="store_true",
                        help="do not start the semantic place-memory service "
                             "(see bebop_vision/places; default: started)")
    parser.add_argument("--places-port", type=int, default=9095,
                        help="port for the places HTTP service (default: 9095)")
    parser.add_argument("--places-root", default="/var/lib/bebop-places",
                        help="directory for the place DB + keyframes "
                             "(default: /var/lib/bebop-places)")
    parser.add_argument("--place-scout-interval", type=float, default=1.0,
                        help="seconds between lofi pending-place scans "
                             "(default: 1.0; the scout always runs with "
                             "--places)")
    parser.add_argument("--place-pending-max", type=int, default=50,
                        help="maximum pending places kept for review "
                             "(default: 50)")
    parser.add_argument("--place-pending-ttl-days", type=float, default=14.0,
                        help="delete unreviewed pending places after this many "
                             "days (default: 14)")
    parser.add_argument("--place-router", action="store_true",
                        help="enable the legacy every-tick System 1 LLM router "
                             "(place memory + map capture; costs an LLM call "
                             "per tick; the lofi scout runs regardless)")
    parser.add_argument("--router-url", default="http://127.0.0.1:8012",
                        help="OpenAI-compatible VLM base URL for the router "
                             "(default: the voice cascade brain on :8012)")
    parser.add_argument("--router-model", default="RadixArk/Qwen3.8-27B-NVFP4",
                        help="model name sent to --router-url")
    parser.add_argument("--router-interval", type=float, default=20.0,
                        help="seconds between background router assessments "
                             "(default: 20)")
    parser.add_argument("--router-cooldown", type=float, default=180.0,
                        help="seconds during which a second remember folds into "
                             "the place just created (default: 180)")
    parser.add_argument("--router-timeout", type=float, default=120.0,
                        help="VLM request timeout in seconds (default: 120)")
    parser.add_argument("--router-voice-url",
                        default="http://127.0.0.1:9093/healthz",
                        help="voice /healthz URL probed to skip router ticks "
                             "while a voice turn is in flight (fail-open)")
    parser.add_argument("--robot-url", default=DEFAULT_URL)
    args = parser.parse_args()

    if args.place_router and args.no_places:
        parser.error("--place-router needs the place store; drop --no-places")
    if args.live_slam_imu:
        parser.error(
            "--live-slam-imu is rejected in this POC: there is no validated "
            "camera-to-IMU calibration or device/host timestamp alignment, and "
            "identity extrinsics are not fabricated. Live SLAM is RGB-D only.")
    if (args.live_slam_map or args.live_slam_log) and not args.live_slam:
        parser.error("--live-slam-map/--live-slam-log need --live-slam")

    if not args.record_navd and not args.bev:
        parser.error(
            "nothing to do — use --bev for the live master-BEV view, or "
            "--record-navd <dir> [--auto] for data collection. Drive the "
            "robot from the app. (Model/drive code lives on the "
            "exp/navd-ai branch.)")

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

    if args.bev and not args.record_navd:
        run_bev(args)
        return
    run_record_navd(args)


if __name__ == "__main__":
    main()
