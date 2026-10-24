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
# tiny matmul.  The recorder uses small matrices where one thread is
# faster, and the idle pool otherwise burns several cores (measured: 5 BLAS
# threads at 70-90% each in a standalone BEV run, since removed).  Set these
# before numpy is imported anywhere in this process.
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


def _session_path(out_dir, stamp):
    """navd_session_<stamp>.mcap, suffixed _1, _2, ... when that name exists.

    Segment names have whole-second resolution and the recorder opens its
    file exclusively; without this a roll completing within the same second
    would collide with the segment just closed.
    """
    base = Path(out_dir) / f"navd_session_{stamp}.mcap"
    if not base.exists():
        return base
    for i in range(1, 1000):
        cand = Path(out_dir) / f"navd_session_{stamp}_{i}.mcap"
        if not cand.exists():
            return cand
    raise OSError(f"too many segments named {base.name}")


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
    from bebop_vision.session_notes import SessionNotes
    import time as _time
    import math

    _acquire_recorder_lock()
    if args.stall_watch:
        from bebop_vision.stall_watch import StallWatch
        StallWatch(args.stall_watch, threshold_ms=args.stall_ms,
                   gc_ms=args.stall_gc_ms).start()
        print(f"[stall-watch] logging stalls >= {args.stall_ms:.0f} ms to {args.stall_watch}")
    # Operator notes (intent / events / places) shared by the video server's
    # routes, the scout and every recorder segment.
    notes = SessionNotes()
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
        vserver = VideoServer(
            rig, port=args.video_port, notes=notes,
            bitrates={"color": args.video_color_kbps * 1000,
                      "depth": args.video_depth_kbps * 1000},
            max_viewers=args.video_max_viewers,
            idr_interval_s=args.video_idr_s)
        vserver.start()

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
            on_known=notes.add_place,
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
        def _warm_places():
            penc.warmup()
            if not args.no_gc_freeze:
                # The ViT-L model is ~280 k tracked objects; freeze them so a
                # full GC stops scanning them (bebop_vision/gc_tuning.py).
                from bebop_vision.gc_tuning import freeze_heap
                freeze_heap("places encoder warmup")

        threading.Thread(target=_warm_places, name="places-warmup",
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
            if vserver is not None:
                vserver.stop()
            rig.stop()
            robot.stop()
            _release_recorder_lock()
            raise SystemExit(f"live-slam failed to start: {exc}")

    from bebop_vision.orbbec import load_rig_config
    safety = load_rig_config()["robots"]["default"].get("safety", {})
    max_frame_age_s = float(safety.get("max_frame_age_s", 0.3))

    if not args.no_gc_freeze:
        # Startup objects live as long as the process: keep full garbage
        # collections from walking them (150-180 ms recorder stalls before).
        from bebop_vision.gc_tuning import freeze_heap
        freeze_heap("startup")

    budget = args.disk_budget_gb * 1e9
    rate = args.record_rate if args.record_rate is not None else 30.0
    rec = None
    rec_holder = {"rec": None}   # active segment recorder (set by new_segment)

    def new_segment():
        path = _session_path(out_dir, _time.strftime('%Y%m%d_%H%M%S'))
        rec = NavdRecorder(
            rig, robot, path, rate_hz=rate,
            max_frame_age_s=max_frame_age_s, notes=notes)
        rec.start()
        rec_holder["rec"] = rec
        print(f"\n[record-navd] recording -> {path}")
        return rec, path, _time.monotonic()

    def close_segment(rec, path):
        rec.stop()
        if rec_holder["rec"] is rec:
            rec_holder["rec"] = None
        print(f"\n[record-navd] closed {path.name} "
              f"({rec.frames} frames, {rec.bytes_written/1e6:.1f} MB, "
              f"{rec.runtime_frames} telemetry frames"
              f"{', stale notes dropped: ' + str(rec.notes_stale) if rec.notes_stale else ''})")
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
            stop_live_slam()
            if places is not None:
                if places.router is not None:
                    places.router.stop()
                places.stop()
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
        stop_live_slam()
        if places is not None:
            if places.router is not None:
                places.router.stop()
            places.stop()
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
                        help="do not serve the operator WebRTC (WHEP) "
                             "video on --video-port")
    parser.add_argument("--video-port", type=int, default=9092,
                        help="port for the operator video stream "
                             "(default: 9092)")
    parser.add_argument("--video-color-kbps", type=int, default=3500,
                        help="WHEP color stream CBR bitrate (default: 3500)")
    parser.add_argument("--video-depth-kbps", type=int, default=1000,
                        help="WHEP depth stream CBR bitrate (default: 1000)")
    parser.add_argument("--video-max-viewers", type=int, default=4,
                        help="simultaneous WHEP viewers per stream; they "
                             "share one encoder (default: 4)")
    parser.add_argument("--video-idr-s", type=float, default=1.0,
                        help="periodic IDR interval of the WHEP encoders, "
                             "seconds (joins and PLIs force one sooner; "
                             "default: 1.0)")
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
    parser.add_argument("--stall-watch", metavar="FILE", default=None,
                        help="diagnostic: log GIL stalls (with every thread's "
                             "stack, captured during the stall) and long GC "
                             "pauses to FILE (bebop_vision/stall_watch.py)")
    parser.add_argument("--stall-ms", type=float, default=150.0,
                        help="stall threshold for --stall-watch (default 150)")
    parser.add_argument("--stall-gc-ms", type=float, default=50.0,
                        help="--stall-watch logs garbage collections at least "
                             "this long (default 50)")
    parser.add_argument("--no-gc-freeze", action="store_true",
                        help="do not freeze the startup heap (gc.freeze); only "
                             "for comparing GC pauses, see gc_tuning.py")
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

    if not args.record_navd:
        parser.error(
            "nothing to do — use --record-navd <dir> [--auto] for data "
            "collection. Drive the robot from the app.")

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
