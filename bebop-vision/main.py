"""bebop-vision data recorder (the data-collection pipeline entry point).

Owns the Orbbec camera rig (one process per camera), the operator video
server, and the navd MCAP recorder: teleop sessions -> color (both
cameras) + lossless depth + cmd_vel/odom, the dataset every
labeling/modeling approach trains from.

    python main.py --record-navd /var/lib/bebop-captures --auto

Drive the robot from the app (manual teleop only — this file carries no
driving; the old model/drive stack is archived on exp/navd-ai-20260909).
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
import threading
from pathlib import Path

from bebop_vision.proto.bebop.runtime.v1 import bebop_runtime_pb2 as pb
from bebop_vision.robot import DEFAULT_URL


def _intent_state_path(path):
    """``path`` with its parent directory created, or None (run without
    persistence) when it is empty or the directory cannot be made."""
    if not path:
        return None
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"[record-navd] warning: cannot create {path.parent} ({exc}); "
              f"drive runs will not survive a restart")
        return None
    return path


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
    capture dir is shared with the firmware (and may still hold policy
    captures written by the retired legged firmware).
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

    DIAL_IN is the only armed mode: the app's manual teleop drives in DialIn
    (armed wheels + cmd_vel), which is the data-collection mode.
    """
    st = robot.state
    return (st.connected
            and st.mode == pb.MODE_DIAL_IN
            and not st.estop_latched
            # differential drive: a single armed wheel is a fault (failed
            # enable), not a drivable state — don't record half-armed runs
            and all(st.wheel_armed.values()))


def _stop_cause(robot, rolled):
    """Why the --auto loop closes its segment, as ``(cause, detail)`` with a
    cause from ``session_notes.STOP_CAUSES``.

    Read from ``robot.state`` at the moment the loop decides to close, in
    this order: link lost, E-STOP (``estop_operator`` when the firmware's
    reason starts with "operator", the app's E-STOP button; ``estop_auto``
    otherwise, with the reason as detail), mode left DialIn (``idle``),
    wheels (``disarmed`` for all, the normal stop; ``wheel_disarmed`` with
    the disarmed names for some), then a size/time roll.  ``(None, "")``
    when the drive state is fine and nothing rolled.
    """
    st = robot.state
    if not st.connected:
        return "link_lost", ""
    if st.estop_latched:
        reason = str(st.estop_reason or "").strip()
        cause = ("estop_operator" if reason.lower().startswith("operator")
                 else "estop_auto")
        return cause, reason
    if st.mode != pb.MODE_DIAL_IN:
        return "idle", ""
    wheels = dict(st.wheel_armed)
    off = sorted(name for name, armed in wheels.items() if not armed)
    if off and len(off) == len(wheels):
        return "disarmed", ""
    if off:
        return "wheel_disarmed", ", ".join(off)
    if rolled:
        return "rolled", ""
    return None, ""


def _settled_stop_cause(robot, rolled, settle_s=0.15, max_s=0.6, poll_s=0.03,
                        clock=None, sleep=None):
    """``_stop_cause`` once the firmware state has stopped moving.

    The firmware passes through intermediate states on a normal stop:
    ``disarm_all_wheels`` disarms one wheel at a time (a frame between them
    reads left off, right on, i.e. ``wheel_disarmed``), and ``set_mode(Idle)``
    disarms every wheel before it stores the new mode (a frame in that gap
    reads ``disarmed``, not ``idle``).  So the cause is re-read every
    ``poll_s`` until it has been the same for ``settle_s``, or ``max_s``
    has passed (then the latest reading wins).  The recorder keeps writing
    meanwhile; the delay only moves the ``stop`` event a little later in
    the closing file.  ``(None, "")`` means the state recovered.
    """
    import time
    clock = clock or time.monotonic
    sleep = sleep or time.sleep
    cur = _stop_cause(robot, rolled)
    t0 = since = clock()
    while True:
        now = clock()
        if now - since >= settle_s or now - t0 >= max_s:
            return cur
        sleep(poll_s)
        nxt = _stop_cause(robot, rolled)
        if nxt != cur:
            cur, since = nxt, clock()


def _note_stop(notes, cause, detail=""):
    """``notes.note_stop`` that never takes the recorder loop down."""
    try:
        notes.note_stop(cause, detail)
    except Exception as exc:
        print(f"\n[record-navd] could not note the stop ({cause}): {exc}")


def run_record_navd(args):
    """Recorder v2: teleop session(s) -> MCAP file(s).

    Default: one session for --seconds (or until Ctrl-C). With --auto the
    recorder follows the drive state instead: a segment opens when the
    firmware is in a driveable, armed state (you start driving) and closes
    when that ends, rolling over on size/time and pruning the oldest files
    under a disk budget. Manual drive = captured data, no SSH per run.

    """
    from bebop_vision.orbbec import OrbbecRig
    from bebop_vision.recorder_mcap import NavdRecorder
    from bebop_vision.videoserver import VideoServer
    from bebop_vision.robot import RobotClient
    from bebop_vision.session_notes import SessionNotes
    import time as _time

    _acquire_recorder_lock()
    if args.stall_watch:
        from bebop_vision.stall_watch import StallWatch
        StallWatch(args.stall_watch, threshold_ms=args.stall_ms,
                   gc_ms=args.stall_gc_ms).start()
        print(f"[stall-watch] logging stalls >= {args.stall_ms:.0f} ms to {args.stall_watch}")
    # Operator notes (intent / events / places) shared by the video server's
    # routes, the scout and every recorder segment.  The open run and the
    # recent runs persist in --intent-state so a restart keeps the run id.
    notes = SessionNotes(state_path=_intent_state_path(args.intent_state))
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
        from bebop_vision.vlm import ChatVLMClient
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

    if not args.no_gc_freeze:
        # Startup objects live as long as the process: keep full garbage
        # collections from walking them (150-180 ms recorder stalls before).
        from bebop_vision.gc_tuning import freeze_heap
        freeze_heap("startup")

    budget = args.disk_budget_gb * 1e9
    rate = args.record_rate if args.record_rate is not None else 30.0

    def new_segment():
        path = _session_path(out_dir, _time.strftime('%Y%m%d_%H%M%S'))
        rec = NavdRecorder(
            rig, robot, path, rate_hz=rate,
            max_frame_age_s=max_frame_age_s, notes=notes)
        rec.start()
        print(f"\n[record-navd] recording -> {path}")
        return rec, path, _time.monotonic()

    def close_segment(rec, path):
        rec.stop()
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
            _note_stop(notes, "recorder_stop")
            close_segment(rec, path)
            if places is not None:
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
                # Say why before closing, so the stop event lands in the
                # closing file; read once the firmware state has settled.
                # None: the drive state recovered while settling (a
                # transient, no stop happened), so the segment stays open.
                cause, detail = _settled_stop_cause(robot, rolled)
                if cause is None:
                    continue
                _note_stop(notes, cause, detail)
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
                print("\r[record-navd] waiting for drive state "
                      "(DialIn + armed)...", end="", flush=True)
            _time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        if seg is not None:
            _note_stop(notes, "recorder_stop")
            try:
                close_segment(seg, seg_path)
            except OSError as exc:
                print(f"\n[record-navd] error closing final segment: {exc}")
        if places is not None:
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
    parser.add_argument("--intent-state", metavar="PATH",
                        default="/var/lib/bebop-vision/intent.json",
                        help="JSON file that keeps the open drive run and the "
                             "recent runs across restarts ('' = memory only; "
                             "default: /var/lib/bebop-vision/intent.json)")
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
    parser.add_argument("--router-url", default="http://127.0.0.1:8012",
                        help="OpenAI-compatible VLM base URL for the "
                             "operator-requested pending-place name "
                             "suggestions (default: the voice cascade brain "
                             "on :8012)")
    parser.add_argument("--router-model", default="RadixArk/Qwen3.8-27B-NVFP4",
                        help="model name sent to --router-url")
    parser.add_argument("--router-timeout", type=float, default=120.0,
                        help="name-suggestion VLM request timeout in seconds "
                             "(default: 120)")
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
