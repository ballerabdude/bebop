"""Entry point: run the bebop-vision navigable-path pipeline.

The default source is the firmware's MJPEG endpoint (the firmware owns
the camera; bebop-vision never opens a capture device directly).

Examples:
    python main.py --display
    python main.py --record run.mp4 --seconds 10
    python main.py --record-dataset datasets/indoor-v1 --concepts "floor,wall" --seconds 60
"""

import argparse

from bebop_vision import config
from bebop_vision.pipeline import NavPipeline
from bebop_vision.recorder import DatasetRecorder
from bebop_vision.robot import DEFAULT_URL
from bebop_vision.proto.bebop.runtime.v1 import bebop_runtime_pb2 as pb


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
    parser.add_argument("--record-rate", type=float, default=2.0, help="dataset recording rate in Hz")
    parser.add_argument("--concepts",
                        help="comma-separated teacher concepts for --record-dataset")
    parser.add_argument("--conf", type=float, default=config.DEFAULT_CONFIDENCE)
    parser.add_argument("--sam-model", default="sam3.1", choices=["sam3", "sam3.1"])
    parser.add_argument("--sam-trt", metavar="ENGINE",
                        help="TensorRT engine for the SAM 3 vision encoder")
    parser.add_argument("--drive", action="store_true",
                        help="drive the robot: nav mask -> planner -> SetVelocityCommand")
    parser.add_argument("--robot-url", default=DEFAULT_URL)
    parser.add_argument("--v-max", type=float, default=0.4)
    parser.add_argument("--wz-max", type=float, default=1.2)
    parser.add_argument("--command-hz", type=float, default=10.0)
    parser.add_argument("--drive-any-mode", action="store_true",
                        help="command velocity regardless of firmware mode (bench only)")
    args = parser.parse_args()

    if args.record_dataset:
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
            rate_hz=args.record_rate,
            display=args.display,
        )
        recorder.run(duration=args.seconds)
        return

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