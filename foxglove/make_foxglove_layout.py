#!/usr/bin/env python3
"""Generate Foxglove Studio layouts (.json) for Bebop MCAP review.

Each layout lives in its own module:
  - layout_robot.py  -> bebop_robot_layout.json
  - layout_noise.py  -> bebop_noise_layout.json

Import the emitted file via Foxglove's layouts sidebar -> "Import from
file". Works in both the desktop app and app.foxglove.dev.

Usage:
    python3 make_foxglove_layout.py --layout robot
    python3 make_foxglove_layout.py --layout noise --out my_noise_layout.json
    python3 make_foxglove_layout.py --layout all --force

    python3 layout_robot.py
    python3 layout_noise.py --force
"""
import argparse
import json
import os
import sys

# Allow imports when run from the repo root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import layout_noise
import layout_robot

LAYOUTS = {
    layout_robot.NAME: (layout_robot.OUTPUT, layout_robot.build),
    layout_noise.NAME: (layout_noise.OUTPUT, layout_noise.build),
}


<<<<<<< HEAD
def write_layout(name, out, *, force):
    if os.path.exists(out) and not force:
        print(f"skip {out} (exists; pass --force to overwrite)")
        return
=======
def path(field, label):
    """One plot series: a message path with a human-readable legend label."""
    return {
        "value": f"{TOPIC}.{field}",
        "enabled": True,
        "label": label,
        "timestampMethod": "receiveTime",
    }


def plot(paths, *, title):
    return {
        # NOTE: built-in Foxglove panels do not render a `title` set via
        # raw layout JSON (only UI-set titles persist), so we ALSO encode
        # the quantity into each series `label` below — that's what
        # actually shows in the legend. `title` is kept for documentation
        # and in case a future Foxglove version honors it.
        "title": title,
        "paths": paths,
        "showLegend": True,
        "legendDisplay": "top",
        "showPlotValuesInLegend": True,
        "showXAxisLabels": True,
        "showYAxisLabels": True,
        "isSynced": True,
        # Use the per-file 0-based sim_time_s as the X axis (populated by
        # the firmware). Switch xAxisVal to "timestamp" for wall time.
        "xAxisVal": "custom",
        "xAxisPath": {
            "value": f"{TOPIC}.sim_time_s",
            "enabled": True,
            "timestampMethod": "receiveTime",
        },
        "sidebarDimension": 240,
    }


def main(out):
    gyro_id = "Plot!gyro"
    jvel_id = "Plot!jvel"
    quat_id = "Plot!quat"
    jpos_id = "Plot!jpos"

    config_by_id = {
        gyro_id: plot(
            [
                path("ang_vel_x", "wx (roll rate)"),
                path("ang_vel_y", "wy (pitch rate)"),
                path("ang_vel_z", "wz (yaw rate)"),
            ],
            title="IMU angular velocity (rad/s) - gyro noise",
        ),
        jvel_id: plot(
            [path(f"joint_vel_rad_s[{i}]", f"{JOINTS[i]} vel") for i in range(8)],
            title="Joint velocity (rad/s) - noisiest channel",
        ),
        quat_id: plot(
            [
                path("quat_x", "qx"),
                path("quat_y", "qy"),
                path("quat_z", "qz"),
                path("quat_w", "qw"),
            ],
            title="IMU orientation quaternion (XYZW) - drift",
        ),
        jpos_id: plot(
            [path(f"joint_pos_rad[{i}]", f"{JOINTS[i]} pos") for i in range(8)],
            title="Joint position (rad) - encoder noise",
        ),
    }

    # 2x2 mosaic: left column = gyro / joint-vel, right column = quat / joint-pos.
    layout = {
        "direction": "row",
        "first": {
            "direction": "column",
            "first": gyro_id,
            "second": jvel_id,
            "splitPercentage": 50,
        },
        "second": {
            "direction": "column",
            "first": quat_id,
            "second": jpos_id,
            "splitPercentage": 50,
        },
        "splitPercentage": 50,
    }

    doc = {
        "configById": config_by_id,
        "globalVariables": {},
        "userNodes": {},
        "playbackConfig": {"speed": 1},
        "layout": layout,
    }
>>>>>>> 1b797a4 (chore(foxglove): update bebop_noise_layout)

    _, builder = LAYOUTS[name]
    with open(out, "w") as f:
        json.dump(builder(), f, indent=2)
        f.write("\n")
    print(f"wrote {out}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate one or more Foxglove layout JSON files."
    )
    parser.add_argument(
        "--layout",
        choices=[*LAYOUTS.keys(), "all"],
        default="robot",
        help="Layout to generate. Defaults to robot.",
    )
    parser.add_argument(
        "--out",
        help="Output JSON path. Only valid when generating a single layout.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing layout file.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if args.layout == "all":
        if args.out:
            raise SystemExit("--out can only be used with a single layout")
        for name, (filename, _) in LAYOUTS.items():
            write_layout(name, os.path.join(script_dir, filename), force=args.force)
        return

    default_filename, _ = LAYOUTS[args.layout]
    out = args.out or os.path.join(script_dir, default_filename)
    write_layout(args.layout, out, force=args.force)


if __name__ == "__main__":
    main()
