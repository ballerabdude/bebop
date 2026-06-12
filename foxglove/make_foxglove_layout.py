#!/usr/bin/env python3
"""Generate a Foxglove Studio layout (.json) for reviewing bebop
policy_capture noise.

Foxglove layouts are plain JSON. The top-level object has:
  - configById : per-panel config keyed by "<PanelType>!<id>"
  - layout     : a mosaic tree of those panel ids (or a single id)
  - globalVariables / userNodes / playbackConfig

Each Plot panel gets a descriptive `title` (shown in the panel toolbar)
and each series a short `label` (shown in the legend) so it's obvious
what every plot and trace is.

Import the emitted file via Foxglove's layouts sidebar -> "Import from
file". Works in both the desktop app and app.foxglove.dev.

Usage:
    python3 make_foxglove_layout.py [out.json]   # defaults to ./bebop_noise_layout.json
"""
import json
import os
import sys

TOPIC = "/policy_capture"

# Joint slot order (see firmware/bebop-linux/src/observation.rs::JOINT_NAMES).
JOINTS = [
    "hip_flex_L", "hip_flex_R", "hip_abd_L", "hip_abd_R",
    "knee_L", "knee_R", "foot_L", "foot_R",
]


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
        # `title` shows in the panel toolbar so the user can tell the
        # four plots apart at a glance.
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
            [path(f"joint_vel_rad_s[{i}]", JOINTS[i]) for i in range(8)],
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
            [path(f"joint_pos_rad[{i}]", JOINTS[i]) for i in range(8)],
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

    with open(out, "w") as f:
        json.dump(doc, f, indent=2)
        f.write("\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    default_out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "bebop_noise_layout.json")
    main(sys.argv[1] if len(sys.argv) > 1 else default_out)
