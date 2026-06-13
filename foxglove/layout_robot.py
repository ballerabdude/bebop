"""Foxglove layout: 3D URDF panel plus policy_capture position/IMU plots."""

from layout_common import JOINTS, foxglove_doc, plot, policy_path

NAME = "robot"
OUTPUT = "bebop_robot_layout.json"

MCAP_PATH = "/Users/ahagi/Downloads/policy_capture_20260612_032423.mcap"
URDF_PATH = (
    "/Users/ahagi/Documents/projects/bebop/ros2/src/"
    "bebopv2_description/urdf/bebopv2.urdf"
)
JOINT_STATES_TOPIC = "/joint_states"


def _robot_3d_panel():
    """Foxglove 3D panel config for the local Bebop URDF."""
    return {
        "title": "Bebop robot - URDF",
        "fixedFrame": "base_link",
        "followTf": "base_link",
        "followMode": "follow-none",
        "scene": {},
        "transforms": {
            "base_link": {
                "visible": True,
                "showLabel": True,
                "axisScale": 0.15,
                "lineWidth": 1,
            },
        },
        "layers": {
            "grid": {
                "instanceId": "grid",
                "layerId": "foxglove.Grid",
                "label": "Ground grid",
                "visible": True,
                "size": 1,
                "divisions": 10,
                "lineWidth": 1,
                "color": "#64748b",
                "position": [0, 0, -0.68],
            },
            "bebop_urdf": {
                "instanceId": "bebop_urdf",
                "layerId": "foxglove.Urdf",
                "label": "Bebop V2 URDF",
                "visible": True,
                "sourceType": "filePath",
                "filePath": URDF_PATH,
                "controlMode": "jointStates",
                "jointStatesTopic": JOINT_STATES_TOPIC,
                "displayMode": "visual",
                "fallbackColor": "#94a3b8",
                "opacity": 1,
                "showOutlines": True,
                "showAxis": False,
                "axisScale": 0.1,
            },
        },
        "topics": {
            JOINT_STATES_TOPIC: {"visible": True},
        },
    }


def _markdown_panel():
    return {
        "markdown": "\n".join(
            [
                "# Bebop MCAP Review",
                "",
                f"Open MCAP: `{MCAP_PATH}`",
                "",
                f"URDF: `{URDF_PATH}`",
                "",
                "Policy capture topic: `/policy_capture`",
                "",
                "The plots read joint positions directly from "
                "`/policy_capture.joint_pos_rad[]`.",
                "",
                "The 3D URDF layer is configured for `/joint_states`. If this "
                "MCAP does not include that topic, the robot will render in its "
                "default pose while the position plots show the recorded motion.",
            ]
        )
    }


def build():
    robot_id = "3D!robot"
    jpos_id = "Plot!jpos"
    jvel_id = "Plot!jvel"
    quat_id = "Plot!quat"
    gyro_id = "Plot!gyro"
    notes_id = "Markdown!notes"

    config_by_id = {
        robot_id: _robot_3d_panel(),
        notes_id: _markdown_panel(),
        jpos_id: plot(
            [policy_path(f"joint_pos_rad[{i}]", JOINTS[i]) for i in range(8)],
            title="Recorded joint position (rad)",
        ),
        jvel_id: plot(
            [policy_path(f"joint_vel_rad_s[{i}]", JOINTS[i]) for i in range(8)],
            title="Recorded joint velocity (rad/s)",
        ),
        quat_id: plot(
            [
                policy_path("quat_x", "qx"),
                policy_path("quat_y", "qy"),
                policy_path("quat_z", "qz"),
                policy_path("quat_w", "qw"),
            ],
            title="IMU orientation quaternion (XYZW)",
        ),
        gyro_id: plot(
            [
                policy_path("ang_vel_x", "wx (roll rate)"),
                policy_path("ang_vel_y", "wy (pitch rate)"),
                policy_path("ang_vel_z", "wz (yaw rate)"),
            ],
            title="IMU angular velocity (rad/s)",
        ),
    }

    # Top = robot render + notes. Bottom = recorded position data and IMU plots.
    layout = {
        "direction": "column",
        "first": {
            "direction": "row",
            "first": robot_id,
            "second": notes_id,
            "splitPercentage": 72,
        },
        "second": {
            "direction": "row",
            "first": {
                "direction": "column",
                "first": jpos_id,
                "second": jvel_id,
                "splitPercentage": 55,
            },
            "second": {
                "direction": "column",
                "first": quat_id,
                "second": gyro_id,
                "splitPercentage": 50,
            },
            "splitPercentage": 50,
        },
        "splitPercentage": 58,
    }

    return foxglove_doc(config_by_id, layout)


if __name__ == "__main__":
    import argparse
    import json
    import os

    parser = argparse.ArgumentParser(description=f"Generate {OUTPUT}")
    parser.add_argument("--out", help=f"Output path (default: {OUTPUT})")
    parser.add_argument("--force", action="store_true", help="Overwrite existing file")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out = args.out or os.path.join(script_dir, OUTPUT)
    if os.path.exists(out) and not args.force:
        raise SystemExit(f"{out} exists; pass --force to overwrite")

    with open(out, "w") as f:
        json.dump(build(), f, indent=2)
        f.write("\n")
    print(f"wrote {out}")
