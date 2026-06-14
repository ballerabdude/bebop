"""Foxglove layout: 3D URDF panel plus joint/IMU plots from ROS2 MCAP."""

from layout_common import (
    JOINTS, foxglove_doc, plot,
    jpos_path, jvel_path,
    imu_quat_path, imu_gyro_path,
)

NAME = "robot"
OUTPUT = "bebop_robot_layout.json"

URDF_PATH = (
    "/Users/ahagi/Documents/projects/bebop/ros2/src/"
    "bebopv2_description/urdf/bebopv2.urdf"
)
JOINT_STATES_TOPIC = "/joint_states"


def _robot_3d_panel():
    return {
        "title": "Bebop robot - URDF",
        "fixedFrame": "base_link",
        "followTf": "base_link",
        "followMode": "follow-none",
        # Robot meshes use Z-up (ROS convention); without this Foxglove
        # assumes COLLADA Y-up and the model appears lying on its side.
        "scene": {"meshUpAxis": "z_up"},
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


def build():
    robot_id = "3D!robot"
    jpos_id = "Plot!jpos"
    jvel_id = "Plot!jvel"
    quat_id = "Plot!quat"
    gyro_id = "Plot!gyro"

    config_by_id = {
        robot_id: _robot_3d_panel(),
        jpos_id: plot(
            [jpos_path(i, JOINTS[i]) for i in range(8)],
            title="Joint positions (rad)",
        ),
        jvel_id: plot(
            [jvel_path(i, JOINTS[i]) for i in range(8)],
            title="Joint velocities (rad/s)",
        ),
        quat_id: plot([
            imu_quat_path("x", "qx"),
            imu_quat_path("y", "qy"),
            imu_quat_path("z", "qz"),
            imu_quat_path("w", "qw"),
        ], title="IMU orientation quaternion (XYZW)"),
        gyro_id: plot([
            imu_gyro_path("x", "wx (roll rate)"),
            imu_gyro_path("y", "wy (pitch rate)"),
            imu_gyro_path("z", "wz (yaw rate)"),
        ], title="IMU angular velocity (rad/s)"),
    }

    # 3D robot on the left; the four titled plots stacked in equal quarters
    # down the right-hand column.
    plots_stack = {
        "direction": "column",
        "first": jpos_id,
        "second": {
            "direction": "column",
            "first": jvel_id,
            "second": {
                "direction": "column",
                "first": quat_id,
                "second": gyro_id,
                "splitPercentage": 50,
            },
            "splitPercentage": 100 / 3,
        },
        "splitPercentage": 25,
    }

    layout = {
        "direction": "row",
        "first": robot_id,
        "second": plots_stack,
        "splitPercentage": 55,
    }

    return foxglove_doc(config_by_id, layout)


if __name__ == "__main__":
    import argparse, json, os
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
