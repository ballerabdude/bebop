"""Foxglove layout: 2x2 noise-review plots from ROS2 MCAP channels."""

from layout_common import (
    foxglove_doc, plot,
    jpos_path, jvel_path,
    imu_quat_path, imu_gyro_path,
)

NAME = "noise"
OUTPUT = "bebop_noise_layout.json"

# Short labels for the noise layout legend.
SHORT_JOINTS = [
    "hip_flex_L", "hip_flex_R", "hip_abd_L", "hip_abd_R",
    "knee_L", "knee_R", "foot_L", "foot_R",
]


def build():
    config_by_id = {
        "Plot!gyro": plot([
            imu_gyro_path("x", "wx (roll rate)"),
            imu_gyro_path("y", "wy (pitch rate)"),
            imu_gyro_path("z", "wz (yaw rate)"),
        ], title="IMU angular velocity (rad/s) - gyro noise"),
        "Plot!jvel": plot(
            [jvel_path(i, SHORT_JOINTS[i]) for i in range(8)],
            title="Joint velocity (rad/s) - noisiest channel",
        ),
        "Plot!quat": plot([
            imu_quat_path("x", "qx"),
            imu_quat_path("y", "qy"),
            imu_quat_path("z", "qz"),
            imu_quat_path("w", "qw"),
        ], title="IMU orientation quaternion (XYZW) - drift"),
        "Plot!jpos": plot(
            [jpos_path(i, SHORT_JOINTS[i]) for i in range(8)],
            title="Joint position (rad) - encoder noise",
        ),
    }

    layout = {
        "direction": "row",
        "first": {
            "direction": "column",
            "first": "Plot!gyro",
            "second": "Plot!jvel",
            "splitPercentage": 50,
        },
        "second": {
            "direction": "column",
            "first": "Plot!quat",
            "second": "Plot!jpos",
            "splitPercentage": 50,
        },
        "splitPercentage": 50,
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
