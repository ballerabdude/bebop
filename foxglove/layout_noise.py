"""Foxglove layout: 2x2 policy_capture noise-review plots."""

from layout_common import foxglove_doc, plot, policy_path

NAME = "noise"
OUTPUT = "bebop_noise_layout.json"

# Short labels used in the original noise layout.
JOINTS = [
    "hip_flex_L", "hip_flex_R", "hip_abd_L", "hip_abd_R",
    "knee_L", "knee_R", "foot_L", "foot_R",
]


def build():
    gyro_id = "Plot!gyro"
    jvel_id = "Plot!jvel"
    quat_id = "Plot!quat"
    jpos_id = "Plot!jpos"

    config_by_id = {
        gyro_id: plot(
            [
                policy_path("ang_vel_x", "wx (roll rate)"),
                policy_path("ang_vel_y", "wy (pitch rate)"),
                policy_path("ang_vel_z", "wz (yaw rate)"),
            ],
            title="IMU angular velocity (rad/s) - gyro noise",
        ),
        jvel_id: plot(
            [policy_path(f"joint_vel_rad_s[{i}]", JOINTS[i]) for i in range(8)],
            title="Joint velocity (rad/s) - noisiest channel",
        ),
        quat_id: plot(
            [
                policy_path("quat_x", "qx"),
                policy_path("quat_y", "qy"),
                policy_path("quat_z", "qz"),
                policy_path("quat_w", "qw"),
            ],
            title="IMU orientation quaternion (XYZW) - drift",
        ),
        jpos_id: plot(
            [policy_path(f"joint_pos_rad[{i}]", JOINTS[i]) for i in range(8)],
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
