"""Foxglove layout: navd MCAP session review (color, depth preview, BEV map,
teleop twist / odometry plots).

Channels are the JSON-encoded ones written by
bebop-vision/bebop_vision/recorder_mcap.py (see its module docstring).
Import bebop_navd_layout.json via Foxglove Desktop: Layouts -> Import
from file, then open a navd session .mcap.
"""

from layout_common import foxglove_doc, plot

NAME = "navd"
OUTPUT = "bebop_navd_layout.json"


def image(topic, title, **extra):
    # Topic binding has changed across Foxglove generations. Current builds
    # (like the LeRobot reference layout in foxglove-sdk) use
    # imageMode: {imageTopic}; older builds used imageTopic / topic /
    # imageTopics. Set all of them so the panel binds anywhere.
    return {
        "foxglovePanelTitle": title,
        "imageMode": {"imageTopic": topic, **extra},
        "imageTopics": [{"topic": topic}],
        "topic": topic,
        "imageTopic": topic,
        "mode": "image",
    }


def state_path(topic, field, label):
    return {
        "value": f"{topic}.{field}",
        "enabled": True,
        "label": label,
        "timestampMethod": "receiveTime",
    }


def build():
    config_by_id = {
        "Image!color": image("/color_near", "Color (near)"),
        "Image!depth": image("/depth_near_preview", "Depth preview (near)",
                             minValue=300, maxValue=6000, colormap="turbo"),
        "Image!map": image("/bev_map", "BEV teacher map (60x60, 3x3 m)"),
        "Plot!cmd": plot([
            state_path("/cmd_vel", "vx", "vx (m/s)"),
            state_path("/cmd_vel", "wz", "wz (rad/s)"),
        ], title="Teleop twist (imitation label)"),
        "Plot!odom": plot([
            state_path("/odom", "x", "x (m)"),
            state_path("/odom", "y", "y (m)"),
        ], title="Odometry (m)"),
        "Plot!goal": plot([
            state_path("/goal", "heading_rad", "heading goal (rad)"),
            state_path("/odom", "theta", "odom theta (rad)"),
        ], title="Goal vs heading"),
        "Plot!plane": plot([
            state_path("/bev_teacher.plane_ok", "near", "plane_ok near"),
            state_path("/bev_teacher.plane_ok", "far", "plane_ok far"),
        ], title="Ground-plane fit health"),
    }

    # Left column: the two camera-ish views; right column: BEV map on top,
    # plots below (2x2 mosaic).
    layout = {
        "direction": "row",
        "first": {
            "direction": "column",
            "first": "Image!color",
            "second": "Image!depth",
            "splitPercentage": 55,
        },
        "second": {
            "direction": "column",
            "first": "Image!map",
            "second": {
                "direction": "row",
                "first": {
                    "direction": "column",
                    "first": "Plot!cmd",
                    "second": "Plot!odom",
                    "splitPercentage": 50,
                },
                "second": {
                    "direction": "column",
                    "first": "Plot!goal",
                    "second": "Plot!plane",
                    "splitPercentage": 50,
                },
            },
            "splitPercentage": 45,
        },
        "splitPercentage": 45,
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
