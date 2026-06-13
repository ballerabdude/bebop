"""Shared helpers for generating Foxglove layout JSON."""

TOPIC = "/policy_capture"

# Joint slot order, matching firmware/bebop-linux/src/observation.rs::JOINT_NAMES
# and the revolute joints in ros2/src/bebopv2_description/urdf/bebopv2.urdf.
JOINTS = [
    "hip_flexion_left_joint",
    "hip_flexion_right_joint",
    "hip_abduction_left_joint",
    "hip_abduction_right_joint",
    "knee_flexion_left_joint",
    "knee_flexion_right_joint",
    "foot_left_joint",
    "foot_right_joint",
]


def policy_path(field, label):
    """One plot series: a message path with a human-readable legend label."""
    return {
        "value": f"{TOPIC}.{field}",
        "enabled": True,
        "label": label,
        "timestampMethod": "receiveTime",
    }


def plot(paths, *, title):
    return {
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


def foxglove_doc(config_by_id, layout):
    return {
        "configById": config_by_id,
        "globalVariables": {},
        "userNodes": {},
        "playbackConfig": {"speed": 1},
        "layout": layout,
    }
