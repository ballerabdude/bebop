"""Shared helpers for generating Foxglove layout JSON.

Now targeting ROS2 CDR-encoded MCAP (profile=ros2, schema_encoding=ros2msg).

Channel layout (see firmware/bebop-linux/src/policy_capture.rs):

  /joint_states       -> sensor_msgs/msg/JointState  (8 joints)
  /imu                -> sensor_msgs/msg/Imu
  /policy/status      -> bebop_msgs/msg/PolicyStatus (mode, dry_run, imu_live, sim_time_s)
  /policy/observation -> bebop_msgs/msg/Float32Stamped
  /policy/action      -> bebop_msgs/msg/PolicyAction

Foxglove ROS2 panel resolves CDR-encoded standard types by their .msg
field names, so the plot paths use e.g. /joint_states.position[0].
"""

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


def jpos_path(index, label):
    """Plot path for joint position (float64[] from sensor_msgs/JointState)."""
    return {
        "value": f"/joint_states.position[{index}]",
        "enabled": True,
        "label": label,
        "timestampMethod": "receiveTime",
    }


def jvel_path(index, label):
    """Plot path for joint velocity (float64[] from sensor_msgs/JointState)."""
    return {
        "value": f"/joint_states.velocity[{index}]",
        "enabled": True,
        "label": label,
        "timestampMethod": "receiveTime",
    }


def imu_quat_path(axis, label):
    """Plot path for IMU quaternion component (geometry_msgs/Quaternion)."""
    return {
        "value": f"/imu.orientation.{axis}",
        "enabled": True,
        "label": label,
        "timestampMethod": "receiveTime",
    }


def imu_gyro_path(axis, label):
    """Plot path for IMU angular velocity component (geometry_msgs/Vector3)."""
    return {
        "value": f"/imu.angular_velocity.{axis}",
        "enabled": True,
        "label": label,
        "timestampMethod": "receiveTime",
    }


def status_path(field, label):
    """Plot path for a scalar field in /policy/status."""
    return {
        "value": f"/policy/status.{field}",
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
        # Plot against playback time so the panel shows a moving cursor that
        # tracks playback (a custom message-path X axis renders the whole
        # series statically with no playhead animation).
        "xAxisVal": "timestamp",
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
