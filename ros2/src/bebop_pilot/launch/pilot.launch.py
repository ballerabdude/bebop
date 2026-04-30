"""Launch only the DualSense pilot node.

For the full driving stack (pilot + policy_runner), use ``drive.launch.py``.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    pkg_share = FindPackageShare("bebop_pilot")
    default_params = PathJoinSubstitution(
        [pkg_share, "config", "ps5_dualsense.yaml"]
    )

    params_arg = DeclareLaunchArgument(
        "params_file",
        default_value=default_params,
        description="Path to the pilot parameter YAML.",
    )

    pilot_node = Node(
        package="bebop_pilot",
        executable="pilot_node.py",
        name="pilot_node",
        output="screen",
        parameters=[LaunchConfiguration("params_file")],
        emulate_tty=True,
    )

    return LaunchDescription([params_arg, pilot_node])
