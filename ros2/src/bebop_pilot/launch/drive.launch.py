"""Drive stack: DualSense pilot + trained Bebop V2 policy runner.

Assumes /imu/data and /joint_states are already being published (e.g. by
``bringup.launch.py`` or a separately running bno085_imu node + Teensy).

Pilot publishes /cmd_vel and /policy_enable; policy_runner consumes both
and produces 8-joint /joint_commands for the motor controller.

Example::

    ros2 launch bebop_pilot drive.launch.py \\
        model_path:=/home/jetson/models/policy.onnx
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    pkg_share = FindPackageShare("bebop_pilot")

    default_pilot_params = PathJoinSubstitution(
        [pkg_share, "config", "ps5_dualsense.yaml"]
    )
    default_runner_params = PathJoinSubstitution(
        [pkg_share, "config", "policy_runner.yaml"]
    )

    pilot_params_arg = DeclareLaunchArgument(
        "pilot_params_file",
        default_value=default_pilot_params,
        description="DualSense pilot parameter YAML.",
    )
    runner_params_arg = DeclareLaunchArgument(
        "runner_params_file",
        default_value=default_runner_params,
        description="Policy runner parameter YAML (joint limits + lin_vel source).",
    )
    model_path_arg = DeclareLaunchArgument(
        "model_path",
        description="Absolute path to the trained Bebop V2 policy.onnx file.",
    )
    control_rate_arg = DeclareLaunchArgument(
        "control_rate",
        default_value="50.0",
        description="Policy control loop rate (Hz).",
    )

    pilot_node = Node(
        package="bebop_pilot",
        executable="pilot_node.py",
        name="pilot_node",
        output="screen",
        parameters=[LaunchConfiguration("pilot_params_file")],
        emulate_tty=True,
    )

    policy_runner = Node(
        package="bebop_pilot",
        executable="policy_runner.py",
        name="policy_runner",
        output="screen",
        parameters=[
            LaunchConfiguration("runner_params_file"),
            {
                "model_path": LaunchConfiguration("model_path"),
                "control_rate": LaunchConfiguration("control_rate"),
            },
        ],
        emulate_tty=True,
    )

    return LaunchDescription(
        [
            pilot_params_arg,
            runner_params_arg,
            model_path_arg,
            control_rate_arg,
            pilot_node,
            policy_runner,
        ]
    )
