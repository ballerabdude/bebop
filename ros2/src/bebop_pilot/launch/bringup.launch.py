"""Full Bebop V2 deployment stack on the Jetson.

Brings up, in this order:
    1. bno085_imu      ->  /imu/data
    2. pilot_node      ->  /cmd_vel + /policy_enable (DualSense)
    3. policy_runner   ->  /joint_commands (consumes /imu/data + /joint_states)

The Teensy / motor-controller side (publishing /joint_states and consuming
/joint_commands) is expected to be running separately, since it usually
ships as its own micro-ROS agent or hardware bringup.

Example::

    ros2 launch bebop_pilot bringup.launch.py \\
        model_path:=/home/jetson/models/policy.onnx

Disable the IMU node (e.g. if you're feeding /imu/data from elsewhere)::

    ros2 launch bebop_pilot bringup.launch.py \\
        model_path:=... \\
        launch_imu:=false
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
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
    default_motor_params = PathJoinSubstitution(
        [pkg_share, "config", "motor_bus.yaml"]
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
    motor_params_arg = DeclareLaunchArgument(
        "motor_params_file",
        default_value=default_motor_params,
        description="Motor bus parameter YAML (CAN IDs, models, gains).",
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
    imu_rate_arg = DeclareLaunchArgument(
        "imu_rate",
        default_value="100.0",
        description="BNO085 publish rate (Hz).",
    )
    launch_imu_arg = DeclareLaunchArgument(
        "launch_imu",
        default_value="true",
        description="Set false if /imu/data is published by another process.",
    )
    launch_motors_arg = DeclareLaunchArgument(
        "launch_motors",
        default_value="true",
        description="Set false to skip the CAN motor bus (e.g. dry runs without HW).",
    )

    imu_node = Node(
        package="bebop_pilot",
        executable="bno085_imu.py",
        name="bno085_imu",
        output="screen",
        emulate_tty=True,
        parameters=[{"publish_rate": LaunchConfiguration("imu_rate")}],
        condition=IfCondition(LaunchConfiguration("launch_imu")),
    )

    motor_bus = Node(
        package="bebop_pilot",
        executable="motor_bus.py",
        name="motor_bus",
        output="screen",
        emulate_tty=True,
        parameters=[LaunchConfiguration("motor_params_file")],
        condition=IfCondition(LaunchConfiguration("launch_motors")),
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
            motor_params_arg,
            model_path_arg,
            control_rate_arg,
            imu_rate_arg,
            launch_imu_arg,
            launch_motors_arg,
            imu_node,
            motor_bus,
            pilot_node,
            policy_runner,
        ]
    )
