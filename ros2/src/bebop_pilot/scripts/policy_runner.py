#!/usr/bin/env python3
"""Policy Runner — Bebop V2 humanoid (8-DoF, position-only).

Runs a trained ONNX locomotion / balance policy whose training config lives in
``bebop_training/envs/bebop_v2_base_cfg.py``.

    - 8 joints (hip abduction, femur, shin, foot — both sides), no wheels.
    - Pure position control (single ``JointPositionActionCfg`` with scale 0.8,
      ``use_default_offset=True``).
    - Observation order matches ``PolicyCfg``:

        [base_lin_vel(3), base_ang_vel(3), projected_gravity(3),
         joint_pos_rel(8), joint_vel(8), last_action(8), velocity_cmd(3)]
        = 36 dims

    - ``base_lin_vel`` cannot be measured from an IMU alone. Three strategies
      are supported via the ``lin_vel_source`` parameter:
          ``zeros``    — publish zeros (safe; mildly out of distribution).
          ``odom``     — subscribe to nav_msgs/Odometry on ``odom_topic``.
          ``twist``    — subscribe to geometry_msgs/TwistStamped on
                         ``twist_topic`` (e.g. from a separate state estimator).
      For first bring-up, leave ``lin_vel_source: zeros`` and command low
      velocities (≤ 0.3 m/s).

    - Joint default positions and min/max safety limits are configurable. The
      defaults represent the training "home" pose expressed in the **real
      robot's encoder frame**. Output positions are clamped to
      ``[joint_min_pos, joint_max_pos]`` before publishing.

Subscribes:
    /joint_states (sensor_msgs/JointState)   - real-robot joint feedback
    /imu/data     (sensor_msgs/Imu)          - BNO085 orientation + gyro
    /cmd_vel      (geometry_msgs/Twist)      - commands from pilot_node
    /policy_enable (std_msgs/Bool)           - safety gate
    [optional] odom / twist topic            - external linear vel estimate

Publishes:
    /joint_commands (sensor_msgs/JointState) - 8 position targets

Run::

    ros2 run bebop_pilot policy_runner.py --ros-args \\
        --params-file install/bebop_pilot/share/bebop_pilot/config/policy_runner.yaml \\
        -p model_path:=/path/to/policy.onnx
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Twist, TwistStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool

try:
    import onnxruntime as ort
except ImportError as e:
    raise SystemExit(
        "onnxruntime is required. On Jetson: pip install onnxruntime-gpu "
        "(or `onnxruntime` for CPU)."
    ) from e


# Joint order MUST match bebop_training/envs/bebop_v2_base_cfg.py:JOINT_NAMES_ALL.
JOINT_NAMES: List[str] = [
    "hip_abduction_left_joint",
    "hip_abduction_right_joint",
    "femur_left_joint",
    "femur_right_joint",
    "shin_left_joint",
    "shin_right_joint",
    "foot_left_joint",
    "foot_right_joint",
]
NUM_JOINTS = len(JOINT_NAMES)

# Action scale: matches ActionsCfg.joints_pos.scale (bebop_v2_base_cfg.py:128).
DEFAULT_ACTION_SCALE = 0.8

# Observation dimensions (sum = 36).
OBS_DIM = 3 + 3 + 3 + NUM_JOINTS + NUM_JOINTS + NUM_JOINTS + 3


class PolicyRunner(Node):
    """Runs a trained 8-DoF Bebop V2 policy on the real robot."""

    def __init__(self) -> None:
        super().__init__("policy_runner")

        # ----- parameters ----------------------------------------------------
        self.declare_parameter("model_path", "")
        self.declare_parameter("normalizer_path", "")
        self.declare_parameter("control_rate", 50.0)
        self.declare_parameter("enabled_default", False)

        self.declare_parameter("action_scale", DEFAULT_ACTION_SCALE)

        self.declare_parameter("max_lin_vel", 0.6)
        self.declare_parameter("max_ang_vel", 1.0)

        self.declare_parameter("joint_default_pos", [0.0] * NUM_JOINTS)
        self.declare_parameter("joint_min_pos", [-math.pi] * NUM_JOINTS)
        self.declare_parameter("joint_max_pos", [math.pi] * NUM_JOINTS)

        self.declare_parameter("lin_vel_source", "zeros")  # zeros | odom | twist
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("twist_topic", "/base_twist")

        self.declare_parameter("data_timeout", 0.1)

        model_path: str = self.get_parameter("model_path").value
        normalizer_path: str = self.get_parameter("normalizer_path").value
        self.control_rate: float = float(self.get_parameter("control_rate").value)
        self.enabled: bool = bool(self.get_parameter("enabled_default").value)
        self.action_scale: float = float(self.get_parameter("action_scale").value)
        self.max_lin_vel: float = float(self.get_parameter("max_lin_vel").value)
        self.max_ang_vel: float = float(self.get_parameter("max_ang_vel").value)
        self.lin_vel_source: str = str(self.get_parameter("lin_vel_source").value)
        self.data_timeout: float = float(self.get_parameter("data_timeout").value)

        self.joint_default_pos = self._param_as_array("joint_default_pos")
        self.joint_min_pos = self._param_as_array("joint_min_pos")
        self.joint_max_pos = self._param_as_array("joint_max_pos")

        if not model_path:
            raise ValueError(
                "model_path is required. Pass --ros-args -p model_path:=/path/to/policy.onnx"
            )

        if not normalizer_path:
            normalizer_path = str(Path(model_path).with_suffix(".norm.npz"))

        # ----- model ---------------------------------------------------------
        self.get_logger().info(f"Loading ONNX model: {model_path}")
        providers = self._select_providers()
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        model_obs_dim = int(self.session.get_inputs()[0].shape[1])
        if model_obs_dim != OBS_DIM:
            raise ValueError(
                f"Model expects obs_dim={model_obs_dim} but the V2 runner builds "
                f"obs_dim={OBS_DIM}. The checkpoint was likely trained with a "
                "different observation group (e.g. v1 runner is 30, blind v2 is 33). "
                "Check bebop_training/envs/bebop_v2_base_cfg.py:ObservationsCfg."
            )

        model_action_dim = int(self.session.get_outputs()[0].shape[1])
        if model_action_dim != NUM_JOINTS:
            raise ValueError(
                f"Model emits action_dim={model_action_dim}, expected {NUM_JOINTS} "
                "for V2."
            )

        # ----- normalizer ----------------------------------------------------
        self.obs_mean = np.zeros(OBS_DIM, dtype=np.float32)
        self.obs_std = np.ones(OBS_DIM, dtype=np.float32)
        self.obs_normalization_enabled = False
        if Path(normalizer_path).exists():
            try:
                norm_data = np.load(normalizer_path)
                if "mean" in norm_data:
                    self.obs_mean = norm_data["mean"].astype(np.float32)
                if "std" in norm_data:
                    self.obs_std = np.maximum(
                        norm_data["std"].astype(np.float32), 1e-6
                    )
                self.obs_normalization_enabled = True
                self.get_logger().info(f"Loaded obs normalizer: {normalizer_path}")
            except Exception as e:
                self.get_logger().warn(f"Failed to load normalizer: {e}")
        else:
            self.get_logger().warn(
                f"Normalizer file not found at {normalizer_path}; using raw obs."
            )

        # ----- state ---------------------------------------------------------
        self.joint_positions = np.zeros(NUM_JOINTS, dtype=np.float32)
        self.joint_velocities = np.zeros(NUM_JOINTS, dtype=np.float32)
        self.base_lin_vel = np.zeros(3, dtype=np.float32)
        self.base_ang_vel = np.zeros(3, dtype=np.float32)
        self.projected_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self.velocity_command = np.zeros(3, dtype=np.float32)
        self.previous_actions = np.zeros(NUM_JOINTS, dtype=np.float32)

        self.last_joint_state_time: Optional[float] = None
        self.last_imu_time: Optional[float] = None
        self.last_lin_vel_time: Optional[float] = None

        if self.lin_vel_source == "zeros":
            self.get_logger().warn(
                "lin_vel_source=zeros — base_lin_vel will be reported as 0. "
                "Policy is slightly out of distribution. Keep commanded speeds low."
            )
        elif self.lin_vel_source not in ("odom", "twist"):
            raise ValueError(
                f"Unknown lin_vel_source '{self.lin_vel_source}'. "
                "Expected one of: zeros, odom, twist."
            )

        # ----- I/O -----------------------------------------------------------
        self.joint_sub = self.create_subscription(
            JointState, "/joint_states", self._joint_state_cb, 10
        )
        self.imu_sub = self.create_subscription(
            Imu, "/imu/data", self._imu_cb, 10
        )
        self.cmd_vel_sub = self.create_subscription(
            Twist, "/cmd_vel", self._cmd_vel_cb, 10
        )
        self.enable_sub = self.create_subscription(
            Bool, "/policy_enable", self._enable_cb, 10
        )

        if self.lin_vel_source == "odom":
            topic = str(self.get_parameter("odom_topic").value)
            self.create_subscription(Odometry, topic, self._odom_cb, 10)
            self.get_logger().info(f"Linear velocity source: {topic} (Odometry)")
        elif self.lin_vel_source == "twist":
            topic = str(self.get_parameter("twist_topic").value)
            self.create_subscription(TwistStamped, topic, self._twist_cb, 10)
            self.get_logger().info(f"Linear velocity source: {topic} (TwistStamped)")

        self.cmd_pub = self.create_publisher(JointState, "/joint_commands", 10)

        period = 1.0 / max(self.control_rate, 1.0)
        self.timer = self.create_timer(period, self._control_loop)

        self.get_logger().info(
            f"Policy runner v2 ready @ {self.control_rate} Hz (obs={OBS_DIM}, "
            f"action={NUM_JOINTS}). Waiting for /joint_states and /imu/data..."
        )

    # ------------------------------------------------------------------ helpers
    def _param_as_array(self, name: str) -> np.ndarray:
        values = list(self.get_parameter(name).value)
        if len(values) != NUM_JOINTS:
            raise ValueError(
                f"Parameter '{name}' must have {NUM_JOINTS} entries, got {len(values)}."
            )
        return np.asarray(values, dtype=np.float32)

    def _select_providers(self) -> List:
        available = ort.get_available_providers()
        preferred = []
        if "TensorrtExecutionProvider" in available:
            preferred.append(
                (
                    "TensorrtExecutionProvider",
                    {"trt_fp16_enable": True, "trt_engine_cache_enable": True},
                )
            )
        if "CUDAExecutionProvider" in available:
            preferred.append("CUDAExecutionProvider")
        preferred.append("CPUExecutionProvider")
        return preferred

    @staticmethod
    def _quat_rotate_inverse(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Rotate v from world frame into body frame using quaternion q (w,x,y,z)."""
        w, x, y, z = q_wxyz
        qx, qy, qz = -x, -y, -z
        vx, vy, vz = v
        cx = qy * vz - qz * vy
        cy = qz * vx - qx * vz
        cz = qx * vy - qy * vx
        cx2 = qy * cz - qz * cy
        cy2 = qz * cx - qx * cz
        cz2 = qx * cy - qy * cx
        return np.array(
            [
                vx + 2.0 * w * cx + 2.0 * cx2,
                vy + 2.0 * w * cy + 2.0 * cy2,
                vz + 2.0 * w * cz + 2.0 * cz2,
            ],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------ callbacks
    def _joint_state_cb(self, msg: JointState) -> None:
        self.last_joint_state_time = self._now()
        for i, name in enumerate(msg.name):
            if name in JOINT_NAMES:
                idx = JOINT_NAMES.index(name)
                if i < len(msg.position):
                    self.joint_positions[idx] = msg.position[i]
                if i < len(msg.velocity):
                    self.joint_velocities[idx] = msg.velocity[i]

    def _imu_cb(self, msg: Imu) -> None:
        self.last_imu_time = self._now()
        self.base_ang_vel[:] = (
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z,
        )
        q = msg.orientation
        self.projected_gravity = self._quat_rotate_inverse(
            np.array([q.w, q.x, q.y, q.z], dtype=np.float32),
            np.array([0.0, 0.0, -1.0], dtype=np.float32),
        )

    def _cmd_vel_cb(self, msg: Twist) -> None:
        self.velocity_command[0] = float(np.clip(msg.linear.x, -self.max_lin_vel, self.max_lin_vel))
        self.velocity_command[1] = float(np.clip(msg.linear.y, -self.max_lin_vel, self.max_lin_vel))
        self.velocity_command[2] = float(np.clip(msg.angular.z, -self.max_ang_vel, self.max_ang_vel))

    def _enable_cb(self, msg: Bool) -> None:
        if msg.data != self.enabled:
            self.get_logger().info(f"Policy {'enabled' if msg.data else 'disabled'}")
        self.enabled = bool(msg.data)

    def _odom_cb(self, msg: Odometry) -> None:
        self.last_lin_vel_time = self._now()
        # Odometry's twist is in the base frame by convention.
        self.base_lin_vel[:] = (
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z,
        )

    def _twist_cb(self, msg: TwistStamped) -> None:
        self.last_lin_vel_time = self._now()
        self.base_lin_vel[:] = (
            msg.twist.linear.x,
            msg.twist.linear.y,
            msg.twist.linear.z,
        )

    # ------------------------------------------------------------------ obs / inference
    def _build_observation(self) -> np.ndarray:
        joint_pos_rel = self.joint_positions - self.joint_default_pos

        if self.lin_vel_source == "zeros":
            lin_vel = np.zeros(3, dtype=np.float32)
        elif self.last_lin_vel_time is None or (
            self._now() - self.last_lin_vel_time > self.data_timeout
        ):
            lin_vel = np.zeros(3, dtype=np.float32)
        else:
            lin_vel = self.base_lin_vel

        obs = np.concatenate(
            [
                lin_vel,                  # 3
                self.base_ang_vel,        # 3
                self.projected_gravity,   # 3
                joint_pos_rel,            # 8
                self.joint_velocities,    # 8 (joint_vel_rel == joint_vel since default vel is 0)
                self.previous_actions,    # 8
                self.velocity_command,    # 3
            ]
        ).astype(np.float32)

        if self.obs_normalization_enabled:
            obs = (obs - self.obs_mean) / self.obs_std
        return obs

    def _run_inference(self, obs: np.ndarray) -> np.ndarray:
        outputs = self.session.run(
            [self.output_name], {self.input_name: obs.reshape(1, -1)}
        )
        return outputs[0].flatten().astype(np.float32)

    # ------------------------------------------------------------------ control loop
    def _control_loop(self) -> None:
        if not self.enabled:
            return

        if self.last_joint_state_time is None:
            self.get_logger().warn_throttle(5.0, "Waiting for /joint_states...")
            return
        if self.last_imu_time is None:
            self.get_logger().warn_throttle(5.0, "Waiting for /imu/data...")
            return

        now = self._now()
        if now - self.last_joint_state_time > self.data_timeout:
            self.get_logger().warn_throttle(
                1.0,
                f"Stale joint states ({now - self.last_joint_state_time:.2f}s old)",
            )
            return
        if now - self.last_imu_time > self.data_timeout:
            self.get_logger().warn_throttle(
                1.0, f"Stale IMU data ({now - self.last_imu_time:.2f}s old)"
            )
            return

        obs = self._build_observation()
        actions = self._run_inference(obs)
        self.previous_actions = actions

        # JointPositionActionCfg with use_default_offset=True does:
        #     target = default_pos + scale * action
        target_pos = self.joint_default_pos + self.action_scale * actions
        target_pos = np.clip(target_pos, self.joint_min_pos, self.joint_max_pos)

        cmd = JointState()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.name = JOINT_NAMES
        cmd.position = target_pos.tolist()
        cmd.velocity = [0.0] * NUM_JOINTS
        cmd.effort = [0.0] * NUM_JOINTS
        self.cmd_pub.publish(cmd)

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def destroy_node(self) -> bool:
        try:
            stop = JointState()
            stop.header.stamp = self.get_clock().now().to_msg()
            stop.name = JOINT_NAMES
            stop.position = self.joint_default_pos.tolist()
            stop.velocity = [0.0] * NUM_JOINTS
            stop.effort = [0.0] * NUM_JOINTS
            self.cmd_pub.publish(stop)
        except Exception:
            pass
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = PolicyRunner()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
