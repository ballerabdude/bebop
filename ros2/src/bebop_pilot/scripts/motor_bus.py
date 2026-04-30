#!/usr/bin/env python3
"""Robstride CAN motor bus for the Bebop V2 humanoid (8 DoF).

Bridges ROS 2 ``/joint_commands`` <-> SocketCAN <-> 8x Robstride motors.

    Subscribes:
        /joint_commands (sensor_msgs/JointState)
            Position targets (one per joint, name-keyed). Velocity / effort
            fields are accepted as feedforward but default to zero.
        /motor_enable    (std_msgs/Bool)
            Latch: True enables all motors, False disables. On startup the
            bus respects ``enable_on_start``.

    Publishes:
        /joint_states    (sensor_msgs/JointState)
            Live position / velocity / effort feedback parsed from the
            motors' active-report frames.

    On shutdown, sends ``Disable`` to every motor.

Hardware setup (Jetson Orin Nano Super dev kit, native CAN on the 40-pin
header):

    1. Use ``sudo /opt/nvidia/jetson-io/jetson-io.py`` to enable
       ``CAN0_DIN`` and ``CAN0_DOUT`` on pins 29 / 32, then reboot.
    2. Bring up the bus:
         sudo modprobe can
         sudo modprobe can_raw
         sudo modprobe mttcan
         sudo ip link set can0 up type can bitrate 1000000
       Verify: ``ip -details link show can0``.
    3. Install python-can:
         sudo apt install python3-can

USB-CAN dongle alternative (e.g. CANable / candleLight): the bus name is
typically still ``can0`` once you ``ip link set <iface> up``.
"""

from __future__ import annotations

import threading
import time
from typing import Dict, List, Optional

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool

try:
    import can  # python-can
except ImportError as exc:
    raise SystemExit(
        "python-can is required. On Jetson: sudo apt install python3-can"
    ) from exc

from bebop_pilot import robstride


# Joint order MUST match bebop_training/envs/bebop_v2_base_cfg.py:JOINT_NAMES_ALL
DEFAULT_JOINT_NAMES = [
    "hip_abduction_left_joint",
    "hip_abduction_right_joint",
    "femur_left_joint",
    "femur_right_joint",
    "shin_left_joint",
    "shin_right_joint",
    "foot_left_joint",
    "foot_right_joint",
]
NUM_JOINTS = len(DEFAULT_JOINT_NAMES)


class MotorState:
    """Per-motor command + feedback cache.

    ``motor_id`` is the 8-bit Robstride node ID configured per motor in the
    Robstride debugger (e.g. 31 for hip_abduction_left). It is *not* the
    full 29-bit CAN frame ID; the latter is computed per frame.
    """

    __slots__ = (
        "name",
        "motor_id",
        "model",
        "kp",
        "kd",
        "target_pos",
        "target_vel",
        "target_tau_ff",
        "fb_pos",
        "fb_vel",
        "fb_tau",
        "fb_temp",
        "fb_enabled",
        "fb_fault_bits",
        "fb_last_update",
    )

    def __init__(self, name: str, motor_id: int, model: str, kp: float, kd: float):
        self.name = name
        self.motor_id = motor_id
        self.model = model
        self.kp = kp
        self.kd = kd
        self.target_pos = 0.0
        self.target_vel = 0.0
        self.target_tau_ff = 0.0
        self.fb_pos = 0.0
        self.fb_vel = 0.0
        self.fb_tau = 0.0
        self.fb_temp = 0.0
        self.fb_enabled = False
        self.fb_fault_bits = 0
        self.fb_last_update: Optional[float] = None


class MotorBusNode(Node):
    """ROS 2 node owning the SocketCAN bus and the 8 Robstride motors."""

    def __init__(self) -> None:
        super().__init__("motor_bus")

        self.declare_parameter("can_interface", "can0")
        self.declare_parameter("joint_names", DEFAULT_JOINT_NAMES)
        self.declare_parameter("publish_rate", 100.0)
        self.declare_parameter("command_rate", 100.0)
        self.declare_parameter("feedback_interval_ms", 5)
        self.declare_parameter("watchdog_timeout", 0.25)
        self.declare_parameter("enable_on_start", False)
        self.declare_parameter("set_zero_on_start", False)

        self.can_interface: str = self.get_parameter("can_interface").value
        self.joint_names: List[str] = list(self.get_parameter("joint_names").value)
        self.publish_rate = float(self.get_parameter("publish_rate").value)
        self.command_rate = float(self.get_parameter("command_rate").value)
        self.feedback_interval_ms = int(self.get_parameter("feedback_interval_ms").value)
        self.watchdog_timeout = float(self.get_parameter("watchdog_timeout").value)
        self.enable_on_start = bool(self.get_parameter("enable_on_start").value)
        self.set_zero_on_start = bool(self.get_parameter("set_zero_on_start").value)

        # Per-joint config is declared as nested params:  joints.<name>.<field>
        self._motors_by_name: Dict[str, MotorState] = {}
        self._motors_by_id: Dict[int, MotorState] = {}
        for name in self.joint_names:
            ms = self._declare_motor_from_params(name)
            if ms.motor_id in self._motors_by_id:
                other = self._motors_by_id[ms.motor_id].name
                raise ValueError(
                    f"Duplicate motor_id={ms.motor_id} for joints "
                    f"{other!r} and {name!r}."
                )
            self._motors_by_name[name] = ms
            self._motors_by_id[ms.motor_id] = ms

        self._lock = threading.Lock()
        self._enabled = False
        self._last_command_time: Optional[float] = None
        self._shutdown = False

        self.get_logger().info(f"Opening CAN bus '{self.can_interface}' (1 Mbps)...")
        try:
            self._bus = can.interface.Bus(
                channel=self.can_interface, bustype="socketcan"
            )
        except Exception as exc:
            raise SystemExit(
                f"Failed to open CAN bus '{self.can_interface}': {exc}. "
                "Bring it up with: sudo ip link set can0 up type can bitrate 1000000"
            ) from exc

        self.cmd_sub = self.create_subscription(
            JointState, "/joint_commands", self._on_joint_commands, 10
        )
        self.enable_sub = self.create_subscription(
            Bool, "/motor_enable", self._on_enable, 10
        )
        self.state_pub = self.create_publisher(JointState, "/joint_states", 10)

        self._reader_thread = threading.Thread(
            target=self._read_loop, name="robstride-rx", daemon=True
        )
        self._reader_thread.start()

        if self.set_zero_on_start:
            self._send_set_zero_all()

        self._send_active_report_all(self.feedback_interval_ms)

        if self.enable_on_start:
            self._set_enabled(True)
        else:
            self.get_logger().warn(
                "Motors NOT enabled at startup. Publish True on /motor_enable when ready."
            )

        self.cmd_timer = self.create_timer(
            1.0 / max(self.command_rate, 1.0), self._send_commands_tick
        )
        self.pub_timer = self.create_timer(
            1.0 / max(self.publish_rate, 1.0), self._publish_joint_states
        )

        self.get_logger().info(
            f"Motor bus ready: {len(self._motors_by_name)} motors, "
            f"command @ {self.command_rate} Hz, feedback @ {self.publish_rate} Hz."
        )

    # ------------------------------------------------------------------ param parsing
    def _declare_motor_from_params(self, name: str) -> "MotorState":
        """Read ``joints.<name>.{motor_id, model, kp, kd}`` from the param tree."""
        prefix = f"joints.{name}"

        def _get(field: str, default):
            full = f"{prefix}.{field}"
            self.declare_parameter(full, default)
            return self.get_parameter(full).value

        motor_id = int(_get("motor_id", 0))
        if motor_id <= 0 or motor_id > 0xFF:
            raise ValueError(
                f"Joint {name!r}: motor_id={motor_id} out of range [1, 255]. "
                "Set joints.<joint>.motor_id in motor_bus.yaml."
            )
        model = str(_get("model", "")).upper()
        if model not in robstride.SPECS:
            raise ValueError(
                f"Joint {name!r}: unknown model {model!r}. "
                f"Expected one of {list(robstride.SPECS)}."
            )
        kp = float(_get("kp", 0.0))
        kd = float(_get("kd", 0.0))

        return MotorState(name=name, motor_id=motor_id, model=model, kp=kp, kd=kd)

    # ------------------------------------------------------------------ ROS callbacks
    def _on_joint_commands(self, msg: JointState) -> None:
        with self._lock:
            for i, name in enumerate(msg.name):
                motor = self._motors_by_name.get(name)
                if motor is None:
                    continue
                if i < len(msg.position):
                    motor.target_pos = float(msg.position[i])
                if i < len(msg.velocity):
                    motor.target_vel = float(msg.velocity[i])
                if i < len(msg.effort):
                    motor.target_tau_ff = float(msg.effort[i])
            self._last_command_time = time.monotonic()

    def _on_enable(self, msg: Bool) -> None:
        self._set_enabled(bool(msg.data))

    # ------------------------------------------------------------------ TX path
    def _send(self, frame_id: int, data: bytes) -> bool:
        try:
            self._bus.send(
                can.Message(
                    arbitration_id=frame_id,
                    is_extended_id=True,
                    data=data,
                ),
                timeout=0.01,
            )
            return True
        except can.CanError as exc:
            self.get_logger().warn_throttle(1.0, f"CAN TX error: {exc}")
            return False

    def _set_enabled(self, enabled: bool) -> None:
        if enabled == self._enabled:
            return
        self._enabled = enabled
        if enabled:
            self.get_logger().info("Enabling all Robstride motors.")
            for m in self._motors_by_name.values():
                frame_id, data = robstride.build_enable(m.motor_id)
                self._send(frame_id, data)
                time.sleep(0.01)
            with self._lock:
                self._last_command_time = time.monotonic()
        else:
            self.get_logger().warn("Disabling all Robstride motors.")
            for m in self._motors_by_name.values():
                frame_id, data = robstride.build_disable(m.motor_id)
                self._send(frame_id, data)
                time.sleep(0.01)

    def _send_active_report_all(self, interval_ms: int) -> None:
        self.get_logger().info(
            f"Requesting active feedback from all motors @ {interval_ms} ms."
        )
        for m in self._motors_by_name.values():
            frame_id, data = robstride.build_active_report(m.motor_id, interval_ms)
            self._send(frame_id, data)
            time.sleep(0.01)

    def _send_set_zero_all(self) -> None:
        self.get_logger().warn(
            "set_zero_on_start=True; setting current pose as mechanical zero "
            "for ALL motors. This persists in motor flash."
        )
        for m in self._motors_by_name.values():
            frame_id, data = robstride.build_set_zero(m.motor_id)
            self._send(frame_id, data)
            time.sleep(0.05)

    def _send_commands_tick(self) -> None:
        if not self._enabled:
            return

        with self._lock:
            stale = (
                self._last_command_time is None
                or (time.monotonic() - self._last_command_time) > self.watchdog_timeout
            )
            snapshot = [
                (m.motor_id, m.model, m.target_pos, m.target_vel, m.kp, m.kd, m.target_tau_ff)
                for m in self._motors_by_name.values()
            ]

        if stale:
            self.get_logger().warn_throttle(
                1.0,
                "Command watchdog: /joint_commands stale, holding last target with kd damping.",
            )

        for motor_id, model, pos, vel, kp, kd, tau_ff in snapshot:
            frame_id, data = robstride.build_motor_ctrl(
                motor_id=motor_id,
                model=model,
                position=pos,
                velocity=vel,
                kp=0.0 if stale else kp,
                kd=kd,
                torque_ff=0.0 if stale else tau_ff,
            )
            self._send(frame_id, data)

    # ------------------------------------------------------------------ RX path
    def _read_loop(self) -> None:
        while not self._shutdown:
            try:
                msg = self._bus.recv(timeout=0.1)
            except can.CanError as exc:
                self.get_logger().warn_throttle(1.0, f"CAN RX error: {exc}")
                continue
            if msg is None or not msg.is_extended_id:
                continue

            cmd_type = (msg.arbitration_id >> 24) & 0x1F
            if cmd_type != robstride.CmdType.FEEDBACK:
                continue

            motor_id = (msg.arbitration_id >> 8) & 0xFF
            motor = self._motors_by_id.get(motor_id)
            if motor is None:
                continue

            fb = robstride.parse_feedback(
                msg.arbitration_id, bytes(msg.data), motor.model
            )
            if fb is None:
                continue

            now = time.monotonic()
            with self._lock:
                motor.fb_pos = fb.position
                motor.fb_vel = fb.velocity
                motor.fb_tau = fb.torque
                motor.fb_temp = fb.temperature
                motor.fb_enabled = fb.is_enabled
                motor.fb_fault_bits = fb.fault_bits
                motor.fb_last_update = now

            if fb.has_fault:
                self.get_logger().warn_throttle(
                    1.0,
                    f"Motor {motor.name} (id={motor.motor_id}) fault: "
                    f"{fb.fault_description()} @ {fb.temperature:.1f}°C",
                )

    def _publish_joint_states(self) -> None:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(self.joint_names)
        positions, velocities, efforts = [], [], []
        with self._lock:
            for name in self.joint_names:
                m = self._motors_by_name[name]
                positions.append(m.fb_pos)
                velocities.append(m.fb_vel)
                efforts.append(m.fb_tau)
        msg.position = positions
        msg.velocity = velocities
        msg.effort = efforts
        self.state_pub.publish(msg)

    # ------------------------------------------------------------------ shutdown
    def destroy_node(self) -> bool:
        self._shutdown = True
        try:
            for m in self._motors_by_name.values():
                frame_id, data = robstride.build_disable(m.motor_id)
                self._send(frame_id, data)
        except Exception:
            pass
        try:
            self._bus.shutdown()
        except Exception:
            pass
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = MotorBusNode()
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
