#!/usr/bin/env python3
"""PS5 DualSense teleop for the Bebop V2 humanoid.

Reads a Sony DualSense controller via evdev and publishes
``geometry_msgs/Twist`` on ``/cmd_vel`` plus ``std_msgs/Bool`` on
``/policy_enable``. The trained locomotion policy in
``bebop_control/policy_runner.py`` consumes both topics.

Default mapping:
    Left stick   -> linear velocity (Y up = +x forward, X right = -y strafe)
    Right stick  -> yaw rate (X)
    L1 (BTN_TL)  -> deadman switch (must be held to send commands)
    Circle       -> emergency stop (zero command, disables policy)
    Triangle     -> re-enable policy
    PS button    -> exit

Run::
    ros2 run bebop_pilot pilot_node.py \\
        --ros-args --params-file install/bebop_pilot/share/bebop_pilot/config/ps5_dualsense.yaml
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool

try:
    import evdev
    from evdev import ecodes
except ImportError as e:
    raise SystemExit(
        "python3-evdev is required. Install with: sudo apt install python3-evdev"
    ) from e


# DualSense raw axes are 8-bit (0..255) centered around 128.
# Triggers (Z, RZ) are also 0..255 but rest at 0.
STICK_CENTER = 128.0
STICK_HALF_RANGE = 127.0


class PilotNode(Node):
    """Reads a DualSense gamepad and publishes velocity commands."""

    def __init__(self) -> None:
        super().__init__("pilot_node")

        self.declare_parameter("device_path", "")
        self.declare_parameter("device_name_substring", "DualSense")
        self.declare_parameter("publish_rate", 50.0)
        self.declare_parameter("max_lin_vel_x", 0.6)
        self.declare_parameter("max_lin_vel_y", 0.4)
        self.declare_parameter("max_ang_vel_z", 1.0)
        self.declare_parameter("deadzone", 0.08)
        self.declare_parameter("expo", 0.4)
        self.declare_parameter("invert_left_stick_y", True)
        self.declare_parameter("invert_right_stick_x", False)
        self.declare_parameter("require_deadman", True)
        self.declare_parameter("watchdog_timeout", 0.25)

        self.device_path: str = self.get_parameter("device_path").value
        self.device_name_substring: str = self.get_parameter(
            "device_name_substring"
        ).value
        self.publish_rate: float = float(self.get_parameter("publish_rate").value)
        self.max_lin_vel_x: float = float(self.get_parameter("max_lin_vel_x").value)
        self.max_lin_vel_y: float = float(self.get_parameter("max_lin_vel_y").value)
        self.max_ang_vel_z: float = float(self.get_parameter("max_ang_vel_z").value)
        self.deadzone: float = float(self.get_parameter("deadzone").value)
        self.expo: float = float(self.get_parameter("expo").value)
        self.invert_ly: bool = bool(self.get_parameter("invert_left_stick_y").value)
        self.invert_rx: bool = bool(self.get_parameter("invert_right_stick_x").value)
        self.require_deadman: bool = bool(
            self.get_parameter("require_deadman").value
        )
        self.watchdog_timeout: float = float(
            self.get_parameter("watchdog_timeout").value
        )

        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.enable_pub = self.create_publisher(Bool, "/policy_enable", 10)

        # Stick state, normalized to [-1, 1].
        self._lock = threading.Lock()
        self._left_x = 0.0
        self._left_y = 0.0
        self._right_x = 0.0
        self._deadman_held = False
        self._policy_enabled = False
        self._last_event_time = time.monotonic()
        self._last_enable_state: Optional[bool] = None
        self._shutdown = False

        self.device = self._open_device()
        self.get_logger().info(
            f"Connected to gamepad: {self.device.name} ({self.device.path})"
        )

        self._reader_thread = threading.Thread(
            target=self._read_events, name="dualsense-reader", daemon=True
        )
        self._reader_thread.start()

        period = 1.0 / max(self.publish_rate, 1.0)
        self.timer = self.create_timer(period, self._tick)

        self.get_logger().info(
            "Pilot ready. Hold L1 (deadman) and use the left stick to move."
        )

    def _open_device(self) -> "evdev.InputDevice":
        if self.device_path:
            return evdev.InputDevice(self.device_path)

        candidates = []
        for path in evdev.list_devices():
            try:
                dev = evdev.InputDevice(path)
            except OSError:
                continue
            if self.device_name_substring.lower() in dev.name.lower():
                candidates.append(dev)

        if not candidates:
            raise SystemExit(
                f"No input device matching '{self.device_name_substring}' found. "
                "Is the controller paired and powered on? "
                "List devices with: python3 -m evdev.evtest"
            )

        if len(candidates) > 1:
            self.get_logger().warn(
                f"Multiple matching devices found, using {candidates[0].path}: "
                + ", ".join(f"{d.name}@{d.path}" for d in candidates)
            )
        return candidates[0]

    @staticmethod
    def _normalize_stick(raw: int) -> float:
        return max(-1.0, min(1.0, (raw - STICK_CENTER) / STICK_HALF_RANGE))

    def _shape(self, value: float) -> float:
        """Apply deadzone + exponential curve for finer center control."""
        if abs(value) < self.deadzone:
            return 0.0
        sign = 1.0 if value > 0 else -1.0
        scaled = (abs(value) - self.deadzone) / (1.0 - self.deadzone)
        shaped = (1.0 - self.expo) * scaled + self.expo * (scaled ** 3)
        return sign * shaped

    def _read_events(self) -> None:
        """Background thread that consumes evdev events as they arrive."""
        try:
            for event in self.device.read_loop():
                if self._shutdown:
                    return
                self._last_event_time = time.monotonic()

                if event.type == ecodes.EV_ABS:
                    self._handle_abs(event)
                elif event.type == ecodes.EV_KEY:
                    self._handle_key(event)
        except OSError as e:
            self.get_logger().error(f"Gamepad disconnected: {e}")
            with self._lock:
                self._deadman_held = False

    def _handle_abs(self, event) -> None:
        with self._lock:
            if event.code == ecodes.ABS_X:
                self._left_x = self._normalize_stick(event.value)
            elif event.code == ecodes.ABS_Y:
                v = self._normalize_stick(event.value)
                self._left_y = -v if self.invert_ly else v
            elif event.code == ecodes.ABS_RX:
                v = self._normalize_stick(event.value)
                self._right_x = -v if self.invert_rx else v

    def _handle_key(self, event) -> None:
        pressed = event.value == 1
        released = event.value == 0

        if event.code == ecodes.BTN_TL:  # L1
            with self._lock:
                self._deadman_held = pressed
            if released:
                self.get_logger().info("Deadman released; zeroing command.")

        elif event.code == ecodes.BTN_EAST and pressed:  # Circle
            self.get_logger().warn("EMERGENCY STOP")
            with self._lock:
                self._deadman_held = False
                self._policy_enabled = False

        elif event.code == ecodes.BTN_NORTH and pressed:  # Triangle
            with self._lock:
                self._policy_enabled = True
            self.get_logger().info("Policy re-enabled.")

        elif event.code == ecodes.BTN_MODE and pressed:  # PS button
            self.get_logger().info("PS button pressed; shutting down.")
            self._shutdown = True
            rclpy.shutdown()

    def _tick(self) -> None:
        """Publish a Twist + enable Bool at the configured rate."""
        with self._lock:
            lx, ly, rx = self._left_x, self._left_y, self._right_x
            deadman = self._deadman_held
            policy_enabled = self._policy_enabled

        watchdog_ok = (time.monotonic() - self._last_event_time) < self.watchdog_timeout
        active = watchdog_ok and (deadman or not self.require_deadman)

        twist = Twist()
        if active:
            twist.linear.x = self._shape(ly) * self.max_lin_vel_x
            twist.linear.y = -self._shape(lx) * self.max_lin_vel_y  # stick right -> -y
            twist.angular.z = -self._shape(rx) * self.max_ang_vel_z  # stick right -> CW
        self.cmd_pub.publish(twist)

        enable_now = active and policy_enabled
        if enable_now != self._last_enable_state:
            self.enable_pub.publish(Bool(data=enable_now))
            self._last_enable_state = enable_now

    def destroy_node(self) -> bool:
        self._shutdown = True
        try:
            zero = Twist()
            self.cmd_pub.publish(zero)
            self.enable_pub.publish(Bool(data=False))
        except Exception:
            pass
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = PilotNode()
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
