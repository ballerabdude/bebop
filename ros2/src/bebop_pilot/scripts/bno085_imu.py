#!/usr/bin/env python3
"""
BNO085 IMU Node - Publishes IMU data for policy runner.

The BNO085 provides:
- Rotation Vector (quaternion) - fused orientation
- Gyroscope - angular velocity
- Accelerometer - linear acceleration

Publishes:
    /imu/data (sensor_msgs/Imu) - Orientation, angular velocity, linear acceleration

Hardware:
    Connect BNO085 via I2C:
    - VIN -> 3.3V or 5V
    - GND -> GND
    - SDA -> I2C SDA (GPIO 2 on Raspberry Pi)
    - SCL -> I2C SCL (GPIO 3 on Raspberry Pi)

Usage:
    ros2 run bebop_pilot bno085_imu.py

Requirements:
    pip install adafruit-circuitpython-bno08x
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Vector3
from std_msgs.msg import Header
import time
import math

# BNO085 library
try:
    import board
    import busio
    from adafruit_bno08x.i2c import BNO08X_I2C
    from adafruit_bno08x import (
        BNO_REPORT_ROTATION_VECTOR,
        BNO_REPORT_GYROSCOPE,
        BNO_REPORT_ACCELEROMETER,
    )
    BNO085_AVAILABLE = True
except ImportError as e:
    BNO085_AVAILABLE = False
    IMPORT_ERROR = str(e)


class BNO085IMU(Node):
    """
    ROS2 node for BNO085 IMU sensor.
    """
    
    def __init__(self):
        super().__init__('bno085_imu')
        
        # Parameters
        self.declare_parameter('publish_rate', 100.0)  # Hz
        self.declare_parameter('frame_id', 'imu_link')
        self.declare_parameter('i2c_address', 0x4A)  # Default BNO085 address
        
        self.publish_rate = self.get_parameter('publish_rate').value
        self.frame_id = self.get_parameter('frame_id').value
        self.i2c_address = self.get_parameter('i2c_address').value
        
        # Publisher
        self.imu_pub = self.create_publisher(Imu, '/imu/data', 10)
        
        # Initialize sensor
        if not BNO085_AVAILABLE:
            self.get_logger().error(f'BNO085 library not available: {IMPORT_ERROR}')
            self.get_logger().error('Install with: pip install adafruit-circuitpython-bno08x')
            self.bno = None
        else:
            self.init_sensor()
        
        # Timer for publishing
        period = 1.0 / self.publish_rate
        self.timer = self.create_timer(period, self.publish_imu)
        
        self.get_logger().info(f'BNO085 IMU node started @ {self.publish_rate} Hz')
    
    def init_sensor(self):
        """Initialize BNO085 sensor via I2C."""
        try:
            self.get_logger().info('Initializing BNO085 IMU...')
            
            # Initialize I2C
            i2c = busio.I2C(board.SCL, board.SDA, frequency=400000)
            
            # Initialize BNO085
            self.bno = BNO08X_I2C(i2c, address=self.i2c_address)
            
            # Enable required reports
            self.bno.enable_feature(BNO_REPORT_ROTATION_VECTOR)
            self.bno.enable_feature(BNO_REPORT_GYROSCOPE)
            self.bno.enable_feature(BNO_REPORT_ACCELEROMETER)
            
            # Wait for sensor to stabilize
            time.sleep(0.5)
            
            self.get_logger().info('BNO085 initialized successfully!')
            
        except Exception as e:
            self.get_logger().error(f'Failed to initialize BNO085: {e}')
            self.bno = None
    
    def publish_imu(self):
        """Read sensor and publish IMU message."""
        
        if self.bno is None:
            # Publish dummy data for testing without hardware
            self.publish_dummy_imu()
            return
        
        try:
            # Read quaternion (rotation vector)
            quat = self.bno.quaternion
            if quat is None:
                return
            
            # Read gyroscope (angular velocity in rad/s)
            gyro = self.bno.gyro
            if gyro is None:
                gyro = (0.0, 0.0, 0.0)
            
            # Read accelerometer (linear acceleration in m/s^2)
            accel = self.bno.acceleration
            if accel is None:
                accel = (0.0, 0.0, -9.81)
            
            # Build IMU message
            msg = Imu()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.frame_id
            
            # Orientation (quaternion: x, y, z, w)
            # BNO085 returns [i, j, k, real] which is [x, y, z, w]
            msg.orientation.x = quat[0]
            msg.orientation.y = quat[1]
            msg.orientation.z = quat[2]
            msg.orientation.w = quat[3]
            
            # Angular velocity (rad/s)
            msg.angular_velocity.x = gyro[0]
            msg.angular_velocity.y = gyro[1]
            msg.angular_velocity.z = gyro[2]
            
            # Linear acceleration (m/s^2)
            msg.linear_acceleration.x = accel[0]
            msg.linear_acceleration.y = accel[1]
            msg.linear_acceleration.z = accel[2]
            
            # Covariance (unknown, set to -1)
            msg.orientation_covariance[0] = -1.0
            msg.angular_velocity_covariance[0] = -1.0
            msg.linear_acceleration_covariance[0] = -1.0
            
            self.imu_pub.publish(msg)
            
        except Exception as e:
            self.get_logger().warn_throttle(1.0, f'Error reading BNO085: {e}')
    
    def publish_dummy_imu(self):
        """Publish dummy IMU data for testing without hardware."""
        msg = Imu()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        
        # Identity quaternion (no rotation)
        msg.orientation.x = 0.0
        msg.orientation.y = 0.0
        msg.orientation.z = 0.0
        msg.orientation.w = 1.0
        
        # Zero angular velocity
        msg.angular_velocity.x = 0.0
        msg.angular_velocity.y = 0.0
        msg.angular_velocity.z = 0.0
        
        # Gravity pointing down
        msg.linear_acceleration.x = 0.0
        msg.linear_acceleration.y = 0.0
        msg.linear_acceleration.z = -9.81
        
        self.imu_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = BNO085IMU()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()

