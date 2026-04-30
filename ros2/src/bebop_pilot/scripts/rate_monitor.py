#!/usr/bin/env python3
"""
Rate Monitor - Measure actual publish rates and latency for debugging.

This helps identify where in the communication stack delays occur:
  Isaac Sim Physics → Isaac Sim Publish → ROS2 Transport → Teensy Receive → Teensy Publish → ROS2 Transport → Isaac Sim Receive
"""

import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState, Imu
from collections import deque
import statistics


class RateMonitor(Node):
    def __init__(self):
        super().__init__('rate_monitor')
        
        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
        
        # Track timing for each topic
        self.topics = {
            '/joint_states': {'times': deque(maxlen=100), 'last': None},
            '/joint_commands': {'times': deque(maxlen=100), 'last': None},
            '/imu/data': {'times': deque(maxlen=100), 'last': None},
        }
        
        # Latency tracking (if messages have timestamps)
        self.latencies = {
            '/joint_states': deque(maxlen=100),
            '/imu/data': deque(maxlen=100),
        }
        
        # Subscribers
        self.create_subscription(JointState, '/joint_states', 
            lambda msg: self.callback('/joint_states', msg), qos)
        self.create_subscription(JointState, '/joint_commands',
            lambda msg: self.callback('/joint_commands', msg), qos)
        self.create_subscription(Imu, '/imu/data',
            lambda msg: self.callback('/imu/data', msg), qos)
        
        # Print stats every second
        self.create_timer(1.0, self.print_stats)
        
        self.get_logger().info("Rate Monitor started - watching communication stack")
    
    def callback(self, topic: str, msg):
        now = time.time()
        data = self.topics[topic]
        
        if data['last'] is not None:
            dt = now - data['last']
            data['times'].append(dt)
        
        data['last'] = now
        
        # Check message timestamp for latency (if available)
        if hasattr(msg, 'header') and msg.header.stamp.sec > 0:
            msg_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            # Get ROS time
            ros_now = self.get_clock().now().nanoseconds * 1e-9
            latency_ms = (ros_now - msg_time) * 1000
            if topic in self.latencies and abs(latency_ms) < 1000:  # Sanity check
                self.latencies[topic].append(latency_ms)
    
    def print_stats(self):
        print("\n" + "="*70)
        print(f"{'Topic':<20} {'Rate (Hz)':<12} {'Jitter (ms)':<12} {'Latency (ms)':<15}")
        print("="*70)
        
        for topic, data in self.topics.items():
            if len(data['times']) >= 2:
                dts = list(data['times'])
                avg_dt = statistics.mean(dts)
                rate = 1.0 / avg_dt if avg_dt > 0 else 0
                
                # Jitter = standard deviation of dt
                jitter_ms = statistics.stdev(dts) * 1000 if len(dts) > 2 else 0
                
                # Latency
                latency_str = "N/A"
                if topic in self.latencies and len(self.latencies[topic]) > 0:
                    avg_latency = statistics.mean(self.latencies[topic])
                    latency_str = f"{avg_latency:.2f}"
                
                print(f"{topic:<20} {rate:>8.1f} Hz  {jitter_ms:>8.2f} ms   {latency_str:>10}")
            else:
                print(f"{topic:<20} {'waiting...':<12}")
        
        print("="*70)
        
        # Analysis
        if len(self.topics['/joint_states']['times']) > 10:
            js_rate = 1.0 / statistics.mean(self.topics['/joint_states']['times'])
            jc_times = self.topics['/joint_commands']['times']
            jc_rate = 1.0 / statistics.mean(jc_times) if len(jc_times) > 1 else 0
            
            if js_rate < 90:
                print("⚠️  /joint_states below 90Hz - Isaac Sim may be slow")
            if jc_rate < 45:
                print("⚠️  /joint_commands below 45Hz - Teensy may be slow")
            
            # Check round-trip
            if js_rate > 0 and jc_rate > 0:
                print(f"\n📊 Round-trip estimate: Isaac→Teensy→Isaac")
                print(f"   Isaac publishes at {js_rate:.1f} Hz")
                print(f"   Teensy responds at {jc_rate:.1f} Hz")


def main():
    rclpy.init()
    node = RateMonitor()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

