/**
 * @file RosPublishers.h
 * @brief ROS 2 message publishing and subscription management
 * 
 * Handles all micro-ROS communication for the CAN bridge:
 *   - Joint state feedback (100 Hz)
 *   - IMU data (100 Hz)
 *   - Motor diagnostics (10 Hz)
 *   - Joint command subscription
 */

#ifndef ROS_PUBLISHERS_H
#define ROS_PUBLISHERS_H

#include <micro_ros_arduino.h>
#include <rcl/rcl.h>
#include <rcl/error_handling.h>
#include <rclc/rclc.h>
#include <rclc/executor.h>
#include <sensor_msgs/msg/joint_state.h>
#include <sensor_msgs/msg/imu.h>
#include <geometry_msgs/msg/twist.h>
#include <std_msgs/msg/float32_multi_array.h>
#include <std_msgs/msg/int32_multi_array.h>

#include "RobotConfig.h"
#include "BNO085_IMU.h"

// ============================================================================
// DIAGNOSTIC DATA LAYOUT
// ============================================================================
// 
// /motor_temps (Float32MultiArray) - NUM_JOINTS * 2 floats:
//   [motor0_temp, motor0_board_temp, motor1_temp, motor1_board_temp, ...]
//   For ODrive: [fet_temp, motor_temp]
//   For Robstride: [motor_temp, 0.0]
//
// /motor_status (Int32MultiArray) - NUM_JOINTS * 4 ints:
//   [motor0_error, motor0_state, motor0_enabled, motor0_extra,
//    motor1_error, motor1_state, motor1_enabled, motor1_extra, ...]
//   extra: bus_voltage*100 for ODrive, fault_bits for Robstride
//
// ============================================================================

#define TEMP_ARRAY_SIZE (NUM_JOINTS * 2)
#define STATUS_ARRAY_SIZE (NUM_JOINTS * 4)

// Special effort values for motor control commands (from dashboard)
#define MOTOR_CMD_CLEAR_ERRORS   -999.0f
#define MOTOR_CMD_ENABLE_ALL     -998.0f
#define MOTOR_CMD_DISABLE_ALL    -997.0f
#define MOTOR_CMD_RESET_ENABLE   -996.0f
#define MOTOR_CMD_MODE_POLICY    -995.0f
#define MOTOR_CMD_MODE_MANUAL    -994.0f

// Error handling macros
#define RCCHECK(fn) { rcl_ret_t temp_rc = fn; if((temp_rc != RCL_RET_OK)){errorLoop();}}
#define RCSOFTCHECK(fn) { rcl_ret_t temp_rc = fn; if((temp_rc != RCL_RET_OK)){}}

/**
 * @brief ROS publishers and subscribers manager
 */
class RosPublishers {
public:
    enum ControlMode {
        MODE_PASSTHROUGH,
        MODE_POLICY_BALANCE
    };

    /**
     * @brief Initialize micro-ROS with agent connection
     * @param imu Pointer to IMU instance for publishing IMU data
     * @return true if connected to agent successfully
     */
    bool begin(BNO085_IMU* imu);
    
    /**
     * @brief Spin the executor to process incoming messages
     * @param timeout_us Timeout in microseconds
     */
    void spin(uint32_t timeout_us = 100);
    
    /**
     * @brief Publish joint state feedback
     */
    void publishState();
    
    /**
     * @brief Publish IMU data
     */
    void publishIMU();
    
    /**
     * @brief Publish motor diagnostics (temps and status)
     */
    void publishDiagnostics();
    
    /**
     * @brief Get current control mode
     */
    ControlMode getControlMode() const { return control_mode_; }

    /**
     * @brief Set control mode
     */
    void setControlMode(ControlMode mode) { control_mode_ = mode; }

    /**
     * @brief Get timestamp of last command received
     * @return Timestamp in milliseconds
     */
    uint32_t getLastCmdTime() const { return last_cmd_time_; }

    /**
     * @brief Get latest velocity command
     * @param vx Linear X (m/s)
     * @param vy Linear Y (m/s)
     * @param omega Angular Z (rad/s)
     */
    void getCmdVel(float& vx, float& vy, float& omega) const {
        vx = cmd_vel_x_;
        vy = cmd_vel_y_;
        omega = cmd_vel_omega_;
    }

private:
    // micro-ROS entities
    rcl_node_t node_;
    rcl_subscription_t cmd_subscriber_;
    rcl_subscription_t cmd_vel_subscriber_;
    rcl_publisher_t state_publisher_;
    rcl_publisher_t imu_publisher_;
    rcl_publisher_t temp_publisher_;
    rcl_publisher_t status_publisher_;
    rclc_executor_t executor_;
    rclc_support_t support_;
    rcl_allocator_t allocator_;
    
    // Messages
    sensor_msgs__msg__JointState cmd_msg_;
    geometry_msgs__msg__Twist cmd_vel_msg_;
    sensor_msgs__msg__JointState state_msg_;
    sensor_msgs__msg__Imu imu_msg_;
    std_msgs__msg__Float32MultiArray temp_msg_;
    std_msgs__msg__Int32MultiArray status_msg_;
    
    // IMU reference
    BNO085_IMU* imu_;
    
    // Control Mode
    ControlMode control_mode_ = MODE_PASSTHROUGH; // Default to manual/passthrough

    // Timing
    uint32_t last_cmd_time_ = 0;
    
    // Command Buffer
    float cmd_vel_x_ = 0.0f;
    float cmd_vel_y_ = 0.0f;
    float cmd_vel_omega_ = 0.0f;

    /**
     * @brief Allocate memory for ROS messages
     */
    void allocateMessages();
    
    /**
     * @brief Enter error loop (LED blinks)
     */
    static void errorLoop();
    
    /**
     * @brief Handle special motor control commands
     * @param cmd_value Command value (negative special codes)
     */
    void handleMotorControlCommand(float cmd_value);
    
    /**
     * @brief Command callback (static for micro-ROS)
     */
    static void cmdCallback(const void* msgin);

    /**
     * @brief Velocity command callback
     */
    static void cmdVelCallback(const void* msgin);
    
    // Singleton for callback access
    static RosPublishers* instance_;
};

// Global instance
extern RosPublishers rosPublishers;

#endif // ROS_PUBLISHERS_H
