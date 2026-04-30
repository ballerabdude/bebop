/**
 * @file RosPublishers.cpp
 * @brief ROS 2 message publishing implementation
 */

#include "RosPublishers.h"
#include "SerialCommands.h"

// Global instance
RosPublishers rosPublishers;

// Singleton for callback access
RosPublishers* RosPublishers::instance_ = nullptr;

void RosPublishers::errorLoop() {
    while (1) {
        digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN));
        delay(100);
    }
}

bool RosPublishers::begin(BNO085_IMU* imu) {
    instance_ = this;
    imu_ = imu;
    
    DEBUG_PORT.println("Connecting to micro-ROS agent...");
    set_microros_transports();
    
    allocator_ = rcl_get_default_allocator();
    
    // Retry connecting to agent
    const int MAX_RETRIES = 100;
    int retry_count = 0;
    rcl_ret_t ret;
    
    while (retry_count < MAX_RETRIES) {
        ret = rclc_support_init(&support_, 0, NULL, &allocator_);
        if (ret == RCL_RET_OK) {
            DEBUG_PORT.println("Connected to micro-ROS agent!");
            break;
        }
        
        retry_count++;
        if (retry_count % 10 == 1) {
            DEBUG_PORT.printf("  Waiting for agent... (attempt %d/%d)\n", retry_count, MAX_RETRIES);
        }
        
        digitalWrite(LED_BUILTIN, retry_count % 2);
        delay(500);
    }
    
    if (ret != RCL_RET_OK) {
        DEBUG_PORT.println("ERROR: Failed to connect to micro-ROS agent!");
        return false;
    }
    
    // Create node
    RCCHECK(rclc_node_init_default(&node_, "teensy_can_bridge", "", &support_));
    
    // Allocate message memory
    allocateMessages();
    
    // Create subscriber for joint commands
    RCCHECK(rclc_subscription_init_default(
        &cmd_subscriber_,
        &node_,
        ROSIDL_GET_MSG_TYPE_SUPPORT(sensor_msgs, msg, JointState),
        "/joint_commands"));

    // Create subscriber for velocity commands
    RCCHECK(rclc_subscription_init_default(
        &cmd_vel_subscriber_,
        &node_,
        ROSIDL_GET_MSG_TYPE_SUPPORT(geometry_msgs, msg, Twist),
        "/cmd_vel"));
    
    // Create publisher for joint state feedback
    RCCHECK(rclc_publisher_init_default(
        &state_publisher_,
        &node_,
        ROSIDL_GET_MSG_TYPE_SUPPORT(sensor_msgs, msg, JointState),
        "/joint_states"));
    
    // Create publisher for IMU data
    RCCHECK(rclc_publisher_init_default(
        &imu_publisher_,
        &node_,
        ROSIDL_GET_MSG_TYPE_SUPPORT(sensor_msgs, msg, Imu),
        "/imu/data"));
    
    // Create publisher for temperature data
    RCCHECK(rclc_publisher_init_default(
        &temp_publisher_,
        &node_,
        ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Float32MultiArray),
        "/motor_temps"));
    
    // Create publisher for status/error data
    RCCHECK(rclc_publisher_init_default(
        &status_publisher_,
        &node_,
        ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Int32MultiArray),
        "/motor_status"));
    
    // Create executor
    RCCHECK(rclc_executor_init(&executor_, &support_.context, 2, &allocator_));
    RCCHECK(rclc_executor_add_subscription(&executor_, &cmd_subscriber_, &cmd_msg_, 
                                            &RosPublishers::cmdCallback, ON_NEW_DATA));
    RCCHECK(rclc_executor_add_subscription(&executor_, &cmd_vel_subscriber_, &cmd_vel_msg_, 
                                            &RosPublishers::cmdVelCallback, ON_NEW_DATA));
    
    last_cmd_time_ = millis();
    return true;
}

void RosPublishers::spin(uint32_t timeout_us) {
    RCSOFTCHECK(rclc_executor_spin_some(&executor_, RCL_US_TO_NS(timeout_us)));
}

void RosPublishers::allocateMessages() {
    // ---- Command message (incoming) ----
    // Header frame_id
    cmd_msg_.header.frame_id.capacity = 50;
    cmd_msg_.header.frame_id.data = (char*)malloc(cmd_msg_.header.frame_id.capacity * sizeof(char));
    cmd_msg_.header.frame_id.size = 0;

    // Names
    cmd_msg_.name.capacity = NUM_JOINTS;
    cmd_msg_.name.data = (rosidl_runtime_c__String*)malloc(cmd_msg_.name.capacity * sizeof(rosidl_runtime_c__String));
    cmd_msg_.name.size = 0;
    
    for (size_t i = 0; i < cmd_msg_.name.capacity; i++) {
        cmd_msg_.name.data[i].capacity = 30;
        cmd_msg_.name.data[i].data = (char*)malloc(cmd_msg_.name.data[i].capacity * sizeof(char));
        cmd_msg_.name.data[i].size = 0;
    }

    // Data arrays
    cmd_msg_.position.data = (double*)malloc(NUM_JOINTS * sizeof(double));
    cmd_msg_.position.size = 0;
    cmd_msg_.position.capacity = NUM_JOINTS;
    
    cmd_msg_.velocity.data = (double*)malloc(NUM_JOINTS * sizeof(double));
    cmd_msg_.velocity.size = 0;
    cmd_msg_.velocity.capacity = NUM_JOINTS;
    
    cmd_msg_.effort.data = (double*)malloc(NUM_JOINTS * sizeof(double));
    cmd_msg_.effort.size = 0;
    cmd_msg_.effort.capacity = NUM_JOINTS;

    // ---- State message (outgoing) ----
    state_msg_.name.data = (rosidl_runtime_c__String*)malloc(NUM_JOINTS * sizeof(rosidl_runtime_c__String));
    state_msg_.name.size = NUM_JOINTS;
    state_msg_.name.capacity = NUM_JOINTS;
    
    for (int i = 0; i < NUM_JOINTS; i++) {
        state_msg_.name.data[i].data = (char*)JOINT_NAMES[i];
        state_msg_.name.data[i].size = strlen(JOINT_NAMES[i]);
        state_msg_.name.data[i].capacity = strlen(JOINT_NAMES[i]) + 1;
    }
    
    state_msg_.position.data = (double*)malloc(NUM_JOINTS * sizeof(double));
    state_msg_.position.size = NUM_JOINTS;
    state_msg_.position.capacity = NUM_JOINTS;
    
    state_msg_.velocity.data = (double*)malloc(NUM_JOINTS * sizeof(double));
    state_msg_.velocity.size = NUM_JOINTS;
    state_msg_.velocity.capacity = NUM_JOINTS;
    
    state_msg_.effort.data = (double*)malloc(NUM_JOINTS * sizeof(double));
    state_msg_.effort.size = NUM_JOINTS;
    state_msg_.effort.capacity = NUM_JOINTS;

    // ---- IMU message ----
    imu_msg_.header.frame_id.data = (char*)malloc(20 * sizeof(char));
    imu_msg_.header.frame_id.capacity = 20;
    strcpy(imu_msg_.header.frame_id.data, "imu_link");
    imu_msg_.header.frame_id.size = strlen("imu_link");

    // ---- Temperature message ----
    temp_msg_.data.data = (float*)malloc(TEMP_ARRAY_SIZE * sizeof(float));
    temp_msg_.data.size = TEMP_ARRAY_SIZE;
    temp_msg_.data.capacity = TEMP_ARRAY_SIZE;
    
    // ---- Status message ----
    status_msg_.data.data = (int32_t*)malloc(STATUS_ARRAY_SIZE * sizeof(int32_t));
    status_msg_.data.size = STATUS_ARRAY_SIZE;
    status_msg_.data.capacity = STATUS_ARRAY_SIZE;

    // Initialize all arrays to zero
    for (int i = 0; i < NUM_JOINTS; i++) {
        cmd_msg_.position.data[i] = 0.0;
        cmd_msg_.velocity.data[i] = 0.0;
        cmd_msg_.effort.data[i] = 0.0;
        state_msg_.position.data[i] = 0.0;
        state_msg_.velocity.data[i] = 0.0;
        state_msg_.effort.data[i] = 0.0;
    }
    
    for (int i = 0; i < TEMP_ARRAY_SIZE; i++) {
        temp_msg_.data.data[i] = 0.0f;
    }
    
    for (int i = 0; i < STATUS_ARRAY_SIZE; i++) {
        status_msg_.data.data[i] = 0;
    }
}

void RosPublishers::publishState() {
    for (int i = 0; i < NUM_JOINTS; i++) {
        state_msg_.position.data[i] = joints[i]->current_position;
        state_msg_.velocity.data[i] = joints[i]->current_velocity;
        state_msg_.effort.data[i] = joints[i]->current_torque;
    }
    
    state_msg_.header.stamp.sec = millis() / 1000;
    state_msg_.header.stamp.nanosec = (millis() % 1000) * 1000000;
    
    RCSOFTCHECK(rcl_publish(&state_publisher_, &state_msg_, NULL));
}

void RosPublishers::publishIMU() {
    if (!imu_ || !imu_->initialized) return;

    imu_msg_.header.stamp.sec = millis() / 1000;
    imu_msg_.header.stamp.nanosec = (millis() % 1000) * 1000000;

    // Orientation (Quaternion)
    imu_msg_.orientation.w = imu_->quat_w;
    imu_msg_.orientation.x = imu_->quat_x;
    imu_msg_.orientation.y = imu_->quat_y;
    imu_msg_.orientation.z = imu_->quat_z;
    
    // Angular Velocity (Gyro)
    imu_msg_.angular_velocity.x = imu_->gyro_x;
    imu_msg_.angular_velocity.y = imu_->gyro_y;
    imu_msg_.angular_velocity.z = imu_->gyro_z;
    
    // Linear Acceleration
    imu_msg_.linear_acceleration.x = imu_->accel_x;
    imu_msg_.linear_acceleration.y = imu_->accel_y;
    imu_msg_.linear_acceleration.z = imu_->accel_z;

    RCSOFTCHECK(rcl_publish(&imu_publisher_, &imu_msg_, NULL));
}

void RosPublishers::publishDiagnostics() {
    // ---- Temperature Data ----
    for (int i = 0; i < NUM_JOINTS; i++) {
        int base = i * 2;
        
        if (i >= ROBSTRIDE_START_IDX && i <= ROBSTRIDE_END_IDX) {
            RobstrideMotorType* rs_motor = static_cast<RobstrideMotorType*>(joints[i]);
            temp_msg_.data.data[base + 0] = rs_motor->getTemperature();
            temp_msg_.data.data[base + 1] = 0.0f;
        }
        else if (i >= ODRIVE_START_IDX && i <= ODRIVE_END_IDX) {
            ODriveMotorType* od_motor = static_cast<ODriveMotorType*>(joints[i]);
            temp_msg_.data.data[base + 0] = od_motor->getFetTemperature();
            temp_msg_.data.data[base + 1] = od_motor->getMotorTemperature();
        }
    }
    
    RCSOFTCHECK(rcl_publish(&temp_publisher_, &temp_msg_, NULL));
    
    // ---- Status Data ----
    for (int i = 0; i < NUM_JOINTS; i++) {
        int base = i * 4;
        
        if (i >= ROBSTRIDE_START_IDX && i <= ROBSTRIDE_END_IDX) {
            RobstrideMotorType* rs_motor = static_cast<RobstrideMotorType*>(joints[i]);
            status_msg_.data.data[base + 0] = rs_motor->has_error ? 1 : 0;
            status_msg_.data.data[base + 1] = rs_motor->getMotorStatus();
            status_msg_.data.data[base + 2] = rs_motor->is_enabled ? 1 : 0;
            status_msg_.data.data[base + 3] = rs_motor->getFaultBits();
        }
        else if (i >= ODRIVE_START_IDX && i <= ODRIVE_END_IDX) {
            ODriveMotorType* od_motor = static_cast<ODriveMotorType*>(joints[i]);
            status_msg_.data.data[base + 0] = od_motor->getAxisError();
            status_msg_.data.data[base + 1] = od_motor->getAxisState();
            status_msg_.data.data[base + 2] = od_motor->is_enabled ? 1 : 0;
            status_msg_.data.data[base + 3] = (int32_t)(od_motor->getBusVoltage() * 100);
        }
    }
    
    RCSOFTCHECK(rcl_publish(&status_publisher_, &status_msg_, NULL));
    
    // Request additional data from ODrives
    for (int i = ODRIVE_START_IDX; i <= ODRIVE_END_IDX; i++) {
        ODriveMotorType* od_motor = static_cast<ODriveMotorType*>(joints[i]);
        od_motor->requestTemperature();
        od_motor->requestBusVoltage();
    }
}

void RosPublishers::handleMotorControlCommand(float cmd_value) {
    if (cmd_value == MOTOR_CMD_CLEAR_ERRORS) {
        DEBUG_PORT.println("CMD: Clearing motor errors...");
        for (int i = ROBSTRIDE_START_IDX; i <= ROBSTRIDE_END_IDX; i++) {
            RobstrideMotorType* rs_motor = static_cast<RobstrideMotorType*>(joints[i]);
            rs_motor->clearFault();
        }
        for (int i = ODRIVE_START_IDX; i <= ODRIVE_END_IDX; i++) {
            ODriveMotorType* od_motor = static_cast<ODriveMotorType*>(joints[i]);
            od_motor->clearErrors();
        }
    }
    else if (cmd_value == MOTOR_CMD_ENABLE_ALL) {
        DEBUG_PORT.println("CMD: Enabling all motors...");
        for (int i = 0; i < NUM_JOINTS; i++) {
            joints[i]->enable();
            delay(10);
        }
        
        // Wait for feedback, then send hold position commands
        delay(100);
        for (int i = ROBSTRIDE_START_IDX; i <= ROBSTRIDE_END_IDX; i++) {
            RobstrideMotorType* rs_motor = static_cast<RobstrideMotorType*>(joints[i]);
            rs_motor->holdPosition();
            delay(5);
        }
        
        // Reset watchdog timer to prevent immediate position command with stale data
        last_cmd_time_ = millis();
    }
    else if (cmd_value == MOTOR_CMD_DISABLE_ALL) {
        DEBUG_PORT.println("CMD: Disabling all motors...");
        for (int i = 0; i < NUM_JOINTS; i++) {
            joints[i]->disable();
            delay(5);
        }
    }
    else if (cmd_value == MOTOR_CMD_RESET_ENABLE) {
        DEBUG_PORT.println("CMD: Reset and enable all motors...");
        
        // Step 1: Clear faults on all motors
        for (int i = ROBSTRIDE_START_IDX; i <= ROBSTRIDE_END_IDX; i++) {
            RobstrideMotorType* rs_motor = static_cast<RobstrideMotorType*>(joints[i]);
            rs_motor->clearFault();
        }
        for (int i = ODRIVE_START_IDX; i <= ODRIVE_END_IDX; i++) {
            ODriveMotorType* od_motor = static_cast<ODriveMotorType*>(joints[i]);
            od_motor->clearErrors();
        }
        delay(100);  // Wait for fault clear to process
        
        // Step 2: Enable all motors
        for (int i = 0; i < NUM_JOINTS; i++) {
            joints[i]->enable();
            delay(10);
        }
        
        // Step 3: Wait for feedback to arrive (motors respond to enable with position)
        DEBUG_PORT.println("CMD: Waiting for motor feedback...");
        delay(150);  // Allow time for feedback frames to arrive and be processed
        
        // Step 4: Send "hold position" command to Robstride motors
        // Per docs 4.3.1: After enable, must send Type 1 control command
        // This tells each motor to hold its current position
        DEBUG_PORT.println("CMD: Sending hold position commands...");
        for (int i = ROBSTRIDE_START_IDX; i <= ROBSTRIDE_END_IDX; i++) {
            RobstrideMotorType* rs_motor = static_cast<RobstrideMotorType*>(joints[i]);
            rs_motor->holdPosition();
            delay(5);
        }
        
        // Step 5: Enable active reporting on Robstride motors (Type 24)
        // Per docs 4.1.11: Motors must receive this after each power cycle
        DEBUG_PORT.println("CMD: Enabling active reporting (20ms)...");
        for (int i = ROBSTRIDE_START_IDX; i <= ROBSTRIDE_END_IDX; i++) {
            RobstrideMotorType* rs_motor = static_cast<RobstrideMotorType*>(joints[i]);
            rs_motor->enableActiveReporting(true, 20);
            delay(5);
        }
        
        // Reset watchdog timer to prevent immediate position command with stale data
        last_cmd_time_ = millis();
        
        DEBUG_PORT.println("CMD: All motors reset and enabled");
    }
    else if (cmd_value == MOTOR_CMD_MODE_POLICY) {
        setControlMode(MODE_POLICY_BALANCE);
        DEBUG_PORT.println("CMD: Switched to POLICY_BALANCE mode");
    }
    else if (cmd_value == MOTOR_CMD_MODE_MANUAL) {
        setControlMode(MODE_PASSTHROUGH);
        DEBUG_PORT.println("CMD: Switched to PASSTHROUGH (Manual) mode");
    }
}

void RosPublishers::cmdCallback(const void* msgin) {
    if (!instance_) return;
    
    const sensor_msgs__msg__JointState* msg = (const sensor_msgs__msg__JointState*)msgin;
    
    // Safety check
    if (msg->position.size < NUM_JOINTS || msg->velocity.size < NUM_JOINTS) {
        return;
    }

    // Check for special motor control commands
    if (msg->effort.size > 0 && msg->effort.data[0] < -900.0) {
        instance_->handleMotorControlCommand((float)msg->effort.data[0]);
        return;
    }

    // If in POLICY mode, ignore joint commands (prevent fighting)
    if (instance_->control_mode_ == MODE_POLICY_BALANCE) {
        return;
    }

    // Update watchdog ONLY if in PASSTHROUGH mode
    // (In Policy mode, watchdog is petted by cmdVelCallback)
    instance_->last_cmd_time_ = millis();

    // Process Robstride motors (position + effort control)
    for (int i = ROBSTRIDE_START_IDX; i <= ROBSTRIDE_END_IDX; i++) {
        float pos = msg->position.data[i];
        float vel = 0.0f;
        float torque = 0.0f;
        
        if (msg->effort.size > (size_t)i) {
            torque = msg->effort.data[i];
        }
        
        joints[i]->setCommand(pos, vel, torque);
    }

    // Process ODrive motors (velocity control)
    for (int i = ODRIVE_START_IDX; i <= ODRIVE_END_IDX; i++) {
        float pos = 0.0f;
        float vel = msg->velocity.data[i];
        float torque = 0.0f;
        
        if (msg->effort.size > (size_t)i) {
            torque = msg->effort.data[i];
        }
        
        joints[i]->setCommand(pos, vel, torque);
    }
}

void RosPublishers::cmdVelCallback(const void* msgin) {
    if (!instance_) return;
    
    const geometry_msgs__msg__Twist* msg = (const geometry_msgs__msg__Twist*)msgin;
    
    instance_->cmd_vel_x_ = (float)msg->linear.x;
    instance_->cmd_vel_y_ = (float)msg->linear.y;
    instance_->cmd_vel_omega_ = (float)msg->angular.z;
    
    // Only pet watchdog if in POLICY mode
    if (instance_->control_mode_ == MODE_POLICY_BALANCE) {
        instance_->last_cmd_time_ = millis();
    }
}
