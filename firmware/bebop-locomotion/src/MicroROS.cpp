/**  
 * @file MicroROS.cpp  
 * @brief Micro-ROS communication implementation  
 */  

 #include "MicroROS.h"  
 #include <string.h>  
 
 // Singleton instance  
 MicroROSManager* MicroROSManager::instance_ = nullptr;  
 
 // Error handling macros  
 #define RCCHECK(fn) { rcl_ret_t temp_rc = fn; if((temp_rc != RCL_RET_OK)){ return false; }}  
 #define RCSOFTCHECK(fn) { rcl_ret_t temp_rc = fn; (void)temp_rc; }  
 
 // ============================================================================  
 // CONSTRUCTOR  
 // ============================================================================  
 
 MicroROSManager::MicroROSManager()   
     : state_(WAITING_FOR_AGENT)  
     , last_agent_ping_(0)
     , last_update_us_(0)
 {  
     instance_ = this;  
       
     // Initialize robot state  
     memset(&robot_state_, 0, sizeof(robot_state_));  
     robot_state_.quaternion[0] = 1.0f;  // w = 1 (identity rotation)  
     robot_state_.projected_gravity[2] = -1.0f;  // Upright  
 }  
 
 // ============================================================================  
 // INITIALIZATION  
 // ============================================================================  
 
 void MicroROSManager::init() {  
     set_microros_transports();  
     allocateMessages();  
     state_ = WAITING_FOR_AGENT;
     last_update_us_ = micros();
 }  
 
 void MicroROSManager::allocateMessages() {  
     // Joint state subscription  
     joint_state_msg_.name.data = (rosidl_runtime_c__String*)malloc(NUM_JOINTS * sizeof(rosidl_runtime_c__String));  
     joint_state_msg_.name.size = 0;  
     joint_state_msg_.name.capacity = NUM_JOINTS;  
       
     for (int i = 0; i < NUM_JOINTS; i++) {  
         joint_state_msg_.name.data[i].data = (char*)malloc(32);  
         joint_state_msg_.name.data[i].size = 0;  
         joint_state_msg_.name.data[i].capacity = 32;  
     }  
       
     joint_state_msg_.position.data = (double*)malloc(NUM_JOINTS * sizeof(double));  
     joint_state_msg_.position.size = 0;  
     joint_state_msg_.position.capacity = NUM_JOINTS;  
       
     joint_state_msg_.velocity.data = (double*)malloc(NUM_JOINTS * sizeof(double));  
     joint_state_msg_.velocity.size = 0;  
     joint_state_msg_.velocity.capacity = NUM_JOINTS;  
       
     joint_state_msg_.effort.data = (double*)malloc(NUM_JOINTS * sizeof(double));  
     joint_state_msg_.effort.size = 0;  
     joint_state_msg_.effort.capacity = NUM_JOINTS;  
       
     joint_state_msg_.header.frame_id.data = (char*)malloc(32);  
     joint_state_msg_.header.frame_id.size = 0;  
     joint_state_msg_.header.frame_id.capacity = 32;  
       
     // Joint command publisher message  
     joint_cmd_msg_.position.data = (double*)malloc(NUM_JOINTS * sizeof(double));  
     joint_cmd_msg_.position.size = NUM_JOINTS;  
     joint_cmd_msg_.position.capacity = NUM_JOINTS;  
       
     joint_cmd_msg_.velocity.data = (double*)malloc(NUM_JOINTS * sizeof(double));  
     joint_cmd_msg_.velocity.size = NUM_JOINTS;  
     joint_cmd_msg_.velocity.capacity = NUM_JOINTS;  
       
     joint_cmd_msg_.effort.data = nullptr;  
     joint_cmd_msg_.effort.size = 0;  
     joint_cmd_msg_.effort.capacity = 0;  
       
     // IMPORTANT: Joint names are required for ArticulationController in Isaac Sim!  
     static const char* JOINT_NAMES[NUM_JOINTS] = {  
         "left_hip_pitch",  
         "right_hip_pitch",   
         "left_knee_pitch",  
         "right_knee_pitch",  
         "left_wheel",  
         "right_wheel"  
     };  
       
     joint_cmd_msg_.name.data = (rosidl_runtime_c__String*)malloc(NUM_JOINTS * sizeof(rosidl_runtime_c__String));  
     joint_cmd_msg_.name.size = NUM_JOINTS;  
     joint_cmd_msg_.name.capacity = NUM_JOINTS;  
       
     for (int i = 0; i < NUM_JOINTS; i++) {  
         joint_cmd_msg_.name.data[i].data = (char*)malloc(32);  
         joint_cmd_msg_.name.data[i].capacity = 32;  
         strcpy(joint_cmd_msg_.name.data[i].data, JOINT_NAMES[i]);  
         joint_cmd_msg_.name.data[i].size = strlen(JOINT_NAMES[i]);  
     }  
       
     joint_cmd_msg_.header.frame_id.data = nullptr;  
     joint_cmd_msg_.header.frame_id.size = 0;  
     joint_cmd_msg_.header.frame_id.capacity = 0;  
 }  
 
 // ============================================================================  
 // CONNECTION MANAGEMENT  
 // ============================================================================  
 
 bool MicroROSManager::createEntities() {  
     allocator_ = rcl_get_default_allocator();  
       
     rcl_ret_t ret = rclc_support_init(&support_, 0, NULL, &allocator_);  
     if (ret != RCL_RET_OK) return false;  
       
     ret = rclc_node_init_default(&node_, "teensy_policy", "", &support_);  
     if (ret != RCL_RET_OK) {  
         rclc_support_fini(&support_);  
         return false;  
     }  
       
     // Subscribers  
     RCCHECK(rclc_subscription_init_default(&joint_state_sub_, &node_,  
         ROSIDL_GET_MSG_TYPE_SUPPORT(sensor_msgs, msg, JointState), "/joint_states"));  
       
     RCCHECK(rclc_subscription_init_default(&imu_sub_, &node_,  
         ROSIDL_GET_MSG_TYPE_SUPPORT(sensor_msgs, msg, Imu), "/imu/data"));  
       
     RCCHECK(rclc_subscription_init_default(&cmd_vel_sub_, &node_,  
         ROSIDL_GET_MSG_TYPE_SUPPORT(geometry_msgs, msg, Twist), "/cmd_vel"));  
       
     // Publisher for joint commands  
     RCCHECK(rclc_publisher_init_default(&joint_cmd_pub_, &node_,  
         ROSIDL_GET_MSG_TYPE_SUPPORT(sensor_msgs, msg, JointState), "/joint_commands"));  
       
     RCCHECK(rclc_executor_init(&executor_, &support_.context, 3, &allocator_));  
     RCCHECK(rclc_executor_add_subscription(&executor_, &joint_state_sub_, &joint_state_msg_,   
         &MicroROSManager::jointStateCallback, ON_NEW_DATA));  
     RCCHECK(rclc_executor_add_subscription(&executor_, &imu_sub_, &imu_msg_,   
         &MicroROSManager::imuCallback, ON_NEW_DATA));  
     RCCHECK(rclc_executor_add_subscription(&executor_, &cmd_vel_sub_, &cmd_vel_msg_,   
         &MicroROSManager::cmdVelCallback, ON_NEW_DATA));  
       
     return true;  
 }  
 
 void MicroROSManager::destroyEntities() {  
     rmw_context_t* rmw_context = rcl_context_get_rmw_context(&support_.context);  
     (void)rmw_uros_set_context_entity_destroy_session_timeout(rmw_context, 0);  
       
    RCSOFTCHECK(rcl_subscription_fini(&joint_state_sub_, &node_));
    RCSOFTCHECK(rcl_subscription_fini(&imu_sub_, &node_));
    RCSOFTCHECK(rcl_subscription_fini(&cmd_vel_sub_, &node_));
    RCSOFTCHECK(rcl_publisher_fini(&joint_cmd_pub_, &node_));
    RCSOFTCHECK(rclc_executor_fini(&executor_));
    RCSOFTCHECK(rcl_node_fini(&node_));
    RCSOFTCHECK(rclc_support_fini(&support_));
 }  
 
 // ============================================================================  
 // UPDATE LOOP  
 // ============================================================================  
 
 MicroROSManager::State MicroROSManager::update() {  
     uint32_t now = millis();  
     uint32_t now_us = micros();
       
     switch (state_) {  
         case WAITING_FOR_AGENT:  
             if (createEntities()) {  
                 state_ = AGENT_CONNECTED;  
                 last_agent_ping_ = now;
                 last_update_us_ = now_us;
             } else {  
                 delay(500);  
             }  
             break;  
               
         case AGENT_CONNECTED:  
             {  
                 rcl_ret_t ret = rclc_executor_spin_some(&executor_, RCL_MS_TO_NS(1));  
                 if (ret == RCL_RET_OK) {  
                     last_agent_ping_ = now;  
                 }  
                 
                 // Update velocity estimate
                 float dt = (now_us - last_update_us_) * 1e-6f;
                 if (dt > 0.0f && dt < 0.1f) {  // Sanity check
                     updateVelocityEstimate(dt);
                 }
                 last_update_us_ = now_us;
                   
                 if ((now - last_agent_ping_) > AGENT_TIMEOUT_MS) {  
                     state_ = AGENT_DISCONNECTED;  
                 }  
             }  
             break;  
               
         case AGENT_DISCONNECTED:  
             destroyEntities();  
             state_ = WAITING_FOR_AGENT;  
             delay(1000);  
             break;  
     }  
       
     return state_;  
 }  
 
 // ============================================================================  
 // VELOCITY ESTIMATION (Matches Python logic, but simpler)
 // ============================================================================  
 
 void MicroROSManager::updateVelocityEstimate(float dt) {
     // 1. Wheel Odometry (Forward velocity estimate)
     // Indices 4 and 5 are wheels in the joint array
     float avg_wheel_vel = (robot_state_.joint_velocities[4] + robot_state_.joint_velocities[5]) / 2.0f;
     float odom_vel_x = avg_wheel_vel * WHEEL_RADIUS;
     
     // 2. STATIC SWITCH (Critical for drift prevention)
     // If wheels are stopped, force velocity to zero
     if (fabsf(avg_wheel_vel) < 0.1f) {
         robot_state_.base_lin_vel[0] = 0.0f;
         robot_state_.base_lin_vel[1] = 0.0f;
         robot_state_.base_lin_vel[2] = 0.0f;
         return;
     }
     
     // 3. IMU Integration (with rotation compensation)
     float yaw_rate = robot_state_.base_ang_vel[2];
     float cos_y = cosf(yaw_rate * dt);
     float sin_y = sinf(yaw_rate * dt);
     
     float vx = robot_state_.base_lin_vel[0];
     float vy = robot_state_.base_lin_vel[1];
     
     // Rotate velocity by yaw rate
     robot_state_.base_lin_vel[0] = vx * cos_y + vy * sin_y;
     robot_state_.base_lin_vel[1] = -vx * sin_y + vy * cos_y;
     
     // Integrate acceleration
     robot_state_.base_lin_vel[0] += robot_state_.base_lin_accel[0] * dt;
     robot_state_.base_lin_vel[1] += robot_state_.base_lin_accel[1] * dt;
     robot_state_.base_lin_vel[2] = 0.0f;  // Assume no vertical velocity
     
     // 4. Fusion with Odometry (trust wheels for forward velocity)
     robot_state_.base_lin_vel[0] = (1.0f - VEL_FUSION_ALPHA) * robot_state_.base_lin_vel[0] + 
                                     VEL_FUSION_ALPHA * odom_vel_x;
     
     // 5. Decay lateral velocity (should be minimal for wheeled robot)
     robot_state_.base_lin_vel[1] *= VEL_DECAY;
 }
 
 // ============================================================================  
 // SENSOR DATA  
 // ============================================================================  
 
 bool MicroROSManager::hasValidSensorData() const {  
     uint32_t now = millis();  
     bool have_joints = (now - robot_state_.last_joint_state_time) <= 200;  
     bool have_imu = (now - robot_state_.last_imu_time) <= 200;  
     return have_joints && have_imu;  
 }  
 
// ============================================================================  
// COMMAND PUBLISHING  
// ============================================================================  

// Safety limits for hip and knee joints (radians)
static constexpr float JOINT_POS_MAX = 1.5f;
static constexpr float JOINT_POS_MIN = -1.5f;

static inline float clampJointPosition(float pos) {
    if (pos > JOINT_POS_MAX) return JOINT_POS_MAX;
    if (pos < JOINT_POS_MIN) return JOINT_POS_MIN;
    return pos;
}

void MicroROSManager::publishJointCommand(const JointCommand& cmd) {  
    // Leg positions with safety limits
    joint_cmd_msg_.position.data[0] = clampJointPosition(cmd.leg_positions[0]);  
    joint_cmd_msg_.position.data[1] = clampJointPosition(cmd.leg_positions[1]);  
    joint_cmd_msg_.position.data[2] = clampJointPosition(cmd.leg_positions[2]);  
    joint_cmd_msg_.position.data[3] = clampJointPosition(cmd.leg_positions[3]);  
    joint_cmd_msg_.position.data[4] = 0.0;  // Wheels don't use position  
    joint_cmd_msg_.position.data[5] = 0.0;  
      
    // Wheel velocities  
    joint_cmd_msg_.velocity.data[0] = 0.0;  // Legs don't use velocity  
    joint_cmd_msg_.velocity.data[1] = 0.0;  
    joint_cmd_msg_.velocity.data[2] = 0.0;  
    joint_cmd_msg_.velocity.data[3] = 0.0;  
    joint_cmd_msg_.velocity.data[4] = cmd.wheel_velocities[0];  
    joint_cmd_msg_.velocity.data[5] = cmd.wheel_velocities[1];  
      
    RCSOFTCHECK(rcl_publish(&joint_cmd_pub_, &joint_cmd_msg_, NULL));  
}
 
 // ============================================================================  
 // HELPER FUNCTIONS  
 // ============================================================================  
 
 void MicroROSManager::computeProjectedGravity() {  
     float w = robot_state_.quaternion[0];  
     float x = robot_state_.quaternion[1];  
     float y = robot_state_.quaternion[2];  
     float z = robot_state_.quaternion[3];  
       
     robot_state_.projected_gravity[0] = 2.0f * (x * z - w * y);  
     robot_state_.projected_gravity[1] = 2.0f * (y * z + w * x);  
     robot_state_.projected_gravity[2] = -(w * w - x * x - y * y + z * z);  
 }  
 
 // ============================================================================  
 // CALLBACKS  
 // ============================================================================  
 
void MicroROSManager::jointStateCallback(const void* msgin) {  
    if (!instance_) return;  
      
    const sensor_msgs__msg__JointState* msg = (const sensor_msgs__msg__JointState*)msgin;  
      
    if (msg->position.data == nullptr) return;
    if (msg->name.data == nullptr || msg->name.size == 0) {
        // Fallback: No names provided, assume data is already in training order
        for (size_t i = 0; i < msg->position.size && i < NUM_JOINTS; i++) {  
            instance_->robot_state_.joint_positions[i] = (float)msg->position.data[i];  
        }  
        for (size_t i = 0; i < msg->velocity.size && i < NUM_JOINTS; i++) {  
            instance_->robot_state_.joint_velocities[i] = (float)msg->velocity.data[i];  
        }
    } else {
        // NAME-BASED REORDERING: Convert from ROS message order to training order
        // This ensures joint data matches the policy's expected indices regardless
        // of URDF/USD joint ordering.
        for (size_t train_idx = 0; train_idx < NUM_JOINTS; train_idx++) {
            const char* target_name = TRAINING_JOINT_ORDER[train_idx];
            
            // Find this joint in the incoming message
            for (size_t msg_idx = 0; msg_idx < msg->name.size; msg_idx++) {
                if (msg->name.data[msg_idx].data != nullptr &&
                    strcmp(msg->name.data[msg_idx].data, target_name) == 0) {
                    // Found the joint - copy to correct training index
                    if (msg_idx < msg->position.size) {
                        instance_->robot_state_.joint_positions[train_idx] = 
                            (float)msg->position.data[msg_idx];
                    }
                    if (msg_idx < msg->velocity.size) {
                        instance_->robot_state_.joint_velocities[train_idx] = 
                            (float)msg->velocity.data[msg_idx];
                    }
                    break;
                }
            }
        }
    }

    // Extract Simulation Time from Header
    double sec = (double)msg->header.stamp.sec;
    double nanosec = (double)msg->header.stamp.nanosec;
    instance_->robot_state_.sim_time = sec + (nanosec * 1e-9);

    instance_->robot_state_.last_joint_state_time = millis();  
}
 
 void MicroROSManager::imuCallback(const void* msgin) {  
     if (!instance_) return;  
       
     const sensor_msgs__msg__Imu* msg = (const sensor_msgs__msg__Imu*)msgin;  
       
     instance_->robot_state_.quaternion[0] = (float)msg->orientation.w;  
     instance_->robot_state_.quaternion[1] = (float)msg->orientation.x;  
     instance_->robot_state_.quaternion[2] = (float)msg->orientation.y;  
     instance_->robot_state_.quaternion[3] = (float)msg->orientation.z;  
       
     instance_->robot_state_.base_ang_vel[0] = (float)msg->angular_velocity.x;  
     instance_->robot_state_.base_ang_vel[1] = (float)msg->angular_velocity.y;  
     instance_->robot_state_.base_ang_vel[2] = (float)msg->angular_velocity.z;  
       
     // Extract linear acceleration from IMU (body frame, gravity-compensated)  
     instance_->robot_state_.base_lin_accel[0] = (float)msg->linear_acceleration.x;  
     instance_->robot_state_.base_lin_accel[1] = (float)msg->linear_acceleration.y;  
     instance_->robot_state_.base_lin_accel[2] = (float)msg->linear_acceleration.z;  
       
     instance_->computeProjectedGravity();  
     instance_->robot_state_.last_imu_time = millis();  
 }  
 
 void MicroROSManager::cmdVelCallback(const void* msgin) {  
     if (!instance_) return;  
       
     const geometry_msgs__msg__Twist* msg = (const geometry_msgs__msg__Twist*)msgin;  
     instance_->robot_state_.cmd_vel[0] = (float)msg->linear.x;  
     instance_->robot_state_.cmd_vel[1] = (float)msg->linear.y;  
     instance_->robot_state_.cmd_vel[2] = (float)msg->angular.z;  
 }