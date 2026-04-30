/**  
 * @file MicroROS.h  
 * @brief Micro-ROS communication interface  
 */  

 #ifndef MICROROS_H  
 #define MICROROS_H  
 
 #include <micro_ros_arduino.h>  
 #include <rcl/rcl.h>  
 #include <rclc/rclc.h>  
 #include <rclc/executor.h>  
 #include <sensor_msgs/msg/joint_state.h>  
 #include <sensor_msgs/msg/imu.h>  
 #include <geometry_msgs/msg/twist.h>  
 
// ============================================================================  
// CONSTANTS  
// ============================================================================  

#define NUM_JOINTS 6  
#define AGENT_TIMEOUT_MS 5000

// ============================================================================  
// TRAINING JOINT ORDER (MUST match bebop_base_cfg.py!)
// ============================================================================  
// This is the canonical order used by:
//   - Training (bebop_base_cfg.py)
//   - Policy inference (observation/action indices)
//   - This firmware (after reordering from ROS)
//
// DO NOT CHANGE unless you also update training and Python controller!
static const char* TRAINING_JOINT_ORDER[NUM_JOINTS] = {
    "left_hip_pitch",   // [0] Policy leg action[0], obs joint_pos[0]
    "right_hip_pitch",  // [1] Policy leg action[1], obs joint_pos[1]
    "left_knee_pitch",  // [2] Policy leg action[2], obs joint_pos[2]
    "right_knee_pitch", // [3] Policy leg action[3], obs joint_pos[3]
    "left_wheel",       // [4] Policy wheel action[0], obs joint_pos[4]
    "right_wheel"       // [5] Policy wheel action[1], obs joint_pos[5]
};
 
 // ============================================================================  
 // DATA STRUCTURES  
 // ============================================================================  
 
 struct RobotState {  
     // Joint state (in training order!)
     float joint_positions[NUM_JOINTS];  
     float joint_velocities[NUM_JOINTS];  
     
     // IMU data  
     float quaternion[4];           // [w, x, y, z]  
     float base_ang_vel[3];         // [x, y, z] rad/s (body frame)  
     float base_lin_accel[3];       // [x, y, z] m/s² (body frame, gravity-compensated)  
     
     // Derived state  
     float projected_gravity[3];    // Gravity vector in body frame  
     float base_lin_vel[3];         // [x, y, z] m/s (body frame, ESTIMATED)
     
     // Velocity commands  
     float cmd_vel[3];              // [vx, vy, wz]  
     
     // Timing  
     double sim_time;                 // <--- ADDED THIS FIELD
     uint32_t last_joint_state_time;  
     uint32_t last_imu_time;  
 };  
 
 struct JointCommand {  
     float leg_positions[4];     // Hip/Knee position commands  
     float wheel_velocities[2];  // Wheel velocity commands  
 };  
 
 // ============================================================================  
 // MICRO-ROS MANAGER  
 // ============================================================================  
 
 class MicroROSManager {  
 public:  
     enum State {  
         WAITING_FOR_AGENT,  
         AGENT_CONNECTED,  
         AGENT_DISCONNECTED  
     };  
 
     MicroROSManager();  
     
     void init();  
     State update();  
     
     // Data access  
     const RobotState& getRobotState() const { return robot_state_; }  
     bool hasValidSensorData() const;  
     
     // Command publishing  
     void publishJointCommand(const JointCommand& cmd);  
 
 private:  
     // ROS entities  
     rcl_allocator_t allocator_;  
     rclc_support_t support_;  
     rcl_node_t node_;  
     rclc_executor_t executor_;  
     
     // Subscribers  
     rcl_subscription_t joint_state_sub_;  
     rcl_subscription_t imu_sub_;  
     rcl_subscription_t cmd_vel_sub_;  
     
     // Publisher  
     rcl_publisher_t joint_cmd_pub_;  
     
     // Messages  
     sensor_msgs__msg__JointState joint_state_msg_;  
     sensor_msgs__msg__Imu imu_msg_;  
     geometry_msgs__msg__Twist cmd_vel_msg_;  
     sensor_msgs__msg__JointState joint_cmd_msg_;  
     
     // State  
     State state_;  
     RobotState robot_state_;  
     uint32_t last_agent_ping_;  
     uint32_t last_update_us_;  // For velocity estimation
     
     // Velocity estimation constants
     static constexpr float WHEEL_RADIUS = 0.05f;  // meters
     static constexpr float VEL_FUSION_ALPHA = 0.05f;  // Odometry fusion weight
     static constexpr float VEL_DECAY = 0.90f;  // Lateral velocity decay
     
     // Internal methods  
     bool createEntities();  
     void destroyEntities();  
     void allocateMessages();  
     void computeProjectedGravity();  
     void updateVelocityEstimate(float dt);
     
     // Callbacks  
     static void jointStateCallback(const void* msgin);  
     static void imuCallback(const void* msgin);  
     static void cmdVelCallback(const void* msgin);  
     
     // Singleton for callbacks  
     static MicroROSManager* instance_;  
 };  
 
 #endif // MICROROS_H