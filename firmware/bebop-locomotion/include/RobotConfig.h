/**
 * @file RobotConfig.h
 * @brief Hardware configuration and joint mapping
 * 
 * NAMING CONVENTION: {side}_{joint}_{axis}
 * This allows adding DOF without breaking existing names:
 *   - left_hip_pitch  → can add left_hip_roll, left_hip_yaw
 *   - left_knee_pitch → can add left_knee_roll
 *   - left_wheel      → single axis, no suffix needed
 * 
 * All motors share a single CAN bus at 1 Mbps.
 */

 #ifndef ROBOT_CONFIG_H
 #define ROBOT_CONFIG_H
 
 #include <FlexCAN_T4.h>
 #include "GenericMotor.h"

// ============================================================================
// DEBUG FLAGS (declared before motor includes so they can use them)
// ============================================================================

// Enable to log CAN messages (toggle via 'candebug' serial command)
extern bool can_tx_debug;   // Outgoing CAN messages
extern bool can_rx_debug;   // Incoming CAN messages
extern bool imu_debug;      // IMU data (toggle via 'imudebug' serial command)
extern bool status_logging; // Periodic status output (toggle via 'status off' command)

 #include "RobstrideMotor.h"
 #include "ODriveMotor.h"
 
 // ============================================================================
 // CAN BUS TYPE DEFINITION
 // ============================================================================
 
 // Define the CAN bus type once - used throughout the codebase
 // Pins: TX=22, RX=23 (CAN1 on Teensy 4.1)
 typedef FlexCAN_T4<CAN1, RX_SIZE_256, TX_SIZE_64> CANBusType;
 
 // Motor type aliases for cleaner code
 typedef RobstrideMotor<CANBusType> RobstrideMotorType;
 typedef ODriveMotor<CANBusType> ODriveMotorType;
 
 // ============================================================================
 // CAN BUS CONFIGURATION
 // ============================================================================
 
 #define CAN_BAUD_RATE 1000000  // 1 Mbps (both Robstride and ODrive)
 
 // ============================================================================
 // MOTOR PARAMETERS
 // ============================================================================

// ODrive M8325s Motor Parameters
#define ODRIVE_TORQUE_CONSTANT  0.083f  // Nm/A (for M8325s 100KV)

// ODrive Velocity Limits (match ODrive GUI config)
#define ODRIVE_VEL_LIMIT_REV_S  40.0f   // Soft velocity limit (rev/s) - matches ODrive config
#define ODRIVE_CURRENT_LIMIT_A  60.0f   // Current limit (A) - no torque limit in ODrive config

 // ============================================================================
 // JOINT CONFIGURATION
 // ============================================================================
 
 // Current configuration: 6 joints
 // Expandable by adding new indices (e.g., JOINT_LEFT_HIP_ROLL = 6)
 #define NUM_JOINTS 6
 
 // Joint indices (used in ROS message arrays)
 // Leg joints (Robstride - position/torque control)
 #define JOINT_LEFT_HIP_PITCH    0
 #define JOINT_RIGHT_HIP_PITCH   1
 #define JOINT_LEFT_KNEE_PITCH   2
 #define JOINT_RIGHT_KNEE_PITCH  3
 
 // Wheel joints (ODrive - velocity control)
 #define JOINT_LEFT_WHEEL        4
 #define JOINT_RIGHT_WHEEL       5
 
 // Future expansion examples (uncomment when adding hardware):
 // #define JOINT_LEFT_HIP_ROLL     6
 // #define JOINT_RIGHT_HIP_ROLL    7
 // #define JOINT_LEFT_ANKLE_PITCH  8
 // #define JOINT_RIGHT_ANKLE_PITCH 9
 
 // ============================================================================
 // CAN IDs - Robstride Motors
 // ============================================================================
 
#define ROBSTRIDE_ID_LEFT_HIP_PITCH   31   // 0x1F
#define ROBSTRIDE_ID_RIGHT_HIP_PITCH  41   // 0x29
#define ROBSTRIDE_ID_LEFT_KNEE_PITCH  34   // 0x22
#define ROBSTRIDE_ID_RIGHT_KNEE_PITCH 44   // 0x2C
 
 // Motor model selection (RS01, RS02, RS03, RS04)
 // Change these to match your actual hardware!
 #define ROBSTRIDE_MODEL_HIP   RS_MODEL_04   // 120 N.m peak (larger for hip)
 #define ROBSTRIDE_MODEL_KNEE  RS_MODEL_04   // 120 N.m peak (or RS03 for lighter knee)
 
 // Future expansion (reserve IDs):
 // #define ROBSTRIDE_ID_LEFT_HIP_ROLL    0x14
 // #define ROBSTRIDE_ID_RIGHT_HIP_ROLL   0x15
 
 // ============================================================================
 // CAN IDs - ODrive Node IDs
 // ============================================================================
 
 // ODrive CAN frame ID = (node_id << 5) | cmd_id
 // Node 35 → frames 0x460-0x47F (left wheel)
 // Node 45 → frames 0x5A0-0x5BF (right wheel)
 #define ODRIVE_NODE_LEFT_WHEEL   35
 #define ODRIVE_NODE_RIGHT_WHEEL  45
 
 // ============================================================================
 // JOINT NAME STRINGS (for ROS feedback messages)
 // ============================================================================
 
 // Extensible naming: {side}_{joint}_{axis}
 static const char* JOINT_NAMES[NUM_JOINTS] = {
     "left_hip_pitch",
     "right_hip_pitch",
     "left_knee_pitch",
     "right_knee_pitch",
     "left_wheel",
     "right_wheel"
 };
 
 // ============================================================================
 // CONTROL LOOP TIMING
 // ============================================================================
 
 #define CONTROL_LOOP_HZ     200     // Main control frequency (Hz)
 #define FEEDBACK_PUBLISH_HZ 100     // ROS feedback publish rate (Hz)
 #define WATCHDOG_TIMEOUT_MS 200     // Disable motors if no command received
 
 // ============================================================================
 // MOTOR TYPE BOUNDARIES
 // ============================================================================
 
 // Define which indices are which motor type
 // This makes the main loop cleaner and easier to extend
 #define ROBSTRIDE_START_IDX  0
 #define ROBSTRIDE_END_IDX    3  // Inclusive (indices 0-3)
 #define ODRIVE_START_IDX     4
 #define ODRIVE_END_IDX       5  // Inclusive (indices 4-5)
 
// ============================================================================
// GLOBAL CAN BUS INSTANCE (defined in main.cpp)
// ============================================================================

extern CANBusType canBus;
extern GenericMotor* joints[NUM_JOINTS];
 
 // ============================================================================
 // SETUP FUNCTION
 // ============================================================================
 
inline void setupMotors() {
    // Initialize CAN bus
    canBus.begin();
    canBus.setBaudRate(CAN_BAUD_RATE);
    
    // =========================================================================
    // MAILBOX CONFIGURATION FOR REAL-TIME CONTROL
    // =========================================================================
    // For a balancing robot, we MUST use mailboxes (not FIFO) for sensor data.
    // Mailboxes auto-overwrite with newest data - critical for control loops.
    // FIFO queues old data first, causing phase lag and instability.
    //
    // Layout:
    //   MB0-3: Robstride leg feedback (EXT frames) - position control
    //   MB4-5: ODrive wheel feedback (STD frames) - velocity/balance control
    //   MB6-7: ODrive heartbeat (STD frames) - state monitoring
    //   MB8+:  TX mailboxes (auto-configured)
    // =========================================================================
    
    canBus.setMaxMB(16);
    
    // Robstride motors (Extended frames) - Leg position feedback
    // Each motor gets its own mailbox so newest position is always available
    canBus.setMB(MB0, RX, EXT);  // Left hip pitch (id=31)
    canBus.setMB(MB1, RX, EXT);  // Right hip pitch (id=41)
    canBus.setMB(MB2, RX, EXT);  // Left knee pitch (id=34)
    canBus.setMB(MB3, RX, EXT);  // Right knee pitch (id=44)
    
    // ODrive motors (Standard frames) - Wheel velocity feedback (critical for balance!)
    canBus.setMB(MB4, RX, STD);  // Left wheel encoder estimates
    canBus.setMB(MB5, RX, STD);  // Right wheel encoder estimates
    
    // Extra RX mailboxes for heartbeat/status messages
    canBus.setMB(MB6, RX, STD);  // ODrive heartbeat
    canBus.setMB(MB7, RX, STD);  // ODrive heartbeat
    
    // Enable interrupts for immediate processing
    canBus.enableMBInterrupts();
 
     // ---- Robstride Motors (Legs) ----
     // Specify model for correct torque/velocity scaling (RS_MODEL_01/02/03/04)
     joints[JOINT_LEFT_HIP_PITCH] = new RobstrideMotorType(
         ROBSTRIDE_ID_LEFT_HIP_PITCH, JOINT_LEFT_HIP_PITCH, &canBus, ROBSTRIDE_MODEL_HIP);
     
     joints[JOINT_RIGHT_HIP_PITCH] = new RobstrideMotorType(
         ROBSTRIDE_ID_RIGHT_HIP_PITCH, JOINT_RIGHT_HIP_PITCH, &canBus, ROBSTRIDE_MODEL_HIP);
     
     joints[JOINT_LEFT_KNEE_PITCH] = new RobstrideMotorType(
         ROBSTRIDE_ID_LEFT_KNEE_PITCH, JOINT_LEFT_KNEE_PITCH, &canBus, ROBSTRIDE_MODEL_KNEE);
     
     joints[JOINT_RIGHT_KNEE_PITCH] = new RobstrideMotorType(
         ROBSTRIDE_ID_RIGHT_KNEE_PITCH, JOINT_RIGHT_KNEE_PITCH, &canBus, ROBSTRIDE_MODEL_KNEE);
 
     // ---- ODrive Motors (Wheels) ----
     joints[JOINT_LEFT_WHEEL] = new ODriveMotorType(
         ODRIVE_NODE_LEFT_WHEEL, JOINT_LEFT_WHEEL, &canBus, ODRIVE_TORQUE_CONSTANT);
     
     joints[JOINT_RIGHT_WHEEL] = new ODriveMotorType(
         ODRIVE_NODE_RIGHT_WHEEL, JOINT_RIGHT_WHEEL, &canBus, ODRIVE_TORQUE_CONSTANT);
 
     // Configure ODrives for velocity control (matches ODrive GUI config)
    ODriveMotorType* left_wheel = static_cast<ODriveMotorType*>(joints[JOINT_LEFT_WHEEL]);
    ODriveMotorType* right_wheel = static_cast<ODriveMotorType*>(joints[JOINT_RIGHT_WHEEL]);
    
    left_wheel->setControlMode(ODRIVE_CONTROL_MODE_VELOCITY);
    right_wheel->setControlMode(ODRIVE_CONTROL_MODE_VELOCITY);
    
    // Set velocity and current limits (matches ODrive GUI: 40 rev/s soft limit)
    left_wheel->setLimits(ODRIVE_VEL_LIMIT_REV_S, ODRIVE_CURRENT_LIMIT_A);
    right_wheel->setLimits(ODRIVE_VEL_LIMIT_REV_S, ODRIVE_CURRENT_LIMIT_A);
 }
 
 #endif // ROBOT_CONFIG_H
 