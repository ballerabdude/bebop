/**
 * @file main.cpp
 * @brief Teensy 4.1 CAN Bridge for ROS 2 (micro-ROS)
 * 
 * This firmware acts as a bridge between ROS 2 and motor hardware.
 * It receives JointState commands via micro-ROS and translates them
 * to CAN messages for Robstride and ODrive motors.
 * 
 * Published Topics:
 *   /joint_states     - Position, velocity, torque feedback (100Hz)
 *   /imu/data         - IMU orientation and angular velocity (100Hz)
 *   /motor_temps      - Motor temperatures (10Hz)
 *   /motor_status     - Fault codes, enable states, voltages (10Hz)
 * 
 * Subscribed Topics:
 *   /joint_commands   - Position, velocity, torque commands
 */

#include <Arduino.h>
#include <FlexCAN_T4.h>
#include "BebopPolicy.h"  // Include generated policy

// Robot configuration and hardware
#include "RobotConfig.h"
#include "BNO085_IMU.h"

// Modular components
#include "SerialCommands.h"
#include "RosPublishers.h"

// ============================================================================
// CONFIGURATION
// ============================================================================

// POLICY CONFIGURATION (Must match training!)
#define HISTORY_STEPS 1
#define OBS_FRAME_DIM bebop_policy::OBS_DIM
#define TOTAL_OBS_DIM (OBS_FRAME_DIM * HISTORY_STEPS)

// TIMING
#define POLICY_RATE_HZ 100
#define POLICY_INTERVAL_US (1000000 / POLICY_RATE_HZ)
#define DECIMATION 1  // Run policy every N loops if loop is faster (e.g. 100Hz loop -> decimation 2)

// SAFETY LIMITS (policy output clamps - well within ODrive hardware limits of 40 rev/s)
const float MAX_LEG_POS_RAD = 0.8f;
const float MAX_WHEEL_VEL_RAD_S = 20.0f;  // ~3.2 rev/s - conservative for RL policy

// SCALES
const float SCALE_LIN_VEL = bebop_policy::SCALE_LIN_VEL;
const float SCALE_ANG_VEL = bebop_policy::SCALE_ANG_VEL;
const float SCALE_DOF_POS = bebop_policy::SCALE_DOF_POS;
const float SCALE_DOF_VEL = bebop_policy::SCALE_DOF_VEL;
const float SCALE_ACTION_LEGS = bebop_policy::SCALE_ACTION_LEGS;
const float SCALE_ACTION_WHEELS = bebop_policy::SCALE_ACTION_WHEELS;

const float CLIP_LIN_VEL = bebop_policy::CLIP_LIN_VEL;
const float CLIP_ANG_VEL = bebop_policy::CLIP_ANG_VEL;
const float CLIP_DOF_VEL = bebop_policy::CLIP_DOF_VEL;

// ============================================================================
// GLOBAL VARIABLES
// ============================================================================

// CAN bus instance (type defined in RobotConfig.h)
CANBusType canBus;

// Motor array
GenericMotor* joints[NUM_JOINTS];

// IMU
BNO085_IMU imu;

// Timing
uint32_t last_publish_time = 0;
uint32_t last_imu_publish_time = 0;
uint32_t last_imu_update_time = 0;
uint32_t last_diag_publish_time = 0;
uint32_t last_timing_report_ms = 0;

// Timing intervals
const uint32_t publish_interval_ms = 1000 / FEEDBACK_PUBLISH_HZ;  // 10ms for 100Hz
const uint32_t imu_publish_interval_ms = 10;   // 100 Hz IMU
const uint32_t imu_update_interval_ms = 5;     // 200 Hz IMU update
const uint32_t diag_publish_interval_ms = 100; // 10 Hz diagnostics

// Status LED
#define LED_PIN 13
bool led_state = false;

// CAN debug counters
volatile uint32_t can_rx_count = 0;
volatile uint32_t can_rx_std_count = 0;
volatile uint32_t can_rx_ext_count = 0;

// Debug flags (toggle via serial commands)
bool can_tx_debug = false;
bool can_rx_debug = false;
bool imu_debug = false;
bool status_logging = true;  // Periodic status output (every 2s)

// POLICY STATE
bebop_policy::Policy policy;
float current_obs[OBS_FRAME_DIM];
float history_buffer[TOTAL_OBS_DIM];
float action_buffer[bebop_policy::ACTION_DIM];
float last_action[bebop_policy::ACTION_DIM] = {0};

uint32_t last_policy_us = 0;
uint32_t step_count = 0;

// State Estimation
float est_base_lin_vel[3] = {0}; // [x, y, z] body frame
float est_projected_gravity[3] = {0, 0, -1};
float default_joint_pos[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}; // Should match training

// Velocity estimation params
const float EST_WHEEL_RADIUS = 0.05f;
const float EST_VEL_FUSION = 0.05f;
const float EST_VEL_DECAY = 0.90f;
uint32_t last_vel_est_us = 0;

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

inline float clamp(float val, float min_val, float max_val) {
    return fmaxf(min_val, fminf(max_val, val));
}

void updateHistory(float* new_frame) {
    // Shift and insert logic (Simplified for HISTORY_STEPS=1 which is common)
    if (HISTORY_STEPS > 1) {
        memmove(&history_buffer[OBS_FRAME_DIM], &history_buffer[0], (HISTORY_STEPS - 1) * OBS_FRAME_DIM * sizeof(float));
    }
    memcpy(&history_buffer[0], new_frame, OBS_FRAME_DIM * sizeof(float));
}

void computeProjectedGravity() {
    // Compute R^T * [0, 0, -1] to get gravity in body frame
    // Must match IsaacLab's mdp.projected_gravity = quat_rotate_inverse(root_quat_w, [0,0,-1])
    float w = imu.quat_w;
    float x = imu.quat_x;
    float y = imu.quat_y;
    float z = imu.quat_z;
    
    est_projected_gravity[0] = 2.0f * (w * y - x * z);
    est_projected_gravity[1] = -2.0f * (w * x + y * z);
    est_projected_gravity[2] = -(w * w - x * x - y * y + z * z);
}

void updateVelocityEstimate(float dt) {
    if (dt <= 0.0f || dt > 0.1f) return;

    // 1. Wheel Odometry
    // Joints 4 and 5 are wheels (left, right)
    float avg_wheel_vel = (joints[4]->current_velocity + joints[5]->current_velocity) / 2.0f;
    float odom_vel_x = avg_wheel_vel * EST_WHEEL_RADIUS;
    
    // 2. Static Switch
    if (fabsf(avg_wheel_vel) < 0.1f) {
        est_base_lin_vel[0] = 0.0f;
        est_base_lin_vel[1] = 0.0f;
        est_base_lin_vel[2] = 0.0f;
        return;
    }
    
    // 3. IMU Integration
    float yaw_rate = imu.gyro_z;
    float cos_y = cosf(yaw_rate * dt);
    float sin_y = sinf(yaw_rate * dt);
    
    float vx = est_base_lin_vel[0];
    float vy = est_base_lin_vel[1];
    
    // Rotate
    est_base_lin_vel[0] = vx * cos_y + vy * sin_y;
    est_base_lin_vel[1] = -vx * sin_y + vy * cos_y;
    
    // Integrate Accel (IMU accel is gravity compensated if using LinearAccel feature, check BNO085_IMU)
    // Assuming imu.accel is linear acceleration
    est_base_lin_vel[0] += imu.accel_x * dt;
    est_base_lin_vel[1] += imu.accel_y * dt;
    est_base_lin_vel[2] = 0.0f;
    
    // 4. Fusion
    est_base_lin_vel[0] = (1.0f - EST_VEL_FUSION) * est_base_lin_vel[0] + EST_VEL_FUSION * odom_vel_x;
    
    // 5. Decay
    est_base_lin_vel[1] *= EST_VEL_DECAY;
}

void buildSingleObservation(float* obs) {
    int idx = 0;
    
    // OBSERVATION ORDER MUST MATCH TRAINING (bebop_base_cfg.py ObservationsCfg):
    // [0:3]   base_lin_vel
    // [3:6]   base_ang_vel
    // [6:9]   projected_gravity
    // [9:15]  joint_pos_rel
    // [15:21] joint_vel_rel
    // [21:27] last_action
    // [27:30] velocity_commands
    
    // [0:3] Base Linear Velocity
    obs[idx++] = clamp(est_base_lin_vel[0], -CLIP_LIN_VEL, CLIP_LIN_VEL) * SCALE_LIN_VEL;
    obs[idx++] = clamp(est_base_lin_vel[1], -CLIP_LIN_VEL, CLIP_LIN_VEL) * SCALE_LIN_VEL;
    obs[idx++] = clamp(est_base_lin_vel[2], -CLIP_LIN_VEL, CLIP_LIN_VEL) * SCALE_LIN_VEL;
    
    // [3:6] Base Angular Velocity (IMU)
    obs[idx++] = clamp(imu.gyro_x, -CLIP_ANG_VEL, CLIP_ANG_VEL) * SCALE_ANG_VEL;
    obs[idx++] = clamp(imu.gyro_y, -CLIP_ANG_VEL, CLIP_ANG_VEL) * SCALE_ANG_VEL;
    obs[idx++] = clamp(imu.gyro_z, -CLIP_ANG_VEL, CLIP_ANG_VEL) * SCALE_ANG_VEL;
    
    // [6:9] Projected Gravity
    obs[idx++] = est_projected_gravity[0];
    obs[idx++] = est_projected_gravity[1];
    obs[idx++] = est_projected_gravity[2];
    
    // [9:15] Joint Positions (relative to default)
    for(int i=0; i<4; i++) { // Legs
        obs[idx++] = (joints[i]->current_position - default_joint_pos[i]) * SCALE_DOF_POS;
    }
    obs[idx++] = 0.0f; // Left Wheel (reset - wheels accumulate in real life)
    obs[idx++] = 0.0f; // Right Wheel (reset)
    
    // [15:21] Joint Velocities
    for(int i=0; i<6; i++) {
        obs[idx++] = clamp(joints[i]->current_velocity, -CLIP_DOF_VEL, CLIP_DOF_VEL) * SCALE_DOF_VEL;
    }
    
    // [21:27] Last Action
    for(int i=0; i<6; i++) {
        obs[idx++] = last_action[i];
    }
    
    // [27:30] Velocity Commands (MUST be last to match training!)
    float cmd_vx, cmd_vy, cmd_wz;
    rosPublishers.getCmdVel(cmd_vx, cmd_vy, cmd_wz);
    
    obs[idx++] = cmd_vx * SCALE_LIN_VEL;
    obs[idx++] = cmd_vy * SCALE_LIN_VEL;
    obs[idx++] = cmd_wz * SCALE_ANG_VEL;
}

// ============================================================================
// CAN RECEIVE CALLBACK
// ============================================================================

void canCallback(const CAN_message_t& msg) {
    can_rx_count++;
    if (msg.flags.extended) {
        can_rx_ext_count++;
    } else {
        can_rx_std_count++;
    }
    
    if (can_rx_debug) {
        if (msg.flags.extended) {
            // Robstride extended frame
            uint8_t cmd_type = (msg.id >> 24) & 0x1F;
            uint8_t motor_id = (msg.id >> 8) & 0xFF;
            DEBUG_PORT.printf("RS RX [%08lX] cmd=%02X id=%d len=%d:", 
                              msg.id, cmd_type, motor_id, msg.len);
        } else {
            // ODrive standard frame
            uint8_t node_id = (msg.id >> 5) & 0x3F;
            uint8_t cmd_id = msg.id & 0x1F;
            DEBUG_PORT.printf("OD RX [%03lX] node=%d cmd=%02X len=%d:", 
                              msg.id, node_id, cmd_id, msg.len);
        }
        for (int i = 0; i < msg.len; i++) {
            DEBUG_PORT.printf(" %02X", msg.buf[i]);
        }
        DEBUG_PORT.println();
    }
    
    for (int i = 0; i < NUM_JOINTS; i++) {
        joints[i]->handleCanMessage(msg);
    }
}

// ============================================================================
// WATCHDOG
// ============================================================================

uint32_t last_watchdog_cmd_time = 0;
const uint32_t WATCHDOG_CMD_INTERVAL_MS = 1000;  // Rate-limit watchdog commands to 1Hz

void checkWatchdog() {
    uint32_t now = millis();
    if ((now - rosPublishers.getLastCmdTime()) > WATCHDOG_TIMEOUT_MS) {
        // Rate-limit watchdog commands to avoid flooding CAN bus
        if ((now - last_watchdog_cmd_time) >= WATCHDOG_CMD_INTERVAL_MS) {
            // ODrive wheels: Stop them (they're velocity controlled, so need explicit zero)
            for (int i = ODRIVE_START_IDX; i <= ODRIVE_END_IDX; i++) {
                joints[i]->setCommand(0.0f, 0.0f, 0.0f);
            }
            last_watchdog_cmd_time = now;
        }
    }
}


// ============================================================================
// STATUS REPORTING
// ============================================================================

void printMotorStatus() {
    DEBUG_PORT.println("=== MOTOR STATUS ===");
    DEBUG_PORT.printf("CAN RX (2s): total=%lu (std=%lu ext=%lu)\n", 
                      can_rx_count, can_rx_std_count, can_rx_ext_count);
    
    // Reset CAN counters
    can_rx_count = 0;
    can_rx_std_count = 0;
    can_rx_ext_count = 0;
    
    // Robstride motors
    for (int i = ROBSTRIDE_START_IDX; i <= ROBSTRIDE_END_IDX; i++) {
        RobstrideMotorType* motor = static_cast<RobstrideMotorType*>(joints[i]);
        uint32_t age_ms = millis() - motor->last_feedback_time;
        bool receiving = (age_ms < 500);
        
        DEBUG_PORT.printf("[%d] %s (RS id=%lu): %s | en=%d err=%d (0x%02X) | pos=%.2f vel=%.2f | fb_age=%lums\n",
            i, JOINT_NAMES[i], motor->can_id,
            receiving ? "OK" : "NO CAN",
            motor->is_enabled ? 1 : 0,
            motor->has_error ? 1 : 0,
            motor->getFaultBits(),
            motor->current_position,
            motor->current_velocity,
            age_ms);
    }
    
    // ODrive motors
    for (int i = ODRIVE_START_IDX; i <= ODRIVE_END_IDX; i++) {
        ODriveMotorType* motor = static_cast<ODriveMotorType*>(joints[i]);
        uint32_t age_ms = millis() - motor->last_feedback_time;
        bool receiving = (age_ms < 500);
        
        DEBUG_PORT.printf("[%d] %s (OD node=%lu): %s | state=%d err=0x%08lX | pos=%.2f vel=%.2f | fb_age=%lums\n",
            i, JOINT_NAMES[i], motor->can_id,
            receiving ? "OK" : "NO CAN",
            motor->getAxisState(),
            motor->getAxisError(),
            motor->current_position,
            motor->current_velocity,
            age_ms);
    }
    
    DEBUG_PORT.println();
}

// ============================================================================
// SETUP
// ============================================================================

void setup() {
    // Initialize LED
    pinMode(LED_PIN, OUTPUT);
    digitalWrite(LED_PIN, HIGH);
    
    // Initialize debug serial
    serialCommands.begin(115200);
    delay(500);
    DEBUG_PORT.println("\n=== Teensy CAN Bridge Starting ===");
    DEBUG_PORT.println("Firmware: Bebop Robot micro-ROS Bridge");
    DEBUG_PORT.printf("Joints: %d (Robstride: %d-%d, ODrive: %d-%d)\n", 
                      NUM_JOINTS, ROBSTRIDE_START_IDX, ROBSTRIDE_END_IDX, 
                      ODRIVE_START_IDX, ODRIVE_END_IDX);
    
    // Initialize motors and CAN bus
    DEBUG_PORT.println("Initializing CAN bus...");
    setupMotors();
    DEBUG_PORT.printf("CAN bus initialized: %lu baud, CAN1 (pins TX=22/RX=23)\n", (unsigned long)CAN_BAUD_RATE);

    // Initialize IMU
    DEBUG_PORT.print("Initializing BNO085 IMU... ");
    bool imu_ok = false;
    for (int i = 0; i < 5; i++) {
        if (imu.begin()) {
            imu_ok = true;
            break;
        }
        DEBUG_PORT.print(".");
        delay(200);
    }
    DEBUG_PORT.println(imu_ok ? "OK" : "FAIL");
    
    // Set up CAN callback
    canBus.onReceive(canCallback);

    // Initialize micro-ROS
    if (!rosPublishers.begin(&imu)) {
        // Error loop is handled inside begin()
        while (1) {
            digitalWrite(LED_PIN, !digitalRead(LED_PIN));
            delay(100);
        }
    }

    // Enable all motors
    DEBUG_PORT.println("Enabling motors...");
    for (int i = 0; i < NUM_JOINTS; i++) {
        DEBUG_PORT.printf("  Enabling joint %d: %s\n", i, JOINT_NAMES[i]);
        joints[i]->enable();
        delay(10);
    }
    
    // Enable active reporting on Robstride motors
    DEBUG_PORT.println("Configuring Robstride active reporting...");
    for (int i = ROBSTRIDE_START_IDX; i <= ROBSTRIDE_END_IDX; i++) {
        RobstrideMotorType* rs_motor = static_cast<RobstrideMotorType*>(joints[i]);
        rs_motor->enableActiveReporting(true, 20);  // 20ms interval
    }
    
    policy.init();
    DEBUG_PORT.println("POLICY INITIALIZED (Waiting for activation)");
    last_policy_us = micros();
    last_vel_est_us = micros();
    
    // Initialize timing counters to prevent burst on startup
    uint32_t now = millis();
    last_publish_time = now;
    last_imu_publish_time = now;
    last_diag_publish_time = now;
    last_timing_report_ms = now;
    
    digitalWrite(LED_PIN, LOW);
    DEBUG_PORT.println("=== Teensy CAN Bridge Ready ===\n");
}

// ============================================================================
// MAIN LOOP
// ============================================================================

void loop() {
    uint32_t now = millis();
    uint32_t now_us = micros();
    
    // Process serial debug commands
    serialCommands.update();
    
    // Process CAN messages
    canBus.events();
    
    // Spin micro-ROS executor (non-blocking)
    // Increased timeout to ensure we clear the buffer and handle messages
    rosPublishers.spin(100);  // 1ms timeout
    
    // Update IMU (200 Hz)
    if ((now - last_imu_update_time) >= imu_update_interval_ms) {
        if (imu.initialized) {
            // Check for stale data and attempt hardware reset recovery
            // This uses pin 17 (RST) to reset the BNO085 if data is >2s old
            imu.checkAndRecover(2000);
            
            // Drain IMU queue (handle multiple events if loop is slow)
            int count = 0;
            while (imu.update() && count++ < 10);

            // Always run estimation if policy might be used
            computeProjectedGravity();
            float dt = (now_us - last_vel_est_us) * 1e-6f;
            updateVelocityEstimate(dt);
            last_vel_est_us = now_us;
            
            // IMU debug logging
            if (imu_debug) {
                DEBUG_PORT.printf("IMU: q=[%.3f,%.3f,%.3f,%.3f] gyro=[%.2f,%.2f,%.2f] accel=[%.2f,%.2f,%.2f] grav=[%.2f,%.2f,%.2f]\n",
                    imu.quat_w, imu.quat_x, imu.quat_y, imu.quat_z,
                    imu.gyro_x, imu.gyro_y, imu.gyro_z,
                    imu.accel_x, imu.accel_y, imu.accel_z,
                    est_projected_gravity[0], est_projected_gravity[1], est_projected_gravity[2]);
            }
        }
        last_imu_update_time = now;
    }

    // POLICY CONTROL LOOP
    // Only run if enabled in RosPublishers
    if (rosPublishers.getControlMode() == RosPublishers::MODE_POLICY_BALANCE && (now_us - last_policy_us >= POLICY_INTERVAL_US)) {
        last_policy_us = now_us;
        step_count++;
        
        // Safety Check - only depends on IMU orientation
        bool upright = (est_projected_gravity[2] < -0.5f); // Pointing down (-1)
        
        // ALWAYS run inference if IMU is ready (for debugging/logging)
        if (imu.initialized) {
            // 1. Build Observation
            buildSingleObservation(current_obs);
            updateHistory(current_obs);
            
            // 2. Inference
            policy.infer(history_buffer, action_buffer);
            
            // 3. Store Last Action & Clip
            for(int i=0; i<bebop_policy::ACTION_DIM; i++) {
                last_action[i] = clamp(action_buffer[i], -1.0f, 1.0f);
            }
            
            // 4. Compute scaled commands (for logging)
            float leg_cmds[4];
            float wheel_cmds[2];
            for(int i=0; i<4; i++) {
                float scaled_action = action_buffer[i] * SCALE_ACTION_LEGS;
                leg_cmds[i] = clamp(default_joint_pos[i] + scaled_action, -MAX_LEG_POS_RAD, MAX_LEG_POS_RAD);
            }
            for(int i=0; i<2; i++) {
                wheel_cmds[i] = clamp(action_buffer[4+i] * SCALE_ACTION_WHEELS, -MAX_WHEEL_VEL_RAD_S, MAX_WHEEL_VEL_RAD_S);
            }
            
            // 5. Log actions periodically (every 50 steps = 1 second at 50Hz)
            if (step_count % 50 == 0) {
                DEBUG_PORT.printf("POLICY[%lu]: legs=[%.2f,%.2f,%.2f,%.2f] wheels=[%.1f,%.1f] | UP=%d\n",
                    step_count,
                    leg_cmds[0], leg_cmds[1], leg_cmds[2], leg_cmds[3],
                    wheel_cmds[0], wheel_cmds[1],
                    upright ? 1 : 0);

                // LOG INPUT OBSERVATIONS (First 15 dims)
                DEBUG_PORT.printf("  OBS[0-2] LinVel: [%.3f, %.3f, %.3f]\n", current_obs[0], current_obs[1], current_obs[2]);
                DEBUG_PORT.printf("  OBS[3-5] AngVel: [%.3f, %.3f, %.3f]\n", current_obs[3], current_obs[4], current_obs[5]);
                DEBUG_PORT.printf("  OBS[6-8] Grav:   [%.3f, %.3f, %.3f]\n", current_obs[6], current_obs[7], current_obs[8]);
                DEBUG_PORT.printf("  OBS[9-14] JPos:  [%.2f, %.2f, %.2f, %.2f, %.2f, %.2f]\n", 
                    current_obs[9], current_obs[10], current_obs[11], current_obs[12], current_obs[13], current_obs[14]);
            }
            
            // 6. Only send commands if upright (IMU check)
            if (upright) {
                // Legs (Position)
                for(int i=0; i<4; i++) {
                    joints[i]->setCommand(leg_cmds[i], 0.0f, 0.0f);
                }
                // Wheels (Velocity)
                for(int i=0; i<2; i++) {
                    joints[4+i]->setCommand(0.0f, wheel_cmds[i], 0.0f);
                }
            } else {
                // Safety stop - robot is not upright, zero wheels
                joints[4]->setCommand(0.0f, 0.0f, 0.0f);
                joints[5]->setCommand(0.0f, 0.0f, 0.0f);
            }
        }
    }
    
    // Publish joint state feedback (100 Hz)
    if ((now - last_publish_time) >= publish_interval_ms) {
        rosPublishers.publishState();
        last_publish_time += publish_interval_ms; // Prevent drift
        
        // Blink LED to show activity
        led_state = !led_state;
        digitalWrite(LED_PIN, led_state);
    }
    
    
    // Publish IMU data (100 Hz)
    if ((now - last_imu_publish_time) >= imu_publish_interval_ms) {
        rosPublishers.publishIMU();
        last_imu_publish_time += imu_publish_interval_ms; // Prevent drift
    }
    
    // Publish diagnostics (10 Hz)
    if ((now - last_diag_publish_time) >= diag_publish_interval_ms) {
        rosPublishers.publishDiagnostics();
        last_diag_publish_time += diag_publish_interval_ms; // Prevent drift
    }

    // Check watchdog
    checkWatchdog();
    
    // Print motor status every 2 seconds (if enabled)
    if (status_logging && (now - last_timing_report_ms) >= 2000) {
        // Show current mode in status
        if (rosPublishers.getControlMode() == RosPublishers::MODE_POLICY_BALANCE) {
            DEBUG_PORT.println("[MODE: POLICY_BALANCE]");
        } else {
            DEBUG_PORT.println("[MODE: PASSTHROUGH]");
        }
        
        // IMU diagnostics
        imu.printDiagnostics();
        if (imu.isTimedOut(500)) {
            DEBUG_PORT.println("  WARNING: IMU data is STALE!");
        }
        DEBUG_PORT.printf("Proj Gravity: [%.3f, %.3f, %.3f] (upright=-1.0)\n",
            est_projected_gravity[0], est_projected_gravity[1], est_projected_gravity[2]);
        
        printMotorStatus();
        last_timing_report_ms = now;
    }
}
