/**        
 * @file main_sim.cpp        
 * @brief Teensy 4.1 Controller with Observation History (Latency Compensation)      
 */        
        
 #include <Arduino.h>        
 #include "MicroROS.h"         
 #include "BebopPolicy.h"         
           
 // ============================================================================        
 // CONFIGURATION        
 // ============================================================================        
           
// OBSERVATION CONFIGURATION
// MUST match training! Set HISTORY_STEPS=1 for 30-obs model, 3 for 90-obs model
#define HISTORY_STEPS 1       // 1 = no history (30 obs), 3 = history (90 obs)
#define OBS_FRAME_DIM 30 
#define TOTAL_OBS_DIM (OBS_FRAME_DIM * HISTORY_STEPS) // 30 or 90 floats depending on HISTORY_STEPS

 // PORTS        
 #define UROS_PORT Serial          
 #define DEBUG_PORT SerialUSB1     
           
 // TIMING    
 #define PHYSICS_RATE_HZ 200       
 #define CONTROL_RATE_HZ 100       
 #define DECIMATION (PHYSICS_RATE_HZ / CONTROL_RATE_HZ)      
 #define POLICY_INTERVAL_US (1000000 / CONTROL_RATE_HZ)        
           
 #define DATA_TIMEOUT_MS 500         
 #define TELEMETRY_RATE_HZ 10        
 #define TELEMETRY_INTERVAL_MS (1000 / TELEMETRY_RATE_HZ)        
           
 #define WHEEL_RADIUS 0.05f         
 #define LED_PIN 13        
   
 // ============================================================================  
 // POLICY CONFIGURATION
 // ============================================================================  
   
 // Using constants from generated header for safety
 const float CLIP_LIN_VEL = bebop_policy::CLIP_LIN_VEL;
 const float CLIP_ANG_VEL = bebop_policy::CLIP_ANG_VEL;
 const float CLIP_DOF_VEL = bebop_policy::CLIP_DOF_VEL;
   
 const float SCALE_LIN_VEL = bebop_policy::SCALE_LIN_VEL;
 const float SCALE_ANG_VEL = bebop_policy::SCALE_ANG_VEL;
 const float SCALE_DOF_POS = bebop_policy::SCALE_DOF_POS;
 const float SCALE_DOF_VEL = bebop_policy::SCALE_DOF_VEL;
   
 const float SCALE_ACTION_LEGS = bebop_policy::SCALE_ACTION_LEGS;
 const float SCALE_ACTION_WHEELS = bebop_policy::SCALE_ACTION_WHEELS;
   
const float MAX_LEG_POS_RAD = 1.5f;  
const float MAX_WHEEL_VEL_RAD_S = 20.0f;  
  
// JOINT ORDERING NOTE:
// MicroROS.cpp now handles name-based reordering in jointStateCallback.
// robot_state_.joint_positions/velocities are already in TRAINING order:
//   [0] left_hip_pitch, [1] right_hip_pitch, [2] left_knee_pitch,
//   [3] right_knee_pitch, [4] left_wheel, [5] right_wheel
// NO additional mapping is needed here!
   
 // ============================================================================        
 // GLOBALS        
 // ============================================================================        
           
 MicroROSManager ros;        
 bebop_policy::Policy policy;        
           
 // Data Buffers        
 float current_obs[OBS_FRAME_DIM];       // Snapshot of NOW (30 floats)
 float history_buffer[TOTAL_OBS_DIM];    // Stack of [t, t-1, t-2] (90 floats)
 float action_buffer[bebop_policy::ACTION_DIM];        
 float last_action[bebop_policy::ACTION_DIM] = {0};        
        
 // Command History      
 JointCommand last_cmd = {0};      
           
 // State Tracking    
 uint32_t step_count = 0;    
 uint32_t last_policy_us = 0;        
 uint32_t last_telemetry_ms = 0;        
 uint32_t last_valid_data_ms = 0;         
 bool safety_stop = false;        
 bool is_reset = true;         
 double prev_sim_time = -1.0;    
           
 float default_joint_pos[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};        
      
 // ============================================================================        
 // HELPER FUNCTIONS        
 // ============================================================================        
           
 inline float clamp(float val, float min_val, float max_val) {        
     return fmaxf(min_val, fminf(max_val, val));        
 }        
      
 void resetControllerState(const char* reason) {        
     DEBUG_PORT.print("\n>>> RESET DETECTED: ");    
     DEBUG_PORT.print(reason);    
     DEBUG_PORT.println(" <<<");    
             
     step_count = 0;    
     memset(last_action, 0, sizeof(last_action));        
     memset(action_buffer, 0, sizeof(action_buffer));        
     
     // Clear history on reset
     memset(history_buffer, 0, sizeof(history_buffer));
     memset(current_obs, 0, sizeof(current_obs));
     
     memset(&last_cmd, 0, sizeof(last_cmd));      
     prev_sim_time = -1.0;    
     safety_stop = false;        
     is_reset = true;        
     last_policy_us = micros();        
 } 

 // UPDATE HISTORY STACK
 // Moves [t, t-1] -> [t-1, t-2] and inserts New -> [t]
 void updateHistory(float* new_frame) {
     // 1. Shift existing history to the right
     // We move (HISTORY_STEPS - 1) frames. 
     // Destination: history_buffer + 30
     // Source: history_buffer
     // Size: 30 * 2 * 4 bytes
     memmove(
         &history_buffer[OBS_FRAME_DIM], 
         &history_buffer[0], 
         (HISTORY_STEPS - 1) * OBS_FRAME_DIM * sizeof(float)
     );

     // 2. Copy new frame to the front (Index 0)
     memcpy(&history_buffer[0], new_frame, OBS_FRAME_DIM * sizeof(float));
 }
           
 // BUILD OBSERVATION (Single Frame)
 void buildSingleObservation(const RobotState& state, float* obs) {        
     int idx = 0;        
         
     // 1. Base Linear Velocity  
     float v_x = clamp(state.base_lin_vel[0], -CLIP_LIN_VEL, CLIP_LIN_VEL);      
     float v_y = clamp(state.base_lin_vel[1], -CLIP_LIN_VEL, CLIP_LIN_VEL);      
     float v_z = clamp(state.base_lin_vel[2], -CLIP_LIN_VEL, CLIP_LIN_VEL);      
           
     // 2. Base Angular Velocity
     float w_x = clamp(state.base_ang_vel[0], -CLIP_ANG_VEL, CLIP_ANG_VEL);      
     float w_y = clamp(state.base_ang_vel[1], -CLIP_ANG_VEL, CLIP_ANG_VEL);      
     float w_z = clamp(state.base_ang_vel[2], -CLIP_ANG_VEL, CLIP_ANG_VEL);      
         
     obs[idx++] = v_x * SCALE_LIN_VEL;        
     obs[idx++] = v_y * SCALE_LIN_VEL;        
     obs[idx++] = v_z * SCALE_LIN_VEL;        
         
     obs[idx++] = w_x * SCALE_ANG_VEL;        
     obs[idx++] = w_y * SCALE_ANG_VEL;        
     obs[idx++] = w_z * SCALE_ANG_VEL;        
         
    // 3. Projected Gravity (pass through directly - matches training!)
    obs[idx++] = state.projected_gravity[0];        
    obs[idx++] = state.projected_gravity[1];        
    obs[idx++] = state.projected_gravity[2];
            
         
    // 4. Joint Positions (already in training order from MicroROS)
    // CRITICAL: Wheel positions accumulate during deployment but reset in training!
    // Set wheel positions to 0 since they're velocity-controlled
    for(int i=0; i<4; i++) {  // Legs only (indices 0-3)
        obs[idx++] = (state.joint_positions[i] - default_joint_pos[i]) * SCALE_DOF_POS;        
    }
    // Wheels (indices 4-5): Set to 0, not actual accumulated position
    obs[idx++] = 0.0f;  // left_wheel
    obs[idx++] = 0.0f;  // right_wheel        
        
    // 5. Joint Velocities (already in training order from MicroROS)
    for(int i=0; i<6; i++) {      
        float jv = clamp(state.joint_velocities[i], -CLIP_DOF_VEL, CLIP_DOF_VEL);      
        obs[idx++] = jv * SCALE_DOF_VEL;        
    }
         
     // 6. Last Action  
     for(int i=0; i<6; i++) {        
         obs[idx++] = last_action[i];        
     }        
         
     // 7. Velocity Commands (STEERING!)
     // TODO: Connect this to RC Receiver or Serial Input
     obs[idx++] = 0.0f; // Target Fwd Vel (m/s)        
     obs[idx++] = 0.0f; // Target Lat Vel (m/s)        
     obs[idx++] = 0.0f; // Target Turn Rate (rad/s)        
 }      
           
 // ============================================================================        
 // SETUP        
 // ============================================================================        
 void setup() {        
     pinMode(LED_PIN, OUTPUT);        
     digitalWrite(LED_PIN, HIGH);        
             
     DEBUG_PORT.begin(115200);        
     ros.init();         
     policy.init();        
             
     DEBUG_PORT.println("\n--- Bebop Teensy Controller (History=3) ---");        
     DEBUG_PORT.printf("Obs Dim: %d (Frame) -> %d (Total)\n", OBS_FRAME_DIM, TOTAL_OBS_DIM);
     DEBUG_PORT.println("Waiting for ROS Agent...");        
 }        
           
 // ============================================================================        
 // MAIN LOOP     
 // ============================================================================        
 void loop() {        
     uint32_t now_ms = millis();        
     uint32_t now_us = micros();        
             
     // 1. Update MicroROS        
     MicroROSManager::State ros_status = ros.update();        
             
     // 2. CHECK CONNECTION        
     if (ros_status != MicroROSManager::AGENT_CONNECTED) {        
         if (!is_reset) resetControllerState("Agent Disconnected");         
         digitalWrite(LED_PIN, (now_ms / 100) % 2);     
         return;         
     }        
         
     // 3. CHECK DATA FRESHNESS     
     if (ros.hasValidSensorData()) {        
         last_valid_data_ms = now_ms;        
     }        
         
     if (now_ms - last_valid_data_ms > DATA_TIMEOUT_MS) {        
         if (!is_reset) resetControllerState("Data Timeout");         
         digitalWrite(LED_PIN, (now_ms / 500) % 2);     
         return;         
     }        
         
     digitalWrite(LED_PIN, HIGH);         
         
     if (now_us - last_policy_us >= POLICY_INTERVAL_US) {        
         last_policy_us = now_us;        
                 
         const RobotState& state = ros.getRobotState();        
     
         // SIM TIME RESET DETECTION  
         if (state.sim_time < (prev_sim_time - 0.1) && prev_sim_time > 0.0) {    
             resetControllerState("Sim Time Reset");    
         }    
         prev_sim_time = state.sim_time;    
     
         if (is_reset) {    
             if (state.sim_time > 0.1) {    
                 is_reset = false;    
                 step_count = 0;    
             } else {    
                 return;    
             }    
         }    
     
         step_count++;    
                 
         if (step_count % DECIMATION != 0) return;    
     
         // INITIAL STABILIZATION    
         if (step_count < 15) {    
             JointCommand zero_cmd = {0};    
             ros.publishJointCommand(zero_cmd);    
             last_cmd = zero_cmd;    
             return;    
         } else if (step_count == 15) {    
             DEBUG_PORT.println(">>> ENGAGING CONTROL <<<");    
         }    
         
         // SAFETY CHECK  
         if (state.projected_gravity[2] > -0.5f) {         
             safety_stop = false;        
         } else {        
             if (state.projected_gravity[2] < -0.9f) safety_stop = false;        
         }        
         
         if(safety_stop) {        
             JointCommand zero_cmd = {0};         
             ros.publishJointCommand(zero_cmd);      
             last_cmd = zero_cmd;    
             return;        
         }        
         
         // 1. Build Current Frame (30 floats)
         buildSingleObservation(state, current_obs);
         
         // 2. Update History Stack (Shift & Insert)
         updateHistory(current_obs);

        // 3. Inference (Pass the full 90-float buffer)
        policy.infer(history_buffer, action_buffer);        
         
        // CRITICAL: Clip last_action to prevent feedback explosion!
        // Training expects raw NN output in ~[-1, 1] range.
        for(int i = 0; i < bebop_policy::ACTION_DIM; i++) {
            last_action[i] = clamp(action_buffer[i], -1.0f, 1.0f);
        }
             
         // COMMAND GENERATION  
         JointCommand cmd;        
           
         // Legs (Indices 0-3 in Policy)  
         for(int i=0; i<4; i++) {        
             float scaled_action = action_buffer[i] * SCALE_ACTION_LEGS;    
             float pos = default_joint_pos[i] + scaled_action;    
             cmd.leg_positions[i] = clamp(pos, -MAX_LEG_POS_RAD, MAX_LEG_POS_RAD);    
         }        
                 
      
                 
        // Wheels (Indices 4-5 in Policy)  
        for(int i=0; i<2; i++) {        
            float scaled_action = action_buffer[4+i] * SCALE_ACTION_WHEELS;    
            float vel = scaled_action;    
            
            // Do not negate: Policy matches sim expectation (Positive Action = Forward)
            cmd.wheel_velocities[i] = clamp(vel, -MAX_WHEEL_VEL_RAD_S, MAX_WHEEL_VEL_RAD_S);    
        }
                 
                 
         ros.publishJointCommand(cmd);        
         last_cmd = cmd;    
     }        
             
    // 6. TELEMETRY (10Hz)    
    if (now_ms - last_telemetry_ms > TELEMETRY_INTERVAL_MS) {        
        last_telemetry_ms = now_ms;        
        const RobotState& s = ros.getRobotState();        
    
        // Build LIVE observation for telemetry (even if policy didn't run)
        float live_obs[OBS_FRAME_DIM];
        buildSingleObservation(s, live_obs);
        
        DEBUG_PORT.println();
        DEBUG_PORT.println("######################################################################");
        DEBUG_PORT.printf("# STEP %lu | T:%.3fs", step_count, s.sim_time);
        
        // Show controller state
        if (step_count < 15) {
            DEBUG_PORT.print(" | STATE: STABILIZING");
        } else if (safety_stop) {
            DEBUG_PORT.print(" | STATE: ⚠ SAFETY_STOP");
        } else {
            DEBUG_PORT.print(" | STATE: ACTIVE");
        }
        DEBUG_PORT.println();
        DEBUG_PORT.println("######################################################################");
        
        // === RAW SENSOR VALUES ===
        DEBUG_PORT.println("\n=== RAW SENSOR VALUES ===");
        DEBUG_PORT.printf("Joint Pos (HW order): [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f]\n",
            s.joint_positions[0], s.joint_positions[1], s.joint_positions[2],
            s.joint_positions[3], s.joint_positions[4], s.joint_positions[5]);
        DEBUG_PORT.printf("Joint Vel (HW order): [%.2f, %.2f, %.2f, %.2f, %.2f, %.2f]\n",
            s.joint_velocities[0], s.joint_velocities[1], s.joint_velocities[2],
            s.joint_velocities[3], s.joint_velocities[4], s.joint_velocities[5]);
        DEBUG_PORT.printf("Proj Gravity:         [%.3f, %.3f, %.3f]\n",
            s.projected_gravity[0], s.projected_gravity[1], s.projected_gravity[2]);
        DEBUG_PORT.printf("Base Lin Vel:         [%.3f, %.3f, %.3f] m/s\n",
            s.base_lin_vel[0], s.base_lin_vel[1], s.base_lin_vel[2]);
        DEBUG_PORT.printf("Base Ang Vel:         [%.2f, %.2f, %.2f] rad/s\n",
            s.base_ang_vel[0], s.base_ang_vel[1], s.base_ang_vel[2]);
        
        // === CLIPPING WARNINGS ===
        if (fabs(s.base_lin_vel[0]) > CLIP_LIN_VEL || fabs(s.base_lin_vel[1]) > CLIP_LIN_VEL || fabs(s.base_lin_vel[2]) > CLIP_LIN_VEL) {
            DEBUG_PORT.printf("  ⚠ LIN_VEL CLIPPED! (limit: %.1f)\n", CLIP_LIN_VEL);
        }
        if (fabs(s.base_ang_vel[0]) > CLIP_ANG_VEL || fabs(s.base_ang_vel[1]) > CLIP_ANG_VEL || fabs(s.base_ang_vel[2]) > CLIP_ANG_VEL) {
            DEBUG_PORT.printf("  ⚠ ANG_VEL CLIPPED! Raw max: %.1f (limit: %.1f)\n", 
                fmaxf(fmaxf(fabs(s.base_ang_vel[0]), fabs(s.base_ang_vel[1])), fabs(s.base_ang_vel[2])), CLIP_ANG_VEL);
        }
        for (int i = 0; i < 6; i++) {
            if (fabs(s.joint_velocities[i]) > CLIP_DOF_VEL) {
                DEBUG_PORT.printf("  ⚠ JOINT_VEL[%d] CLIPPED! %.1f (limit: %.1f)\n", i, s.joint_velocities[i], CLIP_DOF_VEL);
            }
        }
        
        // === LIVE OBSERVATION (what policy WOULD see) ===
        DEBUG_PORT.println("\n=== LIVE OBSERVATION (30 dims) ===");
        DEBUG_PORT.printf("  [0:3]   base_lin_vel:  [%.3f, %.3f, %.3f]\n", 
            live_obs[0], live_obs[1], live_obs[2]);
        DEBUG_PORT.printf("  [3:6]   base_ang_vel:  [%.3f, %.3f, %.3f]\n", 
            live_obs[3], live_obs[4], live_obs[5]);
        DEBUG_PORT.printf("  [6:9]   proj_gravity:  [%.3f, %.3f, %.3f]\n", 
            live_obs[6], live_obs[7], live_obs[8]);
        DEBUG_PORT.printf("  [9:15]  joint_pos:     [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f]\n",
            live_obs[9], live_obs[10], live_obs[11], live_obs[12], live_obs[13], live_obs[14]);
        DEBUG_PORT.printf("  [15:21] joint_vel:     [%.2f, %.2f, %.2f, %.2f, %.2f, %.2f]\n",
            live_obs[15], live_obs[16], live_obs[17], live_obs[18], live_obs[19], live_obs[20]);
        DEBUG_PORT.printf("  [21:27] last_action:   [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f]\n",
            live_obs[21], live_obs[22], live_obs[23], live_obs[24], live_obs[25], live_obs[26]);
        DEBUG_PORT.printf("  [27:30] cmd_vel:       [%.2f, %.2f, %.2f]\n",
            live_obs[27], live_obs[28], live_obs[29]);
        
        // === POLICY OUTPUT ===
        DEBUG_PORT.println("\n=== POLICY I/O ===");
        if (safety_stop || step_count < 15) {
            DEBUG_PORT.println("  (Policy NOT running - showing last cached values)");
        }
        DEBUG_PORT.printf("Raw Actions:          [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f]\n",
            action_buffer[0], action_buffer[1], action_buffer[2], 
            action_buffer[3], action_buffer[4], action_buffer[5]);
        DEBUG_PORT.printf("Scaled (Legs*%.1f, Whl*%.1f): [%.3f, %.3f, %.3f, %.3f | %.2f, %.2f]\n",
            SCALE_ACTION_LEGS, SCALE_ACTION_WHEELS,
            action_buffer[0] * SCALE_ACTION_LEGS, action_buffer[1] * SCALE_ACTION_LEGS,
            action_buffer[2] * SCALE_ACTION_LEGS, action_buffer[3] * SCALE_ACTION_LEGS,
            action_buffer[4] * SCALE_ACTION_WHEELS, action_buffer[5] * SCALE_ACTION_WHEELS);
        
        // === COMMANDS SENT ===
        DEBUG_PORT.println("\n=== COMMANDS SENT ===");
        DEBUG_PORT.printf("Leg Pos Cmd:   [%.3f, %.3f, %.3f, %.3f] rad\n",
            last_cmd.leg_positions[0], last_cmd.leg_positions[1], 
            last_cmd.leg_positions[2], last_cmd.leg_positions[3]);
        DEBUG_PORT.printf("Wheel Vel Cmd: [%.2f, %.2f] rad/s\n",
            last_cmd.wheel_velocities[0], last_cmd.wheel_velocities[1]);
        
        // === SANITY CHECK ===
        DEBUG_PORT.println("\n=== SANITY CHECK ===");
        float grav_x = s.projected_gravity[0];
        float grav_z = s.projected_gravity[2];
        float whl_l = last_cmd.wheel_velocities[0];
        float whl_r = last_cmd.wheel_velocities[1];
        
        // Check for zero commands (safety stop or stabilizing)
        bool cmds_zero = (fabs(whl_l) < 0.01f && fabs(whl_r) < 0.01f);
        
        if (grav_x < -0.1f) {
            DEBUG_PORT.printf("Robot tilting FORWARD (proj_grav[0]=%.3f)\n", grav_x);
            if (cmds_zero) {
                DEBUG_PORT.println("  Wheels: ZERO (safety_stop or stabilizing)");
            } else if (whl_l > 0 && whl_r > 0) {
                DEBUG_PORT.println("  Wheels: BOTH FORWARD ✓ (should correct forward tilt)");
            } else if (whl_l < 0 && whl_r < 0) {
                DEBUG_PORT.println("  Wheels: BOTH BACKWARD ✗ (will fall more!)");
            } else {
                DEBUG_PORT.printf("  Wheels: OPPOSITE SIGNS [%.2f, %.2f] (spinning in place!)\n", whl_l, whl_r);
            }
        } else if (grav_x > 0.1f) {
            DEBUG_PORT.printf("Robot tilting BACKWARD (proj_grav[0]=%.3f)\n", grav_x);
            if (cmds_zero) {
                DEBUG_PORT.println("  Wheels: ZERO (safety_stop or stabilizing)");
            } else if (whl_l < 0 && whl_r < 0) {
                DEBUG_PORT.println("  Wheels: BOTH BACKWARD ✓ (should correct backward tilt)");
            } else if (whl_l > 0 && whl_r > 0) {
                DEBUG_PORT.println("  Wheels: BOTH FORWARD ✗ (will fall more!)");
            } else {
                DEBUG_PORT.printf("  Wheels: OPPOSITE SIGNS [%.2f, %.2f] (spinning in place!)\n", whl_l, whl_r);
            }
        } else {
            DEBUG_PORT.printf("Robot roughly UPRIGHT (proj_grav[0]=%.3f)\n", grav_x);
        }
        
        if (grav_z > -0.5f) {
            DEBUG_PORT.printf("  ⚠ SAFETY THRESHOLD! grav_z=%.2f (limit: -0.5)\n", grav_z);
        }
        
        DEBUG_PORT.println("######################################################################\n");
    }
}