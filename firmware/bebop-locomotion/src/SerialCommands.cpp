/**
 * @file SerialCommands.cpp
 * @brief Debug serial command interface implementation
 */

#include "SerialCommands.h"
#include "RobotConfig.h"

// Global instance
SerialCommands serialCommands;

void SerialCommands::begin(uint32_t baud_rate) {
    DEBUG_PORT.begin(baud_rate);
}

void SerialCommands::update() {
    while (DEBUG_PORT.available()) {
        char c = DEBUG_PORT.read();
        if (c == '\n' || c == '\r') {
            if (cmd_index_ > 0) {
                cmd_buffer_[cmd_index_] = '\0';
                processCommand(cmd_buffer_);
                cmd_index_ = 0;
            }
        } else if (cmd_index_ < 63) {
            cmd_buffer_[cmd_index_++] = c;
        }
    }
}

void SerialCommands::processCommand(const char* cmd) {
    DEBUG_PORT.printf("CMD: %s\n", cmd);
    
    if (strncmp(cmd, "zero ", 5) == 0) {
        int joint_idx = atoi(cmd + 5);
        setZero(joint_idx);
    }
    else if (strncmp(cmd, "enable ", 7) == 0) {
        enableMotor(cmd + 7);
    }
    else if (strncmp(cmd, "disable ", 8) == 0) {
        disableMotor(cmd + 8);
    }
    else if (strcmp(cmd, "status") == 0 || strcmp(cmd, "pos") == 0) {
        printStatus();
    }
    else if (strcmp(cmd, "help") == 0) {
        printHelp();
    }
    else if (strcmp(cmd, "candebug") == 0) {
        // Toggle both
        can_tx_debug = !can_tx_debug;
        can_rx_debug = can_tx_debug;
        DEBUG_PORT.printf("  CAN debug (TX+RX): %s\n", can_tx_debug ? "ON" : "OFF");
    }
    else if (strcmp(cmd, "candebug tx") == 0) {
        can_tx_debug = !can_tx_debug;
        DEBUG_PORT.printf("  CAN TX debug: %s\n", can_tx_debug ? "ON" : "OFF");
    }
    else if (strcmp(cmd, "candebug rx") == 0) {
        can_rx_debug = !can_rx_debug;
        DEBUG_PORT.printf("  CAN RX debug: %s\n", can_rx_debug ? "ON" : "OFF");
    }
    else if (strcmp(cmd, "candebug off") == 0) {
        can_tx_debug = false;
        can_rx_debug = false;
        DEBUG_PORT.println("  CAN debug: OFF");
    }
    else if (strcmp(cmd, "imudebug") == 0) {
        imu_debug = !imu_debug;
        DEBUG_PORT.printf("  IMU debug: %s\n", imu_debug ? "ON" : "OFF");
    }
    else if (strcmp(cmd, "debug") == 0) {
        // Toggle all debug flags
        bool all_on = can_tx_debug || can_rx_debug || imu_debug;
        can_tx_debug = can_rx_debug = imu_debug = !all_on;
        DEBUG_PORT.printf("  All debug: %s\n", !all_on ? "ON" : "OFF");
    }
    else if (strcmp(cmd, "debug off") == 0) {
        can_tx_debug = can_rx_debug = imu_debug = false;
        DEBUG_PORT.println("  All debug: OFF");
    }
    else if (strcmp(cmd, "quiet") == 0 || strcmp(cmd, "silent") == 0) {
        // Disable ALL logging including status
        can_tx_debug = can_rx_debug = imu_debug = false;
        status_logging = false;
        DEBUG_PORT.println("  Silent mode: all logging OFF");
    }
    else if (strcmp(cmd, "verbose") == 0) {
        // Re-enable status logging
        status_logging = true;
        DEBUG_PORT.println("  Status logging: ON");
    }
    else {
        DEBUG_PORT.println("  Unknown command. Type 'help' for available commands.");
    }
}

void SerialCommands::printHelp() {
    DEBUG_PORT.println("  Commands:");
    DEBUG_PORT.println("    pos             - Show all joint positions (rad & deg)");
    DEBUG_PORT.println("    zero <joint>    - Set current pos as zero (Robstride 0-3)");
    DEBUG_PORT.println("    enable <joint|all>  - Enable motor(s)");
    DEBUG_PORT.println("    disable <joint|all> - Disable motor(s)");
    DEBUG_PORT.println("    candebug        - Toggle CAN TX+RX message logging");
    DEBUG_PORT.println("    candebug tx/rx  - Toggle TX or RX only");
    DEBUG_PORT.println("    imudebug        - Toggle IMU data logging");
    DEBUG_PORT.println("    debug           - Toggle all debug logging");
    DEBUG_PORT.println("    debug off       - Disable all debug logging");
    DEBUG_PORT.println("    quiet           - Silence all output (incl. status)");
    DEBUG_PORT.println("    verbose         - Re-enable status output");
    DEBUG_PORT.println("    help            - Show this help");
    DEBUG_PORT.println("  Joints: 0=L_hip, 1=R_hip, 2=L_knee, 3=R_knee, 4=L_wheel, 5=R_wheel");
}

void SerialCommands::printStatus() {
    DEBUG_PORT.println("  Joint Status:");
    for (int i = 0; i < NUM_JOINTS; i++) {
        const char* status = (millis() - joints[i]->last_feedback_time < 500) ? "OK" : "NO CAN";
        DEBUG_PORT.printf("    [%d] %s: %s pos=%.3f rad (%.1f deg)\n",
            i, JOINT_NAMES[i], status,
            joints[i]->current_position,
            joints[i]->current_position * 57.2958f);
    }
}

void SerialCommands::setZero(int joint_idx) {
    if (joint_idx >= 0 && joint_idx <= ROBSTRIDE_END_IDX) {
        RobstrideMotorType* motor = static_cast<RobstrideMotorType*>(joints[joint_idx]);
        
        DEBUG_PORT.printf("  Setting zero for joint %d (%s)...\n", joint_idx, JOINT_NAMES[joint_idx]);
        
        // 1. Disable motor first (must be in reset mode)
        DEBUG_PORT.println("    1. Disabling motor...");
        motor->disable();
        delay(100);
        
        // 2. Set mechanical zero
        DEBUG_PORT.println("    2. Setting mechanical zero...");
        motor->setMechanicalZero();
        delay(100);
        
        // 3. Save to flash
        DEBUG_PORT.println("    3. Saving to flash...");
        motor->saveParameters();
        delay(200);
        
        // 4. Re-enable motor
        DEBUG_PORT.println("    4. Re-enabling motor...");
        motor->enable();
        delay(50);
        
        DEBUG_PORT.printf("  Done! Zero position set for %s\n", JOINT_NAMES[joint_idx]);
    } else {
        DEBUG_PORT.printf("  ERROR: Joint %d is not a Robstride motor (0-%d)\n", 
                          joint_idx, ROBSTRIDE_END_IDX);
    }
}

void SerialCommands::enableMotor(const char* arg) {
    if (strncmp(arg, "all", 3) == 0) {
        for (int i = 0; i < NUM_JOINTS; i++) {
            joints[i]->enable();
            delay(5);
        }
        DEBUG_PORT.println("  Enabled all motors");
    } else {
        int joint_idx = atoi(arg);
        if (joint_idx >= 0 && joint_idx < NUM_JOINTS) {
            joints[joint_idx]->enable();
            DEBUG_PORT.printf("  Enabled joint %d (%s)\n", joint_idx, JOINT_NAMES[joint_idx]);
        }
    }
}

void SerialCommands::disableMotor(const char* arg) {
    if (strncmp(arg, "all", 3) == 0) {
        for (int i = 0; i < NUM_JOINTS; i++) {
            joints[i]->disable();
            delay(5);
        }
        DEBUG_PORT.println("  Disabled all motors");
    } else {
        int joint_idx = atoi(arg);
        if (joint_idx >= 0 && joint_idx < NUM_JOINTS) {
            joints[joint_idx]->disable();
            DEBUG_PORT.printf("  Disabled joint %d (%s)\n", joint_idx, JOINT_NAMES[joint_idx]);
        }
    }
}
