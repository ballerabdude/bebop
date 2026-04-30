/**
 * @file SerialCommands.h
 * @brief Debug serial command interface
 * 
 * Provides a simple command-line interface over serial for:
 *   - Enabling/disabling motors
 *   - Setting zero positions (Robstride)
 *   - Viewing motor status
 */

#ifndef SERIAL_COMMANDS_H
#define SERIAL_COMMANDS_H

#include <Arduino.h>

// Debug serial port
#define DEBUG_PORT SerialUSB1

/**
 * @brief Serial command handler class
 */
class SerialCommands {
public:
    /**
     * @brief Initialize the debug serial port
     * @param baud_rate Serial baud rate (default 115200)
     */
    void begin(uint32_t baud_rate = 115200);
    
    /**
     * @brief Check for and process incoming serial commands
     * Call this in the main loop
     */
    void update();
    
private:
    char cmd_buffer_[64];
    int cmd_index_ = 0;
    
    /**
     * @brief Process a complete command string
     * @param cmd Null-terminated command string
     */
    void processCommand(const char* cmd);
    
    /**
     * @brief Print help message
     */
    void printHelp();
    
    /**
     * @brief Print motor status
     */
    void printStatus();
    
    /**
     * @brief Set zero position for a Robstride motor
     * @param joint_idx Joint index (0-3 for Robstride)
     */
    void setZero(int joint_idx);
    
    /**
     * @brief Enable motor(s)
     * @param arg Command argument ("all" or joint index)
     */
    void enableMotor(const char* arg);
    
    /**
     * @brief Disable motor(s)
     * @param arg Command argument ("all" or joint index)
     */
    void disableMotor(const char* arg);
};

// Global instance
extern SerialCommands serialCommands;

#endif // SERIAL_COMMANDS_H
