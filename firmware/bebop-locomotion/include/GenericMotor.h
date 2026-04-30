/**
 * @file GenericMotor.h
 * @brief Abstract base class for all motor types
 * 
 * This provides a unified interface for controlling different motor
 * hardware (Robstride, ODrive, etc.) through polymorphism.
 */

#ifndef GENERIC_MOTOR_H
#define GENERIC_MOTOR_H

#include <FlexCAN_T4.h>

class GenericMotor {
public:
    uint32_t can_id;
    uint8_t joint_index;
    
    // Current state (updated from CAN feedback)
    float current_position = 0.0f;
    float current_velocity = 0.0f;
    float current_torque = 0.0f;
    
    // Target state
    float target_position = 0.0f;
    float target_velocity = 0.0f;
    float target_torque = 0.0f;
    
    // Status flags
    bool is_enabled = false;
    bool has_error = false;
    uint32_t last_feedback_time = 0;
    
    GenericMotor(uint32_t id, uint8_t index) : can_id(id), joint_index(index) {}
    virtual ~GenericMotor() {}
    
    // Pure virtual methods - must be implemented by subclasses
    virtual void setCommand(float position, float velocity, float torque) = 0;
    virtual void handleCanMessage(const CAN_message_t &msg) = 0;
    virtual void enable() = 0;
    virtual void disable() = 0;
    virtual void requestFeedback() = 0;
    
    // Common utility
    bool isTimedOut(uint32_t timeout_ms = 100) {
        return (millis() - last_feedback_time) > timeout_ms;
    }
};

#endif // GENERIC_MOTOR_H

