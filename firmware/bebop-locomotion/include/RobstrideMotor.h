/**
 * @file RobstrideMotor.h
 * @brief Driver for Robstride actuators (RS01, RS02, RS03, RS04)
 * 
 * Based on Robstride User Manual - supports multiple motor models
 * Uses CAN 2.0 Extended Frame (29-bit ID) at 1 Mbps
 * 
 * Extended Frame ID Structure:
 *   Bit 28-24: Communication type
 *   Bit 23-8:  Data area 2 (varies by command)
 *   Bit 7-0:   Target motor CAN_ID
 * 
 * Supports multiple control modes:
 *   - Operation Control Mode (MIT-like): Position + Velocity + Kp + Kd + Torque
 *   - Position Mode (CSP): Cyclic Synchronous Position
 *   - Velocity Mode
 *   - Current Mode
 * 
 * Protocol is identical across all models - only scaling differs!
 */

#ifndef ROBSTRIDE_MOTOR_H
#define ROBSTRIDE_MOTOR_H

#include "GenericMotor.h"

// ============================================================================
// ROBSTRIDE MOTOR MODEL SPECIFICATIONS
// ============================================================================

// Motor model enumeration
enum RobstrideModel {
    RS_MODEL_01 = 1,    
    RS_MODEL_02 = 2,
    RS_MODEL_03 = 3,
    RS_MODEL_04 = 4
};

// Motor specifications structure
struct RobstrideSpecs {
    float torque_min;       // N.m
    float torque_max;       // N.m
    float velocity_min;     // rad/s
    float velocity_max;     // rad/s
    float current_max;      // A
    float kt;               // N.m/Arms (torque constant)
    float rated_torque;     // N.m
};

// Model-specific specifications (from datasheets)
// Note: Position range (-4π to 4π) and Kp/Kd ranges are same for all models
const RobstrideSpecs RS01_SPECS = { -12.0f, 12.0f, -45.0f, 45.0f, 27.0f, 0.45f, 4.0f };
const RobstrideSpecs RS02_SPECS = { -25.0f, 25.0f, -30.0f, 30.0f, 35.0f, 0.72f, 8.0f };
const RobstrideSpecs RS03_SPECS = { -60.0f, 60.0f, -20.0f, 20.0f, 43.0f, 2.36f, 20.0f };
const RobstrideSpecs RS04_SPECS = { -120.0f, 120.0f, -15.0f, 15.0f, 90.0f, 2.1f, 40.0f };

// Helper to get specs by model
inline const RobstrideSpecs& getModelSpecs(RobstrideModel model) {
    switch (model) {
        case RS_MODEL_01: return RS01_SPECS;
        case RS_MODEL_02: return RS02_SPECS;
        case RS_MODEL_03: return RS03_SPECS;
        case RS_MODEL_04: 
        default:          return RS04_SPECS;
    }
}

// ============================================================================
// COMMON PROTOCOL CONSTANTS (same for all models)
// ============================================================================

// Position range (all models use -4π to 4π with cycle counting)
#define RS_P_MIN     -12.57f    // -4π rad
#define RS_P_MAX      12.57f    // +4π rad

// PID gains for operation control mode (same for all)
#define RS_KP_MIN      0.0f
#define RS_KP_MAX   5000.0f
#define RS_KD_MIN      0.0f
#define RS_KD_MAX    100.0f

// ============================================================================
// COMMUNICATION TYPES (29-bit extended frame, bits 28-24)
// ============================================================================

#define RS_CMD_GET_ID           0x00    // Get device ID
#define RS_CMD_MOTOR_CTRL       0x01    // Operation control mode command
#define RS_CMD_FEEDBACK         0x02    // Motor feedback (response)
#define RS_CMD_ENABLE           0x03    // Motor enable
#define RS_CMD_STOP             0x04    // Motor stop
#define RS_CMD_SET_ZERO         0x06    // Set mechanical zero
#define RS_CMD_SET_CAN_ID       0x07    // Set motor CAN_ID
#define RS_CMD_PARAM_READ       0x11    // Single parameter read
#define RS_CMD_PARAM_WRITE      0x12    // Single parameter write
#define RS_CMD_FAULT_FEEDBACK   0x15    // Fault feedback
#define RS_CMD_SAVE_DATA        0x16    // Save data to flash
#define RS_CMD_SET_BAUD         0x17    // Set baud rate
#define RS_CMD_ACTIVE_REPORT    0x18    // Enable active reporting
#define RS_CMD_SET_PROTOCOL     0x19    // Set protocol type

// ============================================================================
// PARAMETER INDICES (for Type 17/18 read/write)
// ============================================================================

#define RS_PARAM_RUN_MODE       0x7005  // Run mode (0=operation, 1=PP, 2=vel, 3=current, 5=CSP)
#define RS_PARAM_IQ_REF         0x7006  // Current mode Iq command (A)
#define RS_PARAM_SPD_REF        0x700A  // Velocity mode speed command (rad/s)
#define RS_PARAM_LIMIT_TORQUE   0x700B  // Torque limit (N.m)
#define RS_PARAM_LOC_REF        0x7016  // Position mode angle command (rad)
#define RS_PARAM_LIMIT_SPD      0x7017  // Position mode speed limit (rad/s)
#define RS_PARAM_LIMIT_CUR      0x7018  // Velocity/position mode current limit (A)
#define RS_PARAM_MECH_POS       0x7019  // Mechanical angle (read-only)
#define RS_PARAM_IQF            0x701A  // Iq filtered (read-only)
#define RS_PARAM_MECH_VEL       0x701B  // Mechanical velocity (read-only)
#define RS_PARAM_VBUS           0x701C  // Bus voltage (read-only)

// ============================================================================
// RUN MODES
// ============================================================================

#define RS_MODE_OPERATION       0x00    // Operation control mode (MIT-like)
#define RS_MODE_POSITION_PP     0x01    // Position mode (Profile Position)
#define RS_MODE_VELOCITY        0x02    // Velocity mode
#define RS_MODE_CURRENT         0x03    // Current mode
#define RS_MODE_POSITION_CSP    0x05    // Position mode (Cyclic Synchronous Position)

// ============================================================================
// MOTOR STATUS (from feedback frame bits 22-23)
// ============================================================================

#define RS_STATUS_RESET         0x00    // Reset mode
#define RS_STATUS_CALI          0x01    // Calibration mode
#define RS_STATUS_MOTOR         0x02    // Motor running mode

// ============================================================================
// ROBSTRIDE MOTOR CLASS
// ============================================================================

template<typename CANType>
class RobstrideMotor : public GenericMotor {
private:
    CANType* can_bus;
    uint8_t host_id;        // Host CAN ID (master)
    uint8_t run_mode;       // Current run mode
    RobstrideModel model;   // Motor model (RS01-RS04)
    const RobstrideSpecs* specs;  // Model-specific specifications
    
    // PID gains for operation control mode
    float kp = 50.0f;
    float kd = 2.0f;
    
    // Motor status from feedback
    uint8_t motor_status = RS_STATUS_RESET;
    uint8_t fault_bits = 0;
    float motor_temp = 0.0f;

    // ========================================================================
    // UTILITY FUNCTIONS
    // ========================================================================

    // Build 29-bit extended frame ID
    uint32_t buildExtendedId(uint8_t cmd_type, uint16_t data_area2, uint8_t target_id) {
        return ((uint32_t)cmd_type << 24) | ((uint32_t)data_area2 << 8) | target_id;
    }

    // Map float to uint16_t for CAN packing
    uint16_t floatToUint16(float x, float x_min, float x_max) {
        float span = x_max - x_min;
        x = constrain(x, x_min, x_max);
        return (uint16_t)((x - x_min) * 65535.0f / span);
    }

    // Map uint16_t back to float for unpacking
    float uint16ToFloat(uint16_t x_int, float x_min, float x_max) {
        float span = x_max - x_min;
        return ((float)x_int) * span / 65535.0f + x_min;
    }

    // Send extended CAN frame
    void sendExtFrame(uint8_t cmd_type, uint16_t data_area2, uint8_t* data, uint8_t len) {
        CAN_message_t msg;
        msg.id = buildExtendedId(cmd_type, data_area2, can_id);
        msg.len = len;
        msg.flags.extended = 1;  // Use extended frame!
        
        for (int i = 0; i < len && i < 8; i++) {
            msg.buf[i] = data[i];
        }
        
        if (can_tx_debug) {
            SerialUSB1.printf("RS TX [%08lX] cmd=%02X id=%lu len=%d:", 
                              msg.id, cmd_type, can_id, msg.len);
            for (int i = 0; i < msg.len; i++) {
                SerialUSB1.printf(" %02X", msg.buf[i]);
            }
            SerialUSB1.println();
        }
        
        can_bus->write(msg);
    }

public:
    // Constructor with model selection (default RS04 for backward compatibility)
    RobstrideMotor(uint32_t id, uint8_t index, CANType* bus, 
                   RobstrideModel motor_model = RS_MODEL_04, uint8_t master_id = 0xFD) 
        : GenericMotor(id, index), can_bus(bus), host_id(master_id), model(motor_model) {
        specs = &getModelSpecs(model);
        run_mode = RS_MODE_OPERATION;  // Default to operation control mode
    }
    
    // Get model info
    RobstrideModel getModel() const { return model; }
    const RobstrideSpecs* getSpecs() const { return specs; }

    // ========================================================================
    // CONFIGURATION
    // ========================================================================

    void setGains(float new_kp, float new_kd) {
        kp = constrain(new_kp, RS_KP_MIN, RS_KP_MAX);
        kd = constrain(new_kd, RS_KD_MIN, RS_KD_MAX);
    }

    void setRunMode(uint8_t mode) {
        run_mode = mode;
        
        // Write run_mode parameter via Type 18
        uint8_t data[8] = {0};
        data[0] = RS_PARAM_RUN_MODE & 0xFF;
        data[1] = (RS_PARAM_RUN_MODE >> 8) & 0xFF;
        data[4] = mode;
        
        sendExtFrame(RS_CMD_PARAM_WRITE, host_id, data, 8);
    }

    // ========================================================================
    // CONTROL INTERFACE (implements GenericMotor)
    // ========================================================================

    void setCommand(float position, float velocity, float torque) override {
        target_position = position;
        target_velocity = velocity;
        target_torque = torque;

        if (run_mode == RS_MODE_OPERATION) {
            // Operation Control Mode (Type 1)
            // Control logic: t_ref = Kd*(v_set - v_actual) + Kp*(p_set - p_actual) + t_ff
            
            // Use model-specific ranges for velocity and torque!
            uint16_t p_int = floatToUint16(position, RS_P_MIN, RS_P_MAX);
            uint16_t v_int = floatToUint16(velocity, specs->velocity_min, specs->velocity_max);
            uint16_t kp_int = floatToUint16(kp, RS_KP_MIN, RS_KP_MAX);
            uint16_t kd_int = floatToUint16(kd, RS_KD_MIN, RS_KD_MAX);
            uint16_t t_int = floatToUint16(torque, specs->torque_min, specs->torque_max);

            // Data area 2 contains torque (bits 23-8 of extended ID)
            uint16_t data_area2 = t_int;

            uint8_t data[8];
            // Position (high byte first)
            data[0] = (p_int >> 8) & 0xFF;
            data[1] = p_int & 0xFF;
            // Velocity (high byte first)
            data[2] = (v_int >> 8) & 0xFF;
            data[3] = v_int & 0xFF;
            // Kp (high byte first)
            data[4] = (kp_int >> 8) & 0xFF;
            data[5] = kp_int & 0xFF;
            // Kd (high byte first)
            data[6] = (kd_int >> 8) & 0xFF;
            data[7] = kd_int & 0xFF;

            sendExtFrame(RS_CMD_MOTOR_CTRL, data_area2, data, 8);
        }
        else if (run_mode == RS_MODE_POSITION_CSP) {
            // Position Mode (CSP) - write loc_ref parameter
            uint8_t data[8] = {0};
            data[0] = RS_PARAM_LOC_REF & 0xFF;
            data[1] = (RS_PARAM_LOC_REF >> 8) & 0xFF;
            memcpy(&data[4], &position, 4);  // Float, little-endian
            
            sendExtFrame(RS_CMD_PARAM_WRITE, host_id, data, 8);
        }
        else if (run_mode == RS_MODE_VELOCITY) {
            // Velocity Mode - write spd_ref parameter
            uint8_t data[8] = {0};
            data[0] = RS_PARAM_SPD_REF & 0xFF;
            data[1] = (RS_PARAM_SPD_REF >> 8) & 0xFF;
            memcpy(&data[4], &velocity, 4);  // Float, little-endian
            
            sendExtFrame(RS_CMD_PARAM_WRITE, host_id, data, 8);
        }
        else if (run_mode == RS_MODE_CURRENT) {
            // Current Mode - write iq_ref parameter
            // Convert torque to current using model-specific Kt
            float iq = torque / specs->kt;  // Use model-specific torque constant
            uint8_t data[8] = {0};
            data[0] = RS_PARAM_IQ_REF & 0xFF;
            data[1] = (RS_PARAM_IQ_REF >> 8) & 0xFF;
            memcpy(&data[4], &iq, 4);  // Float, little-endian
            
            sendExtFrame(RS_CMD_PARAM_WRITE, host_id, data, 8);
        }
    }

    void handleCanMessage(const CAN_message_t &msg) override {
        // Only process extended frames
        if (!msg.flags.extended) return;

        // Parse extended frame ID
        // Frame ID: [Type 5b] [Status/Fault 8b] [MotorID 8b] [HostID 8b]
        uint8_t cmd_type = (msg.id >> 24) & 0x1F;
        uint16_t data_area2 = (msg.id >> 8) & 0xFFFF; // Contains Status (high byte) and MotorID (low byte)
        
        // Check if this is a feedback frame (Type 2) or Active Report (Type 24)
        if (cmd_type == RS_CMD_FEEDBACK || cmd_type == RS_CMD_ACTIVE_REPORT) {
            uint8_t motor_id = data_area2 & 0xFF; // Motor ID is in Bits 8-15 (Low byte of data_area2)
            
            if (motor_id != can_id) return;

            // Status bits are in Bits 16-23 (High byte of data_area2)
            uint8_t status_byte = (data_area2 >> 8) & 0xFF;

            // Parse fault and status
            fault_bits = status_byte & 0x3F;       // Bits 16-21 (0-5 of status byte)
            motor_status = (status_byte >> 6) & 0x03; // Bits 22-23 (6-7 of status byte)
            
            // Parse data bytes
            uint16_t pos_int = ((uint16_t)msg.buf[0] << 8) | msg.buf[1];
            uint16_t vel_int = ((uint16_t)msg.buf[2] << 8) | msg.buf[3];
            uint16_t torque_int = ((uint16_t)msg.buf[4] << 8) | msg.buf[5];
            int16_t temp_raw = ((int16_t)msg.buf[6] << 8) | msg.buf[7];

            // Use model-specific ranges for velocity and torque
            current_position = uint16ToFloat(pos_int, RS_P_MIN, RS_P_MAX);
            current_velocity = uint16ToFloat(vel_int, specs->velocity_min, specs->velocity_max);
            current_torque = uint16ToFloat(torque_int, specs->torque_min, specs->torque_max);
            motor_temp = (float)temp_raw / 10.0f;  // Temperature * 10

            has_error = (fault_bits != 0);
            is_enabled = (motor_status == RS_STATUS_MOTOR);
            last_feedback_time = millis();
        }
    }

    void enable() override {
        // Communication Type 3: Motor Enable
        uint8_t data[8] = {0};
        sendExtFrame(RS_CMD_ENABLE, host_id, data, 8);
        is_enabled = true;
    }
    
    // Send a "hold current position" command after enabling
    // Per docs section 4.3.1: Enable -> Control Command -> Feedback
    // Call this AFTER enable() and AFTER receiving valid feedback!
    void holdPosition() {
        if (run_mode != RS_MODE_OPERATION) return;  // Only valid in operation mode
        
        float hold_pos = current_position;
        uint16_t p_int = floatToUint16(hold_pos, RS_P_MIN, RS_P_MAX);
        uint16_t v_int = floatToUint16(0.0f, specs->velocity_min, specs->velocity_max);
        uint16_t kp_int = floatToUint16(kp, RS_KP_MIN, RS_KP_MAX);
        uint16_t kd_int = floatToUint16(kd, RS_KD_MIN, RS_KD_MAX);
        uint16_t t_int = floatToUint16(0.0f, specs->torque_min, specs->torque_max);
        
        uint8_t ctrl_data[8];
        ctrl_data[0] = (p_int >> 8) & 0xFF;
        ctrl_data[1] = p_int & 0xFF;
        ctrl_data[2] = (v_int >> 8) & 0xFF;
        ctrl_data[3] = v_int & 0xFF;
        ctrl_data[4] = (kp_int >> 8) & 0xFF;
        ctrl_data[5] = kp_int & 0xFF;
        ctrl_data[6] = (kd_int >> 8) & 0xFF;
        ctrl_data[7] = kd_int & 0xFF;
        
        sendExtFrame(RS_CMD_MOTOR_CTRL, t_int, ctrl_data, 8);
    }

    void disable() override {
        // Communication Type 4: Motor Stop
        uint8_t data[8] = {0};
        sendExtFrame(RS_CMD_STOP, host_id, data, 8);
        is_enabled = false;
    }

    void requestFeedback() override {
        // Robstride sends feedback after each command
        // No explicit request needed in operation mode
    }

    // ========================================================================
    // ROBSTRIDE-SPECIFIC FUNCTIONS
    // ========================================================================

    void setMechanicalZero() {
        // Communication Type 6: Set mechanical zero
        uint8_t data[8] = {0};
        data[0] = 0x01;  // Byte[0] = 1 to set zero
        sendExtFrame(RS_CMD_SET_ZERO, host_id, data, 8);
    }

    void clearFault() {
        // Communication Type 4 with Byte[0] = 1 clears fault
        uint8_t data[8] = {0};
        data[0] = 0x01;
        sendExtFrame(RS_CMD_STOP, host_id, data, 8);
        has_error = false;
    }

    void enableActiveReporting(bool enable_report, uint16_t interval_ms = 10) {
        // Communication Type 24: Enable/disable active reporting
        uint8_t data[8] = {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x00};
        data[6] = enable_report ? 0x01 : 0x00;
        sendExtFrame(RS_CMD_ACTIVE_REPORT, host_id, data, 7);
        
        if (enable_report && interval_ms != 10) {
            // Set reporting interval via parameter write
            // EPScan_time: 1 = 10ms, each +1 adds 5ms
            uint16_t scan_time = (interval_ms <= 10) ? 1 : 1 + (interval_ms - 10) / 5;
            uint8_t param_data[8] = {0};
            param_data[0] = 0x26;  // EPScan_time index low
            param_data[1] = 0x70;  // EPScan_time index high
            param_data[4] = scan_time & 0xFF;
            param_data[5] = (scan_time >> 8) & 0xFF;
            sendExtFrame(RS_CMD_PARAM_WRITE, host_id, param_data, 8);
        }
    }

    void saveParameters() {
        // Communication Type 22: Save data to flash
        uint8_t data[8] = {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08};
        sendExtFrame(RS_CMD_SAVE_DATA, host_id, data, 8);
    }

    // Getters for motor status
    float getTemperature() const { return motor_temp; }
    uint8_t getFaultBits() const { return fault_bits; }
    uint8_t getMotorStatus() const { return motor_status; }
    
    // Fault bit meanings (from Type 2 feedback, bits 16-21 of extended frame):
    // Bit 0 (frame bit 16): Undervoltage fault (<12V)
    // Bit 1 (frame bit 17): Three-phase overcurrent fault
    // Bit 2 (frame bit 18): Overtemperature (>145°C)
    // Bit 3 (frame bit 19): Magnetic encoding fault
    // Bit 4 (frame bit 20): Gridlock/stall overload protection
    // Bit 5 (frame bit 21): Encoder not calibrated
    bool isUndervoltage() const { return fault_bits & 0x01; }
    bool hasOvercurrent() const { return fault_bits & 0x02; }
    bool isOverTemp() const { return fault_bits & 0x04; }
    bool hasMagneticEncodingFault() const { return fault_bits & 0x08; }
    bool hasStallFault() const { return fault_bits & 0x10; }
    bool isEncoderUncalibrated() const { return fault_bits & 0x20; }
};

#endif // ROBSTRIDE_MOTOR_H
