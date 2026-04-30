/**
 * @file ODriveMotor.h
 * @brief Driver for ODrive S1 motor controllers
 * 
 * Based on ODrive CAN Protocol (CANSimple)
 * Uses CAN 2.0 Standard Frame (11-bit ID)
 * 
 * CAN ID Format: (node_id << 5) | cmd_id
 * 
 * IMPORTANT UNITS:
 *   - Position: revolutions (NOT radians!)
 *   - Velocity: rev/s (NOT rad/s!)
 *   - Torque: Nm
 *   - All values: little-endian, IEEE 754 floats
 */

#ifndef ODRIVE_MOTOR_H
#define ODRIVE_MOTOR_H

#include "GenericMotor.h"

// ============================================================================
// ODRIVE CAN COMMAND IDs (combined with node_id to form CAN ID)
// ============================================================================

#define ODRIVE_CMD_GET_VERSION          0x00
#define ODRIVE_CMD_HEARTBEAT            0x01    // Cyclic (100ms default)
#define ODRIVE_CMD_ESTOP                0x02
#define ODRIVE_CMD_GET_ERROR            0x03
#define ODRIVE_CMD_RXSDO                0x04
#define ODRIVE_CMD_TXSDO                0x05
#define ODRIVE_CMD_ADDRESS              0x06
#define ODRIVE_CMD_SET_AXIS_STATE       0x07
#define ODRIVE_CMD_GET_ENCODER_EST      0x09    // Cyclic (10ms default)
#define ODRIVE_CMD_SET_CONTROLLER_MODE  0x0B
#define ODRIVE_CMD_SET_INPUT_POS        0x0C
#define ODRIVE_CMD_SET_INPUT_VEL        0x0D
#define ODRIVE_CMD_SET_INPUT_TORQUE     0x0E
#define ODRIVE_CMD_SET_LIMITS           0x0F
#define ODRIVE_CMD_SET_TRAJ_VEL_LIMIT   0x11
#define ODRIVE_CMD_SET_TRAJ_ACCEL       0x12
#define ODRIVE_CMD_SET_TRAJ_INERTIA     0x13
#define ODRIVE_CMD_GET_IQ               0x14
#define ODRIVE_CMD_GET_TEMPERATURE      0x15
#define ODRIVE_CMD_REBOOT               0x16
#define ODRIVE_CMD_GET_BUS_VOLTAGE      0x17
#define ODRIVE_CMD_CLEAR_ERRORS         0x18
#define ODRIVE_CMD_SET_ABS_POSITION     0x19
#define ODRIVE_CMD_SET_POS_GAIN         0x1A
#define ODRIVE_CMD_SET_VEL_GAINS        0x1B
#define ODRIVE_CMD_GET_TORQUES          0x1C
#define ODRIVE_CMD_GET_POWERS           0x1D
#define ODRIVE_CMD_ENTER_DFU            0x1F

// ============================================================================
// AXIS STATES
// ============================================================================

#define ODRIVE_AXIS_STATE_UNDEFINED             0
#define ODRIVE_AXIS_STATE_IDLE                  1
#define ODRIVE_AXIS_STATE_STARTUP_SEQUENCE      2
#define ODRIVE_AXIS_STATE_FULL_CALIBRATION      3
#define ODRIVE_AXIS_STATE_MOTOR_CALIBRATION     4
#define ODRIVE_AXIS_STATE_ENCODER_INDEX_SEARCH  6
#define ODRIVE_AXIS_STATE_ENCODER_OFFSET_CALIB  7
#define ODRIVE_AXIS_STATE_CLOSED_LOOP_CONTROL   8
#define ODRIVE_AXIS_STATE_LOCKIN_SPIN           9
#define ODRIVE_AXIS_STATE_ENCODER_DIR_FIND      10
#define ODRIVE_AXIS_STATE_HOMING                11
#define ODRIVE_AXIS_STATE_ENCODER_HALL_POLARITY 12
#define ODRIVE_AXIS_STATE_ENCODER_HALL_PHASE    13

// ============================================================================
// CONTROL MODES
// ============================================================================

#define ODRIVE_CONTROL_MODE_VOLTAGE     0
#define ODRIVE_CONTROL_MODE_TORQUE      1
#define ODRIVE_CONTROL_MODE_VELOCITY    2
#define ODRIVE_CONTROL_MODE_POSITION    3

// ============================================================================
// INPUT MODES
// ============================================================================

#define ODRIVE_INPUT_MODE_INACTIVE      0
#define ODRIVE_INPUT_MODE_PASSTHROUGH   1
#define ODRIVE_INPUT_MODE_VEL_RAMP      2
#define ODRIVE_INPUT_MODE_POS_FILTER    3
#define ODRIVE_INPUT_MODE_MIX_CHANNELS  4
#define ODRIVE_INPUT_MODE_TRAP_TRAJ     5
#define ODRIVE_INPUT_MODE_TORQUE_RAMP   6
#define ODRIVE_INPUT_MODE_MIRROR        7
#define ODRIVE_INPUT_MODE_TUNING        8

// ============================================================================
// UNIT CONVERSION (ODrive uses revolutions, we use radians internally)
// ============================================================================

#define RAD_TO_REV (1.0f / (2.0f * 3.14159265359f))  // rad to revolutions
#define REV_TO_RAD (2.0f * 3.14159265359f)           // revolutions to rad

// ============================================================================
// ODRIVE MOTOR CLASS
// ============================================================================

template<typename CANType>
class ODriveMotor : public GenericMotor {
private:
    CANType* can_bus;
    uint8_t node_id;
    
    uint8_t control_mode = ODRIVE_CONTROL_MODE_VELOCITY;
    uint8_t input_mode = ODRIVE_INPUT_MODE_PASSTHROUGH;
    
    // Status from heartbeat
    uint32_t axis_error = 0;
    uint8_t axis_state = ODRIVE_AXIS_STATE_IDLE;
    uint8_t procedure_result = 0;
    bool trajectory_done = false;
    
    // Additional telemetry
    float bus_voltage = 0.0f;
    float bus_current = 0.0f;
    float fet_temp = 0.0f;
    float motor_temp = 0.0f;
    float iq_setpoint = 0.0f;
    float iq_measured = 0.0f;
    float torque_constant = 0.083f; // Default for M8325s 100KV

    // ========================================================================
    // HELPER FUNCTIONS
    // ========================================================================

    // Build standard 11-bit CAN ID
    uint32_t makeCanId(uint8_t cmd_id) {
        return ((uint32_t)node_id << 5) | cmd_id;
    }

    // Extract node_id and cmd_id from CAN ID
    void parseCanId(uint32_t can_id, uint8_t &out_node_id, uint8_t &out_cmd_id) {
        out_node_id = (can_id >> 5) & 0x3F;
        out_cmd_id = can_id & 0x1F;
    }

    // Pack float into buffer (little-endian)
    void packFloat(uint8_t* buf, uint8_t offset, float value) {
        memcpy(&buf[offset], &value, 4);
    }

    // Unpack float from buffer (little-endian)
    float unpackFloat(const uint8_t* buf, uint8_t offset) {
        float value;
        memcpy(&value, &buf[offset], 4);
        return value;
    }

    // Pack uint32 into buffer (little-endian)
    void packUint32(uint8_t* buf, uint8_t offset, uint32_t value) {
        buf[offset + 0] = (value >> 0) & 0xFF;
        buf[offset + 1] = (value >> 8) & 0xFF;
        buf[offset + 2] = (value >> 16) & 0xFF;
        buf[offset + 3] = (value >> 24) & 0xFF;
    }

    // Unpack uint32 from buffer (little-endian)
    uint32_t unpackUint32(const uint8_t* buf, uint8_t offset) {
        return ((uint32_t)buf[offset + 0]) |
               ((uint32_t)buf[offset + 1] << 8) |
               ((uint32_t)buf[offset + 2] << 16) |
               ((uint32_t)buf[offset + 3] << 24);
    }

    // Send standard CAN frame
    void sendFrame(uint8_t cmd_id, uint8_t* data, uint8_t len) {
        CAN_message_t msg;
        msg.id = makeCanId(cmd_id);
        msg.len = len;
        msg.flags.extended = 0;  // Standard frame!
        
        for (int i = 0; i < len && i < 8; i++) {
            msg.buf[i] = data[i];
        }
        
        if (can_tx_debug) {
            SerialUSB1.printf("OD TX [%03lX] node=%d cmd=%02X len=%d:", 
                              msg.id, node_id, cmd_id, msg.len);
            for (int i = 0; i < msg.len; i++) {
                SerialUSB1.printf(" %02X", msg.buf[i]);
            }
            // Decode floats for velocity/position commands
            if (cmd_id == ODRIVE_CMD_SET_INPUT_VEL && len >= 8) {
                float vel = unpackFloat(msg.buf, 0);
                float torque = unpackFloat(msg.buf, 4);
                SerialUSB1.printf(" (vel=%.3f rev/s, torque=%.3f Nm)", vel, torque);
            } else if (cmd_id == ODRIVE_CMD_SET_INPUT_POS && len >= 4) {
                float pos = unpackFloat(msg.buf, 0);
                SerialUSB1.printf(" (pos=%.3f rev)", pos);
            }
            SerialUSB1.println();
        }
        
        can_bus->write(msg);
    }

public:
    ODriveMotor(uint8_t node, uint8_t index, CANType* bus, float kt = 0.083f) 
        : GenericMotor(node, index), can_bus(bus), node_id(node), torque_constant(kt) {
        can_id = node;
    }

    // ========================================================================
    // CONFIGURATION
    // ========================================================================

    void setControlMode(uint8_t mode, uint8_t in_mode = ODRIVE_INPUT_MODE_PASSTHROUGH) {
        control_mode = mode;
        input_mode = in_mode;
        
        // Set_Controller_Mode (0x0B)
        // Byte 0-3: Control_Mode (uint32)
        // Byte 4-7: Input_Mode (uint32)
        uint8_t data[8];
        packUint32(data, 0, mode);
        packUint32(data, 4, in_mode);
        sendFrame(ODRIVE_CMD_SET_CONTROLLER_MODE, data, 8);
    }

    void setLimits(float vel_limit_rev_s, float current_limit_a) {
        // Set_Limits (0x0F)
        // Byte 0-3: Velocity_Limit (float, rev/s)
        // Byte 4-7: Current_Limit (float, A)
        uint8_t data[8];
        packFloat(data, 0, vel_limit_rev_s);
        packFloat(data, 4, current_limit_a);
        sendFrame(ODRIVE_CMD_SET_LIMITS, data, 8);
    }

    void setVelGains(float vel_gain, float vel_integrator_gain) {
        // Set_Vel_Gains (0x1B)
        // Byte 0-3: Vel_Gain (float, Nm/(rev/s))
        // Byte 4-7: Vel_Integrator_Gain (float, Nm/rev)
        uint8_t data[8];
        packFloat(data, 0, vel_gain);
        packFloat(data, 4, vel_integrator_gain);
        sendFrame(ODRIVE_CMD_SET_VEL_GAINS, data, 8);
    }

    void setPosGain(float pos_gain) {
        // Set_Pos_Gain (0x1A)
        // Byte 0-3: Pos_Gain (float, (rev/s)/rev)
        uint8_t data[4];
        packFloat(data, 0, pos_gain);
        sendFrame(ODRIVE_CMD_SET_POS_GAIN, data, 4);
    }

    // ========================================================================
    // CONTROL INTERFACE (implements GenericMotor)
    // ========================================================================

    void setCommand(float position, float velocity, float torque) override {
        // Store targets (in radians internally)
        target_position = position;
        target_velocity = velocity;
        target_torque = torque;

        // Convert radians to revolutions for ODrive
        float pos_rev = position * RAD_TO_REV;
        float vel_rev_s = velocity * RAD_TO_REV;

        switch (control_mode) {
            case ODRIVE_CONTROL_MODE_POSITION: {
                // Set_Input_Pos (0x0C)
                // Byte 0-3: Input_Pos (float, rev)
                // Byte 4-5: Vel_FF (int16, 0.001 rev/s scale)
                // Byte 6-7: Torque_FF (int16, 0.001 Nm scale)
                uint8_t data[8];
                packFloat(data, 0, pos_rev);
                
                // Velocity feedforward (0.001 rev/s per LSB)
                int16_t vel_ff = (int16_t)(vel_rev_s * 1000.0f);
                data[4] = vel_ff & 0xFF;
                data[5] = (vel_ff >> 8) & 0xFF;
                
                // Torque feedforward (0.001 Nm per LSB)
                int16_t torque_ff = (int16_t)(torque * 1000.0f);
                data[6] = torque_ff & 0xFF;
                data[7] = (torque_ff >> 8) & 0xFF;
                
                sendFrame(ODRIVE_CMD_SET_INPUT_POS, data, 8);
                break;
            }

            case ODRIVE_CONTROL_MODE_VELOCITY: {
                // Set_Input_Vel (0x0D)
                // Byte 0-3: Input_Vel (float, rev/s)
                // Byte 4-7: Input_Torque_FF (float, Nm)
                uint8_t data[8];
                packFloat(data, 0, vel_rev_s);
                packFloat(data, 4, torque);
                sendFrame(ODRIVE_CMD_SET_INPUT_VEL, data, 8);
                break;
            }

            case ODRIVE_CONTROL_MODE_TORQUE: {
                // Set_Input_Torque (0x0E)
                // Byte 0-3: Input_Torque (float, Nm)
                uint8_t data[4];
                packFloat(data, 0, torque);
                sendFrame(ODRIVE_CMD_SET_INPUT_TORQUE, data, 4);
                break;
            }
        }
    }

    void handleCanMessage(const CAN_message_t &msg) override {
        // Only handle standard frames
        if (msg.flags.extended) return;

        uint8_t msg_node_id, cmd_id;
        parseCanId(msg.id, msg_node_id, cmd_id);

        if (msg_node_id != node_id) return;

        switch (cmd_id) {
            case ODRIVE_CMD_HEARTBEAT: {
                // Heartbeat (0x01) - Sent every 100ms by default
                // Byte 0-3: Axis_Error (uint32)
                // Byte 4: Axis_State (uint8)
                // Byte 5: Procedure_Result (uint8)
                // Byte 6: Trajectory_Done_Flag (uint8)
                axis_error = unpackUint32(msg.buf, 0);
                axis_state = msg.buf[4];
                procedure_result = msg.buf[5];
                trajectory_done = (msg.buf[6] != 0);
                
                has_error = (axis_error != 0);
                is_enabled = (axis_state == ODRIVE_AXIS_STATE_CLOSED_LOOP_CONTROL);
                last_feedback_time = millis();
                break;
            }

            case ODRIVE_CMD_GET_ENCODER_EST: {
                // Get_Encoder_Estimates (0x09) - Sent every 10ms by default
                // Byte 0-3: Pos_Estimate (float, rev)
                // Byte 4-7: Vel_Estimate (float, rev/s)
                float pos_rev = unpackFloat(msg.buf, 0);
                float vel_rev_s = unpackFloat(msg.buf, 4);
                
                // Convert to radians for internal use
                current_position = pos_rev * REV_TO_RAD;
                current_velocity = vel_rev_s * REV_TO_RAD;
                last_feedback_time = millis();
                break;
            }

            case ODRIVE_CMD_GET_IQ: {
                // Get_Iq (0x14)
                // Byte 0-3: Iq_Setpoint (float, A)
                // Byte 4-7: Iq_Measured (float, A)
                iq_setpoint = unpackFloat(msg.buf, 0);
                iq_measured = unpackFloat(msg.buf, 4);
                // Calculate torque from measured current
                current_torque = iq_measured * torque_constant;
                break;
            }

            case ODRIVE_CMD_GET_TEMPERATURE: {
                // Get_Temperature (0x15)
                // Byte 0-3: FET_Temperature (float, °C)
                // Byte 4-7: Motor_Temperature (float, °C)
                fet_temp = unpackFloat(msg.buf, 0);
                motor_temp = unpackFloat(msg.buf, 4);
                break;
            }

            case ODRIVE_CMD_GET_BUS_VOLTAGE: {
                // Get_Bus_Voltage_Current (0x17)
                // Byte 0-3: Bus_Voltage (float, V)
                // Byte 4-7: Bus_Current (float, A)
                bus_voltage = unpackFloat(msg.buf, 0);
                bus_current = unpackFloat(msg.buf, 4);
                break;
            }

            case ODRIVE_CMD_GET_TORQUES: {
                // Get_Torques (0x1C)
                // Byte 0-3: Torque_Target (float, Nm)
                // Byte 4-7: Torque_Estimate (float, Nm)
                current_torque = unpackFloat(msg.buf, 4);  // Use estimate
                break;
            }

            case ODRIVE_CMD_GET_ERROR: {
                // Get_Error (0x03)
                // Byte 0-3: Active_Errors (uint32)
                // Byte 4-7: Disarm_Reason (uint32)
                axis_error = unpackUint32(msg.buf, 0);
                has_error = (axis_error != 0);
                break;
            }
        }
    }

    void enable() override {
        // Set_Axis_State (0x07) to CLOSED_LOOP_CONTROL
        uint8_t data[4];
        packUint32(data, 0, ODRIVE_AXIS_STATE_CLOSED_LOOP_CONTROL);
        sendFrame(ODRIVE_CMD_SET_AXIS_STATE, data, 4);
        is_enabled = true;
    }

    void disable() override {
        // Set_Axis_State (0x07) to IDLE
        uint8_t data[4];
        packUint32(data, 0, ODRIVE_AXIS_STATE_IDLE);
        sendFrame(ODRIVE_CMD_SET_AXIS_STATE, data, 4);
        is_enabled = false;
    }

    void requestFeedback() override {
        // Request encoder estimates via RTR frame
        // Note: ODrive sends these cyclically by default (every 10ms)
        // This is for explicit requests if needed
        CAN_message_t msg;
        msg.id = makeCanId(ODRIVE_CMD_GET_ENCODER_EST);
        msg.len = 0;
        msg.flags.extended = 0;
        msg.flags.remote = 1;  // RTR bit
        can_bus->write(msg);
    }

    // ========================================================================
    // ODRIVE-SPECIFIC FUNCTIONS
    // ========================================================================

    void estop() {
        // Estop (0x02) - Empty payload
        uint8_t data[1] = {0};
        sendFrame(ODRIVE_CMD_ESTOP, data, 0);
        is_enabled = false;
    }

    void clearErrors(bool identify = false) {
        // Clear_Errors (0x18)
        // Byte 0: Identify (uint8) - blinks LED if true
        uint8_t data[1] = { identify ? (uint8_t)1 : (uint8_t)0 };
        sendFrame(ODRIVE_CMD_CLEAR_ERRORS, data, 1);
        has_error = false;
        axis_error = 0;
    }

    void reboot(uint8_t action = 0) {
        // Reboot (0x16)
        // action: 0=reboot, 1=save_config, 2=erase_config, 3=enter_dfu
        uint8_t data[1] = { action };
        sendFrame(ODRIVE_CMD_REBOOT, data, 1);
    }

    void setAbsolutePosition(float position_rad) {
        // Set_Absolute_Position (0x19)
        // Sets the reference frame origin
        uint8_t data[4];
        packFloat(data, 0, position_rad * RAD_TO_REV);
        sendFrame(ODRIVE_CMD_SET_ABS_POSITION, data, 4);
    }

    void setTrajectoryLimits(float vel_limit, float accel_limit, float decel_limit) {
        // Set_Traj_Vel_Limit (0x11)
        uint8_t vel_data[4];
        packFloat(vel_data, 0, vel_limit * RAD_TO_REV);
        sendFrame(ODRIVE_CMD_SET_TRAJ_VEL_LIMIT, vel_data, 4);
        
        // Set_Traj_Accel_Limits (0x12)
        uint8_t accel_data[8];
        packFloat(accel_data, 0, accel_limit * RAD_TO_REV);
        packFloat(accel_data, 4, decel_limit * RAD_TO_REV);
        sendFrame(ODRIVE_CMD_SET_TRAJ_ACCEL, accel_data, 8);
    }

    // Request temperature (for non-cyclic operation)
    void requestTemperature() {
        CAN_message_t msg;
        msg.id = makeCanId(ODRIVE_CMD_GET_TEMPERATURE);
        msg.len = 0;
        msg.flags.extended = 0;
        msg.flags.remote = 1;
        can_bus->write(msg);
    }

    // Request bus voltage (for non-cyclic operation)
    void requestBusVoltage() {
        CAN_message_t msg;
        msg.id = makeCanId(ODRIVE_CMD_GET_BUS_VOLTAGE);
        msg.len = 0;
        msg.flags.extended = 0;
        msg.flags.remote = 1;
        can_bus->write(msg);
    }

    // ========================================================================
    // GETTERS FOR TELEMETRY
    // ========================================================================

    uint32_t getAxisError() const { return axis_error; }
    uint8_t getAxisState() const { return axis_state; }
    bool isTrajectoryDone() const { return trajectory_done; }
    float getBusVoltage() const { return bus_voltage; }
    float getBusCurrent() const { return bus_current; }
    float getFetTemperature() const { return fet_temp; }
    float getMotorTemperature() const { return motor_temp; }
    float getIqSetpoint() const { return iq_setpoint; }
    float getIqMeasured() const { return iq_measured; }
    
    // Axis state helpers
    bool isIdle() const { return axis_state == ODRIVE_AXIS_STATE_IDLE; }
    bool isClosedLoop() const { return axis_state == ODRIVE_AXIS_STATE_CLOSED_LOOP_CONTROL; }
    bool isCalibrating() const { 
        return axis_state == ODRIVE_AXIS_STATE_FULL_CALIBRATION ||
               axis_state == ODRIVE_AXIS_STATE_MOTOR_CALIBRATION ||
               axis_state == ODRIVE_AXIS_STATE_ENCODER_OFFSET_CALIB;
    }
};

#endif // ODRIVE_MOTOR_H
