/**
 * @file BNO085_IMU.h
 * @brief BNO085 IMU driver for Teensy using I2C
 * 
 * Wiring (Teensy 4.1):
 *   - SCL: Pin 19
 *   - SDA: Pin 18
 *   - INT: Pin 36
 *   - RST: Pin 35
 *   - VCC: 3.3V
 *   - GND: GND
 */

#ifndef BNO085_IMU_H
#define BNO085_IMU_H

#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_BNO08x.h>

// I2C address
#define BNO085_I2C_ADDR 0x4A

// Interrupt pin (active low, optional)
#define BNO085_INT_PIN 36

// Reset pin (optional, set to -1 if not used)
#define BNO085_RST_PIN 35

class BNO085_IMU {
public:
    // Orientation quaternion (w, x, y, z)
    float quat_w = 1.0f;
    float quat_x = 0.0f;
    float quat_y = 0.0f;
    float quat_z = 0.0f;
    
    // Angular velocity (rad/s)
    float gyro_x = 0.0f;
    float gyro_y = 0.0f;
    float gyro_z = 0.0f;
    
    // Linear acceleration (m/s^2)
    float accel_x = 0.0f;
    float accel_y = 0.0f;
    float accel_z = 0.0f;
    
    // Projected gravity in body frame
    float gravity_x = 0.0f;
    float gravity_y = 0.0f;
    float gravity_z = -1.0f;
    
    // Status
    bool initialized = false;
    uint32_t last_update_ms = 0;
    uint32_t last_recovery_attempt_ms = 0;
    
    bool begin() {
        // Hardware reset if pin is defined (important for reliable init)
        if (BNO085_RST_PIN >= 0) {
            pinMode(BNO085_RST_PIN, OUTPUT);
            digitalWrite(BNO085_RST_PIN, LOW);
            delay(10);
            digitalWrite(BNO085_RST_PIN, HIGH);
            delay(300);  // BNO085 needs ~300ms after reset
        }
        
        // Initialize I2C
        Wire.begin();
        Wire.setClock(100000);  // 100kHz is more reliable for BNO085
        
        delay(100);  // Allow I2C to stabilize

        if (!bno.begin_I2C(BNO085_I2C_ADDR, &Wire)) {
            SerialUSB1.println("IMU: Failed to initialize I2C");
            return false;
        }

        delay(100);  // Allow sensor to stabilize before enabling reports
        
        // Enable reports with retry logic
        bool rv_ok = false, gyro_ok = false, accel_ok = false;
        
        for (int attempt = 0; attempt < 3; attempt++) {
            if (!rv_ok && bno.enableReport(SH2_ARVR_STABILIZED_RV, 10000)) {
                rv_ok = true;
            }
            if (!gyro_ok && bno.enableReport(SH2_GYROSCOPE_CALIBRATED, 10000)) {
                gyro_ok = true;
            }
            if (!accel_ok && bno.enableReport(SH2_LINEAR_ACCELERATION, 10000)) {
                accel_ok = true;
            }
            
            if (rv_ok && gyro_ok && accel_ok) break;
            delay(50);  // Wait before retry
        }
        
        if (!rv_ok) SerialUSB1.println("IMU: Failed to enable Rotation Vector");
        if (!gyro_ok) SerialUSB1.println("IMU: Failed to enable Gyro");
        if (!accel_ok) SerialUSB1.println("IMU: Failed to enable Accel");
        
        // Only mark as initialized if at least rotation vector works (critical for balance)
        if (!rv_ok) {
            SerialUSB1.println("IMU: CRITICAL - No rotation vector, init failed!");
            return false;
        }
        
        initialized = true;
        SerialUSB1.println("IMU: Initialized successfully");
        return true;
    }
    
    bool update() {
        if (!initialized) return false;
        
        // Check if sensor was reset
        if (bno.wasReset()) {
            SerialUSB1.println("IMU: Sensor reset detected, re-enabling reports...");
            delay(50);
            bno.enableReport(SH2_ARVR_STABILIZED_RV, 10000);
            bno.enableReport(SH2_GYROSCOPE_CALIBRATED, 10000);
            bno.enableReport(SH2_LINEAR_ACCELERATION, 10000);
        }
        
        sh2_SensorValue_t sensorValue;
        
        // Process only ONE event per call (non-blocking)
        if (bno.getSensorEvent(&sensorValue)) {
            switch (sensorValue.sensorId) {
                case SH2_ARVR_STABILIZED_RV:
                    quat_w = sensorValue.un.arvrStabilizedRV.real;
                    quat_x = sensorValue.un.arvrStabilizedRV.i;
                    quat_y = sensorValue.un.arvrStabilizedRV.j;
                    quat_z = sensorValue.un.arvrStabilizedRV.k;
                    
                    // Compute projected gravity
                    computeProjectedGravity();
                    last_update_ms = millis();
                    break;
                    
                case SH2_GYROSCOPE_CALIBRATED:
                    gyro_x = sensorValue.un.gyroscope.x;
                    gyro_y = sensorValue.un.gyroscope.y;
                    gyro_z = sensorValue.un.gyroscope.z;
                    last_update_ms = millis();
                    break;
                    
                case SH2_LINEAR_ACCELERATION:
                    accel_x = sensorValue.un.linearAcceleration.x;
                    accel_y = sensorValue.un.linearAcceleration.y;
                    accel_z = sensorValue.un.linearAcceleration.z;
                    last_update_ms = millis();
                    break;
            }
            return true;
        }
        
        return false;
    }
    
    bool isTimedOut(uint32_t timeout_ms = 100) {
        return (millis() - last_update_ms) > timeout_ms;
    }
    
    // Get age of last update in milliseconds
    uint32_t getUpdateAge() {
        return millis() - last_update_ms;
    }
    
    // Check for stale data and attempt recovery via hardware reset
    // Call this periodically (e.g., every loop or every second)
    // Returns true if a reset was triggered
    bool checkAndRecover(uint32_t stale_threshold_ms = 2000) {
        if (!initialized) return false;
        
        uint32_t age = getUpdateAge();
        uint32_t now = millis();
        
        // Prevent reset spam - minimum 5 seconds between recovery attempts
        if ((now - last_recovery_attempt_ms) < 5000) {
            return false;
        }
        
        if (age > stale_threshold_ms) {
            SerialUSB1.printf("IMU: Data stale (%lums), attempting hardware reset...\n", age);
            last_recovery_attempt_ms = now;
            
            // Hardware reset using RST pin
            if (BNO085_RST_PIN >= 0) {
                digitalWrite(BNO085_RST_PIN, LOW);
                delay(10);
                digitalWrite(BNO085_RST_PIN, HIGH);
                delay(300);  // BNO085 needs ~300ms after reset
                
                // Reinitialize I2C (clear any stuck state)
                Wire.end();
                delay(10);
                Wire.begin();
                Wire.setClock(100000);
                delay(100);
                
                // Reinitialize sensor
                if (!bno.begin_I2C(BNO085_I2C_ADDR, &Wire)) {
                    SerialUSB1.println("IMU: Recovery failed - I2C init failed");
                    return true;
                }
                
                delay(100);
                
                // Re-enable reports
                bno.enableReport(SH2_ARVR_STABILIZED_RV, 10000);
                bno.enableReport(SH2_GYROSCOPE_CALIBRATED, 10000);
                bno.enableReport(SH2_LINEAR_ACCELERATION, 10000);
                
                SerialUSB1.println("IMU: Hardware reset complete, reports re-enabled");
                return true;
            } else {
                SerialUSB1.println("IMU: No reset pin defined, cannot recover!");
            }
        }
        return false;
    }
    
    // Print diagnostic info
    void printDiagnostics() {
        SerialUSB1.printf("IMU: init=%d, age=%lums, quat=[%.2f,%.2f,%.2f,%.2f], gyro=[%.2f,%.2f,%.2f]\n",
            initialized ? 1 : 0,
            getUpdateAge(),
            quat_w, quat_x, quat_y, quat_z,
            gyro_x, gyro_y, gyro_z);
    }
    
private:
    Adafruit_BNO08x bno;
    
    void computeProjectedGravity() {
        // Compute R^T * [0, 0, -1] to get gravity in body frame
        // Matches IsaacLab's quat_rotate_inverse(q, [0,0,-1])
        
        float w = quat_w, x = quat_x, y = quat_y, z = quat_z;
        
        gravity_x = 2.0f * (w * y - x * z);
        gravity_y = -2.0f * (w * x + y * z);
        gravity_z = -(w * w - x * x - y * y + z * z);
    }
};

#endif // BNO085_IMU_H

