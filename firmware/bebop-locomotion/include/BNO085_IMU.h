/**
 * @file BNO085_IMU.h
 * @brief BNO085 IMU driver for Teensy using SPI
 *
 * ## Why SPI (was I2C)
 *
 * Earlier revisions of this driver talked to the BNO over I2C. That
 * worked on the bench but locked up near the leg motors: the brushless
 * drive currents bias the BNO's magnetometer enough that its fusion
 * filter rejects all subsequent mag updates, freezing yaw indefinitely.
 * We migrate to SPI (dedicated INT line for low-latency reads) and keep
 * the AR/VR-Stabilized Rotation Vector (report 0x28), whose fusion
 * pipeline aggressively filters magnetometer disturbances. This mirrors
 * the Linux runtime (`firmware/bebop-linux/src/imu.rs`, report 0x28 over
 * SPI) and the validated standalone test (`src/main_imu_spi_test.cpp`).
 *
 * ## Wiring (Adafruit BNO085 breakout -> Teensy 4.1, hardware SPI0)
 *   Board SCL  -> Pin 13  (SCK)   *** also the onboard LED ***
 *   Board DI   -> Pin 11  (MOSI / data INTO sensor)
 *   Board SDA  -> Pin 12  (MISO / data OUT of sensor; SDA = MISO in SPI)
 *   Board CS   -> Pin 10  (chip select)
 *   Board INT  -> Pin 37  (REQUIRED for SPI - data-ready / host-interrupt)
 *   Board RST  -> Pin 36  (reset; passed to the driver constructor)
 *   Board P0   -> 3Vo     (HIGH -> SPI mode)
 *   Board P1   -> 3Vo     (HIGH -> SPI mode)
 *   Board VIN  -> 3.3V
 *   Board GND  -> GND
 *
 * IMPORTANT - BNO085 mode straps for SPI (Adafruit board pins P0 / P1):
 *   P1 = 1, P0 = 1 (both tied HIGH) -> SPI mode. (P1=0, P0=0 is I2C.)
 *
 * NOTE - Pin 13 conflict: SCK shares the Teensy onboard LED. Any code
 * that drives pin 13 as a status LED (e.g. a heartbeat in main.cpp) will
 * fight the SPI clock. Move the status LED to another pin when this
 * driver is active.
 */

#ifndef BNO085_IMU_H
#define BNO085_IMU_H

#include <Arduino.h>
#include <SPI.h>
#include <Adafruit_BNO08x.h>

// SPI chip select.
#define BNO085_CS_PIN 10

// Interrupt pin (REQUIRED in SPI mode; the SH-2 SPI HAL waits on it to
// know when the sensor has data and when it is ready after reset).
#define BNO085_INT_PIN 37

// Reset pin (passed to the Adafruit_BNO08x constructor).
#define BNO085_RST_PIN 36

// 5ms -> 200Hz requested report rate, matching main_imu_spi_test.cpp.
// (The Linux runtime still uses its own default report period.)
#define BNO085_REPORT_INTERVAL_US 5000

// Minimum spacing between report re-enable attempts after a sensor reset.
// Bounds how often update() makes blocking SH-2 round-trips when the chip
// is boot-looping, so it can never starve a fixed-rate caller loop.
#define BNO085_REENABLE_THROTTLE_MS 250

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
    // Diagnostics / non-blocking reset handling.
    uint32_t reset_count = 0;        // total SH-2 resets observed since boot
    uint32_t last_reenable_ms = 0;   // throttles report re-enable on reset
    
    bool begin() {
        // Bring up the SPI bus. The Adafruit driver pulses RST (passed to
        // the constructor) and runs the SHTP boot handshake inside
        // begin_SPI(), so we don't toggle RST by hand here.
        SPI.begin();
        delay(50);

        if (!bno.begin_SPI(BNO085_CS_PIN, BNO085_INT_PIN, &SPI)) {
            SerialUSB1.println("IMU: Failed to initialize SPI (check PS1=PS0=1, INT/RST wiring, power)");
            return false;
        }

        delay(100);  // Allow sensor to stabilize before enabling reports
        
        // Enable reports with retry logic
        bool rv_ok = false, gyro_ok = false, accel_ok = false;
        
        for (int attempt = 0; attempt < 3; attempt++) {
            if (!rv_ok && bno.enableReport(SH2_ARVR_STABILIZED_RV, BNO085_REPORT_INTERVAL_US)) {
                rv_ok = true;
            }
            if (!gyro_ok && bno.enableReport(SH2_GYROSCOPE_CALIBRATED, BNO085_REPORT_INTERVAL_US)) {
                gyro_ok = true;
            }
            if (!accel_ok && bno.enableReport(SH2_LINEAR_ACCELERATION, BNO085_REPORT_INTERVAL_US)) {
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
        SerialUSB1.println("IMU: Initialized successfully (SPI, AR/VR RV 0x28)");
        return true;
    }
    
    bool update() {
        if (!initialized) return false;

        // INT-gate the read. getSensorEvent() -> sh2_service() unconditionally
        // calls the SPI HAL read, which busy-waits up to 500ms on the INT line
        // (500 x delay(1)) and then *hardware-resets* the chip if no packet
        // arrives. Polling it while the FIFO is empty therefore stalls the
        // caller's loop to a few Hz. The BNO085 drives INT LOW only when a
        // report is ready, so if INT is high there is nothing to service:
        // return immediately and let the caller keep its loop/emit cadence.
        if (digitalRead(BNO085_INT_PIN) != LOW) {
            return false;
        }

        // Check if sensor was reset. A flapping/boot-looping sensor sets
        // this flag continuously; re-enabling reports here involves blocking
        // SH-2 round-trips, so doing it on *every* reset event stalls the
        // caller's loop (and starves any fixed-rate emit cadence built on top
        // of update()). Keep this path non-blocking: never delay(), and
        // throttle the report re-enable so a wedged chip can't monopolize the
        // loop. A healthy reset re-enables within REENABLE_THROTTLE_MS anyway.
        if (bno.wasReset()) {
            reset_count++;
            uint32_t now = millis();
            if ((now - last_reenable_ms) >= BNO085_REENABLE_THROTTLE_MS) {
                last_reenable_ms = now;
                bno.enableReport(SH2_ARVR_STABILIZED_RV, BNO085_REPORT_INTERVAL_US);
                bno.enableReport(SH2_GYROSCOPE_CALIBRATED, BNO085_REPORT_INTERVAL_US);
                bno.enableReport(SH2_LINEAR_ACCELERATION, BNO085_REPORT_INTERVAL_US);
            }
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
    
    // Check for stale data and attempt recovery by re-running the SPI
    // bring-up (begin_SPI internally pulses RST and re-does the SHTP
    // handshake). Call this periodically (e.g. every loop or every
    // second). Returns true if a recovery was triggered.
    bool checkAndRecover(uint32_t stale_threshold_ms = 2000) {
        if (!initialized) return false;
        
        uint32_t age = getUpdateAge();
        uint32_t now = millis();
        
        // Prevent reset spam - minimum 5 seconds between recovery attempts
        if ((now - last_recovery_attempt_ms) < 5000) {
            return false;
        }
        
        if (age > stale_threshold_ms) {
            SerialUSB1.printf("IMU: Data stale (%lums), attempting SPI re-init...\n", age);
            last_recovery_attempt_ms = now;

            // Re-run the SPI bring-up. begin_SPI() pulses the RST line
            // (passed to the constructor) and walks the SHTP boot
            // handshake again, giving a wedged chip a clean restart
            // without any I2C bus teardown.
            if (!bno.begin_SPI(BNO085_CS_PIN, BNO085_INT_PIN, &SPI)) {
                SerialUSB1.println("IMU: Recovery failed - begin_SPI() failed");
                return true;
            }

            delay(100);

            // Re-enable reports
            bno.enableReport(SH2_ARVR_STABILIZED_RV, BNO085_REPORT_INTERVAL_US);
            bno.enableReport(SH2_GYROSCOPE_CALIBRATED, BNO085_REPORT_INTERVAL_US);
            bno.enableReport(SH2_LINEAR_ACCELERATION, BNO085_REPORT_INTERVAL_US);

            SerialUSB1.println("IMU: SPI re-init complete, reports re-enabled");
            return true;
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
    // Reset pin is supplied to the constructor; CS/INT go to begin_SPI().
    Adafruit_BNO08x bno{BNO085_RST_PIN};
    
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
