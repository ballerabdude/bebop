/**
 * @file main_imu_test.cpp
 * @brief Standalone BNO085 IMU test (I2C) for Teensy 4.1
 *
 * Purpose:
 *   Minimal, self-contained program to verify a BNO085 over I2C before
 *   integrating it into the locomotion firmware. All output goes to the
 *   primary USB Serial port so it can be monitored with:
 *
 *       pio run -e imu_test --target upload
 *       pio device monitor -e imu_test
 *
 * Wiring (Teensy 4.1, I2C0 / Wire):
 *   - SCL: Pin 19
 *   - SDA: Pin 18
 *   - INT: Pin 37   (optional, not required for this test)
 *   - RST: Pin 36   (optional, set RST_PIN to -1 if unused)
 *   - VCC: 3.3V
 *   - GND: GND
 *
 * What it does:
 *   1. Scans the I2C bus and lists detected addresses.
 *   2. Hardware-resets the BNO085 (if RST_PIN >= 0).
 *   3. Initializes the sensor and enables rotation vector, gyro, accel.
 *   4. Continuously prints quaternion, gyro, accel, projected gravity,
 *      and a per-second report rate so you can confirm live data.
 */

#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_BNO08x.h>

// ============================================================================
// CONFIGURATION
// ============================================================================

#define IMU_I2C_ADDR  0x4A     // BNO085 default; some bots use 0x4B
#define IMU_INT_PIN   37       // optional
#define IMU_RST_PIN   36       // optional, set to -1 if not connected
#define I2C_CLOCK_HZ  100000   // 100kHz is the most reliable for BNO085
#define REPORT_INTERVAL_US 10000  // 10ms -> 100Hz requested report rate

// Use the primary USB serial for all output (easy monitoring).
#define OUT Serial

// ============================================================================
// GLOBALS
// ============================================================================

Adafruit_BNO08x bno(IMU_RST_PIN);
sh2_SensorValue_t sensorValue;

// Latest values
float quat_w = 1, quat_x = 0, quat_y = 0, quat_z = 0;
float gyro_x = 0, gyro_y = 0, gyro_z = 0;
float accel_x = 0, accel_y = 0, accel_z = 0;
float gravity_x = 0, gravity_y = 0, gravity_z = -1;

// Stats
uint32_t rv_count = 0, gyro_count = 0, accel_count = 0;
uint32_t last_stats_ms = 0;
uint32_t last_print_ms = 0;

// ============================================================================
// HELPERS
// ============================================================================

// Scan a single I2C bus and report any devices that ACK.
uint8_t scanBus(TwoWire &wire, const char *name, uint8_t sda, uint8_t scl) {
    wire.begin();
    wire.setClock(I2C_CLOCK_HZ);
    delay(10);

    OUT.printf("[I2C] Scanning %s (SDA=%u, SCL=%u)...\n", name, sda, scl);
    uint8_t found = 0;
    for (uint8_t addr = 1; addr < 127; addr++) {
        wire.beginTransmission(addr);
        uint8_t err = wire.endTransmission();
        if (err == 0) {
            OUT.printf("[I2C]   Found device at 0x%02X%s\n", addr,
                       (addr == IMU_I2C_ADDR || addr == 0x4B)
                           ? "  <- possible BNO085"
                           : "");
            found++;
        }
    }
    if (found == 0) {
        OUT.printf("[I2C]   %s: no devices.\n", name);
    } else {
        OUT.printf("[I2C]   %s: %u device(s) found.\n", name, found);
    }
    return found;
}

// Scan every I2C bus available on the Teensy 4.1 so we can discover which
// pins the sensor is actually wired to (a common cause of "no devices").
void i2cScanAll() {
    OUT.println(F("\n[I2C] Scanning all Teensy 4.1 I2C buses..."));
    uint8_t total = 0;
    total += scanBus(Wire,  "Wire",  18, 19);
    total += scanBus(Wire1, "Wire1", 17, 16);
    total += scanBus(Wire2, "Wire2", 25, 24);
    if (total == 0) {
        OUT.println(F("[I2C] NO devices on ANY bus."));
        OUT.println(F("[I2C]   -> Check 3.3V + GND to the sensor."));
        OUT.println(F("[I2C]   -> Check SDA/SCL not swapped, wired to scanned pins."));
        OUT.println(F("[I2C]   -> Need pull-ups on SDA/SCL (most breakouts have them)."));
        OUT.println(F("[I2C]   -> Confirm board is in I2C mode (PS0/PS1 pins),"));
        OUT.println(F("[I2C]      not SPI/UART. SPI mode will NOT answer on I2C."));
    }
    // Leave the primary bus initialized for the driver below.
    Wire.begin();
    Wire.setClock(I2C_CLOCK_HZ);
    delay(50);
}

// Project gravity into body frame: R^T * [0,0,-1]
// (matches IsaacLab quat_rotate_inverse(q, [0,0,-1]))
void computeProjectedGravity() {
    float w = quat_w, x = quat_x, y = quat_y, z = quat_z;
    gravity_x = 2.0f * (w * y - x * z);
    gravity_y = -2.0f * (w * x + y * z);
    gravity_z = -(w * w - x * x - y * y + z * z);
}

void enableReports() {
    OUT.println(F("[IMU] Enabling reports..."));
    if (!bno.enableReport(SH2_ARVR_STABILIZED_RV, REPORT_INTERVAL_US)) {
        OUT.println(F("[IMU]   FAILED to enable Rotation Vector"));
    }
    if (!bno.enableReport(SH2_GYROSCOPE_CALIBRATED, REPORT_INTERVAL_US)) {
        OUT.println(F("[IMU]   FAILED to enable Gyroscope"));
    }
    if (!bno.enableReport(SH2_LINEAR_ACCELERATION, REPORT_INTERVAL_US)) {
        OUT.println(F("[IMU]   FAILED to enable Linear Acceleration"));
    }
}

bool initIMU() {
    // Hardware reset (Adafruit lib also toggles RST, but be explicit).
    if (IMU_RST_PIN >= 0) {
        pinMode(IMU_RST_PIN, OUTPUT);
        digitalWrite(IMU_RST_PIN, LOW);
        delay(10);
        digitalWrite(IMU_RST_PIN, HIGH);
        delay(300);  // BNO085 needs ~300ms after reset
    }

    i2cScanAll();

    OUT.printf("[IMU] Initializing BNO085 at 0x%02X...\n", IMU_I2C_ADDR);
    if (!bno.begin_I2C(IMU_I2C_ADDR, &Wire)) {
        OUT.println(F("[IMU] begin_I2C() FAILED."));
        return false;
    }
    OUT.println(F("[IMU] begin_I2C() OK."));

    // Print product/version info if available.
    for (int i = 0; i < bno.prodIds.numEntries; i++) {
        OUT.printf("[IMU]   Part %lu : SW %u.%u.%lu\n",
                   (unsigned long)bno.prodIds.entry[i].swPartNumber,
                   bno.prodIds.entry[i].swVersionMajor,
                   bno.prodIds.entry[i].swVersionMinor,
                   (unsigned long)bno.prodIds.entry[i].swBuildNumber);
    }

    delay(100);
    enableReports();
    return true;
}

// ============================================================================
// ARDUINO ENTRY POINTS
// ============================================================================

void setup() {
    pinMode(LED_BUILTIN, OUTPUT);
    digitalWrite(LED_BUILTIN, HIGH);

    OUT.begin(115200);
    uint32_t start = millis();
    while (!OUT && (millis() - start) < 3000) {
        // wait up to 3s for USB serial
    }

    OUT.println(F("\n================================"));
    OUT.println(F("  BNO085 I2C IMU Test (Teensy)"));
    OUT.println(F("================================"));

    if (!initIMU()) {
        OUT.println(F("[IMU] Initialization failed. Halting (will retry in 3s)."));
    }

    last_stats_ms = millis();
    last_print_ms = millis();
}

void loop() {
    // If the sensor reset itself, re-enable reports.
    if (bno.wasReset()) {
        OUT.println(F("[IMU] Sensor reset detected, re-enabling reports..."));
        delay(50);
        enableReports();
    }

    // Drain all available events this loop.
    while (bno.getSensorEvent(&sensorValue)) {
        switch (sensorValue.sensorId) {
            case SH2_ARVR_STABILIZED_RV:
                quat_w = sensorValue.un.arvrStabilizedRV.real;
                quat_x = sensorValue.un.arvrStabilizedRV.i;
                quat_y = sensorValue.un.arvrStabilizedRV.j;
                quat_z = sensorValue.un.arvrStabilizedRV.k;
                computeProjectedGravity();
                rv_count++;
                break;
            case SH2_GYROSCOPE_CALIBRATED:
                gyro_x = sensorValue.un.gyroscope.x;
                gyro_y = sensorValue.un.gyroscope.y;
                gyro_z = sensorValue.un.gyroscope.z;
                gyro_count++;
                break;
            case SH2_LINEAR_ACCELERATION:
                accel_x = sensorValue.un.linearAcceleration.x;
                accel_y = sensorValue.un.linearAcceleration.y;
                accel_z = sensorValue.un.linearAcceleration.z;
                accel_count++;
                break;
        }
    }

    uint32_t now = millis();

    // Print live values at ~10Hz.
    if (now - last_print_ms >= 100) {
        last_print_ms = now;
        OUT.printf(
            "quat[w,x,y,z]=[% .3f % .3f % .3f % .3f]  "
            "gyro=[% .2f % .2f % .2f]  "
            "accel=[% .2f % .2f % .2f]  "
            "grav=[% .2f % .2f % .2f]\n",
            quat_w, quat_x, quat_y, quat_z,
            gyro_x, gyro_y, gyro_z,
            accel_x, accel_y, accel_z,
            gravity_x, gravity_y, gravity_z);
    }

    // Print report rates once per second.
    if (now - last_stats_ms >= 1000) {
        float dt = (now - last_stats_ms) / 1000.0f;
        OUT.printf("[RATE] rv=%.0fHz gyro=%.0fHz accel=%.0fHz\n",
                   rv_count / dt, gyro_count / dt, accel_count / dt);
        rv_count = gyro_count = accel_count = 0;
        last_stats_ms = now;

        // Heartbeat LED.
        digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN));
    }
}
