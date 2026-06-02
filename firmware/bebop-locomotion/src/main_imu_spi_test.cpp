/**
 * @file main_imu_spi_test.cpp
 * @brief Standalone BNO085 IMU test (SPI) for Teensy 4.1
 *
 * Purpose:
 *   Minimal, self-contained program to verify a BNO085 over SPI before
 *   deploying the SPI path on the Jetson. All output goes to the primary
 *   USB Serial port:
 *
 *       pio run -e imu_spi_test --target upload
 *       pio device monitor -e imu_spi_test    (or: cat /dev/ttyACM0)
 *
 * IMPORTANT - BNO085 mode straps for SPI (Adafruit board pins P0 / P1):
 *   P1 = 1, P0 = 1   (both tied to 3Vo / HIGH)  -> SPI mode
 *   (For reference: P1=0, P0=0 is I2C.)
 *
 * Wiring - Adafruit BNO085 breakout silk labels -> Teensy 4.1 (hardware SPI0):
 *   Board SCL  -> Pin 13  (SCK)
 *   Board DI   -> Pin 11  (MOSI / data INTO sensor)
 *   Board SDA  -> Pin 12  (MISO / data OUT of sensor; SDA = MISO in SPI mode)
 *   Board CS   -> Pin 10  (chip select)
 *   Board INT  -> Pin 37  (REQUIRED for SPI - data-ready / host-interrupt)
 *   Board RST  -> Pin 36  (reset, passed to driver constructor)
 *   Board P0   -> 3Vo     (HIGH -> SPI mode)
 *   Board P1   -> 3Vo     (HIGH -> SPI mode)
 *   Board VIN  -> 3.3V
 *   Board GND  -> GND
 *   Board BT   -> leave unconnected
 *
 * Notes:
 *   - On the Adafruit breakout the I2C pins are reused for SPI: SCL=SCK,
 *     SDA=MISO, and the dedicated DI pin is MOSI.
 *   - INT is mandatory in SPI mode; the SH2 SPI HAL polls/waits on it to know
 *     when the sensor has data and when it is ready after reset.
 *   - Pin 13 is also the onboard LED, so we do NOT use the LED heartbeat here
 *     (it would conflict with SCK). Watch the serial output instead.
 */

#include <Arduino.h>
#include <SPI.h>
#include <Adafruit_BNO08x.h>

// ============================================================================
// CONFIGURATION
// ============================================================================

#define IMU_CS_PIN    10       // SPI chip select
#define IMU_INT_PIN   37       // REQUIRED for SPI
#define IMU_RST_PIN   36       // reset (constructor)
#define REPORT_INTERVAL_US 5000   // 5ms -> 200Hz requested report rate

// Use the primary USB serial for all output.
#define OUT Serial

// ============================================================================
// GLOBALS
// ============================================================================

// Reset pin is supplied to the constructor; CS/INT go to begin_SPI().
Adafruit_BNO08x bno(IMU_RST_PIN);
sh2_SensorValue_t sensorValue;

float quat_w = 1, quat_x = 0, quat_y = 0, quat_z = 0;
float gyro_x = 0, gyro_y = 0, gyro_z = 0;
float accel_x = 0, accel_y = 0, accel_z = 0;
float gravity_x = 0, gravity_y = 0, gravity_z = -1;

uint32_t rv_count = 0, gyro_count = 0, accel_count = 0;
uint32_t last_stats_ms = 0;
uint32_t last_print_ms = 0;

// Lifetime counters / diagnostics (not reset every second).
uint32_t loop_count = 0;          // how many times loop() has run
uint32_t event_count = 0;         // total sensor events ever drained
uint32_t reset_count = 0;         // total wasReset() events
uint32_t last_event_ms = 0;       // millis() of the most recent sensor event

// Cap events drained per loop() pass so we always reach the print/heartbeat
// code even if the sensor floods us (avoids getting "stuck" draining events).
#define MAX_EVENTS_PER_LOOP 64

// ============================================================================
// HELPERS
// ============================================================================

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

// Print the raw INT pin state a few times. In SPI mode the BNO085 drives INT
// low when it has data to send / is ready. If INT never moves, the sensor is
// likely unpowered, not in SPI mode (PS1/PS0), or INT is miswired.
void probeIntPin() {
    pinMode(IMU_INT_PIN, INPUT_PULLUP);
    OUT.printf("[SPI] Sampling INT pin (%d) for 500ms...\n", IMU_INT_PIN);
    int lows = 0, highs = 0;
    uint32_t t0 = millis();
    while (millis() - t0 < 500) {
        if (digitalRead(IMU_INT_PIN) == LOW) lows++; else highs++;
        delayMicroseconds(200);
    }
    OUT.printf("[SPI]   INT samples: LOW=%d HIGH=%d %s\n", lows, highs,
               (lows == 0) ? "(INT never asserted - check power/mode/wiring)"
                           : "(INT toggling - good sign)");
}

bool initIMU() {
    SPI.begin();
    delay(50);

    probeIntPin();

    OUT.printf("[IMU] Initializing BNO085 over SPI (CS=%d, INT=%d, RST=%d)...\n",
               IMU_CS_PIN, IMU_INT_PIN, IMU_RST_PIN);

    if (!bno.begin_SPI(IMU_CS_PIN, IMU_INT_PIN, &SPI)) {
        OUT.println(F("[IMU] begin_SPI() FAILED."));
        OUT.println(F("[IMU]   -> Confirm PS1=1, PS0=1 (SPI mode)."));
        OUT.println(F("[IMU]   -> Confirm SCK=13, MOSI=11, MISO=12, CS, INT, RST."));
        OUT.println(F("[IMU]   -> Confirm 3.3V power and common ground."));
        return false;
    }
    OUT.println(F("[IMU] begin_SPI() OK."));

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
    OUT.begin(115200);
    uint32_t start = millis();
    while (!OUT && (millis() - start) < 3000) {
        // wait up to 3s for USB serial
    }

    OUT.println(F("\n================================"));
    OUT.println(F("  BNO085 SPI IMU Test (Teensy)"));
    OUT.println(F("================================"));

    if (!initIMU()) {
        OUT.println(F("[IMU] Initialization failed."));
    }

    uint32_t now = millis();
    last_stats_ms = now;
    last_print_ms = now;
    last_event_ms = now;
}

void loop() {
    loop_count++;

    if (bno.wasReset()) {
        reset_count++;
        OUT.printf("[IMU] Sensor reset detected (#%lu), re-enabling reports...\n",
                   (unsigned long)reset_count);
        delay(50);
        enableReports();
    }

    // Drain a bounded number of events so we always fall through to the
    // heartbeat/print code below, even if the sensor is flooding us.
    int drained = 0;
    while (drained < MAX_EVENTS_PER_LOOP && bno.getSensorEvent(&sensorValue)) {
        drained++;
        event_count++;
        last_event_ms = millis();
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

    if (now - last_print_ms >= 100) {
        last_print_ms = now;
        uint32_t stale = now - last_event_ms;
        OUT.printf(
            "[DATA] quat[w,x,y,z]=[% .3f % .3f % .3f % .3f]  "
            "gyro=[% .2f % .2f % .2f]  "
            "accel=[% .2f % .2f % .2f]  "
            "grav=[% .2f % .2f % .2f]  "
            "evts=%lu age=%lums%s\n",
            quat_w, quat_x, quat_y, quat_z,
            gyro_x, gyro_y, gyro_z,
            accel_x, accel_y, accel_z,
            gravity_x, gravity_y, gravity_z,
            (unsigned long)event_count, (unsigned long)stale,
            (stale > 500) ? "  <-- NO RECENT DATA" : "");
    }

    if (now - last_stats_ms >= 1000) {
        float dt = (now - last_stats_ms) / 1000.0f;
        OUT.printf("[RATE] rv=%.0fHz gyro=%.0fHz accel=%.0fHz  "
                   "loops/s=%.0f totalEvts=%lu resets=%lu\n",
                   rv_count / dt, gyro_count / dt, accel_count / dt,
                   loop_count / dt,
                   (unsigned long)event_count, (unsigned long)reset_count);
        rv_count = gyro_count = accel_count = 0;
        loop_count = 0;
        last_stats_ms = now;
    }
}
