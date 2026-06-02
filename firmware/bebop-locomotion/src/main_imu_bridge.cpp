/**
 * @file main_imu_bridge.cpp
 * @brief BNO085 IMU -> USB serial bridge for Teensy 4.1
 *
 * Purpose:
 *   Read the BNO085 over SPI (using the shared `BNO085_IMU` driver) and
 *   stream fused orientation + calibrated gyro + linear accel to the host
 *   (Jetson) over USB serial as fixed-size binary frames. This lets the
 *   Jetson consume the IMU through the Teensy instead of wiring the BNO
 *   to the Jetson's own SPI bus.
 *
 *       pio run -e imu_bridge --target upload
 *
 * Ports (USB_DUAL_SERIAL):
 *   Serial      (primary, /dev/ttyACM0) -> BINARY IMU frames only
 *   SerialUSB1  (secondary, /dev/ttyACM1) -> human-readable diagnostics
 *
 *   Keeping binary frames on a dedicated channel means the Jetson parser
 *   never has to filter out log lines, and you can still `cat` the debug
 *   port to watch rates while the bridge runs.
 *
 * Wiring: identical to `main_imu_spi_test.cpp` / `BNO085_IMU.h`
 *   (BNO over hardware SPI0, INT=37, RST=36, CS=10, PS1=PS0=1 for SPI mode).
 *
 * Frame format: see `include/ImuSerialProtocol.h`. The Jetson-side parser
 * lives in `firmware/bebop-linux/src/imu_serial.rs`.
 *
 * NOTE: the Teensy streams the RAW sensor-frame quaternion / gyro. The
 * chassis mount rotation is applied on the Jetson (mirroring the SPI
 * path in `imu.rs`) so the two IMU sources are interchangeable.
 */

#include <Arduino.h>
#include "BNO085_IMU.h"
#include "ImuSerialProtocol.h"

// 200 Hz emit cadence (5 ms). Decoupled from sensor event arrival: we
// always send the latest cached values, so the host sees a steady stream.
#define EMIT_INTERVAL_US 5000

// Cap sensor events drained per loop so a flood (or a wedged FIFO) can't
// monopolize the loop and starve the emit cadence below.
#define MAX_EVENTS_PER_LOOP 64

// Primary USB serial carries the binary frames; SerialUSB1 is debug.
#define FRAME_PORT Serial
#define DBG_PORT   SerialUSB1

static_assert(sizeof(ImuSerialFrame) == IMU_SERIAL_FRAME_SIZE,
              "ImuSerialFrame must be 52 bytes (packed) to match the wire format");

BNO085_IMU imu;

uint32_t seq = 0;
uint32_t last_emit_us = 0;
uint32_t last_stats_ms = 0;
uint32_t frames_sent = 0;

void emitFrame() {
    ImuSerialFrame f;
    f.seq = seq++;
    f.t_us = micros();

    // BNO085_IMU stores the quaternion as (w, x, y, z); the wire format is
    // XYZW (scalar last) to match the Jetson's ImuSnapshot contract.
    f.quat_xyzw[0] = imu.quat_x;
    f.quat_xyzw[1] = imu.quat_y;
    f.quat_xyzw[2] = imu.quat_z;
    f.quat_xyzw[3] = imu.quat_w;

    f.gyro_xyz[0] = imu.gyro_x;
    f.gyro_xyz[1] = imu.gyro_y;
    f.gyro_xyz[2] = imu.gyro_z;

    f.accel_xyz[0] = imu.accel_x;
    f.accel_xyz[1] = imu.accel_y;
    f.accel_xyz[2] = imu.accel_z;

    imu_serial_finalize(&f);

    FRAME_PORT.write((const uint8_t*)&f, sizeof(f));
    frames_sent++;
}

void setup() {
    FRAME_PORT.begin(115200);   // baud is ignored for USB CDC; full USB speed
    DBG_PORT.begin(115200);

    uint32_t start = millis();
    while (!DBG_PORT && (millis() - start) < 2000) {
        // brief wait for the debug port; don't block the frame stream forever
    }

    DBG_PORT.println(F("\n===================================="));
    DBG_PORT.println(F("  BNO085 -> USB Serial IMU Bridge"));
    DBG_PORT.println(F("===================================="));
    DBG_PORT.printf("[BRIDGE] Frame size: %u bytes, emit @ %u Hz\n",
                    (unsigned)sizeof(ImuSerialFrame), 1000000u / EMIT_INTERVAL_US);

    if (!imu.begin()) {
        DBG_PORT.println(F("[BRIDGE] IMU init FAILED (will keep retrying via recovery)"));
    } else {
        DBG_PORT.println(F("[BRIDGE] IMU init OK; streaming binary frames on primary Serial"));
    }

    uint32_t now = millis();
    last_stats_ms = now;
    last_emit_us = micros();
}

void loop() {
    // 1) EMIT FIRST, on cadence. The host relies on a steady stream, so the
    //    frame stream is the top priority: emitting before any IMU servicing
    //    guarantees a slow/blocking IMU op (event drain, report re-enable on
    //    reset, or a stale-recovery begin_SPI) below can never delay a frame
    //    that is already due this cycle.
    uint32_t now_us = micros();
    if ((uint32_t)(now_us - last_emit_us) >= EMIT_INTERVAL_US) {
        last_emit_us += EMIT_INTERVAL_US;
        // If we fell badly behind (e.g. a blocking re-init ate several ms),
        // don't burst to "catch up" — resync the cadence to now.
        if ((uint32_t)(now_us - last_emit_us) > EMIT_INTERVAL_US) {
            last_emit_us = now_us;
        }
        emitFrame();
    }

    // 2) Drain a BOUNDED number of sensor events so the cached quat/gyro/accel
    //    stay fresh without letting a flood starve the emit cadence above.
    //    update() handles one event per call and re-enables reports on sensor
    //    reset internally.
    int drained = 0;
    while (drained < MAX_EVENTS_PER_LOOP && imu.update()) {
        drained++;
    }

    // 3) Rate-limited stale recovery (the driver enforces >=5s between
    //    attempts). begin_SPI() can block briefly; we already emitted this
    //    cycle and the cadence resync in (1) absorbs the gap on the next pass.
    imu.checkAndRecover();

    uint32_t now_ms = millis();
    if (now_ms - last_stats_ms >= 1000) {
        float dt = (now_ms - last_stats_ms) / 1000.0f;
        DBG_PORT.printf("[BRIDGE] tx=%.0fHz init=%d resets=%lu age=%lums quat=[% .3f % .3f % .3f % .3f] gyro=[% .2f % .2f % .2f]\n",
                        frames_sent / dt, imu.initialized ? 1 : 0,
                        (unsigned long)imu.reset_count, (unsigned long)imu.getUpdateAge(),
                        imu.quat_w, imu.quat_x, imu.quat_y, imu.quat_z,
                        imu.gyro_x, imu.gyro_y, imu.gyro_z);
        frames_sent = 0;
        last_stats_ms = now_ms;
    }
}
