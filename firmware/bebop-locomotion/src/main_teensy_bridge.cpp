/**
 * @file main_teensy_bridge.cpp
 * @brief BNO085 IMU + physical controls -> USB serial bridge for Teensy 4.1
 *
 * Purpose:
 *   Read the BNO085 over SPI (using the shared `BNO085_IMU` driver) and
 *   stream fused orientation + calibrated gyro + linear accel to the host
 *   (Jetson) over USB serial as fixed-size binary frames. This lets the
 *   Jetson consume the IMU through the Teensy instead of wiring the BNO
 *   to the Jetson's own SPI bus.
 *
 *   Also acts as the robot's control input: the Thor has no usable GPIO
 *   header, so buttons (network-mode toggle today, E-stop and more later)
 *   are wired here and forwarded to the Jetson on a dedicated channel.
 *
 *       pio run -e teensy_bridge --target upload
 *
 * Ports (USB_TRIPLE_SERIAL):
 *   Serial      (primary, if00, /dev/bebop-imu)       -> BINARY IMU frames
 *   SerialUSB1  (secondary, if02, /dev/bebop-imu-debug) -> diagnostics
 *   SerialUSB2  (tertiary, if04, /dev/bebop-control)  -> control frames
 *
 *   Keeping binary frames on a dedicated channel means the Jetson parser
 *   never has to filter out log lines, and you can still `cat` the debug
 *   port to watch rates while the bridge runs. The control channel carries
 *   button/event frames (see `include/ControlProtocol.h`) so the Jetson can
 *   read physical controls without a GPIO header (Thor has none) — the
 *   network-mode toggle today, emergency stop and more later.
 *
 * Wiring: identical to `main_imu_spi_test.cpp` / `BNO085_IMU.h`
 *   (BNO over hardware SPI0, INT=37, RST=36, CS=10, PS1=PS0=1 for SPI mode).
 *
 * Button wiring (network-mode toggle): a momentary switch from `BUTTON_PIN`
 *   to GND. The pin uses the Teensy's internal pull-up, so no external
 *   resistor is needed (idle high, pressed low). It is debounced here and
 *   reported on the control port; the Jetson (`bebop-agent`) decides what a
 *   press does.
 *
 * Frame format: see `include/ImuSerialProtocol.h` (IMU) and
 * `include/ControlProtocol.h` (control). The Jetson-side parsers live in
 * `firmware/bebop-linux/src/imu_serial.rs` and
 * `jetson-agent/bebop-agent/src/control_serial.rs`.
 *
 * NOTE: the Teensy streams the RAW sensor-frame quaternion / gyro. The
 * chassis mount rotation is applied on the Jetson (mirroring the SPI
 * path in `imu.rs`) so the two IMU sources are interchangeable.
 */

#include <Arduino.h>
#include "BNO085_IMU.h"
#include "ImuSerialProtocol.h"
#include "ControlProtocol.h"

// 200 Hz emit cadence (5 ms). Decoupled from sensor event arrival: we
// always send the latest cached values, so the host sees a steady stream.
#define EMIT_INTERVAL_US 5000

// Cap sensor events drained per loop so a flood (or a wedged FIFO) can't
// monopolize the loop and starve the emit cadence below.
#define MAX_EVENTS_PER_LOOP 64

// Primary USB serial carries the binary IMU frames; SerialUSB1 is debug.
#define FRAME_PORT Serial
#define DBG_PORT   SerialUSB1
// Control channel (buttons/events) on the third CDC.
#define CTRL_PORT  SerialUSB2

// Network-mode toggle button: switch to GND, internal pull-up (active low).
#define BUTTON_PIN 2
// A level must hold this long before it is accepted as a real edge.
#define BUTTON_DEBOUNCE_MS 30

static_assert(sizeof(ImuSerialFrame) == IMU_SERIAL_FRAME_SIZE,
              "ImuSerialFrame must be 52 bytes (packed) to match the wire format");
static_assert(sizeof(CtrlFrame) == CTRL_FRAME_SIZE,
              "CtrlFrame must be 14 bytes (packed) to match the wire format");

BNO085_IMU imu;

uint32_t seq = 0;
uint32_t last_emit_us = 0;
uint32_t last_stats_ms = 0;
uint32_t frames_sent = 0;

// Control-channel state.
uint8_t  ctrl_seq = 0;
uint32_t buttons_pressed = 0;
bool     button_raw = false;       // last raw active-high read
bool     button_stable = false;    // debounced active-high state
uint32_t button_last_edge_ms = 0;

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

// Emit one control frame on the control CDC. Payload is type-specific;
// button frames pack (state << 8) | id.
void emitControl(uint8_t type, uint32_t payload) {
    CtrlFrame f;
    f.type = type;
    f.seq = ctrl_seq++;
    f.t_us = micros();
    f.payload = payload;
    ctrl_finalize(&f);
    CTRL_PORT.write((const uint8_t*)&f, sizeof(f));
}

void emitButton(uint8_t button_id, bool pressed) {
    uint32_t payload = ((uint32_t)(pressed ? CTRL_BUTTON_PRESSED
                                            : CTRL_BUTTON_RELEASED) << 8) | button_id;
    emitControl(CTRL_MSG_BUTTON, payload);
    if (pressed) {
        buttons_pressed++;
    }
}

// Poll the mode button and emit an event only on a debounced level change.
// Non-blocking, so it never disturbs the 200 Hz IMU emit cadence.
void pollButton() {
    bool raw = (digitalRead(BUTTON_PIN) == LOW);  // active-low (pull-up)
    if (raw != button_raw) {
        button_raw = raw;
        button_last_edge_ms = millis();
    }
    if (raw != button_stable && (millis() - button_last_edge_ms) >= BUTTON_DEBOUNCE_MS) {
        button_stable = raw;
        emitButton(CTRL_BUTTON_NETWORK, button_stable);
    }
}

void setup() {
    FRAME_PORT.begin(115200);   // baud is ignored for USB CDC; full USB speed
    DBG_PORT.begin(115200);
    CTRL_PORT.begin(115200);

    // Mode button: internal pull-up, switch to GND when pressed.
    pinMode(BUTTON_PIN, INPUT_PULLUP);
    button_raw = (digitalRead(BUTTON_PIN) == LOW);
    button_stable = button_raw;
    button_last_edge_ms = millis();

    uint32_t start = millis();
    while (!DBG_PORT && (millis() - start) < 2000) {
        // brief wait for the debug port; don't block the frame stream forever
    }

    DBG_PORT.println(F("\n===================================="));
    DBG_PORT.println(F("  Teensy IMU + Control USB Bridge"));
    DBG_PORT.println(F("===================================="));
    DBG_PORT.printf("[BRIDGE] Frame size: %u bytes, emit @ %u Hz\n",
                    (unsigned)sizeof(ImuSerialFrame), 1000000u / EMIT_INTERVAL_US);
    DBG_PORT.printf("[BRIDGE] Control frame: %u bytes on SerialUSB2; button pin %u\n",
                    (unsigned)sizeof(CtrlFrame), (unsigned)BUTTON_PIN);

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

    // 4) Poll the control button; emits only on a debounced edge, so this is
    //    effectively free on cycles with no change.
    pollButton();

    uint32_t now_ms = millis();
    if (now_ms - last_stats_ms >= 1000) {
        float dt = (now_ms - last_stats_ms) / 1000.0f;
        DBG_PORT.printf("[BRIDGE] tx=%.0fHz init=%d resets=%lu age=%lums quat=[% .3f % .3f % .3f % .3f] gyro=[% .2f % .2f % .2f] btn=%lu\n",
                        frames_sent / dt, imu.initialized ? 1 : 0,
                        (unsigned long)imu.reset_count, (unsigned long)imu.getUpdateAge(),
                        imu.quat_w, imu.quat_x, imu.quat_y, imu.quat_z,
                        imu.gyro_x, imu.gyro_y, imu.gyro_z,
                        (unsigned long)buttons_pressed);
        frames_sent = 0;
        last_stats_ms = now_ms;
    }
}
