/**
 * @file ControlProtocol.h
 * @brief Binary wire format for the Teensy <-> Jetson **control** channel
 *        over USB serial (USB CDC).
 *
 * ## Why a separate channel
 *
 * The Jetson AGX Thor has no exposed GPIO header, so buttons (network-mode
 * toggle today, emergency stop and more later) are wired to the Teensy and
 * forwarded to the Jetson over USB. The `teensy_bridge` firmware uses
 * `USB_TRIPLE_SERIAL`, which presents three independent CDC ports:
 *
 *   Serial      (if00, /dev/bebop-imu)       -> 52-byte IMU frames
 *   SerialUSB1  (if02, /dev/bebop-imu-debug) -> human-readable diagnostics
 *   SerialUSB2  (if04, /dev/bebop-control)   -> control frames (this file)
 *
 * Keeping control on its own channel means the IMU parser never has to
 * filter button bytes, diagnostics stay `cat`-able, and a future host ->
 * Teensy command path (e.g. drive an LED or force a safe state) has a clean
 * framed transport. The channel is bidirectional; the Jetson-side reader
 * currently only consumes Teensy -> host events.
 *
 * ## Frame layout (little-endian; fixed 14 bytes)
 *
 *   offset  size  field
 *   ------  ----  ---------------------------------------------------------
 *    0      1     magic0   = 0xBE
 *    1      1     magic1   = 0xC0
 *    2      1     type     message type (see CTRL_MSG_*)
 *    3      1     seq      increments once per emitted frame (wraps)
 *    4      4     t_us     Teensy micros() at emit time (wraps)
 *    8      4     payload  type-specific; little-endian
 *   12      2     crc16    CRC-16/CCITT-FALSE over bytes [0, 12)
 *   ------  ----  ---------------------------------------------------------
 *   total  14 bytes
 *
 * `magic0`/`magic1` are deliberately different from the IMU frame's
 * `0xBE 0xB0` so a byte stream can never be mis-parsed across channels.
 *
 * ### Payload encoding
 *
 *   CTRL_MSG_BUTTON: (state << 8) | button_id
 *     state      0 = released, 1 = pressed
 *     button_id  0 = network-mode toggle (CTRL_BUTTON_NETWORK)
 *
 * The Jetson-side parser lives in
 * `jetson-agent/bebop-agent/src/control_serial.rs`.
 */

#ifndef CONTROL_PROTOCOL_H
#define CONTROL_PROTOCOL_H

#include <stdint.h>

#define CTRL_FRAME_MAGIC0 0xBE
#define CTRL_FRAME_MAGIC1 0xC0

// Teensy -> host message types.
#define CTRL_MSG_BUTTON 0x01

// Host -> Teensy message types (reserved for future use; the Jetson reader
// currently ignores inbound bytes).
#define CTRL_MSG_CMD 0x81

// Button identifiers.
#define CTRL_BUTTON_NETWORK 0

// Button states.
#define CTRL_BUTTON_RELEASED 0
#define CTRL_BUTTON_PRESSED  1

#pragma pack(push, 1)
typedef struct {
    uint8_t  magic0;   // 0xBE
    uint8_t  magic1;   // 0xC0
    uint8_t  type;     // CTRL_MSG_*
    uint8_t  seq;      // increments per frame
    uint32_t t_us;     // micros() at emit
    uint32_t payload;  // type-specific
    uint16_t crc16;    // CRC-16/CCITT-FALSE over bytes [0, 12)
} CtrlFrame;
#pragma pack(pop)

#define CTRL_FRAME_SIZE 14
// Number of bytes the CRC is computed over (everything except the CRC).
#define CTRL_CRC_LEN (CTRL_FRAME_SIZE - 2)

/**
 * @brief CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF, no reflection, no
 *        final XOR). Must stay byte-compatible with `crc16_ccitt()` in the
 *        Jetson-side parser (`control_serial.rs`). Kept separate from the
 *        IMU helper so the two protocols can evolve independently.
 */
static inline uint16_t ctrl_crc16(const uint8_t* data, uint32_t len) {
    uint16_t crc = 0xFFFF;
    for (uint32_t i = 0; i < len; i++) {
        crc ^= (uint16_t)data[i] << 8;
        for (int b = 0; b < 8; b++) {
            if (crc & 0x8000) {
                crc = (uint16_t)((crc << 1) ^ 0x1021);
            } else {
                crc = (uint16_t)(crc << 1);
            }
        }
    }
    return crc;
}

/**
 * @brief Populate a frame's magic + CRC. Call after filling type, seq,
 *        t_us and payload.
 */
static inline void ctrl_finalize(CtrlFrame* f) {
    f->magic0 = CTRL_FRAME_MAGIC0;
    f->magic1 = CTRL_FRAME_MAGIC1;
    f->crc16 = ctrl_crc16((const uint8_t*)f, CTRL_CRC_LEN);
}

#endif // CONTROL_PROTOCOL_H
