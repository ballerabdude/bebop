/**
 * @file ImuSerialProtocol.h
 * @brief Binary wire format for streaming BNO085 IMU data Teensy -> Jetson
 *        over USB serial (USB CDC).
 *
 * ## Why a binary frame
 *
 * The Teensy reads the BNO085 over SPI and forwards fused orientation +
 * calibrated gyro + linear accel to the Jetson over USB. USB CDC on the
 * Teensy 4.1 runs at full USB speed regardless of the nominal baud, so a
 * fixed-size 52-byte frame at 200 Hz (~10.4 KB/s) is trivial. A magic
 * prefix + CRC lets the Jetson resynchronize after a partial read or a
 * hot-plug without a stateful handshake.
 *
 * ## Frame layout (little-endian; both Teensy Cortex-M7 and Jetson
 * ##                aarch64 are little-endian, so the struct is sent /
 * ##                parsed as raw bytes with no byte-swapping)
 *
 *   offset  size  field
 *   ------  ----  ---------------------------------------------------------
 *    0      1     magic0      = 0xBE
 *    1      1     magic1      = 0xB0
 *    2      4     seq         u32, increments once per emitted frame
 *    6      4     t_us        u32, Teensy micros() at emit time
 *   10     16     quat_xyzw   4 x f32, body-frame-from-world rotation in
 *                             XYZW (scalar-last) order. RAW sensor frame:
 *                             the Teensy does NOT apply the chassis mount
 *                             rotation; the Jetson does (see imu_serial.rs).
 *   26     12     gyro_xyz    3 x f32, calibrated gyro (rad/s), sensor frame
 *   38     12     accel_xyz   3 x f32, linear accel (m/s^2), sensor frame
 *   50      2     crc16       u16, CRC-16/CCITT-FALSE over bytes [0, 50)
 *   ------  ----  ---------------------------------------------------------
 *   total  52 bytes
 *
 * The CRC covers every byte before it (magic through accel). The Jetson
 * scans the byte stream for the 0xBE 0xB0 magic, reads the following 50
 * bytes, and accepts the frame only if the CRC matches.
 *
 * NOTE: quaternion order is XYZW to match the Jetson runtime's
 * `ImuSnapshot.quaternion` contract (see `firmware/bebop-linux/src/imu.rs`).
 * The BNO085 reports (real, i, j, k); map that to (i, j, k, real) on the
 * wire, i.e. (x, y, z, w).
 */

#ifndef IMU_SERIAL_PROTOCOL_H
#define IMU_SERIAL_PROTOCOL_H

#include <stdint.h>
#include <string.h>

#define IMU_FRAME_MAGIC0 0xBE
#define IMU_FRAME_MAGIC1 0xB0

#pragma pack(push, 1)
typedef struct {
    uint8_t  magic0;        // 0xBE
    uint8_t  magic1;        // 0xB0
    uint32_t seq;           // increments per frame
    uint32_t t_us;          // micros() at emit
    float    quat_xyzw[4];  // x, y, z, w (sensor frame, world->body)
    float    gyro_xyz[3];   // rad/s, sensor frame
    float    accel_xyz[3];  // m/s^2, sensor frame
    uint16_t crc16;         // CRC-16/CCITT-FALSE over bytes [0, 50)
} ImuSerialFrame;
#pragma pack(pop)

#define IMU_SERIAL_FRAME_SIZE 52
// Number of bytes the CRC is computed over (everything except the CRC itself).
#define IMU_SERIAL_CRC_LEN (IMU_SERIAL_FRAME_SIZE - 2)

/**
 * @brief CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF, no reflection, no
 *        final XOR). Same algorithm the Jetson-side parser uses.
 */
static inline uint16_t imu_serial_crc16(const uint8_t* data, uint32_t len) {
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
 * @brief Populate a frame's magic + CRC. Call after filling seq, t_us,
 *        quat, gyro and accel.
 */
static inline void imu_serial_finalize(ImuSerialFrame* f) {
    f->magic0 = IMU_FRAME_MAGIC0;
    f->magic1 = IMU_FRAME_MAGIC1;
    f->crc16 = imu_serial_crc16((const uint8_t*)f, IMU_SERIAL_CRC_LEN);
}

#endif // IMU_SERIAL_PROTOCOL_H
