//! BNO085 IMU reader that consumes pre-fused frames streamed by the
//! Teensy `imu_bridge` firmware over USB serial, instead of talking to
//! the chip over the Jetson's own SPI bus.
//!
//! ## Why a serial backend
//!
//! The BNO can be wired to the Teensy (which already sits on the robot
//! for motor control) rather than to the Jetson's SPI header. The Teensy
//! reads the chip and forwards each fused sample to the Jetson as a
//! fixed 52-byte binary frame. This module parses that stream and writes
//! the result into the **same** [`ImuShared`] snapshot the SPI path
//! ([`crate::imu::spawn_imu_thread`]) fills, so the policy runner and the
//! telemetry pump don't care which backend is active.
//!
//! ## Frame contract
//!
//! The wire format is defined in
//! `firmware/bebop-locomotion/include/ImuSerialProtocol.h`:
//!
//! ```text
//!   0   1   magic 0xBE 0xB0
//!   2   4   seq        u32 LE
//!   6   4   t_us       u32 LE  (Teensy micros)
//!  10  16   quat_xyzw  4 x f32 LE  (RAW sensor frame, world->body, XYZW)
//!  26  12   gyro_xyz   3 x f32 LE  (rad/s, sensor frame)
//!  38  12   accel_xyz  3 x f32 LE  (m/s^2, sensor frame)
//!  50   2   crc16      u16 LE  (CRC-16/CCITT-FALSE over bytes [0, 50))
//! ```
//!
//! The Teensy sends the **raw sensor-frame** quaternion / gyro; the
//! chassis mount rotation (YAML `imu.mount:`) is applied here, exactly
//! as [`crate::imu`] does for the SPI path, so the published snapshot is
//! always a body-frame (REP-103 / FLU) attitude + angular velocity.

use std::io::Read;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use tracing::{error, info, warn};

use crate::config::ImuConfig;
use crate::imu::{quat_mul_xyzw, quat_normalize_xyzw, rotate_vec_by_quat_xyzw, ImuShared};

/// Total on-wire frame size, must match `IMU_SERIAL_FRAME_SIZE` in
/// `ImuSerialProtocol.h`.
const FRAME_SIZE: usize = 52;
/// Number of bytes the CRC is computed over (everything except the CRC).
const CRC_LEN: usize = FRAME_SIZE - 2;
const MAGIC0: u8 = 0xBE;
const MAGIC1: u8 = 0xB0;

/// CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF, no reflection, no final
/// XOR). Mirrors `imu_serial_crc16()` on the Teensy side.
fn crc16_ccitt(data: &[u8]) -> u16 {
    let mut crc: u16 = 0xFFFF;
    for &b in data {
        crc ^= (b as u16) << 8;
        for _ in 0..8 {
            if crc & 0x8000 != 0 {
                crc = (crc << 1) ^ 0x1021;
            } else {
                crc <<= 1;
            }
        }
    }
    crc
}

/// A decoded serial IMU frame (raw sensor-frame values, pre-mount).
struct ParsedFrame {
    quat_xyzw: [f32; 4],
    gyro_xyz: [f32; 3],
}

/// Parse one frame from a 52-byte slice that starts at the magic. The
/// caller guarantees `buf.len() >= FRAME_SIZE`. Returns `None` if the CRC
/// fails (caller should resync by one byte).
fn parse_frame(buf: &[u8]) -> Option<ParsedFrame> {
    debug_assert!(buf.len() >= FRAME_SIZE);
    let expected = u16::from_le_bytes([buf[50], buf[51]]);
    let actual = crc16_ccitt(&buf[..CRC_LEN]);
    if expected != actual {
        return None;
    }
    let f32_at = |off: usize| f32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]]);
    Some(ParsedFrame {
        quat_xyzw: [f32_at(10), f32_at(14), f32_at(18), f32_at(22)],
        gyro_xyz: [f32_at(26), f32_at(30), f32_at(34)],
    })
}

/// Spawn a background thread that opens the Teensy serial port, parses
/// the binary IMU stream, and publishes body-frame attitude + angular
/// velocity into `shared`.
///
/// Mirrors the lifecycle of [`crate::imu::spawn_imu_thread`]: it retries
/// opening the port forever (with backoff) so the runtime starts even if
/// the Teensy is not yet enumerated, and it honors `shutdown` for a clean
/// exit. Returns `None` only if the OS thread itself fails to spawn.
pub fn spawn_imu_serial_thread(
    cfg: ImuConfig,
    shutdown: Arc<AtomicBool>,
    shared: ImuShared,
) -> Option<JoinHandle<()>> {
    const OPEN_BACKOFF_MIN_MS: u64 = 500;
    const OPEN_BACKOFF_MAX_MS: u64 = 5_000;
    const OPEN_LOUD_ATTEMPTS: u32 = 5;
    /// USB CDC ignores the baud rate, but the `serialport` builder still
    /// requires one. Any value works; 115200 matches the Teensy's nominal.
    const NOMINAL_BAUD: u32 = 115_200;

    let device = cfg.serial_device.clone();
    let period_ms = cfg.rotation_vector_period_ms;
    let mount_quat = cfg.mount_quat_sensor_body;
    let mount_is_identity = mount_quat == ImuConfig::IDENTITY_QUAT;

    // Seed the staleness budget so the telemetry stale check is sane even
    // before the first frame arrives.
    if let Ok(mut g) = shared.lock() {
        g.report_period_ms = period_ms;
    }

    info!(
        device = %device,
        period_ms,
        mount_is_identity,
        "IMU(serial): thread spawning; will retry opening the Teensy bridge \
         port forever in background (runtime continues without a live IMU \
         until frames arrive)"
    );

    fn sleep_or_shutdown(dur: Duration, shutdown: &Arc<AtomicBool>) -> bool {
        let wake = Instant::now() + dur;
        while let Some(remaining) = wake.checked_duration_since(Instant::now()) {
            if shutdown.load(Ordering::SeqCst) {
                return true;
            }
            std::thread::sleep(remaining.min(Duration::from_millis(50)));
        }
        false
    }

    Some(std::thread::spawn(move || {
        let mut backoff = Duration::from_millis(OPEN_BACKOFF_MIN_MS);
        let mut attempt: u32 = 0;

        // Reopen loop: an unplugged / reset Teensy drops the port; we fall
        // back out here, re-open, and resume streaming.
        'reopen: while !shutdown.load(Ordering::SeqCst) {
            attempt += 1;

            let port = serialport::new(&device, NOMINAL_BAUD)
                .timeout(Duration::from_millis(100))
                .open();

            let mut port = match port {
                Ok(p) => {
                    info!(
                        device = %device,
                        attempt,
                        "IMU(serial): opened Teensy bridge port; reading binary frames"
                    );
                    attempt = 0;
                    backoff = Duration::from_millis(OPEN_BACKOFF_MIN_MS);
                    p
                }
                Err(e) => {
                    if attempt <= OPEN_LOUD_ATTEMPTS {
                        warn!(
                            device = %device,
                            attempt,
                            backoff_ms = backoff.as_millis() as u64,
                            error = %e,
                            "IMU(serial): open failed; retrying after backoff"
                        );
                    } else if attempt == OPEN_LOUD_ATTEMPTS + 1 {
                        error!(
                            device = %device,
                            "IMU(serial): still cannot open the port after {} attempts; \
                             will keep retrying every {} ms (suppressing per-attempt \
                             warnings) — hint: check the Teensy is flashed with the \
                             `imu_bridge` firmware and the USB cable / device path",
                            OPEN_LOUD_ATTEMPTS, OPEN_BACKOFF_MAX_MS
                        );
                    }
                    if sleep_or_shutdown(backoff, &shutdown) {
                        break 'reopen;
                    }
                    backoff = (backoff * 2).min(Duration::from_millis(OPEN_BACKOFF_MAX_MS));
                    continue 'reopen;
                }
            };

            // Rolling byte buffer for resync across reads. A handful of
            // frames of headroom is plenty; we compact it after parsing.
            let mut buf: Vec<u8> = Vec::with_capacity(FRAME_SIZE * 8);
            let mut chunk = [0u8; 256];
            let mut frames_ok: u64 = 0;
            let mut crc_errs: u64 = 0;
            let mut last_seen = Instant::now();
            let mut gyro_ever_seen = false;
            let mut last_stats = Instant::now();

            while !shutdown.load(Ordering::SeqCst) {
                match port.read(&mut chunk) {
                    Ok(0) => {}
                    Ok(n) => {
                        buf.extend_from_slice(&chunk[..n]);
                        last_seen = Instant::now();
                    }
                    Err(ref e) if e.kind() == std::io::ErrorKind::TimedOut => {
                        // No bytes within the timeout. If the stream has been
                        // silent for a while the Teensy likely went away; drop
                        // out to the reopen loop.
                        if last_seen.elapsed() > Duration::from_secs(2) {
                            warn!(
                                device = %device,
                                "IMU(serial): no frames for >2 s; reopening port"
                            );
                            continue 'reopen;
                        }
                        continue;
                    }
                    Err(e) => {
                        warn!(device = %device, error = %e, "IMU(serial): read error; reopening port");
                        if sleep_or_shutdown(Duration::from_millis(250), &shutdown) {
                            break 'reopen;
                        }
                        continue 'reopen;
                    }
                }

                // Parse as many complete frames as the buffer holds,
                // resyncing on the magic prefix and dropping bad CRCs.
                let mut i = 0usize;
                let now = Instant::now();
                while i + FRAME_SIZE <= buf.len() {
                    if buf[i] != MAGIC0 || buf[i + 1] != MAGIC1 {
                        i += 1;
                        continue;
                    }
                    match parse_frame(&buf[i..i + FRAME_SIZE]) {
                        Some(frame) => {
                            frames_ok += 1;
                            let q_world_sensor = frame.quat_xyzw;
                            let q_world_body = if mount_is_identity {
                                quat_normalize_xyzw(q_world_sensor)
                            } else {
                                quat_normalize_xyzw(quat_mul_xyzw(q_world_sensor, mount_quat))
                            };

                            let [wx, wy, wz] = frame.gyro_xyz;
                            let gyro_mag_sq = wx * wx + wy * wy + wz * wz;

                            if let Ok(mut g) = shared.lock() {
                                g.quaternion = Some(q_world_body);
                                g.last_update = Some(now);
                                g.report_period_ms = period_ms;
                                // Match the SPI path: only surface a gyro once
                                // a real (non-zero) sample lands, so the policy
                                // can tell "no gyro yet" from "perfectly still".
                                if gyro_mag_sq > 1e-9 {
                                    let omega_body =
                                        rotate_vec_by_quat_xyzw(frame.gyro_xyz, mount_quat);
                                    g.angular_velocity_body = Some(omega_body);
                                    g.gyro_last_update = Some(now);
                                }
                            }
                            if gyro_mag_sq > 1e-9 && !gyro_ever_seen {
                                info!(
                                    target: "bebop_linux::imu_serial",
                                    "IMU(serial): first gyro sample received; base_ang_vel is live"
                                );
                                gyro_ever_seen = true;
                            }
                            i += FRAME_SIZE;
                        }
                        None => {
                            // Bad CRC at this magic — likely a false-positive
                            // magic inside payload bytes. Skip one byte and
                            // keep scanning.
                            crc_errs += 1;
                            i += 1;
                        }
                    }
                }
                // Drop everything we consumed / scanned past; keep the tail
                // (a possibly-partial frame) for the next read.
                if i > 0 {
                    buf.drain(..i);
                }
                // Guard against unbounded growth if we somehow never sync.
                if buf.len() > FRAME_SIZE * 64 {
                    let drop = buf.len() - FRAME_SIZE * 4;
                    buf.drain(..drop);
                }

                if last_stats.elapsed() >= Duration::from_secs(5) {
                    info!(
                        target: "bebop_linux::imu_serial",
                        rate_hz = frames_ok / 5,
                        crc_errs,
                        "IMU(serial): streaming"
                    );
                    frames_ok = 0;
                    crc_errs = 0;
                    last_stats = Instant::now();
                }
            }

            // shutdown requested
            break 'reopen;
        }

        info!(
            target: "bebop_linux::imu_serial",
            "IMU(serial) thread exiting"
        );
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a valid frame the way the Teensy does, then confirm the
    /// parser round-trips it and the CRC matches both implementations.
    fn build_frame(seq: u32, quat: [f32; 4], gyro: [f32; 3], accel: [f32; 3]) -> [u8; FRAME_SIZE] {
        let mut b = [0u8; FRAME_SIZE];
        b[0] = MAGIC0;
        b[1] = MAGIC1;
        b[2..6].copy_from_slice(&seq.to_le_bytes());
        b[6..10].copy_from_slice(&123u32.to_le_bytes());
        for (k, v) in quat.iter().enumerate() {
            b[10 + k * 4..14 + k * 4].copy_from_slice(&v.to_le_bytes());
        }
        for (k, v) in gyro.iter().enumerate() {
            b[26 + k * 4..30 + k * 4].copy_from_slice(&v.to_le_bytes());
        }
        for (k, v) in accel.iter().enumerate() {
            b[38 + k * 4..42 + k * 4].copy_from_slice(&v.to_le_bytes());
        }
        let crc = crc16_ccitt(&b[..CRC_LEN]);
        b[50..52].copy_from_slice(&crc.to_le_bytes());
        b
    }

    #[test]
    fn parses_valid_frame() {
        let q = [0.1f32, -0.2, 0.3, 0.927_362];
        let g = [0.5f32, -0.25, 0.1];
        let frame = build_frame(7, q, g, [0.0, 0.0, -9.81]);
        let parsed = parse_frame(&frame).expect("valid CRC");
        assert_eq!(parsed.quat_xyzw, q);
        assert_eq!(parsed.gyro_xyz, g);
    }

    #[test]
    fn rejects_corrupted_frame() {
        let mut frame = build_frame(1, [0.0, 0.0, 0.0, 1.0], [0.0; 3], [0.0; 3]);
        frame[12] ^= 0xFF; // flip a quaternion byte, leave CRC stale
        assert!(parse_frame(&frame).is_none());
    }

    /// CRC-16/CCITT-FALSE check value: the ASCII string "123456789" must
    /// hash to 0x29B1. Anchors our implementation against the spec so it
    /// stays byte-compatible with the Teensy header.
    #[test]
    fn crc_matches_known_vector() {
        assert_eq!(crc16_ccitt(b"123456789"), 0x29B1);
    }
}
