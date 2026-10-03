//! Teensy control-channel reader.
//!
//! The Jetson AGX Thor exposes no usable GPIO header, so the physical
//! network-mode button is wired to the Teensy 4.1 that already runs the
//! `teensy_bridge` firmware. That firmware uses `USB_TRIPLE_SERIAL` and emits
//! debounced press/release events as fixed 14-byte frames on its third CDC
//! port, which udev exposes as `/dev/bebop-control`.
//!
//! ## Frame contract
//!
//! The wire format is defined in
//! `firmware/bebop-locomotion/include/ControlProtocol.h`:
//!
//! ```text
//!   0   1   magic0 0xBE
//!   1   1   magic1 0xC0
//!   2   1   type      (0x01 = button)
//!   3   1   seq       u8
//!   4   4   t_us      u32 LE  (Teensy micros)
//!   8   4   payload   u32 LE  ((state << 8) | button_id)
//!  12   2   crc16     u16 LE  (CRC-16/CCITT-FALSE over bytes [0, 12))
//! ```
//!
//! This module turns button frames into [`RawEvent`]s, applies the
//! configured hold/debounce policy on the async side, and calls
//! [`crate::button::toggle_mode`] — the exact same path the GPIO source uses,
//! so the two are interchangeable.

use std::io::Read;
use std::time::{Duration, Instant};

use tokio::sync::mpsc;
use tracing::{info, warn};

use crate::state::AppState;

/// Total on-wire frame size, must match `CTRL_FRAME_SIZE` in
/// `ControlProtocol.h`.
const FRAME_SIZE: usize = 14;
/// Number of bytes the CRC is computed over (everything except the CRC).
const CRC_LEN: usize = FRAME_SIZE - 2;
const MAGIC0: u8 = 0xBE;
const MAGIC1: u8 = 0xC0;

const MSG_BUTTON: u8 = 0x01;
const BUTTON_NETWORK: u32 = 0;

const BUTTON_RELEASED: u32 = 0;
const BUTTON_PRESSED: u32 = 1;

/// USB CDC ignores the baud rate, but the `serialport` builder still
/// requires one. Any value works; 115200 matches the Teensy's nominal.
const NOMINAL_BAUD: u32 = 115_200;
const OPEN_BACKOFF_MIN_MS: u64 = 500;
const OPEN_BACKOFF_MAX_MS: u64 = 5_000;
const OPEN_LOUD_ATTEMPTS: u32 = 5;

/// A debounced button edge as decoded from the control channel. The hold
/// policy is applied by the async consumer, not the firmware.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RawEvent {
    Pressed,
    Released,
}

/// CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF, no reflection, no final
/// XOR). Mirrors `ctrl_crc16()` in `ControlProtocol.h`.
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

/// Decode the first frame in `buf` (which must start at the magic and be at
/// least `FRAME_SIZE` bytes). Returns `None` on a CRC mismatch so the caller
/// can resync by one byte.
fn parse_frame(buf: &[u8]) -> Option<(u8, u32)> {
    debug_assert!(buf.len() >= FRAME_SIZE);
    let expected = u16::from_le_bytes([buf[12], buf[13]]);
    let actual = crc16_ccitt(&buf[..CRC_LEN]);
    if expected != actual {
        return None;
    }
    let frame_type = buf[2];
    let payload = u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]);
    Some((frame_type, payload))
}

/// Entry point. Retries opening the control port forever (with backoff) so
/// the agent starts even if the Teensy is not yet enumerated. Once a press
/// event is accepted it toggles the network mode. Never returns while
/// enabled: on thread failure it parks so the supervisor doesn't treat the
/// exit as fatal.
pub async fn run(state: AppState) -> anyhow::Result<()> {
    let cfg = state.config().await.network;
    let device = cfg.control_device.clone();
    let hold = Duration::from_secs(cfg.button_hold_secs);
    let immediate = hold.is_zero();

    let (tx, mut rx) = mpsc::channel::<RawEvent>(64);

    let dev_for_thread = device.clone();
    let spawned = std::thread::Builder::new()
        .name("bebop-control".into())
        .spawn(move || reader_loop(&dev_for_thread, tx));
    match spawned {
        Ok(_) => info!(
            device = %device,
            hold_secs = hold.as_secs(),
            trigger = if immediate { "press" } else { "hold" },
            "mode button armed on Teensy control channel (toggles Known/Hosted)"
        ),
        Err(e) => {
            warn!(error = %e, "failed to spawn control reader thread; button unavailable");
            std::future::pending::<()>().await;
            return Ok(());
        }
    }

    let mut pressed_at: Option<Instant> = None;
    loop {
        // Sleeps until the hold deadline once a press is in progress;
        // pending forever otherwise.
        let hold_sleep = async {
            match pressed_at {
                Some(t0) => {
                    let deadline = t0 + hold;
                    if let Some(remaining) = deadline.checked_duration_since(Instant::now()) {
                        tokio::time::sleep(remaining).await;
                    }
                }
                None => std::future::pending::<()>().await,
            }
        };

        tokio::select! {
            ev = rx.recv() => {
                match ev {
                    Some(RawEvent::Pressed) => {
                        if immediate {
                            info!("button pressed (serial); toggling mode");
                            crate::button::toggle_mode(&state).await;
                        } else {
                            info!("button pressed (serial); hold to toggle");
                            pressed_at = Some(Instant::now());
                        }
                    }
                    Some(RawEvent::Released) => {
                        if pressed_at.take().is_some() {
                            info!("button released (serial)");
                        }
                    }
                    None => break,
                }
            }
            _ = hold_sleep => {
                if let Some(t0) = pressed_at.take() {
                    info!(
                        held_ms = t0.elapsed().as_millis(),
                        "button long-press (serial) detected"
                    );
                    crate::button::toggle_mode(&state).await;
                }
            }
        }
    }

    warn!("control reader thread exited; button unavailable until restart");
    std::future::pending::<()>().await;
    Ok(())
}

/// Blocking read + parse loop, runs on a dedicated thread. Reopens the port
/// on read errors or if it disappears.
fn reader_loop(device: &str, tx: mpsc::Sender<RawEvent>) {
    let mut backoff = Duration::from_millis(OPEN_BACKOFF_MIN_MS);
    let mut attempt: u32 = 0;

    'reopen: loop {
        attempt += 1;

        let opened = serialport::new(device, NOMINAL_BAUD)
            .timeout(Duration::from_millis(100))
            .open();

        let mut port = match opened {
            Ok(mut p) => {
                // Teensy transmits regardless of DTR; harmless and matches
                // what `cat`/most terminals do on open.
                let _ = p.write_data_terminal_ready(true);
                info!(
                    device = %device,
                    attempt,
                    "control: opened Teensy control port"
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
                        "control: open failed; retrying after backoff"
                    );
                } else if attempt == OPEN_LOUD_ATTEMPTS + 1 {
                    warn!(
                        device = %device,
                        "control: still cannot open the port after {} attempts; \
                         will keep retrying every {} ms (suppressing per-attempt \
                         warnings) — hint: check the Teensy is flashed with \
                         USB_TRIPLE_SERIAL and udev created /dev/bebop-control",
                        OPEN_LOUD_ATTEMPTS, OPEN_BACKOFF_MAX_MS
                    );
                }
                std::thread::sleep(backoff);
                backoff = (backoff * 2).min(Duration::from_millis(OPEN_BACKOFF_MAX_MS));
                continue 'reopen;
            }
        };

        let mut buf: Vec<u8> = Vec::with_capacity(FRAME_SIZE * 8);
        let mut chunk = [0u8; 128];
        let mut last_stats = Instant::now();
        let mut frames_ok: u64 = 0;
        let mut crc_errs: u64 = 0;

        loop {
            match port.read(&mut chunk) {
                Ok(0) => {}
                Ok(n) => buf.extend_from_slice(&chunk[..n]),
                Err(ref e) if e.kind() == std::io::ErrorKind::TimedOut => continue,
                Err(e) => {
                    warn!(device = %device, error = %e, "control: read error; reopening port");
                    std::thread::sleep(Duration::from_millis(250));
                    continue 'reopen;
                }
            }

            let mut i = 0usize;
            while i + FRAME_SIZE <= buf.len() {
                if buf[i] != MAGIC0 || buf[i + 1] != MAGIC1 {
                    i += 1;
                    continue;
                }
                match parse_frame(&buf[i..i + FRAME_SIZE]) {
                    Some((frame_type, payload)) => {
                        frames_ok += 1;
                        if frame_type == MSG_BUTTON {
                            let button_id = payload & 0xFF;
                            let state = (payload >> 8) & 0xFF;
                            if button_id == BUTTON_NETWORK {
                                let ev = if state == BUTTON_PRESSED {
                                    Some(RawEvent::Pressed)
                                } else if state == BUTTON_RELEASED {
                                    Some(RawEvent::Released)
                                } else {
                                    None
                                };
                                if let Some(ev) = ev {
                                    if tx.blocking_send(ev).is_err() {
                                        // Consumer gone (shutdown); stop.
                                        return;
                                    }
                                }
                            }
                        }
                        i += FRAME_SIZE;
                    }
                    None => {
                        crc_errs += 1;
                        i += 1;
                    }
                }
            }
            if i > 0 {
                buf.drain(..i);
            }
            if buf.len() > FRAME_SIZE * 64 {
                let drop = buf.len() - FRAME_SIZE * 4;
                buf.drain(..drop);
            }

            if last_stats.elapsed() >= Duration::from_secs(60) {
                info!(
                    target: "bebop_agent::control_serial",
                    frames = frames_ok,
                    crc_errs,
                    "control: channel alive"
                );
                last_stats = Instant::now();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_frame(frame_type: u8, seq: u8, payload: u32) -> [u8; FRAME_SIZE] {
        let mut b = [0u8; FRAME_SIZE];
        b[0] = MAGIC0;
        b[1] = MAGIC1;
        b[2] = frame_type;
        b[3] = seq;
        b[4..8].copy_from_slice(&123u32.to_le_bytes());
        b[8..12].copy_from_slice(&payload.to_le_bytes());
        let crc = crc16_ccitt(&b[..CRC_LEN]);
        b[12..14].copy_from_slice(&crc.to_le_bytes());
        b
    }

    #[test]
    fn parses_button_frame() {
        let f = build_frame(MSG_BUTTON, 1, (BUTTON_PRESSED << 8) | BUTTON_NETWORK);
        assert_eq!(parse_frame(&f), Some((MSG_BUTTON, BUTTON_PRESSED << 8)));
    }

    #[test]
    fn rejects_corrupted_frame() {
        let mut f = build_frame(MSG_BUTTON, 1, (BUTTON_PRESSED << 8) | BUTTON_NETWORK);
        f[8] ^= 0xFF;
        assert!(parse_frame(&f).is_none());
    }

    /// Same CRC-16/CCITT-FALSE check value as the IMU protocol: the ASCII
    /// string "123456789" must hash to 0x29B1.
    #[test]
    fn crc_matches_known_vector() {
        assert_eq!(crc16_ccitt(b"123456789"), 0x29B1);
    }
}
