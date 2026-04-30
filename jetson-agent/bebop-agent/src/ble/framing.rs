//! Chunked framing for BLE writes/notifications.
//!
//! ATT MTU varies (default 23, negotiated up to ~517). Protobuf messages are
//! typically small but Wi-Fi scan results or OTA status bursts can exceed a
//! single frame. We use a very simple length-prefixed scheme:
//!
//! ```text
//!  +--------+--------+-----------+========================+
//!  |  ver   | flags  |  length   |        payload         |
//!  | u8     | u8     |  u16 BE   |       <= length        |
//!  +--------+--------+-----------+========================+
//! ```
//!
//! * `ver`    — protocol version, currently `1`.
//! * `flags`  — bit 0: `FRAGMENT` (more to come), bit 1: `FINAL`.
//! * `length` — number of payload bytes in *this* frame.
//!
//! A logical message is the concatenation of one or more frames where only
//! the last one has `FINAL` set. Reassembly is per-connection.

use bytes::{Buf, BufMut, BytesMut};

pub const PROTO_VERSION: u8 = 1;
pub const FLAG_FRAGMENT: u8 = 1 << 0;
pub const FLAG_FINAL: u8 = 1 << 1;

/// Bytes consumed by the per-frame header (`ver | flags | len_be`).
pub const HEADER_LEN: usize = 4;

/// Split a logical message into BLE frames no bigger than `max_frame`
/// (which must already account for header overhead).
pub fn encode(payload: &[u8], max_payload_per_frame: usize) -> Vec<Vec<u8>> {
    if payload.is_empty() {
        return vec![make_frame(&[], FLAG_FINAL)];
    }

    let mut out = Vec::new();
    let mut remaining = payload;
    while !remaining.is_empty() {
        let take = remaining.len().min(max_payload_per_frame);
        let chunk = &remaining[..take];
        remaining = &remaining[take..];
        let flags = if remaining.is_empty() {
            FLAG_FINAL
        } else {
            FLAG_FRAGMENT
        };
        out.push(make_frame(chunk, flags));
    }
    out
}

fn make_frame(payload: &[u8], flags: u8) -> Vec<u8> {
    let mut buf = BytesMut::with_capacity(4 + payload.len());
    buf.put_u8(PROTO_VERSION);
    buf.put_u8(flags);
    buf.put_u16(payload.len() as u16);
    buf.put_slice(payload);
    buf.to_vec()
}

/// Accumulates inbound frames into complete messages.
#[derive(Default)]
pub struct Reassembler {
    buffer: BytesMut,
}

impl Reassembler {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed one frame. Returns `Some(message)` when a full message is ready.
    pub fn push(&mut self, frame: &[u8]) -> anyhow::Result<Option<Vec<u8>>> {
        if frame.len() < 4 {
            anyhow::bail!("frame too short");
        }
        let mut cursor = frame;
        let ver = cursor.get_u8();
        if ver != PROTO_VERSION {
            anyhow::bail!("unsupported frame version {ver}");
        }
        let flags = cursor.get_u8();
        let len = cursor.get_u16() as usize;
        if cursor.remaining() < len {
            anyhow::bail!("frame payload truncated");
        }
        self.buffer.extend_from_slice(&cursor[..len]);
        if flags & FLAG_FINAL != 0 {
            let complete = self.buffer.split().to_vec();
            Ok(Some(complete))
        } else {
            Ok(None)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_single_frame() {
        let msg = b"hello world";
        let frames = encode(msg, 64);
        assert_eq!(frames.len(), 1);

        let mut r = Reassembler::new();
        let out = r.push(&frames[0]).unwrap();
        assert_eq!(out.unwrap(), msg);
    }

    #[test]
    fn round_trip_multi_frame() {
        let msg: Vec<u8> = (0..=200u8).collect();
        let frames = encode(&msg, 32);
        assert!(frames.len() > 1);

        let mut r = Reassembler::new();
        let mut result = None;
        for f in &frames {
            result = r.push(f).unwrap();
        }
        assert_eq!(result.unwrap(), msg);
    }
}
