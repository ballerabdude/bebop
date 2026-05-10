//! Length-prefixed BLE framing.
//!
//! Mirrors `bebop-agent/src/ble/framing.rs` and
//! `bebop-app/src/ble/protocol.ts` — keep all three in sync.

use bytes::{Buf, BufMut, BytesMut};

pub const PROTO_VERSION: u8 = 1;
pub const FLAG_FRAGMENT: u8 = 1 << 0;
pub const FLAG_FINAL: u8 = 1 << 1;

/// Split a logical message into BLE frames no bigger than `max_payload_per_frame`
/// bytes of payload (the 4-byte header is added on top).
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

#[derive(Default)]
pub struct Reassembler {
    buffer: BytesMut,
}

impl Reassembler {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed one inbound frame. Returns `Some(message)` once the FINAL flag is seen.
    pub fn push(&mut self, frame: &[u8]) -> Result<Option<Vec<u8>>, String> {
        if frame.len() < 4 {
            return Err("frame too short".into());
        }
        let mut cursor = frame;
        let ver = cursor.get_u8();
        if ver != PROTO_VERSION {
            return Err(format!("unsupported frame version {ver}"));
        }
        let flags = cursor.get_u8();
        let len = cursor.get_u16() as usize;
        if cursor.remaining() < len {
            return Err("frame payload truncated".into());
        }
        self.buffer.extend_from_slice(&cursor[..len]);
        if flags & FLAG_FINAL != 0 {
            Ok(Some(self.buffer.split().to_vec()))
        } else {
            Ok(None)
        }
    }
}
