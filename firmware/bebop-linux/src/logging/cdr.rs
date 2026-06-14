//! Minimal CDR (Common Data Representation) encoder.
//!
//! CDR is the OMG wire format used by DDS / ROS2. It is little-endian,
//! prefixed with a 4-byte encapsulation header (endianness marker).
//!
//! Each primitive is aligned to its own size (4 bytes for u32/f32,
//! 8 bytes for u64/f64), measured from the *start of the body* — i.e.
//! the byte right after the 4-byte encapsulation header. This matches
//! the classic CDR alignment that ROS2 (FastCDR / CycloneDDS) and
//! Foxglove's `@foxglove/cdr` reader expect. Getting the 8-byte
//! alignment wrong shifts every `float64` and corrupts downstream
//! sequence-length reads ("Invalid typed array length" in Foxglove).

pub struct CdrEncoder {
    buf: Vec<u8>,
    pos: usize,
    /// Offset where the aligned body begins (after the encapsulation
    /// header). Alignment is computed relative to this point.
    body_start: usize,
}

impl CdrEncoder {
    pub fn with_capacity(cap: usize) -> Self {
        Self { buf: Vec::with_capacity(cap), pos: 0, body_start: 0 }
    }

    pub fn write_header(&mut self) {
        // CDR encapsulation header: a 2-byte representation identifier in
        // big-endian byte order, followed by 2 option bytes. 0x0001 selects
        // PLAIN_CDR with a little-endian payload. Readers (e.g. Foxglove's
        // `@foxglove/cdr`) take the endianness from byte index 1, so this
        // MUST be `[0x00, 0x01, ...]` — writing the id as a little-endian
        // u32 would put 0x00 there and be misread as big-endian.
        self.buf.extend_from_slice(&[0x00, 0x01, 0x00, 0x00]);
        self.pos += 4;
        // CDR alignment restarts after the encapsulation header.
        self.body_start = self.pos;
    }

    /// Pad with zero bytes so the next write lands on an `alignment`-byte
    /// boundary relative to the start of the body.
    fn align(&mut self, alignment: usize) {
        let remainder = (self.pos - self.body_start) % alignment;
        if remainder != 0 {
            let pad = alignment - remainder;
            self.buf.resize(self.buf.len() + pad, 0);
            self.pos += pad;
        }
    }

    pub fn write_u32(&mut self, value: u32) {
        self.align(4);
        self.write_u32_le_raw(value);
    }

    pub fn write_u64(&mut self, value: u64) {
        self.align(8);
        self.buf.extend_from_slice(&value.to_le_bytes());
        self.pos += 8;
    }

    pub fn write_f32(&mut self, value: f32) {
        self.align(4);
        self.write_u32_le_raw(value.to_bits());
    }

    pub fn write_f64(&mut self, value: f64) {
        self.align(8);
        self.buf.extend_from_slice(&value.to_le_bytes());
        self.pos += 8;
    }

    pub fn write_bool(&mut self, value: bool) {
        self.buf.push(value as u8);
        self.pos += 1;
    }

    pub fn write_string(&mut self, s: &str) {
        let bytes = s.as_bytes();
        let len = (bytes.len() + 1) as u32;
        self.write_u32(len);
        self.buf.extend_from_slice(bytes);
        self.pos += bytes.len();
        self.buf.push(0);
        self.pos += 1;
    }

    pub fn write_f32_seq(&mut self, values: &[f32]) {
        self.write_u32(values.len() as u32);
        for &v in values {
            self.write_u32_le_raw(v.to_bits());
        }
    }

    pub fn write_bool_seq(&mut self, values: &[bool]) {
        self.write_u32(values.len() as u32);
        for &v in values {
            self.buf.push(v as u8);
            self.pos += 1;
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.buf[..self.pos]
    }

    pub fn clear(&mut self) {
        self.buf.clear();
        self.pos = 0;
        self.body_start = 0;
    }

    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.pos
    }

    fn write_u32_le_raw(&mut self, value: u32) {
        self.buf.extend_from_slice(&value.to_le_bytes());
        self.pos += 4;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_is_le_tag() {
        let mut e = CdrEncoder::with_capacity(64);
        e.write_header();
        // Representation id 0x0001 (PLAIN_CDR, little-endian payload) is
        // stored big-endian, so byte index 1 carries the endianness flag.
        assert_eq!(e.as_bytes(), &[0x00, 0x01, 0x00, 0x00]);
    }

    #[test]
    fn u32_round_trips() {
        let mut e = CdrEncoder::with_capacity(4);
        e.write_u32(42);
        let bytes = e.as_bytes();
        let val = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        assert_eq!(val, 42);
    }

    #[test]
    fn f32_round_trips() {
        let mut e = CdrEncoder::with_capacity(8);
        e.write_f32(1.5);
        let bytes = e.as_bytes();
        assert_eq!(bytes.len(), 4);
        let val = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        assert!((val - 1.5).abs() < 1e-7);
    }

    #[test]
    fn f64_round_trips() {
        let mut e = CdrEncoder::with_capacity(16);
        e.write_f64(std::f64::consts::PI);
        let bytes = e.as_bytes();
        assert_eq!(bytes.len(), 8);
        let val = f64::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7]]);
        assert!((val - std::f64::consts::PI).abs() < 1e-15);
    }

    #[test]
    fn bool_is_one_byte() {
        let mut e = CdrEncoder::with_capacity(16);
        e.write_bool(true);
        assert_eq!(e.len(), 1);
    }

    #[test]
    fn string_with_null_terminator() {
        let mut e = CdrEncoder::with_capacity(64);
        e.write_string("hello");
        let bytes = e.as_bytes();
        // 4 (length) + 5 ("hello") + 1 (NUL); no trailing padding — the
        // next field aligns itself when it is written.
        assert_eq!(bytes.len(), 10);
        let len = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        assert_eq!(len, 6);
        assert_eq!(&bytes[4..9], b"hello");
        assert_eq!(bytes[9], 0);
    }

    /// Decode a Header-style message the way a classic-CDR reader
    /// (e.g. Foxglove's `@foxglove/cdr`) does: little-endian payload,
    /// alignment relative to the body start, 8-byte aligned float64.
    /// Guards against the encapsulation byte order, the body alignment,
    /// and the field layout all at once.
    #[test]
    fn decodes_like_classic_cdr_reader() {
        let mut e = CdrEncoder::with_capacity(64);
        e.write_header();
        e.write_u32(7); // stamp.sec
        e.write_u32(8); // stamp.nanosec
        e.write_string("imu_link"); // frame_id
        e.write_f64(std::f64::consts::PI); // a float64 that must be 8-aligned

        let b = e.as_bytes();
        // Byte index 1 selects little-endian; getting this wrong makes the
        // whole payload read as big-endian.
        assert_eq!(b[1], 0x01);

        let origin = 4usize; // body starts after the 4-byte encapsulation header
        let mut off = 4usize;
        let mut align = |off: &mut usize, n: usize| {
            let rem = (*off - origin) % n;
            if rem != 0 {
                *off += n - rem;
            }
        };
        let read_u32 = |off: &mut usize, align: &mut dyn FnMut(&mut usize, usize)| {
            align(off, 4);
            let v = u32::from_le_bytes(b[*off..*off + 4].try_into().unwrap());
            *off += 4;
            v
        };

        let sec = read_u32(&mut off, &mut align);
        let nsec = read_u32(&mut off, &mut align);
        assert_eq!((sec, nsec), (7, 8));

        let slen = read_u32(&mut off, &mut align) as usize;
        let frame = std::str::from_utf8(&b[off..off + slen - 1]).unwrap();
        off += slen;
        assert_eq!(frame, "imu_link");

        align(&mut off, 8);
        assert_eq!((off - origin) % 8, 0, "float64 must be 8-byte aligned");
        let val = f64::from_le_bytes(b[off..off + 8].try_into().unwrap());
        assert!((val - std::f64::consts::PI).abs() < 1e-12);
    }

    #[test]
    fn f64_is_8_byte_aligned_within_body() {
        // After the 4-byte encapsulation header, alignment restarts.
        // A u32 then an f64 must leave 4 bytes of padding so the f64
        // begins at an 8-byte boundary relative to the body start.
        let mut e = CdrEncoder::with_capacity(64);
        e.write_header(); // 4-byte encapsulation, body starts here
        e.write_u32(1); // body offset 0..4
        e.write_f64(2.0); // aligned to body offset 8
        // 4 (header) + 4 (u32) + 4 (pad) + 8 (f64) = 20
        assert_eq!(e.len(), 20);
    }

    #[test]
    fn f32_seq_round_trips() {
        let mut e = CdrEncoder::with_capacity(64);
        e.write_f32_seq(&[1.0_f32, 2.5, -3.0]);
        let bytes = e.as_bytes();
        assert_eq!(bytes.len(), 16);
        let count = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        assert_eq!(count, 3);
    }

    #[test]
    fn alignment_between_fields() {
        let mut e = CdrEncoder::with_capacity(64);
        e.write_bool(true);
        e.write_u32(0xDEAD_BEEF_u32);
        let bytes = e.as_bytes();
        assert_eq!(bytes.len(), 8);
        let val = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        assert_eq!(val, 0xDEAD_BEEF);
    }

    #[test]
    fn clear_resets_position() {
        let mut e = CdrEncoder::with_capacity(64);
        e.write_u32(42);
        e.clear();
        assert_eq!(e.len(), 0);
        e.write_u32(99);
        assert_eq!(e.len(), 4);
    }
}
