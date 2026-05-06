//! Robstride PowerBoard CAN driver.
//!
//! The Robstride PowerBoard is the smart power-distribution module that
//! sits between the battery pack and the motor / 24 V / 12 V / 5 V rails.
//! On Bebop V2 it lives on its own CAN bus (typically `can4`) and reports:
//!
//!   - Battery (VBUS) voltage  — the raw pack voltage.  Used as the
//!     primary "fuel gauge" input together with the configured pack
//!     chemistry (cell count, full / empty thresholds).
//!   - Motor (VMBUS) voltage   — voltage on the motor branch after the
//!     soft-start FET; differs from VBUS during pre-charge / discharge.
//!   - Board temperature       — internal sensor, °C.
//!   - 24-bit fault word       — over-/under-voltage, over-current,
//!     over-temp, INA238 sense fault, etc. (see `describe_faults`).
//!   - Per-branch currents (AL / AR / LL / LR) when polled with type 04.
//!
//! ## Wire format (User Manual V1.3, §七)
//!
//! All frames are CAN 2.0 extended (29-bit ID), 1 Mbps, 8-byte payload.
//!
//! Outgoing (host → power board) ID layout:
//!
//! ```text
//!   bit 28..24 : communication type (5 bits, see `cmd::*` below)
//!   bit 23..8  : data area 2 (16 bits, command-specific; usually 0)
//!   bit 7..0   : POWER_ID (target board address; default 0xAA)
//! ```
//!
//! Incoming (power board → host) Type-03 / Type-04 / Type-05 frames keep
//! the same `comm_type` in bits 24..28, but **the source POWER_ID is
//! carried in bits 16..23** (verified against a real board: a Type-03
//! response from POWER_ID 0xAA arrives as ID `0x03AA0000`). Bits 0..15
//! hold the target/host address, which is zero unless we explicitly
//! set one. The user manual labels bits 16..23 "Data Area 2 / current
//! sampling", but the live board uses that byte to echo the source
//! POWER_ID — we route on that, not on the low byte.
//!
//! The user manual is sometimes contradictory about how many bits hold
//! which counter; this implementation only consumes fields where the
//! payload byte mapping is unambiguous (battery V, motor V, board temp,
//! fault bits, branch currents). The "total current" sample that the
//! manual describes as living in the ID itself is omitted — the four
//! per-branch currents from a Type-04 query give you the same number
//! summed, with a clean encoding.
//!
//! ## Why not auto-report?
//!
//! The board supports an "auto-report" mode (CMD 0x06) that pushes
//! Type-03 frames at a configurable interval. We instead poll at 1 Hz
//! from `safety::power_monitor`, which keeps the bus quiet, lets us
//! tag a host-side wall-clock to each sample, and means a missing
//! response shows up as `feedback_stale` rather than a missed push
//! we'd never notice.

use crate::can_interface::{CanInterface, ReceivedFrame};
use anyhow::Result;
use tracing::{debug, trace};

/// CAN communication types (bits 28..24 of the 29-bit ID).
mod cmd {
    /// Control + query frame: enable/disable rails, restart, clear faults.
    pub const CONTROL: u8 = 0x01;
    /// Query frame. `data[0]` selects which response we want:
    ///   - `0x00` → board returns a Type-03 status frame
    ///   - `0xCA` → board returns a Type-04 per-branch current frame
    pub const QUERY: u8 = 0x02;
    /// Status response (battery V, motor V, temp, fault word).
    pub const STATUS: u8 = 0x03;
    /// Per-branch current response (AL / AR / LL / LR amps).
    pub const CURRENTS: u8 = 0x04;
    /// Version query / response. `data[0..]` ASCII "PBVx.yz".
    pub const VERSION: u8 = 0x05;
    /// Auto-report enable/disable: `data[0] = 0x01` on, `0x00` off.
    pub const AUTO_REPORT: u8 = 0x06;
    /// Read parameter (data[0..2] = LE index).
    pub const PARAM_READ: u8 = 0x07;
    /// Write parameter (data[0..2] = index, data[4..8] = LE value).
    pub const PARAM_WRITE: u8 = 0x08;
}

/// Selector byte for `CONTROL` (0x01) frames.
mod ctrl {
    pub const ENABLE: u8 = 0xA1;
    pub const DISABLE: u8 = 0xA0;
}

/// Default board address per manual §六.七 (parameter 6006, default 0xAA).
pub const DEFAULT_POWER_ID: u8 = 0xAA;

/// Sentinel selectors for `query()` / response routing.
const QUERY_STATUS: u8 = 0x00;
const QUERY_CURRENTS: u8 = 0xCA;

/// Driver handle for one PowerBoard.
///
/// Stateless apart from `power_id`; all live data lives in
/// [`PowerBoardSnapshot`] which is owned by the supervisor.
#[derive(Debug, Clone, Copy)]
pub struct PowerBoard {
    pub power_id: u8,
}

impl PowerBoard {
    pub fn new(power_id: u8) -> Self {
        Self { power_id }
    }

    /// Build the 29-bit extended CAN ID for an outgoing frame.
    fn make_can_id(&self, comm_type: u8, data_area2: u16) -> u32 {
        ((comm_type as u32) << 24) | ((data_area2 as u32) << 8) | (self.power_id as u32)
    }

    /// Send a Type-02 status query (`data[0] = 0x00`). Power board will
    /// respond with one Type-03 frame containing battery + motor V, board
    /// temperature, and the 24-bit fault word.
    pub fn query_status(&self, can: &CanInterface) -> Result<()> {
        let id = self.make_can_id(cmd::QUERY, 0);
        let mut data = [0u8; 8];
        data[0] = QUERY_STATUS;
        can.send_extended(id, &data)?;
        trace!(power_id = self.power_id, "powerboard query_status TX");
        Ok(())
    }

    /// Send a Type-02 current query (`data[0] = 0xCA`). Power board will
    /// respond with one Type-04 frame containing the four branch currents.
    pub fn query_currents(&self, can: &CanInterface) -> Result<()> {
        let id = self.make_can_id(cmd::QUERY, 0);
        let mut data = [0u8; 8];
        data[0] = QUERY_CURRENTS;
        can.send_extended(id, &data)?;
        trace!(power_id = self.power_id, "powerboard query_currents TX");
        Ok(())
    }

    /// Ask the board for its firmware version. Reply is one Type-05 frame
    /// whose 8-byte payload holds an ASCII string like `"PBV1.00 "`.
    pub fn query_version(&self, can: &CanInterface) -> Result<()> {
        let id = self.make_can_id(cmd::VERSION, 0);
        let data = [0u8; 8];
        can.send_extended(id, &data)?;
        Ok(())
    }

    /// Enable / disable the board's auto-report Type-03 stream. We
    /// generally leave this off and poll instead (see module docs).
    pub fn set_auto_report(&self, can: &CanInterface, enabled: bool) -> Result<()> {
        let id = self.make_can_id(cmd::AUTO_REPORT, 0);
        let mut data = [0u8; 8];
        data[0] = if enabled { 0x01 } else { 0x00 };
        can.send_extended(id, &data)?;
        debug!(
            power_id = self.power_id,
            enabled, "powerboard auto-report toggled"
        );
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

/// Decoded contents of one Type-03 status response.
#[derive(Debug, Clone)]
pub struct PowerStatus {
    pub power_id: u8,
    /// Battery / VBUS voltage in volts. Per manual: raw `u16` big-endian
    /// in payload bytes 0..2, scale = 1/100 V (0 → 0 V, 65535 → 655.35 V).
    pub battery_voltage_v: f32,
    /// Motor / VMBUS voltage in volts. Same encoding as battery_voltage.
    pub motor_voltage_v: f32,
    /// Board temperature in °C (payload byte 4, 1 °C per LSB).
    pub board_temperature_c: f32,
    /// 24-bit fault word from payload bytes 5..8 (little-endian).
    /// See [`describe_faults`] for human-readable bit names.
    pub fault_bits: u32,
    /// Whether the soft-start FET is currently on (bit 21 of the fault word).
    pub soft_start_on: bool,
    /// VMBUS rail enabled (bit 22).
    pub motor_rail_on: bool,
    /// 24 V rail enabled (bit 23).
    pub rail_24v_on: bool,
    /// 12 V rail enabled (bit 20).
    pub rail_12v_on: bool,
}

/// Decoded contents of one Type-04 per-branch current response.
#[derive(Debug, Clone, Copy, Default)]
pub struct PowerCurrents {
    /// Front-left motor branch (per manual labelling AL / AR / LL / LR).
    pub al_current_a: f32,
    pub ar_current_a: f32,
    pub ll_current_a: f32,
    pub lr_current_a: f32,
}

/// Outcome of running [`parse_frame`] over a raw extended CAN frame.
#[derive(Debug, Clone)]
pub enum PowerFrame {
    Status(PowerStatus),
    Currents {
        power_id: u8,
        currents: PowerCurrents,
    },
    Version {
        power_id: u8,
        version: String,
    },
}

/// Try to interpret `frame` as a PowerBoard response. Returns `None` if
/// the frame doesn't look like one of ours (wrong comm-type, wrong
/// source POWER_ID, or short payload).
///
/// The board encodes the source `POWER_ID` in bits 16..23 of the
/// response ID (verified live: a 0xAA board responds with
/// `0x03AA0000`, `0x04AA0000`, …). We filter on that — the low 16 bits
/// are the host/target address which is typically zero.
pub fn parse_frame(frame: &ReceivedFrame, power_id: u8) -> Option<PowerFrame> {
    if !frame.is_extended || frame.data.len() < 8 {
        return None;
    }

    let comm_type = ((frame.id >> 24) & 0x1F) as u8;
    let source_id = ((frame.id >> 16) & 0xFF) as u8;
    if source_id != power_id {
        return None;
    }

    match comm_type {
        cmd::STATUS => Some(PowerFrame::Status(parse_status(power_id, &frame.data))),
        cmd::CURRENTS => Some(PowerFrame::Currents {
            power_id,
            currents: parse_currents(&frame.data),
        }),
        cmd::VERSION => Some(PowerFrame::Version {
            power_id,
            version: parse_version_payload(&frame.data),
        }),
        _ => None,
    }
}

fn parse_status(power_id: u8, payload: &[u8]) -> PowerStatus {
    debug_assert!(payload.len() >= 8);
    let battery_raw = u16::from_be_bytes([payload[0], payload[1]]);
    let motor_raw = u16::from_be_bytes([payload[2], payload[3]]);
    let temp_raw = payload[4];

    // 24-bit fault / status word from payload bytes 5..8, **big-endian**
    // (verified against a healthy live board: payload `… 23 D0 00 00`
    // decodes to 0xD00000 = bits 21+22+23 set = soft-start + VMBUS +
    // 24V rails on, no actual faults — which matches the operating
    // state of a powered-up board with all rails enabled).
    //
    // Bit layout per User Manual V1.3:
    //   bits 0..15 : fault flags (overcurrent, OV, UV, …)
    //   bit 20     : 12V rail on/off
    //   bit 21     : soft-start on/off
    //   bit 22     : VMBUS rail on/off
    //   bit 23     : 24V rail on/off
    let fault_bits = ((payload[5] as u32) << 16) | ((payload[6] as u32) << 8) | (payload[7] as u32);

    PowerStatus {
        power_id,
        battery_voltage_v: battery_raw as f32 / 100.0,
        motor_voltage_v: motor_raw as f32 / 100.0,
        board_temperature_c: temp_raw as f32,
        fault_bits,
        rail_12v_on: (fault_bits >> 20) & 1 != 0,
        soft_start_on: (fault_bits >> 21) & 1 != 0,
        motor_rail_on: (fault_bits >> 22) & 1 != 0,
        rail_24v_on: (fault_bits >> 23) & 1 != 0,
    }
}

fn parse_currents(payload: &[u8]) -> PowerCurrents {
    debug_assert!(payload.len() >= 8);
    // Manual: 0..65535 → 0..655.36A per branch, big-endian per byte pair.
    let al = u16::from_be_bytes([payload[0], payload[1]]) as f32 / 100.0;
    let ar = u16::from_be_bytes([payload[2], payload[3]]) as f32 / 100.0;
    let ll = u16::from_be_bytes([payload[4], payload[5]]) as f32 / 100.0;
    let lr = u16::from_be_bytes([payload[6], payload[7]]) as f32 / 100.0;
    PowerCurrents {
        al_current_a: al,
        ar_current_a: ar,
        ll_current_a: ll,
        lr_current_a: lr,
    }
}

fn parse_version_payload(payload: &[u8]) -> String {
    // The board responds with an ASCII version string like "PBV1.00 "
    // (whitespace-padded to 8 bytes). Trim trailing nuls / spaces.
    String::from_utf8_lossy(payload)
        .trim_end_matches(|c: char| c.is_whitespace() || c == '\0')
        .to_string()
}

// ---------------------------------------------------------------------------
// Fault decoding
// ---------------------------------------------------------------------------

/// Friendly human-readable string for a Type-03 fault word. Returns
/// `"normal"` when there are no fault bits set (status bits 20..23 are
/// rail on/off indicators, not faults — they're masked out).
pub fn describe_faults(bits: u32) -> String {
    // Mask out the rail status bits (20..23); they're not faults.
    let fault_only = bits & 0x000F_FFFF;
    if fault_only == 0 {
        return "normal".to_string();
    }
    let mut parts = Vec::new();
    if fault_only & (1 << 0) != 0 {
        parts.push("powerchip_overcurrent");
    }
    if fault_only & (1 << 1) != 0 {
        parts.push("powerchip_overtemp");
    }
    if fault_only & (1 << 2) != 0 {
        parts.push("board_overtemp_100c");
    }
    if fault_only & (1 << 3) != 0 {
        parts.push("sample_overcurrent");
    }
    if fault_only & (1 << 4) != 0 {
        parts.push("vbus_overvoltage");
    }
    if fault_only & (1 << 5) != 0 {
        parts.push("vbus_undervoltage");
    }
    if fault_only & (1 << 6) != 0 {
        parts.push("vmbus_overvoltage");
    }
    if fault_only & (1 << 7) != 0 {
        parts.push("vmbus_undervoltage");
    }
    if fault_only & (1 << 8) != 0 {
        parts.push("rail24v_overvoltage");
    }
    if fault_only & (1 << 9) != 0 {
        parts.push("rail24v_undervoltage");
    }
    if fault_only & (1 << 10) != 0 {
        parts.push("rail12v_overvoltage");
    }
    if fault_only & (1 << 11) != 0 {
        parts.push("rail12v_undervoltage");
    }
    if fault_only & (1 << 12) != 0 {
        parts.push("output_branch_overcurrent");
    }
    if fault_only & (1 << 13) != 0 {
        parts.push("ina238_fault");
    }
    if fault_only & (1 << 14) != 0 {
        parts.push("softstart_fault");
    }
    if parts.is_empty() {
        format!("0x{fault_only:06X}")
    } else {
        parts.join(",")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn id_layout_round_trip() {
        let pb = PowerBoard::new(0xAA);
        // Type 02, data2 = 0
        let id = pb.make_can_id(cmd::QUERY, 0);
        assert_eq!(id & 0xFF, 0xAA);
        assert_eq!((id >> 24) & 0x1F, 0x02);
    }

    #[test]
    fn parse_status_decodes_battery_voltage() {
        // 5460 raw / 100 = 54.60 V — a fully charged 13s Li-ion at 4.20 V/cell.
        let mut payload = [0u8; 8];
        payload[0..2].copy_from_slice(&5460u16.to_be_bytes());
        payload[2..4].copy_from_slice(&5440u16.to_be_bytes());
        payload[4] = 32; // 32 °C
                         // Set fault bit 5 = vbus undervoltage. With BE byte order the
                         // low fault byte lives at payload[7].
        payload[7] = 0b0010_0000;

        let s = parse_status(0xAA, &payload);
        assert!((s.battery_voltage_v - 54.60).abs() < 1e-3);
        assert!((s.motor_voltage_v - 54.40).abs() < 1e-3);
        assert!((s.board_temperature_c - 32.0).abs() < 1e-3);
        assert!(s.fault_bits & (1 << 5) != 0);
        assert!(!s.rail_24v_on);
    }

    #[test]
    fn fault_word_status_bits_arent_treated_as_faults() {
        // Only the "12V on" rail-status bit set (bit 20).
        // With BE byte order, bit 20 (0x100000) lives in payload[5] bit 4.
        let mut payload = [0u8; 8];
        payload[5] = 0x10;
        let s = parse_status(0xAA, &payload);
        assert!(s.rail_12v_on);
        assert_eq!(describe_faults(s.fault_bits), "normal");
    }

    #[test]
    fn parse_status_matches_live_board_capture() {
        // Real candump capture from a healthy 13s2p Bebop V2 pack:
        //   RX 03AA0000  [8]  12 10 12 0C 23 D0 00 00
        //
        // Expected decode:
        //   battery V = 0x1210 / 100 = 46.24 V
        //   motor V   = 0x120C / 100 = 46.20 V
        //   temp      = 0x23 = 35 °C
        //   fault BE  = 0xD00000 → bits 20+22+23 set: 12V rail, VMBUS
        //               rail and 24V rail all ON; soft-start (bit 21)
        //               OFF (pre-charge is already complete on a
        //               powered-up board, so the FET hands off to the
        //               main rail). No real faults in bits 0..15.
        let payload = [0x12, 0x10, 0x12, 0x0C, 0x23, 0xD0, 0x00, 0x00];
        let s = parse_status(0xAA, &payload);
        assert!(
            (s.battery_voltage_v - 46.24).abs() < 1e-3,
            "v={}",
            s.battery_voltage_v
        );
        assert!((s.motor_voltage_v - 46.20).abs() < 1e-3);
        assert_eq!(s.board_temperature_c, 35.0);
        assert!(s.rail_12v_on);
        assert!(s.motor_rail_on);
        assert!(s.rail_24v_on);
        assert!(!s.soft_start_on);
        assert_eq!(describe_faults(s.fault_bits), "normal");
    }

    #[test]
    fn parse_frame_routes_live_status_response() {
        // Same candump capture wrapped in a `ReceivedFrame`. Confirms
        // the bits-16..23 source-id filter accepts the real board's
        // response (which puts POWER_ID = 0xAA at bits 16..23, NOT at
        // bits 0..7 / 8..15 like the manual's outgoing frames).
        let frame = ReceivedFrame {
            id: 0x03AA_0000,
            is_extended: true,
            data: vec![0x12, 0x10, 0x12, 0x0C, 0x23, 0xD0, 0x00, 0x00],
        };
        match parse_frame(&frame, 0xAA) {
            Some(PowerFrame::Status(s)) => {
                assert!((s.battery_voltage_v - 46.24).abs() < 1e-3);
            }
            other => panic!("expected Status, got {other:?}"),
        }
    }

    #[test]
    fn parse_frame_rejects_other_power_ids() {
        // A response from a different POWER_ID on the same bus must
        // not be misattributed to ours.
        let frame = ReceivedFrame {
            id: 0x03BB_0000,
            is_extended: true,
            data: vec![0; 8],
        };
        assert!(parse_frame(&frame, 0xAA).is_none());
    }

    #[test]
    fn parse_frame_routes_live_currents_response() {
        // Real capture: RX 04AA0000  [8]  00 02 00 02 00 00 00 00
        // Decodes to AL=0.02A AR=0.02A LL=0 LR=0 (board at idle,
        // tiny quiescent draw on the front-half rails).
        let frame = ReceivedFrame {
            id: 0x04AA_0000,
            is_extended: true,
            data: vec![0x00, 0x02, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00],
        };
        match parse_frame(&frame, 0xAA) {
            Some(PowerFrame::Currents { currents, .. }) => {
                assert!((currents.al_current_a - 0.02).abs() < 1e-4);
                assert!((currents.ar_current_a - 0.02).abs() < 1e-4);
                assert_eq!(currents.ll_current_a, 0.0);
                assert_eq!(currents.lr_current_a, 0.0);
            }
            other => panic!("expected Currents, got {other:?}"),
        }
    }

    #[test]
    fn parse_currents_decodes_all_four_branches() {
        let mut payload = [0u8; 8];
        payload[0..2].copy_from_slice(&1500u16.to_be_bytes()); // 15.00 A
        payload[2..4].copy_from_slice(&1234u16.to_be_bytes()); // 12.34 A
        payload[4..6].copy_from_slice(&500u16.to_be_bytes()); //  5.00 A
        payload[6..8].copy_from_slice(&0u16.to_be_bytes());
        let c = parse_currents(&payload);
        assert!((c.al_current_a - 15.00).abs() < 1e-3);
        assert!((c.ar_current_a - 12.34).abs() < 1e-3);
        assert!((c.ll_current_a - 5.00).abs() < 1e-3);
        assert_eq!(c.lr_current_a, 0.0);
    }

    #[test]
    fn version_payload_trims_padding() {
        let payload = b"PBV1.00 ";
        assert_eq!(parse_version_payload(payload), "PBV1.00");
    }
}
