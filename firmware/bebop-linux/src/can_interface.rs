//! CAN bus interface using Linux SocketCAN
//!
//! This module provides low-level CAN communication using the socketcan crate.
//! Supports both standard (11-bit) and extended (29-bit) CAN frames.

use crate::config::RobstrideSpecs;
use anyhow::{Context, Result};
use socketcan::{CanFrame, CanSocket, EmbeddedFrame, ExtendedId, Socket, StandardId};
use std::time::Duration;
use tracing::{debug, trace, warn};

/// CAN bus interface wrapper
pub struct CanInterface {
    socket: CanSocket,
    interface_name: String,
}

impl std::fmt::Debug for CanInterface {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CanInterface")
            .field("interface_name", &self.interface_name)
            .finish()
    }
}

impl CanInterface {
    /// Open a CAN interface
    pub fn open(interface: &str) -> Result<Self> {
        let socket = CanSocket::open(interface)
            .with_context(|| format!("Failed to open CAN interface: {}", interface))?;

        // Set non-blocking mode with timeout
        socket
            .set_read_timeout(Duration::from_millis(1))
            .context("Failed to set read timeout")?;

        socket
            .set_write_timeout(Duration::from_millis(10))
            .context("Failed to set write timeout")?;

        debug!("Opened CAN interface: {}", interface);

        Ok(Self {
            socket,
            interface_name: interface.to_string(),
        })
    }

    /// Send an extended frame (29-bit ID) - used by Robstride
    pub fn send_extended(&self, id: u32, data: &[u8]) -> Result<()> {
        let extended_id = ExtendedId::new(id)
            .ok_or_else(|| anyhow::anyhow!("Invalid extended CAN ID: {}", id))?;

        let frame = CanFrame::new(extended_id, data)
            .ok_or_else(|| anyhow::anyhow!("Failed to create CAN frame"))?;

        self.socket
            .write_frame(&frame)
            .with_context(|| format!("Failed to send extended frame to {}", self.interface_name))?;

        trace!("TX EXT [0x{:08X}]: {:02X?}", id, data);

        Ok(())
    }

    /// Send a standard frame (11-bit ID) - used by ODrive
    pub fn send_standard(&self, id: u16, data: &[u8]) -> Result<()> {
        let standard_id = StandardId::new(id)
            .ok_or_else(|| anyhow::anyhow!("Invalid standard CAN ID: {}", id))?;

        let frame = CanFrame::new(standard_id, data)
            .ok_or_else(|| anyhow::anyhow!("Failed to create CAN frame"))?;

        self.socket
            .write_frame(&frame)
            .with_context(|| format!("Failed to send standard frame to {}", self.interface_name))?;

        trace!("TX STD [0x{:03X}]: {:02X?}", id, data);

        Ok(())
    }

    /// Try to receive a frame (non-blocking)
    pub fn try_receive(&self) -> Result<Option<ReceivedFrame>> {
        match self.socket.read_frame() {
            Ok(frame) => {
                let received = ReceivedFrame::from_socketcan_frame(&frame);
                trace!(
                    "RX {} [0x{:08X}]: {:02X?}",
                    if received.is_extended { "EXT" } else { "STD" },
                    received.id,
                    received.data
                );
                Ok(Some(received))
            }
            Err(e) => {
                // Check if it's a timeout (WouldBlock)
                if e.kind() == std::io::ErrorKind::WouldBlock {
                    Ok(None)
                } else {
                    Err(e).context("Failed to receive CAN frame")
                }
            }
        }
    }

    /// Receive with timeout
    pub fn receive_timeout(&self, timeout: Duration) -> Result<Option<ReceivedFrame>> {
        let start = std::time::Instant::now();

        while start.elapsed() < timeout {
            if let Some(frame) = self.try_receive()? {
                return Ok(Some(frame));
            }
            std::thread::sleep(Duration::from_micros(100));
        }

        Ok(None)
    }

    /// Drain all pending frames from receive buffer
    pub fn drain(&self) -> Vec<ReceivedFrame> {
        let mut frames = Vec::new();
        while let Ok(Some(frame)) = self.try_receive() {
            frames.push(frame);
        }
        frames
    }

    /// Get interface name
    pub fn interface_name(&self) -> &str {
        &self.interface_name
    }
}

/// A received CAN frame
#[derive(Debug, Clone)]
pub struct ReceivedFrame {
    pub id: u32,
    pub is_extended: bool,
    pub data: Vec<u8>,
}

impl ReceivedFrame {
    fn from_socketcan_frame(frame: &CanFrame) -> Self {
        let (id, is_extended) = match frame.id() {
            socketcan::Id::Standard(std_id) => (std_id.as_raw() as u32, false),
            socketcan::Id::Extended(ext_id) => (ext_id.as_raw(), true),
        };

        Self {
            id,
            is_extended,
            data: frame.data().to_vec(),
        }
    }

    /// Parse as Robstride feedback frame.
    ///
    /// Note on scaling: position and temperature use universal MIT-mode
    /// ranges (`±4π rad`, `raw / 10.0` °C) and so are pre-decoded into
    /// physical units here. Velocity and torque, on the other hand, are
    /// encoded against per-model full-scale ranges
    /// (`RobstrideSpecs::RSxx.velocity_min/max` / `torque_min/max`); we
    /// can't decode them at this layer because the parser only sees the
    /// motor_id, not the model. They are returned as raw `u16` and must
    /// be decoded by the consumer via [`RobstrideFeedback::velocity`] /
    /// [`RobstrideFeedback::torque`] using the matching spec.
    pub fn parse_robstride(&self) -> Option<RobstrideFeedback> {
        if !self.is_extended || self.data.len() < 8 {
            return None;
        }

        // Extract fields from 29-bit extended ID
        let cmd_type = ((self.id >> 24) & 0x1F) as u8;
        let motor_id = ((self.id >> 8) & 0xFF) as u8;
        let host_id = (self.id & 0xFF) as u8;

        // Fault bits from ID (bits 16-21)
        let fault_bits = ((self.id >> 16) & 0x3F) as u8;

        // Mode status from ID (bits 22-23)
        let mode_status = ((self.id >> 22) & 0x03) as u8;

        // Parse payload (big-endian)
        let position_raw = u16::from_be_bytes([self.data[0], self.data[1]]);
        let velocity_raw = u16::from_be_bytes([self.data[2], self.data[3]]);
        let torque_raw = u16::from_be_bytes([self.data[4], self.data[5]]);
        let temperature_raw = u16::from_be_bytes([self.data[6], self.data[7]]);

        // Universal-range fields decoded inline; per-model fields are
        // forwarded as raw u16 (see struct doc comment).
        let position = uint16_to_float(
            position_raw,
            -4.0 * std::f32::consts::PI,
            4.0 * std::f32::consts::PI,
        );
        let temperature = temperature_raw as f32 / 10.0;

        Some(RobstrideFeedback {
            motor_id,
            cmd_type,
            host_id,
            fault_bits,
            mode_status,
            position,
            velocity_raw,
            torque_raw,
            temperature,
        })
    }

    /// Parse as ODrive encoder estimate frame
    pub fn parse_odrive_encoder(&self) -> Option<ODriveEncoderFeedback> {
        if self.is_extended || self.data.len() < 8 {
            return None;
        }

        let node_id = ((self.id >> 5) & 0x3F) as u8;
        let cmd_id = (self.id & 0x1F) as u8;

        // Only parse encoder estimate frames (cmd_id = 0x09)
        if cmd_id != 0x09 {
            return None;
        }

        // Parse payload (little-endian floats)
        let position_rev =
            f32::from_le_bytes([self.data[0], self.data[1], self.data[2], self.data[3]]);
        let velocity_rev_s =
            f32::from_le_bytes([self.data[4], self.data[5], self.data[6], self.data[7]]);

        // Convert to radians
        let position = position_rev * 2.0 * std::f32::consts::PI;
        let velocity = velocity_rev_s * 2.0 * std::f32::consts::PI;

        Some(ODriveEncoderFeedback {
            node_id,
            position,
            velocity,
        })
    }

    /// Parse as ODrive heartbeat frame
    pub fn parse_odrive_heartbeat(&self) -> Option<ODriveHeartbeat> {
        if self.is_extended || self.data.len() < 8 {
            return None;
        }

        let node_id = ((self.id >> 5) & 0x3F) as u8;
        let cmd_id = (self.id & 0x1F) as u8;

        // Only parse heartbeat frames (cmd_id = 0x01)
        if cmd_id != 0x01 {
            return None;
        }

        let axis_error =
            u32::from_le_bytes([self.data[0], self.data[1], self.data[2], self.data[3]]);
        let axis_state = self.data[4];
        let procedure_result = self.data[5];
        let trajectory_done = (self.data[6] & 0x01) != 0;

        Some(ODriveHeartbeat {
            node_id,
            axis_error,
            axis_state,
            procedure_result,
            trajectory_done,
        })
    }

}

/// MIT-mode style uint16 → float decode.
///
/// Maps `value` linearly from `[0, 65535]` onto `[min, max]`. This is
/// the inverse of the Robstride MIT-mode payload encoding used by
/// [`crate::robstride::RobstrideMotor::send_command`] (see
/// `float_to_uint16` there).
pub(crate) fn uint16_to_float(value: u16, min: f32, max: f32) -> f32 {
    let proportion = value as f32 / 65535.0;
    min + proportion * (max - min)
}

/// Parsed Robstride feedback frame.
///
/// `position` and `temperature` are pre-decoded into physical units
/// because their scaling is universal across all RS0x models. By
/// contrast, `velocity_raw` and `torque_raw` are the **raw** little-
/// fingers off the wire — their full-scale range is per-model
/// (`RobstrideSpecs::RSxx`), and the parser doesn't know the model.
/// Callers must decode them via [`Self::velocity`] / [`Self::torque`]
/// using the matching [`RobstrideSpecs`].
#[derive(Debug, Clone)]
pub struct RobstrideFeedback {
    pub motor_id: u8,
    pub cmd_type: u8,
    pub host_id: u8,
    pub fault_bits: u8,
    pub mode_status: u8,
    pub position: f32,
    /// Raw 16-bit velocity from the MIT-mode feedback frame. Decode
    /// via [`Self::velocity`] with the motor's [`RobstrideSpecs`].
    pub velocity_raw: u16,
    /// Raw 16-bit torque from the MIT-mode feedback frame. Decode via
    /// [`Self::torque`] with the motor's [`RobstrideSpecs`].
    pub torque_raw: u16,
    pub temperature: f32,
}

impl RobstrideFeedback {
    /// Decode `velocity_raw` (rad/s) using the motor model's full-scale.
    pub fn velocity(&self, specs: &RobstrideSpecs) -> f32 {
        uint16_to_float(self.velocity_raw, specs.velocity_min, specs.velocity_max)
    }

    /// Decode `torque_raw` (Nm) using the motor model's full-scale.
    pub fn torque(&self, specs: &RobstrideSpecs) -> f32 {
        uint16_to_float(self.torque_raw, specs.torque_min, specs.torque_max)
    }
}

/// Parsed ODrive encoder feedback
#[derive(Debug, Clone)]
pub struct ODriveEncoderFeedback {
    pub node_id: u8,
    pub position: f32, // radians
    pub velocity: f32, // rad/s
}

/// Parsed ODrive heartbeat
#[derive(Debug, Clone)]
pub struct ODriveHeartbeat {
    pub node_id: u8,
    pub axis_error: u32,
    pub axis_state: u8,
    pub procedure_result: u8,
    pub trajectory_done: bool,
}

/// Multi-bus CAN manager
pub struct CanBusManager {
    buses: Vec<CanInterface>,
}

impl CanBusManager {
    /// Create a new manager with multiple CAN interfaces
    pub fn new(interfaces: &[&str]) -> Result<Self> {
        let mut buses = Vec::new();

        for interface in interfaces {
            match CanInterface::open(interface) {
                Ok(bus) => {
                    buses.push(bus);
                    debug!("Opened CAN interface: {}", interface);
                }
                Err(e) => {
                    warn!("Failed to open CAN interface {}: {}", interface, e);
                }
            }
        }

        if buses.is_empty() {
            return Err(anyhow::anyhow!("No CAN interfaces available"));
        }

        Ok(Self { buses })
    }

    /// Get bus by index
    pub fn get_bus(&self, index: usize) -> Option<&CanInterface> {
        self.buses.get(index)
    }

    /// Get bus by name
    pub fn get_bus_by_name(&self, name: &str) -> Option<&CanInterface> {
        self.buses.iter().find(|b| b.interface_name() == name)
    }

    /// Number of active buses
    pub fn bus_count(&self) -> usize {
        self.buses.len()
    }

    /// Drain all buses
    pub fn drain_all(&self) -> Vec<ReceivedFrame> {
        let mut frames = Vec::new();
        for bus in &self.buses {
            frames.extend(bus.drain());
        }
        frames
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RobstrideModel;

    /// Mirror of `RobstrideMotor::float_to_uint16` (private there). We
    /// duplicate it here intentionally so the test asserts encode/decode
    /// reciprocity *across* implementations — if the two ever drift, a
    /// round-trip assertion will fail loudly.
    fn float_to_uint16(value: f32, min: f32, max: f32) -> u16 {
        let clamped = value.clamp(min, max);
        let proportion = (clamped - min) / (max - min);
        (proportion * 65535.0) as u16
    }

    fn make_feedback(velocity_raw: u16, torque_raw: u16) -> RobstrideFeedback {
        RobstrideFeedback {
            motor_id: 1,
            cmd_type: 0x02,
            host_id: 0xFD,
            fault_bits: 0,
            mode_status: 0x02,
            position: 0.0,
            velocity_raw,
            torque_raw,
            temperature: 25.0,
        }
    }

    /// Encoding a torque with model M's full-scale and decoding with
    /// the same model's full-scale must recover the original value to
    /// within one LSB of quantization.
    #[test]
    fn torque_round_trip_per_model() {
        let one_lsb = |range: f32| range / 65535.0;
        for (model, sample_nm) in [
            (RobstrideModel::RS01, 5.0_f32),
            (RobstrideModel::RS02, 5.0),
            (RobstrideModel::RS03, 30.0),
            (RobstrideModel::RS04, 50.0),
        ] {
            let specs = model.specs();
            let raw = float_to_uint16(sample_nm, specs.torque_min, specs.torque_max);
            let fb = make_feedback(0, raw);
            let decoded = fb.torque(&specs);
            let tol = one_lsb(specs.torque_max - specs.torque_min);
            assert!(
                (decoded - sample_nm).abs() <= tol,
                "{model:?}: encoded {sample_nm} Nm -> raw {raw} -> decoded {decoded} Nm (tol {tol})",
            );
        }
    }

    #[test]
    fn velocity_round_trip_per_model() {
        let one_lsb = |range: f32| range / 65535.0;
        for (model, sample_rad_s) in [
            (RobstrideModel::RS01, 10.0_f32),
            (RobstrideModel::RS02, 10.0),
            (RobstrideModel::RS03, 8.0),
            (RobstrideModel::RS04, 5.0),
        ] {
            let specs = model.specs();
            let raw = float_to_uint16(sample_rad_s, specs.velocity_min, specs.velocity_max);
            let fb = make_feedback(raw, 0);
            let decoded = fb.velocity(&specs);
            let tol = one_lsb(specs.velocity_max - specs.velocity_min);
            assert!(
                (decoded - sample_rad_s).abs() <= tol,
                "{model:?}: encoded {sample_rad_s} rad/s -> raw {raw} -> decoded {decoded} rad/s (tol {tol})",
            );
        }
    }

    /// Cross-model decode must NOT silently look "close enough": a
    /// torque frame encoded against one model's full-scale and decoded
    /// against another's produces a predictable multiplicative error
    /// (the ratio of the two full-scales). Asserting that ratio
    /// explicitly here means any future "let's just hardcode one
    /// range" regression in the parser fails this test loudly instead
    /// of silently inflating telemetry on the smaller motors.
    #[test]
    fn torque_decode_with_wrong_model_scales_by_full_scale_ratio() {
        let true_nm = 5.0_f32;
        let rs02 = RobstrideModel::RS02.specs();
        let rs04 = RobstrideModel::RS04.specs();
        let raw = float_to_uint16(true_nm, rs02.torque_min, rs02.torque_max);
        let fb = make_feedback(0, raw);
        let correct = fb.torque(&rs02);
        let buggy = fb.torque(&rs04);
        // Correct decode lands within 1 LSB of truth.
        assert!((correct - true_nm).abs() <= (rs02.torque_max - rs02.torque_min) / 65535.0);
        // Wrong decode is inflated by the ratio of the full-scales (~7.06×).
        let expected_inflation = (rs04.torque_max - rs04.torque_min) / (rs02.torque_max - rs02.torque_min);
        let observed_inflation = buggy / true_nm;
        assert!(
            (observed_inflation - expected_inflation).abs() < 0.05,
            "buggy decode inflation: expected ~{expected_inflation:.2}×, observed {observed_inflation:.2}× (true={true_nm}, buggy={buggy})",
        );
    }
}
