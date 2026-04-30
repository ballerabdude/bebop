//! CAN bus interface using Linux SocketCAN
//!
//! This module provides low-level CAN communication using the socketcan crate.
//! Supports both standard (11-bit) and extended (29-bit) CAN frames.

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
        let extended_id =
            ExtendedId::new(id).ok_or_else(|| anyhow::anyhow!("Invalid extended CAN ID: {}", id))?;

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

    /// Parse as Robstride feedback frame
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

        // Convert to physical units (Robstride RS04 ranges)
        let position =
            Self::uint16_to_float(position_raw, -4.0 * std::f32::consts::PI, 4.0 * std::f32::consts::PI);
        let velocity = Self::uint16_to_float(velocity_raw, -15.0, 15.0);
        let torque = Self::uint16_to_float(torque_raw, -120.0, 120.0);
        let temperature = temperature_raw as f32 / 10.0;

        Some(RobstrideFeedback {
            motor_id,
            cmd_type,
            host_id,
            fault_bits,
            mode_status,
            position,
            velocity,
            torque,
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
        let position_rev = f32::from_le_bytes([self.data[0], self.data[1], self.data[2], self.data[3]]);
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

        let axis_error = u32::from_le_bytes([self.data[0], self.data[1], self.data[2], self.data[3]]);
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

    /// Convert uint16 to float with given range
    fn uint16_to_float(value: u16, min: f32, max: f32) -> f32 {
        let proportion = value as f32 / 65535.0;
        min + proportion * (max - min)
    }
}

/// Parsed Robstride feedback
#[derive(Debug, Clone)]
pub struct RobstrideFeedback {
    pub motor_id: u8,
    pub cmd_type: u8,
    pub host_id: u8,
    pub fault_bits: u8,
    pub mode_status: u8,
    pub position: f32,
    pub velocity: f32,
    pub torque: f32,
    pub temperature: f32,
}

/// Parsed ODrive encoder feedback
#[derive(Debug, Clone)]
pub struct ODriveEncoderFeedback {
    pub node_id: u8,
    pub position: f32,  // radians
    pub velocity: f32,  // rad/s
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
