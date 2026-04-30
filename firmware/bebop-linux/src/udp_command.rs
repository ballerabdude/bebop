//! UDP command interface for remote control
//!
//! Receives velocity commands over UDP from a remote controller.
//! Protocol: JSON messages over UDP port 10000.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::net::UdpSocket;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use tracing::{debug, info, trace, warn};

use crate::observation::VelocityCommand;

/// UDP command message format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommandMessage {
    /// Message type (optional, for future expansion)
    #[serde(rename = "type", default)]
    pub message_type: Option<String>,

    /// Velocity commands (can be nested under "commands" or at root)
    #[serde(flatten)]
    pub commands: CommandData,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CommandData {
    /// Forward/backward velocity (m/s)
    #[serde(default, alias = "linear_x", alias = "vx")]
    pub xvel: f32,

    /// Left/right velocity (m/s)
    #[serde(default, alias = "linear_y", alias = "vy")]
    pub yvel: f32,

    /// Yaw rate (rad/s)
    #[serde(default, alias = "angular_z", alias = "wz")]
    pub angvel: f32,
}

/// Shared command state
#[derive(Debug, Clone)]
pub struct CommandState {
    pub cmd: VelocityCommand,
    pub last_update: Instant,
    pub connected: bool,
}

impl Default for CommandState {
    fn default() -> Self {
        Self {
            cmd: VelocityCommand::default(),
            last_update: Instant::now(),
            connected: false,
        }
    }
}

/// UDP command listener
pub struct UdpCommandListener {
    state: Arc<Mutex<CommandState>>,
    port: u16,
    timeout_ms: u64,
}

impl UdpCommandListener {
    /// Create a new UDP command listener
    pub fn new(port: u16) -> Self {
        Self {
            state: Arc::new(Mutex::new(CommandState::default())),
            port,
            timeout_ms: 500,
        }
    }

    /// Set command timeout (ms)
    pub fn set_timeout(&mut self, timeout_ms: u64) {
        self.timeout_ms = timeout_ms;
    }

    /// Get shared state handle
    pub fn state_handle(&self) -> Arc<Mutex<CommandState>> {
        Arc::clone(&self.state)
    }

    /// Start the listener in a background thread
    pub fn start(&self) -> Result<()> {
        let socket = UdpSocket::bind(format!("0.0.0.0:{}", self.port))?;
        socket.set_read_timeout(Some(Duration::from_millis(100)))?;

        info!("UDP command listener started on port {}", self.port);

        let state = Arc::clone(&self.state);
        let timeout_ms = self.timeout_ms;

        thread::spawn(move || {
            let mut buf = [0u8; 1024];

            loop {
                match socket.recv_from(&mut buf) {
                    Ok((len, addr)) => {
                        if let Ok(text) = std::str::from_utf8(&buf[..len]) {
                            match serde_json::from_str::<CommandMessage>(text) {
                                Ok(msg) => {
                                    // Handle reset command
                                    if msg.message_type.as_deref() == Some("reset") {
                                        if let Ok(mut state) = state.lock() {
                                            state.cmd = VelocityCommand::default();
                                            state.last_update = Instant::now();
                                            debug!("Command reset");
                                        }
                                        continue;
                                    }

                                    // Update command state
                                    if let Ok(mut state) = state.lock() {
                                        state.cmd = VelocityCommand {
                                            linear_x: msg.commands.xvel,
                                            linear_y: msg.commands.yvel,
                                            angular_z: msg.commands.angvel,
                                        };
                                        state.last_update = Instant::now();
                                        state.connected = true;

                                        trace!(
                                            "UDP from {}: vx={:.2} vy={:.2} wz={:.2}",
                                            addr,
                                            state.cmd.linear_x,
                                            state.cmd.linear_y,
                                            state.cmd.angular_z
                                        );
                                    }
                                }
                                Err(e) => {
                                    debug!("Failed to parse UDP message: {}", e);
                                }
                            }
                        }
                    }
                    Err(e) => {
                        if e.kind() != std::io::ErrorKind::WouldBlock
                            && e.kind() != std::io::ErrorKind::TimedOut
                        {
                            warn!("UDP receive error: {}", e);
                        }
                    }
                }

                // Check timeout
                if let Ok(mut state) = state.lock() {
                    if state.last_update.elapsed() > Duration::from_millis(timeout_ms) {
                        if state.connected {
                            debug!("Command timeout, zeroing commands");
                            state.connected = false;
                        }
                        state.cmd = VelocityCommand::default();
                    }
                }
            }
        });

        Ok(())
    }

    /// Get current command (thread-safe)
    pub fn get_command(&self) -> VelocityCommand {
        self.state
            .lock()
            .map(|s| s.cmd.clone())
            .unwrap_or_default()
    }

    /// Check if connected (receiving commands)
    pub fn is_connected(&self) -> bool {
        self.state
            .lock()
            .map(|s| s.connected)
            .unwrap_or(false)
    }
}

/// Async UDP command listener using tokio
pub struct AsyncUdpListener {
    state: Arc<tokio::sync::Mutex<CommandState>>,
    port: u16,
    timeout_ms: u64,
}

impl AsyncUdpListener {
    /// Create a new async UDP listener
    pub fn new(port: u16) -> Self {
        Self {
            state: Arc::new(tokio::sync::Mutex::new(CommandState::default())),
            port,
            timeout_ms: 500,
        }
    }

    /// Get shared state handle
    pub fn state_handle(&self) -> Arc<tokio::sync::Mutex<CommandState>> {
        Arc::clone(&self.state)
    }

    /// Start the listener as a tokio task
    pub async fn start(&self) -> Result<()> {
        let socket = tokio::net::UdpSocket::bind(format!("0.0.0.0:{}", self.port)).await?;

        info!("Async UDP command listener started on port {}", self.port);

        let state = Arc::clone(&self.state);
        let timeout_ms = self.timeout_ms;

        tokio::spawn(async move {
            let mut buf = [0u8; 1024];

            loop {
                let timeout = tokio::time::timeout(
                    Duration::from_millis(100),
                    socket.recv_from(&mut buf),
                );

                match timeout.await {
                    Ok(Ok((len, addr))) => {
                        if let Ok(text) = std::str::from_utf8(&buf[..len]) {
                            if let Ok(msg) = serde_json::from_str::<CommandMessage>(text) {
                                // Handle reset
                                if msg.message_type.as_deref() == Some("reset") {
                                    let mut state = state.lock().await;
                                    state.cmd = VelocityCommand::default();
                                    state.last_update = Instant::now();
                                    continue;
                                }

                                // Update command
                                let mut state = state.lock().await;
                                state.cmd = VelocityCommand {
                                    linear_x: msg.commands.xvel,
                                    linear_y: msg.commands.yvel,
                                    angular_z: msg.commands.angvel,
                                };
                                state.last_update = Instant::now();
                                state.connected = true;

                                trace!(
                                    "UDP from {}: vx={:.2} vy={:.2} wz={:.2}",
                                    addr,
                                    state.cmd.linear_x,
                                    state.cmd.linear_y,
                                    state.cmd.angular_z
                                );
                            }
                        }
                    }
                    Ok(Err(e)) => {
                        warn!("UDP receive error: {}", e);
                    }
                    Err(_) => {
                        // Timeout - check for command timeout
                        let mut state = state.lock().await;
                        if state.last_update.elapsed() > Duration::from_millis(timeout_ms) {
                            if state.connected {
                                debug!("Command timeout");
                                state.connected = false;
                            }
                            state.cmd = VelocityCommand::default();
                        }
                    }
                }
            }
        });

        Ok(())
    }

    /// Get current command
    pub async fn get_command(&self) -> VelocityCommand {
        self.state.lock().await.cmd.clone()
    }

    /// Check if connected
    pub async fn is_connected(&self) -> bool {
        self.state.lock().await.connected
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_command() {
        let json = r#"{"xvel": 0.5, "yvel": 0.0, "angvel": 0.1}"#;
        let msg: CommandMessage = serde_json::from_str(json).unwrap();
        assert!((msg.commands.xvel - 0.5).abs() < 0.01);
        assert!((msg.commands.angvel - 0.1).abs() < 0.01);
    }

    #[test]
    fn test_parse_nested_command() {
        let json = r#"{"type": "cmd", "xvel": 1.0, "yvel": -0.5, "angvel": 0.0}"#;
        let msg: CommandMessage = serde_json::from_str(json).unwrap();
        assert!((msg.commands.xvel - 1.0).abs() < 0.01);
        assert!((msg.commands.yvel - (-0.5)).abs() < 0.01);
    }

    #[test]
    fn test_parse_reset() {
        let json = r#"{"type": "reset"}"#;
        let msg: CommandMessage = serde_json::from_str(json).unwrap();
        assert_eq!(msg.message_type.as_deref(), Some("reset"));
    }
}
