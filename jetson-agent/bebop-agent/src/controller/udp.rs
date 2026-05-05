//! Thin wrapper around a connected UDP socket that emits the JSON
//! teleop packets [`firmware/bebop-linux`] expects on port 10000.
//!
//! Wire format (per `firmware/bebop-linux/README.md`):
//!
//! ```json
//! {"xvel": 0.5, "yvel": 0.0, "angvel": 0.1}
//! {"type": "reset"}
//! ```

use anyhow::{Context, Result};
use serde::Serialize;
use tokio::net::UdpSocket;

use super::teleop::TeleopCmd;

/// Wire-shape for the one-shot reset frame. Defined here (rather than
/// in `teleop.rs`) because `teleop.rs` is meant to stay free of
/// serialisation concerns.
#[derive(Serialize)]
struct ResetFrame<'a> {
    #[serde(rename = "type")]
    kind: &'a str,
}

/// Owns a connected UDP socket. `connect` so subsequent `send`s skip
/// the destination resolution path.
pub struct TeleopSink {
    socket: UdpSocket,
    target: String,
}

impl TeleopSink {
    pub async fn connect(target: &str) -> Result<Self> {
        let socket = UdpSocket::bind("0.0.0.0:0")
            .await
            .context("bind UDP source socket")?;
        socket
            .connect(target)
            .await
            .with_context(|| format!("connect UDP socket to {target}"))?;
        Ok(Self {
            socket,
            target: target.to_owned(),
        })
    }

    pub fn target(&self) -> &str {
        &self.target
    }

    /// Send a velocity command. Errors here are treated as transient by
    /// the supervisor — we log and keep going.
    pub async fn send_velocity(&self, cmd: &TeleopCmd) -> Result<()> {
        let bytes = serde_json::to_vec(cmd).context("serialise TeleopCmd")?;
        self.socket
            .send(&bytes)
            .await
            .context("UDP send velocity")?;
        Ok(())
    }

    /// Send the one-shot `{"type":"reset"}` frame the firmware uses
    /// to clear its internal command state on e-stop entry.
    pub async fn send_reset(&self) -> Result<()> {
        let bytes =
            serde_json::to_vec(&ResetFrame { kind: "reset" }).context("serialise ResetFrame")?;
        self.socket.send(&bytes).await.context("UDP send reset")?;
        Ok(())
    }
}
