//! Bluetooth gamepad subsystem: pair, persist, read evdev, forward
//! body-velocity teleop to bebop-linux over UDP.
//!
//! Public API mirrors the wifi/containers modules:
//!
//!   * [`run`] — supervisor task spawned from `main.rs`.
//!   * [`scan`], [`pair`], [`unpair`], [`status`] — request handlers
//!     called by the BLE dispatcher.
//!
//! On non-Linux builds (macOS/Windows dev) everything stubs out via
//! `stub.rs`, mirroring `ble::server_stub` so `cargo check` keeps
//! working off-target.

pub mod bluez;
pub mod mapping;
pub mod teleop;
pub mod udp;

#[cfg(target_os = "linux")]
pub mod evdev_input;
#[cfg(target_os = "linux")]
mod supervisor;
#[cfg(target_os = "linux")]
pub use supervisor::*;

#[cfg(not(target_os = "linux"))]
mod stub;
#[cfg(not(target_os = "linux"))]
pub use stub::*;

use crate::error::AgentError;
use crate::state::AppState;

// ---------------------------------------------------------------------------
// BLE dispatcher entrypoints — small wrappers so dispatcher.rs doesn't
// have to know about the supervisor's internals.
// ---------------------------------------------------------------------------

/// Discover nearby Bluetooth devices (gamepads + others) for the given
/// duration in milliseconds. Falls back to ~8s if `timeout_ms` is 0.
pub async fn scan(timeout_ms: u32) -> Result<Vec<bluez::DiscoveredDevice>, AgentError> {
    let timeout = std::time::Duration::from_millis(if timeout_ms == 0 {
        8_000
    } else {
        timeout_ms as u64
    });
    bluez::discover(timeout).await
}

/// Pair, trust, connect, and persist `mac` as the bound controller.
/// Triggers a supervisor wake-up on success so the teleop loop picks
/// the device up immediately.
pub async fn pair(state: &AppState, mac: &str) -> Result<(), AgentError> {
    bluez::pair_and_connect(mac).await?;

    // Capture a friendly name from the existing devices listing for
    // UI display. Best-effort; absence isn't fatal.
    let device_name = match bluez::discover(std::time::Duration::from_millis(500)).await {
        Ok(devs) => devs
            .into_iter()
            .find(|d| d.mac.eq_ignore_ascii_case(mac))
            .map(|d| d.name)
            .unwrap_or_default(),
        Err(_) => String::new(),
    };

    persist_pairing(state, mac.to_owned(), device_name.clone())
        .await
        .map_err(|e| AgentError::Controller(e.to_string()))?;

    state
        .update_controller_status(|s| {
            s.paired_mac = mac.to_owned();
            s.device_name = device_name;
            // The supervisor will flip `connected` true once the
            // evdev node is open; pairing alone doesn't guarantee
            // we can read inputs yet.
            s.connected = false;
            s.armed = false;
            s.estop_latched = false;
        })
        .await;

    Ok(())
}

/// Forget `mac` (or the currently bound controller, if `mac` is empty).
pub async fn unpair(state: &AppState, mac: &str) -> Result<(), AgentError> {
    let cfg = state.config().await;
    let target = if mac.is_empty() {
        cfg.controller.paired_mac.clone()
    } else {
        mac.to_owned()
    };
    if target.is_empty() {
        return Err(AgentError::InvalidRequest("no controller paired".into()));
    }
    bluez::unpair(&target).await?;

    if cfg.controller.paired_mac.eq_ignore_ascii_case(&target) {
        persist_pairing(state, String::new(), String::new())
            .await
            .map_err(|e| AgentError::Controller(e.to_string()))?;
        state
            .update_controller_status(|s| {
                s.paired_mac.clear();
                s.device_name.clear();
                s.connected = false;
                s.armed = false;
                s.estop_latched = false;
            })
            .await;
    }
    Ok(())
}

/// Snapshot the current controller status for the dispatcher.
pub async fn status(state: &AppState) -> bebop_proto::v1::ControllerStatus {
    let cfg = state.config().await;
    let s = state.controller_status().await;
    bebop_proto::v1::ControllerStatus {
        enabled: cfg.controller.enabled,
        paired_mac: s.paired_mac,
        device_name: s.device_name,
        connected: s.connected,
        armed: s.armed,
        estop_latched: s.estop_latched,
        last_event_unix_ms: s.last_event_unix_ms,
        target_addr: cfg.controller.target_addr,
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async fn persist_pairing(state: &AppState, mac: String, name: String) -> anyhow::Result<()> {
    use crate::config;

    let mut next = state.config().await;
    next.controller.paired_mac = mac;
    next.controller.device_name = name;
    let path = config::config_path();
    config::save(&next, &path)?;
    state.update_config(|c| *c = next).await;
    Ok(())
}
