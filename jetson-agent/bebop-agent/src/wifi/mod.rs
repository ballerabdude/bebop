//! Wi-Fi provisioning, implemented by shelling out to NetworkManager's
//! `nmcli`. This is the path NVIDIA's default L4T images already support,
//! and it avoids depending on a specific NM D-Bus crate version.
//!
//! All public functions in this module are safe to call from the BLE
//! dispatcher.

use anyhow::{Context, Result};
use tokio::process::Command;

use crate::error::AgentError;
use crate::state::{AppState, WifiRuntimeStatus};

/// A scanned Wi-Fi network.
#[derive(Debug, Clone)]
pub struct Network {
    pub ssid: String,
    pub signal_dbm: i32,
    pub security: String,
    pub saved: bool,
}

impl From<Network> for bebop_proto::v1::WifiNetwork {
    fn from(n: Network) -> Self {
        Self {
            ssid: n.ssid,
            signal_dbm: n.signal_dbm,
            security: n.security,
            saved: n.saved,
        }
    }
}

impl From<WifiRuntimeStatus> for bebop_proto::v1::WifiStatus {
    fn from(s: WifiRuntimeStatus) -> Self {
        Self {
            connected: s.connected,
            ssid: s.ssid,
            ip_address: s.ip_address,
            signal_dbm: s.signal_dbm,
        }
    }
}

/// Trigger a scan and parse the results.
pub async fn scan(_state: &AppState) -> Result<Vec<Network>, AgentError> {
    let out = nmcli(&[
        "-t",
        "-f",
        "IN-USE,SSID,SIGNAL,SECURITY",
        "device",
        "wifi",
        "list",
        "--rescan",
        "yes",
    ])
    .await
    .map_err(|e| AgentError::Wifi(e.to_string()))?;

    let mut nets = Vec::new();
    for line in out.lines() {
        // `-t` terminal output is colon separated; escape sequences use '\:'.
        let fields = split_nmcli(line);
        if fields.len() < 4 {
            continue;
        }
        let ssid = fields[1].trim().to_owned();
        if ssid.is_empty() {
            continue;
        }
        let signal_percent: i32 = fields[2].trim().parse().unwrap_or(0);
        // NM reports signal as 0-100; approximate dBm.
        let signal_dbm = percent_to_dbm(signal_percent);
        let security = fields[3].trim().to_owned();
        nets.push(Network {
            ssid,
            signal_dbm,
            security: if security.is_empty() {
                "OPEN".into()
            } else {
                security
            },
            saved: false,
        });
    }

    Ok(nets)
}

/// Connect to a Wi-Fi network, persisting the connection profile.
pub async fn connect(
    state: &AppState,
    ssid: &str,
    password: &str,
    hidden: bool,
) -> Result<WifiRuntimeStatus, AgentError> {
    let mut args: Vec<&str> = vec!["device", "wifi", "connect", ssid];
    if !password.is_empty() {
        args.push("password");
        args.push(password);
    }
    if hidden {
        args.push("hidden");
        args.push("yes");
    }

    nmcli(&args)
        .await
        .map_err(|e| AgentError::Wifi(e.to_string()))?;

    let status = query_status().await.unwrap_or_default();
    state.set_wifi_status(status.clone()).await;
    Ok(status)
}

/// Read current Wi-Fi status from NetworkManager.
pub async fn query_status() -> Result<WifiRuntimeStatus> {
    let out = nmcli(&["-t", "-f", "DEVICE,TYPE,STATE,CONNECTION", "device"]).await?;
    let mut wifi_dev = None;
    for line in out.lines() {
        let fields = split_nmcli(line);
        if fields.len() >= 4 && fields[1] == "wifi" {
            wifi_dev = Some((
                fields[0].to_owned(),
                fields[2].to_owned(),
                fields[3].to_owned(),
            ));
            break;
        }
    }
    let Some((device, state, connection)) = wifi_dev else {
        return Ok(WifiRuntimeStatus::default());
    };

    let connected = state == "connected";
    let ip_address = if connected {
        ip_of(&device).await.unwrap_or_default()
    } else {
        String::new()
    };

    Ok(WifiRuntimeStatus {
        connected,
        ssid: if connected { connection } else { String::new() },
        ip_address,
        signal_dbm: 0,
    })
}

async fn ip_of(device: &str) -> Result<String> {
    let out = nmcli(&["-g", "IP4.ADDRESS", "device", "show", device]).await?;
    Ok(out.lines().next().unwrap_or("").trim().to_owned())
}

async fn nmcli(args: &[&str]) -> Result<String> {
    let output = Command::new("nmcli")
        .args(args)
        .output()
        .await
        .context("spawn nmcli")?;
    if !output.status.success() {
        anyhow::bail!(
            "nmcli {:?} failed: {}",
            args,
            String::from_utf8_lossy(&output.stderr)
        );
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

fn split_nmcli(line: &str) -> Vec<String> {
    // Split on unescaped ':', honouring `\:` escapes.
    let mut parts = Vec::new();
    let mut current = String::new();
    let mut chars = line.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\\' {
            if let Some(&next) = chars.peek() {
                current.push(next);
                chars.next();
                continue;
            }
        }
        if c == ':' {
            parts.push(std::mem::take(&mut current));
        } else {
            current.push(c);
        }
    }
    parts.push(current);
    parts
}

fn percent_to_dbm(percent: i32) -> i32 {
    // Rough linear mapping: 0% -> -100 dBm, 100% -> -50 dBm.
    -100 + percent / 2
}
