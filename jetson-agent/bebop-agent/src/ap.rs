//! SoftAP provisioning fallback.
//!
//! The robot normally joins a known Wi-Fi network as a client. When none is
//! available (first boot, new venue, wrong credentials) this module raises a
//! WPA2 hotspot named `Bebop-<id>` so a phone can join it and reach the setup
//! server on a fixed gateway address.
//!
//! Mode is driven by `[network] mode` in `agent.toml`:
//!   * `auto`   — client first, AP after `ap_auto_after_secs` if still offline.
//!   * `client` — never raise the AP.
//!   * `ap`     — always raise the AP.
//!
//! Bring-up uses NetworkManager (`nmcli`) in shared IPv4 mode, which also
//! provides DHCP/DNS to the phone. The Wi-Fi hardware is a single radio, so
//! client and AP are mutually exclusive — hence the "fallback" design.

use std::time::{Duration, Instant};

use tracing::{info, warn};

use crate::config::NetworkConfig;
use crate::state::AppState;
use crate::wifi;

/// NetworkManager connection profile name for the setup AP. Fixed so we can
/// reliably find/replace it, and so `wifi::query_status` can exclude it from
/// the "client network" check (an active AP otherwise looks `connected`).
pub const AP_CON_NAME: &str = "bebop-setup";

/// Static gateway assigned to the AP interface. Clients get addresses in
/// this /24 via NetworkManager's shared mode.
const AP_GATEWAY: &str = "192.168.42.1";

/// How often the supervisor re-evaluates the desired state.
const TICK: Duration = Duration::from_secs(5);

/// Long-running supervisor. Mirrors the shape of the other subsystems:
/// reconcile desired vs actual state each tick, log transitions once.
pub async fn run(state: AppState) -> anyhow::Result<()> {
    info!("network supervisor online");
    let started = Instant::now();
    let mut last_mode_logged: Option<String> = None;

    loop {
        let cfg = state.config().await;
        let wifi_status = state.wifi_status().await;
        let ap_status = state.ap_status().await;

        let mode = cfg.network.mode.to_ascii_lowercase();
        let should_host = match mode.as_str() {
            "ap" => true,
            "client" => false,
            // Auto: host only once we've given the client path a fair chance.
            _ => {
                !wifi_status.connected
                    && started.elapsed().as_secs() >= cfg.network.ap_auto_after_secs
            }
        } && !ap_status.connecting;

        if should_host && !ap_status.active {
            match raise(&state, &cfg.network).await {
                Ok(()) => {
                    let ssid = state.ap_status().await.ssid;
                    info!(ssid = %ssid, "setup SoftAP raised");
                }
                Err(e) => {
                    warn!(error = %e, "failed to raise setup SoftAP; will retry");
                    state
                        .update_ap_status(|s| {
                            s.active = false;
                            s.last_error = Some(e.to_string());
                        })
                        .await;
                }
            }
        } else if !should_host && ap_status.active {
            if let Err(e) = lower(&state).await {
                warn!(error = %e, "failed to lower setup SoftAP");
            } else {
                info!("setup SoftAP lowered");
            }
        }

        if last_mode_logged.as_deref() != Some(mode.as_str()) {
            info!(
                mode = %mode,
                wifi_connected = wifi_status.connected,
                "network mode active"
            );
            last_mode_logged = Some(mode);
        }

        tokio::time::sleep(TICK).await;
    }
}

/// Create + activate the AP profile. Idempotent: safe to call when already up.
async fn raise(state: &AppState, cfg: &NetworkConfig) -> anyhow::Result<()> {
    let iface = wifi::wifi_device()
        .await
        .map_err(|e| anyhow::anyhow!("no wifi interface: {e}"))?;
    let ssid = cfg.ap_ssid();

    if cfg.ap_password.len() < 8 {
        anyhow::bail!(
            "network.ap_password must be 8..=63 chars for WPA2 (got {})",
            cfg.ap_password.len()
        );
    }

    // Recreate the profile each time so config edits (ssid/password) apply.
    let _ = nmcli(&["con", "delete", AP_CON_NAME]).await;
    nmcli(&[
        "con",
        "add",
        "type",
        "wifi",
        "ifname",
        &iface,
        "con-name",
        AP_CON_NAME,
        "autoconnect",
        "no",
        "ssid",
        &ssid,
        "802-11-wireless.mode",
        "ap",
        "802-11-wireless.band",
        "bg",
        "wifi-sec.key-mgmt",
        "wpa-psk",
        "wifi-sec.psk",
        &cfg.ap_password,
        "ipv4.method",
        "shared",
        "ipv4.addresses",
        &format!("{AP_GATEWAY}/24"),
    ])
    .await?;
    nmcli(&["con", "up", AP_CON_NAME]).await?;

    // Ask NM for the address it actually assigned; fall back to the
    // configured gateway if the query is unavailable.
    let gateway = query_gateway(&iface)
        .await
        .unwrap_or_else(|| AP_GATEWAY.into());
    let port = setup_port(&cfg.setup_bind_addr);
    state
        .update_ap_status(|s| {
            s.active = true;
            s.ssid = ssid.clone();
            s.address = format!("{gateway}:{port}");
            s.last_error = None;
        })
        .await;
    Ok(())
}

/// Deactivate the AP profile (if present). Idempotent.
async fn lower(state: &AppState) -> anyhow::Result<()> {
    let _ = nmcli(&["con", "down", AP_CON_NAME]).await;
    state
        .update_ap_status(|s| {
            s.active = false;
            s.last_error = None;
        })
        .await;
    Ok(())
}

/// Public wrapper for the Wi-Fi-join path: drop the AP before the radio
/// switches to client mode. Does not set `connecting` (the caller owns it).
pub async fn lower_now(state: &AppState) -> anyhow::Result<()> {
    lower(state).await
}

async fn query_gateway(iface: &str) -> Option<String> {
    let out = nmcli(&["-g", "IP4.ADDRESS", "dev", "show", iface])
        .await
        .ok()?;
    let first = out.lines().next()?.trim();
    let ip = first.split('/').next()?.trim();
    if ip.is_empty() || ip == "0.0.0.0" {
        None
    } else {
        Some(ip.to_owned())
    }
}

fn setup_port(bind_addr: &str) -> String {
    bind_addr
        .rsplit(':')
        .next()
        .filter(|s| !s.is_empty())
        .unwrap_or("9091")
        .to_owned()
}

async fn nmcli(args: &[&str]) -> anyhow::Result<String> {
    let output = tokio::process::Command::new("nmcli")
        .args(args)
        .output()
        .await
        .map_err(|e| anyhow::anyhow!("spawn nmcli: {e}"))?;
    if !output.status.success() {
        anyhow::bail!(
            "nmcli {:?} failed: {}",
            args,
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn setup_port_parsed() {
        assert_eq!(setup_port("0.0.0.0:9091"), "9091");
        assert_eq!(setup_port("127.0.0.1:8080"), "8080");
    }
}
