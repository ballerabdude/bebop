//! Hosted Network ("SoftAP") supervisor.
//!
//! Network mode is a two-way switch with no automatic fallback:
//!   * `mode = "ap"`     — host the `Bebop-XXXX` hotspot continuously.
//!   * `mode = "client"` — join a saved network; never host.
//!
//! The physical button long-press toggles the mode; this module reconciles
//! the running hotspot with whichever mode is configured. The Wi-Fi radio is
//! single-ended, so hosting the AP necessarily drops any client link.
//!
//! Bring-up uses NetworkManager shared IPv4 mode, which also provides DHCP
//! and DNS to connected devices.

use std::time::Duration;

use anyhow::Context;
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
const TICK: Duration = Duration::from_secs(1);

/// Long-running supervisor.
pub async fn run(state: AppState) -> anyhow::Result<()> {
    info!("network supervisor online");

    // Startup reconciliation: a previous process (or a crash while hosting)
    // may have left the AP profile active while the configured mode is now
    // `client`. Tear it down so the radio can rejoin a known network.
    {
        let cfg = state.config().await;
        if !cfg.network.hosts_ap() && profile_active().await == Some(true) {
            info!("lowering leftover Hosted Network from a previous run");
            let _ = lower(&state).await;
        }
    }

    let mut last_mode_logged: Option<String> = None;

    loop {
        let cfg = state.config().await;
        let network = cfg.network.clone();
        let mut ap = state.ap_status().await;
        let desired = network.hosts_ap();
        let fingerprint = network.ap_fingerprint();

        // Detect an AP that was externally deactivated (NetworkManager
        // restart, an admin `nmcli con down`, a stale profile delete). Without
        // this the supervisor would keep believing it is hosting and never
        // re-raise. A failed query is treated as "still active" so a transient
        // nmcli error can't cause thrash.
        if ap.active && profile_active().await == Some(false) {
            warn!("setup AP is no longer active; re-raising");
            state
                .update_ap_status(|s| {
                    s.active = false;
                    s.fingerprint.clear();
                })
                .await;
            ap = state.ap_status().await;
        }

        if desired && (!ap.active || ap.fingerprint != fingerprint) {
            match raise(&state, &network).await {
                Ok(()) => {
                    let ssid = state.ap_status().await.ssid;
                    info!(ssid = %ssid, %fingerprint, "Hosted Network raised");
                }
                Err(e) => {
                    warn!(error = %e, "failed to raise Hosted Network; will retry");
                    state
                        .update_ap_status(|s| {
                            s.active = false;
                            s.last_error = Some(e.to_string());
                        })
                        .await;
                }
            }
        } else if !desired && ap.active {
            if let Err(e) = lower(&state).await {
                warn!(error = %e, "failed to lower Hosted Network");
            } else {
                info!("Hosted Network lowered");
            }
        }

        if last_mode_logged.as_deref() != Some(network.mode.as_str()) {
            info!(mode = %network.mode, "network mode active");
            last_mode_logged = Some(network.mode.clone());
        }

        tokio::time::sleep(TICK).await;
    }
}

/// Create + activate the AP profile from `cfg`. Idempotent.
async fn raise(state: &AppState, cfg: &NetworkConfig) -> anyhow::Result<()> {
    let iface = wifi::wifi_device()
        .await
        .map_err(|e| anyhow::anyhow!("no wifi interface: {e}"))?;
    let ssid = cfg.ap_ssid.trim();
    if ssid.is_empty() {
        anyhow::bail!("network.ap_ssid must not be empty");
    }
    if cfg.ap_password.len() < 8 {
        anyhow::bail!(
            "network.ap_password must be 8..=63 chars for WPA2 (got {})",
            cfg.ap_password.len()
        );
    }
    let band = cfg.nm_band();

    // Start from a clean radio state: drop any client association and the
    // previous AP profile before recreating, so NetworkManager doesn't leave
    // the interface half-switched (which can yield a beaconing AP that
    // clients cannot complete the handshake with).
    let _ = nmcli(&["device", "disconnect", &iface]).await;
    let _ = nmcli(&["con", "delete", AP_CON_NAME]).await;
    tokio::time::sleep(Duration::from_millis(500)).await;

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
        ssid,
        "802-11-wireless.mode",
        "ap",
        "802-11-wireless.band",
        band,
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
    nmcli(&["con", "up", AP_CON_NAME])
        .await
        .context("activating AP profile; is the radio free?")?;
    // Give the driver a beat to finish entering AP mode before we read the
    // assigned address or report success.
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Ask NM for the address it actually assigned; fall back to the
    // configured gateway if the query is unavailable.
    let gateway = query_gateway(&iface)
        .await
        .unwrap_or_else(|| AP_GATEWAY.into());
    let port = setup_port(&cfg.setup_bind_addr);
    let fingerprint = cfg.ap_fingerprint();
    state
        .update_ap_status(|s| {
            s.active = true;
            s.ssid = ssid.to_owned();
            s.address = format!("{gateway}:{port}");
            s.fingerprint = fingerprint;
            s.last_error = None;
        })
        .await;
    Ok(())
}

/// Force a lower + raise on the next supervisor pass, e.g. after the app
/// edits the hosted SSID/password. Safe to call when not hosting.
pub async fn reconfigure(state: &AppState) {
    let _ = lower(state).await;
}

/// Deactivate the AP profile (if present). Idempotent.
async fn lower(state: &AppState) -> anyhow::Result<()> {
    let _ = nmcli(&["con", "down", AP_CON_NAME]).await;
    state
        .update_ap_status(|s| {
            s.active = false;
            s.fingerprint.clear();
            s.last_error = None;
        })
        .await;
    Ok(())
}

/// True if the agent's AP profile is an active NetworkManager connection.
/// `None` when the query fails (treated as "unknown" by the caller).
async fn profile_active() -> Option<bool> {
    let out = nmcli(&["-t", "-f", "NAME", "con", "show", "--active"])
        .await
        .ok()?;
    Some(out.lines().any(|l| l.trim() == AP_CON_NAME))
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
