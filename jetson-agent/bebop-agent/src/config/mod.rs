//! Agent configuration: loaded from a TOML file on disk (default
//! `/etc/bebop/agent.toml`) with environment-variable overrides.

use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

pub const DEFAULT_CONFIG_PATH: &str = "/etc/bebop/agent.toml";
pub const CONFIG_PATH_ENV: &str = "BEBOP_AGENT_CONFIG";

/// Default WPA2 passphrase for the Hosted Network hotspot. 8..=63 bytes.
/// Deliberately simple for now; replace with a per-device derived code
/// before shipping to customers.
pub const DEFAULT_AP_PASSWORD: &str = "bebopbebop";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    /// Human readable robot identifier (defaults to hostname).
    #[serde(default = "default_robot_name")]
    pub robot_name: String,

    /// Persistent state / config directory.
    #[serde(default = "default_state_dir")]
    pub state_dir: PathBuf,

    #[serde(default)]
    pub network: NetworkConfig,
}

/// Wi-Fi behaviour.
///
/// `mode` is a two-way switch with **no automatic fallback**:
///   * `client` — join a saved network ("Known Network").
///   * `ap` — host the setup hotspot ("Hosted Network"); stays up
///     indefinitely and is only changed by the physical button.
///
/// The GPIO button long-press toggles `mode`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    /// `"ap"` (default) or `"client"`.
    #[serde(default = "default_network_mode")]
    pub mode: String,

    /// SSID broadcast by the setup hotspot.
    #[serde(default = "default_ap_ssid")]
    pub ap_ssid: String,

    /// WPA2 passphrase for the hotspot (8..=63 chars).
    #[serde(default = "default_ap_password")]
    pub ap_password: String,

    /// Hotspot band: `"2.4"` (default) or `"5"`.
    #[serde(default = "default_ap_band")]
    pub ap_band: String,

    /// Bind address for the setup server. `0.0.0.0` listens on every
    /// interface (LAN + Hosted Network); `127.0.0.1` restricts to the robot.
    #[serde(default = "default_setup_bind_addr")]
    pub setup_bind_addr: String,

    /// Enable the physical mode-toggle button.
    #[serde(default = "default_true")]
    pub button_enabled: bool,

    /// GPIO chip the button is wired to.
    #[serde(default = "default_button_chip")]
    pub button_chip: String,

    /// GPIO line offset on `button_chip`. Default 105 = header **pin 29**.
    #[serde(default = "default_button_line")]
    pub button_line: u32,

    /// True when the button shorts the line to GND when pressed.
    #[serde(default = "default_true")]
    pub button_active_low: bool,

    /// Internal line bias while idle: `"pull-up"`, `"pull-down"`, `"none"`.
    #[serde(default = "default_button_bias")]
    pub button_bias: String,

    /// Hold time (seconds) required to toggle the mode on release.
    #[serde(default = "default_button_hold_secs")]
    pub button_hold_secs: u64,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            mode: default_network_mode(),
            ap_ssid: default_ap_ssid(),
            ap_password: default_ap_password(),
            ap_band: default_ap_band(),
            setup_bind_addr: default_setup_bind_addr(),
            button_enabled: true,
            button_chip: default_button_chip(),
            button_line: default_button_line(),
            button_active_low: true,
            button_bias: default_button_bias(),
            button_hold_secs: default_button_hold_secs(),
        }
    }
}

impl NetworkConfig {
    /// True when the robot should host the setup hotspot.
    pub fn hosts_ap(&self) -> bool {
        self.mode.eq_ignore_ascii_case("ap")
    }

    /// NetworkManager `802-11-wireless.band` value.
    pub fn nm_band(&self) -> &'static str {
        if self.ap_band.trim() == "5" {
            "a"
        } else {
            "bg"
        }
    }

    /// Fingerprint of the AP settings an active profile was built from.
    /// Used to detect edits that require re-raising the hotspot.
    pub fn ap_fingerprint(&self) -> String {
        format!(
            "{}|{}|{}",
            self.ap_ssid,
            self.ap_band,
            short_hash(&self.ap_password)
        )
    }
}

impl AgentConfig {
    pub fn load() -> Result<Self> {
        let path = config_path();

        if path.exists() {
            let raw = std::fs::read_to_string(&path)
                .with_context(|| format!("reading config {}", path.display()))?;
            let cfg: AgentConfig = toml::from_str(&raw)
                .with_context(|| format!("parsing config {}", path.display()))?;
            Ok(cfg)
        } else {
            // First boot / dev: fall back to defaults and keep going.
            tracing::warn!(
                path = %path.display(),
                "agent config not found; using defaults"
            );
            Ok(Self::default_instance())
        }
    }

    fn default_instance() -> Self {
        Self {
            robot_name: default_robot_name(),
            state_dir: default_state_dir(),
            network: NetworkConfig::default(),
        }
    }
}

/// Resolve the on-disk config path the same way [`AgentConfig::load`] does.
/// Honours `BEBOP_AGENT_CONFIG`, falling back to [`DEFAULT_CONFIG_PATH`].
pub fn config_path() -> PathBuf {
    std::env::var(CONFIG_PATH_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(DEFAULT_CONFIG_PATH))
}

/// Atomically persist `cfg` to `path`. Writes to a sibling `.tmp` file,
/// fsyncs, and renames into place so a crash mid-write can't leave a
/// half-written config behind.
///
/// Note: this serialises via `toml::to_string_pretty`, which loses any
/// comments that were present in the source file. The shipped template at
/// `deploy/examples/agent.toml` is fully commented; the first call to
/// `save` after an app- or button-driven edit will replace it with concrete
/// values.
pub fn save(cfg: &AgentConfig, path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating {}", parent.display()))?;
        }
    }
    let serialized = toml::to_string_pretty(cfg).context("serialising AgentConfig to TOML")?;

    let tmp = path.with_extension("toml.tmp");
    {
        let mut f =
            std::fs::File::create(&tmp).with_context(|| format!("creating {}", tmp.display()))?;
        f.write_all(serialized.as_bytes())
            .with_context(|| format!("writing {}", tmp.display()))?;
        f.sync_all()
            .with_context(|| format!("fsync {}", tmp.display()))?;
    }
    std::fs::rename(&tmp, path)
        .with_context(|| format!("renaming {} -> {}", tmp.display(), path.display()))?;
    Ok(())
}

fn default_robot_name() -> String {
    hostname_or("bebop".into())
}

fn default_state_dir() -> PathBuf {
    PathBuf::from("/var/lib/bebop")
}

fn default_network_mode() -> String {
    // A brand-new robot hosts its hotspot so it is reachable immediately.
    "ap".into()
}

fn default_ap_ssid() -> String {
    format!("Bebop-{}", short_id())
}

fn default_ap_password() -> String {
    DEFAULT_AP_PASSWORD.into()
}

fn default_ap_band() -> String {
    "2.4".into()
}

fn default_setup_bind_addr() -> String {
    "0.0.0.0:9091".into()
}

fn default_button_chip() -> String {
    "gpiochip0".into()
}

fn default_button_line() -> u32 {
    105 // Orin Nano 40-pin header pin 29 (PQ.05)
}

fn default_button_bias() -> String {
    "pull-up".into()
}

fn default_button_hold_secs() -> u64 {
    5
}

fn default_true() -> bool {
    true
}

fn hostname_or(fallback: String) -> String {
    std::fs::read_to_string("/etc/hostname")
        .map(|s| s.trim().to_owned())
        .unwrap_or(fallback)
}

/// Short, stable-ish per-device id used to make AP SSIDs unique.
/// Uses the machine-id (truncated) when available.
fn short_id() -> String {
    std::fs::read_to_string("/etc/machine-id")
        .ok()
        .map(|s| s.trim().chars().take(6).collect::<String>())
        .unwrap_or_else(|| "000000".into())
}

/// Tiny FNV-1a hash so the AP fingerprint never contains the raw password.
fn short_hash(s: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_valid() {
        let cfg = NetworkConfig::default();
        assert_eq!(cfg.mode, "ap");
        assert!(cfg.hosts_ap());
        assert!(cfg.ap_password.len() >= 8);
        assert_eq!(cfg.nm_band(), "bg");
        assert_eq!(cfg.button_line, 105);
    }

    #[test]
    fn parses_minimal_config() {
        let cfg: AgentConfig = toml::from_str("robot_name = \"lab\"").unwrap();
        assert_eq!(cfg.robot_name, "lab");
        assert_eq!(cfg.network.mode, "ap");
    }

    #[test]
    fn band_mapping() {
        let cfg = NetworkConfig {
            ap_band: "5".into(),
            ..Default::default()
        };
        assert_eq!(cfg.nm_band(), "a");
        let cfg = NetworkConfig {
            ap_band: "2.4".into(),
            ..Default::default()
        };
        assert_eq!(cfg.nm_band(), "bg");
    }

    #[test]
    fn fingerprint_changes_with_password() {
        let cfg = NetworkConfig::default();
        let a = cfg.ap_fingerprint();
        let cfg = NetworkConfig {
            ap_password: "differentpw".into(),
            ..Default::default()
        };
        assert_ne!(a, cfg.ap_fingerprint());
    }
}
