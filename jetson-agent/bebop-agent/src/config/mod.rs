//! Agent configuration: loaded from a TOML file on disk (default
//! `/etc/bebop/agent.toml`) with environment-variable overrides.

use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

pub const DEFAULT_CONFIG_PATH: &str = "/etc/bebop/agent.toml";
pub const CONFIG_PATH_ENV: &str = "BEBOP_AGENT_CONFIG";

/// Default WPA2 passphrase for the setup SoftAP. Must be 8..=63 bytes.
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

/// Wi-Fi / SoftAP provisioning behaviour.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    /// `auto` | `client` | `ap`. See `proto/bebop.proto` for semantics.
    #[serde(default = "default_network_mode")]
    pub mode: String,

    /// SSID prefix for the setup SoftAP. A short device id is appended.
    #[serde(default = "default_ap_ssid_prefix")]
    pub ap_ssid_prefix: String,

    /// WPA2 passphrase for the SoftAP (8..=63 chars).
    #[serde(default = "default_ap_password")]
    pub ap_password: String,

    /// In `auto` mode, how long to wait for a known network before
    /// raising the SoftAP.
    #[serde(default = "default_ap_auto_after_secs")]
    pub ap_auto_after_secs: u64,

    /// Bind address for the setup server. `0.0.0.0` listens on every
    /// interface (LAN + SoftAP); `127.0.0.1` restricts to the robot.
    #[serde(default = "default_setup_bind_addr")]
    pub setup_bind_addr: String,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            mode: default_network_mode(),
            ap_ssid_prefix: default_ap_ssid_prefix(),
            ap_password: default_ap_password(),
            ap_auto_after_secs: default_ap_auto_after_secs(),
            setup_bind_addr: default_setup_bind_addr(),
        }
    }
}

impl NetworkConfig {
    /// Current AP SSID: prefix + short device id.
    pub fn ap_ssid(&self) -> String {
        format!("{}-{}", self.ap_ssid_prefix, short_id())
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
/// `save` after an app-driven edit will replace it with concrete values.
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
    "auto".into()
}

fn default_ap_ssid_prefix() -> String {
    "Bebop".into()
}

fn default_ap_password() -> String {
    DEFAULT_AP_PASSWORD.into()
}

fn default_ap_auto_after_secs() -> u64 {
    25
}

fn default_setup_bind_addr() -> String {
    "0.0.0.0:9091".into()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_valid() {
        let cfg = NetworkConfig::default();
        assert_eq!(cfg.mode, "auto");
        assert!(cfg.ap_password.len() >= 8);
        assert!(!cfg.ap_ssid().is_empty());
    }

    #[test]
    fn parses_minimal_config() {
        let cfg: AgentConfig = toml::from_str("robot_name = \"lab\"").unwrap();
        assert_eq!(cfg.robot_name, "lab");
        assert_eq!(cfg.network.mode, "auto");
    }
}
