//! Agent configuration: loaded from a TOML file on disk (default
//! `/etc/bebop/agent.toml`) with environment-variable overrides.

use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

pub const DEFAULT_CONFIG_PATH: &str = "/etc/bebop/agent.toml";
pub const CONFIG_PATH_ENV: &str = "BEBOP_AGENT_CONFIG";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    /// Human readable robot identifier (defaults to hostname).
    #[serde(default = "default_robot_name")]
    pub robot_name: String,

    /// Persistent state / config directory.
    #[serde(default = "default_state_dir")]
    pub state_dir: PathBuf,

    #[serde(default)]
    pub ble: BleConfig,

    #[serde(default)]
    pub app: AppConfig,

    #[serde(default)]
    pub ota: OtaConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BleConfig {
    /// Adapter name to use. `None` means use the default adapter.
    #[serde(default)]
    pub adapter: Option<String>,

    /// Advertised BLE local name (what users see in their phone's scanner).
    #[serde(default = "default_ble_local_name")]
    pub local_name: String,

    /// If true, require the mobile app to complete a pairing challenge
    /// (using a pre-shared pairing code) before any writes take effect.
    #[serde(default = "default_true")]
    pub require_pairing: bool,
}

impl Default for BleConfig {
    fn default() -> Self {
        Self {
            adapter: None,
            local_name: default_ble_local_name(),
            require_pairing: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    /// Name of the primary robot application container.
    #[serde(default = "default_app_name")]
    pub name: String,

    /// Image to pull and run for the robot application container.
    ///
    /// `None` means "no app configured": the container supervisor stays
    /// idle and makes no pull attempts. This is the right default for a
    /// freshly-flashed device that hasn't been pointed at a registry yet.
    #[serde(default)]
    pub image: Option<String>,

    /// Use nvidia container runtime (passes `--runtime=nvidia`).
    #[serde(default = "default_true")]
    pub use_nvidia_runtime: bool,

    /// Extra environment variables to inject into the robot app container.
    #[serde(default)]
    pub env: Vec<String>,

    /// Host paths to mount into the container (`/host:/container[:ro]`).
    #[serde(default)]
    pub volumes: Vec<String>,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            name: default_app_name(),
            image: None,
            use_nvidia_runtime: true,
            env: vec![],
            volumes: vec![],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OtaConfig {
    /// How often (seconds) to poll the update channel.
    #[serde(default = "default_ota_poll_secs")]
    pub poll_interval_secs: u64,

    /// URL returning a manifest describing the desired image for this channel.
    /// e.g. `https://updates.bebop.example.com/channels/stable.json`.
    #[serde(default)]
    pub manifest_url: Option<String>,

    /// Update channel name (purely informational; the URL is authoritative).
    #[serde(default = "default_channel")]
    pub channel: String,
}

impl Default for OtaConfig {
    fn default() -> Self {
        Self {
            poll_interval_secs: default_ota_poll_secs(),
            manifest_url: None,
            channel: default_channel(),
        }
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
            ble: BleConfig::default(),
            app: AppConfig::default(),
            ota: OtaConfig::default(),
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
/// `save` after a BLE-driven edit will replace it with concrete values.
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

fn default_ble_local_name() -> String {
    format!("Bebop-{}", short_id())
}

fn default_app_name() -> String {
    "bebop-app".into()
}

fn default_ota_poll_secs() -> u64 {
    300
}

fn default_channel() -> String {
    "stable".into()
}

fn default_true() -> bool {
    true
}

fn hostname_or(fallback: String) -> String {
    std::fs::read_to_string("/etc/hostname")
        .map(|s| s.trim().to_owned())
        .unwrap_or(fallback)
}

/// Short, stable-ish per-device id for advertising names.
/// Uses the machine-id (truncated) when available.
fn short_id() -> String {
    std::fs::read_to_string("/etc/machine-id")
        .ok()
        .map(|s| s.trim().chars().take(6).collect::<String>())
        .unwrap_or_else(|| "000000".into())
}
