//! Shared state handed to every subsystem.
//!
//! Subsystems hold a cheap-to-clone `AppState` (an `Arc` wrapper) and use
//! async locks inside it for interior mutability.

use std::sync::Arc;

use tokio::sync::RwLock;

use crate::config::AgentConfig;

#[derive(Clone)]
pub struct AppState {
    inner: Arc<Inner>,
}

struct Inner {
    config: RwLock<AgentConfig>,
    app: RwLock<AppRuntimeStatus>,
    ota: RwLock<OtaRuntimeStatus>,
    wifi: RwLock<WifiRuntimeStatus>,
    controller: RwLock<ControllerRuntimeStatus>,
}

#[derive(Debug, Clone, Default)]
pub struct AppRuntimeStatus {
    pub image: String,
    pub image_digest: String,
    pub container_id: String,
    pub state: AppLifecycle,
    pub started_at_unix: i64,
    pub restart_count: i32,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum AppLifecycle {
    #[default]
    Stopped,
    Starting,
    Running,
    Crashed,
    // Scaffolding: set by the OTA updater while a container swap is in
    // flight. Wiring lands when `ota::apply` is taught to flip app
    // lifecycle (currently it only mutates `OtaLifecycle`).
    #[allow(dead_code)]
    Updating,
}

#[derive(Debug, Clone, Default)]
pub struct OtaRuntimeStatus {
    pub state: OtaLifecycle,
    pub current_image: String,
    pub target_image: String,
    pub progress_percent: u32,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum OtaLifecycle {
    #[default]
    Idle,
    Checking,
    Downloading,
    Applying,
    Success,
    Failed,
}

#[derive(Debug, Clone, Default)]
pub struct WifiRuntimeStatus {
    pub connected: bool,
    pub ssid: String,
    pub ip_address: String,
    pub signal_dbm: i32,
}

/// Live state of the Bluetooth-controller subsystem. Mirrors the
/// `bebop.v1.ControllerStatus` proto so the dispatcher can convert
/// without hand-writing field maps in two places.
#[derive(Debug, Clone, Default)]
pub struct ControllerRuntimeStatus {
    pub paired_mac: String,
    pub device_name: String,
    pub connected: bool,
    pub armed: bool,
    pub estop_latched: bool,
    pub last_event_unix_ms: i64,
}

impl AppState {
    pub async fn new(config: AgentConfig) -> anyhow::Result<Self> {
        Ok(Self {
            inner: Arc::new(Inner {
                config: RwLock::new(config),
                app: RwLock::new(AppRuntimeStatus::default()),
                ota: RwLock::new(OtaRuntimeStatus::default()),
                wifi: RwLock::new(WifiRuntimeStatus::default()),
                controller: RwLock::new(ControllerRuntimeStatus::default()),
            }),
        })
    }

    pub async fn config(&self) -> AgentConfig {
        self.inner.config.read().await.clone()
    }

    pub async fn update_config<F>(&self, f: F)
    where
        F: FnOnce(&mut AgentConfig),
    {
        let mut g = self.inner.config.write().await;
        f(&mut g);
    }

    pub async fn app_status(&self) -> AppRuntimeStatus {
        self.inner.app.read().await.clone()
    }

    pub async fn set_app_status(&self, s: AppRuntimeStatus) {
        *self.inner.app.write().await = s;
    }

    pub async fn ota_status(&self) -> OtaRuntimeStatus {
        self.inner.ota.read().await.clone()
    }

    pub async fn set_ota_status(&self, s: OtaRuntimeStatus) {
        *self.inner.ota.write().await = s;
    }

    pub async fn wifi_status(&self) -> WifiRuntimeStatus {
        self.inner.wifi.read().await.clone()
    }

    pub async fn set_wifi_status(&self, s: WifiRuntimeStatus) {
        *self.inner.wifi.write().await = s;
    }

    pub async fn controller_status(&self) -> ControllerRuntimeStatus {
        self.inner.controller.read().await.clone()
    }

    #[allow(dead_code)] // parallel API to set_wifi_status; reserved for future use
    pub async fn set_controller_status(&self, s: ControllerRuntimeStatus) {
        *self.inner.controller.write().await = s;
    }

    /// Apply `f` to the current controller status in-place. Used by the
    /// teleop loop where we only flip a couple of fields and want to
    /// avoid cloning + re-storing the whole struct.
    pub async fn update_controller_status<F>(&self, f: F)
    where
        F: FnOnce(&mut ControllerRuntimeStatus),
    {
        let mut g = self.inner.controller.write().await;
        f(&mut g);
    }
}
