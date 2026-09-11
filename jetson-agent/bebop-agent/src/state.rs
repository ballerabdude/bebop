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
    wifi: RwLock<WifiRuntimeStatus>,
    ap: RwLock<ApRuntimeStatus>,
}

#[derive(Debug, Clone, Default)]
pub struct WifiRuntimeStatus {
    pub connected: bool,
    pub ssid: String,
    pub ip_address: String,
    pub signal_dbm: i32,
}

/// Live state of the setup SoftAP.
#[derive(Debug, Clone, Default)]
pub struct ApRuntimeStatus {
    pub active: bool,
    pub ssid: String,
    /// `host:port` the app should use to reach the setup server while the
    /// AP is up (e.g. `192.168.42.1:9091`).
    pub address: String,
    /// True while a Wi-Fi join triggered from the setup UI is in flight.
    /// Suppresses the AP supervisor so it doesn't race the connection.
    pub connecting: bool,
    pub last_error: Option<String>,
}

impl AppState {
    pub async fn new(config: AgentConfig) -> anyhow::Result<Self> {
        Ok(Self {
            inner: Arc::new(Inner {
                config: RwLock::new(config),
                wifi: RwLock::new(WifiRuntimeStatus::default()),
                ap: RwLock::new(ApRuntimeStatus::default()),
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

    pub async fn wifi_status(&self) -> WifiRuntimeStatus {
        self.inner.wifi.read().await.clone()
    }

    pub async fn set_wifi_status(&self, s: WifiRuntimeStatus) {
        *self.inner.wifi.write().await = s;
    }

    pub async fn ap_status(&self) -> ApRuntimeStatus {
        self.inner.ap.read().await.clone()
    }

    pub async fn update_ap_status<F>(&self, f: F)
    where
        F: FnOnce(&mut ApRuntimeStatus),
    {
        let mut g = self.inner.ap.write().await;
        f(&mut g);
    }
}
