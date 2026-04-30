//! Dev-only stub used when compiling on a non-Linux host (e.g. macOS).
//!
//! BlueZ / `bluer` are Linux-only, so on other platforms we simply log that
//! BLE is unavailable and sleep forever. This lets `cargo check` / `cargo
//! test` run everywhere while the real implementation lives in `server.rs`.

use tracing::warn;

use crate::state::AppState;

pub async fn serve(_state: AppState) -> anyhow::Result<()> {
    warn!("BLE server disabled: this build target is not Linux/BlueZ");
    loop {
        tokio::time::sleep(std::time::Duration::from_secs(3600)).await;
    }
}
