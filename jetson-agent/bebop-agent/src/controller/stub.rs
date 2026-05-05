//! Dev-only stub used when compiling on a non-Linux host.
//!
//! `evdev` and BlueZ are Linux-only, so on macOS/Windows the supervisor
//! task simply logs once and parks forever. Mirrors the pattern used by
//! `ble::server_stub`.

use tracing::warn;

use crate::state::AppState;

pub async fn run(_state: AppState) -> anyhow::Result<()> {
    warn!(
        "controller subsystem disabled: this build target is not Linux/BlueZ. \
         Pair-from-app and teleop will not work until you run on a Jetson."
    );
    loop {
        tokio::time::sleep(std::time::Duration::from_secs(3600)).await;
    }
}
