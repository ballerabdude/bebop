//! OTA updater.
//!
//! Strategy: containerised OTA. The agent periodically fetches a small JSON
//! manifest from `ota.manifest_url` describing the image the robot should be
//! running. If the manifest's target image differs from what's currently
//! deployed, the updater:
//!
//!   1. Pulls the new image tag (this happens in the container manager).
//!   2. Swaps the app config to point at the new image.
//!   3. Triggers a container recreate (`containers::restart`).
//!
//! Because the Bebop agent itself runs as a systemd unit (outside Docker),
//! this flow only updates the robot *application*, not the agent. Agent
//! self-update is a separate (future) concern — typically handled by
//! `apt`/a separate signed debian repo, or Mender/SWUpdate for full system
//! image updates.

use anyhow::{Context, Result};
use serde::Deserialize;
use tracing::{error, info, warn};

use crate::error::AgentError;
use crate::state::{AppState, OtaLifecycle, OtaRuntimeStatus};

#[derive(Debug, Deserialize)]
struct Manifest {
    /// Fully-qualified image reference the robot should run.
    image: String,
    /// Optional digest to pin the exact image content.
    #[serde(default)]
    digest: Option<String>,
    /// Optional notes — shown in the mobile app.
    #[serde(default)]
    notes: Option<String>,
}

pub async fn run(state: AppState) -> Result<()> {
    loop {
        let cfg = state.config().await;
        let interval = std::time::Duration::from_secs(cfg.ota.poll_interval_secs.max(30));
        if let Err(e) = poll_once(&state).await {
            warn!(error = ?e, "ota poll failed");
        }
        tokio::time::sleep(interval).await;
    }
}

/// Manual trigger from the mobile app.
pub async fn trigger(state: &AppState, target_image: Option<String>) -> Result<(), AgentError> {
    let mut status = state.ota_status().await;
    status.state = OtaLifecycle::Checking;
    status.error = None;
    status.target_image = target_image.clone().unwrap_or_default();
    state.set_ota_status(status).await;

    let state = state.clone();
    tokio::spawn(async move {
        if let Err(e) = apply(&state, target_image).await {
            error!(error = ?e, "ota apply failed");
            let mut s = state.ota_status().await;
            s.state = OtaLifecycle::Failed;
            s.error = Some(e.to_string());
            state.set_ota_status(s).await;
        }
    });

    Ok(())
}

async fn poll_once(state: &AppState) -> Result<()> {
    let cfg = state.config().await;
    let Some(url) = cfg.ota.manifest_url.clone() else {
        return Ok(());
    };

    info!(%url, "checking ota manifest");
    let manifest: Manifest = reqwest::get(&url)
        .await
        .with_context(|| format!("fetch {url}"))?
        .error_for_status()?
        .json()
        .await
        .context("parse manifest")?;

    let current = state.app_status().await.image;
    if manifest.image == current {
        return Ok(());
    }
    info!(%current, target = %manifest.image, "update available, applying");
    apply(state, Some(manifest.image)).await
}

async fn apply(state: &AppState, target_image: Option<String>) -> Result<()> {
    let mut s = state.ota_status().await;
    s.state = OtaLifecycle::Downloading;
    state.set_ota_status(s.clone()).await;

    if let Some(img) = target_image {
        state
            .update_config(|c| {
                c.app.image = img.clone();
            })
            .await;
        s.target_image = img;
    }

    s.state = OtaLifecycle::Applying;
    state.set_ota_status(s.clone()).await;

    // Delegate the actual pull + container swap to the container manager.
    crate::containers::restart(state)
        .await
        .map_err(|e| anyhow::anyhow!("restart failed: {e}"))?;

    let current = state.app_status().await.image;
    s = OtaRuntimeStatus {
        state: OtaLifecycle::Success,
        current_image: current,
        target_image: s.target_image,
        progress_percent: 100,
        error: None,
    };
    state.set_ota_status(s).await;
    info!("ota applied successfully");
    Ok(())
}
