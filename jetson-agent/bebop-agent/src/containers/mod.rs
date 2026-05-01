//! Robot application lifecycle, backed by the local Docker daemon.
//!
//! On a Jetson with `nvidia-container-toolkit` installed, we pass
//! `HostConfig.runtime = "nvidia"` so the container sees the GPU and CUDA
//! libraries. The configured image is pulled on boot, then a container is
//! created + started. Logs and restarts are supervised by this module.
//!
//! Public surface:
//!   * [`run`]   — long-running supervisor task (spawned from `main`).
//!   * [`start`] / [`stop`] / [`restart`] — used by the BLE dispatcher.

use std::collections::HashMap;

use anyhow::Context;
use bollard::container::{
    Config, CreateContainerOptions, RemoveContainerOptions, StartContainerOptions,
    StopContainerOptions,
};
use bollard::image::CreateImageOptions;
use bollard::models::HostConfig;
use bollard::Docker;
use futures::StreamExt;
use tracing::{error, info, warn};

use crate::error::AgentError;
use crate::state::{AppLifecycle, AppRuntimeStatus, AppState};

const NVIDIA_RUNTIME: &str = "nvidia";

pub async fn run(state: AppState) -> anyhow::Result<()> {
    let docker = connect().context("connect to docker")?;
    info!("container supervisor online");

    // Loop drives three states:
    //   * no image configured  -> idle silently after one info log.
    //   * image set, not yet bootstrapped -> ensure_running once.
    //   * bootstrapped, image unchanged -> poll health via reconcile.
    // OTA can flip image None<->Some at runtime; flags below let us
    // re-bootstrap without spamming the log.
    let mut ticker = tokio::time::interval(std::time::Duration::from_secs(10));
    let mut bootstrapped = false;
    let mut idle_logged = false;

    loop {
        let cfg = state.config().await;
        match cfg.app.image.as_ref() {
            None => {
                if !idle_logged {
                    info!(
                        "[app] image not configured; container supervisor idling. \
                         Set [app] image in /etc/bebop/agent.toml (then \
                         `systemctl restart bebop-agent`) to enable the robot app."
                    );
                    idle_logged = true;
                }
                bootstrapped = false;
            }
            Some(_) => {
                idle_logged = false;
                if !bootstrapped {
                    match ensure_running(&docker, &state).await {
                        Ok(()) => bootstrapped = true,
                        Err(e) => error!(error = ?e, "initial container start failed"),
                    }
                } else if let Err(e) = reconcile(&docker, &state).await {
                    warn!(error = ?e, "container reconcile error");
                }
            }
        }

        ticker.tick().await;
    }
}

fn connect() -> anyhow::Result<Docker> {
    Docker::connect_with_local_defaults().context("docker connect")
}

/// BLE dispatcher entrypoint: ensure the app is running.
pub async fn start(state: &AppState) -> Result<(), AgentError> {
    let docker = connect().map_err(|e| AgentError::Container(e.to_string()))?;
    ensure_running(&docker, state)
        .await
        .map_err(|e| AgentError::Container(e.to_string()))
}

/// BLE dispatcher entrypoint: stop the app.
pub async fn stop(state: &AppState) -> Result<(), AgentError> {
    let docker = connect().map_err(|e| AgentError::Container(e.to_string()))?;
    let cfg = state.config().await;
    stop_container(&docker, &cfg.app.name)
        .await
        .map_err(|e| AgentError::Container(e.to_string()))?;
    let mut status = state.app_status().await;
    status.state = AppLifecycle::Stopped;
    state.set_app_status(status).await;
    Ok(())
}

/// BLE dispatcher entrypoint: restart the app.
pub async fn restart(state: &AppState) -> Result<(), AgentError> {
    stop(state).await?;
    start(state).await
}

async fn ensure_running(docker: &Docker, state: &AppState) -> anyhow::Result<()> {
    let cfg = state.config().await;
    let name = cfg.app.name.clone();
    let image = cfg
        .app
        .image
        .clone()
        .ok_or_else(|| anyhow::anyhow!("no [app] image configured in agent.toml"))?;

    // Pull image (no-op if already present + tag unchanged).
    info!(%image, "pulling image");
    let mut pull = docker.create_image(
        Some(CreateImageOptions {
            from_image: image.clone(),
            ..Default::default()
        }),
        None,
        None,
    );
    while let Some(ev) = pull.next().await {
        match ev {
            Ok(info) => tracing::debug!(?info, "pull progress"),
            Err(e) => anyhow::bail!("pull failed: {e}"),
        }
    }

    // Remove any pre-existing container with this name (simplest way to
    // honour config changes like env/volumes/image).
    let _ = docker
        .remove_container(
            &name,
            Some(RemoveContainerOptions {
                force: true,
                ..Default::default()
            }),
        )
        .await;

    let host = HostConfig {
        runtime: if cfg.app.use_nvidia_runtime {
            Some(NVIDIA_RUNTIME.to_string())
        } else {
            None
        },
        network_mode: Some("host".to_string()),
        restart_policy: Some(bollard::models::RestartPolicy {
            name: Some(bollard::models::RestartPolicyNameEnum::UNLESS_STOPPED),
            ..Default::default()
        }),
        binds: if cfg.app.volumes.is_empty() {
            None
        } else {
            Some(cfg.app.volumes.clone())
        },
        ..Default::default()
    };

    let container_cfg = Config {
        image: Some(image.clone()),
        env: if cfg.app.env.is_empty() {
            None
        } else {
            Some(cfg.app.env.clone())
        },
        host_config: Some(host),
        labels: Some(HashMap::from([
            ("bebop.managed".to_string(), "true".to_string()),
            ("bebop.app".to_string(), name.clone()),
        ])),
        ..Default::default()
    };

    info!(%image, %name, "creating container");
    let created = docker
        .create_container(
            Some(CreateContainerOptions {
                name: name.clone(),
                ..Default::default()
            }),
            container_cfg,
        )
        .await?;

    docker
        .start_container(&name, None::<StartContainerOptions<String>>)
        .await?;

    let mut status = AppRuntimeStatus {
        image,
        container_id: created.id,
        state: AppLifecycle::Starting,
        started_at_unix: now_unix(),
        ..Default::default()
    };
    state.set_app_status(status.clone()).await;

    // Update to Running after a short settle.
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;
    status.state = AppLifecycle::Running;
    state.set_app_status(status).await;

    Ok(())
}

async fn reconcile(docker: &Docker, state: &AppState) -> anyhow::Result<()> {
    let cfg = state.config().await;
    let name = &cfg.app.name;
    let inspection = match docker.inspect_container(name, None).await {
        Ok(i) => i,
        Err(_) => {
            // Container vanished; recreate.
            return ensure_running(docker, state).await;
        }
    };
    let running = inspection
        .state
        .as_ref()
        .and_then(|s| s.running)
        .unwrap_or(false);

    let mut status = state.app_status().await;
    status.state = if running {
        AppLifecycle::Running
    } else {
        AppLifecycle::Crashed
    };
    status.container_id = inspection.id.unwrap_or_default();
    state.set_app_status(status).await;
    Ok(())
}

async fn stop_container(docker: &Docker, name: &str) -> anyhow::Result<()> {
    docker
        .stop_container(name, Some(StopContainerOptions { t: 10 }))
        .await?;
    Ok(())
}

fn now_unix() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}
