// On non-Linux hosts the real BLE server is cfg'd out, so much of the
// scaffolding (dispatcher, framing, UUIDs, ...) appears dead to the
// compiler. Silence that noise while keeping it meaningful on Linux.
#![cfg_attr(not(target_os = "linux"), allow(dead_code))]

//! Bebop Agent — entrypoint.
//!
//! Orchestrates all long-running subsystems:
//!   * BLE GATT server (provisioning / control surface for the mobile app)
//!   * Wi-Fi provisioner (wraps NetworkManager)
//!   * Container manager (Docker / NVIDIA runtime)
//!   * OTA updater
//!
//! Each subsystem runs on its own tokio task and communicates with the others
//! through the shared [`AppState`] handle.

mod ble;
mod config;
mod containers;
mod controller;
mod error;
mod ota;
mod state;
mod wifi;
mod ws;

use anyhow::Context;
use tracing::{error, info};
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

use crate::state::AppState;

const AGENT_VERSION: &str = env!("CARGO_PKG_VERSION");

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    init_tracing();
    info!(version = AGENT_VERSION, "starting bebop-agent");

    let cfg = config::AgentConfig::load().context("failed to load agent configuration")?;
    info!(?cfg, "configuration loaded");

    let state = AppState::new(cfg.clone()).await?;

    // Spawn long-running subsystems. Each returns a JoinHandle so we can
    // supervise them and exit if any of them crashes fatally.
    let mut tasks = tokio::task::JoinSet::new();

    {
        let s = state.clone();
        tasks.spawn(async move {
            if let Err(e) = containers::run(s).await {
                error!(error = ?e, "container manager exited");
            }
        });
    }

    {
        let s = state.clone();
        tasks.spawn(async move {
            if let Err(e) = ota::run(s).await {
                error!(error = ?e, "ota updater exited");
            }
        });
    }

    {
        let s = state.clone();
        tasks.spawn(async move {
            if let Err(e) = ble::run(s).await {
                error!(error = ?e, "ble server exited");
            }
        });
    }

    {
        let s = state.clone();
        tasks.spawn(async move {
            if let Err(e) = controller::run(s).await {
                error!(error = ?e, "controller subsystem exited");
            }
        });
    }

    // Network control surface — WS mirror of the BLE GATT API. Lets the
    // operator app pair controllers / read status without going through
    // BLE (the IP-only path in `bebop-app`).
    {
        let s = state.clone();
        tasks.spawn(async move {
            if let Err(e) = ws::run(s).await {
                error!(error = ?e, "agent WS server exited");
            }
        });
    }

    // Graceful shutdown on SIGINT / SIGTERM.
    tokio::select! {
        _ = shutdown_signal() => {
            info!("shutdown signal received; exiting");
        }
        Some(res) = tasks.join_next() => {
            match res {
                Ok(()) => info!("a subsystem completed; exiting"),
                Err(e) => error!(error = ?e, "a subsystem panicked"),
            }
        }
    }

    tasks.shutdown().await;
    Ok(())
}

fn init_tracing() {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info,bebop_agent=debug"));

    tracing_subscriber::registry()
        .with(fmt::layer().with_target(true))
        .with(filter)
        .init();
}

async fn shutdown_signal() {
    use tokio::signal::unix::{signal, SignalKind};

    let mut sigterm = signal(SignalKind::terminate()).expect("install SIGTERM handler");
    let mut sigint = signal(SignalKind::interrupt()).expect("install SIGINT handler");

    tokio::select! {
        _ = sigterm.recv() => {}
        _ = sigint.recv() => {}
    }
}
