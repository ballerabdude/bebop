//! Bebop agent — entrypoint.
//!
//! Provisioning-only daemon:
//!   * Wi-Fi status poller (wraps NetworkManager)
//!   * Hosted Network supervisor (`ap` / `client` modes)
//!   * GPIO mode button (long-press toggles Known/Hosted)
//!   * Setup server (protobuf-over-WebSocket + a status page), reachable on
//!     the LAN or directly over the robot's setup hotspot.
//!
//! Each subsystem runs on its own tokio task and communicates through the
//! shared [`AppState`].

mod ap;
mod button;
mod config;
mod dispatcher;
mod error;
mod server;
mod state;
mod wifi;

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

    let state = AppState::new(cfg).await?;

    let mut tasks = tokio::task::JoinSet::new();

    {
        let s = state.clone();
        tasks.spawn(async move {
            if let Err(e) = wifi::run(s).await {
                error!(error = ?e, "wifi poller exited");
            }
        });
    }

    {
        let s = state.clone();
        tasks.spawn(async move {
            if let Err(e) = ap::run(s).await {
                error!(error = ?e, "network supervisor exited");
            }
        });
    }

    {
        let s = state.clone();
        tasks.spawn(async move {
            if let Err(e) = button::run(s).await {
                error!(error = ?e, "mode button exited");
            }
        });
    }

    {
        let s = state.clone();
        tasks.spawn(async move {
            if let Err(e) = server::run(s).await {
                error!(error = ?e, "setup server exited");
            }
        });
    }

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
