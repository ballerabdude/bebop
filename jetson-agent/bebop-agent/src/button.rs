//! Physical mode-toggle button.
//!
//! A momentary switch wired between a GPIO header pin (default: Orin Nano
//! pin 29 → `gpiochip0` line 105, which idles low) and 3.3 V. A **long press** (default 5 s,
//! fired on release) toggles the robot between the two network modes:
//! `client` ("Known Network") and `ap` ("Hosted Network").
//!
//! Implemented with the pure-Rust [`gpiocdev`] crate (GPIO uAPI v2), so it
//! needs no `libgpiod` build dependency and can set the line's internal
//! pull-up/pull-down bias.

use std::time::Duration;

use gpiocdev::line::{Bias, EdgeDetection, EdgeKind};
use gpiocdev::Request;
use tokio::sync::mpsc;
use tracing::{info, warn};

use crate::ap;
use crate::state::AppState;

/// Entry point. Never returns while enabled; parks if the button is
/// disabled or the line cannot be requested, so it can't take the agent down.
pub async fn run(state: AppState) -> anyhow::Result<()> {
    let cfg = state.config().await.network;
    if !cfg.button_enabled {
        info!("mode button disabled in config; idling");
        std::future::pending::<()>().await;
        return Ok(());
    }

    let (tx, mut rx) = mpsc::channel::<()>(4);

    let chip = cfg.button_chip.clone();
    let line = cfg.button_line;
    let active_low = cfg.button_active_low;
    let hold = Duration::from_secs(cfg.button_hold_secs.max(1));
    let bias = parse_bias(&cfg.button_bias);

    let spawned = std::thread::Builder::new()
        .name("bebop-button".into())
        .spawn(move || button_loop(&chip, line, active_low, hold, bias, tx));
    match spawned {
        Ok(_) => info!(
            chip = %cfg.button_chip,
            line = cfg.button_line,
            hold_secs = hold.as_secs(),
            "mode button armed (long-press to toggle Known/Hosted)"
        ),
        Err(e) => {
            warn!(error = %e, "failed to spawn button thread; button unavailable");
            std::future::pending::<()>().await;
            return Ok(());
        }
    }

    while rx.recv().await.is_some() {
        toggle_mode(&state).await;
    }

    // Thread exited (line error). Park rather than completing — a returning
    // subsystem would be treated as fatal by the supervisor.
    warn!("button thread exited; button unavailable until restart");
    std::future::pending::<()>().await;
    Ok(())
}

/// Blocking edge-event loop, runs on a dedicated thread.
fn button_loop(
    chip: &str,
    line: u32,
    active_low: bool,
    hold: Duration,
    bias: Option<Bias>,
    tx: mpsc::Sender<()>,
) {
    // gpiocdev wants a device path; accept either "gpiochip0" or the full
    // "/dev/gpiochip0".
    let chip_path = if chip.starts_with('/') {
        chip.to_owned()
    } else {
        format!("/dev/{chip}")
    };

    let request = Request::builder()
        .on_chip(&chip_path)
        .with_consumer("bebop-agent")
        .with_line(line)
        .with_edge_detection(EdgeDetection::BothEdges)
        .with_bias(bias)
        .with_debounce_period(Duration::from_millis(50))
        .request();
    let request = match request {
        Ok(r) => r,
        Err(e) => {
            warn!(error = %e, chip = %chip_path, line, "failed to request button GPIO line");
            return;
        }
    };

    // Diagnostic: idle level with the configured bias. A pull-up should
    // read Active (high) while the switch is open.
    match request.value(line) {
        Ok(v) => info!(?v, "button line idle level"),
        Err(e) => warn!(error = %e, "failed to read button idle level"),
    }

    let mut press_ts: Option<u64> = None;
    for event in request.edge_events() {
        let event = match event {
            Ok(e) => e,
            Err(e) => {
                warn!(error = %e, "button GPIO event error");
                return;
            }
        };
        let pressed = match event.kind {
            EdgeKind::Falling => active_low,
            EdgeKind::Rising => !active_low,
        };
        if pressed {
            press_ts = Some(event.timestamp_ns);
        } else if let Some(t0) = press_ts.take() {
            let held = Duration::from_nanos(event.timestamp_ns.saturating_sub(t0));
            if held >= hold {
                info!(held_ms = held.as_millis(), "button long-press detected");
                // Fire and forget; the async side applies the toggle.
                let _ = tx.blocking_send(());
            } else {
                info!(held_ms = held.as_millis(), "button short tap ignored");
            }
        }
    }
}

/// Flip the persisted network mode and reconcile the hotspot immediately.
async fn toggle_mode(state: &AppState) {
    let mut new_mode = String::new();
    state
        .update_config(|c| {
            new_mode = if c.network.hosts_ap() {
                "client".into()
            } else {
                "ap".into()
            };
            c.network.mode = new_mode.clone();
        })
        .await;

    if let Err(e) = state.persist_config().await {
        warn!(error = %e, "failed to persist toggled network mode");
    }
    info!(mode = %new_mode, "mode button toggled network mode");

    // Lower immediately when switching to client; the supervisor raises the
    // AP on its next tick when switching to ap.
    ap::reconfigure(state).await;
}

fn parse_bias(s: &str) -> Option<Bias> {
    match s.trim().to_ascii_lowercase().as_str() {
        "pull-up" | "pullup" | "up" => Some(Bias::PullUp),
        "pull-down" | "pulldown" | "down" => Some(Bias::PullDown),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bias_parsing() {
        assert_eq!(parse_bias("pull-up"), Some(Bias::PullUp));
        assert_eq!(parse_bias("PULL-DOWN"), Some(Bias::PullDown));
        assert_eq!(parse_bias("none"), None);
    }
}
