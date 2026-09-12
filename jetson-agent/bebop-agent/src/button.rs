//! Physical mode-toggle button.
//!
//! A momentary switch wired between a GPIO header pin (default: Orin Nano
//! pin 32 → `gpiochip0` line 41, which has an internal pull-down) and
//! 3.3 V (no external resistor needed). Pressing it
//! toggles the robot between `client` ("Known Network") and `ap` ("Hosted
//! Network"). By default (`button_hold_secs = 0`) a debounced press toggles
//! immediately; set `button_hold_secs > 0` to require holding the line
//! active for that long instead.
//!
//! The line level is **polled**, not edge-timed, so it works with momentary
//! push-buttons and latching switches alike and does not depend on a release
//! edge arriving.
//!
//! Implemented with the pure-Rust [`gpiocdev`] crate (GPIO uAPI v2), so it
//! needs no `libgpiod` build dependency and can request a line bias.

use std::time::{Duration, Instant};

use gpiocdev::line::{Bias, Direction, Value};
use gpiocdev::Request;
use tokio::sync::mpsc;
use tracing::{info, warn};

use crate::ap;
use crate::state::AppState;

/// How often the button thread samples the line.
const POLL: Duration = Duration::from_millis(100);

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
    let hold = Duration::from_secs(cfg.button_hold_secs);
    let immediate = hold.is_zero();
    let bias = parse_bias(&cfg.button_bias);

    let spawned = std::thread::Builder::new()
        .name("bebop-button".into())
        .spawn(move || button_loop(&chip, line, active_low, hold, bias, tx));
    match spawned {
        Ok(_) => info!(
            chip = %cfg.button_chip,
            line = cfg.button_line,
            hold_secs = hold.as_secs(),
            trigger = if immediate { "press" } else { "hold" },
            "mode button armed (toggles Known/Hosted)"
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

/// Blocking level-polling loop, runs on a dedicated thread.
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
        .with_direction(Direction::Input)
        .with_bias(bias)
        .request();
    let request = match request {
        Ok(r) => r,
        Err(e) => {
            warn!(error = %e, chip = %chip_path, line, "failed to request button GPIO line");
            return;
        }
    };

    let read_pressed = |r: &Request| -> bool {
        match r.value(line) {
            Ok(Value::Active) => !active_low,
            Ok(Value::Inactive) => active_low,
            Err(_) => false,
        }
    };

    let idle = if read_pressed(&request) {
        "active"
    } else {
        "inactive"
    };
    info!(idle, "button line idle level");

    let immediate = hold.is_zero();

    // Let the line settle after the request (bias/direction take effect
    // asynchronously) before sampling the initial level, so a startup
    // transient isn't mistaken for a press.
    std::thread::sleep(Duration::from_millis(300));
    let mut prev_pressed = read_pressed(&request);
    if prev_pressed {
        info!("button line active at startup; waiting for a fresh press");
    }

    let mut pressed_since: Option<Instant> = None;
    let mut fired = false;
    // Debounce: a new level must persist for two polls before it is accepted.
    let mut stable = prev_pressed;
    let mut stable_count = 0u32;

    loop {
        std::thread::sleep(POLL);
        let raw = read_pressed(&request);
        if raw == stable {
            stable_count = stable_count.saturating_add(1);
        } else {
            stable = raw;
            stable_count = 1;
        }

        if stable != prev_pressed && stable_count >= 2 {
            if stable {
                if immediate {
                    info!("button pressed; toggling mode");
                    fired = true;
                    let _ = tx.blocking_send(());
                } else {
                    info!("button pressed; hold to toggle");
                    pressed_since = Some(Instant::now());
                    fired = false;
                }
            } else {
                info!("button released");
                pressed_since = None;
                fired = false;
            }
            prev_pressed = stable;
        }

        if !immediate && prev_pressed && !fired {
            if let Some(t0) = pressed_since {
                if t0.elapsed() >= hold {
                    info!(
                        held_ms = t0.elapsed().as_millis(),
                        "button long-press detected"
                    );
                    fired = true;
                    let _ = tx.blocking_send(());
                }
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
