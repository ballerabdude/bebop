//! Linux-only background task that owns the evdev → teleop → UDP
//! pipeline. Mirrors the shape of `containers::run`: idle when not
//! configured, otherwise reconcile in a loop with backoff on errors.

use std::time::{Duration, Instant};

use tokio::time::{sleep, timeout};
use tracing::{debug, info, warn};

use super::{
    bluez,
    evdev_input::{self, OpenedDevice},
    mapping::{self, GamepadButton, GamepadState},
    teleop::{self, TeleopOut, TeleopParams, TeleopState},
    udp::TeleopSink,
};
use crate::config::ControllerConfig;
use crate::state::AppState;

/// Wait this long between attach attempts when the controller isn't
/// reachable (powered off, out of range, BlueZ refusing to connect).
/// Kept fixed-rate rather than exponential because a wheeled robot
/// teleop UX needs the controller to come back fast when the user
/// turns it on, not after a 30 s exponential climb.
const RETRY_BACKOFF: Duration = Duration::from_secs(2);

/// How long we wait for an evdev event before we tick the watchdog. The
/// teleop module's own watchdog uses the user-configured `watchdog_ms`,
/// but we need a smaller poll-step so we can flush the periodic
/// "command at rest" packet at the configured send rate even when the
/// user isn't moving the sticks.
const EVENT_WAIT_TICK: Duration = Duration::from_millis(20);

pub async fn run(state: AppState) -> anyhow::Result<()> {
    info!("controller supervisor online");
    let mut idle_logged = false;
    // Tracks the last failure message we logged so we can suppress
    // repeated identical errors (e.g. "controller is off" logged every
    // 2 s for hours). Reset to None on success so the next failure is
    // always logged once.
    let mut last_logged_error: Option<String> = None;
    let mut suppressed_count: u32 = 0;

    loop {
        let cfg = state.config().await.controller.clone();

        if !cfg.enabled || cfg.paired_mac.is_empty() {
            if !idle_logged {
                if !cfg.enabled {
                    info!("[controller] disabled in agent.toml; supervisor idling");
                } else {
                    info!(
                        "[controller] no paired_mac configured; pair a gamepad \
                         from the bebop-app to enable teleop"
                    );
                }
                idle_logged = true;
            }
            // Reset error de-dupe so the next failure after a config
            // change (e.g. user re-pairs) gets logged immediately.
            last_logged_error = None;
            suppressed_count = 0;
            sleep(Duration::from_secs(5)).await;
            continue;
        }
        idle_logged = false;

        match attach_and_run(&state, &cfg).await {
            Ok(()) => {
                debug!("controller session ended cleanly; re-attaching");
                if suppressed_count > 0 {
                    info!(
                        suppressed = suppressed_count,
                        "controller recovered; suppressed repeated errors"
                    );
                }
                last_logged_error = None;
                suppressed_count = 0;
            }
            Err(e) => {
                let msg = e.to_string();
                if last_logged_error.as_deref() == Some(msg.as_str()) {
                    // Same failure as last time — keep counting silently
                    // and only re-log once a minute so the journal isn't
                    // flooded when the controller is just off.
                    suppressed_count += 1;
                    if suppressed_count % REPEAT_LOG_EVERY == 0 {
                        warn!(
                            error = %msg,
                            attempts = suppressed_count + 1,
                            mac = %cfg.paired_mac,
                            "controller still failing; will keep retrying"
                        );
                    }
                } else {
                    warn!(
                        error = %msg,
                        mac = %cfg.paired_mac,
                        "controller session ended with error"
                    );
                    last_logged_error = Some(msg);
                    suppressed_count = 0;
                }
            }
        }

        // Mark disconnected then back off before the next attach attempt.
        state
            .update_controller_status(|s| {
                s.connected = false;
                s.armed = false;
            })
            .await;

        sleep(RETRY_BACKOFF).await;
    }
}

/// Re-log a sustained failure once every N consecutive attempts so the
/// user sees periodic confirmation that the agent is still trying,
/// without filling the journal at the retry cadence. With a 2 s
/// backoff and ~7 s per failed attach attempt, 30 ≈ once every ~3-4
/// minutes.
const REPEAT_LOG_EVERY: u32 = 30;

/// One full attach cycle: BlueZ connect → evdev open → teleop loop
/// until the controller drops or an unrecoverable error fires.
async fn attach_and_run(state: &AppState, cfg: &ControllerConfig) -> anyhow::Result<()> {
    // Try to get BlueZ to (re-)connect first. If the controller is off
    // this returns an error; we bail and let the caller back off.
    if !bluez::is_connected(&cfg.paired_mac).await {
        debug!(mac = %cfg.paired_mac, "asking bluez to connect");
        bluez::try_connect(&cfg.paired_mac)
            .await
            .map_err(|e| anyhow::anyhow!(e))?;
    }

    // The kernel takes a beat to enumerate the input nodes after
    // BlueZ marks the device "Connected: yes". Try a couple of times
    // before giving up.
    let opened = wait_for_event_node(&cfg.paired_mac).await?;
    info!(
        path = ?opened.event_path,
        target = %cfg.target_addr,
        "controller attached, starting teleop loop"
    );

    state
        .update_controller_status(|s| {
            s.connected = true;
            s.armed = false;
            s.estop_latched = false;
        })
        .await;

    let sink = TeleopSink::connect(&cfg.target_addr).await?;
    teleop_loop(state, cfg, opened, sink).await
}

async fn wait_for_event_node(mac: &str) -> anyhow::Result<OpenedDevice> {
    for attempt in 0..10 {
        if let Some(opened) = evdev_input::open_for_mac(mac)? {
            return Ok(opened);
        }
        sleep(Duration::from_millis(200 * (attempt + 1))).await;
    }
    anyhow::bail!("no /dev/input/event* matched MAC {mac} after 10 retries");
}

async fn teleop_loop(
    state: &AppState,
    cfg: &ControllerConfig,
    mut opened: OpenedDevice,
    sink: TeleopSink,
) -> anyhow::Result<()> {
    let params = TeleopParams {
        deadzone: cfg.deadzone,
        max_lin_vel: cfg.max_lin_vel,
        max_ang_vel: cfg.max_ang_vel,
        deadman_threshold: cfg.deadman_threshold,
        watchdog: Duration::from_millis(cfg.watchdog_ms as u64),
    };
    let estop_btn = evdev_input::parse_button_name(&cfg.controller_estop_btn_name())
        .unwrap_or(GamepadButton::East);
    let arm_btn = evdev_input::parse_button_name(&cfg.controller_arm_btn_name())
        .unwrap_or(GamepadButton::South);

    let mut gp = GamepadState::default();
    let mut tstate = TeleopState::default();

    let send_period = Duration::from_millis((1000 / cfg.send_rate_hz.max(1) as u64).max(1));
    let mut send_ticker = tokio::time::interval(send_period);
    send_ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    loop {
        tokio::select! {
            // Block briefly waiting for the next evdev event so we
            // don't spin. `timeout` returning Err just means "no event
            // arrived yet"; we then fall through to the periodic send
            // tick below.
            evt_result = timeout(EVENT_WAIT_TICK, opened.stream.next_event()) => {
                match evt_result {
                    Ok(Ok(event)) => {
                        if let Some(raw) = evdev_input::translate(&event, &opened.cal) {
                            mapping::apply_event(&mut gp, raw);
                            tstate.note_event(Instant::now());
                            // Push the latest event timestamp into the
                            // shared status so the BLE notify loop can
                            // report it.
                            let now_ms = chrono_now_ms();
                            state
                                .update_controller_status(|s| {
                                    s.last_event_unix_ms = now_ms;
                                })
                                .await;
                        }
                    }
                    Ok(Err(e)) => {
                        warn!(error = %e, "evdev read error; assuming controller dropped");
                        let _ = sink.send_velocity(&teleop::idle_zero()).await;
                        return Ok(());
                    }
                    Err(_) => {
                        // Timeout: no event in EVENT_WAIT_TICK; that's
                        // fine, we'll still send a packet on the rate
                        // ticker below.
                    }
                }
            }
            _ = send_ticker.tick() => {
                let now = Instant::now();
                let out = teleop::tick(&mut tstate, &gp, &params, estop_btn, arm_btn, now);
                let armed = tstate.armed(&gp, &params);
                let latched = tstate.estop_latched();
                state
                    .update_controller_status(|s| {
                        s.armed = armed;
                        s.estop_latched = latched;
                    })
                    .await;

                match out {
                    TeleopOut::Velocity(cmd) => {
                        if let Err(e) = sink.send_velocity(&cmd).await {
                            warn!(error = %e, "udp send_velocity failed");
                        }
                    }
                    TeleopOut::Reset => {
                        if let Err(e) = sink.send_reset().await {
                            warn!(error = %e, "udp send_reset failed");
                        }
                        // Follow the reset with an immediate zero so
                        // the firmware doesn't keep coasting at the
                        // last commanded velocity until the next tick.
                        if let Err(e) = sink.send_velocity(&teleop::idle_zero()).await {
                            warn!(error = %e, "udp follow-up zero failed");
                        }
                    }
                }
            }
        }
    }
}

fn chrono_now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

/// Tiny helper trait so the supervisor can read the configured button
/// names without `cfg.controller.estop_button` literals scattered
/// around. Lives here to keep `ControllerConfig` itself generic.
trait ControllerCfgExt {
    fn controller_estop_btn_name(&self) -> String;
    fn controller_arm_btn_name(&self) -> String;
}

impl ControllerCfgExt for ControllerConfig {
    fn controller_estop_btn_name(&self) -> String {
        // Hard-coded for now — config-driven button mapping is a
        // future enhancement (see plan).
        "BTN_EAST".into()
    }
    fn controller_arm_btn_name(&self) -> String {
        "BTN_SOUTH".into()
    }
}
