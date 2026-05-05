//! Power-board monitor: 1 Hz polling + RX caching + safety hooks.
//!
//! Owns one [`PowerBoardSnapshot`] (behind a `Mutex`) that holds the
//! latest battery voltage / motor voltage / board temperature / fault
//! word reported by the Robstride PowerBoard. The supervisor exposes
//! this snapshot to the WS server so every telemetry frame includes
//! a "fuel gauge" reading.
//!
//! Two pieces collaborate:
//!
//!  - **Poller** ([`spawn_power_monitor`]): a dedicated OS thread that
//!    sends a Type-02 status query every `poll_interval_ms`. We poll
//!    rather than rely on the board's auto-report mode so the host
//!    wall-clock on each sample is meaningful and a missing response
//!    surfaces as `feedback_stale` instead of silently going stale.
//!  - **RX dispatch** ([`PowerMonitor::on_frame`]): called by the
//!    per-bus RX thread for any frame that arrives on the power bus
//!    with a comm-type of 0x03 / 0x04 / 0x05. Updates the cached
//!    snapshot atomically.
//!
//! The poller deliberately doesn't talk to the supervisor's E-STOP
//! latch directly; if we ever want a hard E-STOP on under-voltage we
//! can wire it through here, but for now the operator UI just shows
//! the reading and lets a human decide.

use crate::config::PowerBoardConfig;
use crate::powerboard::{describe_faults, PowerBoard, PowerCurrents, PowerFrame, PowerStatus};
use crate::safety::bus_pool::BusPool;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};
use tracing::{debug, info, trace, warn};

/// Cached power-board state shared between the RX thread, the poller,
/// and the WS telemetry pump.
#[derive(Debug, Clone, Default)]
pub struct PowerBoardSnapshot {
    /// Last Type-03 status response we successfully decoded.
    pub status: Option<PowerStatus>,
    /// Last Type-04 per-branch current response, if anyone has queried.
    pub currents: Option<PowerCurrents>,
    /// Last successfully decoded firmware version string ("PBV1.00").
    pub version: Option<String>,
    /// Wall-clock instant when `status` was last updated. `None` means
    /// no Type-03 frame has arrived yet (board offline / wrong bus).
    pub last_status_rx: Option<Instant>,
    /// Wall-clock instant of the last *outgoing* poll. Useful for the
    /// "stale" check: if poll_at - last_status_rx > timeout we know
    /// the board missed a beat, even if we haven't polled again yet.
    pub last_poll_at: Option<Instant>,
}

impl PowerBoardSnapshot {
    /// Whether `status` is older than `staleness_ms` from `now`. Returns
    /// `false` if no status has ever been received (we don't want to
    /// bait the operator with "stale" before the first poll completes).
    pub fn is_stale(&self, now: Instant, staleness_ms: u64) -> bool {
        match self.last_status_rx {
            None => false,
            Some(t) => now.duration_since(t).as_millis() as u64 > staleness_ms,
        }
    }
}

/// Owner of the power-board cache + poller handle. Cheap to clone
/// (everything inside is `Arc`-wrapped).
#[derive(Clone)]
pub struct PowerMonitor {
    pub cfg: PowerBoardConfig,
    pub board: PowerBoard,
    pub state: Arc<Mutex<PowerBoardSnapshot>>,
}

impl PowerMonitor {
    pub fn new(cfg: PowerBoardConfig) -> Self {
        let board = PowerBoard::new(cfg.power_id);
        Self {
            cfg,
            board,
            state: Arc::new(Mutex::new(PowerBoardSnapshot::default())),
        }
    }

    /// Take a cheap copy of the latest snapshot. The clone is `O(1)` —
    /// the inner `PowerStatus` etc. are small Plain-Old-Data structs.
    pub fn snapshot(&self) -> PowerBoardSnapshot {
        self.state
            .lock()
            .map(|g| g.clone())
            .unwrap_or_else(|p| p.into_inner().clone())
    }

    /// Called by the per-bus RX thread for every parsed power frame.
    /// Updates the cached state under the mutex.
    pub fn on_frame(&self, frame: PowerFrame) {
        let now = Instant::now();
        let mut g = match self.state.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        match frame {
            PowerFrame::Status(s) => {
                // First-status-frame log at INFO so the operator can
                // confirm in the firmware terminal that the board is
                // alive and being decoded; subsequent frames stay at
                // TRACE to keep the log quiet during normal operation.
                if g.status.is_none() {
                    info!(
                        battery_v = s.battery_voltage_v,
                        motor_v = s.motor_voltage_v,
                        board_temp_c = s.board_temperature_c,
                        rails = format!(
                            "24V={} VMBUS={} 12V={} softstart={}",
                            on_off(s.rail_24v_on),
                            on_off(s.motor_rail_on),
                            on_off(s.rail_12v_on),
                            on_off(s.soft_start_on),
                        ),
                        "powerboard first status frame received",
                    );
                }
                trace!(
                    battery_v = s.battery_voltage_v,
                    motor_v = s.motor_voltage_v,
                    temp_c = s.board_temperature_c,
                    fault_bits = format!("0x{:06X}", s.fault_bits),
                    "powerboard status RX"
                );
                if s.fault_bits & 0x000F_FFFF != 0 {
                    // Don't latch E-STOP from here; just log so the
                    // operator notices on the firmware terminal.
                    warn!(
                        fault = %describe_faults(s.fault_bits),
                        battery_v = s.battery_voltage_v,
                        motor_v = s.motor_voltage_v,
                        "powerboard reports faults"
                    );
                }
                g.status = Some(s);
                g.last_status_rx = Some(now);
            }
            PowerFrame::Currents { currents, .. } => {
                g.currents = Some(currents);
            }
            PowerFrame::Version { version, .. } => {
                if g.version.as_deref() != Some(version.as_str()) {
                    info!(version = %version, "powerboard firmware version detected");
                }
                g.version = Some(version);
            }
        }
    }
}

fn on_off(b: bool) -> &'static str {
    if b {
        "on"
    } else {
        "off"
    }
}

/// Spawn the power-board poller thread. Returns the join handle so the
/// process can wait for it on shutdown. The thread also issues an
/// initial version query so the firmware version shows up in the first
/// few telemetry frames.
pub fn spawn_power_monitor(
    monitor: Arc<PowerMonitor>,
    bus_pool: Arc<BusPool>,
    shutdown: Arc<AtomicBool>,
) -> Option<JoinHandle<()>> {
    let iface = monitor.cfg.can_interface.clone();
    let can = match bus_pool.get(&iface) {
        Some(c) => c.clone(),
        None => {
            warn!(
                can_interface = %iface,
                "power monitor: no bus pool entry for {iface}; skipping poll thread"
            );
            return None;
        }
    };
    let interval = Duration::from_millis(monitor.cfg.poll_interval_ms);
    let board = monitor.board;
    let state = monitor.state.clone();
    let handle = std::thread::Builder::new()
        .name(format!("powerboard-{iface}"))
        .spawn(move || {
            info!(
                can_interface = %iface,
                power_id = board.power_id,
                interval_ms = interval.as_millis() as u64,
                "powerboard monitor thread started"
            );
            // Kick off with a version query so the operator UI gets a
            // firmware string in the first few frames. Failure here is
            // benign — the bus may be down at boot and recover later.
            if let Err(e) = board.query_version(&can) {
                debug!(
                    can_interface = %iface,
                    error = format!("{:#}", e),
                    "powerboard version query failed (ok at boot)",
                );
            }
            std::thread::sleep(Duration::from_millis(50));

            while !shutdown.load(Ordering::SeqCst) {
                let poll_at = Instant::now();
                if let Ok(mut g) = state.lock() {
                    g.last_poll_at = Some(poll_at);
                }
                if let Err(e) = board.query_status(&can) {
                    // CAN write failed. Most likely cause: the gs_usb
                    // controller went BUS-OFF because there's no peer
                    // ACKing on this bus, or the kernel TX queue is
                    // full. Logged at DEBUG with the full cause chain
                    // so the operator can `RUST_LOG=debug` and see the
                    // underlying errno (e.g. "No buffer space available
                    // (os error 105)") instead of just our context line.
                    debug!(
                        can_interface = %iface,
                        error = format!("{:#}", e),
                        "powerboard status TX failed",
                    );
                }
                // Stagger the per-branch current query a bit so the two
                // requests don't race the same response window.
                std::thread::sleep(Duration::from_millis(50));
                if let Err(e) = board.query_currents(&can) {
                    debug!(
                        can_interface = %iface,
                        error = format!("{:#}", e),
                        "powerboard currents TX failed",
                    );
                }
                // Sleep the rest of the interval. `saturating_sub`
                // means we never sleep negative if the two TX calls
                // ate the whole budget.
                let elapsed = poll_at.elapsed();
                let remainder = interval.saturating_sub(elapsed);
                if remainder > Duration::ZERO {
                    std::thread::sleep(remainder);
                }
            }
            info!(can_interface = %iface, "powerboard monitor thread exiting");
        })
        .expect("spawn powerboard monitor thread");
    Some(handle)
}
