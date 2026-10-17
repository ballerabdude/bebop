//! Multi-bus CAN management with pre-flight health check.
//!
//! Wraps one [`CanInterface`] per distinct CAN channel referenced by the
//! config. The pre-flight reads each bus's controller state and logs a
//! prominent warning for any in `ERROR-PASSIVE` / `BUS-OFF` (commonly: the
//! peer is unpowered, unplugged, or there's no peer wired up yet for that
//! bus). The startup proceeds either way so that a partially-wired robot
//! can still bring up the half it has — armoring against silent runaway
//! is handled by the supervisor refusing to arm motors on unhealthy buses
//! (see [`BusPool::is_healthy`]).

use crate::can_interface::CanInterface;
use anyhow::Result;
use std::collections::HashMap;
use std::process::Command;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

/// How often `refresh_if_due` is allowed to probe (and possibly re-open) the
/// buses. The callers are per-telemetry-tick, i.e. tens of Hz, so the throttle
/// is what keeps this to a couple of `ip` calls a minute.
const REFRESH_INTERVAL: Duration = Duration::from_secs(5);

/// Read the current CAN controller state for `interface` by parsing
/// `ip -details link show <iface>`.
///
/// Returns one of `"ERROR-ACTIVE"`, `"ERROR-PASSIVE"`, `"BUS-OFF"`, or `None`
/// if the state line couldn't be located (driver or netdev variant). Doing
/// this via netlink directly would avoid a fork+exec, but `ip` is universally
/// available on Jetson L4T images and the call only happens at bring-up.
pub fn read_can_state(interface: &str) -> Option<String> {
    let out = Command::new("ip")
        .args(["-details", "link", "show", interface])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&out.stdout);
    for line in stdout.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("can state ") {
            return rest.split_whitespace().next().map(|s| s.to_string());
        }
    }
    None
}

/// Healthy = `ERROR-ACTIVE`. Treat anything else as suspect.
pub fn is_state_healthy(state: Option<&str>) -> bool {
    matches!(state, Some("ERROR-ACTIVE"))
}

pub struct BusPool {
    /// Re-openable: a USB-CAN adapter that comes back with a new ifindex
    /// leaves the old socket dead, so entries get replaced by
    /// [`BusPool::refresh_if_due`].
    by_iface: RwLock<HashMap<String, Arc<CanInterface>>>,
    last_refresh: Mutex<Instant>,
}

impl BusPool {
    /// Open every interface in `interfaces`.
    ///
    /// For each bus we log the controller state at INFO if healthy, or a
    /// loud WARN with remediation steps if not. The bus is still opened
    /// either way so the rest of the runtime (telemetry, `BusEntry.healthy`,
    /// the operator's UI) can tell the operator about the situation.
    ///
    /// Hard failure (returning `Err`) is reserved for cases where the
    /// SocketCAN socket can't be opened at all (interface doesn't exist,
    /// permissions, kernel module missing). A bus stuck in `ERROR-PASSIVE`
    /// is *not* such a case — the kernel will happily hand us a socket.
    pub fn open(interfaces: &[String]) -> Result<Self> {
        let mut by_iface = HashMap::new();
        for iface in interfaces {
            match read_can_state(iface).as_deref() {
                Some("ERROR-PASSIVE") | Some("BUS-OFF") => {
                    let state = read_can_state(iface).unwrap_or_else(|| "?".into());
                    warn!(
                        can_interface = iface,
                        state = %state,
                        "CAN bus is in {state}: this usually means the motor / \
                         power board on this bus is unpowered, unplugged, or \
                         simply not wired up yet. Opening anyway; arming any \
                         motor on this bus will be refused until it returns to \
                         ERROR-ACTIVE.\n  \
                         Recovery (if you expect a peer here):\n  \
                            1. Verify the device on this bus has power.\n  \
                            2. Briefly unplug+replug the CANHub USB cable.\n  \
                            3. Re-bring up the link:\n     \
                                  sudo ip link set {iface} down\n     \
                                  sudo ip link set {iface} type can bitrate 1000000\n     \
                                  sudo ip link set {iface} up\n  \
                            4. Verify with: ip -details link show {iface}\n     \
                               (should report 'can state ERROR-ACTIVE').",
                        state = state,
                        iface = iface,
                    );
                }
                Some(s) => {
                    info!(can_interface = iface, state = %s, "opening CAN bus");
                }
                None => {
                    warn!(
                        can_interface = iface,
                        "could not read CAN controller state \
                         (ip command unavailable or unexpected output); \
                         attempting to open anyway"
                    );
                }
            }
            let can = CanInterface::open(iface)?;
            by_iface.insert(iface.clone(), Arc::new(can));
        }
        Ok(Self {
            by_iface: RwLock::new(by_iface),
            last_refresh: Mutex::new(Instant::now()),
        })
    }

    pub fn get(&self, iface: &str) -> Option<Arc<CanInterface>> {
        self.by_iface.read().ok()?.get(iface).cloned()
    }

    /// Every bus as `(interface, socket)`. Used by the supervisor to spawn one
    /// RX thread per bus.
    pub fn snapshot(&self) -> Vec<(String, Arc<CanInterface>)> {
        match self.by_iface.read() {
            Ok(g) => g.iter().map(|(k, v)| (k.clone(), Arc::clone(v))).collect(),
            Err(_) => Vec::new(),
        }
    }

    pub fn interfaces(&self) -> Vec<String> {
        let mut v: Vec<String> = match self.by_iface.read() {
            Ok(g) => g.keys().cloned().collect(),
            Err(_) => Vec::new(),
        };
        v.sort();
        v
    }

    /// Re-open a bus whose socket can no longer be the one the kernel is
    /// using. Returns the interfaces it re-opened.
    ///
    /// This automates the remediation `open` prints. The case that matters is a
    /// USB-CAN adapter unplugged and replugged: systemd-networkd re-applies the
    /// link config on hotplug (so `ip` reports the *new* netdev as
    /// `ERROR-ACTIVE`), but the socket this process holds still refers to the
    /// netdev that went away — every send fails forever, the supervisor latches
    /// an E-STOP, and only a restart used to clear it.
    ///
    /// Triggered only by:
    ///   * the netdev being gone (`read_can_state` → `None`); the next tick
    ///     retries and succeeds once it is back, or
    ///   * a send failing *while the controller still reads `ERROR-ACTIVE`* —
    ///     the stale-socket signature, since a socket on the right netdev
    ///     cannot both fail and look healthy, or
    ///   * `BUS-OFF`, the one state a fresh socket does not fix and only a
    ///     link restart does.
    ///
    /// Deliberately *not* triggered by `ERROR-PASSIVE`, nor by a send failure
    /// on a bus that is already unhealthy. An unpowered peer — the documented
    /// common case, and the normal state of a partly-wired robot — produces
    /// exactly that, and bouncing the link there flaps a bus that is fine apart
    /// from having nobody on the other end.
    ///
    /// Throttled to [`REFRESH_INTERVAL`], so it is cheap on every telemetry
    /// tick.
    pub fn refresh_if_due(&self) -> Vec<String> {
        {
            let mut last = match self.last_refresh.lock() {
                Ok(g) => g,
                Err(p) => p.into_inner(),
            };
            if last.elapsed() < REFRESH_INTERVAL {
                return Vec::new();
            }
            *last = Instant::now();
        }

        let mut reopened = Vec::new();
        for iface in self.interfaces() {
            let state = read_can_state(&iface);
            let healthy = is_state_healthy(state.as_deref());
            let broken_socket = self.get(&iface).map(|c| c.io_error()).unwrap_or(true);
            let stale_socket = broken_socket && healthy;
            let bus_off = matches!(state.as_deref(), Some("BUS-OFF"));
            if !stale_socket && !bus_off && state.is_some() {
                continue;
            }
            if bus_off {
                for args in [
                    ["link", "set", iface.as_str(), "down"],
                    ["link", "set", iface.as_str(), "up"],
                ] {
                    let _ = Command::new("ip").args(args).output();
                }
            }
            match CanInterface::open(&iface) {
                Ok(can) => {
                    if let Ok(mut g) = self.by_iface.write() {
                        g.insert(iface.clone(), Arc::new(can));
                    }
                    warn!(
                        can_interface = %iface,
                        state = state.as_deref().unwrap_or("gone"),
                        bounced = bus_off,
                        "CAN bus re-opened (adapter came back / controller recovered)"
                    );
                    reopened.push(iface);
                }
                Err(e) => {
                    // Still absent (or unopenable) — the next tick retries.
                    // `debug`, not `info`: a robot whose hub is unplugged for
                    // hours should not write a line every few seconds. The bus
                    // state is in telemetry and the app banner shows it.
                    debug!(can_interface = %iface, error = %e,
                           "CAN bus not openable yet; will retry");
                }
            }
        }
        reopened
    }

    /// Live re-read the controller state for `iface`. Cheap (~1 ms via `ip`)
    /// but not free; call from the supervisor on demand, not in a hot loop.
    pub fn is_healthy(&self, iface: &str) -> bool {
        is_state_healthy(read_can_state(iface).as_deref())
    }
}
