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
use std::sync::Arc;
use tracing::{info, warn};

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
    by_iface: HashMap<String, Arc<CanInterface>>,
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
        Ok(Self { by_iface })
    }

    pub fn get(&self, iface: &str) -> Option<&Arc<CanInterface>> {
        self.by_iface.get(iface)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&String, &Arc<CanInterface>)> {
        self.by_iface.iter()
    }

    pub fn interfaces(&self) -> Vec<String> {
        let mut v: Vec<String> = self.by_iface.keys().cloned().collect();
        v.sort();
        v
    }

    /// Live re-read the controller state for `iface`. Cheap (~1 ms via `ip`)
    /// but not free; call from the supervisor on demand, not in a hot loop.
    pub fn is_healthy(&self, iface: &str) -> bool {
        is_state_healthy(read_can_state(iface).as_deref())
    }
}
