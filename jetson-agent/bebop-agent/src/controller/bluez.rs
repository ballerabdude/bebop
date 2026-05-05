//! BlueZ pairing surface, implemented by shelling out to `bluetoothctl`.
//!
//! We use the same approach as `wifi/mod.rs` (which calls `nmcli`):
//! shelling out keeps us decoupled from a specific BlueZ DBus crate
//! version and avoids the BR/EDR + LE adapter contention that
//! concurrent `bluer` clients on the same adapter can produce.
//!
//! Public API is intentionally narrow: discover, pair, unpair, list,
//! is_connected. The supervisor in `mod.rs` orchestrates these. Output
//! parsers are kept private but defensively unit-tested in this file
//! against captured `bluetoothctl` output.

use std::time::Duration;

use anyhow::{anyhow, Result};
use tokio::process::Command;
use tracing::debug;

use crate::error::AgentError;

/// One row in `ControllerScanResult`. Same shape as the proto struct;
/// the dispatcher converts.
#[derive(Debug, Clone, PartialEq)]
pub struct DiscoveredDevice {
    pub mac: String,
    pub name: String,
    pub rssi: i32,
    pub paired: bool,
    pub connected: bool,
    pub kind: DeviceKind,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DeviceKind {
    Gamepad,
    Unknown,
}

impl DeviceKind {
    pub fn as_proto_str(self) -> &'static str {
        match self {
            DeviceKind::Gamepad => "gamepad",
            DeviceKind::Unknown => "unknown",
        }
    }
}

/// Run a discovery cycle for `timeout` and return everything BlueZ saw,
/// enriched with per-device `info` so we can classify gamepads.
pub async fn discover(timeout: Duration) -> Result<Vec<DiscoveredDevice>, AgentError> {
    // `--timeout N` runs scan for N seconds then exits cleanly. Nicer
    // than spawning a long-lived `scan on` and remembering to stop it.
    let secs = timeout.as_secs().max(1);
    debug!(secs, "starting bluetoothctl scan");
    let _ = bluetoothctl(&["--timeout", &secs.to_string(), "scan", "on"])
        .await
        .map_err(|e| AgentError::Controller(format!("scan: {e}")))?;

    let listing = bluetoothctl(&["devices"])
        .await
        .map_err(|e| AgentError::Controller(format!("devices: {e}")))?;

    let mut out = Vec::new();
    for (mac, name) in parse_devices_listing(&listing) {
        let info = bluetoothctl(&["info", &mac]).await.unwrap_or_default();
        let mut dev = parse_info(&mac, &name, &info);
        // The `Name:` field from `info` is more reliable than the
        // listing, but fall back to the listing when info is silent
        // (e.g. unpaired devices that BlueZ hasn't queried fully yet).
        if dev.name.is_empty() {
            dev.name = name;
        }
        out.push(dev);
    }
    Ok(out)
}

/// Pair, trust (so the agent can auto-reconnect on boot without the
/// user re-confirming), and connect. Idempotent: running it twice on a
/// paired device is a no-op apart from the `connect`.
pub async fn pair_and_connect(mac: &str) -> Result<(), AgentError> {
    bluetoothctl(&["pair", mac])
        .await
        .map_err(|e| AgentError::Controller(format!("pair {mac}: {e}")))?;
    bluetoothctl(&["trust", mac])
        .await
        .map_err(|e| AgentError::Controller(format!("trust {mac}: {e}")))?;
    bluetoothctl(&["connect", mac])
        .await
        .map_err(|e| AgentError::Controller(format!("connect {mac}: {e}")))?;
    Ok(())
}

/// Best-effort connect to an already-paired device. Used by the
/// supervisor to re-attach after the controller is power-cycled. We
/// don't care if this fails — the supervisor will back off and retry.
pub async fn try_connect(mac: &str) -> Result<(), AgentError> {
    bluetoothctl(&["connect", mac])
        .await
        .map_err(|e| AgentError::Controller(format!("connect {mac}: {e}")))?;
    Ok(())
}

/// Forget the device entirely (`bluetoothctl remove`). After this the
/// agent will need a full pair flow to re-bind.
pub async fn unpair(mac: &str) -> Result<(), AgentError> {
    bluetoothctl(&["remove", mac])
        .await
        .map_err(|e| AgentError::Controller(format!("remove {mac}: {e}")))?;
    Ok(())
}

/// True iff `bluetoothctl info` reports `Connected: yes` for `mac`.
pub async fn is_connected(mac: &str) -> bool {
    let Ok(info) = bluetoothctl(&["info", mac]).await else {
        return false;
    };
    parse_info_field(&info, "Connected")
        .map(|v| v == "yes")
        .unwrap_or(false)
}

// ---------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------

async fn bluetoothctl(args: &[&str]) -> Result<String> {
    let output = Command::new("bluetoothctl")
        .args(args)
        .output()
        .await
        .map_err(|e| anyhow!("spawn bluetoothctl: {e}"))?;
    if !output.status.success() {
        return Err(anyhow!(
            "bluetoothctl {:?} failed: {}",
            args,
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

/// Parse the output of `bluetoothctl devices`. Each line is of the
/// form `Device AA:BB:CC:DD:EE:FF Friendly Name`.
fn parse_devices_listing(out: &str) -> Vec<(String, String)> {
    let mut v = Vec::new();
    for line in out.lines() {
        let line = line.trim();
        // BlueZ sometimes prefixes lines with ANSI control chars when
        // running interactively; defensive trim.
        let line = strip_ansi(line);
        if let Some(rest) = line.strip_prefix("Device ") {
            let mut parts = rest.splitn(2, ' ');
            let Some(mac) = parts.next() else { continue };
            let name = parts.next().unwrap_or("").trim().to_owned();
            v.push((mac.to_owned(), name));
        }
    }
    v
}

/// Parse `bluetoothctl info <mac>` output. We extract Name, Paired,
/// Connected, RSSI, and the Class-of-Device + UUIDs so we can guess
/// whether the device is a gamepad.
fn parse_info(mac: &str, fallback_name: &str, info: &str) -> DiscoveredDevice {
    let name = parse_info_field(info, "Name").unwrap_or_else(|| fallback_name.to_owned());
    let paired = parse_info_field(info, "Paired")
        .map(|v| v == "yes")
        .unwrap_or(false);
    let connected = parse_info_field(info, "Connected")
        .map(|v| v == "yes")
        .unwrap_or(false);
    let rssi = parse_info_field(info, "RSSI")
        .and_then(|s| s.parse::<i32>().ok())
        .unwrap_or(0);

    let class_hex = parse_info_field(info, "Class");
    let uuids: Vec<String> = info
        .lines()
        .map(strip_ansi)
        .filter(|l| l.trim_start().starts_with("UUID:"))
        .filter_map(|l| {
            // `UUID: Human Readable           (00001124-0000-1000-8000-00805f9b34fb)`
            let open = l.find('(')?;
            let close = l.find(')')?;
            Some(l[open + 1..close].to_lowercase())
        })
        .collect();

    let kind = if looks_like_gamepad(class_hex.as_deref(), &uuids, &name) {
        DeviceKind::Gamepad
    } else {
        DeviceKind::Unknown
    };

    DiscoveredDevice {
        mac: mac.to_owned(),
        name,
        rssi,
        paired,
        connected,
        kind,
    }
}

fn parse_info_field(info: &str, field: &str) -> Option<String> {
    let needle = format!("{field}:");
    for line in info.lines().map(strip_ansi) {
        let trimmed = line.trim_start();
        if let Some(rest) = trimmed.strip_prefix(needle.as_str()) {
            return Some(rest.trim().to_owned());
        }
    }
    None
}

/// Heuristic: a device is a gamepad if any of these hold:
///   * Class-of-Device's major class is 0x05 (Peripheral) AND minor
///     class indicates joystick (0x04), gamepad (0x08), or
///     remote control (0x0c — many "controllers" report this).
///   * It advertises the HID service UUID `0x1124` (BR/EDR) or the BLE
///     HID service UUID `0x1812`.
///   * Heuristic name match for known gamepad brands. Last resort but
///     catches new hardware before we can update class/UUID heuristics.
fn looks_like_gamepad(class_hex: Option<&str>, uuids: &[String], name: &str) -> bool {
    if let Some(class_str) = class_hex {
        if let Some(class) = parse_class_hex(class_str) {
            // Class-of-Device is 24 bits. Major device class lives in
            // bits [12:8], minor in bits [7:2].
            let major = (class >> 8) & 0x1f;
            let minor = (class >> 2) & 0x3f;
            if major == 0x05 && (minor == 0x04 || minor == 0x08 || minor == 0x0c) {
                return true;
            }
        }
    }
    for u in uuids {
        if u.starts_with("00001124-") || u.starts_with("00001812-") {
            return true;
        }
    }
    let lname = name.to_lowercase();
    const NAME_HINTS: &[&str] = &[
        "controller",
        "gamepad",
        "joystick",
        "dualsense",
        "dualshock",
        "xbox",
        "8bitdo",
        "switch pro",
    ];
    NAME_HINTS.iter().any(|h| lname.contains(h))
}

fn parse_class_hex(s: &str) -> Option<u32> {
    let trimmed = s.trim();
    let body = trimmed
        .strip_prefix("0x")
        .or_else(|| trimmed.strip_prefix("0X"))
        .unwrap_or(trimmed);
    // BlueZ sometimes prints `0x002508 (9480)` — split off a trailing
    // decimal-in-parens if present.
    let body = body.split_whitespace().next().unwrap_or(body);
    u32::from_str_radix(body, 16).ok()
}

/// Drop ANSI escape sequences. `bluetoothctl` mostly behaves under
/// non-interactive use, but we've seen the occasional `\x1b[...m` slip
/// through depending on terminal env vars in service contexts.
fn strip_ansi(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\x1b' {
            // Skip until the terminating letter of the CSI sequence.
            for next in chars.by_ref() {
                if next.is_ascii_alphabetic() {
                    break;
                }
            }
        } else {
            out.push(c);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_devices_listing_basic() {
        let out = "\
Device 1A:2B:3C:4D:5E:6F DualSense Wireless Controller
Device AA:BB:CC:DD:EE:FF Some Headset
";
        let v = parse_devices_listing(out);
        assert_eq!(v.len(), 2);
        assert_eq!(v[0].0, "1A:2B:3C:4D:5E:6F");
        assert_eq!(v[0].1, "DualSense Wireless Controller");
    }

    #[test]
    fn parse_info_classifies_dualsense_as_gamepad() {
        // Trimmed to the fields we care about; real output is more verbose.
        let info = "\
Device 1A:2B:3C:4D:5E:6F (public)
\tName: DualSense Wireless Controller
\tAlias: DualSense Wireless Controller
\tClass: 0x002508
\tIcon: input-gaming
\tPaired: no
\tBonded: no
\tTrusted: no
\tBlocked: no
\tConnected: no
\tLegacyPairing: no
\tUUID: Human Interface Device   (00001124-0000-1000-8000-00805f9b34fb)
\tRSSI: -52
";
        let dev = parse_info("1A:2B:3C:4D:5E:6F", "DualSense Wireless Controller", info);
        assert_eq!(dev.kind, DeviceKind::Gamepad);
        assert_eq!(dev.name, "DualSense Wireless Controller");
        assert!(!dev.paired);
        assert!(!dev.connected);
        assert_eq!(dev.rssi, -52);
    }

    #[test]
    fn parse_info_rejects_a_headset() {
        let info = "\
Device AA:BB:CC:DD:EE:FF (public)
\tName: Awesome Headphones
\tClass: 0x240414
\tIcon: audio-card
\tPaired: yes
\tConnected: no
\tUUID: Audio Sink                (0000110b-0000-1000-8000-00805f9b34fb)
";
        let dev = parse_info("AA:BB:CC:DD:EE:FF", "Awesome Headphones", info);
        assert_eq!(dev.kind, DeviceKind::Unknown);
        assert!(dev.paired);
        assert!(!dev.connected);
    }

    #[test]
    fn name_hint_classifies_unknown_class_xbox_pad() {
        // Some Xbox-licensed pads under-report Class but advertise HID.
        let info = "\
\tName: Xbox Wireless Controller
\tConnected: no
";
        let dev = parse_info("11:22:33:44:55:66", "Xbox Wireless Controller", info);
        assert_eq!(dev.kind, DeviceKind::Gamepad);
    }

    #[test]
    fn ble_hid_uuid_classifies_as_gamepad() {
        let info = "\
\tName: NoNameThing
\tConnected: no
\tUUID: Human Interface Device   (00001812-0000-1000-8000-00805f9b34fb)
";
        let dev = parse_info("11:22:33:44:55:66", "NoNameThing", info);
        assert_eq!(dev.kind, DeviceKind::Gamepad);
    }

    #[test]
    fn class_hex_with_trailing_decimal_in_parens() {
        assert_eq!(parse_class_hex("0x002508 (9480)"), Some(0x002508));
        assert_eq!(parse_class_hex("002508"), Some(0x002508));
    }
}
