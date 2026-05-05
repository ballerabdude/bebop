//! Linux-only adapter between `evdev` and the platform-agnostic
//! [`super::mapping::RawEvent`] model.
//!
//! Two responsibilities:
//!
//! 1. Resolve a Bluetooth MAC to its `/dev/input/eventX` path. A single
//!    BT controller often exposes several input nodes (the PS5
//!    DualSense, for example, has separate gamepad / motion-sensor /
//!    touchpad nodes). We pick the one whose `EV_KEY` capabilities
//!    include the standard gamepad button range so we never bind to
//!    the touchpad by mistake.
//!
//! 2. Translate `evdev::InputEvent` values into [`RawEvent`]s, looking
//!    up calibration via the device's `AbsInfo` table on first use.

use std::collections::HashMap;
use std::path::PathBuf;

use evdev::{AbsInfo, AbsoluteAxisCode, Device, EventStream, EventSummary, KeyCode};
use tracing::debug;

use super::mapping::{AbsAxis, AbsCalibration, GamepadButton, RawEvent};

/// Result of resolving a MAC + opening the right input node. The
/// `event_path` is kept around for logging; the supervisor only needs
/// the stream + the calibration cache.
pub struct OpenedDevice {
    pub event_path: PathBuf,
    pub stream: EventStream,
    pub cal: AbsCalibrationCache,
}

/// Snapshot of the device's `AbsInfo` table, indexed by the axes we
/// care about. Captured once at `open` time; we don't re-query during
/// the teleop loop because the kernel guarantees these are stable for
/// the lifetime of the device.
#[derive(Debug, Default, Clone)]
pub struct AbsCalibrationCache {
    table: HashMap<AbsoluteAxisCode, AbsCalibration>,
}

impl AbsCalibrationCache {
    fn from_device(dev: &Device) -> Self {
        let mut table = HashMap::new();
        if let Ok(iter) = dev.get_absinfo() {
            for (code, info) in iter {
                table.insert(code, abs_to_calibration(&info));
            }
        }
        Self { table }
    }

    fn get(&self, code: AbsoluteAxisCode) -> Option<AbsCalibration> {
        self.table.get(&code).copied()
    }
}

fn abs_to_calibration(info: &AbsInfo) -> AbsCalibration {
    AbsCalibration {
        min: info.minimum(),
        max: info.maximum(),
        flat: info.flat(),
    }
}

/// Walk `/dev/input/event*`, find the gamepad node bound to `mac`, and
/// open it as an async event stream.
///
/// Returns `Ok(None)` if no node matches (controller not connected, or
/// connected but kernel hasn't enumerated yet — caller should back off
/// and retry).
pub fn open_for_mac(mac: &str) -> std::io::Result<Option<OpenedDevice>> {
    let normalised_mac = mac.to_ascii_lowercase();
    let dir = std::fs::read_dir("/dev/input")?;
    for entry in dir.flatten() {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if !name.starts_with("event") {
            continue;
        }
        let Ok(device) = Device::open(&path) else {
            continue;
        };
        let uniq = device
            .unique_name()
            .unwrap_or_default()
            .to_ascii_lowercase();
        if uniq != normalised_mac {
            continue;
        }
        if !is_gamepad_node(&device) {
            // Right MAC, wrong node (motion sensors, touchpad, ...).
            continue;
        }
        debug!(
            ?path,
            name = device.name().unwrap_or(""),
            "opened gamepad node"
        );
        let cal = AbsCalibrationCache::from_device(&device);
        let stream = device.into_event_stream()?;
        return Ok(Some(OpenedDevice {
            event_path: path,
            stream,
            cal,
        }));
    }
    Ok(None)
}

/// Heuristic: a node is "the gamepad" if it advertises BTN_SOUTH (the
/// universal "primary face button" code) or one of its near siblings.
/// On the DualSense the touchpad node has BTN_LEFT/BTN_RIGHT only and
/// the motion-sensors node has no keys — both fail this check.
fn is_gamepad_node(device: &Device) -> bool {
    let Some(keys) = device.supported_keys() else {
        return false;
    };
    keys.contains(KeyCode::BTN_SOUTH)
        || keys.contains(KeyCode::BTN_GAMEPAD)
        || keys.contains(KeyCode::BTN_JOYSTICK)
}

/// Translate a single `evdev::InputEvent` into the platform-agnostic
/// [`RawEvent`] used by `mapping::apply_event`. Returns `None` for
/// event types we don't care about (`SYN_REPORT`, MSC, FF, etc.) or
/// for axes we don't expose to the teleop layer (e.g. `ABS_HAT0X`).
pub fn translate(event: &evdev::InputEvent, cal: &AbsCalibrationCache) -> Option<RawEvent> {
    match event.destructure() {
        EventSummary::Key(_, code, value) => {
            let button = key_to_gamepad_button(code)?;
            // evdev convention: 0 = released, 1 = pressed, 2 = autorepeat.
            // Treat autorepeat as "still held" so we don't oscillate the
            // bitmask on long holds.
            let pressed = value != 0;
            Some(RawEvent::Button { button, pressed })
        }
        EventSummary::AbsoluteAxis(_, code, value) => {
            let axis = absolute_to_axis(code)?;
            let cal = cal.get(code)?;
            Some(RawEvent::Axis {
                axis,
                raw: value,
                cal,
            })
        }
        _ => None,
    }
}

fn key_to_gamepad_button(code: KeyCode) -> Option<GamepadButton> {
    Some(match code {
        KeyCode::BTN_SOUTH => GamepadButton::South,
        KeyCode::BTN_EAST => GamepadButton::East,
        KeyCode::BTN_WEST => GamepadButton::West,
        KeyCode::BTN_NORTH => GamepadButton::North,
        KeyCode::BTN_TL => GamepadButton::ShoulderL,
        KeyCode::BTN_TR => GamepadButton::ShoulderR,
        KeyCode::BTN_SELECT => GamepadButton::Select,
        KeyCode::BTN_START => GamepadButton::Start,
        KeyCode::BTN_MODE => GamepadButton::Mode,
        KeyCode::BTN_THUMBL => GamepadButton::ThumbL,
        KeyCode::BTN_THUMBR => GamepadButton::ThumbR,
        _ => return None,
    })
}

fn absolute_to_axis(code: AbsoluteAxisCode) -> Option<AbsAxis> {
    Some(match code {
        AbsoluteAxisCode::ABS_X => AbsAxis::LeftStickX,
        AbsoluteAxisCode::ABS_Y => AbsAxis::LeftStickY,
        AbsoluteAxisCode::ABS_RX => AbsAxis::RightStickX,
        AbsoluteAxisCode::ABS_RY => AbsAxis::RightStickY,
        AbsoluteAxisCode::ABS_Z => AbsAxis::LeftTrigger,
        AbsoluteAxisCode::ABS_RZ => AbsAxis::RightTrigger,
        _ => return None,
    })
}

/// Resolve the configured deadman/e-stop/arm button names from
/// `agent.toml` to typed [`GamepadButton`]s. Returns `None` for an
/// unknown button name; the supervisor falls back to defaults in that
/// case.
pub fn parse_button_name(name: &str) -> Option<GamepadButton> {
    Some(match name {
        "BTN_SOUTH" => GamepadButton::South,
        "BTN_EAST" => GamepadButton::East,
        "BTN_WEST" => GamepadButton::West,
        "BTN_NORTH" => GamepadButton::North,
        "BTN_TL" => GamepadButton::ShoulderL,
        "BTN_TR" => GamepadButton::ShoulderR,
        // We treat the analog trigger as the deadman by reading its
        // axis value (`gp.rt`), but accept the button name for symmetry
        // and to allow controllers without analog triggers to use the
        // digital trigger button.
        "BTN_TL2" => GamepadButton::ShoulderL,
        "BTN_TR2" => GamepadButton::ShoulderR,
        "BTN_SELECT" => GamepadButton::Select,
        "BTN_START" => GamepadButton::Start,
        "BTN_MODE" => GamepadButton::Mode,
        "BTN_THUMBL" => GamepadButton::ThumbL,
        "BTN_THUMBR" => GamepadButton::ThumbR,
        _ => return None,
    })
}
