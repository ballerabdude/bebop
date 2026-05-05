//! Pure (no-IO, no-evdev) gamepad event model + axis normalisation.
//!
//! `evdev_input.rs` translates platform `evdev::InputEvent`s into the
//! [`RawEvent`] values defined here, and the supervisor folds them into a
//! [`GamepadState`] via [`apply_event`]. The teleop layer in `teleop.rs`
//! consumes [`GamepadState`] and never touches evdev directly.
//!
//! Splitting the model out like this means:
//!   * the unit tests in this module and in `teleop.rs` build on macOS,
//!     where the `evdev` crate isn't available;
//!   * a future mapping backend (SDL2 GameControllerDB, in-app
//!     calibration) can plug in by emitting the same [`RawEvent`]s
//!     without rewriting the teleop state machine.

/// Logical analog axes exposed by a "standard" gamepad. Maps cleanly to
/// SDL2 / Linux-evdev / XInput naming.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AbsAxis {
    LeftStickX,
    LeftStickY,
    RightStickX,
    RightStickY,
    LeftTrigger,
    RightTrigger,
}

/// Logical buttons we care about for teleop. The d-pad is intentionally
/// omitted — we drive motion exclusively from the analog sticks so the
/// d-pad is free for higher-level UI bindings later.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum GamepadButton {
    South = 0, // PS5 Cross / Xbox A
    East = 1,  // PS5 Circle / Xbox B
    West = 2,  // PS5 Square / Xbox X
    North = 3, // PS5 Triangle / Xbox Y
    ShoulderL = 4,
    ShoulderR = 5,
    Select = 6,
    Start = 7,
    Mode = 8, // PS button
    ThumbL = 9,
    ThumbR = 10,
}

impl GamepadButton {
    fn bit(self) -> u32 {
        1u32 << (self as u8 as u32)
    }
}

/// Raw axis calibration as reported by the kernel. We snapshot it once
/// when the device is opened and use it to normalise raw values.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AbsCalibration {
    pub min: i32,
    pub max: i32,
    /// Hardware-reported flat zone around the resting position. We
    /// honour it, then layer the user-configured radial deadzone in
    /// `teleop.rs` on top.
    pub flat: i32,
}

/// A single platform-agnostic gamepad event. Produced by
/// `evdev_input::stream_events` on Linux; consumed by [`apply_event`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RawEvent {
    Axis {
        axis: AbsAxis,
        raw: i32,
        cal: AbsCalibration,
    },
    Button {
        button: GamepadButton,
        pressed: bool,
    },
}

/// Latest known state of the gamepad. Sticks and triggers are
/// post-calibration but pre-deadzone; the radial deadzone is applied
/// per-tick in `teleop.rs` so the user can tune it without restarting.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct GamepadState {
    /// Left stick X, normalised to `[-1.0, 1.0]`. +X = right.
    pub lx: f32,
    /// Left stick Y, normalised to `[-1.0, 1.0]`. +Y = up (we invert
    /// the kernel's "down is positive" axis here so callers don't have
    /// to remember).
    pub ly: f32,
    pub rx: f32,
    pub ry: f32,
    /// Left trigger, normalised to `[0.0, 1.0]`.
    pub lt: f32,
    /// Right trigger, normalised to `[0.0, 1.0]`. Used as the deadman
    /// gate in `teleop.rs`.
    pub rt: f32,
    /// Bitmask indexed by [`GamepadButton`].
    buttons: u32,
}

impl GamepadState {
    pub fn is_pressed(&self, b: GamepadButton) -> bool {
        (self.buttons & b.bit()) != 0
    }

    fn set(&mut self, b: GamepadButton, pressed: bool) {
        if pressed {
            self.buttons |= b.bit();
        } else {
            self.buttons &= !b.bit();
        }
    }
}

/// Fold `event` into `state`. Returns the updated state for
/// convenience; callers may also rely on the in-place mutation.
pub fn apply_event(state: &mut GamepadState, event: RawEvent) {
    match event {
        RawEvent::Axis { axis, raw, cal } => match axis {
            AbsAxis::LeftStickX => state.lx = normalise_stick(raw, cal),
            // Kernel reports "stick down" as positive on Y axes; flip
            // so callers can write `xvel = ly` and have "up = forward".
            AbsAxis::LeftStickY => state.ly = -normalise_stick(raw, cal),
            AbsAxis::RightStickX => state.rx = normalise_stick(raw, cal),
            AbsAxis::RightStickY => state.ry = -normalise_stick(raw, cal),
            AbsAxis::LeftTrigger => state.lt = normalise_trigger(raw, cal),
            AbsAxis::RightTrigger => state.rt = normalise_trigger(raw, cal),
        },
        RawEvent::Button { button, pressed } => state.set(button, pressed),
    }
}

/// Map a raw stick value (`min..max` inclusive) into `[-1.0, 1.0]`,
/// snapping the hardware-reported flat zone to exactly 0.
fn normalise_stick(raw: i32, cal: AbsCalibration) -> f32 {
    let min = cal.min as f32;
    let max = cal.max as f32;
    if max <= min {
        return 0.0;
    }
    let centre = 0.5 * (min + max);
    let half = 0.5 * (max - min);
    let v = (raw as f32 - centre) / half;
    let flat_norm = if half > 0.0 {
        cal.flat as f32 / half
    } else {
        0.0
    };
    if v.abs() <= flat_norm {
        0.0
    } else {
        v.clamp(-1.0, 1.0)
    }
}

/// Map a raw trigger value (`min..max`) into `[0.0, 1.0]`. The kernel
/// reports trigger axes as unipolar (0 at rest, max at full press) so
/// we don't centre them.
fn normalise_trigger(raw: i32, cal: AbsCalibration) -> f32 {
    let min = cal.min as f32;
    let max = cal.max as f32;
    if max <= min {
        return 0.0;
    }
    ((raw as f32 - min) / (max - min)).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ds_stick_cal() -> AbsCalibration {
        // PS5 DualSense over BT reports 0..255 with flat=15 in practice.
        AbsCalibration {
            min: 0,
            max: 255,
            flat: 15,
        }
    }

    fn ds_trigger_cal() -> AbsCalibration {
        AbsCalibration {
            min: 0,
            max: 255,
            flat: 0,
        }
    }

    #[test]
    fn stick_centred_value_clamps_to_zero() {
        let mut s = GamepadState::default();
        apply_event(
            &mut s,
            RawEvent::Axis {
                axis: AbsAxis::LeftStickX,
                raw: 128,
                cal: ds_stick_cal(),
            },
        );
        assert_eq!(s.lx, 0.0);
    }

    #[test]
    fn stick_full_right_normalises_to_plus_one() {
        let mut s = GamepadState::default();
        apply_event(
            &mut s,
            RawEvent::Axis {
                axis: AbsAxis::LeftStickX,
                raw: 255,
                cal: ds_stick_cal(),
            },
        );
        assert!((s.lx - 1.0).abs() < 1e-6, "got {}", s.lx);
    }

    #[test]
    fn stick_full_left_normalises_to_minus_one() {
        let mut s = GamepadState::default();
        apply_event(
            &mut s,
            RawEvent::Axis {
                axis: AbsAxis::LeftStickX,
                raw: 0,
                cal: ds_stick_cal(),
            },
        );
        assert!((s.lx + 1.0).abs() < 1e-6, "got {}", s.lx);
    }

    #[test]
    fn left_stick_y_is_inverted_so_up_is_positive() {
        let mut s = GamepadState::default();
        // raw=0 means "stick pushed up" on the kernel's convention;
        // we want that to surface as a positive ly.
        apply_event(
            &mut s,
            RawEvent::Axis {
                axis: AbsAxis::LeftStickY,
                raw: 0,
                cal: ds_stick_cal(),
            },
        );
        assert!(s.ly > 0.0, "expected up-stick to give positive ly");
    }

    #[test]
    fn trigger_full_press_is_one() {
        let mut s = GamepadState::default();
        apply_event(
            &mut s,
            RawEvent::Axis {
                axis: AbsAxis::RightTrigger,
                raw: 255,
                cal: ds_trigger_cal(),
            },
        );
        assert!((s.rt - 1.0).abs() < 1e-6);
    }

    #[test]
    fn trigger_at_rest_is_zero() {
        let mut s = GamepadState::default();
        apply_event(
            &mut s,
            RawEvent::Axis {
                axis: AbsAxis::RightTrigger,
                raw: 0,
                cal: ds_trigger_cal(),
            },
        );
        assert_eq!(s.rt, 0.0);
    }

    #[test]
    fn button_press_release_round_trip() {
        let mut s = GamepadState::default();
        assert!(!s.is_pressed(GamepadButton::East));
        apply_event(
            &mut s,
            RawEvent::Button {
                button: GamepadButton::East,
                pressed: true,
            },
        );
        assert!(s.is_pressed(GamepadButton::East));
        assert!(!s.is_pressed(GamepadButton::South));
        apply_event(
            &mut s,
            RawEvent::Button {
                button: GamepadButton::East,
                pressed: false,
            },
        );
        assert!(!s.is_pressed(GamepadButton::East));
    }
}
