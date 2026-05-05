//! Pure teleop state machine: maps a [`GamepadState`] + wall-clock to a
//! UDP teleop command (or a one-shot reset / nothing at all).
//!
//! Lives outside the supervisor so the safety-critical bits (deadman
//! gate, latching e-stop, watchdog) are unit-tested without needing
//! BlueZ, evdev, or a UDP socket.

use std::time::{Duration, Instant};

use serde::Serialize;

use super::mapping::{GamepadButton, GamepadState};

/// Velocity command in bebop-linux's UDP wire format. Body-frame, ROS
/// REP-103 (x forward, y left, yaw counterclockwise positive).
///
/// Matches the `{"xvel","yvel","angvel"}` JSON shape documented in
/// `firmware/bebop-linux/README.md`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct TeleopCmd {
    pub xvel: f32,
    pub yvel: f32,
    pub angvel: f32,
}

impl TeleopCmd {
    pub const ZERO: Self = Self {
        xvel: 0.0,
        yvel: 0.0,
        angvel: 0.0,
    };
}

/// Tunables drawn from `ControllerConfig`. Carried as a separate
/// `Copy` struct so tests don't need to construct a full config.
#[derive(Debug, Clone, Copy)]
pub struct TeleopParams {
    pub deadzone: f32,
    pub max_lin_vel: f32,
    pub max_ang_vel: f32,
    pub deadman_threshold: f32,
    pub watchdog: Duration,
}

/// One side-effect surfaced from a single tick.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TeleopOut {
    /// Send this velocity command.
    Velocity(TeleopCmd),
    /// Latch entered this tick: send a one-shot `{"type":"reset"}`
    /// before the next velocity command. The supervisor is responsible
    /// for emitting the reset frame *and then* the zero-velocity
    /// command that follows on subsequent ticks.
    Reset,
}

/// Persistent state across ticks. Lives in the supervisor task; no
/// global state.
#[derive(Debug, Default, Clone)]
pub struct TeleopState {
    /// True iff the user has pressed the e-stop button since the
    /// previous arm. While latched, no velocity commands are emitted.
    estop_latched: bool,
    /// Edge-detect for the e-stop button: only latch on press, not on
    /// hold.
    prev_estop_pressed: bool,
    /// Edge-detect for the arm/clear button.
    prev_arm_pressed: bool,
    /// Set when we've emitted the one-shot Reset corresponding to the
    /// current latch.
    reset_emitted_for_latch: bool,
    /// Wall-clock of the most recent input event we've folded into the
    /// gamepad state. Drives the watchdog. `None` until the first
    /// event arrives.
    pub last_event_at: Option<Instant>,
}

impl TeleopState {
    pub fn estop_latched(&self) -> bool {
        self.estop_latched
    }

    /// Mark that an input event just arrived. Called by the supervisor
    /// before each `tick`.
    pub fn note_event(&mut self, now: Instant) {
        self.last_event_at = Some(now);
    }

    /// True iff the deadman is held AND no e-stop is latched. Surfaced
    /// up to the BLE status snapshot so the UI can render a green
    /// "armed" pill in real time.
    pub fn armed(&self, gp: &GamepadState, params: &TeleopParams) -> bool {
        !self.estop_latched && gp.rt >= params.deadman_threshold
    }
}

/// Drive one teleop tick.
///
/// `estop_btn` and `arm_btn` are the buttons configured in `agent.toml`;
/// the supervisor resolves them from string names once at startup.
///
/// Returns `Velocity(ZERO)` (rather than `None`) when the deadman is
/// released or a watchdog fires, because the firmware needs an explicit
/// "stop" rather than an absence of packets.
pub fn tick(
    state: &mut TeleopState,
    gp: &GamepadState,
    params: &TeleopParams,
    estop_btn: GamepadButton,
    arm_btn: GamepadButton,
    now: Instant,
) -> TeleopOut {
    // ----- e-stop edge detection ------------------------------------
    let estop_pressed_now = gp.is_pressed(estop_btn);
    if estop_pressed_now && !state.prev_estop_pressed && !state.estop_latched {
        state.estop_latched = true;
        state.reset_emitted_for_latch = false;
    }
    state.prev_estop_pressed = estop_pressed_now;

    let arm_pressed_now = gp.is_pressed(arm_btn);
    if arm_pressed_now && !state.prev_arm_pressed && state.estop_latched {
        state.estop_latched = false;
        state.reset_emitted_for_latch = false;
    }
    state.prev_arm_pressed = arm_pressed_now;

    // ----- emit one-shot reset on latch entry -----------------------
    if state.estop_latched && !state.reset_emitted_for_latch {
        state.reset_emitted_for_latch = true;
        return TeleopOut::Reset;
    }

    // ----- watchdog -------------------------------------------------
    let watchdog_fired = match state.last_event_at {
        Some(t) => now.saturating_duration_since(t) > params.watchdog,
        // No events yet — treat as watchdog'd; we'll still emit a zero
        // command so the firmware sees us as "alive but idle" rather
        // than absent.
        None => true,
    };

    if state.estop_latched || watchdog_fired || gp.rt < params.deadman_threshold {
        return TeleopOut::Velocity(TeleopCmd::ZERO);
    }

    // ----- shape stick input into a body-velocity command -----------
    TeleopOut::Velocity(stick_to_velocity(gp, params))
}

/// Convenience: compute the watchdog-zero command without any state
/// machine work. Used by the supervisor when the controller disconnects
/// mid-loop and we want to emit one final stop before tearing down.
pub fn idle_zero() -> TeleopCmd {
    TeleopCmd::ZERO
}

fn stick_to_velocity(gp: &GamepadState, params: &TeleopParams) -> TeleopCmd {
    let (lx, ly) = apply_radial_deadzone(gp.lx, gp.ly, params.deadzone);
    // Yaw is one-dimensional, so a 1-D deadzone is fine.
    let rx = apply_axial_deadzone(gp.rx, params.deadzone);

    TeleopCmd {
        xvel: ly * params.max_lin_vel,
        // Forward is x; positive yvel is "left" in REP-103, so map a
        // right-stick of "left" (lx < 0) to a positive yvel.
        yvel: -lx * params.max_lin_vel,
        // Right-stick to the right (rx > 0) should yaw clockwise, i.e.
        // negative angvel in REP-103.
        angvel: -rx * params.max_ang_vel,
    }
}

fn apply_radial_deadzone(x: f32, y: f32, deadzone: f32) -> (f32, f32) {
    let mag = (x * x + y * y).sqrt();
    if mag <= deadzone {
        return (0.0, 0.0);
    }
    // Re-scale so the post-deadzone vector still spans full magnitude
    // (otherwise the user gets a "dead" zone with no smooth ramp out).
    let scale = ((mag - deadzone) / (1.0 - deadzone)) / mag;
    (x * scale, y * scale)
}

fn apply_axial_deadzone(v: f32, deadzone: f32) -> f32 {
    if v.abs() <= deadzone {
        0.0
    } else {
        let sign = v.signum();
        sign * (v.abs() - deadzone) / (1.0 - deadzone)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::controller::mapping::{
        apply_event, AbsAxis, AbsCalibration, GamepadButton, RawEvent,
    };

    fn params() -> TeleopParams {
        TeleopParams {
            deadzone: 0.10,
            max_lin_vel: 0.6,
            max_ang_vel: 1.5,
            deadman_threshold: 0.5,
            watchdog: Duration::from_millis(200),
        }
    }

    fn stick_cal() -> AbsCalibration {
        AbsCalibration {
            min: 0,
            max: 255,
            flat: 0,
        }
    }

    fn trigger_cal() -> AbsCalibration {
        AbsCalibration {
            min: 0,
            max: 255,
            flat: 0,
        }
    }

    fn fresh(now: Instant) -> TeleopState {
        let mut s = TeleopState::default();
        s.note_event(now);
        s
    }

    #[test]
    fn centred_sticks_with_deadman_held_emit_zero() {
        let now = Instant::now();
        let mut state = fresh(now);
        let mut gp = GamepadState::default();
        gp.rt = 1.0; // deadman fully held

        let out = tick(
            &mut state,
            &gp,
            &params(),
            GamepadButton::East,
            GamepadButton::South,
            now,
        );
        assert_eq!(out, TeleopOut::Velocity(TeleopCmd::ZERO));
    }

    #[test]
    fn deadman_released_zeroes_motion_even_with_full_stick() {
        let now = Instant::now();
        let mut state = fresh(now);
        let mut gp = GamepadState::default();
        apply_event(
            &mut gp,
            RawEvent::Axis {
                axis: AbsAxis::LeftStickY,
                raw: 0,
                cal: stick_cal(),
            },
        ); // full forward
           // rt left at 0 -> deadman released

        let out = tick(
            &mut state,
            &gp,
            &params(),
            GamepadButton::East,
            GamepadButton::South,
            now,
        );
        assert_eq!(out, TeleopOut::Velocity(TeleopCmd::ZERO));
    }

    #[test]
    fn deadman_held_full_forward_yields_positive_xvel() {
        let now = Instant::now();
        let mut state = fresh(now);
        let mut gp = GamepadState::default();
        apply_event(
            &mut gp,
            RawEvent::Axis {
                axis: AbsAxis::LeftStickY,
                raw: 0,
                cal: stick_cal(),
            },
        );
        apply_event(
            &mut gp,
            RawEvent::Axis {
                axis: AbsAxis::RightTrigger,
                raw: 255,
                cal: trigger_cal(),
            },
        );

        let out = tick(
            &mut state,
            &gp,
            &params(),
            GamepadButton::East,
            GamepadButton::South,
            now,
        );
        match out {
            TeleopOut::Velocity(cmd) => {
                assert!(cmd.xvel > 0.0, "expected positive xvel, got {}", cmd.xvel);
                assert!((cmd.xvel - 0.6).abs() < 1e-3, "expected ~max_lin_vel");
                assert_eq!(cmd.yvel, 0.0);
                assert_eq!(cmd.angvel, 0.0);
            }
            other => panic!("expected Velocity, got {other:?}"),
        }
    }

    #[test]
    fn estop_button_press_emits_reset_then_zero() {
        let now = Instant::now();
        let mut state = fresh(now);
        let mut gp = GamepadState::default();
        gp.rt = 1.0;
        apply_event(
            &mut gp,
            RawEvent::Button {
                button: GamepadButton::East,
                pressed: true,
            },
        );

        let first = tick(
            &mut state,
            &gp,
            &params(),
            GamepadButton::East,
            GamepadButton::South,
            now,
        );
        assert_eq!(first, TeleopOut::Reset);
        assert!(state.estop_latched());

        // Subsequent tick after release should still be zero (latched).
        apply_event(
            &mut gp,
            RawEvent::Button {
                button: GamepadButton::East,
                pressed: false,
            },
        );
        let second = tick(
            &mut state,
            &gp,
            &params(),
            GamepadButton::East,
            GamepadButton::South,
            now,
        );
        assert_eq!(second, TeleopOut::Velocity(TeleopCmd::ZERO));
    }

    #[test]
    fn arm_button_clears_estop_latch() {
        let now = Instant::now();
        let mut state = fresh(now);
        let mut gp = GamepadState::default();
        gp.rt = 1.0;

        // Latch.
        apply_event(
            &mut gp,
            RawEvent::Button {
                button: GamepadButton::East,
                pressed: true,
            },
        );
        let _ = tick(
            &mut state,
            &gp,
            &params(),
            GamepadButton::East,
            GamepadButton::South,
            now,
        );
        assert!(state.estop_latched());

        // Release e-stop, press arm.
        apply_event(
            &mut gp,
            RawEvent::Button {
                button: GamepadButton::East,
                pressed: false,
            },
        );
        apply_event(
            &mut gp,
            RawEvent::Button {
                button: GamepadButton::South,
                pressed: true,
            },
        );
        let _ = tick(
            &mut state,
            &gp,
            &params(),
            GamepadButton::East,
            GamepadButton::South,
            now,
        );
        assert!(!state.estop_latched(), "arm button should clear latch");
    }

    #[test]
    fn watchdog_fires_after_quiet_window() {
        let t0 = Instant::now();
        let mut state = fresh(t0);
        let mut gp = GamepadState::default();
        apply_event(
            &mut gp,
            RawEvent::Axis {
                axis: AbsAxis::LeftStickY,
                raw: 0,
                cal: stick_cal(),
            },
        );
        gp.rt = 1.0;

        // Within window: motion flows.
        let p = params();
        let later = t0 + Duration::from_millis(50);
        let out = tick(
            &mut state,
            &gp,
            &p,
            GamepadButton::East,
            GamepadButton::South,
            later,
        );
        match out {
            TeleopOut::Velocity(cmd) => assert!(cmd.xvel > 0.0),
            _ => panic!(),
        }

        // After window: zero.
        let way_later = t0 + Duration::from_millis(500);
        let out = tick(
            &mut state,
            &gp,
            &p,
            GamepadButton::East,
            GamepadButton::South,
            way_later,
        );
        assert_eq!(out, TeleopOut::Velocity(TeleopCmd::ZERO));
    }

    #[test]
    fn deadzone_is_applied_radially() {
        let now = Instant::now();
        let mut state = fresh(now);
        let mut gp = GamepadState::default();
        gp.rt = 1.0;
        // Tiny stick deflection inside the deadzone.
        gp.lx = 0.05;
        gp.ly = 0.05;
        let out = tick(
            &mut state,
            &gp,
            &params(),
            GamepadButton::East,
            GamepadButton::South,
            now,
        );
        assert_eq!(out, TeleopOut::Velocity(TeleopCmd::ZERO));
    }

    #[test]
    fn armed_helper_reflects_deadman_and_estop() {
        let p = params();
        let now = Instant::now();
        let mut state = fresh(now);
        let mut gp = GamepadState::default();

        assert!(!state.armed(&gp, &p), "deadman not held");
        gp.rt = 1.0;
        assert!(state.armed(&gp, &p), "deadman held, no estop");

        apply_event(
            &mut gp,
            RawEvent::Button {
                button: GamepadButton::East,
                pressed: true,
            },
        );
        let _ = tick(
            &mut state,
            &gp,
            &p,
            GamepadButton::East,
            GamepadButton::South,
            now,
        );
        assert!(!state.armed(&gp, &p), "estop latched -> not armed");
    }
}
