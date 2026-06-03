//! Standalone actuator system-identification (sysid) driver.
//!
//! Drives **one** Robstride actuator through a chosen excitation maneuver
//! and logs synchronized CAN feedback (`position` / `velocity` / `torque` /
//! `temperature`) to a CSV file. The CSV is then fed to
//! `sim/bebop_training/tools/sysid_fit.py`, which fits the four
//! Isaac Lab `DCMotorCfg` parameters used in
//! `sim/bebop_training/experiments/exp_standing.py`:
//!
//! | Parameter           | Maneuver that measures it          |
//! |---------------------|------------------------------------|
//! | `friction` (Nm)     | `friction-sweep`                   |
//! | `armature` (kg·m²)  | `torque-step` / `torque-chirp`     |
//! | `velocity_limit`    | `noload-speed`                     |
//! | `saturation_effort` | `stall-torque`                     |
//!
//! This bin reuses the production motor stack (`RobotConfig`,
//! `CanInterface`, `RobstrideMotor`) but **bypasses** the `Supervisor` /
//! WebSocket server so it has full control over `kp` / `kd` / `torque` /
//! `velocity` — the MIT-mode channels the wire API does not expose. It
//! enforces its own self-contained safety envelope (per-tick clamp,
//! feedback watchdog, limit/fault abort, Ctrl-C ramp-down).
//!
//! # Workflow
//!
//! 1. **Stop the runtime first** so two controllers don't fight on the
//!    same bus:
//!
//!    ```text
//!    sudo systemctl stop bebop-linux
//!    ```
//!
//! 2. Run each maneuver, once per joint. The required `--setup` flag
//!    declares the physical rig and gates which maneuvers may run.
//!    Example (left knee, friction, robot hanging on a stand):
//!
//!    ```text
//!    cargo run --release --bin sysid -- \
//!        --config config/bebop_v2.yaml \
//!        --joint knee_flexion_left_joint \
//!        --maneuver friction-sweep \
//!        --setup hanging
//!    ```
//!
//!    Maneuvers that must exceed the deploy safety caps (true no-load
//!    speed > YAML `vel_max`, true stall torque > YAML `tau_max`) are only
//!    permitted on `--setup bench` and also require an explicit, confirmed
//!    `--allow-cap-override`:
//!
//!    ```text
//!    cargo run --release --bin sysid -- \
//!        --joint hip_flexion_left_joint \
//!        --maneuver noload-speed --setup bench --allow-cap-override
//!    ```
//!
//! 3. Fit the logs:
//!
//!    ```text
//!    python sim/bebop_training/tools/sysid_fit.py ~/bebop-sysid-logs
//!    ```
//!
//! # Test setups (`--setup`, required)
//!
//! - **`bench`** — single motor on a bench/fixture. All maneuvers,
//!   including `stall-torque` (block the output) and `noload-speed` (free
//!   shaft). The only setup permitted for the cap-override T-N maneuvers.
//! - **`hanging`** — robot suspended with the leg free to swing its full
//!   travel: `friction-sweep`, `torque-step`, `torque-chirp`.
//!   `friction-sweep` is bidirectional so gravity bias cancels out.
//! - **`stand`** — robot supported / feet on the ground: `friction-sweep`
//!   and `torque-chirp` only (keep amplitudes low).
//!
//! # Maneuvers — what movement to expect
//!
//! All maneuvers send `position = current` so there is no position "kick";
//! motion comes only from the velocity / torque channels described below.
//! Every command is clamped to the joint's limits and the run aborts +
//! disables on any limit / fault / feedback-timeout breach, so the
//! worst case is an early safety stop, not a runaway.
//!
//! ### `friction-sweep`  (measures Coulomb + viscous friction)
//! - **Commands:** `kp = 0`, velocity-tracking `kd`; steps through
//!   `--vel-steps` speed magnitudes (default 6), each in **both**
//!   directions, holding each for `--dwell-s` (default 1.2 s). Top speed is
//!   `--vel-max-frac` x `vel_max` (default 0.6).
//! - **Expected motion:** the joint **rotates continuously**, first one way
//!   then the other, in steps of increasing speed. Total ~15 s at defaults.
//! - **Heads-up:** because this is velocity control, the joint keeps
//!   turning. On a free **bench** shaft it just spins. On an assembled
//!   joint with limited travel it will reach its position limit and
//!   safety-abort within a fraction of a second unless you reduce
//!   `--vel-max-frac` (e.g. 0.1) and `--dwell-s` (e.g. 0.2) so each segment
//!   stays inside the joint's range. Bench is best for this test.
//!
//! ### `torque-step`  (measures rotor inertia / `armature`)
//! - **Commands:** `kp = kd = 0`, open-loop torque. Sequence: 0.3 s idle,
//!   `+tau` for `--step-duration` (0.3 s), 0.6 s idle, `-tau` for 0.3 s,
//!   0.3 s idle. `tau = --tau-frac` x `tau_max` (default 0.3). Total ~1.8 s.
//! - **Expected motion:** two sharp **kicks** — the joint snaps one way,
//!   coasts, then snaps back the other way. Brief but abrupt; give the leg
//!   room to swing.
//!
//! ### `torque-chirp`  (measures rotor inertia / `armature`)
//! - **Commands:** `kp = kd = 0`, sinusoidal torque whose frequency sweeps
//!   `--chirp-f0` -> `--chirp-f1` (0.5 -> 6 Hz) over `--chirp-duration`
//!   (10 s), amplitude `--tau-frac` x `tau_max`.
//! - **Expected motion:** the joint **oscillates / shakes** about its start
//!   pose, with the **largest swings at the start** (low frequency) shrinking
//!   to a fast buzz as the frequency rises. If the early low-frequency swing
//!   is too wide, lower `--tau-frac`.
//!
//! ### `noload-speed`  (measures `velocity_limit`; **bench only**)
//! - **Commands:** `kp = kd = 0`, torque ramps 0 -> amp over `--ramp-s`
//!   (3 s) then holds for `--hold-s` (2 s). With `--allow-cap-override` the
//!   amplitude uses the motor's full torque envelope.
//! - **Expected motion:** a **free shaft spins up to its top (no-load)
//!   speed** and holds it — fast continuous rotation. Never run this on an
//!   assembled leg; it would drive the link into its hard stop at speed.
//!
//! ### `stall-torque`  (measures `saturation_effort`; **bench only**)
//! - **Commands:** `kp = kd = 0`, torque ramps 0 -> amp over `--ramp-s`
//!   then holds. Output must be **mechanically blocked**.
//! - **Expected motion:** ideally **none** — the blocked shaft holds while
//!   torque climbs (you'll feel/hear the load build in the fixture). If the
//!   shaft slips and turns, the run warns that the peak torque is
//!   underestimated. Reaction torque is large; clamp the fixture solidly.
//!
//! # Exit codes
//!
//! | Code | Meaning                                             |
//! |------|-----------------------------------------------------|
//! | 0    | Maneuver completed (or clean Ctrl-C ramp-down)      |
//! | 1    | Setup error (config / CAN / unknown joint)          |
//! | 2    | Safety abort (limit / fault / feedback watchdog)    |

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};
use clap::Parser;

use bebop_linux::can_interface::CanInterface;
use bebop_linux::config::{JointCommand, JointConfig, RobotConfig, RobstrideSpecs};
use bebop_linux::robstride::RobstrideMotor;

// ===========================================================================
// CLI
// ===========================================================================

#[derive(Debug, Parser)]
#[command(
    version,
    about = "Robstride actuator system-identification driver (single joint)."
)]
struct Args {
    /// Robot config YAML (joint table, CAN buses, hard limits).
    #[arg(long, default_value = "config/bebop_v2.yaml")]
    config: String,

    /// Joint to excite, by name (e.g. `knee_flexion_left_joint`). Mutually
    /// exclusive with `--motor-id` / `--bus`.
    #[arg(long)]
    joint: Option<String>,

    /// Joint to excite, by Robstride CAN node id (use with `--bus`).
    #[arg(long)]
    motor_id: Option<u8>,

    /// CAN interface for `--motor-id` (e.g. `can0`). Ignored when `--joint`
    /// is given.
    #[arg(long)]
    bus: Option<String>,

    /// Excitation maneuver. One of: `friction-sweep`, `torque-step`,
    /// `torque-chirp`, `noload-speed`, `stall-torque`.
    #[arg(long, default_value = "friction-sweep")]
    maneuver: String,

    /// Physical test setup (REQUIRED) — gates which maneuvers may run:
    ///   `bench`   single motor on a bench/fixture -> all maneuvers
    ///   `hanging` robot suspended, leg swings free -> friction-sweep, torque-step, torque-chirp
    ///   `stand`   robot supported / feet down -> friction-sweep, torque-chirp (low amplitude)
    #[arg(long)]
    setup: String,

    /// Output directory for CSV logs. `~` is expanded.
    #[arg(long, default_value = "~/bebop-sysid-logs")]
    out: String,

    /// Command / feedback loop rate (Hz).
    #[arg(long, default_value_t = 500.0)]
    rate_hz: f32,

    /// Issue a Robstride `set_zero` (mechanical zero) before enabling.
    /// The motor is disabled while zeroing, then re-enabled.
    #[arg(long, default_value_t = false)]
    set_zero: bool,

    /// Allow `noload-speed` / `stall-torque` to widen the joint's
    /// `vel_max` / `tau_max` beyond the deploy safety envelope. Prompts
    /// for confirmation. Required for a faithful T-N curve.
    #[arg(long, default_value_t = false)]
    allow_cap_override: bool,

    // --- friction-sweep ----------------------------------------------------
    /// Number of distinct |velocity| set-points in the friction sweep.
    #[arg(long, default_value_t = 6)]
    vel_steps: usize,

    /// Highest swept velocity as a fraction of the joint's `vel_max`.
    #[arg(long, default_value_t = 0.6)]
    vel_max_frac: f32,

    /// Dwell time at each velocity set-point (s).
    #[arg(long, default_value_t = 1.2)]
    dwell_s: f32,

    /// Velocity-tracking `kd` for the friction sweep (defaults to the
    /// joint's `test_gains.kd`). `kp` is held at 0 so torque == load.
    #[arg(long)]
    kd: Option<f32>,

    // --- torque maneuvers --------------------------------------------------
    /// Torque amplitude as a fraction of the effective `tau_max` cap.
    #[arg(long, default_value_t = 0.3)]
    tau_frac: f32,

    /// Per-step duration for `torque-step` (s).
    #[arg(long, default_value_t = 0.3)]
    step_duration: f32,

    /// `torque-chirp` start / end frequency (Hz) and total duration (s).
    #[arg(long, default_value_t = 0.5)]
    chirp_f0: f32,
    #[arg(long, default_value_t = 6.0)]
    chirp_f1: f32,
    #[arg(long, default_value_t = 10.0)]
    chirp_duration: f32,

    /// Torque ramp time for `noload-speed` / `stall-torque` (s).
    #[arg(long, default_value_t = 3.0)]
    ramp_s: f32,

    /// Hold time after the ramp for `noload-speed` / `stall-torque` (s).
    #[arg(long, default_value_t = 2.0)]
    hold_s: f32,
}

// ===========================================================================
// Maneuvers
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Maneuver {
    FrictionSweep,
    TorqueStep,
    TorqueChirp,
    NoloadSpeed,
    StallTorque,
}

impl Maneuver {
    fn parse(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().replace('_', "-").as_str() {
            "friction-sweep" | "friction" => Ok(Self::FrictionSweep),
            "torque-step" | "step" => Ok(Self::TorqueStep),
            "torque-chirp" | "chirp" => Ok(Self::TorqueChirp),
            "noload-speed" | "noload" => Ok(Self::NoloadSpeed),
            "stall-torque" | "stall" => Ok(Self::StallTorque),
            other => Err(anyhow!(
                "unknown maneuver {other:?} (expected friction-sweep, torque-step, \
                 torque-chirp, noload-speed, stall-torque)"
            )),
        }
    }

    fn as_str(&self) -> &'static str {
        match self {
            Self::FrictionSweep => "friction-sweep",
            Self::TorqueStep => "torque-step",
            Self::TorqueChirp => "torque-chirp",
            Self::NoloadSpeed => "noload-speed",
            Self::StallTorque => "stall-torque",
        }
    }

    /// True when the maneuver intends to exceed the deploy safety caps and
    /// therefore requires `--allow-cap-override`.
    fn needs_cap_override(&self) -> bool {
        matches!(self, Self::NoloadSpeed | Self::StallTorque)
    }

    /// One-line description of the motion the operator should expect, shown
    /// just before the motor is enabled.
    fn expectation(&self) -> &'static str {
        match self {
            Self::FrictionSweep => {
                "joint ROTATES continuously, both directions, stepping up in speed \
                 (free shaft on a bench; on a constrained joint lower --vel-max-frac / --dwell-s)"
            }
            Self::TorqueStep => "two sharp KICKS — joint snaps one way, coasts, then snaps back",
            Self::TorqueChirp => {
                "joint OSCILLATES about its start pose; widest swings at the start, \
                 shrinking to a fast buzz as frequency rises"
            }
            Self::NoloadSpeed => "free shaft SPINS UP to its top no-load speed and holds (bench only)",
            Self::StallTorque => {
                "blocked shaft holds ~STILL while torque ramps up; large reaction load in the fixture (bench only)"
            }
        }
    }
}

/// Physical test setup. Gates which maneuvers are allowed to run so that,
/// e.g., a free-spinning no-load run can't be commanded on an assembled
/// leg, and a stall run can't be commanded with nothing to react against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Setup {
    /// Single motor on a bench / fixture: full range, including the
    /// cap-override T-N-curve maneuvers.
    Bench,
    /// Robot suspended with the leg free to swing through its full travel:
    /// in-envelope maneuvers only (no cap override — a free leg driven to
    /// the true no-load speed would slam its hard stop, and there is
    /// nothing to stall against).
    Hanging,
    /// Robot supported / feet on the ground: only small, low-amplitude
    /// in-envelope excitation.
    Stand,
}

impl Setup {
    fn parse(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "bench" => Ok(Self::Bench),
            "hanging" | "hang" | "suspended" => Ok(Self::Hanging),
            "stand" | "ground" | "standing" => Ok(Self::Stand),
            other => Err(anyhow!(
                "unknown setup {other:?} (expected bench, hanging, or stand)"
            )),
        }
    }

    fn as_str(&self) -> &'static str {
        match self {
            Self::Bench => "bench",
            Self::Hanging => "hanging",
            Self::Stand => "stand",
        }
    }

    /// Maneuvers permitted in this setup.
    fn allowed(&self) -> &'static [Maneuver] {
        match self {
            Self::Bench => &[
                Maneuver::FrictionSweep,
                Maneuver::TorqueStep,
                Maneuver::TorqueChirp,
                Maneuver::NoloadSpeed,
                Maneuver::StallTorque,
            ],
            Self::Hanging => &[
                Maneuver::FrictionSweep,
                Maneuver::TorqueStep,
                Maneuver::TorqueChirp,
            ],
            Self::Stand => &[Maneuver::FrictionSweep, Maneuver::TorqueChirp],
        }
    }

    fn allows(&self, maneuver: Maneuver) -> bool {
        self.allowed().contains(&maneuver)
    }
}

/// Effective safety envelope for this run (a copy of the joint's hard
/// limits, optionally widened for the T-N-curve maneuvers).
#[derive(Debug, Clone, Copy)]
struct EffLimits {
    pos_min: f32,
    pos_max: f32,
    vel_max: f32,
    tau_max: f32,
    temp_max: f32,
    feedback_timeout_ms: f32,
}

/// Stateless excitation generator: given elapsed time `t` (s) and the
/// latest measured position, return the next pre-clamp command, or `None`
/// when the maneuver is finished.
struct Excitation {
    maneuver: Maneuver,
    kd_track: f32,
    /// Friction sweep velocity schedule (rad/s), one entry per dwell slot.
    vel_schedule: Vec<f32>,
    dwell_s: f32,
    warmup_s: f32,
    tau_amp: f32,
    step_duration: f32,
    chirp_f0: f32,
    chirp_f1: f32,
    chirp_duration: f32,
    ramp_s: f32,
    hold_s: f32,
}

impl Excitation {
    fn build(args: &Args, maneuver: Maneuver, limits: &EffLimits, joint: &JointConfig) -> Self {
        let kd_track = args.kd.unwrap_or(joint.test_gains.kd);

        // Symmetric, bidirectional velocity set-points: +v1, -v1, +v2, -v2 ...
        let mut vel_schedule = Vec::new();
        let vmax = (limits.vel_max * args.vel_max_frac).max(0.0);
        let steps = args.vel_steps.max(1);
        for i in 1..=steps {
            let v = vmax * (i as f32) / (steps as f32);
            vel_schedule.push(v);
            vel_schedule.push(-v);
        }

        let tau_amp = (limits.tau_max * args.tau_frac).max(0.0);

        Self {
            maneuver,
            kd_track,
            vel_schedule,
            dwell_s: args.dwell_s.max(0.05),
            warmup_s: 0.4,
            tau_amp,
            step_duration: args.step_duration.max(0.05),
            chirp_f0: args.chirp_f0.max(0.0),
            chirp_f1: args.chirp_f1.max(0.0),
            chirp_duration: args.chirp_duration.max(0.5),
            ramp_s: args.ramp_s.max(0.1),
            hold_s: args.hold_s.max(0.0),
        }
    }

    fn command(&self, t: f32, fb_pos: f32) -> Option<JointCommand> {
        match self.maneuver {
            Maneuver::FrictionSweep => self.friction_sweep(t, fb_pos),
            Maneuver::TorqueStep => self.torque_step(t, fb_pos),
            Maneuver::TorqueChirp => self.torque_chirp(t, fb_pos),
            Maneuver::NoloadSpeed => self.torque_ramp(t, fb_pos),
            Maneuver::StallTorque => self.torque_ramp(t, fb_pos),
        }
    }

    /// `kp = 0`, `kd > 0` velocity tracking. At steady state the motor's
    /// torque equals the load (friction ± gravity), which is exactly the
    /// quantity we record. Bidirectional set-points let the fitter cancel
    /// the constant gravity bias.
    fn friction_sweep(&self, t: f32, fb_pos: f32) -> Option<JointCommand> {
        if t < self.warmup_s {
            return Some(JointCommand {
                position: fb_pos,
                velocity: 0.0,
                torque: 0.0,
                kp: 0.0,
                kd: self.kd_track,
            });
        }
        let idx = ((t - self.warmup_s) / self.dwell_s) as usize;
        let v = *self.vel_schedule.get(idx)?;
        Some(JointCommand {
            position: fb_pos,
            velocity: v,
            torque: 0.0,
            kp: 0.0,
            kd: self.kd_track,
        })
    }

    /// Open-loop torque (`kp = kd = 0`): warmup, +step, settle, -step,
    /// tail. The fitter differentiates velocity to get acceleration and
    /// fits the rotor inertia.
    fn torque_step(&self, t: f32, fb_pos: f32) -> Option<JointCommand> {
        let warmup = 0.3;
        let settle = 0.6;
        let tail = 0.3;
        let t1 = warmup;
        let t2 = t1 + self.step_duration; // +step
        let t3 = t2 + settle;
        let t4 = t3 + self.step_duration; // -step
        let t5 = t4 + tail;

        let torque = if t < t1 {
            0.0
        } else if t < t2 {
            self.tau_amp
        } else if t < t3 {
            0.0
        } else if t < t4 {
            -self.tau_amp
        } else if t < t5 {
            0.0
        } else {
            return None;
        };
        Some(JointCommand {
            position: fb_pos,
            velocity: 0.0,
            torque,
            kp: 0.0,
            kd: 0.0,
        })
    }

    /// Open-loop linear-frequency-sweep torque about the start position.
    fn torque_chirp(&self, t: f32, fb_pos: f32) -> Option<JointCommand> {
        if t >= self.chirp_duration {
            return None;
        }
        let f0 = self.chirp_f0;
        let f1 = self.chirp_f1;
        let dur = self.chirp_duration;
        // Instantaneous phase of a linear chirp: 2π (f0 t + (f1-f0) t²/2dur).
        let phase =
            2.0 * std::f32::consts::PI * (f0 * t + 0.5 * (f1 - f0) * t * t / dur);
        Some(JointCommand {
            position: fb_pos,
            velocity: 0.0,
            torque: self.tau_amp * phase.sin(),
            kp: 0.0,
            kd: 0.0,
        })
    }

    /// Open-loop torque ramp then hold — used by both `noload-speed`
    /// (free shaft reaches terminal velocity) and `stall-torque` (blocked
    /// shaft, record peak torque). `kp = kd = 0`.
    fn torque_ramp(&self, t: f32, fb_pos: f32) -> Option<JointCommand> {
        let total = self.ramp_s + self.hold_s;
        if t >= total {
            return None;
        }
        let torque = if t < self.ramp_s {
            self.tau_amp * (t / self.ramp_s)
        } else {
            self.tau_amp
        };
        Some(JointCommand {
            position: fb_pos,
            velocity: 0.0,
            torque,
            kp: 0.0,
            kd: 0.0,
        })
    }
}

// ===========================================================================
// Helpers
// ===========================================================================

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

fn expand_tilde(path: &str) -> PathBuf {
    if let Some(rest) = path.strip_prefix("~/") {
        if let Ok(home) = std::env::var("HOME") {
            return PathBuf::from(home).join(rest);
        }
    } else if path == "~" {
        if let Ok(home) = std::env::var("HOME") {
            return PathBuf::from(home);
        }
    }
    PathBuf::from(path)
}

fn clamp_command(cmd: &mut JointCommand, lim: &EffLimits, specs: &RobstrideSpecs) {
    cmd.position = cmd.position.clamp(lim.pos_min, lim.pos_max);
    cmd.velocity = cmd.velocity.clamp(-lim.vel_max, lim.vel_max);
    cmd.torque = cmd.torque.clamp(-lim.tau_max, lim.tau_max);
    cmd.kp = cmd.kp.clamp(0.0, specs.kp_max);
    cmd.kd = cmd.kd.clamp(0.0, specs.kd_max);
}

/// Drain pending CAN frames into the motor's cached state.
fn pump_feedback(can: &CanInterface, motor: &mut RobstrideMotor) {
    for frame in can.drain() {
        if let Some(fb) = frame.parse_robstride() {
            motor.process_feedback(&fb);
        }
    }
}

/// Returns an abort reason if any safety condition is breached.
fn check_abort(motor: &RobstrideMotor, lim: &EffLimits, elapsed_s: f32, grace_s: f32) -> Option<String> {
    let st = &motor.state;

    if st.has_error {
        return Some(format!(
            "motor fault (code 0x{:X}: {})",
            st.error_code,
            motor.fault_description().unwrap_or_else(|| "unknown".into())
        ));
    }
    // Only trust feedback-derived limits once we have at least one frame.
    if st.last_update_ms != 0 {
        // 5% margin so quantization / overshoot doesn't false-trip.
        if st.position < lim.pos_min - 0.05 || st.position > lim.pos_max + 0.05 {
            return Some(format!("position {:.3} rad outside limits", st.position));
        }
        if st.velocity.abs() > lim.vel_max * 1.2 {
            return Some(format!("velocity {:.2} rad/s exceeds limit", st.velocity));
        }
        if st.torque.abs() > lim.tau_max * 1.2 {
            return Some(format!("torque {:.2} Nm exceeds limit", st.torque));
        }
        if st.temperature > lim.temp_max {
            return Some(format!("temperature {:.1} °C exceeds limit", st.temperature));
        }
        let age = now_ms().saturating_sub(st.last_update_ms) as f32;
        if age > lim.feedback_timeout_ms {
            return Some(format!("feedback stale ({age:.0} ms > timeout)"));
        }
    } else if elapsed_s > grace_s {
        return Some("no feedback received from motor".into());
    }
    None
}

/// Ramp gains/torque to zero, then disable the motor. Best-effort.
fn ramp_down_and_disable(can: &CanInterface, motor: &RobstrideMotor) {
    let zero = JointCommand::default();
    for _ in 0..40 {
        let _ = motor.send_command(can, &zero);
        std::thread::sleep(Duration::from_millis(5));
    }
    let _ = motor.disable(can);
}

fn confirm_cap_override(joint: &JointConfig, lim: &EffLimits) -> Result<()> {
    eprintln!(
        "\n!! CAP OVERRIDE: this maneuver will drive {} (model {}) up to\n   \
         vel_max={:.1} rad/s, tau_max={:.1} Nm — BEYOND the deploy safety envelope.\n   \
         Ensure the shaft is free (no-load) or solidly blocked (stall) as appropriate,\n   \
         and that bystanders are clear.\n   \
         Type 'yes' to proceed: ",
        joint.name,
        joint.model.as_str(),
        lim.vel_max,
        lim.tau_max,
    );
    let mut line = String::new();
    std::io::stdin()
        .read_line(&mut line)
        .context("read confirmation from stdin")?;
    if line.trim().eq_ignore_ascii_case("yes") {
        Ok(())
    } else {
        bail!("cap override not confirmed");
    }
}

// ===========================================================================
// Core run
// ===========================================================================

/// Outcome of a maneuver run, mapped to a process exit code by `main`.
enum RunOutcome {
    Completed,
    Interrupted,
    Aborted(String),
}

fn run(args: Args, stop: Arc<AtomicBool>) -> Result<RunOutcome> {
    // --- resolve joint -----------------------------------------------------
    let cfg = RobotConfig::from_yaml(&args.config)
        .with_context(|| format!("load robot config {:?}", args.config))?;

    let joint: JointConfig = match (&args.joint, args.motor_id, &args.bus) {
        (Some(name), _, _) => cfg
            .get_joint(name)
            .cloned()
            .ok_or_else(|| anyhow!("joint {name:?} not found in config"))?,
        (None, Some(id), Some(bus)) => cfg
            .joints
            .iter()
            .find(|j| j.can_id == id && &j.can_bus == bus)
            .cloned()
            .ok_or_else(|| anyhow!("no joint with motor_id {id} on bus {bus:?}"))?,
        _ => bail!("specify either --joint <name> or both --motor-id <id> and --bus <iface>"),
    };

    let maneuver = Maneuver::parse(&args.maneuver)?;
    let setup = Setup::parse(&args.setup)?;
    let specs = joint.model.specs();

    // Gate maneuvers by physical setup before touching the bus.
    if !setup.allows(maneuver) {
        let allowed: Vec<&str> = setup.allowed().iter().map(|m| m.as_str()).collect();
        let setups_allowing: Vec<&str> = [Setup::Bench, Setup::Hanging, Setup::Stand]
            .iter()
            .filter(|s| s.allows(maneuver))
            .map(|s| s.as_str())
            .collect();
        bail!(
            "maneuver {} is not allowed with --setup {} (permitted here: {}).\n\
             Run it on one of these setups instead: {}.",
            maneuver.as_str(),
            setup.as_str(),
            allowed.join(", "),
            setups_allowing.join(", "),
        );
    }

    // --- effective limits (optionally widened) -----------------------------
    let hl = &joint.hard_limits;
    let mut limits = EffLimits {
        pos_min: hl.pos_min,
        pos_max: hl.pos_max,
        vel_max: hl.vel_max,
        tau_max: hl.tau_max,
        temp_max: hl.temp_max,
        feedback_timeout_ms: hl.feedback_timeout_ms.max(50.0),
    };

    if maneuver.needs_cap_override() {
        if !args.allow_cap_override {
            bail!(
                "maneuver {} needs --allow-cap-override (it must exceed the deploy \
                 vel_max/tau_max to measure the true T-N curve)",
                maneuver.as_str()
            );
        }
        // Widen to the motor model's full electrical envelope.
        limits.vel_max = specs.velocity_max.abs().max(specs.velocity_min.abs());
        limits.tau_max = specs.torque_max.abs().max(specs.torque_min.abs());
        confirm_cap_override(&joint, &limits)?;
    }

    // --- open CSV ----------------------------------------------------------
    let out_dir = expand_tilde(&args.out);
    fs::create_dir_all(&out_dir)
        .with_context(|| format!("create output dir {}", out_dir.display()))?;
    let stamp = chrono::Local::now().format("%Y%m%d_%H%M%S");
    let csv_path = out_dir.join(format!(
        "sysid_{}_{}_{}.csv",
        joint.name,
        maneuver.as_str().replace('-', "_"),
        stamp
    ));
    let file = File::create(&csv_path)
        .with_context(|| format!("create CSV {}", csv_path.display()))?;
    let mut csv = BufWriter::new(file);
    writeln!(
        csv,
        "t_s,maneuver,joint,model,cmd_pos,cmd_vel,cmd_tau,cmd_kp,cmd_kd,fb_pos,fb_vel,fb_tau,fb_temp"
    )?;

    println!(
        "sysid: joint={} model={} bus={} id={} maneuver={} setup={}",
        joint.name,
        joint.model.as_str(),
        joint.can_bus,
        joint.can_id,
        maneuver.as_str(),
        setup.as_str(),
    );
    println!(
        "       limits: pos=[{:.3},{:.3}] vel_max={:.2} tau_max={:.2} temp_max={:.1}",
        limits.pos_min, limits.pos_max, limits.vel_max, limits.tau_max, limits.temp_max
    );
    println!("       logging to {}", csv_path.display());
    println!("       expect: {}", maneuver.expectation());

    // --- open CAN + motor --------------------------------------------------
    let can = CanInterface::open(&joint.can_bus)
        .with_context(|| format!("open CAN bus {:?}", joint.can_bus))?;
    let mut motor = RobstrideMotor::new(joint.can_id, joint.model);

    if args.set_zero {
        println!("       setting mechanical zero (motor disabled during zero)...");
        motor.disable(&can).ok();
        std::thread::sleep(Duration::from_millis(50));
        motor.set_zero(&can)?;
        std::thread::sleep(Duration::from_millis(100));
    }

    motor.enable(&can).context("enable motor")?;
    std::thread::sleep(Duration::from_millis(50));
    pump_feedback(&can, &mut motor);

    let excitation = Excitation::build(&args, maneuver, &limits, &joint);

    // --- control loop ------------------------------------------------------
    let period = Duration::from_secs_f32(1.0 / args.rate_hz.max(1.0));
    let grace_s = 0.5_f32;
    let start = Instant::now();

    let outcome = loop {
        let tick_start = Instant::now();
        let t = start.elapsed().as_secs_f32();

        if stop.load(Ordering::SeqCst) {
            break RunOutcome::Interrupted;
        }

        pump_feedback(&can, &mut motor);

        if let Some(reason) = check_abort(&motor, &limits, t, grace_s) {
            break RunOutcome::Aborted(reason);
        }

        match excitation.command(t, motor.state.position) {
            None => {
                break RunOutcome::Completed;
            }
            Some(mut cmd) => {
                clamp_command(&mut cmd, &limits, &specs);
                motor.send_command(&can, &cmd).context("send command")?;

                let st = &motor.state;
                writeln!(
                    csv,
                    "{:.6},{},{},{},{:.6},{:.6},{:.6},{:.4},{:.4},{:.6},{:.6},{:.6},{:.2}",
                    t,
                    maneuver.as_str(),
                    joint.name,
                    joint.model.as_str(),
                    cmd.position,
                    cmd.velocity,
                    cmd.torque,
                    cmd.kp,
                    cmd.kd,
                    st.position,
                    st.velocity,
                    st.torque,
                    st.temperature,
                )?;
            }
        }

        // Maintain the loop rate.
        if let Some(remaining) = period.checked_sub(tick_start.elapsed()) {
            std::thread::sleep(remaining);
        }
    };

    // --- shutdown ----------------------------------------------------------
    csv.flush().ok();
    ramp_down_and_disable(&can, &motor);

    match &outcome {
        RunOutcome::Completed => println!("sysid: maneuver complete."),
        RunOutcome::Interrupted => println!("sysid: interrupted — ramped down and disabled."),
        RunOutcome::Aborted(reason) => {
            eprintln!("sysid: SAFETY ABORT — {reason}. Ramped down and disabled.")
        }
    }
    Ok(outcome)
}

#[tokio::main]
async fn main() -> ExitCode {
    let args = Args::parse();

    let stop = Arc::new(AtomicBool::new(false));
    let stop_signal = stop.clone();
    tokio::spawn(async move {
        if tokio::signal::ctrl_c().await.is_ok() {
            eprintln!("\nsysid: Ctrl-C received — stopping...");
            stop_signal.store(true, Ordering::SeqCst);
        }
    });

    match tokio::task::spawn_blocking(move || run(args, stop)).await {
        Ok(Ok(RunOutcome::Completed)) | Ok(Ok(RunOutcome::Interrupted)) => ExitCode::SUCCESS,
        Ok(Ok(RunOutcome::Aborted(_))) => ExitCode::from(2),
        Ok(Err(e)) => {
            eprintln!("sysid: error: {e:#}");
            ExitCode::from(1)
        }
        Err(e) => {
            eprintln!("sysid: task panicked: {e}");
            ExitCode::from(1)
        }
    }
}
