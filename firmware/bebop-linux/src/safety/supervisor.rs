//! Motor supervisor: shared-state owner, TX clamp, watchdog, E-STOP latch.
//!
//! Threading model:
//!
//! - **One OS thread per CAN bus** drains the socket and dispatches feedback
//!   to per-motor entries. (Blocking SocketCAN reads play poorly with tokio,
//!   and we don't want a slow socket starving the executor.)
//! - **One async task** runs the supervisor "tick" at 100 Hz: re-sends
//!   hold-gain commands to every armed motor, runs the watchdog.
//! - **Server tasks** (WS handlers) call into the supervisor via the
//!   methods on this struct. All shared state is protected by per-motor
//!   `std::sync::Mutex`es to avoid one slow motor blocking the others.
//!
//! Drop on the supervisor disables every motor on every bus three times.

use crate::config::{JointCommand, RobotConfig, SafetyLimits};
use crate::mode::Mode;
use crate::powerboard;
use crate::safety::bus_pool::{read_can_state, BusPool};
use crate::safety::limits::{BreachReason, MotorRuntimeState, MotorSnapshot};
use crate::safety::power_monitor::{PowerBoardSnapshot, PowerMonitor};
use anyhow::{anyhow, Context, Result};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tracing::{debug, error, info, warn};

const TICK_RATE_HZ: u64 = 100;

/// Async event broadcast to interested subscribers (server, logger, etc.).
#[derive(Debug, Clone)]
pub enum SupervisorEvent {
    ModeChanged(Mode),
    EStopLatched(String),
    EStopReset,
    MotorArmed { joint: String },
    MotorDisarmed { joint: String },
}

pub struct Supervisor {
    cfg: Arc<RobotConfig>,
    bus_pool: Arc<BusPool>,
    /// One `Mutex<MotorRuntimeState>` per joint, in joint-index order.
    /// Per-motor locks let the RX thread for bus A update motor 0 without
    /// blocking the supervisor tick reading motor 1.
    motors: Vec<Arc<Mutex<MotorRuntimeState>>>,
    /// Lookup: `(can_interface, motor_id) -> motors[index]`.
    by_can_id: HashMap<(String, u8), usize>,
    /// Lookup: joint name -> motors[index]
    by_name: HashMap<String, usize>,
    /// Power-board monitor: `Some` iff `cfg.power` is set. Owns the
    /// cached battery / VBUS snapshot exposed through telemetry.
    power: Option<Arc<PowerMonitor>>,
    mode: Arc<AtomicU8>,
    estop: Arc<AtomicBool>,
    estop_reason: Arc<Mutex<Option<BreachReason>>>,
    events: tokio::sync::broadcast::Sender<SupervisorEvent>,
}

impl Supervisor {
    pub fn new(cfg: Arc<RobotConfig>, bus_pool: Arc<BusPool>) -> Self {
        let mut motors = Vec::with_capacity(cfg.joints.len());
        let mut by_can_id = HashMap::new();
        let mut by_name = HashMap::new();
        for (i, joint) in cfg.joints.iter().enumerate() {
            by_can_id.insert((joint.can_bus.clone(), joint.can_id), i);
            by_name.insert(joint.name.clone(), i);
            motors.push(Arc::new(Mutex::new(MotorRuntimeState::new(joint.clone()))));
        }
        let power = cfg
            .power
            .as_ref()
            .map(|p| Arc::new(PowerMonitor::new(p.clone())));
        let (events, _rx) = tokio::sync::broadcast::channel(64);
        Self {
            cfg,
            bus_pool,
            motors,
            by_can_id,
            by_name,
            power,
            mode: Arc::new(AtomicU8::new(Mode::Idle as u8)),
            estop: Arc::new(AtomicBool::new(false)),
            estop_reason: Arc::new(Mutex::new(None)),
            events,
        }
    }

    pub fn cfg(&self) -> &RobotConfig {
        &self.cfg
    }

    /// Shared handle to the power-board monitor (or `None` when no
    /// `power:` block was configured). Cloning the `Arc` is cheap and
    /// safe to do from background tasks.
    pub fn power_monitor(&self) -> Option<Arc<PowerMonitor>> {
        self.power.clone()
    }

    /// Most recent power-board snapshot, or `None` when no power board
    /// is configured. Used by the WS server to populate `Snapshot.power`.
    pub fn power_snapshot(&self) -> Option<PowerBoardSnapshot> {
        self.power.as_ref().map(|p| p.snapshot())
    }

    pub fn subscribe(&self) -> tokio::sync::broadcast::Receiver<SupervisorEvent> {
        self.events.subscribe()
    }

    pub fn mode(&self) -> Mode {
        Mode::from_u8(self.mode.load(Ordering::SeqCst)).unwrap_or(Mode::Idle)
    }

    pub fn estop_active(&self) -> bool {
        self.estop.load(Ordering::SeqCst)
    }

    pub fn estop_reason_human(&self) -> Option<String> {
        self.estop_reason
            .lock()
            .ok()
            .and_then(|g| g.as_ref().map(|r| r.human()))
    }

    // ---------------------------------------------------------------- snapshots

    pub fn snapshot_motors(&self) -> Vec<MotorSnapshot> {
        let now = Instant::now();
        self.motors
            .iter()
            .map(|m| {
                m.lock()
                    .map(|g| g.snapshot(now))
                    .unwrap_or_else(|p| p.into_inner().snapshot(now))
            })
            .collect()
    }

    // ---------------------------------------------------------------- mode transitions

    pub fn set_mode(&self, requested: Mode) -> Result<()> {
        if self.estop_active() && requested != Mode::Idle {
            return Err(anyhow!(
                "cannot change mode while E-STOP is latched; reset first"
            ));
        }
        let prev = self.mode();
        if prev == requested {
            return Ok(());
        }
        // Always disarm motors when leaving a mode that might have armed them.
        if matches!(prev, Mode::DialIn | Mode::RunPolicy) {
            self.disarm_all_internal(BreachReason::Operator(format!(
                "mode change {:?} -> {:?}",
                prev, requested
            )));
        }
        self.mode.store(requested as u8, Ordering::SeqCst);
        info!(?prev, ?requested, "mode changed");
        let _ = self.events.send(SupervisorEvent::ModeChanged(requested));
        Ok(())
    }

    // ---------------------------------------------------------------- E-STOP

    /// Latch E-STOP. Idempotent; subsequent calls don't replace the reason.
    pub fn trigger_estop(&self, reason: BreachReason) {
        if self
            .estop
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return;
        }
        let human = reason.human();
        if let Ok(mut g) = self.estop_reason.lock() {
            *g = Some(reason);
        }
        warn!(reason = %human, "E-STOP latched");
        // Best-effort: send disable to every motor on every bus, multiple
        // times in case the bus is dropping frames.
        for _ in 0..3 {
            self.disable_all_no_state();
            std::thread::sleep(Duration::from_millis(5));
        }
        // Mark every entry as disarmed locally.
        for entry in &self.motors {
            if let Ok(mut g) = entry.lock() {
                g.armed = false;
            }
        }
        let _ = self.events.send(SupervisorEvent::EStopLatched(human));
    }

    pub fn reset_estop(&self) -> bool {
        if !self.estop_active() {
            return false;
        }
        self.estop.store(false, Ordering::SeqCst);
        if let Ok(mut g) = self.estop_reason.lock() {
            *g = None;
        }
        info!("E-STOP latch cleared by operator");
        let _ = self.events.send(SupervisorEvent::EStopReset);
        true
    }

    // ---------------------------------------------------------------- arm / disarm

    pub fn arm(&self, joint: &str) -> Result<()> {
        if self.estop_active() {
            return Err(anyhow!("cannot arm while E-STOP is latched"));
        }
        if self.mode() != Mode::DialIn {
            return Err(anyhow!(
                "arming is only allowed in DialIn mode (currently {:?})",
                self.mode()
            ));
        }
        let idx = *self
            .by_name
            .get(joint)
            .ok_or_else(|| anyhow!("unknown joint {joint:?}"))?;
        let entry = self.motors[idx].clone();
        let cfg = &self.cfg.joints[idx];

        // Refuse to arm on a bus that's not ERROR-ACTIVE. The motor on
        // an ERROR-PASSIVE bus is presumed unpowered / unwired; sending
        // Enable would be wasted TX (and might push the controller toward
        // BUS-OFF).
        if !self.bus_pool.is_healthy(&cfg.can_bus) {
            return Err(anyhow!(
                "refusing to arm {joint}: bus {} is not healthy. \
                 Verify the motor is powered + wired and the link is \
                 ERROR-ACTIVE (`ip -details link show {}`).",
                cfg.can_bus,
                cfg.can_bus
            ));
        }

        // Refuse to arm if the joint is currently outside its hard limits.
        let pos_now = {
            let g = entry.lock().unwrap();
            g.motor.state.position
        };
        let hl = cfg.hard_limits;
        if pos_now < hl.pos_min - 1e-3 || pos_now > hl.pos_max + 1e-3 {
            return Err(anyhow!(
                "refusing to arm {joint}: pos {pos_now:+.3} outside [{:+.3}, {:+.3}]; \
                 move into range manually first",
                hl.pos_min,
                hl.pos_max
            ));
        }

        let can = self
            .bus_pool
            .get(&cfg.can_bus)
            .ok_or_else(|| anyhow!("no bus pool entry for {}", &cfg.can_bus))?
            .clone();

        // Send Enable + lock the slew tracker to current pose before any
        // hold-cycle TX so the first command can't request a far-away target.
        // Wrap the I/O call with diagnostics so the operator-side error
        // chain reads "<joint>: failed to enable motor 31 on can1 (bus state
        // ERROR-PASSIVE): No buffer space available" rather than the bare
        // socket error. The bus state is re-read here, *after* the TX
        // attempt, so it reflects the moment the kernel rejected us — a
        // bus that was ERROR-ACTIVE during pre-flight can degrade
        // mid-arm if there's no peer to ACK.
        {
            let g = entry.lock().unwrap();
            g.motor.enable(&can).with_context(|| {
                let bus_state =
                    read_can_state(&cfg.can_bus).unwrap_or_else(|| "?".into());
                format!(
                    "{}: failed to enable motor id {} on {} (bus state {}); \
                     check that the motor is powered, the CAN cable is \
                     plugged in, and the bus has a working terminator",
                    joint, cfg.can_id, cfg.can_bus, bus_state
                )
            })?;
        }
        std::thread::sleep(Duration::from_millis(20));
        {
            let mut g = entry.lock().unwrap();
            g.last_target_pos = g.motor.state.position;
            g.armed = true;
        }
        info!(joint, "armed");
        let _ = self.events.send(SupervisorEvent::MotorArmed {
            joint: joint.to_string(),
        });
        Ok(())
    }

    pub fn disarm(&self, joint: &str) -> Result<()> {
        let idx = *self
            .by_name
            .get(joint)
            .ok_or_else(|| anyhow!("unknown joint {joint:?}"))?;
        let entry = &self.motors[idx];
        let cfg = self.cfg.joints[idx].clone();
        let can = match self.bus_pool.get(&cfg.can_bus) {
            Some(c) => c.clone(),
            None => return Err(anyhow!("no bus pool entry for {}", cfg.can_bus)),
        };
        // Best-effort disable; ignore CAN errors (motor may already be down).
        if let Ok(g) = entry.lock() {
            let _ = g.motor.disable(&can);
        }
        if let Ok(mut g) = entry.lock() {
            g.armed = false;
        }
        info!(joint, "disarmed");
        let _ = self.events.send(SupervisorEvent::MotorDisarmed {
            joint: joint.to_string(),
        });
        Ok(())
    }

    /// Set the hold-target position for one armed motor. The supervisor's
    /// 100 Hz `tick_dial_in_hold` will pick the new target up on its next
    /// pass and slew toward it. `safe_send_ctrl` already clamps to
    /// hard_limits, so a target outside the YAML envelope is silently
    /// pulled back to the limit (the operator sees this via the
    /// `target_position` field in the next telemetry frame).
    ///
    /// The dial-in tool should drive armed joints toward the current
    /// envelope edges to confirm the motor tracks; once a wider envelope
    /// is wanted, the operator edits the YAML and restarts.
    pub fn set_target_position(&self, joint: &str, position: f32) -> Result<()> {
        if self.estop_active() {
            return Err(anyhow!("cannot set target while E-STOP is latched"));
        }
        if self.mode() != Mode::DialIn {
            return Err(anyhow!(
                "set_target_position is only allowed in DialIn mode (currently {:?})",
                self.mode()
            ));
        }
        let idx = *self
            .by_name
            .get(joint)
            .ok_or_else(|| anyhow!("unknown joint {joint:?}"))?;
        let h = self.cfg.joints[idx].hard_limits;
        // Pre-clamp to hard limits so the snapshot we'll write is honest.
        // The slew limiter in `safe_send_ctrl` will further cap the per-tick
        // step on the next 100 Hz pass.
        let clamped = position.max(h.pos_min).min(h.pos_max);
        let entry = &self.motors[idx];
        let g = entry
            .lock()
            .map_err(|p| anyhow!("motor mutex poisoned: {p}"))?;
        if !g.armed {
            return Err(anyhow!(
                "cannot set target on disarmed joint {joint}; arm it first"
            ));
        }
        // Drop the read lock and reacquire mutably to update the target.
        drop(g);
        let mut g = entry
            .lock()
            .map_err(|p| anyhow!("motor mutex poisoned: {p}"))?;
        g.last_target_pos = clamped;
        Ok(())
    }

    /// Re-zero a single motor's mechanical origin to its current physical
    /// position. Drives the motor through the safe sequence:
    ///
    ///   1. Send Disable (CMD 0x04) — even though the joint is already
    ///      disarmed at the supervisor level, the motor on the bus may
    ///      have been left in an enabled state by a previous arm-fail
    ///      mid-sequence, or by a peer (CAN is shared). Reference
    ///      Robstride driver implementations also belt-and-suspender
    ///      this; the cost is one CAN frame.
    ///   2. Sleep ~20 ms so the motor processes the Disable before the
    ///      SET_ZERO frame lands. SET_ZERO is most reliably honoured
    ///      from the stopped state.
    ///   3. Send SET_ZERO (CMD 0x06) with `data[0] = 1`. The motor
    ///      treats its current physical position as the new 0 rad
    ///      reference and commits the origin to flash.
    ///   4. Sleep ~20 ms so the next feedback frame is post-zero before
    ///      we overwrite our cached state.
    ///   5. Zero out our cached `position`, `velocity`, and
    ///      `last_target_pos` so a follow-on `arm()` reads "in range"
    ///      from the supervisor's perspective without having to wait for
    ///      the rx thread to deliver the post-zero feedback frame.
    ///
    /// We deliberately do NOT auto re-enable the motor (some reference
    /// implementations do). Re-enabling on the back of a re-zero would
    /// arm the motor without an explicit operator gesture; the operator
    /// must click the arm toggle again, which gives them a chance to
    /// see the new ~0 rad position in telemetry first.
    ///
    /// Preconditions (rejected with a descriptive error otherwise):
    ///   - No E-STOP latched.
    ///   - Joint exists.
    ///   - Joint is currently *disarmed*. SET_ZERO mid-hold would shift
    ///     the position reference under a live PD loop and wrench the
    ///     motor toward the new "zero" minus our cached `last_target_pos`.
    ///   - The joint's CAN bus is ERROR-ACTIVE. On a degraded bus the
    ///     SET_ZERO frame would be silently dropped and the operator
    ///     would falsely believe re-zero succeeded.
    pub fn set_mechanical_zero(&self, joint: &str) -> Result<()> {
        if self.estop_active() {
            return Err(anyhow!(
                "cannot set mechanical zero while E-STOP is latched; reset first"
            ));
        }
        let idx = *self
            .by_name
            .get(joint)
            .ok_or_else(|| anyhow!("unknown joint {joint:?}"))?;
        let entry = self.motors[idx].clone();
        let cfg = self.cfg.joints[idx].clone();

        let armed = entry
            .lock()
            .map(|g| g.armed)
            .map_err(|p| anyhow!("motor mutex poisoned: {p}"))?;
        if armed {
            return Err(anyhow!(
                "refusing to set mechanical zero on armed joint {joint}; \
                 disarm it first"
            ));
        }

        if !self.bus_pool.is_healthy(&cfg.can_bus) {
            return Err(anyhow!(
                "refusing to set mechanical zero on {joint}: bus {} is not \
                 healthy. Verify the motor is powered + wired and the link \
                 is ERROR-ACTIVE (`ip -details link show {}`).",
                cfg.can_bus,
                cfg.can_bus
            ));
        }

        let can = self
            .bus_pool
            .get(&cfg.can_bus)
            .ok_or_else(|| anyhow!("no bus pool entry for {}", &cfg.can_bus))?
            .clone();

        // Step 1: explicit Disable. Mirrors the reference Robstride
        // driver. Cheap insurance against the motor still being in
        // enabled state on the bus (e.g. a mid-arm failure that returned
        // an error before disarm could be called).
        {
            let g = entry
                .lock()
                .map_err(|p| anyhow!("motor mutex poisoned: {p}"))?;
            g.motor.disable(&can).with_context(|| {
                let bus_state =
                    read_can_state(&cfg.can_bus).unwrap_or_else(|| "?".into());
                format!(
                    "{}: failed to send Disable before SET_ZERO to motor id {} on {} (bus state {})",
                    joint, cfg.can_id, cfg.can_bus, bus_state
                )
            })?;
        }
        // Step 2: let the motor process the Disable.
        std::thread::sleep(Duration::from_millis(20));

        // Step 3: SET_ZERO.
        {
            let g = entry
                .lock()
                .map_err(|p| anyhow!("motor mutex poisoned: {p}"))?;
            g.motor.set_zero(&can).with_context(|| {
                let bus_state =
                    read_can_state(&cfg.can_bus).unwrap_or_else(|| "?".into());
                format!(
                    "{}: failed to send SET_ZERO to motor id {} on {} (bus state {})",
                    joint, cfg.can_id, cfg.can_bus, bus_state
                )
            })?;
        }
        // Step 4: let the motor commit + start reporting against the
        // new origin before we touch cached state.
        std::thread::sleep(Duration::from_millis(20));

        // Step 5: zero cached state. The rx thread will overwrite this
        // with authoritative post-zero feedback shortly; we set it
        // optimistically so an immediate re-arm sees "in range".
        if let Ok(mut g) = entry.lock() {
            g.motor.state.position = 0.0;
            g.motor.state.velocity = 0.0;
            g.last_target_pos = 0.0;
        }
        info!(joint, "mechanical zero set");
        Ok(())
    }

    pub fn arm_all(&self) -> Vec<(String, anyhow::Error)> {
        let mut errors = Vec::new();
        let names: Vec<String> = self.cfg.joints.iter().map(|j| j.name.clone()).collect();
        for name in names {
            if let Err(e) = self.arm(&name) {
                errors.push((name, e));
            }
        }
        errors
    }

    pub fn disarm_all(&self) -> Vec<(String, anyhow::Error)> {
        let mut errors = Vec::new();
        let names: Vec<String> = self.cfg.joints.iter().map(|j| j.name.clone()).collect();
        for name in names {
            if let Err(e) = self.disarm(&name) {
                errors.push((name, e));
            }
        }
        errors
    }

    /// Internal: disarm all without recording the per-joint events. Used
    /// during E-STOP and mode transitions where we already have a top-level
    /// reason.
    fn disarm_all_internal(&self, _why: BreachReason) {
        for idx in 0..self.motors.len() {
            let cfg = &self.cfg.joints[idx];
            if let Some(can) = self.bus_pool.get(&cfg.can_bus) {
                if let Ok(g) = self.motors[idx].lock() {
                    let _ = g.motor.disable(can);
                }
            }
            if let Ok(mut g) = self.motors[idx].lock() {
                g.armed = false;
            }
        }
    }

    /// Send Disable to every motor on every known bus, ignoring per-motor
    /// state (used in E-STOP and Drop). Doesn't take any locks beyond what
    /// CanInterface needs.
    fn disable_all_no_state(&self) {
        for joint in &self.cfg.joints {
            if let Some(can) = self.bus_pool.get(&joint.can_bus) {
                let motor = crate::robstride::RobstrideMotor::new(joint.can_id, joint.model);
                let _ = motor.disable(can);
            }
        }
    }

    // ---------------------------------------------------------------- safe TX

    /// Send a clamp+slew-limited control command to a single motor.
    ///
    /// All TX traffic must go through here. `target_pos`, `velocity_ff`,
    /// and `torque_ff` are clamped against the joint's hard limits *before*
    /// the slew limiter; the slew limiter then caps the per-tick step
    /// against `last_target_pos`.
    pub fn safe_send_ctrl(
        &self,
        idx: usize,
        target_pos: f32,
        kp: f32,
        kd: f32,
        velocity_ff: f32,
        torque_ff: f32,
    ) -> Result<()> {
        if self.estop_active() {
            return Ok(());
        }
        let cfg = &self.cfg.joints[idx];
        let h = cfg.hard_limits;
        let s = cfg.slew;

        let entry = &self.motors[idx];
        let (clamped, vel_ff, tau_ff) = {
            let mut g = entry.lock().unwrap();
            let mut clamped = clamp(target_pos, h.pos_min, h.pos_max);
            clamped = clamp(
                clamped,
                g.last_target_pos - s.max_pos_step_per_tick,
                g.last_target_pos + s.max_pos_step_per_tick,
            );
            g.last_target_pos = clamped;
            let vel_ff = clamp(velocity_ff, -h.vel_max, h.vel_max);
            let tau_ff = clamp(torque_ff, -h.tau_max, h.tau_max);
            (clamped, vel_ff, tau_ff)
        };

        let can = match self.bus_pool.get(&cfg.can_bus) {
            Some(c) => c,
            None => return Err(anyhow!("no bus pool entry for {}", cfg.can_bus)),
        };

        let cmd = JointCommand {
            position: clamped,
            velocity: vel_ff,
            torque: tau_ff,
            kp,
            kd,
        };
        let send_result = {
            let g = entry.lock().unwrap();
            g.motor.send_command(can, &cmd)
        };
        if let Err(e) = send_result {
            self.trigger_estop(BreachReason::BusError {
                can_interface: cfg.can_bus.clone(),
                message: e.to_string(),
            });
            return Err(e);
        }
        Ok(())
    }

    // ---------------------------------------------------------------- RX path

    /// Called by the bus RX thread for each parsed feedback frame.
    /// Updates the motor's cached state and runs limit checks.
    pub fn on_feedback(&self, can_iface: &str, fb: &crate::can_interface::RobstrideFeedback) {
        let idx = match self.by_can_id.get(&(can_iface.to_string(), fb.motor_id)) {
            Some(i) => *i,
            None => return, // unknown motor — could be a different bus user
        };
        let now = Instant::now();
        {
            let mut g = match self.motors[idx].lock() {
                Ok(g) => g,
                Err(p) => p.into_inner(),
            };
            g.motor.process_feedback(fb);
            g.last_rx = Some(now);
        }
        // Run limit checks while we hold our own snapshot.
        self.check_feedback(idx, fb);
    }

    fn check_feedback(&self, idx: usize, fb: &crate::can_interface::RobstrideFeedback) {
        let cfg = &self.cfg.joints[idx];
        let h = cfg.hard_limits;
        if fb.fault_bits != 0 {
            self.trigger_estop(BreachReason::MotorFault {
                joint: cfg.name.clone(),
                bits: fb.fault_bits,
                description: describe_fault(fb.fault_bits),
            });
            return;
        }
        if let Some(reason) = check_pos(&cfg.name, fb.position, h) {
            self.trigger_estop(reason);
            return;
        }
        if let Some(reason) = check_vel(&cfg.name, fb.velocity, h) {
            self.trigger_estop(reason);
            return;
        }
        if let Some(reason) = check_tau(&cfg.name, fb.torque, h) {
            self.trigger_estop(reason);
            return;
        }
        if let Some(reason) = check_temp(&cfg.name, fb.temperature, h) {
            self.trigger_estop(reason);
        }
    }

    /// Watchdog: latch E-STOP if any armed motor hasn't received feedback
    /// within its timeout. Called from the supervisor tick.
    pub fn run_watchdog(&self) {
        if self.estop_active() {
            return;
        }
        if self.mode() == Mode::Idle {
            return; // motors are off; no traffic expected
        }
        let now = Instant::now();
        for idx in 0..self.motors.len() {
            let (armed, last_rx, joint, timeout) = {
                let g = match self.motors[idx].lock() {
                    Ok(g) => g,
                    Err(p) => p.into_inner(),
                };
                (
                    g.armed,
                    g.last_rx,
                    g.joint_cfg.name.clone(),
                    g.joint_cfg.hard_limits.feedback_timeout_ms,
                )
            };
            if !armed {
                continue;
            }
            let Some(t) = last_rx else { continue };
            let elapsed_ms = now.duration_since(t).as_secs_f32() * 1000.0;
            if elapsed_ms > timeout {
                self.trigger_estop(BreachReason::FeedbackWatchdog {
                    joint,
                    elapsed_ms,
                    timeout_ms: timeout,
                });
                return;
            }
        }
    }

    // ---------------------------------------------------------------- 100 Hz tick

    /// Periodic hold-gain TX for every armed motor in DialIn mode. Keeps
    /// the motor's own watchdog alive and re-asserts our slew-limited
    /// setpoint each cycle.
    pub fn tick_dial_in_hold(&self) {
        if self.estop_active() || self.mode() != Mode::DialIn {
            return;
        }
        for idx in 0..self.motors.len() {
            let (armed, target, kp, kd) = {
                let g = match self.motors[idx].lock() {
                    Ok(g) => g,
                    Err(p) => p.into_inner(),
                };
                (
                    g.armed,
                    g.last_target_pos,
                    g.joint_cfg.hold_gains.kp,
                    g.joint_cfg.hold_gains.kd,
                )
            };
            if !armed {
                continue;
            }
            if let Err(e) = self.safe_send_ctrl(idx, target, kp, kd, 0.0, 0.0) {
                debug!(joint = %self.cfg.joints[idx].name, error = %e, "hold TX failed");
            }
        }
    }
}

impl Drop for Supervisor {
    fn drop(&mut self) {
        // Always-disable on exit. Don't lock motors first — this runs
        // from arbitrary contexts (panic, ctrl-c). Issue raw disable frames
        // multiple times in case the bus drops them.
        warn!("supervisor dropping; disabling all motors");
        for _ in 0..3 {
            self.disable_all_no_state();
            std::thread::sleep(Duration::from_millis(5));
        }
    }
}

// ---------------------------------------------------------------------------
// Limit check helpers
// ---------------------------------------------------------------------------

fn clamp(v: f32, lo: f32, hi: f32) -> f32 {
    v.max(lo).min(hi)
}

fn check_pos(joint: &str, value: f32, h: SafetyLimits) -> Option<BreachReason> {
    if value < h.pos_min - 1e-3 || value > h.pos_max + 1e-3 {
        Some(BreachReason::PositionOutOfRange {
            joint: joint.to_string(),
            value,
            min: h.pos_min,
            max: h.pos_max,
        })
    } else {
        None
    }
}

fn check_vel(joint: &str, value: f32, h: SafetyLimits) -> Option<BreachReason> {
    if value.abs() > h.vel_max {
        Some(BreachReason::VelocityExceeded {
            joint: joint.to_string(),
            value,
            max: h.vel_max,
        })
    } else {
        None
    }
}

fn check_tau(joint: &str, value: f32, h: SafetyLimits) -> Option<BreachReason> {
    if value.abs() > h.tau_max {
        Some(BreachReason::TorqueExceeded {
            joint: joint.to_string(),
            value,
            max: h.tau_max,
        })
    } else {
        None
    }
}

fn check_temp(joint: &str, value: f32, h: SafetyLimits) -> Option<BreachReason> {
    if value > h.temp_max {
        Some(BreachReason::TemperatureExceeded {
            joint: joint.to_string(),
            value,
            max: h.temp_max,
        })
    } else {
        None
    }
}

fn describe_fault(bits: u8) -> String {
    let mut flags = Vec::new();
    if bits & 0x01 != 0 {
        flags.push("undervoltage");
    }
    if bits & 0x02 != 0 {
        flags.push("overcurrent");
    }
    if bits & 0x04 != 0 {
        flags.push("overtemperature");
    }
    if bits & 0x08 != 0 {
        flags.push("encoder_fault");
    }
    if bits & 0x10 != 0 {
        flags.push("gridlock_overload");
    }
    if bits & 0x20 != 0 {
        flags.push("uncalibrated");
    }
    if flags.is_empty() {
        format!("0x{bits:02X}")
    } else {
        flags.join(",")
    }
}

/// Spawn one OS thread per CAN bus that drains feedback frames and routes
/// them to the supervisor. Returns join handles so the caller can wait
/// for them on shutdown (although in practice the supervisor's Drop will
/// race them on exit; the threads are daemon-style).
pub fn spawn_rx_threads(
    sup: Arc<Supervisor>,
    bus_pool: Arc<BusPool>,
    shutdown: Arc<AtomicBool>,
) -> Vec<std::thread::JoinHandle<()>> {
    let mut handles = Vec::new();
    // Pre-bind the power monitor (if any) to its bus so we don't take
    // the supervisor lock on every frame just to check `Option::is_some`.
    let power_iface = sup
        .power_monitor()
        .as_ref()
        .map(|m| (m.cfg.can_interface.clone(), m.board.power_id));

    for (iface, can) in bus_pool.iter() {
        let iface = iface.clone();
        let can = can.clone();
        let sup = sup.clone();
        let shutdown = shutdown.clone();
        let power_for_this_bus = power_iface
            .as_ref()
            .filter(|(p_iface, _)| *p_iface == iface)
            .and_then(|_| sup.power_monitor());
        let handle = std::thread::Builder::new()
            .name(format!("rx-{}", iface))
            .spawn(move || {
                info!(
                    can_interface = %iface,
                    powerboard_attached = power_for_this_bus.is_some(),
                    "RX thread started",
                );
                while !shutdown.load(Ordering::SeqCst) {
                    match can.try_receive() {
                        Ok(Some(frame)) => {
                            // Try to parse the frame as a Robstride motor
                            // feedback first (most common case on motor
                            // buses); if that doesn't fit, see if it's a
                            // power-board status response on the bus we
                            // configured for one.
                            if let Some(fb) = frame.parse_robstride() {
                                if fb.cmd_type == 0x02 {
                                    sup.on_feedback(&iface, &fb);
                                    continue;
                                }
                            }
                            if let Some(monitor) = power_for_this_bus.as_ref() {
                                if let Some(pf) =
                                    powerboard::parse_frame(&frame, monitor.board.power_id)
                                {
                                    monitor.on_frame(pf);
                                }
                            }
                        }
                        Ok(None) => {
                            // No frame; small sleep to avoid spinning.
                            std::thread::sleep(Duration::from_micros(200));
                        }
                        Err(e) => {
                            error!(can_interface = %iface, error = %e, "RX error");
                            std::thread::sleep(Duration::from_millis(10));
                        }
                    }
                }
                info!(can_interface = %iface, "RX thread exiting");
            })
            .expect("spawn rx thread");
        handles.push(handle);
    }
    handles
}
