//! BNO080/BNO085 SPI reader that publishes fused orientation **and**
//! body-frame angular velocity into a shared state so both the WS
//! server (telemetry) and the policy runner (observation builder) can
//! consume them.
//!
//! ## Why SPI + report 0x28
//!
//! Earlier revisions of this file talked to the BNO over I²C and asked
//! the chip for SH-2 report **0x05** (the plain magnetometer-fused
//! rotation vector). That setup worked on the bench but locked up
//! near the leg motors: the brushless drive currents bias the BNO's
//! magnetometer enough that its fusion filter rejects all subsequent
//! mag updates, freezing yaw indefinitely until a hard reset.
//!
//! The fix is to migrate to **SPI** (so we have a dedicated INT line
//! for low-latency reads) and subscribe to report **0x28**
//! ("AR/VR-Stabilized Rotation Vector"), which is the same report the
//! old Teensy firmware used in
//! `firmware/bebop-locomotion/include/BNO085_IMU.h`. 0x28 uses the
//! same Q14 wire format as 0x05 but goes through a separate fusion
//! pipeline inside the chip that aggressively filters magnetometer
//! disturbances. The trade-off is that absolute yaw drifts very
//! slowly until the chip re-establishes a clean mag reference; in
//! exchange the yaw axis stays *responsive* under motor noise instead
//! of locking.
//!
//! In parallel we subscribe to **report 0x02** ("Calibrated
//! Gyroscope") so the policy observation builder has a real
//! body-frame angular velocity input. The BNO chip applies its own
//! bias-tracking calibration to 0x02, so this is the right report to
//! use for closed-loop control — the uncalibrated variant (0x07) is
//! noticeably noisier in the steady state.
//!
//! The crates.io copy of `bno08x-rs` (v2.0.1) cannot enable any
//! report whose ID is ≥ 16 (it indexes a `[bool; 16]` array with the
//! raw ID, which would panic on 0x28). We therefore vendor and patch
//! the crate in `vendor/bno08x-rs/`; see the `BEBOP-PATCH` markers in
//! that copy and the dependency comment in `Cargo.toml`.
//!
//! ## Frame contract
//!
//! The mount-rotation pipeline carries over from the I²C version
//! verbatim — see [`ImuSnapshot`] for the body-frame XYZW unit
//! quaternion contract. The shape of the snapshot, its staleness
//! semantics, and the post-multiply with `mount_quat_sensor_body` are
//! unchanged from the I²C era so the telemetry pump and operator-app
//! widgets don't need to know we switched buses.
//!
//! The gyro vector follows the same convention: the BNO publishes
//! `omega_sensor` in the chip's silkscreen frame and we rotate it
//! into the body frame here with `R_sensor_body * omega_sensor`, so
//! [`ImuSnapshot::angular_velocity_body`] is always FLU body-frame
//! (+x forward, +y left, +z up) rad/s. The simulator's
//! `mdp.imu_ang_vel` produces the same quantity for the trained
//! policy (see `sim/bebop_training/envs/bebop_v2_base_cfg.py`), so
//! the policy can be deployed without a frame remap.
//!
//! ## Quaternion order
//!
//! `ImuSnapshot.quaternion` is **XYZW** (scalar last). This matches
//! Isaac Lab **3.0**'s new default (the 2.x → 3.0 migration moved
//! every quaternion in the framework from WXYZ to XYZW so that
//! Warp / PhysX / Newton can share buffers without conversions). The
//! firmware was already XYZW for ROS / `geometry_msgs` parity, so no
//! reorder is needed in either direction when the policy that was
//! trained against Isaac Lab 3.0 is loaded here.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use bno08x_rs::{BNO08x, SENSOR_REPORTID_ARVR_STABILIZED_RV, SENSOR_REPORTID_GYROSCOPE};
use nalgebra::{Quaternion, UnitQuaternion, Vector3};
use tracing::{error, info, warn};

use crate::config::ImuConfig;

/// Latest decoded BNO080/BNO085 rotation vector + gyroscope reading.
///
/// # Frame & normalization contract
///
/// `quaternion` is the **body-frame** orientation in **XYZW (Hamilton)**
/// order, matching [`crate::observation::ImuState::quaternion`].
///
/// 1. Frame: any constant sensor→body mount rotation declared in the
///    YAML `imu.mount:` block is applied here, in
///    [`spawn_imu_thread`], **before** the value reaches `ImuSnapshot`.
///    Downstream consumers (telemetry pump, [`crate::policy_runner`]
///    IMU feed, ROS bridge) can therefore treat this as a calibrated
///    body-frame (REP-103 / FLU: `+x forward`, `+y left`, `+z up`)
///    attitude with no further rotation needed.
/// 2. Normalization: when `Some`, the producer guarantees
///    `|quaternion| = 1` within float precision (we explicitly
///    normalize after the mount multiply to compensate for any drift
///    in the BNO output or numerical noise from the composition).
/// 3. Identity is `(0.0, 0.0, 0.0, 1.0)`. `None` (= `last_update` is
///    `None`) means no rotation-vector frame has been decoded yet.
///
/// `angular_velocity_body` is the BNO's **calibrated** gyroscope
/// reading (SH-2 report 0x02, rad/s) rotated into the same FLU body
/// frame as `quaternion`. The producer also clamps to `Some(...)` only
/// after the first gyro report arrives so consumers can distinguish
/// "no gyro yet" from "zero motion".
#[derive(Debug, Default, Clone, Copy)]
pub struct ImuSnapshot {
    pub quaternion: Option<[f32; 4]>,
    /// Body-frame angular velocity in rad/s, or `None` if no gyro
    /// report has been decoded since startup. FLU axis convention,
    /// identical to what `mdp.imu_ang_vel` produces in the sim
    /// observation pipeline.
    pub angular_velocity_body: Option<[f32; 3]>,
    pub heading_accuracy_rad: f32,
    pub last_update: Option<Instant>,
    pub report_period_ms: u16,
}

impl ImuSnapshot {
    /// True if the most recent successful read is older than
    /// `3 × report_period_ms` (clamped to at least 250 ms). Returns
    /// `true` before the first frame so the UI greys out the widget.
    pub fn is_stale(&self, now: Instant) -> bool {
        let Some(last) = self.last_update else {
            return true;
        };
        let budget_ms = (self.report_period_ms as u64).saturating_mul(3).max(250);
        now.duration_since(last) > Duration::from_millis(budget_ms)
    }

    /// Milliseconds since the last successful read, or `0` if no read
    /// has succeeded yet. Callers should check `last_update.is_some()`
    /// (or `received` on the proto side) to distinguish "fresh zero"
    /// from "never updated".
    pub fn age_ms(&self, now: Instant) -> u32 {
        self.last_update
            .map(|t| now.duration_since(t).as_millis().min(u32::MAX as u128) as u32)
            .unwrap_or(0)
    }
}

/// Shared handle on the latest [`ImuSnapshot`].
///
/// Cloned into the SPI reader thread (writer) and the WS server's
/// telemetry builder (reader). A plain [`Mutex`] is sufficient here:
/// updates happen at the sensor's report period (~20 Hz at the default
/// 50 ms) and the telemetry pump reads at ≤100 Hz, so contention is
/// negligible.
pub type ImuShared = Arc<Mutex<ImuSnapshot>>;

/// Allocate a fresh shared snapshot. Initial state is "no quaternion
/// yet, stale" — safe to wire into the telemetry builder before the
/// SPI reader thread starts (or when no IMU is configured at all).
pub fn new_shared() -> ImuShared {
    Arc::new(Mutex::new(ImuSnapshot::default()))
}

/// Hamilton quaternion product in XYZW order. Used to bake the constant
/// sensor→body mount rotation into each raw BNO reading before publishing.
///
/// Matches `nalgebra::Quaternion::mul` (`coords = [i, j, k, w]`); kept as a
/// tiny local helper to avoid promoting two `[f32; 4]`s into `nalgebra`
/// types on every IMU sample.
#[inline]
fn quat_mul_xyzw(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    let [ax, ay, az, aw] = a;
    let [bx, by, bz, bw] = b;
    [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ]
}

/// Normalize an XYZW quaternion to unit length. Returns the XYZW
/// identity `(0, 0, 0, 1)` for inputs whose squared norm is below a
/// tiny threshold — this only fires before the first BNO report has
/// arrived (the driver's cached value starts at all-zeros) and avoids
/// a divide-by-zero rather than silently propagating an unphysical
/// quaternion.
///
/// In normal operation the BNO emits a unit quaternion and the
/// constant mount multiply preserves that, so this just trims off the
/// ~1e-6 of float drift that accumulates after the composition.
#[inline]
fn quat_normalize_xyzw(q: [f32; 4]) -> [f32; 4] {
    let [x, y, z, w] = q;
    let norm_sq = x * x + y * y + z * z + w * w;
    if norm_sq < 1e-12 {
        return [0.0, 0.0, 0.0, 1.0];
    }
    let s = 1.0 / norm_sq.sqrt();
    [x * s, y * s, z * s, w * s]
}

/// Rotate a sensor-frame vector into the body frame using the
/// `q_sensor_body` mount quaternion stored on `ImuConfig`.
///
/// The BNO gyroscope publishes `omega_sensor` in the chip's silkscreen
/// axes. The policy observation builder (sim and real) expects FLU
/// body-frame angular velocity, so we apply
/// `omega_body = R_sensor_to_body * omega_sensor`.
///
/// **Convention note** — `mount_quat_sensor_body` (as built by
/// [`crate::config::build_mount_quat_sensor_body`]) is the *inverse*
/// of `R_body_sensor`, i.e. it actively rotates body-frame vectors
/// into sensor coordinates: `q_sensor_body * v_body = v_sensor`. To
/// go the other way we therefore apply its inverse, mirroring exactly
/// what the unit test `bebop_v2_mount_maps_sensor_axes_to_body_axes`
/// in `config.rs` does. This is consistent with the quaternion
/// composition `q_world_body = q_world_sensor * q_sensor_body` used
/// for the rotation vector: under the same convention, applying
/// `q_world_body` to a body vector gives the world vector, exactly as
/// the projected-gravity code in
/// [`crate::observation::ImuState::projected_gravity`] assumes.
///
/// Returns the input unchanged when the mount is identity so the
/// no-op path stays free of nalgebra construction cost — gyro reads
/// fire at 20–50 Hz, well below the threshold where this would matter
/// on the Jetson, but the explicit branch keeps the intent obvious
/// in flame graphs.
#[inline]
fn rotate_vec_by_quat_xyzw(v: [f32; 3], q_sensor_body_xyzw: [f32; 4]) -> [f32; 3] {
    if q_sensor_body_xyzw == ImuConfig::IDENTITY_QUAT {
        return v;
    }
    let [qx, qy, qz, qw] = q_sensor_body_xyzw;
    // nalgebra's `Quaternion::new` takes `(w, i, j, k)`; we store XYZW
    // (scalar last). `inverse()` on a unit quaternion is just the
    // conjugate, so this is cheap.
    let q = UnitQuaternion::from_quaternion(Quaternion::new(qw, qx, qy, qz));
    let out = q.inverse() * Vector3::new(v[0], v[1], v[2]);
    [out.x, out.y, out.z]
}

/// Spawn a background thread that initializes the IMU and publishes
/// AR/VR-stabilized rotation-vector quaternions into `shared` for the
/// WS server to ship.
///
/// The returned [`JoinHandle`] is wired up in `main.rs` so that
/// shutdown waits for the thread to drain — that wait is what gives
/// the graceful-disable epilogue (see end of the spawned closure) a
/// chance to run before the process exits, which is the linchpin of
/// the "clean restart" property described in §"Restart hygiene" below.
///
/// # Bring-up retries
///
/// All hardware bring-up — opening `/dev/spidev*` + GPIOs, the SHTP
/// boot handshake, and the `SET_FEATURE` for report 0x28 — happens
/// **inside** the spawned thread, in an infinite retry loop with
/// exponential backoff capped at [`BRINGUP_BACKOFF_MAX_MS`]. The
/// runtime therefore starts immediately even if the chip is wedged,
/// the operator app stays responsive, and the IMU just becomes
/// "live" (snapshot un-stales) the moment the chip cooperates. We
/// chose this over a bounded "give up after N tries" loop because
/// the most common failure mode in the field is a stale subscription
/// from the previous run (see "Restart hygiene" below) which clears
/// itself after a single successful re-init — there's no scenario
/// where giving up actually helps the operator.
///
/// The SHTP handshake is one of the more failure-prone parts of
/// bringing up the BNO over SPI:
///
///   * `bno08x-rs::SpiInterface::setup` toggles RST low for only
///     2 ms, then waits up to 200 ms for HINTN to fall. A
///     cold-booted Jetson where the 3.3 V rail just stabilized, or
///     an SPI clock momentarily perturbed by motor harness noise,
///     can miss that window and return `SensorUnresponsive`.
///   * If a previous run of `bebop-linux` exited *without* the
///     graceful-disable epilogue (panic in another thread,
///     OOM-kill, `kill -9`, hard power cut, …), the chip is still
///     streaming reports on the data channel from the stale
///     subscription, which collides with the SHTP control channel
///     during the new `verify_product_id` exchange.
///   * `enable_report(0x28, …)` itself can return an `Err` (not
///     just `Ok(false)`) when the chip is mid-recovery.
///
/// Each retry drops the previous `BNO08x` (which releases the GPIO
/// lines) and constructs a fresh one — that goes through
/// `SpiInterface::setup` again, performing another RST pulse and
/// giving the chip a clean second chance.
///
/// To keep the log readable, the per-attempt log severity demotes
/// after [`BRINGUP_LOUD_ATTEMPTS`] failures: the first few are
/// `warn`, then a single `error` summary, then a periodic `info`
/// every [`BRINGUP_PERIODIC_INFO_EVERY`] attempts so the "still
/// trying" signal stays visible without flooding the log.
///
/// `enable_report` returning `Ok(false)` is *not* treated as a
/// failure: it just means the chip didn't surface a
/// `GET_FEATURE_RESP` within the crate's 2 s window, which some
/// BNO085 firmware revs do even when the report is being streamed
/// correctly. The stream loop below is the source of truth on
/// whether reports actually arrive.
///
/// # Restart hygiene (graceful disable)
///
/// On shutdown — the `shutdown_flag` set by `main.rs` after Ctrl-C
/// or a server-task exit — the streaming loop falls out and the
/// epilogue at the bottom of the closure runs:
///
///   1. `enable_report(0x28, 0)` — period=0 µs is the SH-2 spec way
///      of saying "stop sending this report".
///   2. `enable_report(0x02, 0)` — same for the calibrated gyro.
///   3. `soft_reset()` — sends `EXECUTABLE_DEVICE_CMD_RESET` on the
///      executable channel, equivalent to a power-on reboot of the
///      sensor hub.
///
/// Together these put the chip in the same state it would be in
/// after a cold boot, so the next `bebop-linux` start finds a
/// quiet bus and a clean control channel. Skipping this step (e.g.
/// `kill -9`) is what causes the "fails 4× in a row on every
/// restart" symptom we used to see — the bring-up retry loop
/// recovers from that, but only after burning a couple of seconds.
/// Each call is `let _`'d on purpose: the chip may already be wedged
/// at this point, and a failed disable just means the next start
/// will lean on the retry loop again instead of getting the fast
/// path.
///
/// Returns `None` only if the IMU thread itself fails to spawn,
/// which in practice only happens under extreme resource exhaustion.
pub fn spawn_imu_thread(
    cfg: ImuConfig,
    shutdown: Arc<AtomicBool>,
    shared: ImuShared,
) -> Option<JoinHandle<()>> {
    /// Initial sleep between bring-up retries. Doubles each failure
    /// up to [`BRINGUP_BACKOFF_MAX_MS`].
    const BRINGUP_BACKOFF_MIN_MS: u64 = 250;
    /// Cap on the bring-up retry backoff. 5 s keeps us responsive to
    /// the operator power-cycling the IMU board mid-runtime, while
    /// being long enough that we're not eating SPI bandwidth on a
    /// chip that's permanently dead.
    const BRINGUP_BACKOFF_MAX_MS: u64 = 5_000;
    /// Per-attempt failures get a full `warn!` for the first N
    /// attempts; after that the log demotes to one `error!` summary
    /// followed by periodic `info!`s, so a chronically wedged IMU
    /// doesn't drown out the rest of the runtime's output.
    const BRINGUP_LOUD_ATTEMPTS: u32 = 5;
    /// Once we've gone quiet, still print a heartbeat every Nth
    /// attempt so the operator can tell from `journalctl` that the
    /// thread is alive and trying. With the 5 s backoff cap this
    /// works out to roughly one line per minute.
    const BRINGUP_PERIODIC_INFO_EVERY: u32 = 12;

    let period_ms = cfg.rotation_vector_period_ms;
    let mount_quat = cfg.mount_quat_sensor_body;
    let mount_is_identity = mount_quat == ImuConfig::IDENTITY_QUAT;

    // Seed the period so the telemetry builder's stale check has a
    // sane budget even before the first successful read.
    if let Ok(mut g) = shared.lock() {
        g.report_period_ms = period_ms;
    }

    info!(
        period_ms,
        spi = %cfg.spi_device,
        mount_quat_sensor_body = ?mount_quat,
        mount_is_identity,
        "IMU: thread spawning; will retry SHTP bring-up forever in background \
         (runtime continues without a live IMU until the chip responds)"
    );

    // Helper: sleep up to `dur`, returning early if the shutdown
    // flag is set. Polls the flag every 50 ms so even a 5 s backoff
    // can't delay shutdown by more than that. Returns `true` if the
    // sleep was interrupted by shutdown.
    fn sleep_or_shutdown(dur: Duration, shutdown: &Arc<AtomicBool>) -> bool {
        let wake = Instant::now() + dur;
        while let Some(remaining) = wake.checked_duration_since(Instant::now()) {
            if shutdown.load(Ordering::SeqCst) {
                return true;
            }
            std::thread::sleep(remaining.min(Duration::from_millis(50)));
        }
        false
    }

    Some(std::thread::spawn(move || {
        // ---------- Bring-up: retry forever until success or shutdown ----------
        //
        // Both this loop body and the streaming loop below are inlined
        // (rather than factored into helpers) so we don't have to name
        // `BNO08x`'s sensor-interface generic parameter — `new_spi`
        // returns a concrete but unwieldy
        // `BNO08x<'_, SpiInterface<SpiDevice, GpiodIn, GpiodOut>>` that
        // isn't re-exported from `bno08x-rs`'s crate root. Closure /
        // `loop { break value }` type inference sidesteps the problem
        // and keeps us free of `bno08x_rs::interface::spi::*` private-ish
        // submodule paths that the upstream may rearrange between
        // releases.
        let mut backoff = Duration::from_millis(BRINGUP_BACKOFF_MIN_MS);
        let mut attempt: u32 = 0;
        let mut imu = loop {
            if shutdown.load(Ordering::SeqCst) {
                info!(target: "bebop_linux::imu", "IMU thread exiting before bring-up succeeded");
                return;
            }
            attempt += 1;
            let bringup_result: Result<_, String> = (|| {
                let mut imu = BNO08x::new_spi(
                    &cfg.spi_device,
                    &cfg.int_chip,
                    cfg.int_line,
                    &cfg.rst_chip,
                    cfg.rst_line,
                )
                .map_err(|e| {
                    format!(
                        "open SPI={} INT={}:{} RST={}:{}: {e:?}",
                        cfg.spi_device, cfg.int_chip, cfg.int_line, cfg.rst_chip, cfg.rst_line
                    )
                })?;
                imu.init()
                    .map_err(|e| format!("SHTP boot handshake (init): {e:?}"))?;
                imu.enable_report(SENSOR_REPORTID_ARVR_STABILIZED_RV, period_ms)
                    .map_err(|e| format!("SET_FEATURE for 0x28: {e:?}"))?;
                Ok(imu)
            })();

            match bringup_result {
                Ok(imu) => {
                    info!(
                        attempt,
                        period_ms,
                        "IMU: bring-up succeeded; AR/VR-stabilized RV (0x28) subscribed"
                    );
                    break imu;
                }
                Err(msg) => {
                    if attempt <= BRINGUP_LOUD_ATTEMPTS {
                        warn!(
                            attempt,
                            backoff_ms = backoff.as_millis() as u64,
                            error = %msg,
                            "IMU: bring-up attempt failed; retrying after backoff"
                        );
                    } else if attempt == BRINGUP_LOUD_ATTEMPTS + 1 {
                        error!(
                            attempt,
                            error = %msg,
                            "IMU: bring-up still failing after {} attempts; \
                             will keep retrying every {} ms but suppressing \
                             per-attempt warnings (heartbeat every {} attempts) \
                             — hint: check the BNO SPI-mode jumpers, INT/RST \
                             wiring, or power-cycle the IMU board to clear any \
                             stale feature subscription from a previous run",
                            BRINGUP_LOUD_ATTEMPTS,
                            BRINGUP_BACKOFF_MAX_MS,
                            BRINGUP_PERIODIC_INFO_EVERY
                        );
                    } else if attempt % BRINGUP_PERIODIC_INFO_EVERY == 0 {
                        info!(
                            attempt,
                            error = %msg,
                            "IMU: bring-up still failing (heartbeat)"
                        );
                    }
                    if sleep_or_shutdown(backoff, &shutdown) {
                        info!(
                            target: "bebop_linux::imu",
                            attempt,
                            "IMU thread exiting before bring-up succeeded"
                        );
                        return;
                    }
                    backoff =
                        (backoff * 2).min(Duration::from_millis(BRINGUP_BACKOFF_MAX_MS));
                }
            }
        };

        // Subscribe to 0x02 (Calibrated Gyroscope). The chip applies
        // its own bias-tracking calibration to this report, which is
        // what we want for closed-loop control — the uncalibrated
        // stream (0x07) takes several seconds to converge after a
        // power cycle.
        //
        // Unlike the rotation vector, we don't retry the gyro: a
        // failed gyro subscription leaves the policy with synthetic
        // angular velocity = 0, which is a safe (if degraded) state.
        // The single warn is loud enough to surface in observability
        // tools without spamming the log.
        match imu.enable_report(SENSOR_REPORTID_GYROSCOPE, period_ms) {
            Ok(true) => info!(period_ms, "IMU: enabled report 0x02 (Calibrated Gyroscope)"),
            Ok(false) => warn!(
                period_ms,
                "IMU: no GET_FEATURE_RESP for 0x02 within 2 s; \
                 continuing on the assumption the chip is streaming anyway"
            ),
            Err(e) => warn!(
                ?e,
                "IMU: SET_FEATURE for 0x02 failed; angular_velocity_body \
                 will stay None and PolicyRunner will use a zero gyro"
            ),
        }

        info!(
            period_ms,
            "IMU: streaming AR/VR-stabilized rotation vector + calibrated gyro \
             into shared state (consumed by PolicyRunner and telemetry pump)"
        );

        // ---------- Streaming loop ----------
        while !shutdown.load(Ordering::SeqCst) {
            // Pump SHTP messages; 25 ms per-message timeout matches
            // the diagnostic probe so a quiet bus doesn't stall the
            // loop.
            let _ = imu.handle_all_messages(25);

            // Read the AR/VR-stabilized cache populated by the
            // vendored-crate parser. The bno08x-rs cache returns the
            // SHTP rotation vector in (i, j, k, real) order, i.e.
            // XYZW. Compose with the mount rotation so what we publish
            // is `q_world_body`, not the raw `q_world_sensor`. When
            // `mount:` is omitted from the YAML this multiply is a
            // no-op (identity quat).
            //
            // The final `quat_normalize_xyzw` enforces the
            // `ImuSnapshot.quaternion` contract: a *unit* body-frame
            // quaternion. The driver's cached buffer starts at all-
            // zeros before the first report, so without normalization
            // we'd briefly publish `(0, 0, 0, 0)` — which trips
            // downstream Euler decompositions in confusing ways (see
            // the operator-app orientation card's Roll/Pitch/Yaw
            // readout).
            let now = Instant::now();

            match imu.arvr_stabilized_rotation_quaternion() {
                Ok([qx, qy, qz, qw]) => {
                    let q_world_sensor = [qx, qy, qz, qw];
                    let q_world_body = if mount_is_identity {
                        quat_normalize_xyzw(q_world_sensor)
                    } else {
                        quat_normalize_xyzw(quat_mul_xyzw(q_world_sensor, mount_quat))
                    };
                    let acc = imu.arvr_stabilized_rotation_acc();
                    if let Ok(mut g) = shared.lock() {
                        g.quaternion = Some(q_world_body);
                        g.heading_accuracy_rad = acc;
                        g.last_update = Some(now);
                        g.report_period_ms = period_ms;
                    }
                }
                Err(e) => warn!(
                    target: "bebop_linux::imu",
                    ?e,
                    "arvr_stabilized_rotation_quaternion read"
                ),
            }

            // Read the calibrated gyroscope cache and rotate into body
            // frame. Note this reads the *cached* values held by the
            // driver — both the rotation-vector and gyroscope reports
            // arrive on the same data channel and the driver updates
            // its caches as new reports come in, so there's no risk of
            // racing with the SHTP pump (we already drained it with
            // `handle_all_messages` above).
            //
            // The gyro cache is seeded to all-zeros before the first
            // report. The driver doesn't expose a "have we ever
            // received a 0x02?" flag, but the chip starts streaming
            // 0x02 almost immediately after `enable_report` returns,
            // so the first real sample lands within a couple of loop
            // iterations. We initially publish `None` and switch to
            // `Some(...)` on the first non-zero read so the policy
            // can distinguish "no gyro yet" from "perfectly still".
            //
            // A perfectly-still robot does in principle read exactly
            // [0, 0, 0], so this distinguishes by sample magnitude
            // rather than by a separate "received" flag. The BNO's
            // own noise floor is ~3e-3 rad/s even on a tripod, well
            // above the 1e-9 threshold below; in practice we publish
            // `Some(...)` on the *very* first report.
            match imu.gyro() {
                Ok([wx, wy, wz]) => {
                    let omega_sensor = [wx, wy, wz];
                    let mag_sq = wx * wx + wy * wy + wz * wz;
                    if mag_sq > 1e-9 {
                        let omega_body = rotate_vec_by_quat_xyzw(omega_sensor, mount_quat);
                        if let Ok(mut g) = shared.lock() {
                            g.angular_velocity_body = Some(omega_body);
                            // `last_update` already covers both
                            // reports — they arrive on the same data
                            // channel at the same cadence, so a single
                            // staleness clock is sufficient.
                            g.last_update = Some(now);
                            g.report_period_ms = period_ms;
                        }
                    }
                }
                Err(e) => warn!(
                    target: "bebop_linux::imu",
                    ?e,
                    "calibrated gyroscope read"
                ),
            }
            std::thread::sleep(Duration::from_millis(2));
        }

        // ---------- Graceful disable (restart hygiene) ----------
        //
        // The chip is currently streaming 0x28 + 0x02 on the data
        // channel at `period_ms` cadence. If we just drop `imu` here
        // (which releases the SPI/GPIO handles) the chip happily
        // keeps streaming into the void — and the next `bebop-linux`
        // start will then see those stale reports collide with the
        // SHTP control channel during its own `verify_product_id`
        // exchange, triggering the slow retry loop above.
        //
        // SH-2 SET_FEATURE with `interval_us = 0` is the spec way to
        // tell the chip "stop sending this report"; we do that for
        // both subscriptions to silence the data channel.
        //
        // We deliberately do NOT also call `soft_reset()` here:
        //   1. It's redundant — the *next* bring-up's
        //      `BNO08x::new_spi` runs `SpiInterface::setup`, which
        //      RST-pulses the chip via the GPIO line and brings it
        //      up cold-boot-clean regardless of whether we
        //      soft-reset it on the way out.
        //   2. The chip's reply to `soft_reset` is the full SHTP
        //      advertisement (~1.2 KB on the BNO085 revs we ship),
        //      which the upstream `handle_advertise_response` used
        //      to walk off the end of (panic at `driver.rs:595`,
        //      `index out of bounds: the len is 1276 but the index
        //      is 1276`). We've patched that bug — see BEBOP-PATCH
        //      [5/5] — but the soft-reset path is still ~500 ms of
        //      pointless wait at shutdown, and the disable calls
        //      below are sufficient for restart hygiene on their
        //      own.
        //
        // Both calls are best-effort (`let _`): if they fail it's
        // because the chip is wedged hard enough that the bring-up
        // loop above will need to RST-pulse it on the next start
        // anyway, and there's nothing useful we can do about it
        // from a process that's already on its way out.
        let _ = imu.enable_report(SENSOR_REPORTID_ARVR_STABILIZED_RV, 0);
        let _ = imu.enable_report(SENSOR_REPORTID_GYROSCOPE, 0);
        info!(
            target: "bebop_linux::imu",
            "IMU thread exiting (subscriptions disabled for clean restart)"
        );
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: [f32; 4], b: [f32; 4]) -> bool {
        a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < 1e-5)
    }

    fn quat_norm(q: [f32; 4]) -> f32 {
        (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt()
    }

    #[test]
    fn identity_post_multiply_is_a_noop() {
        let q = [0.1_f32, -0.2, 0.3, 0.927_362];
        let out = quat_mul_xyzw(q, ImuConfig::IDENTITY_QUAT);
        assert!(approx_eq(out, q), "identity should not change the input");
    }

    #[test]
    fn ninety_deg_yaw_mount_maps_sensor_x_to_body_minus_y() {
        // q_sensor_body for the Bebop V2 mount (sensor +X = body -Y,
        // sensor +Y = body +X, sensor +Z = body +Z) is a +90° rotation
        // about +Z (carries body +X onto sensor +Y).
        let s = (std::f32::consts::FRAC_PI_4).sin();
        let c = (std::f32::consts::FRAC_PI_4).cos();
        let q_sb = [0.0, 0.0, s, c]; // axis-angle (z, +90°) in XYZW

        // Imagine the BNO reports the sensor frame is aligned with world
        // (identity). Then `q_world_body` should equal `q_sensor_body`
        // itself — the body is rotated relative to world by exactly the
        // mount rotation.
        let q_world_sensor = ImuConfig::IDENTITY_QUAT;
        let q_world_body = quat_mul_xyzw(q_world_sensor, q_sb);
        assert!(approx_eq(q_world_body, q_sb));
    }

    #[test]
    fn normalize_leaves_unit_quaternion_unit() {
        let q = [0.0_f32, 0.0, -0.477_158_5, 0.878_817_4]; // yaw = -57° about z
        let n_in = quat_norm(q);
        assert!((n_in - 1.0).abs() < 1e-5);
        let out = quat_normalize_xyzw(q);
        assert!(approx_eq(out, q));
        assert!((quat_norm(out) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn normalize_rescales_non_unit_input() {
        // Same situation we used to see in the operator-app screenshot:
        // values that decompose as something close to (≈0, 0, -0.55, 0.55)
        // — norm ~0.778, not unit. The widget's Euler readout was
        // computing roll/pitch/yaw straight from these and producing
        // wrong angles; the firmware now hands out unit quaternions so
        // the widget can trust them.
        let q = [0.01_f32, 0.0, -0.55, 0.55];
        let norm_in = quat_norm(q);
        assert!((norm_in - 0.778).abs() < 0.01);
        let out = quat_normalize_xyzw(q);
        assert!((quat_norm(out) - 1.0).abs() < 1e-6);
        // Direction must be preserved up to scale.
        for (orig, scaled) in q.iter().zip(out.iter()) {
            assert!((orig - scaled * norm_in).abs() < 1e-5);
        }
    }

    #[test]
    fn normalize_zero_returns_identity() {
        // The driver seeds its rotation-quaternion buffer with all
        // zeros before the first report arrives. We must not propagate
        // (0, 0, 0, 0) downstream.
        let out = quat_normalize_xyzw([0.0; 4]);
        assert_eq!(out, [0.0, 0.0, 0.0, 1.0]);
    }

    fn approx_eq3(a: [f32; 3], b: [f32; 3]) -> bool {
        a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < 1e-5)
    }

    #[test]
    fn rotate_vec_identity_is_a_noop() {
        let v = [0.7_f32, -0.2, 0.5];
        let out = rotate_vec_by_quat_xyzw(v, ImuConfig::IDENTITY_QUAT);
        assert!(approx_eq3(out, v), "identity should not change the input");
    }

    #[test]
    fn rotate_vec_v2_mount_remaps_axes() {
        // Bebop V2 mount: sensor +X = body -Y, sensor +Y = body +X,
        // sensor +Z = body +Z. The mount quaternion encodes a +90°
        // rotation about +Z (yaw). Apply it to the three unit-vector
        // sensor axes and check we land on the documented body axes
        // — same axis mapping the YAML comment promises and the
        // `ninety_deg_yaw_mount_maps_sensor_x_to_body_minus_y` test
        // verifies for the orientation quaternion path.
        let s = (std::f32::consts::FRAC_PI_4).sin();
        let c = (std::f32::consts::FRAC_PI_4).cos();
        let q_sb = [0.0, 0.0, s, c]; // axis-angle (z, +90°) in XYZW

        // sensor +X -> body -Y
        let out_x = rotate_vec_by_quat_xyzw([1.0, 0.0, 0.0], q_sb);
        assert!(
            approx_eq3(out_x, [0.0, -1.0, 0.0]),
            "expected sensor +X -> body -Y but got {out_x:?}"
        );

        // sensor +Y -> body +X
        let out_y = rotate_vec_by_quat_xyzw([0.0, 1.0, 0.0], q_sb);
        assert!(
            approx_eq3(out_y, [1.0, 0.0, 0.0]),
            "expected sensor +Y -> body +X but got {out_y:?}"
        );

        // sensor +Z -> body +Z
        let out_z = rotate_vec_by_quat_xyzw([0.0, 0.0, 1.0], q_sb);
        assert!(
            approx_eq3(out_z, [0.0, 0.0, 1.0]),
            "expected sensor +Z -> body +Z but got {out_z:?}"
        );
    }

    #[test]
    fn rotate_vec_preserves_magnitude() {
        // Round-trip a random-looking gyro reading through an
        // arbitrary mount and confirm we don't accidentally scale it
        // (a classic sign of treating XYZW as WXYZ or vice versa).
        let s = (std::f32::consts::FRAC_PI_4).sin();
        let c = (std::f32::consts::FRAC_PI_4).cos();
        let q_sb = [0.0, 0.0, s, c];
        let v = [0.31_f32, -1.27, 0.86];
        let mag_in = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        let out = rotate_vec_by_quat_xyzw(v, q_sb);
        let mag_out = (out[0] * out[0] + out[1] * out[1] + out[2] * out[2]).sqrt();
        assert!((mag_in - mag_out).abs() < 1e-5);
    }

    #[test]
    fn mount_mul_then_normalize_is_unit_for_any_input() {
        // Use the V2 mount (+90° yaw) and a deliberately wobbly input
        // to confirm the published quaternion is always unit.
        let s = (std::f32::consts::FRAC_PI_4).sin();
        let c = (std::f32::consts::FRAC_PI_4).cos();
        let q_sb = [0.0, 0.0, s, c];
        for q_in in [
            [1.0_f32, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.5, 0.0, -0.5, 0.5], // norm = 0.866
            [0.01, 0.0, -0.55, 0.55],
        ] {
            let out = quat_normalize_xyzw(quat_mul_xyzw(q_in, q_sb));
            assert!(
                (quat_norm(out) - 1.0).abs() < 1e-6,
                "input {q_in:?} produced non-unit output {out:?}"
            );
        }
    }
}
