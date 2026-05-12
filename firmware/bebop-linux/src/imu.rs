//! BNO080/BNO085 SPI reader that publishes fused orientation into a
//! shared state so the WS server can ship it to operator clients.
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

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use bno08x_rs::{BNO08x, SENSOR_REPORTID_ARVR_STABILIZED_RV};
use tracing::{error, info, warn};

use crate::config::ImuConfig;

/// Latest decoded BNO080/BNO085 rotation vector reading.
///
/// # Frame & normalization contract
///
/// `quaternion` is the **body-frame** orientation in **XYZW (Hamilton)**
/// order, matching [`crate::observation::ImuState::quaternion`].
///
/// 1. Frame: any constant sensor→body mount rotation declared in the
///    YAML `imu.mount:` block is applied here, in
///    [`run_imu_loop`], **before** the value reaches `ImuSnapshot`.
///    Downstream consumers (telemetry pump, future `PolicyRunner` IMU
///    feed, ROS bridge) can therefore treat this as a calibrated
///    body-frame (REP-103 / FLU: `+x forward`, `+y left`, `+z up`)
///    attitude with no further rotation needed.
/// 2. Normalization: when `Some`, the producer guarantees
///    `|quaternion| = 1` within float precision (we explicitly
///    normalize after the mount multiply to compensate for any drift
///    in the BNO output or numerical noise from the composition).
/// 3. Identity is `(0.0, 0.0, 0.0, 1.0)`. `None` (= `last_update` is
///    `None`) means no rotation-vector frame has been decoded yet.
#[derive(Debug, Default, Clone, Copy)]
pub struct ImuSnapshot {
    pub quaternion: Option<[f32; 4]>,
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

/// Spawn a background thread that initializes the IMU and publishes
/// AR/VR-stabilized rotation-vector quaternions into `shared` for the
/// WS server to ship.
///
/// Returns `None` if the SPI device or either GPIO line can't be
/// opened, or if the SHTP boot handshake fails. The caller logs the
/// error and continues — the rest of the runtime operates fine
/// without a live IMU (synthetic IMU is used in the policy path).
pub fn spawn_imu_thread(
    cfg: ImuConfig,
    shutdown: Arc<AtomicBool>,
    shared: ImuShared,
) -> Option<JoinHandle<()>> {
    let period_ms = cfg.rotation_vector_period_ms;
    let mount_quat = cfg.mount_quat_sensor_body;

    // Seed the period so the telemetry builder's stale check has a
    // sane budget even before the first successful read.
    if let Ok(mut g) = shared.lock() {
        g.report_period_ms = period_ms;
    }

    // Note we bring up the chip on the *caller's* thread so any
    // hardware failure (missing /dev/spidev*, wrong GPIO line, dead
    // chip) surfaces synchronously and gets logged at startup
    // instead of vanishing into a spawned thread that exits silently.
    let mut imu = match BNO08x::new_spi(
        &cfg.spi_device,
        &cfg.int_chip,
        cfg.int_line,
        &cfg.rst_chip,
        cfg.rst_line,
    ) {
        Ok(imu) => imu,
        Err(e) => {
            error!(
                ?e,
                spi = %cfg.spi_device,
                int = format!("{}:{}", cfg.int_chip, cfg.int_line),
                rst = format!("{}:{}", cfg.rst_chip, cfg.rst_line),
                "IMU: failed to open SPI / GPIO lines; thread not started"
            );
            return None;
        }
    };

    if let Err(e) = imu.init() {
        error!(
            ?e,
            "IMU: SHTP boot handshake failed; thread not started \
             (hint: check the BNO SPI-mode jumpers, INT/RST wiring, \
             or that no previous run left a feature subscription \
             active without a power-cycle)"
        );
        return None;
    }

    // Subscribe to 0x28 (AR/VR-Stabilized Rotation Vector). Some
    // BNO085 firmware revisions don't auto-emit the GET_FEATURE_RESP
    // bno08x-rs polls for, so `enable_report` may return `Ok(false)`
    // even when the chip is happily streaming the report on the data
    // channel. We treat that case as a soft warning (the stream loop
    // below is the source of truth on whether reports actually
    // arrive). See `src/bin/imu_probe.rs` for the same logic.
    match imu.enable_report(SENSOR_REPORTID_ARVR_STABILIZED_RV, period_ms) {
        Ok(true) => info!(
            period_ms,
            "IMU: enabled report 0x28 (AR/VR-Stabilized Rotation Vector)"
        ),
        Ok(false) => warn!(
            period_ms,
            "IMU: no GET_FEATURE_RESP for 0x28 within 2 s; \
             continuing on the assumption the chip is streaming anyway"
        ),
        Err(e) => {
            error!(
                ?e,
                "IMU: SET_FEATURE for 0x28 failed; thread not started"
            );
            return None;
        }
    }

    let mount_is_identity = mount_quat == ImuConfig::IDENTITY_QUAT;
    info!(
        period_ms,
        spi = %cfg.spi_device,
        mount_quat_sensor_body = ?mount_quat,
        mount_is_identity,
        "IMU: streaming AR/VR-stabilized rotation vector into shared state \
         (policy still uses synthetic IMU)"
    );

    Some(std::thread::spawn(move || {
        // Loop body inlined here (rather than a `run_imu_loop` helper)
        // so we don't have to name `BNO08x`'s sensor-interface generic
        // parameter — `new_spi` returns a concrete but unwieldy type
        // that's easier to capture-by-move into a closure than to
        // declare in a function signature.
        let mut imu = imu; // re-bind as `mut` so the closure can call &mut self methods
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
                        g.last_update = Some(Instant::now());
                        g.report_period_ms = period_ms;
                    }
                }
                Err(e) => warn!(
                    target: "bebop_linux::imu",
                    ?e,
                    "arvr_stabilized_rotation_quaternion read"
                ),
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        info!(target: "bebop_linux::imu", "IMU thread exiting");
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
