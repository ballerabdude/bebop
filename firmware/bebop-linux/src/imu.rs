//! Optional BNO080/BNO085 I²C reader that publishes fused orientation
//! into a shared state so the WS server can ship it to operator clients.
//!
//! The policy stack still uses synthetic IMU in [`crate::policy_runner`];
//! this module exists to surface the live rotation vector in the operator
//! app (motor-bench orientation card) before the policy path is wired up
//! to consume real IMU readings.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use bno080::interface::I2cInterface;
use bno080::wrapper::BNO080;
use embedded_hal::blocking::delay::DelayMs;
use embedded_hal::blocking::i2c::{Read, Write, WriteRead};
use i2cdev::core::I2CDevice;
use i2cdev::linux::{LinuxI2CDevice, LinuxI2CError};
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
/// Cloned into the I²C reader thread (writer) and the WS server's
/// telemetry builder (reader). A plain [`Mutex`] is sufficient here:
/// updates happen at the sensor's report period (~20 Hz at the default
/// 50 ms) and the telemetry pump reads at ≤100 Hz, so contention is
/// negligible.
pub type ImuShared = Arc<Mutex<ImuSnapshot>>;

/// Allocate a fresh shared snapshot. Initial state is "no quaternion
/// yet, stale" — safe to wire into the telemetry builder before the
/// I²C reader thread starts (or when no IMU is configured at all).
pub fn new_shared() -> ImuShared {
    Arc::new(Mutex::new(ImuSnapshot::default()))
}

/// `embedded-hal` delay backed by `std::thread::sleep`.
struct ThreadDelay;

impl DelayMs<u8> for ThreadDelay {
    fn delay_ms(&mut self, ms: u8) {
        std::thread::sleep(Duration::from_millis(ms as u64));
    }
}

/// Bridges `LinuxI2CDevice` (single-slave `i2cdev` file) to `embedded-hal` I²C
/// traits expected by `bno080::I2cInterface`.
struct EhLinuxI2c {
    dev: LinuxI2CDevice,
    slave: u8,
}

impl EhLinuxI2c {
    fn open(path: &str, slave: u8) -> Result<Self, LinuxI2CError> {
        let dev = LinuxI2CDevice::new(path, u16::from(slave))?;
        Ok(Self { dev, slave })
    }

    fn check_addr(&self, addr: u8) -> Result<(), LinuxI2CError> {
        if addr != self.slave {
            return Err(LinuxI2CError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "unexpected I²C address 0x{addr:02x} (device is 0x{:02x})",
                    self.slave
                ),
            )));
        }
        Ok(())
    }
}

impl Read for EhLinuxI2c {
    type Error = LinuxI2CError;

    fn read(&mut self, address: u8, buffer: &mut [u8]) -> Result<(), Self::Error> {
        self.check_addr(address)?;
        self.dev.read(buffer)
    }
}

impl Write for EhLinuxI2c {
    type Error = LinuxI2CError;

    fn write(&mut self, address: u8, bytes: &[u8]) -> Result<(), Self::Error> {
        self.check_addr(address)?;
        self.dev.write(bytes)
    }
}

impl WriteRead for EhLinuxI2c {
    type Error = LinuxI2CError;

    fn write_read(
        &mut self,
        address: u8,
        bytes: &[u8],
        buffer: &mut [u8],
    ) -> Result<(), Self::Error> {
        // BNO08x does not use a repeated-start read; separate transactions match
        // `bno080::interface::i2c` expectations.
        self.write(address, bytes)?;
        self.read(address, buffer)
    }
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
/// arrived (the bno080 crate's cached value starts at all-zeros) and
/// avoids a divide-by-zero rather than silently propagating an
/// unphysical quaternion.
///
/// In normal operation the BNO080 emits a unit quaternion and the
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
/// rotation-vector quaternions into `shared` for the WS server to ship.
/// Returns `None` if the I²C device cannot be opened.
pub fn spawn_imu_thread(
    cfg: ImuConfig,
    shutdown: Arc<AtomicBool>,
    shared: ImuShared,
) -> Option<JoinHandle<()>> {
    let path = cfg.i2c_device.clone();
    let addr = cfg.i2c_address;
    let rv_ms = cfg.rotation_vector_period_ms;
    let mount_quat = cfg.mount_quat_sensor_body;

    // Seed the period so the telemetry builder's stale check has a
    // sane budget even before the first successful read.
    if let Ok(mut g) = shared.lock() {
        g.report_period_ms = rv_ms;
    }

    match EhLinuxI2c::open(&path, addr) {
        Ok(i2c) => Some(std::thread::spawn(move || {
            run_imu_loop(i2c, addr, rv_ms, mount_quat, shutdown, shared);
        })),
        Err(e) => {
            error!(
                error = %e,
                device = %path,
                address = format!("0x{addr:02x}"),
                "IMU: failed to open I²C device; thread not started"
            );
            None
        }
    }
}

fn run_imu_loop(
    i2c: EhLinuxI2c,
    i2c_address: u8,
    rotation_vector_period_ms: u16,
    mount_quat_sensor_body: [f32; 4],
    shutdown: Arc<AtomicBool>,
    shared: ImuShared,
) {
    let iface = I2cInterface::new(i2c, i2c_address);
    let mut imu = BNO080::new_with_interface(iface);
    let mut delay = ThreadDelay;

    if let Err(e) = imu.init(&mut delay) {
        error!(?e, "IMU: init failed");
        return;
    }
    if let Err(e) = imu.enable_rotation_vector(rotation_vector_period_ms) {
        error!(?e, "IMU: enable_rotation_vector failed");
        return;
    }

    let mount_is_identity = mount_quat_sensor_body == ImuConfig::IDENTITY_QUAT;
    info!(
        period_ms = rotation_vector_period_ms,
        mount_quat_sensor_body = ?mount_quat_sensor_body,
        mount_is_identity,
        "IMU: streaming rotation vector into shared state (policy still uses synthetic IMU)"
    );

    while !shutdown.load(Ordering::SeqCst) {
        let _ = imu.handle_all_messages(&mut delay, 25);
        match imu.rotation_quaternion() {
            Ok([qx, qy, qz, qw]) => {
                // The bno080 crate returns the SHTP rotation vector in
                // (i, j, k, real) order, i.e. XYZW. Compose with the
                // mount rotation so what we publish is `q_world_body`,
                // not the raw `q_world_sensor`. When `mount:` is omitted
                // from the YAML this multiply is a no-op (identity quat).
                //
                // The final `quat_normalize_xyzw` enforces the
                // `ImuSnapshot.quaternion` contract: a *unit* body-frame
                // quaternion. The bno080 crate's cached buffer starts
                // at all-zeros before the first report, so without
                // normalization we'd briefly publish `(0, 0, 0, 0)` —
                // which trips downstream Euler decompositions in
                // confusing ways (see the operator-app orientation
                // card's Roll/Pitch/Yaw readout).
                let q_world_sensor = [qx, qy, qz, qw];
                let q_world_body = if mount_is_identity {
                    quat_normalize_xyzw(q_world_sensor)
                } else {
                    quat_normalize_xyzw(quat_mul_xyzw(q_world_sensor, mount_quat_sensor_body))
                };
                let acc = imu.heading_accuracy();
                if let Ok(mut g) = shared.lock() {
                    g.quaternion = Some(q_world_body);
                    g.heading_accuracy_rad = acc;
                    g.last_update = Some(Instant::now());
                    g.report_period_ms = rotation_vector_period_ms;
                }
            }
            Err(e) => warn!(target: "bebop_linux::imu", ?e, "rotation_quaternion read"),
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    info!(target: "bebop_linux::imu", "IMU thread exiting");
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
        // Same situation observed in the operator-app screenshot:
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
        // bno080 crate seeds its rotation_quaternion buffer with all
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
