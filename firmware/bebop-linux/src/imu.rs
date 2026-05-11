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
/// `quaternion` is XYZW (Hamilton), matching
/// [`crate::observation::ImuState::quaternion`]. Identity is
/// `(0.0, 0.0, 0.0, 1.0)`. `last_update` is `None` until the first
/// successful read; downstream consumers use it (together with
/// `report_period_ms`) to mark the reading stale.
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

    // Seed the period so the telemetry builder's stale check has a
    // sane budget even before the first successful read.
    if let Ok(mut g) = shared.lock() {
        g.report_period_ms = rv_ms;
    }

    match EhLinuxI2c::open(&path, addr) {
        Ok(i2c) => Some(std::thread::spawn(move || {
            run_imu_loop(i2c, addr, rv_ms, shutdown, shared);
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

    info!(
        period_ms = rotation_vector_period_ms,
        "IMU: streaming rotation vector into shared state (policy still uses synthetic IMU)"
    );

    while !shutdown.load(Ordering::SeqCst) {
        let _ = imu.handle_all_messages(&mut delay, 25);
        match imu.rotation_quaternion() {
            Ok([qx, qy, qz, qw]) => {
                let acc = imu.heading_accuracy();
                if let Ok(mut g) = shared.lock() {
                    g.quaternion = Some([qx, qy, qz, qw]);
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
