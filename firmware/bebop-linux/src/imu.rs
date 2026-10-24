//! Shared BNO085 IMU state: the latest fused orientation **and**
//! body-frame angular velocity, plus the quaternion helpers the reader
//! uses to rotate raw sensor-frame readings into the body frame.
//!
//! The BNO085 is wired to the Teensy, which runs the `teensy_bridge`
//! firmware: it subscribes to SH-2 report **0x28** ("AR/VR-Stabilized
//! Rotation Vector", whose fusion filter rejects the magnetometer
//! disturbances from nearby motor currents instead of freezing yaw) and
//! report **0x02** ("Calibrated Gyroscope"), and streams both to the Jetson
//! over USB serial. [`crate::imu_serial`] parses that stream and writes
//! each sample into the [`ImuShared`] snapshot defined here; the telemetry
//! builders read it.
//!
//! ## Frame contract
//!
//! See [`ImuSnapshot`] for the body-frame XYZW unit quaternion contract:
//! the reader post-multiplies every raw reading by the YAML
//! `mount_quat_sensor_body` before publishing.
//!
//! The gyro vector follows the same convention: the BNO publishes
//! `omega_sensor` in the chip's silkscreen frame and the reader rotates it
//! into the body frame with `rotate_vec_by_quat_xyzw`, so
//! [`ImuSnapshot::angular_velocity_body`] is always FLU body-frame
//! (+x forward, +y left, +z up) rad/s.
//!
//! ## Quaternion order
//!
//! `ImuSnapshot.quaternion` is **XYZW** (scalar last), the ROS /
//! `geometry_msgs` order and the Isaac Lab 3.0 default.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use nalgebra::{Quaternion, UnitQuaternion, Vector3};

use crate::config::ImuConfig;

/// Latest decoded BNO085 rotation vector + gyroscope reading.
///
/// # Frame & normalization contract
///
/// `quaternion` is the **body-frame** orientation in **XYZW (Hamilton)**
/// order.
///
/// 1. Frame: any constant sensor→body mount rotation declared in the
///    YAML `imu.mount:` block is applied by the reader
///    ([`crate::imu_serial::spawn_imu_serial_thread`]) **before** the
///    value reaches `ImuSnapshot`. Downstream consumers (the telemetry
///    builders) can therefore treat this as a calibrated body-frame
///    (REP-103 / FLU: `+x forward`, `+y left`, `+z up`) attitude with no
///    further rotation needed.
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
    /// report has been decoded since startup. FLU axis convention.
    pub angular_velocity_body: Option<[f32; 3]>,
    pub last_update: Option<Instant>,
    pub report_period_ms: u16,
}

impl ImuSnapshot {
    /// True if the most recent successful read is older than
    /// `3 × report_period_ms` (clamped to at least 250 ms). Returns
    /// `true` before the first frame so the UI greys out the widget.
    ///
    /// This is the *only* IMU liveness signal — the gyro and rotation
    /// vector arrive together in every serial frame, so when this reads
    /// "fresh" the calibrated-gyro stream is fresh too.
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
/// Cloned into the serial reader thread (writer) and the WS server's
/// telemetry builder (reader). A plain [`Mutex`] is sufficient here:
/// updates arrive at the bridge's frame rate (~200 Hz) and the telemetry
/// pump reads at ≤100 Hz, so contention is negligible.
pub type ImuShared = Arc<Mutex<ImuSnapshot>>;

/// Allocate a fresh shared snapshot. Initial state is "no quaternion
/// yet, stale" — safe to wire into the telemetry builder before the
/// reader thread starts (or when no IMU is configured at all).
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
pub(crate) fn quat_mul_xyzw(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
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
/// tiny threshold — an all-zero quaternion (e.g. a bridge that has not
/// read the chip yet) — avoiding a divide-by-zero rather than silently
/// propagating an unphysical quaternion.
///
/// In normal operation the BNO emits a unit quaternion and the
/// constant mount multiply preserves that, so this just trims off the
/// ~1e-6 of float drift that accumulates after the composition.
#[inline]
pub(crate) fn quat_normalize_xyzw(q: [f32; 4]) -> [f32; 4] {
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
/// axes. Consumers expect FLU body-frame angular velocity, so we apply
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
/// `q_world_body` to a body vector gives the world vector.
///
/// Returns the input unchanged when the mount is identity so the
/// no-op path stays free of nalgebra construction cost — gyro frames
/// arrive at ~200 Hz, well below the threshold where this would matter
/// on the Jetson, but the explicit branch keeps the intent obvious
/// in flame graphs.
#[inline]
pub(crate) fn rotate_vec_by_quat_xyzw(v: [f32; 3], q_sensor_body_xyzw: [f32; 4]) -> [f32; 3] {
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
        // A bridge that has not read the chip yet sends an all-zero
        // quaternion. We must not propagate (0, 0, 0, 0) downstream.
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
        // sensor axes and check we land on the documented body axes —
        // the same axis mapping the
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
