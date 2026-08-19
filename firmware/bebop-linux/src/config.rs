//! Robot configuration — joint definitions, motor specs, safety limits.
//!
//! The configuration is loaded at startup from a YAML file via
//! [`RobotConfig::from_yaml`]. There is no longer a hardcoded fallback
//! preset — running without a config is an error, by design, so that the
//! safety limits in the YAML are always the source of truth.

use anyhow::{anyhow, Context, Result};
use serde::Deserialize;
use std::collections::HashSet;
use std::path::Path;

// ===========================================================================
// Control loop / observation / scaling — kept in code (training-coupled).
// ===========================================================================

/// Control loop timing configuration.
///
/// NOTE: the authoritative policy/control-loop rate is `main.rs`'s
/// `TICK_PERIOD` (10 ms = 100 Hz), which is what actually drives
/// `PolicyRunner::tick`. It MUST match the sim control rate
/// (`exp_standing.py`: `sim.dt 0.005 × decimation 2 = 100 Hz`). A stale
/// `POLICY_RATE_HZ = 50` constant used to live here and was never wired to
/// the real loop; it was removed so nothing can silently re-introduce a
/// half-rate control loop (which would desync the slew limit + action delay
/// the policy was trained against).
pub mod timing {
    pub const FEEDBACK_PUBLISH_HZ: u64 = 100;
    pub const WATCHDOG_TIMEOUT_MS: u64 = 500;
    pub const UDP_PORT: u16 = 10000;
}

/// Observation and action dimensions (must match training!).
///
/// Bebop V2 (humanoid, 8 leg joints, no wheels). Observation layout
/// matches `sim/bebop_training/envs/bebop_v2_base_cfg.py::PolicyCfg`
/// declaration order:
///
/// ```text
///   [ 0.. 3)  base_ang_vel       (rad/s, body frame)
///   [ 3.. 6)  projected_gravity  (unit vector in body frame)
///   [ 6..14)  joint_pos_rel      (rad, JOINT_NAMES order)
///   [14..22)  joint_vel_rel      (rad/s, JOINT_NAMES order)
///   [22..46)  last_action        (raw NN output, 24 dims: pos | kp | kd)
///   [46..49)  velocity_commands  (vx, vy, wz)
/// ```
///
/// NOTE: there is deliberately **no** `base_lin_vel` channel. Bebop V2 has
/// no wheel odometry / VIO, so the real robot could only ever feed zeros;
/// training against the sim's privileged ground-truth velocity was a
/// sim-to-real gap, so it was removed from the observation entirely.
///
/// Action layout: 24 floats. MIT-mode variable impedance, 3 channels
/// per joint in JOINT_NAMES order:
///
/// - `[ 0.. 8)` raw position commands -> `target = default + SCALE_ACTION * clamp(raw, -1, 1)`
/// - `[ 8..16)` raw kp commands       -> affine [-1, 1] to each joint's `(kp_min, kp_max)`
/// - `[16..24)` raw kd commands       -> affine [-1, 1] to each joint's `(kd_min, kd_max)`
///
/// Per-joint kp/kd clamps are loaded from the YAML
/// `policy_gain_clamps` block — see [`PolicyGainClamps`]. They MUST
/// mirror `POLICY_KP_MIN/MAX` and `POLICY_KD_MIN/MAX` in the sim
/// config.
pub mod dims {
    pub const OBS_DIM: usize = 49;
    pub const ACTION_DIM: usize = 24;
    pub const HISTORY_STEPS: usize = 1;
    pub const TOTAL_OBS_DIM: usize = OBS_DIM * HISTORY_STEPS;
}

/// Scaling factors used by [`crate::observation`] (must match training!).
pub mod scales {
    pub const SCALE_LIN_VEL: f32 = 1.0;
    pub const SCALE_ANG_VEL: f32 = 1.0;
    pub const SCALE_DOF_POS: f32 = 1.0;
    pub const SCALE_DOF_VEL: f32 = 1.0;
    /// Action -> position-target gain. MUST match the ``pos_scale``
    /// field on the sim's ``VariableImpedanceJointActionCfg`` in
    /// ``sim/bebop_training/experiments/exp_standing.py``. The policy
    /// learns the mapping ``target = default + SCALE_ACTION * raw``
    /// during training; if this constant differs from the sim value
    /// the deployed commands silently mis-scale (a too-large value
    /// here pushes the joints past their safe envelope on every
    /// command; a too-small value makes the policy weak / unable to
    /// reach its trained pose).
    ///
    /// 0.5 means raw=±1.0 maps to ±0.5 rad per joint -- enough range
    /// for standing and slow walking, well inside every joint's
    /// hard safety limit. Raise this (and retrain) when transitioning
    /// to highly dynamic motions that need more per-tick reach.
    pub const SCALE_ACTION: f32 = 0.5;

    pub const CLIP_LIN_VEL: f32 = 3.0;
    pub const CLIP_ANG_VEL: f32 = 10.0;
    pub const CLIP_DOF_VEL: f32 = 15.0;
}

// ===========================================================================
// Motor model specs — used for uint16 ↔ float scaling on the wire.
// ===========================================================================

#[derive(Debug, Clone, Copy)]
pub struct RobstrideSpecs {
    pub torque_min: f32,
    pub torque_max: f32,
    pub velocity_min: f32,
    pub velocity_max: f32,
    pub kp_max: f32,
    pub kd_max: f32,
}

impl RobstrideSpecs {
    pub const RS01: Self = Self {
        torque_min: -12.0,
        torque_max: 12.0,
        velocity_min: -45.0,
        velocity_max: 45.0,
        kp_max: 500.0,
        kd_max: 5.0,
    };

    pub const RS02: Self = Self {
        torque_min: -17.0,
        torque_max: 17.0,
        velocity_min: -30.0,
        velocity_max: 30.0,
        kp_max: 500.0,
        kd_max: 5.0,
    };

    pub const RS03: Self = Self {
        torque_min: -60.0,
        torque_max: 60.0,
        velocity_min: -20.0,
        velocity_max: 20.0,
        kp_max: 5000.0,
        kd_max: 100.0,
    };

    pub const RS04: Self = Self {
        torque_min: -120.0,
        torque_max: 120.0,
        velocity_min: -15.0,
        velocity_max: 15.0,
        kp_max: 5000.0,
        kd_max: 100.0,
    };
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RobstrideModel {
    RS01,
    RS02,
    RS03,
    RS04,
}

impl RobstrideModel {
    pub fn specs(&self) -> RobstrideSpecs {
        match self {
            RobstrideModel::RS01 => RobstrideSpecs::RS01,
            RobstrideModel::RS02 => RobstrideSpecs::RS02,
            RobstrideModel::RS03 => RobstrideSpecs::RS03,
            RobstrideModel::RS04 => RobstrideSpecs::RS04,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            RobstrideModel::RS01 => "RS01",
            RobstrideModel::RS02 => "RS02",
            RobstrideModel::RS03 => "RS03",
            RobstrideModel::RS04 => "RS04",
        }
    }
}

impl std::str::FromStr for RobstrideModel {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        match s.to_ascii_uppercase().as_str() {
            "RS01" => Ok(Self::RS01),
            "RS02" => Ok(Self::RS02),
            "RS03" => Ok(Self::RS03),
            "RS04" => Ok(Self::RS04),
            other => Err(anyhow!(
                "unknown Robstride model {other:?} (expected RS01..RS04)"
            )),
        }
    }
}

// ===========================================================================
// Per-joint runtime types
// ===========================================================================

/// Hard safety limits for a joint. Enforced both on outgoing setpoints and
/// on every incoming feedback frame; breach latches E-STOP.
#[derive(Debug, Clone, Copy)]
pub struct SafetyLimits {
    pub pos_min: f32,
    pub pos_max: f32,
    pub vel_max: f32,
    pub tau_max: f32,
    pub temp_max: f32,
    pub feedback_timeout_ms: f32,
}

impl SafetyLimits {
    pub const CONSERVATIVE: Self = Self {
        pos_min: -0.5,
        pos_max: 0.5,
        vel_max: 2.0,
        tau_max: 8.0,
        temp_max: 65.0,
        feedback_timeout_ms: 100.0,
    };
}

#[derive(Debug, Clone, Copy)]
pub struct Gains {
    pub kp: f32,
    pub kd: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct SlewParams {
    pub max_pos_step_per_tick: f32,
    pub arm_ramp_s: f32,
    pub abort_ramp_s: f32,
}

impl SlewParams {
    pub const DEFAULT: Self = Self {
        max_pos_step_per_tick: 0.005,
        arm_ramp_s: 0.5,
        abort_ramp_s: 0.3,
    };
}

/// Global policy action-decode parameters (`defaults.policy` in the YAML).
/// These MUST mirror the sim action term
/// (`VariableImpedanceJointActionCfg` in
/// `sim/bebop_training/envs/bebop_v2_actions.py`) — the trained policy
/// baked in whatever bandwidth these values allow.
#[derive(Debug, Clone, Copy)]
pub struct PolicyConfig {
    /// First-order low-pass time constant (seconds) applied to the decoded
    /// kp/kd channels every 100 Hz policy tick:
    /// `g += alpha * (cmd - g)`, `alpha = 1 - exp(-tick_dt / tau)`.
    /// Hard anti-snap guarantee on the gain channels (the position
    /// channel's analog is the slew clamp): tick-rate gain flipping is
    /// attenuated ~16x at tau = 0.08 while impedance shifts slower than
    /// ~2 Hz pass through. `0` disables the filter — do NOT deploy
    /// disabled against an EMA-trained policy. See
    /// [`crate::observation::GainEma`].
    pub gain_ema_tau_s: f32,
}

impl PolicyConfig {
    pub const DEFAULT: Self = Self {
        // Mirrors the sim ActionsCfg `gain_ema_tau_s` (round 6, Jul 22
        // 2026). Explicitly set in bebop_v2.yaml; the fallback only
        // guards older configs.
        gain_ema_tau_s: 0.08,
    };
}

/// Per-joint bounds on the policy-emitted kp / kd values for the
/// MIT-mode variable-impedance action. The decode path affine-maps each
/// raw `[-1, 1]` channel into `(min, max)` for its joint. Values MUST
/// mirror `POLICY_KP_MIN/MAX` and `POLICY_KD_MIN/MAX` in
/// `sim/bebop_training/envs/bebop_v2_base_cfg.py`.
#[derive(Debug, Clone, Copy)]
pub struct PolicyGainClamps {
    pub kp_min: f32,
    pub kp_max: f32,
    pub kd_min: f32,
    pub kd_max: f32,
}

impl PolicyGainClamps {
    /// Conservative fallback if YAML omits both defaults and per-joint
    /// overrides. Matches the most-conservative joint in the Bebop V2
    /// envelope (foot kd_min).
    pub const FALLBACK: Self = Self {
        kp_min: 5.0,
        kp_max: 100.0,
        kd_min: 0.5,
        kd_max: 5.0,
    };
}

#[derive(Debug, Clone)]
pub struct JointConfig {
    pub name: String,
    pub index: usize,
    pub can_id: u8,
    pub can_bus: String,
    pub model: RobstrideModel,
    pub hard_limits: SafetyLimits,
    pub hold_gains: Gains,
    pub test_gains: Gains,
    pub slew: SlewParams,
    pub default_position: f32,
    /// Per-joint policy gain clamps (kp / kd ranges). The MIT-mode
    /// policy emits raw kp / kd in [-1, 1]; the decode path maps each
    /// to this joint's range before TX.
    pub policy_gain_clamps: PolicyGainClamps,
    /// Sign that converts the motor/encoder rotation direction into the
    /// robot (URDF / sim) joint convention. Either `+1.0` (motor and
    /// URDF agree, the default) or `-1.0` (the motor is mounted / zeroed
    /// with the opposite polarity, so its positive rotation is the URDF's
    /// negative rotation).
    ///
    /// The supervisor applies this at the motor⇄robot boundary: every
    /// incoming feedback `position` / `velocity` / `torque` is multiplied
    /// by `direction` before it reaches any consumer (policy observation,
    /// `/joint_states` log, telemetry, DialIn, limit checks), and every
    /// outgoing position / velocity / torque setpoint is multiplied by it
    /// again on the way back to the wire. So everything above
    /// [`crate::robstride`] speaks the URDF convention and the trained
    /// policy sees the same joint signs it did in sim.
    ///
    /// NOTE: `hard_limits.pos_min` / `pos_max` are interpreted in the
    /// robot frame. They're symmetric for the joints we currently flip
    /// (hip abduction, ±1.0), so the sign is harmless there; if you ever
    /// flip a joint with an *asymmetric* position envelope, express that
    /// envelope in robot-frame coordinates.
    pub direction: f32,
}

/// Configuration for a single ODrive BotWheel (velocity-controlled,
/// continuous rotation). Unlike [`JointConfig`] a wheel has no position
/// envelope, no kp/kd — the ODrive runs its own velocity loop and we
/// command a `rev/s` setpoint with an optional torque feed-forward.
#[derive(Debug, Clone)]
pub struct WheelConfig {
    pub name: String,
    pub can_interface: String,
    /// ODrive `node_id` (0..=63; distinct from Robstride's 1..=255).
    pub node_id: u8,
    /// Motor→robot rotation sign (`+1` / `-1`), mirroring
    /// [`JointConfig::direction`]. Applied at the wheel⇄robot boundary so
    /// feedback and setpoints speak the URDF/sim forward convention.
    pub direction: f32,
    /// Wheel angular-velocity hard limit (rad/s). E-STOP breaker.
    pub vel_max: f32,
    /// Optional torque feed-forward clamp (Nm). Applies to the
    /// `Input_Torque_FF` field on `Set_Input_Vel`; currently the policy
    /// path leaves it at 0.
    pub torque_limit: f32,
    /// E-STOP if no encoder/heartbeat arrives within this many ms.
    pub feedback_timeout_ms: f32,
}

/// Differential-drive geometry + wheel naming. The twist→wheel and
/// wheel→odometry transforms in [`crate::drive`] key off this.
#[derive(Debug, Clone)]
pub struct DiffDriveConfig {
    /// Name of the left wheel (must match a [`WheelConfig::name`]).
    pub left_wheel: String,
    /// Name of the right wheel (must match a [`WheelConfig::name`]).
    pub right_wheel: String,
    /// Wheel radius in metres.
    pub wheel_radius: f32,
    /// Half the left-right wheel spacing (metres). "Track" here is the
    /// *half*-track: `omega = (v_r - v_l) / (2 * half_track)`.
    pub half_track: f32,
}

#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub bind_addr: String,
    pub telemetry_default_hz: u32,
    pub telemetry_max_hz: u32,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind_addr: "0.0.0.0:9090".into(),
            telemetry_default_hz: 30,
            telemetry_max_hz: 100,
        }
    }
}

/// Power-board (Robstride PowerBoard) configuration.
///
/// Optional: when present, a 1 Hz monitor task polls the board for its
/// status frame (battery voltage, motor voltage, board temp, fault word)
/// and republishes the result through `Snapshot.power` / telemetry frames.
///
/// Battery chemistry parameters are independent of the board itself and
/// are used purely for the operator-facing "fuel gauge" (state-of-charge
/// linear-interp from voltage). The defaults applied when YAML omits these
/// fields are 13 cells × 4.20 V / 3.00 V (a generic Li-ion NMC pack);
/// Bebop V2 and the wheeled chassis both ship with the 13s2p NMC pack,
/// see `bebop_v2.yaml` / `bebop_wheeled.yaml` for the matching 4.20 V /
/// 3.00 V endpoints.
#[derive(Debug, Clone)]
pub struct PowerBoardConfig {
    pub can_interface: String,
    pub power_id: u8,
    pub poll_interval_ms: u64,
    pub battery_cells: u32,
    pub cell_full_voltage: f32,
    pub cell_empty_voltage: f32,
}

impl PowerBoardConfig {
    /// Pack-level full-charge voltage (cell_full × cell_count).
    pub fn pack_full_voltage(&self) -> f32 {
        self.cell_full_voltage * self.battery_cells as f32
    }

    /// Pack-level empty / cutoff voltage (cell_empty × cell_count).
    pub fn pack_empty_voltage(&self) -> f32 {
        self.cell_empty_voltage * self.battery_cells as f32
    }

    /// Crude linear state-of-charge in percent from a measured pack voltage.
    /// Returns `None` for negative or NaN inputs (likely a stale / missing
    /// reading) so the UI can render "—" instead of a fake 0 %.
    pub fn estimate_soc_pct(&self, pack_voltage_v: f32) -> Option<f32> {
        if !pack_voltage_v.is_finite() || pack_voltage_v <= 0.0 {
            return None;
        }
        let full = self.pack_full_voltage();
        let empty = self.pack_empty_voltage();
        if full <= empty {
            return None;
        }
        let soc = (pack_voltage_v - empty) / (full - empty) * 100.0;
        Some(soc.clamp(0.0, 100.0))
    }
}

#[derive(Debug, Clone)]
pub struct RobotConfig {
    pub joints: Vec<JointConfig>,
    /// ODrive wheel actuators (empty on the legged humanoid). Mutually
    /// exclusive with `joints` in practice — see `drive` below.
    pub wheels: Vec<WheelConfig>,
    /// Differential-drive geometry; `Some` on a wheeled chassis, `None`
    /// on the humanoid. References wheel names in `wheels`.
    pub drive: Option<DiffDriveConfig>,
    pub can_interfaces: Vec<String>,
    pub server: ServerConfig,
    pub power: Option<PowerBoardConfig>,
    /// Global policy action-decode parameters (`defaults.policy`).
    pub policy: PolicyConfig,
    /// When set, [`crate::imu`] streams fused orientation into the shared
    /// state consumed by the WS telemetry pump. Policy observations still
    /// use synthetic IMU until explicitly wired.
    pub imu: Option<ImuConfig>,
}

/// Optional BNO080/BNO085 SPI reader (see `imu:` in the robot YAML).
///
/// The BNO talks SHTP over SPI (mode 3, ≤ 3 MHz) with two host-side
/// GPIOs: an active-low **INT** (`HINTN`) so the host knows when the
/// chip has a packet ready, and an active-low **RST** the host pulses
/// at boot to bring the chip into a clean SPI-mode session. See
/// `config/bebop_v2.yaml`'s `imu:` block for the canonical Bebop V2
/// pinout and a wiring diagram.
/// Where the runtime gets its BNO085 data from.
///
/// Both backends fill the same [`crate::imu::ImuShared`] snapshot with a
/// body-frame quaternion + gyro, so they are interchangeable from the
/// policy runner / telemetry pump's point of view. The chassis `mount:`
/// rotation is applied identically in both paths.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImuSource {
    /// Read the BNO085 directly over the Jetson's own SPI bus + GPIOs
    /// (the historical path; see [`crate::imu::spawn_imu_thread`]).
    Spi,
    /// Read pre-fused frames streamed by the Teensy over USB serial
    /// (see [`crate::imu_serial::spawn_imu_serial_thread`] and the Teensy
    /// `imu_bridge` firmware). The BNO is wired to the Teensy, not the
    /// Jetson, so the SPI/INT/RST fields are unused.
    Serial,
}

impl std::str::FromStr for ImuSource {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "spi" => Ok(Self::Spi),
            "serial" | "usb" | "teensy" => Ok(Self::Serial),
            other => Err(anyhow!(
                "unknown imu.source {other:?} (expected \"spi\" or \"serial\")"
            )),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ImuConfig {
    /// Which backend feeds the shared IMU snapshot.
    pub source: ImuSource,
    /// Serial device the Teensy `imu_bridge` enumerates as, e.g.
    /// `/dev/ttyACM0`. Only meaningful when `source == Serial`.
    pub serial_device: String,
    /// SPI character device — e.g. `/dev/spidev0.0` for the Jetson
    /// header pins 19/21/23/24 (`SPI0_*` in the pinout). Only meaningful
    /// when `source == Spi`.
    pub spi_device: String,
    /// GPIO chip (e.g. `gpiochip0`) hosting the BNO `INT`/`HINTN` line.
    pub int_chip: String,
    /// Line offset within `int_chip`. Look up with `gpioinfo`; for
    /// Jetson Orin Nano this is the gpiochip line that backs the
    /// physical header pin INT is wired to (see the YAML pinout table).
    pub int_line: u32,
    /// GPIO chip hosting the BNO `RST`/`RSTN` line.
    pub rst_chip: String,
    /// Line offset within `rst_chip` for `RST`.
    pub rst_line: u32,
    /// SH-2 sensor-report cadence hint, in milliseconds. Lower = more
    /// samples/sec; bounded by the chip's gyro rate (1 kHz) and the SPI
    /// throughput (see `spi_max_speed_hz`). 5 ms = 200 Hz; 50 ms = 20 Hz.
    pub rotation_vector_period_ms: u16,
    /// SPI bus clock (Hz) for the BNO08x. The chip is rated to 3 MHz;
    /// 80 kHz (the driver's historical default) cannot carry high report
    /// rates. Defaults to 1 MHz, which sustains a 200 Hz RV + gyro pair
    /// with margin. Lower it if long / unshielded jumpers make reads flaky.
    pub spi_max_speed_hz: u32,
    /// Constant **sensor-to-body** rotation, stored as a unit quaternion
    /// in XYZW order. Every raw BNO reading is post-multiplied by this
    /// before publishing, so downstream consumers always see a body-frame
    /// (REP-103 / FLU) orientation regardless of how the chip is glued
    /// to the chassis.
    ///
    /// Concretely, if the BNO reports `q_world_sensor`, we publish
    /// `q_world_body = q_world_sensor · q_sensor_body`, where
    /// `q_sensor_body` is the rotation that takes a body-frame vector
    /// and expresses it in sensor coordinates.
    ///
    /// Identity `(0, 0, 0, 1)` when the YAML omits `imu.mount`, so
    /// pre-existing configs behave exactly as before.
    pub mount_quat_sensor_body: [f32; 4],
}

impl ImuConfig {
    /// XYZW identity — sensor frame and body frame coincide.
    pub const IDENTITY_QUAT: [f32; 4] = [0.0, 0.0, 0.0, 1.0];
}

impl RobotConfig {
    /// Load and validate a v2 humanoid config from a YAML file.
    ///
    /// Validates:
    /// - Each `model` parses as a known [`RobstrideModel`].
    /// - `motor_id` is in `1..=255`.
    /// - `pos_min < pos_max`, `vel_max > 0`, etc.
    /// - No two joints share `(can_interface, motor_id)`.
    /// - Joint names are unique.
    ///
    /// Joint order in the resulting `Vec<JointConfig>` matches the YAML
    /// insertion order (preserved by `serde_yaml::Mapping`).
    pub fn from_yaml<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("read config file {}", path.display()))?;
        let raw: RawConfig = serde_yaml::from_str(&text)
            .with_context(|| format!("parse YAML from {}", path.display()))?;

        let defaults = raw.defaults.unwrap_or_default();
        let mut joints = Vec::new();
        let mut seen_names: HashSet<String> = HashSet::new();
        let mut seen_pairs: HashSet<(String, u8)> = HashSet::new();
        let mut interfaces: HashSet<String> = HashSet::new();

        // serde_yaml::Mapping preserves YAML insertion order, which is what
        // we want so that `index` matches the file ordering.
        for (key, value) in raw.joints.iter() {
            let name = key
                .as_str()
                .ok_or_else(|| anyhow!("joint key must be a string"))?
                .to_string();
            let raw_joint: RawJoint = serde_yaml::from_value(value.clone())
                .with_context(|| format!("parse joint {name:?}"))?;

            if !seen_names.insert(name.clone()) {
                return Err(anyhow!("duplicate joint name {name:?} in config"));
            }

            let can_bus = raw_joint
                .can_interface
                .as_ref()
                .cloned()
                .ok_or_else(|| anyhow!("joint {name:?}: missing can_interface"))?;

            let motor_id_u32 = raw_joint
                .motor_id
                .ok_or_else(|| anyhow!("joint {name:?}: missing motor_id"))?;
            if !(1..=0xFF).contains(&motor_id_u32) {
                return Err(anyhow!(
                    "joint {name:?}: motor_id {motor_id_u32} out of range [1, 255]"
                ));
            }
            let motor_id = motor_id_u32 as u8;
            if !seen_pairs.insert((can_bus.clone(), motor_id)) {
                return Err(anyhow!(
                    "joint {name:?}: duplicate (can_interface, motor_id) = ({can_bus:?}, {motor_id})"
                ));
            }
            interfaces.insert(can_bus.clone());

            let model_str = raw_joint
                .model
                .as_ref()
                .ok_or_else(|| anyhow!("joint {name:?}: missing model"))?;
            let model: RobstrideModel = model_str
                .parse()
                .with_context(|| format!("joint {name:?}: parse model"))?;

            let hard_limits = merge_limits(
                defaults.hard_limits.as_ref(),
                raw_joint.hard_limits.as_ref(),
            );

            if hard_limits.pos_min >= hard_limits.pos_max {
                return Err(anyhow!(
                    "joint {name:?}: pos_min ({}) must be < pos_max ({})",
                    hard_limits.pos_min,
                    hard_limits.pos_max
                ));
            }
            if hard_limits.vel_max <= 0.0 || hard_limits.tau_max <= 0.0 {
                return Err(anyhow!("joint {name:?}: vel_max and tau_max must be > 0"));
            }
            if hard_limits.feedback_timeout_ms <= 0.0 {
                return Err(anyhow!("joint {name:?}: feedback_timeout_ms must be > 0"));
            }

            let hold_gains = merge_gains(
                defaults.hold_gains.as_ref(),
                raw_joint.hold_gains.as_ref(),
                Gains { kp: 5.0, kd: 1.0 },
            );
            let test_gains = merge_gains(
                defaults.test_gains.as_ref(),
                raw_joint.test_gains.as_ref(),
                Gains { kp: 30.0, kd: 2.0 },
            );
            let slew = merge_slew(defaults.slew.as_ref(), raw_joint.slew.as_ref());

            let policy_gain_clamps = merge_policy_gain_clamps(
                defaults.policy_gain_clamps.as_ref(),
                raw_joint.policy_gain_clamps.as_ref(),
            );
            validate_policy_gain_clamps(&name, &model, &policy_gain_clamps)?;

            let direction = raw_joint
                .direction
                .or(defaults.direction)
                .unwrap_or(1.0);
            validate_direction(&name, direction)?;

            let index = joints.len();
            joints.push(JointConfig {
                name,
                index,
                can_id: motor_id,
                can_bus,
                model,
                hard_limits,
                hold_gains,
                test_gains,
                slew,
                default_position: 0.0,
                policy_gain_clamps,
                direction,
            });
        }

        // ----- Wheels (ODrive BotWheel) -----------------------------------
        // Parse the optional `wheels:` mapping. A wheeled chassis has
        // wheels (and usually no `joints`); the humanoid has `joints` and
        // no `wheels`. At least one of the two must be present.
        let mut wheels = Vec::new();
        let mut seen_wheel_names: HashSet<String> = HashSet::new();
        let mut seen_wheel_pairs: HashSet<(String, u8)> = HashSet::new();
        for (key, value) in raw.wheels.iter() {
            let name = key
                .as_str()
                .ok_or_else(|| anyhow!("wheel key must be a string"))?
                .to_string();
            let raw_wheel: RawWheel = serde_yaml::from_value(value.clone())
                .with_context(|| format!("parse wheel {name:?}"))?;

            if !seen_wheel_names.insert(name.clone()) {
                return Err(anyhow!("duplicate wheel name {name:?} in config"));
            }

            let can_bus = raw_wheel
                .can_interface
                .as_ref()
                .cloned()
                .ok_or_else(|| anyhow!("wheel {name:?}: missing can_interface"))?;

            let node_id = raw_wheel
                .node_id
                .ok_or_else(|| anyhow!("wheel {name:?}: missing node_id"))?;
            if node_id > 0x3F {
                return Err(anyhow!(
                    "wheel {name:?}: node_id {node_id} out of range [0, 63]"
                ));
            }
            if !seen_wheel_pairs.insert((can_bus.clone(), node_id as u8)) {
                return Err(anyhow!(
                    "wheel {name:?}: duplicate (can_interface, node_id) = ({can_bus:?}, {node_id})"
                ));
            }
            interfaces.insert(can_bus.clone());

            let vel_max = raw_wheel.vel_max.ok_or_else(|| {
                anyhow!("wheel {name:?}: missing vel_max (rad/s)")
            })?;
            if vel_max <= 0.0 {
                return Err(anyhow!("wheel {name:?}: vel_max must be > 0"));
            }
            let torque_limit = raw_wheel.torque_limit.unwrap_or(0.0);
            let feedback_timeout_ms = raw_wheel.feedback_timeout_ms.unwrap_or(100.0);
            if feedback_timeout_ms <= 0.0 {
                return Err(anyhow!(
                    "wheel {name:?}: feedback_timeout_ms must be > 0"
                ));
            }
            let direction = raw_wheel.direction.unwrap_or(1.0);
            validate_direction(&name, direction)?;

            wheels.push(WheelConfig {
                name,
                can_interface: can_bus,
                node_id: node_id as u8,
                direction,
                vel_max,
                torque_limit,
                feedback_timeout_ms,
            });
        }

        // ----- Differential drive -----------------------------------------
        // Optional; only meaningful when `wheels` is populated.
        let drive = raw
            .drive
            .map(|d| -> Result<DiffDriveConfig> {
                let left_wheel = d
                    .left_wheel
                    .ok_or_else(|| anyhow!("drive.left_wheel is required"))?;
                let right_wheel = d
                    .right_wheel
                    .ok_or_else(|| anyhow!("drive.right_wheel is required"))?;
                let wheel_radius = d
                    .wheel_radius
                    .ok_or_else(|| anyhow!("drive.wheel_radius is required"))?;
                let track_width = d
                    .track_width
                    .ok_or_else(|| anyhow!("drive.track_width is required"))?;
                if wheel_radius <= 0.0 {
                    return Err(anyhow!("drive.wheel_radius must be > 0"));
                }
                if track_width <= 0.0 {
                    return Err(anyhow!("drive.track_width must be > 0"));
                }
                Ok(DiffDriveConfig {
                    left_wheel,
                    right_wheel,
                    wheel_radius,
                    half_track: track_width / 2.0,
                })
            })
            .transpose()
            .context("invalid `drive:` section")?;

        if joints.is_empty() && wheels.is_empty() {
            return Err(anyhow!("config has neither joints nor wheels"));
        }
        if let Some(d) = &drive {
            for wn in [&d.left_wheel, &d.right_wheel] {
                if !wheels.iter().any(|w| &w.name == wn) {
                    return Err(anyhow!(
                        "drive references wheel {wn:?} not present in `wheels:`"
                    ));
                }
            }
        }

        let server = raw
            .server
            .map(|s| ServerConfig {
                bind_addr: s.bind_addr.unwrap_or_else(|| "0.0.0.0:9090".into()),
                telemetry_default_hz: s.telemetry_default_hz.unwrap_or(30),
                telemetry_max_hz: s.telemetry_max_hz.unwrap_or(100),
            })
            .unwrap_or_default();

        // Power-board section is optional; an absent / commented-out
        // `power:` block just means we won't poll for battery telemetry.
        let power = raw
            .power
            .map(|p| -> Result<PowerBoardConfig> {
                let can_interface = p
                    .can_interface
                    .ok_or_else(|| anyhow!("power.can_interface is required"))?;
                let power_id = p
                    .power_id
                    .map(|v| {
                        if !(0..=0xFF).contains(&v) {
                            Err(anyhow!("power.power_id {v} out of range [0, 255]"))
                        } else {
                            Ok(v as u8)
                        }
                    })
                    .transpose()?
                    .unwrap_or(crate::powerboard::DEFAULT_POWER_ID);
                let battery_cells = p.battery_cells.unwrap_or(13);
                if battery_cells == 0 {
                    return Err(anyhow!("power.battery_cells must be >= 1"));
                }
                let cell_full_voltage = p.cell_full_voltage.unwrap_or(4.20);
                let cell_empty_voltage = p.cell_empty_voltage.unwrap_or(3.00);
                if cell_full_voltage <= cell_empty_voltage {
                    return Err(anyhow!(
                        "power.cell_full_voltage ({cell_full_voltage}) must be > \
                         power.cell_empty_voltage ({cell_empty_voltage})"
                    ));
                }
                let poll_interval_ms = p.poll_interval_ms.unwrap_or(1000);
                if poll_interval_ms < 50 {
                    return Err(anyhow!(
                        "power.poll_interval_ms {poll_interval_ms} too aggressive (min 50)"
                    ));
                }
                interfaces.insert(can_interface.clone());
                Ok(PowerBoardConfig {
                    can_interface,
                    power_id,
                    poll_interval_ms,
                    battery_cells,
                    cell_full_voltage,
                    cell_empty_voltage,
                })
            })
            .transpose()
            .context("invalid `power:` section")?;

        let mut can_interfaces: Vec<String> = interfaces.into_iter().collect();
        can_interfaces.sort();

        let policy = {
            let tau = defaults
                .policy
                .as_ref()
                .and_then(|p| p.gain_ema_tau_s)
                .unwrap_or(PolicyConfig::DEFAULT.gain_ema_tau_s);
            if tau < 0.0 {
                return Err(anyhow!(
                    "defaults.policy.gain_ema_tau_s must be >= 0 (0 disables the \
                     filter); got {tau}"
                ));
            }
            PolicyConfig {
                gain_ema_tau_s: tau,
            }
        };

        let imu = raw
            .imu
            .map(|raw_imu| -> Result<ImuConfig> {
                // The defaults match the validated Bebop V2 pinout (see
                // `config/bebop_v2.yaml`): SPI device + INT/RST GPIOs
                // matching Jetson Orin Nano's `gpiochip0:144` (header
                // pin 7, PAC.06) and `gpiochip0:85` (header pin 15,
                // PN.01 / GPIO12). Override per-robot when the wiring differs.
                let source = match raw_imu.source.as_deref() {
                    Some(s) => s.parse::<ImuSource>()?,
                    None => ImuSource::Spi,
                };

                let serial_device = raw_imu
                    .serial_device
                    .unwrap_or_else(|| "/dev/ttyACM0".to_string());

                let spi_device = raw_imu
                    .spi_device
                    .unwrap_or_else(|| "/dev/spidev0.0".to_string());
                let int_chip = raw_imu.int_chip.unwrap_or_else(|| "gpiochip0".to_string());
                let rst_chip = raw_imu.rst_chip.unwrap_or_else(|| "gpiochip0".to_string());
                // INT/RST GPIO lines only matter for the SPI backend; the
                // serial bridge has the BNO wired to the Teensy. Require them
                // only when `source: spi`, otherwise default to 0 (unused).
                let (int_line, rst_line) = if source == ImuSource::Spi {
                    let int_line = raw_imu.int_line.ok_or_else(|| {
                        anyhow!("imu.int_line is required for source: spi \
                                 (GPIO line offset within `int_chip`)")
                    })?;
                    let rst_line = raw_imu.rst_line.ok_or_else(|| {
                        anyhow!("imu.rst_line is required for source: spi \
                                 (GPIO line offset within `rst_chip`)")
                    })?;
                    (int_line, rst_line)
                } else {
                    (raw_imu.int_line.unwrap_or(0), raw_imu.rst_line.unwrap_or(0))
                };
                let rotation_vector_period_ms = raw_imu.rotation_vector_period_ms.unwrap_or(50);
                if rotation_vector_period_ms == 0 {
                    return Err(anyhow!("imu.rotation_vector_period_ms must be >= 1"));
                }
                // Default 1 MHz: ~12× the historical 80 kHz, enough to carry
                // a 200 Hz RV + gyro pair with headroom, while staying well
                // under the BNO08x's 3 MHz SPI ceiling. Tune down for long /
                // unshielded jumper wiring if reads get flaky at speed.
                let spi_max_speed_hz = raw_imu.spi_max_speed_hz.unwrap_or(1_000_000);
                if spi_max_speed_hz == 0 || spi_max_speed_hz > 3_000_000 {
                    return Err(anyhow!(
                        "imu.spi_max_speed_hz must be in 1..=3_000_000 (BNO08x SPI ceiling \
                         is 3 MHz); got {spi_max_speed_hz}"
                    ));
                }
                let mount_quat_sensor_body = match raw_imu.mount {
                    Some(m) => build_mount_quat_sensor_body(&m)?,
                    None => ImuConfig::IDENTITY_QUAT,
                };
                Ok(ImuConfig {
                    source,
                    serial_device,
                    spi_device,
                    int_chip,
                    int_line,
                    rst_chip,
                    rst_line,
                    rotation_vector_period_ms,
                    spi_max_speed_hz,
                    mount_quat_sensor_body,
                })
            })
            .transpose()
            .context("invalid `imu:` section")?;

        Ok(Self {
            joints,
            wheels,
            drive,
            can_interfaces,
            server,
            power,
            policy,
            imu,
        })
    }

    pub fn get_joint(&self, name: &str) -> Option<&JointConfig> {
        self.joints.iter().find(|j| j.name == name)
    }

    pub fn get_joint_by_index(&self, index: usize) -> Option<&JointConfig> {
        self.joints.iter().find(|j| j.index == index)
    }

    pub fn num_joints(&self) -> usize {
        self.joints.len()
    }

    pub fn get_wheel(&self, name: &str) -> Option<&WheelConfig> {
        self.wheels.iter().find(|w| w.name == name)
    }

    pub fn num_wheels(&self) -> usize {
        self.wheels.len()
    }
}

// ===========================================================================
// Joint state / command (used by Robstride driver + observation builder).
// ===========================================================================

#[derive(Debug, Clone, Default)]
pub struct JointState {
    pub position: f32,
    pub velocity: f32,
    pub torque: f32,
    pub temperature: f32,
    pub is_enabled: bool,
    pub has_error: bool,
    pub error_code: u32,
    pub last_update_ms: u64,
}

#[derive(Debug, Clone, Default)]
pub struct JointCommand {
    pub position: f32,
    pub velocity: f32,
    pub torque: f32,
    pub kp: f32,
    pub kd: f32,
}

// ===========================================================================
// Raw YAML deserialization (private)
// ===========================================================================

#[derive(Debug, Default, Deserialize)]
struct RawConfig {
    defaults: Option<RawDefaults>,
    /// `serde_yaml::Mapping` preserves YAML insertion order, which we rely
    /// on so that `index` reflects the file ordering.
    #[serde(default)]
    joints: serde_yaml::Mapping,
    #[serde(default)]
    wheels: serde_yaml::Mapping,
    drive: Option<RawDrive>,
    server: Option<RawServer>,
    power: Option<RawPower>,
    imu: Option<RawImu>,
}

#[derive(Debug, Default, Deserialize)]
struct RawImu {
    /// `"spi"` (default) or `"serial"`. Selects the IMU backend.
    source: Option<String>,
    /// Serial device for the Teensy `imu_bridge`. Defaults to
    /// `/dev/ttyACM0`. Only used when `source: serial`.
    serial_device: Option<String>,
    /// SPI character device. Defaults to `/dev/spidev0.0` for the Jetson
    /// header pins 19/21/23/24 (`SPI0_*` in the pinout).
    spi_device: Option<String>,
    /// GPIO chip for `INT` (`HINTN`). Defaults to `gpiochip0`.
    int_chip: Option<String>,
    /// Line offset within `int_chip`. **Required.**
    int_line: Option<u32>,
    /// GPIO chip for `RST`. Defaults to `gpiochip0`.
    rst_chip: Option<String>,
    /// Line offset within `rst_chip`. **Required.**
    rst_line: Option<u32>,
    rotation_vector_period_ms: Option<u16>,
    /// SPI clock in Hz. Defaults to 1 MHz. See [`ImuConfig::spi_max_speed_hz`].
    spi_max_speed_hz: Option<u32>,
    mount: Option<RawMount>,
}

/// Per-axis mount remap: for each sensor axis, "which body axis does this
/// sensor axis point along when the robot is at rest at identity?"
///
/// Body frame is REP-103 / FLU (`+x = forward`, `+y = left`, `+z = up`).
/// Accepted values are `"+x"`, `"-x"`, `"+y"`, `"-y"`, `"+z"`, `"-z"`.
/// All three fields are required when `mount:` is present.
#[derive(Debug, Default, Deserialize)]
struct RawMount {
    sensor_x: Option<String>,
    sensor_y: Option<String>,
    sensor_z: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct RawDefaults {
    hard_limits: Option<RawLimits>,
    hold_gains: Option<RawGains>,
    test_gains: Option<RawGains>,
    slew: Option<RawSlew>,
    policy: Option<RawPolicy>,
    policy_gain_clamps: Option<RawPolicyGainClamps>,
    direction: Option<f32>,
}

#[derive(Debug, Default, Deserialize)]
struct RawJoint {
    can_interface: Option<String>,
    motor_id: Option<u32>,
    model: Option<String>,
    hard_limits: Option<RawLimits>,
    hold_gains: Option<RawGains>,
    test_gains: Option<RawGains>,
    slew: Option<RawSlew>,
    policy_gain_clamps: Option<RawPolicyGainClamps>,
    /// Motor→robot direction sign (`+1` / `-1`). See
    /// [`JointConfig::direction`]. Omitted ⇒ `+1` (default, motor agrees
    /// with the URDF). Inherits the `defaults.direction` block.
    direction: Option<f32>,
}

#[derive(Debug, Default, Deserialize)]
struct RawPolicyGainClamps {
    kp_min: Option<f32>,
    kp_max: Option<f32>,
    kd_min: Option<f32>,
    kd_max: Option<f32>,
}

#[derive(Debug, Default, Deserialize)]
struct RawLimits {
    pos_min: Option<f32>,
    pos_max: Option<f32>,
    vel_max: Option<f32>,
    tau_max: Option<f32>,
    temp_max: Option<f32>,
    feedback_timeout_ms: Option<f32>,
}

#[derive(Debug, Default, Deserialize)]
struct RawGains {
    kp: Option<f32>,
    kd: Option<f32>,
}

#[derive(Debug, Default, Deserialize)]
struct RawSlew {
    max_pos_step_per_tick: Option<f32>,
    arm_ramp_s: Option<f32>,
    abort_ramp_s: Option<f32>,
}

#[derive(Debug, Default, Deserialize)]
struct RawPolicy {
    gain_ema_tau_s: Option<f32>,
}

#[derive(Debug, Default, Deserialize)]
struct RawServer {
    bind_addr: Option<String>,
    telemetry_default_hz: Option<u32>,
    telemetry_max_hz: Option<u32>,
}

#[derive(Debug, Default, Deserialize)]
struct RawPower {
    can_interface: Option<String>,
    power_id: Option<u32>,
    poll_interval_ms: Option<u64>,
    battery_cells: Option<u32>,
    cell_full_voltage: Option<f32>,
    cell_empty_voltage: Option<f32>,
}

#[derive(Debug, Default, Deserialize)]
struct RawWheel {
    can_interface: Option<String>,
    node_id: Option<u32>,
    direction: Option<f32>,
    vel_max: Option<f32>,
    torque_limit: Option<f32>,
    feedback_timeout_ms: Option<f32>,
}

#[derive(Debug, Default, Deserialize)]
struct RawDrive {
    left_wheel: Option<String>,
    right_wheel: Option<String>,
    /// Left-right wheel separation (metres). The config expresses the full
    /// track; we store the half-track in [`DiffDriveConfig`].
    track_width: Option<f32>,
    wheel_radius: Option<f32>,
}

/// Parse one of `"+x"`, `"-x"`, `"+y"`, `"-y"`, `"+z"`, `"-z"` (a leading
/// `+` is optional) into the corresponding signed unit vector in body
/// frame. Whitespace is tolerated, casing is not significant.
fn parse_axis_remap(spec: &str, field: &str) -> Result<[f32; 3]> {
    let trimmed = spec.trim();
    let (sign, rest) = match trimmed.as_bytes().first() {
        Some(b'+') => (1.0_f32, &trimmed[1..]),
        Some(b'-') => (-1.0_f32, &trimmed[1..]),
        _ => (1.0_f32, trimmed),
    };
    let axis = match rest.trim().to_ascii_lowercase().as_str() {
        "x" => [1.0, 0.0, 0.0],
        "y" => [0.0, 1.0, 0.0],
        "z" => [0.0, 0.0, 1.0],
        other => {
            return Err(anyhow!(
                "imu.mount.{field}: invalid axis spec {other:?} \
                 (expected one of +x, -x, +y, -y, +z, -z)"
            ));
        }
    };
    Ok([axis[0] * sign, axis[1] * sign, axis[2] * sign])
}

/// Build the constant `q_sensor_body` rotation from a `mount:` block.
///
/// Each `sensor_*` field describes where that sensor axis points in body
/// frame. The columns of the implied 3×3 matrix are therefore the sensor
/// axes expressed in body coordinates (= `R_body_sensor`). We:
///
/// 1. Build `R_body_sensor` directly from the three columns.
/// 2. Reject anything that isn't a proper right-handed rotation
///    (`det != +1` — covers duplicate axes, parallel columns, and
///    accidental reflections).
/// 3. Invert and convert to a unit quaternion `q_sensor_body` in XYZW.
///
/// Used by `imu.rs` as a post-multiplier: `q_world_body = q_world_sensor *
/// q_sensor_body`.
fn build_mount_quat_sensor_body(raw: &RawMount) -> Result<[f32; 4]> {
    use nalgebra::{Matrix3, Rotation3, UnitQuaternion, Vector3};

    let sensor_x = raw
        .sensor_x
        .as_deref()
        .ok_or_else(|| anyhow!("imu.mount: missing sensor_x"))?;
    let sensor_y = raw
        .sensor_y
        .as_deref()
        .ok_or_else(|| anyhow!("imu.mount: missing sensor_y"))?;
    let sensor_z = raw
        .sensor_z
        .as_deref()
        .ok_or_else(|| anyhow!("imu.mount: missing sensor_z"))?;

    let cx = parse_axis_remap(sensor_x, "sensor_x")?;
    let cy = parse_axis_remap(sensor_y, "sensor_y")?;
    let cz = parse_axis_remap(sensor_z, "sensor_z")?;

    let r_body_sensor = Matrix3::from_columns(&[
        Vector3::new(cx[0], cx[1], cx[2]),
        Vector3::new(cy[0], cy[1], cy[2]),
        Vector3::new(cz[0], cz[1], cz[2]),
    ]);

    let det = r_body_sensor.determinant();
    if (det - 1.0).abs() > 1e-3 {
        return Err(anyhow!(
            "imu.mount: (sensor_x={sensor_x:?}, sensor_y={sensor_y:?}, sensor_z={sensor_z:?}) \
             does not form a proper right-handed rotation (det = {det:.3}). \
             Two axes likely collide or the handedness is flipped — pick a permutation \
             whose determinant is +1."
        ));
    }

    // Safe after the det check above: the columns are an orthonormal basis.
    let rot_b_s = Rotation3::from_matrix_unchecked(r_body_sensor);
    let q_sensor_body = UnitQuaternion::from_rotation_matrix(&rot_b_s)
        .inverse()
        .into_inner();
    // nalgebra::Quaternion stores `coords = [i, j, k, w]` (= XYZW), which is
    // exactly the convention `ImuShared` / `ImuState` expect.
    Ok([
        q_sensor_body.coords.x,
        q_sensor_body.coords.y,
        q_sensor_body.coords.z,
        q_sensor_body.coords.w,
    ])
}

fn merge_limits(defaults: Option<&RawLimits>, joint: Option<&RawLimits>) -> SafetyLimits {
    let pick = |get: fn(&RawLimits) -> Option<f32>, fallback: f32| {
        joint
            .and_then(get)
            .or_else(|| defaults.and_then(get))
            .unwrap_or(fallback)
    };
    SafetyLimits {
        pos_min: pick(|l| l.pos_min, SafetyLimits::CONSERVATIVE.pos_min),
        pos_max: pick(|l| l.pos_max, SafetyLimits::CONSERVATIVE.pos_max),
        vel_max: pick(|l| l.vel_max, SafetyLimits::CONSERVATIVE.vel_max),
        tau_max: pick(|l| l.tau_max, SafetyLimits::CONSERVATIVE.tau_max),
        temp_max: pick(|l| l.temp_max, SafetyLimits::CONSERVATIVE.temp_max),
        feedback_timeout_ms: pick(
            |l| l.feedback_timeout_ms,
            SafetyLimits::CONSERVATIVE.feedback_timeout_ms,
        ),
    }
}

fn merge_gains(defaults: Option<&RawGains>, joint: Option<&RawGains>, fallback: Gains) -> Gains {
    let kp = joint
        .and_then(|g| g.kp)
        .or_else(|| defaults.and_then(|g| g.kp))
        .unwrap_or(fallback.kp);
    let kd = joint
        .and_then(|g| g.kd)
        .or_else(|| defaults.and_then(|g| g.kd))
        .unwrap_or(fallback.kd);
    Gains { kp, kd }
}

fn merge_slew(defaults: Option<&RawSlew>, joint: Option<&RawSlew>) -> SlewParams {
    let pick = |get: fn(&RawSlew) -> Option<f32>, fallback: f32| {
        joint
            .and_then(get)
            .or_else(|| defaults.and_then(get))
            .unwrap_or(fallback)
    };
    SlewParams {
        max_pos_step_per_tick: pick(
            |s| s.max_pos_step_per_tick,
            SlewParams::DEFAULT.max_pos_step_per_tick,
        ),
        arm_ramp_s: pick(|s| s.arm_ramp_s, SlewParams::DEFAULT.arm_ramp_s),
        abort_ramp_s: pick(|s| s.abort_ramp_s, SlewParams::DEFAULT.abort_ramp_s),
    }
}

fn merge_policy_gain_clamps(
    defaults: Option<&RawPolicyGainClamps>,
    joint: Option<&RawPolicyGainClamps>,
) -> PolicyGainClamps {
    let pick = |get: fn(&RawPolicyGainClamps) -> Option<f32>, fallback: f32| {
        joint
            .and_then(get)
            .or_else(|| defaults.and_then(get))
            .unwrap_or(fallback)
    };
    PolicyGainClamps {
        kp_min: pick(|c| c.kp_min, PolicyGainClamps::FALLBACK.kp_min),
        kp_max: pick(|c| c.kp_max, PolicyGainClamps::FALLBACK.kp_max),
        kd_min: pick(|c| c.kd_min, PolicyGainClamps::FALLBACK.kd_min),
        kd_max: pick(|c| c.kd_max, PolicyGainClamps::FALLBACK.kd_max),
    }
}

/// Reject a `direction` that isn't exactly `+1` or `-1`. The sign is a
/// pure rotation-polarity flip; any other value would silently rescale
/// every position / velocity / torque crossing the motor⇄robot boundary.
fn validate_direction(joint_name: &str, direction: f32) -> Result<()> {
    if (direction - 1.0).abs() > 1e-6 && (direction + 1.0).abs() > 1e-6 {
        return Err(anyhow!(
            "actuator {joint_name:?}: direction must be +1 or -1 (got {direction})"
        ));
    }
    Ok(())
}

/// Reject clamps that don't form a valid range or that exceed the
/// motor model's electrical envelope. The supervisor uses the wire
/// scaler's full encoder range when packing the CAN frame, so writing
/// kp > `RobstrideSpecs::kp_max` would saturate at the encoder peak
/// silently; we'd rather fail-fast at boot.
fn validate_policy_gain_clamps(
    joint_name: &str,
    model: &RobstrideModel,
    clamps: &PolicyGainClamps,
) -> Result<()> {
    if !(clamps.kp_min.is_finite() && clamps.kp_max.is_finite()) {
        return Err(anyhow!(
            "joint {joint_name:?}: policy_gain_clamps kp_min/kp_max must be finite"
        ));
    }
    if !(clamps.kd_min.is_finite() && clamps.kd_max.is_finite()) {
        return Err(anyhow!(
            "joint {joint_name:?}: policy_gain_clamps kd_min/kd_max must be finite"
        ));
    }
    if clamps.kp_min < 0.0 || clamps.kd_min < 0.0 {
        return Err(anyhow!(
            "joint {joint_name:?}: policy_gain_clamps kp_min/kd_min must be >= 0 \
             (got kp_min={}, kd_min={})",
            clamps.kp_min,
            clamps.kd_min
        ));
    }
    if clamps.kp_min >= clamps.kp_max {
        return Err(anyhow!(
            "joint {joint_name:?}: policy_gain_clamps kp_min ({}) must be < kp_max ({})",
            clamps.kp_min,
            clamps.kp_max
        ));
    }
    if clamps.kd_min >= clamps.kd_max {
        return Err(anyhow!(
            "joint {joint_name:?}: policy_gain_clamps kd_min ({}) must be < kd_max ({})",
            clamps.kd_min,
            clamps.kd_max
        ));
    }
    let specs = model.specs();
    if clamps.kp_max > specs.kp_max {
        return Err(anyhow!(
            "joint {joint_name:?}: policy_gain_clamps kp_max ({}) exceeds {} encoder \
             ceiling ({}). Lower kp_max or move to a higher-spec motor model.",
            clamps.kp_max,
            model.as_str(),
            specs.kp_max
        ));
    }
    if clamps.kd_max > specs.kd_max {
        return Err(anyhow!(
            "joint {joint_name:?}: policy_gain_clamps kd_max ({}) exceeds {} encoder \
             ceiling ({}). Lower kd_max or move to a higher-spec motor model.",
            clamps.kd_max,
            model.as_str(),
            specs.kd_max
        ));
    }
    Ok(())
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod direction_tests {
    use super::*;
    use std::io::Write;
    use std::path::PathBuf;

    fn write_temp_config(body: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        let unique = format!(
            "bebop_dir_test_{}_{}.yaml",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        path.push(unique);
        let mut f = std::fs::File::create(&path).expect("create temp config");
        f.write_all(body.as_bytes()).expect("write temp config");
        path
    }

    const TWO_JOINTS: &str = r#"
joints:
  a_joint:
    can_interface: can0
    motor_id: 1
    model: RS03
    direction: -1
  b_joint:
    can_interface: can0
    motor_id: 2
    model: RS03
"#;

    #[test]
    fn direction_defaults_to_plus_one_and_parses_minus_one() {
        let path = write_temp_config(TWO_JOINTS);
        let cfg = RobotConfig::from_yaml(&path).expect("load config");
        let a = cfg.get_joint("a_joint").unwrap();
        let b = cfg.get_joint("b_joint").unwrap();
        assert_eq!(a.direction, -1.0, "explicit direction: -1 should parse");
        assert_eq!(b.direction, 1.0, "omitted direction should default to +1");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn invalid_direction_is_rejected() {
        let body = r#"
joints:
  a_joint:
    can_interface: can0
    motor_id: 1
    model: RS03
    direction: -2
"#;
        let path = write_temp_config(body);
        let err = RobotConfig::from_yaml(&path).unwrap_err();
        assert!(
            err.to_string().contains("direction must be +1 or -1")
                || format!("{err:#}").contains("direction must be +1 or -1"),
            "unexpected error: {err:#}"
        );
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn defaults_block_direction_is_inherited_and_overridden() {
        let body = r#"
defaults:
  direction: -1
joints:
  inherits_joint:
    can_interface: can0
    motor_id: 1
    model: RS03
  overrides_joint:
    can_interface: can0
    motor_id: 2
    model: RS03
    direction: 1
"#;
        let path = write_temp_config(body);
        let cfg = RobotConfig::from_yaml(&path).expect("load config");
        assert_eq!(cfg.get_joint("inherits_joint").unwrap().direction, -1.0);
        assert_eq!(cfg.get_joint("overrides_joint").unwrap().direction, 1.0);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn shipped_bebop_v2_yaml_flips_reversed_polarity_joints() {
        // Guards the actual deployed config: the hip-abduction, hip-flexion,
        // and knee-flexion motors are mounted/zeroed with reversed encoder
        // polarity vs the URDF/sim (confirmed by passive back-drive captures),
        // so they carry direction -1. Only the foot/ankle joints agree with
        // the URDF (direction +1).
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("config/bebop_v2.yaml");
        let cfg = RobotConfig::from_yaml(&path).expect("load shipped bebop_v2.yaml");
        for joint in &cfg.joints {
            let reversed = joint.name.starts_with("hip_abduction_")
                || joint.name.starts_with("hip_flexion_")
                || joint.name.starts_with("knee_flexion_");
            let expected = if reversed { -1.0 } else { 1.0 };
            assert_eq!(
                joint.direction, expected,
                "joint {} has unexpected direction {}",
                joint.name, joint.direction
            );
        }
    }

    #[test]
    fn shipped_bebop_wheeled_yaml_parses_power_block() {
        // Guards the deployed wheeled config: the chassis reuses the V2
        // power board (can4, addr 0xAA) and 13s NMC pack, and the parser
        // must auto-register both `can2` (wheels) and `can4` (power) in
        // the bus pool — the YAML deliberately carries no explicit
        // `can_interfaces:` list. Losing the `power:` block silently
        // hides the operator app's battery card (present=false).
        let path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("config/bebop_wheeled.yaml");
        let cfg = RobotConfig::from_yaml(&path).expect("load shipped bebop_wheeled.yaml");
        assert_eq!(cfg.num_joints(), 0);
        assert_eq!(cfg.num_wheels(), 2);
        assert!(cfg.drive.is_some());
        let power = cfg.power.as_ref().expect("power block present");
        assert_eq!(power.can_interface, "can4");
        assert_eq!(power.power_id, 170);
        assert_eq!(power.poll_interval_ms, 1000);
        assert_eq!(power.battery_cells, 13);
        assert!((power.cell_full_voltage - 4.20).abs() < 1e-6);
        assert!((power.cell_empty_voltage - 3.00).abs() < 1e-6);
        assert!(cfg.can_interfaces.contains(&"can2".to_string()));
        assert!(cfg.can_interfaces.contains(&"can4".to_string()));
    }
}

#[cfg(test)]
mod imu_mount_tests {
    use super::*;

    /// Apply `q_sensor_body` to a sensor-frame vector and return the body-
    /// frame components. Mirrors what `imu.rs` does at runtime (via the
    /// `q_world_body = q_world_sensor · q_sensor_body` post-multiply) but
    /// exposed here as a plain rotation so the assertions are obvious.
    fn rotate_sensor_to_body(q_sb_xyzw: [f32; 4], v_sensor: [f32; 3]) -> [f32; 3] {
        use nalgebra::{Quaternion, UnitQuaternion, Vector3};
        // q_sensor_body takes a body-frame vector → sensor coords, so the
        // inverse takes a sensor-frame vector → body coords.
        let q = UnitQuaternion::from_quaternion(Quaternion::new(
            q_sb_xyzw[3],
            q_sb_xyzw[0],
            q_sb_xyzw[1],
            q_sb_xyzw[2],
        ));
        let v = q.inverse() * Vector3::new(v_sensor[0], v_sensor[1], v_sensor[2]);
        [v.x, v.y, v.z]
    }

    fn approx(a: [f32; 3], b: [f32; 3]) -> bool {
        (a[0] - b[0]).abs() < 1e-5 && (a[1] - b[1]).abs() < 1e-5 && (a[2] - b[2]).abs() < 1e-5
    }

    #[test]
    fn identity_mount_is_default() {
        assert_eq!(ImuConfig::IDENTITY_QUAT, [0.0, 0.0, 0.0, 1.0]);
    }

    /// Sensor +X = right (body -Y), sensor +Y = forward (body +X),
    /// sensor +Z = up (body +Z) — the Bebop V2 mounting. Each sensor unit
    /// vector should map to the expected body unit vector.
    fn bebop_v2_mount() -> [f32; 4] {
        build_mount_quat_sensor_body(&RawMount {
            sensor_x: Some("-y".into()),
            sensor_y: Some("+x".into()),
            sensor_z: Some("+z".into()),
        })
        .expect("valid right-handed mount")
    }

    #[test]
    fn bebop_v2_mount_maps_sensor_axes_to_body_axes() {
        let q = bebop_v2_mount();
        assert!(
            approx(rotate_sensor_to_body(q, [1.0, 0.0, 0.0]), [0.0, -1.0, 0.0]),
            "sensor +X should map to body -Y (right)"
        );
        assert!(
            approx(rotate_sensor_to_body(q, [0.0, 1.0, 0.0]), [1.0, 0.0, 0.0]),
            "sensor +Y should map to body +X (forward)"
        );
        assert!(
            approx(rotate_sensor_to_body(q, [0.0, 0.0, 1.0]), [0.0, 0.0, 1.0]),
            "sensor +Z should map to body +Z (up)"
        );
    }

    #[test]
    fn rejects_reflection() {
        // sensor_x=+y, sensor_y=+x, sensor_z=+z is a permutation with
        // determinant -1 — a reflection, not a rotation.
        let err = build_mount_quat_sensor_body(&RawMount {
            sensor_x: Some("+y".into()),
            sensor_y: Some("+x".into()),
            sensor_z: Some("+z".into()),
        })
        .unwrap_err();
        assert!(
            err.to_string().contains("proper right-handed rotation"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn rejects_duplicate_axes() {
        let err = build_mount_quat_sensor_body(&RawMount {
            sensor_x: Some("+x".into()),
            sensor_y: Some("+x".into()),
            sensor_z: Some("+z".into()),
        })
        .unwrap_err();
        assert!(
            err.to_string().contains("proper right-handed rotation"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn rejects_unknown_axis_token() {
        let err = build_mount_quat_sensor_body(&RawMount {
            sensor_x: Some("+w".into()),
            sensor_y: Some("+x".into()),
            sensor_z: Some("+z".into()),
        })
        .unwrap_err();
        assert!(
            err.to_string().contains("invalid axis spec"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn missing_axis_fails_with_clear_error() {
        let err = build_mount_quat_sensor_body(&RawMount {
            sensor_x: None,
            sensor_y: Some("+x".into()),
            sensor_z: Some("+z".into()),
        })
        .unwrap_err();
        assert!(
            err.to_string().contains("missing sensor_x"),
            "unexpected error: {err}"
        );
    }
}

#[cfg(test)]
mod wheel_tests {
    use super::*;
    use std::io::Write;
    use std::path::PathBuf;

    fn write_temp_config(body: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        let unique = format!(
            "bebop_wheel_test_{}_{}.yaml",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        path.push(unique);
        let mut f = std::fs::File::create(&path).expect("create temp config");
        f.write_all(body.as_bytes()).expect("write temp config");
        path
    }

    const DIFF_DRIVE: &str = r#"
wheels:
  left:
    can_interface: can0
    node_id: 1
    vel_max: 20.0
  right:
    can_interface: can0
    node_id: 2
    direction: -1
    vel_max: 20.0
drive:
  left_wheel: left
  right_wheel: right
  wheel_radius: 0.05
  track_width: 0.30
"#;

    #[test]
    fn parses_wheels_without_joints() {
        let path = write_temp_config(DIFF_DRIVE);
        let cfg = RobotConfig::from_yaml(&path).expect("load wheeled config");
        assert_eq!(cfg.num_joints(), 0);
        assert_eq!(cfg.num_wheels(), 2);
        assert!(cfg.get_wheel("left").is_some());
        assert_eq!(cfg.get_wheel("right").unwrap().direction, -1.0);
        assert_eq!(cfg.get_wheel("left").unwrap().direction, 1.0);
        let d = cfg.drive.as_ref().expect("drive parsed");
        assert_eq!(d.left_wheel, "left");
        assert_eq!(d.right_wheel, "right");
        assert!((d.half_track - 0.15).abs() < 1e-6);
        assert!((d.wheel_radius - 0.05).abs() < 1e-6);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn rejects_config_with_neither_joints_nor_wheels() {
        let path = write_temp_config("server:\n  bind_addr: 0.0.0.0:9090\n");
        let err = RobotConfig::from_yaml(&path).unwrap_err();
        assert!(
            format!("{err:#}").contains("neither joints nor wheels"),
            "unexpected error: {err:#}"
        );
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn rejects_drive_reference_to_unknown_wheel() {
        let body = r#"
wheels:
  left:
    can_interface: can0
    node_id: 1
    vel_max: 20.0
drive:
  left_wheel: left
  right_wheel: missing
  wheel_radius: 0.05
  track_width: 0.30
"#;
        let path = write_temp_config(body);
        let err = RobotConfig::from_yaml(&path).unwrap_err();
        assert!(
            format!("{err:#}").contains("not present in `wheels:`"),
            "unexpected error: {err:#}"
        );
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn rejects_odrive_node_id_out_of_range() {
        let body = r#"
wheels:
  left:
    can_interface: can0
    node_id: 64
    vel_max: 20.0
"#;
        let path = write_temp_config(body);
        let err = RobotConfig::from_yaml(&path).unwrap_err();
        assert!(
            format!("{err:#}").contains("out of range [0, 63]"),
            "unexpected error: {err:#}"
        );
        std::fs::remove_file(&path).ok();
    }
}
