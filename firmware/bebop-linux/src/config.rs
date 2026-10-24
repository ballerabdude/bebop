//! Robot configuration — joints, wheels, drive geometry, power board, IMU
//! and server settings, plus the motor specs and safety limits they use.
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
}

impl SlewParams {
    pub const DEFAULT: Self = Self {
        max_pos_step_per_tick: 0.005,
    };
}

#[derive(Debug, Clone)]
pub struct JointConfig {
    pub name: String,
    pub can_id: u8,
    pub can_bus: String,
    pub model: RobstrideModel,
    pub hard_limits: SafetyLimits,
    pub hold_gains: Gains,
    pub test_gains: Gains,
    pub slew: SlewParams,
    /// Sign that converts the motor/encoder rotation direction into the
    /// robot (URDF / sim) joint convention. Either `+1.0` (motor and
    /// URDF agree, the default) or `-1.0` (the motor is mounted / zeroed
    /// with the opposite polarity, so its positive rotation is the URDF's
    /// negative rotation).
    ///
    /// The supervisor applies this at the motor⇄robot boundary: every
    /// incoming feedback `position` / `velocity` / `torque` is multiplied
    /// by `direction` before it reaches any consumer (telemetry, DialIn,
    /// limit checks), and every outgoing position / velocity / torque
    /// setpoint is multiplied by it again on the way back to the wire. So
    /// everything above [`crate::robstride`] speaks the URDF convention.
    ///
    /// NOTE: `hard_limits.pos_min` / `pos_max` are interpreted in the
    /// robot frame. If you flip a joint with an *asymmetric* position
    /// envelope, express that envelope in robot-frame coordinates.
    pub direction: f32,
}

/// Configuration for a single ODrive BotWheel (velocity-controlled,
/// continuous rotation). Unlike [`JointConfig`] a wheel has no position
/// envelope, no kp/kd — the ODrive runs its own velocity loop and we
/// command a `rev/s` setpoint.
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
    /// Wheel angular-velocity limit (rad/s). Every commanded wheel speed is
    /// clamped to it; the measured speed is not checked against it.
    pub vel_max: f32,
    /// E-STOP if an armed wheel goes this many ms without an encoder /
    /// heartbeat frame (once it has reported at least once).
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
    /// Operator-link freshness budget for the drive twist (ms). A
    /// non-zero twist that isn't refreshed within this window is
    /// zeroed by the supervisor watchdog (controlled stop — the
    /// operator went silent mid-drive and their stop never arrived).
    /// `0` disables the check. The operator app re-sends the current
    /// twist at ~10 Hz while driving, so 500 ms tolerates several
    /// dropped frames without halting a healthy session.
    pub operator_timeout_ms: u64,
    /// Operator-arbitration grace (ms): how long after the active
    /// operator's last drive command their assignment still blocks
    /// other clients' non-zero drive input. Once this window passes
    /// without input from the holder, anyone's next drive command
    /// takes over. `0` disables arbitration entirely (any drive input
    /// takes over immediately — last writer wins). Stops (zero twist)
    /// are always accepted from any client regardless.
    pub operator_grace_ms: u64,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind_addr: "0.0.0.0:9090".into(),
            telemetry_default_hz: 30,
            telemetry_max_hz: 100,
            operator_timeout_ms: 500,
            operator_grace_ms: 2000,
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
/// fields are 13 cells × 4.20 V / 3.00 V (a generic Li-ion NMC pack); the
/// wheeled chassis ships with a 13s2p NMC pack, see `bebop_wheeled.yaml`
/// for the matching 4.20 V / 3.00 V endpoints.
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
    /// Robstride MIT-mode joints (empty without a `joints:` block).
    pub joints: Vec<JointConfig>,
    /// ODrive wheel actuators (empty without a `wheels:` block).
    pub wheels: Vec<WheelConfig>,
    /// Differential-drive geometry; `None` without a `drive:` block.
    /// References wheel names in `wheels`.
    pub drive: Option<DiffDriveConfig>,
    pub can_interfaces: Vec<String>,
    pub server: ServerConfig,
    pub power: Option<PowerBoardConfig>,
    /// When set, [`crate::imu_serial`] streams the BNO085's body-frame
    /// orientation into the shared state the telemetry builders read.
    pub imu: Option<ImuConfig>,
}

/// Where the runtime gets its BNO085 data from (`imu.source` in the robot
/// YAML; `serial` when omitted).
///
/// The backend fills the shared [`crate::imu::ImuShared`] snapshot with a
/// body-frame quaternion + gyro, after applying the chassis `mount:`
/// rotation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImuSource {
    /// Read pre-fused frames streamed by the Teensy over USB serial
    /// (see [`crate::imu_serial::spawn_imu_serial_thread`] and the Teensy
    /// `teensy_bridge` firmware). The BNO is wired to the Teensy, not the
    /// Jetson.
    Serial,
}

impl std::str::FromStr for ImuSource {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "serial" | "usb" | "teensy" => Ok(Self::Serial),
            other => Err(anyhow!(
                "unknown imu.source {other:?} (expected \"serial\")"
            )),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ImuConfig {
    /// Which backend feeds the shared IMU snapshot.
    pub source: ImuSource,
    /// Serial device the Teensy `teensy_bridge` enumerates as, e.g.
    /// `/dev/ttyACM0`.
    pub serial_device: String,
    /// Expected report period in milliseconds (the Teensy sets the BNO's
    /// actual cadence). Seeds the staleness budget: the snapshot reads
    /// stale after 3× this, at least 250 ms. 5 ms = 200 Hz; 50 ms = 20 Hz.
    pub rotation_vector_period_ms: u16,
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
    /// Load and validate a robot config from a YAML file.
    ///
    /// Validates:
    /// - Each `model` parses as a known [`RobstrideModel`].
    /// - `motor_id` is in `1..=255`; a wheel's `node_id` in `0..=63`.
    /// - `pos_min < pos_max`, `vel_max > 0`, etc.
    /// - No two joints share `(can_interface, motor_id)`, no two wheels
    ///   `(can_interface, node_id)`.
    /// - Joint and wheel names are unique.
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
        // we want so that `joints` keeps the file ordering.
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

            let direction = raw_joint.direction.or(defaults.direction).unwrap_or(1.0);
            validate_direction(&name, direction)?;

            joints.push(JointConfig {
                name,
                can_id: motor_id,
                can_bus,
                model,
                hard_limits,
                hold_gains,
                test_gains,
                slew,
                direction,
            });
        }

        // ----- Wheels (ODrive BotWheel) -----------------------------------
        // Parse the optional `wheels:` mapping. A config may carry wheels,
        // joints, or both; at least one of the two must be present.
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

            let vel_max = raw_wheel
                .vel_max
                .ok_or_else(|| anyhow!("wheel {name:?}: missing vel_max (rad/s)"))?;
            if vel_max <= 0.0 {
                return Err(anyhow!("wheel {name:?}: vel_max must be > 0"));
            }
            let feedback_timeout_ms = raw_wheel.feedback_timeout_ms.unwrap_or(100.0);
            if feedback_timeout_ms <= 0.0 {
                return Err(anyhow!("wheel {name:?}: feedback_timeout_ms must be > 0"));
            }
            let direction = raw_wheel.direction.unwrap_or(1.0);
            validate_direction(&name, direction)?;

            wheels.push(WheelConfig {
                name,
                can_interface: can_bus,
                node_id: node_id as u8,
                direction,
                vel_max,
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
                operator_timeout_ms: s.operator_timeout_ms.unwrap_or(500),
                operator_grace_ms: s.operator_grace_ms.unwrap_or(2000),
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

        let imu = raw
            .imu
            .map(|raw_imu| -> Result<ImuConfig> {
                let source = match raw_imu.source.as_deref() {
                    Some(s) => s.parse::<ImuSource>()?,
                    None => ImuSource::Serial,
                };
                let serial_device = raw_imu
                    .serial_device
                    .unwrap_or_else(|| "/dev/ttyACM0".to_string());
                let rotation_vector_period_ms = raw_imu.rotation_vector_period_ms.unwrap_or(50);
                if rotation_vector_period_ms == 0 {
                    return Err(anyhow!("imu.rotation_vector_period_ms must be >= 1"));
                }
                let mount_quat_sensor_body = match raw_imu.mount {
                    Some(m) => build_mount_quat_sensor_body(&m)?,
                    None => ImuConfig::IDENTITY_QUAT,
                };
                Ok(ImuConfig {
                    source,
                    serial_device,
                    rotation_vector_period_ms,
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
            imu,
        })
    }

    pub fn get_joint(&self, name: &str) -> Option<&JointConfig> {
        self.joints.iter().find(|j| j.name == name)
    }

    pub fn num_joints(&self) -> usize {
        self.joints.len()
    }

    pub fn num_wheels(&self) -> usize {
        self.wheels.len()
    }
}

// ===========================================================================
// Joint state / command (used by the Robstride and ODrive drivers).
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
    /// on so that `RobotConfig::joints` keeps the file ordering.
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
    /// `"serial"` (the default). Selects the IMU backend.
    source: Option<String>,
    /// Serial device for the Teensy `teensy_bridge`. Defaults to
    /// `/dev/ttyACM0`.
    serial_device: Option<String>,
    rotation_vector_period_ms: Option<u16>,
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
    /// Motor→robot direction sign (`+1` / `-1`). See
    /// [`JointConfig::direction`]. Omitted ⇒ `+1` (default, motor agrees
    /// with the URDF). Inherits the `defaults.direction` block.
    direction: Option<f32>,
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
}

#[derive(Debug, Default, Deserialize)]
struct RawServer {
    bind_addr: Option<String>,
    telemetry_default_hz: Option<u32>,
    telemetry_max_hz: Option<u32>,
    operator_timeout_ms: Option<u64>,
    operator_grace_ms: Option<u64>,
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
/// Used by the IMU reader (`imu_serial.rs`) as a post-multiplier:
/// `q_world_body = q_world_sensor * q_sensor_body`.
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
    // exactly the convention `ImuShared` expects.
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

    /// The full `joints:` schema the arms will use — a `defaults:` block,
    /// per-joint overrides, a reversed-polarity joint and two buses —
    /// next to the wheeled chassis it will share a config with.
    const ARMS_AND_WHEELS: &str = r#"
defaults:
  hard_limits:
    pos_min: -0.5
    pos_max: 0.5
    vel_max: 2.0
    tau_max: 8.0
    temp_max: 65.0
    feedback_timeout_ms: 100
  hold_gains: { kp: 5.0, kd: 1.0 }
  test_gains: { kp: 30.0, kd: 2.0 }
  slew:
    max_pos_step_per_tick: 0.01
joints:
  shoulder_pitch_left_joint:
    can_interface: can1
    motor_id: 11
    model: RS03
    direction: -1
    hard_limits: { pos_min: -1.5, pos_max: 1.5, tau_max: 30.0 }
    hold_gains: { kp: 40.0, kd: 3.0 }
  elbow_left_joint:
    can_interface: can1
    motor_id: 12
    model: RS02
  shoulder_pitch_right_joint:
    can_interface: can2
    motor_id: 11
    model: RS03
    slew: { max_pos_step_per_tick: 0.02 }
wheels:
  left:
    can_interface: can4
    node_id: 36
    vel_max: 20.0
  right:
    can_interface: can4
    node_id: 46
    direction: -1
    vel_max: 20.0
drive:
  left_wheel: left
  right_wheel: right
  wheel_radius: 0.0855
  track_width: 0.402
"#;

    #[test]
    fn joint_schema_parses_next_to_wheels_with_reversed_polarity() {
        let path = write_temp_config(ARMS_AND_WHEELS);
        let cfg = RobotConfig::from_yaml(&path).expect("load arms + wheels config");
        std::fs::remove_file(&path).ok();

        // Joints keep the YAML order; a motor_id may repeat on another bus.
        let names: Vec<&str> = cfg.joints.iter().map(|j| j.name.as_str()).collect();
        assert_eq!(
            names,
            [
                "shoulder_pitch_left_joint",
                "elbow_left_joint",
                "shoulder_pitch_right_joint"
            ]
        );
        let shoulder = cfg.get_joint("shoulder_pitch_left_joint").unwrap();
        assert_eq!(shoulder.direction, -1.0, "reversed polarity must parse");
        assert_eq!((shoulder.can_bus.as_str(), shoulder.can_id), ("can1", 11));
        assert_eq!(shoulder.model, RobstrideModel::RS03);
        // Per-joint overrides win field by field; the rest comes from
        // `defaults:`.
        assert_eq!(shoulder.hard_limits.pos_min, -1.5);
        assert_eq!(shoulder.hard_limits.tau_max, 30.0);
        assert_eq!(shoulder.hard_limits.vel_max, 2.0);
        assert_eq!(shoulder.hold_gains.kp, 40.0);
        assert_eq!(shoulder.test_gains.kd, 2.0);

        let elbow = cfg.get_joint("elbow_left_joint").unwrap();
        assert_eq!(elbow.direction, 1.0);
        assert_eq!(elbow.model, RobstrideModel::RS02);
        assert_eq!(elbow.hard_limits.pos_max, 0.5);
        assert!((elbow.slew.max_pos_step_per_tick - 0.01).abs() < 1e-6);
        let right = cfg.get_joint("shoulder_pitch_right_joint").unwrap();
        assert!((right.slew.max_pos_step_per_tick - 0.02).abs() < 1e-6);

        assert_eq!(cfg.num_wheels(), 2);
        assert!(cfg.drive.is_some());
        // Every bus a joint or wheel names is registered, sorted.
        assert_eq!(cfg.can_interfaces, ["can1", "can2", "can4"]);
    }

    #[test]
    fn shipped_bebop_wheeled_yaml_parses_power_block() {
        // Guards the deployed wheeled config: the chassis has its PowerBoard
        // on can0 (addr 0xAA) and a 13s NMC pack, and the parser must
        // auto-register both `can4` (wheels) and `can0` (power) in the bus
        // pool — the YAML deliberately carries no explicit
        // `can_interfaces:` list. Losing the `power:` block silently
        // hides the operator app's battery card (present=false).
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("config/bebop_wheeled.yaml");
        let cfg = RobotConfig::from_yaml(&path).expect("load shipped bebop_wheeled.yaml");
        assert_eq!(cfg.num_joints(), 0);
        assert_eq!(cfg.num_wheels(), 2);
        assert!(cfg.drive.is_some());
        let power = cfg.power.as_ref().expect("power block present");
        assert_eq!(power.can_interface, "can0");
        assert_eq!(power.power_id, 170);
        assert_eq!(power.poll_interval_ms, 1000);
        assert_eq!(power.battery_cells, 13);
        assert!((power.cell_full_voltage - 4.20).abs() < 1e-6);
        assert!((power.cell_empty_voltage - 3.00).abs() < 1e-6);
        assert!(cfg.can_interfaces.contains(&"can4".to_string()));
        assert!(cfg.can_interfaces.contains(&"can0".to_string()));
    }
}

#[cfg(test)]
mod imu_mount_tests {
    use super::*;

    /// Apply `q_sensor_body` to a sensor-frame vector and return the body-
    /// frame components. Mirrors what the IMU reader does at runtime (via the
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
        let wheel = |name: &str| cfg.wheels.iter().find(|w| w.name == name);
        assert_eq!(cfg.num_joints(), 0);
        assert_eq!(cfg.num_wheels(), 2);
        assert!(wheel("left").is_some());
        assert_eq!(wheel("right").unwrap().direction, -1.0);
        assert_eq!(wheel("left").unwrap().direction, 1.0);
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
    fn operator_timeout_defaults_to_500ms() {
        let path = write_temp_config(DIFF_DRIVE);
        let cfg = RobotConfig::from_yaml(&path).expect("load wheeled config");
        assert_eq!(cfg.server.operator_timeout_ms, 500);
        assert_eq!(cfg.server.operator_grace_ms, 2000);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn operator_timeout_is_configurable_and_zero_disables() {
        let body = format!("{DIFF_DRIVE}server:\n  operator_timeout_ms: 250\n");
        let path = write_temp_config(&body);
        let cfg = RobotConfig::from_yaml(&path).expect("load wheeled config");
        assert_eq!(cfg.server.operator_timeout_ms, 250);
        std::fs::remove_file(&path).ok();

        let body = format!("{DIFF_DRIVE}server:\n  operator_timeout_ms: 0\n");
        let path = write_temp_config(&body);
        let cfg = RobotConfig::from_yaml(&path).expect("load wheeled config");
        assert_eq!(cfg.server.operator_timeout_ms, 0);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn operator_grace_is_configurable_and_zero_disables_arbitration() {
        let body = format!("{DIFF_DRIVE}server:\n  operator_grace_ms: 5000\n");
        let path = write_temp_config(&body);
        let cfg = RobotConfig::from_yaml(&path).expect("load wheeled config");
        assert_eq!(cfg.server.operator_grace_ms, 5000);
        std::fs::remove_file(&path).ok();

        let body = format!("{DIFF_DRIVE}server:\n  operator_grace_ms: 0\n");
        let path = write_temp_config(&body);
        let cfg = RobotConfig::from_yaml(&path).expect("load wheeled config");
        assert_eq!(cfg.server.operator_grace_ms, 0);
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
    fn imu_block_without_source_defaults_to_serial() {
        let body = format!("{DIFF_DRIVE}imu:\n  serial_device: /dev/bebop-imu\n");
        let path = write_temp_config(&body);
        let cfg = RobotConfig::from_yaml(&path).expect("load wheeled config with imu");
        std::fs::remove_file(&path).ok();
        let imu = cfg.imu.expect("imu block parsed");
        assert_eq!(imu.source, ImuSource::Serial);
        assert_eq!(imu.serial_device, "/dev/bebop-imu");
        assert_eq!(imu.rotation_vector_period_ms, 50);
        assert_eq!(imu.mount_quat_sensor_body, ImuConfig::IDENTITY_QUAT);
    }

    #[test]
    fn rejects_unknown_imu_source() {
        let body = format!("{DIFF_DRIVE}imu:\n  source: spi\n");
        let path = write_temp_config(&body);
        let err = RobotConfig::from_yaml(&path).unwrap_err();
        std::fs::remove_file(&path).ok();
        assert!(
            format!("{err:#}").contains("unknown imu.source"),
            "unexpected error: {err:#}"
        );
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
