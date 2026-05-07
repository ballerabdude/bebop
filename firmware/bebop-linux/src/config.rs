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
pub mod timing {
    pub const POLICY_RATE_HZ: u64 = 50;
    pub const POLICY_INTERVAL_MS: u64 = 1000 / POLICY_RATE_HZ;
    pub const FEEDBACK_PUBLISH_HZ: u64 = 100;
    pub const WATCHDOG_TIMEOUT_MS: u64 = 500;
    pub const UDP_PORT: u16 = 10000;
}

/// Observation and action dimensions (must match training!).
pub mod dims {
    pub const OBS_DIM: usize = 30;
    pub const ACTION_DIM: usize = 6;
    pub const HISTORY_STEPS: usize = 1;
    pub const TOTAL_OBS_DIM: usize = OBS_DIM * HISTORY_STEPS;
}

/// Scaling factors used by [`crate::observation`] (must match training!).
pub mod scales {
    pub const SCALE_LIN_VEL: f32 = 1.0;
    pub const SCALE_ANG_VEL: f32 = 1.0;
    pub const SCALE_DOF_POS: f32 = 1.0;
    pub const SCALE_DOF_VEL: f32 = 1.0;
    pub const SCALE_ACTION_LEGS: f32 = 0.8;
    /// Legacy v1 (wheeled) constant. Unused on v2; retained so the legacy
    /// observation/policy modules continue to compile while RunPolicy mode
    /// is stubbed pending a v2-trained ONNX model.
    pub const SCALE_ACTION_WHEELS: f32 = 20.0;

    pub const CLIP_LIN_VEL: f32 = 3.0;
    pub const CLIP_ANG_VEL: f32 = 10.0;
    pub const CLIP_DOF_VEL: f32 = 15.0;
}

/// Observation-time clipping (legacy name kept; used by `observation.rs`).
pub mod limits {
    pub const MAX_LEG_POS_RAD: f32 = 1.6;
    /// Legacy v1 (wheeled) constant. See `scales::SCALE_ACTION_WHEELS`.
    pub const MAX_WHEEL_VEL_RAD_S: f32 = 20.0;
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
        torque_min: -25.0,
        torque_max: 25.0,
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
/// Bebop V2 itself ships with a 13s LFP pack, see `bebop_v2.yaml` for the
/// matching 3.45 V / 2.70 V endpoints.
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
    pub can_interfaces: Vec<String>,
    pub server: ServerConfig,
    pub power: Option<PowerBoardConfig>,
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
            });
        }

        if joints.is_empty() {
            return Err(anyhow!("config has no joints"));
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

        Ok(Self {
            joints,
            can_interfaces,
            server,
            power,
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
    server: Option<RawServer>,
    power: Option<RawPower>,
}

#[derive(Debug, Default, Deserialize)]
struct RawDefaults {
    hard_limits: Option<RawLimits>,
    hold_gains: Option<RawGains>,
    test_gains: Option<RawGains>,
    slew: Option<RawSlew>,
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
