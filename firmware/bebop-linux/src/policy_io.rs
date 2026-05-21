//! Shared snapshot of the latest policy observation + action vectors.
//!
//! The [`crate::policy_runner::PolicyRunner`] writes this on every 100 Hz
//! tick while RunPolicy is active; the WS telemetry builder reads it at
//! ~30 Hz and packs it into [`bebop_proto::runtime::v1::PolicyIoStats`].

use std::sync::{Arc, Mutex};

use crate::observation::{JOINT_NAMES, NUM_JOINTS};

/// Latest policy I/O from the inference loop.
#[derive(Debug, Clone, Default)]
pub struct PolicyIoSnapshot {
    /// True when `policy.onnx` loaded at boot.
    pub present: bool,
    /// True while RunPolicy is active and E-STOP is not latched.
    pub active: bool,
    /// Whether observations use live BNO085 readings vs synthetic fallback.
    pub imu_live: bool,
    /// Full 52-dim observation vector fed to ONNX.
    pub observation: Vec<f32>,
    /// Full 24-dim raw NN output.
    pub raw_action: Vec<f32>,
    /// Decoded position targets (rad), policy slot order.
    pub position_targets_rad: [f32; NUM_JOINTS],
    /// Decoded kp gains, policy slot order.
    pub kp: [f32; NUM_JOINTS],
    /// Decoded kd gains, policy slot order.
    pub kd: [f32; NUM_JOINTS],
}

pub type PolicyIoShared = Arc<Mutex<PolicyIoSnapshot>>;

/// Allocate a fresh shared snapshot. Call [`PolicyIoSnapshot::set_present`]
/// after a successful policy load.
pub fn new_shared() -> PolicyIoShared {
    Arc::new(Mutex::new(PolicyIoSnapshot::default()))
}

impl PolicyIoSnapshot {
    /// Mark whether a policy was loaded at boot.
    pub fn set_present(&mut self, present: bool) {
        self.present = present;
        if !present {
            self.clear_tick();
        }
    }

    /// Clear per-tick fields when leaving RunPolicy or on policy unload.
    pub fn clear_tick(&mut self) {
        self.active = false;
        self.imu_live = false;
        self.observation.clear();
        self.raw_action.clear();
        self.position_targets_rad = [0.0; NUM_JOINTS];
        self.kp = [0.0; NUM_JOINTS];
        self.kd = [0.0; NUM_JOINTS];
    }

    /// Publish one inference cycle to the shared snapshot.
    pub fn publish_tick(
        &mut self,
        imu_live: bool,
        observation: &[f32],
        raw_action: &[f32],
        position_targets_rad: &[f32; NUM_JOINTS],
        kp: &[f32; NUM_JOINTS],
        kd: &[f32; NUM_JOINTS],
    ) {
        self.active = true;
        self.imu_live = imu_live;
        self.observation.clear();
        self.observation.extend_from_slice(observation);
        self.raw_action.clear();
        self.raw_action.extend_from_slice(raw_action);
        self.position_targets_rad = *position_targets_rad;
        self.kp = *kp;
        self.kd = *kd;
    }
}

/// Policy slot joint names for UI labels.
pub fn joint_names() -> &'static [&'static str; NUM_JOINTS] {
    &JOINT_NAMES
}
