//! Build telemetry / snapshot frames from the supervisor's current state.

use crate::mode::Mode;
use crate::powerboard::describe_faults;
use crate::safety::limits::MotorSnapshot;
use crate::safety::power_monitor::PowerBoardSnapshot;
use crate::safety::{bus_pool::read_can_state, Supervisor};
use bebop_proto::runtime::v1 as proto;
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

fn motor_state_to_proto(m: &MotorSnapshot) -> proto::MotorState {
    proto::MotorState {
        joint_name: m.joint_name.clone(),
        can_interface: m.can_interface.clone(),
        motor_id: m.motor_id as u32,
        model: m.model.to_string(),
        armed: m.armed,
        feedback_stale: m.feedback_stale,
        fault_bits: m.fault_bits as u32,
        position_rad: m.position,
        velocity_rad_s: m.velocity,
        torque_nm: m.torque,
        temperature_c: m.temperature,
        target_position_rad: m.target_position,
        pos_min_rad: m.pos_min,
        pos_max_rad: m.pos_max,
        vel_max: m.vel_max,
        tau_max: m.tau_max,
        temp_max: m.temp_max,
    }
}

fn collect_buses(sup: &Arc<Supervisor>) -> Vec<proto::BusEntry> {
    sup.cfg()
        .can_interfaces
        .iter()
        .map(|iface| {
            let state = read_can_state(iface);
            let healthy = matches!(state.as_deref(), Some("ERROR-ACTIVE"));
            proto::BusEntry {
                can_interface: iface.clone(),
                state: state.unwrap_or_default(),
                healthy,
            }
        })
        .collect()
}

fn now_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Build the `PowerStats` proto from the supervisor's cached snapshot.
///
/// Always returns a `PowerStats`: when no power board is configured we
/// emit a default with `present = false` so old clients keep decoding
/// cleanly (and new clients hide the card).
fn build_power_stats(sup: &Arc<Supervisor>) -> proto::PowerStats {
    let cfg = match sup.cfg().power.as_ref() {
        Some(c) => c,
        None => return proto::PowerStats::default(),
    };
    let snapshot: PowerBoardSnapshot = sup.power_snapshot().unwrap_or_default();

    let now = Instant::now();
    // Stale = older than 3× the configured poll interval. Generous to
    // avoid false positives on a busy bus, tight enough to flag a
    // genuinely missing board within a few seconds.
    let staleness_ms = cfg.poll_interval_ms.saturating_mul(3).max(2_000);
    let status_stale = snapshot.is_stale(now, staleness_ms);
    let last_status_age_ms = snapshot
        .last_status_rx
        .map(|t| now.duration_since(t).as_millis() as u32)
        .unwrap_or(0);

    let (
        battery_v,
        motor_v,
        temp_c,
        fault_bits,
        fault_desc,
        rail_12v,
        soft_start,
        motor_rail,
        rail_24v,
    ) = if let Some(s) = snapshot.status.as_ref() {
        (
            s.battery_voltage_v,
            s.motor_voltage_v,
            s.board_temperature_c,
            s.fault_bits,
            describe_faults(s.fault_bits),
            s.rail_12v_on,
            s.soft_start_on,
            s.motor_rail_on,
            s.rail_24v_on,
        )
    } else {
        (0.0, 0.0, 0.0, 0, String::new(), false, false, false, false)
    };

    let currents = snapshot.currents.unwrap_or_default();
    let total_motor_current_a = currents.al_current_a
        + currents.ar_current_a
        + currents.ll_current_a
        + currents.lr_current_a;

    // SOC is -1 when we don't have a battery voltage yet; otherwise the
    // linear-interp percent. Frontend treats negative as "unknown".
    let soc_pct = if snapshot.status.is_some() {
        cfg.estimate_soc_pct(battery_v).unwrap_or(-1.0)
    } else {
        -1.0
    };

    proto::PowerStats {
        present: true,
        can_interface: cfg.can_interface.clone(),
        power_id: cfg.power_id as u32,
        firmware_version: snapshot.version.unwrap_or_default(),
        status_received: snapshot.status.is_some(),
        status_stale,
        last_status_age_ms,
        battery_voltage_v: battery_v,
        motor_voltage_v: motor_v,
        board_temperature_c: temp_c,
        fault_bits,
        fault_description: fault_desc,
        rail_12v_on: rail_12v,
        soft_start_on: soft_start,
        motor_rail_on: motor_rail,
        rail_24v_on: rail_24v,
        current_al_a: currents.al_current_a,
        current_ar_a: currents.ar_current_a,
        current_ll_a: currents.ll_current_a,
        current_lr_a: currents.lr_current_a,
        total_motor_current_a,
        battery_cells: cfg.battery_cells,
        pack_full_voltage_v: cfg.pack_full_voltage(),
        pack_empty_voltage_v: cfg.pack_empty_voltage(),
        state_of_charge_pct: soc_pct,
    }
}

pub fn build_snapshot(sup: &Arc<Supervisor>) -> proto::Snapshot {
    let motors = sup.snapshot_motors();
    let mode_proto = sup.mode().as_proto() as i32;
    let estop_latched = sup.estop_active();
    let estop_reason = sup.estop_reason_human().unwrap_or_default();
    proto::Snapshot {
        host_unix_ms: now_unix_ms(),
        mode: mode_proto,
        estop_latched,
        estop_reason,
        motors: motors.iter().map(motor_state_to_proto).collect(),
        buses: collect_buses(sup),
        power: Some(build_power_stats(sup)),
    }
}

pub fn build_telemetry(sup: &Arc<Supervisor>) -> proto::TelemetryFrame {
    let motors = sup.snapshot_motors();
    let mode_proto = sup.mode().as_proto() as i32;
    let estop_latched = sup.estop_active();
    let estop_reason = sup.estop_reason_human().unwrap_or_default();
    proto::TelemetryFrame {
        host_unix_ms: now_unix_ms(),
        mode: mode_proto,
        estop_latched,
        estop_reason,
        motors: motors.iter().map(motor_state_to_proto).collect(),
        buses: collect_buses(sup),
        power: Some(build_power_stats(sup)),
    }
}

/// Wrap a TelemetryFrame in the ServerRuntimeMessage envelope.
pub fn telemetry_envelope(frame: proto::TelemetryFrame) -> proto::ServerRuntimeMessage {
    proto::ServerRuntimeMessage {
        request_id: 0,
        payload: Some(proto::server_runtime_message::Payload::Telemetry(frame)),
    }
}

#[allow(dead_code)]
pub fn _force_mode_used(_m: Mode) {}
