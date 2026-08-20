//! Per-message dispatch for the runtime WebSocket protocol.
//!
//! Each `ClientRuntimeMessage` is decoded and routed to the supervisor.
//! The corresponding `ServerRuntimeMessage` is returned for the caller to
//! write back to the WS sink.

use crate::drive::Twist;
use crate::imu::ImuShared;
use crate::mode::Mode;
use crate::policy_control::PolicyControlShared;
use crate::policy_io::PolicyIoShared;
use crate::safety::limits::BreachReason;
use crate::safety::Supervisor;
use bebop_proto::runtime::v1 as proto;
use bebop_proto::Message;
use bytes::Bytes;
use std::sync::Arc;

/// Format an anyhow chain for the operator. Without the alternate
/// formatter, `e.to_string()` only returns the outermost context message
/// — meaning the user sees "Failed to send extended frame to can1" but
/// not "No buffer space available (os error 105)" underneath. The
/// alternate formatter (`{:#}`) walks the cause chain.
fn fmt_err(e: &anyhow::Error) -> String {
    format!("{e:#}")
}

/// Decode and dispatch one client message. Returns the immediate reply
/// (Ack / Error / Snapshot / etc.) — or `None` for messages that don't
/// produce a response (e.g. SubscribeTelemetry, where the response is the
/// telemetry stream itself).
pub fn handle_client_message(
    sup: &Arc<Supervisor>,
    imu: &ImuShared,
    imu_present: bool,
    policy_io: &PolicyIoShared,
    policy_control: &PolicyControlShared,
    bytes: &[u8],
) -> proto::ServerRuntimeMessage {
    let req = match proto::ClientRuntimeMessage::decode(bytes) {
        Ok(m) => m,
        Err(e) => {
            return error_response(0, format!("decode error: {e}"));
        }
    };
    let request_id = req.request_id;
    let payload = match req.payload {
        Some(p) => p,
        None => {
            return error_response(request_id, "empty client message".into());
        }
    };

    use proto::client_runtime_message::Payload as P;
    match payload {
        P::SubscribeTelemetry(s) => {
            let rate = s.rate_hz;
            ack(
                request_id,
                format!("telemetry subscribed (rate hint = {rate} Hz)"),
            )
        }
        P::UnsubscribeTelemetry(_) => ack(request_id, "telemetry unsubscribed".into()),
        P::GetSnapshot(_) => snapshot_response(request_id, sup, imu, imu_present, policy_io),
        P::SetMotorEnabled(req) => {
            let result = if req.enabled {
                sup.arm(&req.joint_name)
            } else {
                sup.disarm(&req.joint_name)
            };
            match result {
                Ok(()) => ack(
                    request_id,
                    format!(
                        "{} {}",
                        if req.enabled { "armed" } else { "disarmed" },
                        req.joint_name
                    ),
                ),
                Err(e) => error_response(request_id, fmt_err(&e)),
            }
        }
        P::SetAllMotorsEnabled(req) => {
            let errs = if req.enabled {
                sup.arm_all()
            } else {
                sup.disarm_all()
            };
            if errs.is_empty() {
                ack(
                    request_id,
                    format!(
                        "all motors {}",
                        if req.enabled { "armed" } else { "disarmed" }
                    ),
                )
            } else {
                let msg = errs
                    .iter()
                    .map(|(n, e)| format!("{n}: {:#}", e))
                    .collect::<Vec<_>>()
                    .join("; ");
                error_response(request_id, format!("partial failure: {msg}"))
            }
        }
        P::SetMode(req) => {
            let mode_proto = proto::Mode::try_from(req.mode).unwrap_or(proto::Mode::Unspecified);
            let mode = match Mode::from_proto(mode_proto) {
                Some(m) => m,
                None => {
                    return error_response(request_id, format!("unknown mode value {}", req.mode))
                }
            };
            match sup.set_mode(mode) {
                Ok(()) => ack(request_id, format!("mode -> {mode:?}")),
                Err(e) => error_response(request_id, fmt_err(&e)),
            }
        }
        P::EmergencyStop(req) => {
            sup.trigger_estop(BreachReason::Operator(if req.reason.is_empty() {
                "operator E-STOP".into()
            } else {
                req.reason
            }));
            ack(request_id, "E-STOP latched".into())
        }
        P::ResetEstop(_) => {
            if sup.reset_estop() {
                ack(request_id, "E-STOP cleared".into())
            } else {
                error_response(request_id, "E-STOP not active".into())
            }
        }
        P::SetMotorTarget(req) => {
            match sup.set_target_position(&req.joint_name, req.position_rad) {
                Ok(()) => ack(
                    request_id,
                    format!("{} target -> {:+.3} rad", req.joint_name, req.position_rad),
                ),
                Err(e) => error_response(request_id, fmt_err(&e)),
            }
        }
        P::SetMechanicalZero(req) => match sup.set_mechanical_zero(&req.joint_name) {
            Ok(()) => ack(
                request_id,
                format!("{} mechanical zero set", req.joint_name),
            ),
            Err(e) => error_response(request_id, fmt_err(&e)),
        },
        P::SetMechanicalZeroAll(_) => match sup.set_mechanical_zero_all() {
            Ok(outcomes) => {
                let failed: Vec<_> = outcomes.iter().filter(|o| o.error.is_some()).collect();
                if !failed.is_empty() {
                    let msg = failed
                        .iter()
                        .map(|o| format!("{}: {:#}", o.joint_name, o.error.as_ref().unwrap()))
                        .collect::<Vec<_>>()
                        .join("; ");
                    error_response(request_id, format!("partial failure re-zeroing: {msg}"))
                } else {
                    let unverified: Vec<String> = outcomes
                        .iter()
                        .filter(|o| match o.position_after_rad {
                            Some(p) => p.abs() > Supervisor::ZERO_VERIFY_TOLERANCE_RAD,
                            None => true,
                        })
                        .map(|o| match o.position_after_rad {
                            Some(p) => format!("{} ({p:+.3} rad)", o.joint_name),
                            None => format!("{} (no feedback)", o.joint_name),
                        })
                        .collect();
                    let mut msg = format!("all {} actuators re-zeroed", outcomes.len());
                    if !unverified.is_empty() {
                        msg.push_str(&format!(
                            "; VERIFY FAILED (post-zero |pos| > {:.2} rad — motor may have ignored SET_ZERO, or joint not at the reference pose): {}",
                            Supervisor::ZERO_VERIFY_TOLERANCE_RAD,
                            unverified.join(", ")
                        ));
                    }
                    ack(request_id, msg)
                }
            }
            Err(e) => error_response(request_id, fmt_err(&e)),
        },
        P::SetPolicyDryRun(req) => {
            // Flip the flag here; PolicyRunner reads it on the next tick
            // (≤10 ms). We deliberately don't try to validate "is the
            // policy currently running" — toggling in IDLE is fine and
            // takes effect when the operator next enters RUN_POLICY.
            match policy_control.lock() {
                Ok(mut g) => {
                    g.dry_run = req.enabled;
                    ack(
                        request_id,
                        format!(
                            "policy dry-run {}",
                            if req.enabled { "ENABLED" } else { "disabled" }
                        ),
                    )
                }
                Err(_) => error_response(request_id, "policy_control mutex poisoned".into()),
            }
        }
        P::SetVelocityCommand(req) => {
            sup.set_cmd_vel(Twist {
                vx: req.linear_x,
                wz: req.angular_z,
            });
            ack(
                request_id,
                format!(
                    "drive twist -> vx {:.3} m/s, wz {:.3} rad/s",
                    req.linear_x, req.angular_z
                ),
            )
        }
        P::SetWheelEnabled(req) => {
            let result = if req.enabled {
                sup.arm_wheel(&req.wheel_name)
            } else {
                sup.disarm_wheel(&req.wheel_name)
            };
            match result {
                Ok(()) => ack(
                    request_id,
                    format!(
                        "{} {}",
                        if req.enabled { "armed" } else { "disarmed" },
                        req.wheel_name
                    ),
                ),
                Err(e) => error_response(request_id, fmt_err(&e)),
            }
        }
        P::SetAllWheelsEnabled(req) => {
            let errs = if req.enabled {
                sup.arm_all_wheels()
            } else {
                sup.disarm_all_wheels()
            };
            if errs.is_empty() {
                ack(
                    request_id,
                    format!(
                        "all wheels {}",
                        if req.enabled { "armed" } else { "disarmed" }
                    ),
                )
            } else {
                let msg = errs
                    .iter()
                    .map(|(n, e)| format!("{n}: {:#}", e))
                    .collect::<Vec<_>>()
                    .join("; ");
                error_response(request_id, format!("partial failure: {msg}"))
            }
        }
        P::ResetOdometry(_) => {
            sup.reset_odometry();
            ack(request_id, "odometry reset".into())
        }
        P::CalibrateWheel(req) => match sup.calibrate_wheel(&req.wheel_name) {
            Ok(()) => ack(
                request_id,
                format!(
                    "full calibration started on {} (axis spins ~20-30 s; NOT saved to S1 NVM — re-run after power cycle)",
                    req.wheel_name
                ),
            ),
            Err(e) => error_response(request_id, fmt_err(&e)),
        },
    }
}

pub fn ack(request_id: u32, message: String) -> proto::ServerRuntimeMessage {
    proto::ServerRuntimeMessage {
        request_id,
        payload: Some(proto::server_runtime_message::Payload::Ack(proto::Ack {
            ok: true,
            message,
        })),
    }
}

pub fn error_response(request_id: u32, message: String) -> proto::ServerRuntimeMessage {
    proto::ServerRuntimeMessage {
        request_id,
        payload: Some(proto::server_runtime_message::Payload::Error(
            proto::Error { message },
        )),
    }
}

pub fn snapshot_response(
    request_id: u32,
    sup: &Arc<Supervisor>,
    imu: &ImuShared,
    imu_present: bool,
    policy_io: &PolicyIoShared,
) -> proto::ServerRuntimeMessage {
    proto::ServerRuntimeMessage {
        request_id,
        payload: Some(proto::server_runtime_message::Payload::Snapshot(
            crate::server::telemetry::build_snapshot(sup, imu, imu_present, policy_io),
        )),
    }
}

/// Encode a `ServerRuntimeMessage` to bytes for the WS sink.
pub fn encode(msg: &proto::ServerRuntimeMessage) -> Bytes {
    let mut buf = Vec::with_capacity(msg.encoded_len());
    msg.encode(&mut buf).expect("encode runtime message");
    Bytes::from(buf)
}
