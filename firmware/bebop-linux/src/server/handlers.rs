//! Per-message dispatch for the runtime WebSocket protocol.
//!
//! Each `ClientRuntimeMessage` is decoded and routed to the supervisor.
//! The corresponding `ServerRuntimeMessage` is returned for the caller to
//! write back to the WS sink.

use crate::mode::Mode;
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
            ack(request_id, format!("telemetry subscribed (rate hint = {rate} Hz)"))
        }
        P::UnsubscribeTelemetry(_) => ack(request_id, "telemetry unsubscribed".into()),
        P::GetSnapshot(_) => snapshot_response(request_id, sup),
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
                    return error_response(
                        request_id,
                        format!("unknown mode value {}", req.mode),
                    )
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
        P::SetMotorTarget(req) => match sup.set_target_position(&req.joint_name, req.position_rad) {
            Ok(()) => ack(
                request_id,
                format!("{} target -> {:+.3} rad", req.joint_name, req.position_rad),
            ),
            Err(e) => error_response(request_id, fmt_err(&e)),
        },
        P::SetMechanicalZero(req) => match sup.set_mechanical_zero(&req.joint_name) {
            Ok(()) => ack(
                request_id,
                format!("{} mechanical zero set", req.joint_name),
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
        payload: Some(proto::server_runtime_message::Payload::Error(proto::Error {
            message,
        })),
    }
}

pub fn snapshot_response(
    request_id: u32,
    sup: &Arc<Supervisor>,
) -> proto::ServerRuntimeMessage {
    proto::ServerRuntimeMessage {
        request_id,
        payload: Some(proto::server_runtime_message::Payload::Snapshot(
            crate::server::telemetry::build_snapshot(sup),
        )),
    }
}

/// Encode a `ServerRuntimeMessage` to bytes for the WS sink.
pub fn encode(msg: &proto::ServerRuntimeMessage) -> Bytes {
    let mut buf = Vec::with_capacity(msg.encoded_len());
    msg.encode(&mut buf).expect("encode runtime message");
    Bytes::from(buf)
}
