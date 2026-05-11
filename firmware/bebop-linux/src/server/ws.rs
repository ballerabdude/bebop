//! Axum-based WebSocket server.
//!
//! - `GET /healthz` — simple liveness probe (returns "ok").
//! - `GET /ws` — upgrades to a binary WebSocket; framing carries one
//!   `ClientRuntimeMessage` / `ServerRuntimeMessage` per WS message.
//!
//! Each WS connection runs three concurrent tasks:
//!
//! 1. **Inbound**: read frames from the socket, dispatch to
//!    [`super::handlers::handle_client_message`], queue the reply.
//! 2. **Telemetry**: every `1/rate_hz` seconds, build a `TelemetryFrame`
//!    and queue it. Default 30 Hz; clamped to `cfg.server.telemetry_max_hz`.
//!    Sending is gated by whether the client has subscribed.
//! 3. **Events**: forward supervisor events (mode change, E-STOP latched)
//!    as unsolicited frames.
//!
//! All three feed a shared mpsc to the WS sink writer.

use crate::imu::ImuShared;
use crate::safety::{Supervisor, SupervisorEvent};
use crate::server::handlers::{encode, handle_client_message};
use crate::server::telemetry::{build_telemetry, telemetry_envelope};
use anyhow::Result;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::State;
use axum::response::IntoResponse;
use axum::routing::get;
use axum::Router;
use bebop_proto::runtime::v1 as proto;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tower_http::cors::{Any, CorsLayer};
use tracing::{debug, info, warn};

#[derive(Clone)]
pub struct AppState {
    pub sup: Arc<Supervisor>,
    /// Latest IMU rotation-vector reading. Populated by [`crate::imu`]
    /// when the YAML has an `imu:` block; left at default otherwise.
    pub imu: ImuShared,
    /// True when the firmware was configured with an `imu:` block (drives
    /// the `ImuStats.present` proto flag).
    pub imu_present: bool,
}

pub async fn run_server(
    sup: Arc<Supervisor>,
    imu: ImuShared,
    imu_present: bool,
    bind_addr: &str,
) -> Result<()> {
    let state = AppState {
        sup,
        imu,
        imu_present,
    };
    // Permissive CORS: the operator app is served from a different origin
    // (e.g. tauri://localhost or a dev http://localhost:1420), and we're
    // on the LAN. WebSockets aren't subject to CORS but the /healthz
    // pre-flight ping is, so allow any origin to read it.
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);
    let app = Router::new()
        .route("/healthz", get(|| async { "ok" }))
        .route("/ws", get(ws_upgrade))
        .with_state(state)
        .layer(cors);

    let addr: SocketAddr = bind_addr.parse()?;
    info!(%addr, "starting WS runtime server");
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

async fn ws_upgrade(ws: WebSocketUpgrade, State(state): State<AppState>) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_ws(socket, state))
}

async fn handle_ws(socket: WebSocket, state: AppState) {
    let AppState {
        sup,
        imu,
        imu_present,
    } = state;
    info!("ws client connected");
    let (mut sink, mut stream) = socket.split();
    let (tx, mut rx) = mpsc::channel::<proto::ServerRuntimeMessage>(256);

    // Telemetry control: shared subscribed flag + clamped rate.
    let telemetry_state = Arc::new(tokio::sync::RwLock::new(TelemetryState {
        subscribed: false,
        rate_hz: 30,
    }));
    let max_rate_hz = sup.cfg().server.telemetry_max_hz.max(1);
    let default_rate_hz = sup.cfg().server.telemetry_default_hz.max(1);

    // Task: telemetry pump.
    let tx_tele = tx.clone();
    let sup_tele = sup.clone();
    let imu_tele = imu.clone();
    let tele_state_tele = telemetry_state.clone();
    let telemetry_task = tokio::spawn(async move {
        loop {
            let (subscribed, rate_hz) = {
                let g = tele_state_tele.read().await;
                (g.subscribed, g.rate_hz)
            };
            let period = Duration::from_secs_f32(1.0 / rate_hz.max(1) as f32);
            tokio::time::sleep(period).await;
            if !subscribed {
                continue;
            }
            let frame = build_telemetry(&sup_tele, &imu_tele, imu_present);
            let env = telemetry_envelope(frame);
            if tx_tele.send(env).await.is_err() {
                break;
            }
        }
    });

    // Task: forward supervisor events (mode change, e-stop latched).
    let tx_events = tx.clone();
    let mut event_rx = sup.subscribe();
    let event_task = tokio::spawn(async move {
        while let Ok(ev) = event_rx.recv().await {
            let payload = match ev {
                SupervisorEvent::ModeChanged(m) => Some(
                    proto::server_runtime_message::Payload::ModeChanged(proto::ModeChanged {
                        mode: m.as_proto() as i32,
                    }),
                ),
                SupervisorEvent::EStopLatched(reason) => {
                    Some(proto::server_runtime_message::Payload::EstopLatched(
                        proto::EStopLatched { reason },
                    ))
                }
                SupervisorEvent::EStopReset
                | SupervisorEvent::MotorArmed { .. }
                | SupervisorEvent::MotorDisarmed { .. } => None,
            };
            if let Some(p) = payload {
                let msg = proto::ServerRuntimeMessage {
                    request_id: 0,
                    payload: Some(p),
                };
                if tx_events.send(msg).await.is_err() {
                    break;
                }
            }
        }
    });

    // Task: WS writer pulls from the channel and serializes.
    let writer_task = tokio::spawn(async move {
        use futures::SinkExt;
        while let Some(msg) = rx.recv().await {
            let bytes = encode(&msg);
            if let Err(e) = sink.send(Message::Binary(bytes.to_vec())).await {
                debug!(error = %e, "ws send error; closing");
                break;
            }
        }
    });

    // Reader loop: handle incoming frames.
    use futures::StreamExt;
    while let Some(frame) = stream.next().await {
        match frame {
            Ok(Message::Binary(bytes)) => {
                let response = handle_client_message(&sup, &imu, imu_present, &bytes);

                // Side effects for messages that affect telemetry state: do this
                // after dispatch so the response is consistent with the new state.
                if let Ok(req) =
                    <proto::ClientRuntimeMessage as bebop_proto::Message>::decode(bytes.as_ref())
                {
                    if let Some(payload) = req.payload {
                        match payload {
                            proto::client_runtime_message::Payload::SubscribeTelemetry(s) => {
                                let mut g = telemetry_state.write().await;
                                g.subscribed = true;
                                g.rate_hz = if s.rate_hz == 0 {
                                    default_rate_hz
                                } else {
                                    s.rate_hz.min(max_rate_hz)
                                };
                            }
                            proto::client_runtime_message::Payload::UnsubscribeTelemetry(_) => {
                                let mut g = telemetry_state.write().await;
                                g.subscribed = false;
                            }
                            _ => {}
                        }
                    }
                }

                if tx.send(response).await.is_err() {
                    break;
                }
            }
            Ok(Message::Text(t)) => {
                warn!(?t, "ignoring text WS frame (binary protobuf only)");
            }
            Ok(Message::Ping(_)) | Ok(Message::Pong(_)) => {}
            Ok(Message::Close(_)) => break,
            Err(e) => {
                // Most "errors" here are benign client-side disconnects:
                // the browser tears down the TCP socket before completing
                // the WebSocket close handshake (especially during React
                // StrictMode dev double-mount or when the user navigates
                // mid-handshake). Log at DEBUG so they don't pollute the
                // operator's terminal.
                debug!(error = %e, "ws stream ended");
                break;
            }
        }
    }

    drop(tx);
    let _ = writer_task.await;
    telemetry_task.abort();
    event_task.abort();
    info!("ws client disconnected");
}

struct TelemetryState {
    subscribed: bool,
    rate_hz: u32,
}
