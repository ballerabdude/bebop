//! Network control surface — protobuf-over-WebSocket mirror of the BLE
//! GATT server.
//!
//! Why this exists:
//!
//! The BLE server in `crate::ble` is the authoritative provisioning path
//! for first-time setup. Once the robot is on Wi-Fi, it's nicer for the
//! operator app to talk to the agent over the LAN (no pairing dance, no
//! BLE stack on the laptop) — especially in the "Connect by IP" flow,
//! where there is no BLE link at all. This module exposes the same
//! `bebop.v1.ClientRequest` / `AgentResponse` envelope over a binary
//! WebSocket, reusing `crate::ble::dispatcher::handle` verbatim so the
//! two surfaces stay in lockstep.
//!
//! Endpoints:
//!
//! - `GET /healthz` — returns "ok", used by the operator app's
//!   pre-flight probe (matches `bebop-linux`'s pattern so the app can
//!   speak the same dialect to both).
//! - `GET /ws` — upgrades to a binary WebSocket. Each `Message::Binary`
//!   frame carries one `ClientRequest`; the reply (one `AgentResponse`)
//!   comes back as a `Message::Binary` frame.
//!
//! Auth: not implemented yet — same posture as `bebop-linux`'s runtime
//! WS. The agent is expected to live on a trusted LAN; tighten with a
//! reverse proxy / `net.ws_bind_addr = "127.0.0.1:9091"` if untrusted
//! clients are reachable. A pairing-code handshake (paralleling
//! `BleConfig::require_pairing`) is the obvious next step.

use std::net::SocketAddr;

use anyhow::{Context, Result};
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::State;
use axum::response::IntoResponse;
use axum::routing::get;
use axum::Router;
use bebop_proto::{
    v1::{AgentResponse, ClientRequest, ResponseStatus},
    Message as ProtoMessage,
};
use futures::{SinkExt, StreamExt};
use tower_http::cors::{Any, CorsLayer};
use tracing::{debug, info, warn};

use crate::ble::dispatcher;
use crate::state::AppState;

#[derive(Clone)]
struct WsState {
    app: AppState,
}

/// Long-running task that binds the WS server and serves until the
/// listener errors out. Mirrors the supervisor pattern other subsystems
/// use (`crate::ble::run`, `crate::ota::run`, ...).
pub async fn run(state: AppState) -> Result<()> {
    let cfg = state.config().await;
    if cfg.net.disabled {
        info!("network control surface disabled in config; skipping WS server");
        // Stay alive — exiting would trip the supervisor's "subsystem
        // exited" branch and bring down the whole agent. Park forever.
        std::future::pending::<()>().await;
        return Ok(());
    }

    let bind_addr = cfg.net.ws_bind_addr.clone();
    let addr: SocketAddr = bind_addr
        .parse()
        .with_context(|| format!("parsing net.ws_bind_addr {bind_addr:?}"))?;

    // Permissive CORS for the `/healthz` probe. WebSocket upgrade
    // requests bypass CORS in browsers, so the app's WS connection
    // works regardless; only the pre-flight `fetch` to `/healthz` is
    // CORS-checked. See the matching block in `bebop-linux`'s ws.rs.
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let ws_state = WsState { app: state };
    let app = Router::new()
        .route("/healthz", get(|| async { "ok" }))
        .route("/ws", get(ws_upgrade))
        .with_state(ws_state)
        .layer(cors);

    info!(%addr, "starting agent control-surface WS server");
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .with_context(|| format!("binding {addr}"))?;
    axum::serve(listener, app)
        .await
        .context("axum::serve exited")?;
    Ok(())
}

async fn ws_upgrade(ws: WebSocketUpgrade, State(state): State<WsState>) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, state.app))
}

/// One per accepted connection. We read one `ClientRequest` at a time,
/// dispatch it, and write the `AgentResponse` back. There's no streaming
/// (telemetry, push events) on this surface: the dispatcher is strictly
/// request/response, so we don't need the multi-task fan-out that
/// `bebop-linux`'s WS server uses.
async fn handle_socket(socket: WebSocket, state: AppState) {
    info!("agent ws client connected");
    let (mut sink, mut stream) = socket.split();

    while let Some(frame) = stream.next().await {
        let bytes = match frame {
            Ok(Message::Binary(b)) => b,
            Ok(Message::Text(t)) => {
                warn!(
                    ?t,
                    "ignoring text WS frame; agent control surface is binary protobuf only"
                );
                continue;
            }
            Ok(Message::Ping(_)) | Ok(Message::Pong(_)) => continue,
            Ok(Message::Close(_)) => break,
            Err(e) => {
                debug!(error = %e, "agent ws stream ended");
                break;
            }
        };

        // Decode the envelope. A malformed frame gets a synthetic Error
        // response (so the operator app can show a useful message)
        // rather than dropping the connection silently.
        let response = match ClientRequest::decode(bytes.as_slice()) {
            Ok(req) => dispatcher::handle(&state, req).await,
            Err(e) => AgentResponse {
                request_id: 0,
                status: ResponseStatus::Error as i32,
                message: format!("malformed ClientRequest: {e}"),
                payload: None,
            },
        };

        let mut buf = Vec::with_capacity(response.encoded_len());
        if let Err(e) = response.encode(&mut buf) {
            warn!(error = %e, "encoding AgentResponse failed; dropping frame");
            continue;
        }
        if let Err(e) = sink.send(Message::Binary(buf)).await {
            debug!(error = %e, "agent ws send failed; closing");
            break;
        }
    }

    info!("agent ws client disconnected");
}
