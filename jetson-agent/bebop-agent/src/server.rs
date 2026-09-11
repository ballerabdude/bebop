//! Setup server — the provisioning control surface.
//!
//! Exposes the same `bebop.v1.ClientRequest` / `AgentResponse` envelope as
//! the (retired) BLE GATT server over a binary WebSocket, so the companion
//! app can scan/join Wi-Fi and set the robot name on the LAN or directly
//! over the robot's SoftAP.
//!
//! Endpoints:
//! - `GET /healthz` — returns "ok"; used by the app's pre-flight probe.
//! - `GET /ws` — upgrades to a binary WebSocket; one `ClientRequest` per
//!   binary frame, one `AgentResponse` back.
//! - `GET /` — a tiny status page so a phone browser can confirm it reached
//!   the robot without the app installed.

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

use crate::dispatcher;
use crate::state::AppState;

#[derive(Clone)]
struct ServerState {
    app: AppState,
}

/// Long-running task that binds the setup server and serves until the
/// listener errors out.
pub async fn run(state: AppState) -> Result<()> {
    let cfg = state.config().await;
    let bind_addr = cfg.network.setup_bind_addr.clone();
    let addr: SocketAddr = bind_addr
        .parse()
        .with_context(|| format!("parsing network.setup_bind_addr {bind_addr:?}"))?;

    // Permissive CORS for the `/healthz` probe and the status page.
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let server_state = ServerState { app: state };
    let app = Router::new()
        .route("/healthz", get(|| async { "ok" }))
        .route("/", get(status_page))
        .route("/ws", get(ws_upgrade))
        .with_state(server_state)
        .layer(cors);

    info!(%addr, "starting setup server");
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .with_context(|| format!("binding {addr}"))?;
    axum::serve(listener, app)
        .await
        .context("axum::serve exited")?;
    Ok(())
}

async fn status_page(State(state): State<ServerState>) -> impl IntoResponse {
    let wifi = state.app.wifi_status().await;
    let ap = state.app.ap_status().await;
    let cfg = state.app.config().await;
    let wifi_line = if wifi.connected {
        format!(
            "Connected to <b>{}</b> ({})",
            html_escape(&wifi.ssid),
            html_escape(&wifi.ip_address)
        )
    } else {
        "Not connected to Wi-Fi".to_owned()
    };
    let ap_line = if ap.active {
        format!(
            "Setup hotspot <b>{}</b> is active at <code>{}</code>",
            html_escape(&ap.ssid),
            html_escape(&ap.address)
        )
    } else {
        "Setup hotspot is not active".to_owned()
    };
    let body = format!(
        "<!doctype html><html><head><meta charset=utf-8>\
<meta name=viewport content=\"width=device-width,initial-scale=1\">\
<title>Bebop {name}</title>\
<style>body{{font-family:system-ui,sans-serif;max-width:34rem;margin:3rem auto;padding:0 1rem;line-height:1.5}}\
code{{background:#eee;padding:.1rem .3rem;border-radius:.25rem}}</style></head><body>\
<h1>Bebop robot</h1><p>Name: <b>{name}</b></p><p>{wifi}</p><p>{ap}</p>\
<p>Use the Bebop app to finish Wi-Fi setup, or POST to the protobuf WebSocket at <code>/ws</code>.</p>\
</body></html>",
        name = html_escape(&cfg.robot_name),
        wifi = wifi_line,
        ap = ap_line,
    );
    (
        [(axum::http::header::CONTENT_TYPE, "text/html; charset=utf-8")],
        body,
    )
}

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

async fn ws_upgrade(ws: WebSocketUpgrade, State(state): State<ServerState>) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, state.app))
}

/// One per accepted connection. Strictly request/response: the dispatcher
/// has no server-pushed events, so no fan-out task is needed.
async fn handle_socket(socket: WebSocket, state: AppState) {
    info!("setup client connected");
    let (mut sink, mut stream) = socket.split();

    while let Some(frame) = stream.next().await {
        let bytes = match frame {
            Ok(Message::Binary(b)) => b,
            Ok(Message::Text(t)) => {
                warn!(
                    ?t,
                    "ignoring text WS frame; setup surface is binary protobuf only"
                );
                continue;
            }
            Ok(Message::Ping(_)) | Ok(Message::Pong(_)) => continue,
            Ok(Message::Close(_)) => break,
            Err(e) => {
                debug!(error = %e, "setup ws stream ended");
                break;
            }
        };

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
            debug!(error = %e, "setup ws send failed; closing");
            break;
        }
    }

    info!("setup client disconnected");
}
