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
use crate::nav::NavHub;
use crate::policy_control::PolicyControlShared;
use crate::policy_io::PolicyIoShared;
use crate::safety::{Supervisor, SupervisorEvent};
use crate::server::handlers::{encode, handle_client_message};
use crate::server::telemetry::{build_telemetry, telemetry_envelope};
use crate::video::VideoHub;
use anyhow::Result;
use axum::body::{Body, Bytes};
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::Json;
use axum::Router;
use bebop_proto::runtime::v1 as proto;
use serde::Serialize;
use std::convert::Infallible;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, UNIX_EPOCH};
use tokio::sync::broadcast;
use tokio::sync::mpsc;
use tower_http::cors::{Any, CorsLayer};
use tower_http::services::ServeDir;
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
    /// Latest policy observation/action snapshot from the inference loop.
    pub policy_io: PolicyIoShared,
    /// Operator-toggled dry-run flag. Written here from the WS handler;
    /// read by [`crate::policy_runner::PolicyRunner`].
    pub policy_control: PolicyControlShared,
    /// Directory the MCAP writer thread is writing into. Exposed to the
    /// HTTP layer so `GET /captures` can list finished segments and
    /// `GET /captures/dl/<name>` (via `ServeDir`) can stream them out.
    pub capture_dir: PathBuf,
    /// MJPEG camera hub (see [`crate::video`]). `None` on robots without
    /// a `video:` config; `GET /video` answers 503 then.
    pub video: Option<Arc<VideoHub>>,
    /// Navigable-path runner handle (see [`crate::nav`]). `None` on
    /// robots without a `nav:` block or when the model failed to load;
    /// telemetry then reports `NavState.present = false` and the
    /// `subscribe_nav` push sends nothing.
    pub nav: Option<Arc<NavHub>>,
}

pub async fn run_server(state: AppState, bind_addr: &str) -> Result<()> {
    // Permissive CORS: the operator app is served from a different origin
    // (e.g. tauri://localhost or a dev http://localhost:1420), and we're
    // on the LAN. WebSockets aren't subject to CORS but the /healthz
    // pre-flight ping and the /captures download endpoints are, so allow
    // any origin to read them.
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);
    // `ServeDir` does its own path-traversal sanitization and handles
    // range requests (so a partial download / resume works out of the
    // box), Content-Type, and ETags. We nest it under /captures/dl/
    // rather than serving the capture dir at the root so the JSON list
    // endpoint can live next to it without filename collisions.
    let app = Router::new()
        .route("/healthz", get(|| async { "ok" }))
        .route("/ws", get(ws_upgrade))
        .route("/video", get(video_stream))
        .route("/captures", get(list_captures))
        .nest_service("/captures/dl", ServeDir::new(state.capture_dir.clone()))
        .with_state(state)
        .layer(cors);

    let addr: SocketAddr = bind_addr.parse()?;
    info!(%addr, "starting WS runtime server");
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

/// `GET /video` — live MJPEG from the robot camera.
///
/// Serves `multipart/x-mixed-replace` with JPEG parts: render directly in
/// an `<img>` tag from the operator app, or open the URL as a capture
/// source in OpenCV / FFmpeg (bebop-vision does exactly this). Each part
/// carries the host capture timestamp in `X-Timestamp-Us` so consumers can
/// align frames with telemetry. A slow client lags independently — the
/// broadcast channel drops stale frames per subscriber. 503 when the
/// robot has no `video:` config.
async fn video_stream(State(state): State<AppState>) -> Response {
    let hub = match state.video {
        Some(ref hub) => hub.clone(),
        None => {
            return (
                axum::http::StatusCode::SERVICE_UNAVAILABLE,
                "no camera configured (missing `video:` in robot yaml)",
            )
                .into_response()
        }
    };

    const BOUNDARY: &str = "bebopframe";
    // If the capture thread stops producing frames (camera unplugged,
    // hub wedged), end the response after this long so clients get a
    // stream end (→ <img> onerror / reader EOF) instead of an idle
    // socket that never closes.
    const FRAME_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);
    let stream = futures::stream::unfold(hub.subscribe(), |mut rx| async move {
        loop {
            match tokio::time::timeout(FRAME_TIMEOUT, rx.recv()).await {
                Ok(Ok(frame)) => {
                    let mut part = format!(
                        "\r\n--{BOUNDARY}\r\nContent-Type: image/jpeg\r\n\
                         Content-Length: {}\r\nX-Timestamp-Us: {}\r\n\
                         X-Pan-Deg: {:.2}\r\nX-Tilt-Deg: {:.2}\r\n\r\n",
                        frame.jpeg.len(),
                        frame.ts_us,
                        frame.pan_deg,
                        frame.tilt_deg
                    )
                    .into_bytes();
                    part.extend_from_slice(&frame.jpeg);
                    let chunk: Result<Bytes, Infallible> = Ok(Bytes::from(part));
                    return Some((chunk, rx));
                }
                // Slow consumer: skip whatever queued up and continue
                // from the newest frame instead of backlogging.
                Ok(Err(broadcast::error::RecvError::Lagged(dropped))) => {
                    debug!(
                        dropped,
                        "video subscriber lagged; resyncing to newest frame"
                    );
                    continue;
                }
                Ok(Err(broadcast::error::RecvError::Closed)) | Err(_) => return None,
            }
        }
    });

    Response::builder()
        .header(
            header::CONTENT_TYPE,
            format!("multipart/x-mixed-replace; boundary={BOUNDARY}"),
        )
        .header(header::CACHE_CONTROL, "no-store")
        .body(Body::from_stream(stream))
        .unwrap()
}

/// One row of the `GET /captures` response. The TS consumer expects
/// camelCase keys (`name`, `sizeBytes`, `modifiedMs`) — without the
/// `rename_all` attribute serde would emit `size_bytes` / `modified_ms`
/// and the web app would render every row as "NaN GiB" / "—".
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct CaptureEntry {
    /// Filename (e.g. `policy_capture_20260612_021530.mcap`). The
    /// download URL is `/captures/dl/<name>`.
    name: String,
    /// On-disk size in bytes. Operators use this to spot a stuck /
    /// rotating writer vs a finished segment.
    size_bytes: u64,
    /// Modification time as Unix milliseconds. 0 if `stat` doesn't
    /// expose it on this filesystem.
    modified_ms: u64,
}

#[derive(Serialize)]
struct CapturesResponse {
    files: Vec<CaptureEntry>,
}

/// List all `policy_capture_*.mcap` files in the capture dir, newest
/// first. The currently-open segment (still being appended to by the
/// writer) is included so the operator can grab a partial recording
/// if needed — `ServeDir` will stream whatever bytes are on disk at
/// download time.
async fn list_captures(State(state): State<AppState>) -> impl IntoResponse {
    let dir = state.capture_dir.clone();
    // Read on a blocking thread so the axum executor isn't blocked on
    // a slow eMMC `readdir` (cheap on the Jetson, but free safety).
    let result = tokio::task::spawn_blocking(move || -> std::io::Result<Vec<CaptureEntry>> {
        let mut out: Vec<CaptureEntry> = Vec::new();
        for entry in std::fs::read_dir(&dir)? {
            let entry = match entry {
                Ok(e) => e,
                Err(_) => continue,
            };
            let name = match entry.file_name().into_string() {
                Ok(n) => n,
                Err(_) => continue,
            };
            if !(name.starts_with("policy_capture_") && name.ends_with(".mcap")) {
                continue;
            }
            let meta = match entry.metadata() {
                Ok(m) => m,
                Err(_) => continue,
            };
            if !meta.is_file() {
                continue;
            }
            let modified_ms = meta
                .modified()
                .ok()
                .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0);
            out.push(CaptureEntry {
                name,
                size_bytes: meta.len(),
                modified_ms,
            });
        }
        // Newest first so the operator sees the most relevant capture
        // at the top of the list.
        out.sort_by(|a, b| b.modified_ms.cmp(&a.modified_ms));
        Ok(out)
    })
    .await;

    match result {
        Ok(Ok(files)) => (StatusCode::OK, Json(CapturesResponse { files })).into_response(),
        Ok(Err(e)) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            [(header::CONTENT_TYPE, "text/plain")],
            format!("capture dir read failed: {e}"),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            [(header::CONTENT_TYPE, "text/plain")],
            format!("capture listing task failed: {e}"),
        )
            .into_response(),
    }
}

async fn ws_upgrade(ws: WebSocketUpgrade, State(state): State<AppState>) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_ws(socket, state))
}

async fn handle_ws(socket: WebSocket, state: AppState) {
    let AppState {
        sup,
        imu,
        imu_present,
        policy_io,
        policy_control,
        capture_dir: _,
        video,
        nav,
    } = state;
    info!("ws client connected");
    let (mut sink, mut stream) = socket.split();
    let (tx, mut rx) = mpsc::channel::<proto::ServerRuntimeMessage>(256);

    // Telemetry control: shared subscribed flag + clamped rate.
    let telemetry_state = Arc::new(tokio::sync::RwLock::new(TelemetryState {
        subscribed: false,
        rate_hz: 30,
    }));
    // Nav-mask push control: same shape as the telemetry subscription.
    let nav_state = Arc::new(tokio::sync::RwLock::new(NavPushState {
        subscribed: false,
        rate_hz: 10,
    }));
    let max_rate_hz = sup.cfg().server.telemetry_max_hz.max(1);
    let default_rate_hz = sup.cfg().server.telemetry_default_hz.max(1);
    // Cap nav-mask pushes well under the telemetry ceiling: each frame
    // is a ~14 KB grid, and the overlay doesn't need more than the
    // model produces anyway.
    const NAV_PUSH_MAX_HZ: u32 = 15;

    // Task: telemetry pump.
    let tx_tele = tx.clone();
    let sup_tele = sup.clone();
    let imu_tele = imu.clone();
    let policy_io_tele = policy_io.clone();
    let video_tele = video.clone();
    let nav_tele = nav.clone();
    let tele_state_tele = telemetry_state.clone();
    let mut client_telemetry_subscribed = false;
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
            let frame = build_telemetry(
                &sup_tele,
                &imu_tele,
                imu_present,
                &policy_io_tele,
                &video_tele,
                &nav_tele,
            );
            let env = telemetry_envelope(frame);
            if tx_tele.send(env).await.is_err() {
                break;
            }
        }
    });

    // Task: nav-mask pump. Only pushes on *new* masks (seq change) so a
    // slow model never re-sends the same grid; the subscription rate is
    // an upper bound on overlay update frequency.
    let tx_nav = tx.clone();
    let nav_push_state = nav_state.clone();
    let nav_push_hub = nav.clone();
    let nav_task = tokio::spawn(async move {
        let mut rx_hub = nav_push_hub.as_ref().map(|hub| hub.subscribe());
        let mut last_seq: Option<u64> = None;
        loop {
            let (subscribed, rate_hz) = {
                let g = nav_push_state.read().await;
                (g.subscribed, g.rate_hz)
            };
            let period = Duration::from_secs_f32(1.0 / rate_hz.max(1) as f32);
            tokio::time::sleep(period).await;
            if !subscribed {
                continue;
            }
            let Some(rx) = rx_hub.as_mut() else { continue };
            let frame = rx.borrow_and_update().clone();
            let Some(frame) = frame else { continue };
            if last_seq == Some(frame.seq) {
                continue;
            }
            let provider = nav_push_hub
                .as_ref()
                .map(|h| h.provider())
                .unwrap_or_default();
            let msg = proto::ServerRuntimeMessage {
                request_id: 0,
                payload: Some(proto::server_runtime_message::Payload::NavMask(
                    proto::NavMaskFrame {
                        seq: frame.seq,
                        ts_us: frame.ts_us,
                        width: crate::nav::MASK_W as u32,
                        height: crate::nav::MASK_H as u32,
                        grid: frame.grid.clone(),
                        frac_blocked: frame.frac_blocked,
                        frac_navigable: frame.frac_navigable,
                        frac_caution: frame.frac_caution,
                        mask_hz: frame.infer_hz,
                        provider,
                    },
                )),
            };
            last_seq = Some(frame.seq);
            if tx_nav.send(msg).await.is_err() {
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
                | SupervisorEvent::MotorDisarmed { .. }
                | SupervisorEvent::WheelArmed { .. }
                | SupervisorEvent::WheelDisarmed { .. } => None,
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
                let response = handle_client_message(
                    &sup,
                    &imu,
                    imu_present,
                    &policy_io,
                    &policy_control,
                    &video,
                    &nav,
                    &bytes,
                );

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
                                if !client_telemetry_subscribed {
                                    sup.inc_telemetry_subscribers();
                                    client_telemetry_subscribed = true;
                                }
                            }
                            proto::client_runtime_message::Payload::UnsubscribeTelemetry(_) => {
                                let mut g = telemetry_state.write().await;
                                g.subscribed = false;
                                if client_telemetry_subscribed {
                                    sup.dec_telemetry_subscribers();
                                    client_telemetry_subscribed = false;
                                }
                            }
                            proto::client_runtime_message::Payload::SubscribeNav(s) => {
                                let mut g = nav_state.write().await;
                                g.subscribed = true;
                                g.rate_hz = if s.rate_hz == 0 {
                                    10
                                } else {
                                    s.rate_hz.min(NAV_PUSH_MAX_HZ)
                                };
                            }
                            proto::client_runtime_message::Payload::UnsubscribeNav(_) => {
                                let mut g = nav_state.write().await;
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
    nav_task.abort();
    event_task.abort();
    if client_telemetry_subscribed {
        sup.dec_telemetry_subscribers();
    }
    info!("ws client disconnected");
}

struct TelemetryState {
    subscribed: bool,
    rate_hz: u32,
}

/// Nav-mask push subscription state (see `SubscribeNav`).
struct NavPushState {
    subscribed: bool,
    rate_hz: u32,
}
