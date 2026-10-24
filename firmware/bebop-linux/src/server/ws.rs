//! Axum-based WebSocket server.
//!
//! - `GET /healthz` — simple liveness probe (returns "ok").
//! - `GET /ws` — upgrades to a binary WebSocket; framing carries one
//!   `ClientRuntimeMessage` / `ServerRuntimeMessage` per WS message.
//!
//! Each WS connection runs these concurrent tasks:
//!
//! 1. **Inbound** (the connection's own task): decode each frame once,
//!    dispatch to [`super::handlers::handle_client_message`], queue the
//!    reply (if any) on the reply queue.
//! 2. **Telemetry**: every `1/rate_hz` seconds, build a `TelemetryFrame`
//!    and *replace* the connection's latest-telemetry slot
//!    (`tokio::sync::watch`). Default 30 Hz; clamped to
//!    `cfg.server.telemetry_max_hz`. Gated by whether the client has
//!    subscribed.
//! 3. **Events**: forward supervisor events (mode change, E-STOP latched)
//!    onto the reply queue.
//! 4. **Writer**: drains the reply queue first (biased), then sends the
//!    newest telemetry frame. On a slow downlink stale telemetry frames
//!    are overwritten instead of queueing, so replies and events never
//!    wait behind a backlog of telemetry, and the inbound task is never
//!    parked by telemetry (accepted twists have no reply at all).

use crate::capture_control::CaptureControl;
use crate::imu::ImuShared;
use crate::model::ModelShared;
use crate::policy_control::PolicyControlShared;
use crate::policy_io::PolicyIoShared;
use crate::safety::{Supervisor, SupervisorEvent};
use crate::server::handlers::{encode, error_response, handle_client_message};
use crate::server::telemetry::{build_telemetry, telemetry_envelope};
use crate::vision::VisionShared;
use crate::voice::VoiceShared;
use anyhow::Result;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Path as AxumPath, State};
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Redirect};
use axum::routing::{get, post};
use axum::Json;
use axum::Router;
use bebop_proto::runtime::v1 as proto;
use bytes::Bytes;
use serde::Serialize;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, UNIX_EPOCH};
use tokio::sync::{mpsc, watch};
use tower_http::cors::{Any, CorsLayer};
use tower_http::services::ServeDir;
use tracing::{debug, info, warn};

/// Monotonic WS connection ids. Each accepted connection gets a fresh
/// id used for operator arbitration (which client owns the drive twist)
/// and the per-connection telemetry flags. Starts at 1; 0 is reserved
/// as "no connection".
static WS_CONN_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Cap on the total size of cached `.rrd` conversions in the capture dir.
/// They're derived artifacts (not pruned by the capture writers), so the
/// convert endpoint evicts oldest-first above this.
const RRD_BUDGET_BYTES: u64 = 4 * 1024 * 1024 * 1024;

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
    /// bebop-vision service control + status (see [`crate::vision`]).
    pub vision: VisionShared,
    /// Gated model-weight provisioning control + status (see
    /// [`crate::model`]).
    pub model: ModelShared,
    /// Voice (speech-to-speech) service control + status (see
    /// [`crate::voice`]).
    pub voice: VoiceShared,
    /// Finalize / active-segment state shared with the MCAP capture writers.
    /// Used to flag the currently-writing segment in `GET /captures` and to
    /// roll it over before a download (`POST /captures/finalize`).
    pub captures: Arc<CaptureControl>,
    /// Executable that converts an MCAP into a Rerun `.rrd`
    /// (`GET /captures/rerun/<name>`); see `scripts/setup-rerun.sh`.
    pub rerun_converter: PathBuf,
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
        .route("/captures", get(list_captures))
        .route("/captures/finalize", post(finalize_captures))
        .route("/captures/finalize-dl/:name", get(finalize_download))
        .route("/captures/rrd/:stem", get(serve_rrd))
        .nest_service("/captures/dl", ServeDir::new(state.capture_dir.clone()))
        .with_state(state)
        .layer(cors);

    let addr: SocketAddr = bind_addr.parse()?;
    info!(%addr, "starting WS runtime server");
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
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
    /// True for the segment a firmware writer currently has open (still
    /// being appended to). The app calls `/captures/finalize` before
    /// downloading one of these so the served file has a footer.
    active: bool,
}

#[derive(Serialize)]
struct CapturesResponse {
    files: Vec<CaptureEntry>,
}

/// List every MCAP in the capture dir — `policy_capture_*` (firmware),
/// `navd_session_*` (bebop-vision) and `system_*` (always-on system log) —
/// newest first. The currently-open segment (still being appended to by
/// the writer) is included so the operator can grab a partial recording
/// if needed — `ServeDir` will stream whatever bytes are on disk at
/// download time.
async fn list_captures(State(state): State<AppState>) -> impl IntoResponse {
    let dir = state.capture_dir.clone();
    let system_active = state.captures.system_path();
    let policy_active = state.captures.policy_path();
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
            // navd recorder sessions (bebop-vision) and the always-on
            // system log (bebop-linux) share this dir and are
            // listed/downloadable alongside policy captures.
            let is_mcap = name.ends_with(".mcap");
            if !(is_mcap
                && (name.starts_with("policy_capture_")
                    || name.starts_with("navd_session_")
                    || name.starts_with("system_")))
            {
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
            let path = entry.path();
            let active = system_active.as_deref() == Some(path.as_path())
                || policy_active.as_deref() == Some(path.as_path());
            out.push(CaptureEntry {
                name,
                size_bytes: meta.len(),
                modified_ms,
                active,
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

/// Roll the currently-writing segment(s) over so their files have a
/// footer. Waits (bounded) for the writers to open a fresh segment.
async fn finalize_now(control: &CaptureControl) {
    let before = (control.system_path(), control.policy_path());
    control.request_finalize();
    let deadline = Instant::now() + Duration::from_secs(3);
    while Instant::now() < deadline {
        let now = (control.system_path(), control.policy_path());
        let sys_done = before.0.is_none() || now.0 != before.0;
        let pol_done = before.1.is_none() || now.1 != before.1;
        if sys_done && pol_done {
            break;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
}

/// `POST /captures/finalize` — explicit finalize (kept for API/scripting).
async fn finalize_captures(State(state): State<AppState>) -> impl IntoResponse {
    finalize_now(&state.captures).await;
    Json(serde_json::json!({ "ok": true }))
}

/// `GET /captures/finalize-dl/<name>` — finalize `name` if it's the active
/// segment, then redirect to its normal download URL. This lets the app
/// keep a plain, copyable `<a href>` link that still yields a complete file
/// (the browser follows the redirect; Foxglove/Rerun follow it too).
async fn finalize_download(
    State(state): State<AppState>,
    AxumPath(name): AxumPath<String>,
) -> impl IntoResponse {
    let path = match safe_capture_path(&state.capture_dir, &name) {
        Ok(p) => p,
        Err(e) => return (StatusCode::BAD_REQUEST, e).into_response(),
    };
    let active = state.captures.system_path().as_deref() == Some(path.as_path())
        || state.captures.policy_path().as_deref() == Some(path.as_path());
    if active {
        finalize_now(&state.captures).await;
    }
    let location = format!("/captures/dl/{}", url_encode(&name));
    Redirect::temporary(&location).into_response()
}

/// Percent-encode a capture filename for use in a redirect Location.
fn url_encode(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    for b in name.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~') {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

/// Validate a capture filename and resolve it inside the capture dir.
/// Rejects path separators / traversal and non-MCAP names.
fn safe_capture_path(dir: &Path, name: &str) -> Result<PathBuf, String> {
    if name.is_empty()
        || name.contains('/')
        || name.contains('\\')
        || name.contains("..")
        || !name.ends_with(".mcap")
    {
        return Err(format!("invalid capture name {name:?}"));
    }
    let path = dir.join(name);
    if !path.is_file() {
        return Err(format!("no such capture {name:?}"));
    }
    Ok(path)
}

fn rrd_is_fresh(mcap: &Path, rrd: &Path) -> bool {
    let (Ok(m), Ok(r)) = (std::fs::metadata(mcap), std::fs::metadata(rrd)) else {
        return false;
    };
    match (m.modified(), r.modified()) {
        (Ok(mt), Ok(rt)) => rt >= mt,
        _ => false,
    }
}

/// Evict oldest `.rrd` conversions until the cache fits `budget` (never
/// removing `keep`, the file we're about to write).
fn prune_rrds(dir: &Path, budget: u64, keep: &Path) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut files: Vec<(PathBuf, u64, std::time::SystemTime)> = Vec::new();
    let mut total = 0u64;
    for entry in entries.flatten() {
        let path = entry.path();
        if !path
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.ends_with(".rrd"))
        {
            continue;
        }
        let Ok(meta) = entry.metadata() else { continue };
        if !meta.is_file() {
            continue;
        }
        total += meta.len();
        files.push((path, meta.len(), meta.modified().unwrap_or(UNIX_EPOCH)));
    }
    if total <= budget {
        return;
    }
    files.sort_by_key(|(_, _, m)| *m);
    for (path, size, _) in files {
        if total <= budget {
            break;
        }
        if path == keep {
            continue;
        }
        if std::fs::remove_file(&path).is_ok() {
            total = total.saturating_sub(size);
        }
    }
}

/// Ensure a fresh `.rrd` exists for `mcap` (convert if needed). On failure
/// returns an HTTP error response ready to send.
async fn ensure_converted(
    state: &AppState,
    mcap: &Path,
    rrd: &Path,
) -> Result<(), axum::response::Response> {
    if rrd_is_fresh(mcap, rrd) {
        return Ok(());
    }
    let converter = state.rerun_converter.clone();
    if !converter.exists() {
        return Err((
            StatusCode::SERVICE_UNAVAILABLE,
            "Rerun conversion not set up on this robot (run scripts/setup-rerun.sh)",
        )
            .into_response());
    }
    let mcap = mcap.to_path_buf();
    let rrd = rrd.to_path_buf();
    let dir = state.capture_dir.clone();
    let run = tokio::task::spawn_blocking(move || {
        prune_rrds(&dir, RRD_BUDGET_BYTES, &rrd);
        std::process::Command::new(&converter)
            .arg(&mcap)
            .arg(&rrd)
            .output()
            .map_err(|e| format!("spawn {}: {e}", converter.display()))
    })
    .await;
    match run {
        Ok(Ok(out)) if out.status.success() => Ok(()),
        Ok(Ok(out)) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            format!(
                "converter exited {}: {}",
                out.status,
                String::from_utf8_lossy(&out.stderr).trim()
            ),
        )
            .into_response()),
        Ok(Err(e)) => Err((StatusCode::INTERNAL_SERVER_ERROR, e).into_response()),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("conversion task failed: {e}"),
        )
            .into_response()),
    }
}

/// `GET /captures/rrd/<stem>.rrd` — convert `<stem>.mcap` on demand and
/// serve the resulting `.rrd` directly (200, streamed). The URL ends in
/// `.rrd` so Rerun's loader selects the Rerun format; a redirect would make
/// Rerun dispatch on the original `.mcap` extension and fail ("Bad magic
/// number").
async fn serve_rrd(
    State(state): State<AppState>,
    AxumPath(stem): AxumPath<String>,
) -> impl IntoResponse {
    let Some(base) = stem.strip_suffix(".rrd") else {
        return (StatusCode::BAD_REQUEST, "expected a .rrd path").into_response();
    };
    let mcap_name = format!("{base}.mcap");
    let mcap = match safe_capture_path(&state.capture_dir, &mcap_name) {
        Ok(p) => p,
        Err(e) => return (StatusCode::BAD_REQUEST, e).into_response(),
    };
    let rrd = state.capture_dir.join(&stem);
    if let Err(resp) = ensure_converted(&state, &mcap, &rrd).await {
        return resp;
    }
    serve_file(rrd).await
}

/// Stream a file (with range support) via `tower-http`'s `ServeFile`.
async fn serve_file(path: PathBuf) -> axum::response::Response {
    use tower::ServiceExt;
    let req = axum::http::Request::builder()
        .uri("/")
        .body(axum::body::Body::empty())
        .expect("static request");
    match tower_http::services::ServeFile::new(path)
        .oneshot(req)
        .await
    {
        Ok(res) => res.map(axum::body::Body::new).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("serve failed: {e}"),
        )
            .into_response(),
    }
}

async fn ws_upgrade(ws: WebSocketUpgrade, State(state): State<AppState>) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_ws(socket, state))
}

/// Per-connection reply/event queue depth. Telemetry does not ride this
/// queue (it has its own latest-value slot), so it only fills when a
/// client fires many requests while its downlink is stalled.
const REPLY_QUEUE_DEPTH: usize = 32;

/// Telemetry subscription state for one connection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TelemetryState {
    subscribed: bool,
    rate_hz: u32,
}

async fn handle_ws(socket: WebSocket, state: AppState) {
    // Per-connection identity for operator arbitration. Monotonic so a
    // reconnecting client never inherits a stale assignment; never 0
    // (0 reads as "no connection" in the per-connection telemetry
    // flags).
    let conn_id = WS_CONN_COUNTER.fetch_add(1, Ordering::SeqCst) + 1;
    info!(conn_id, "ws client connected");
    let sup = state.sup.clone();
    let (mut sink, mut stream) = socket.split();
    let (reply_tx, mut reply_rx) = mpsc::channel::<Bytes>(REPLY_QUEUE_DEPTH);
    let (tele_tx, mut tele_rx) = watch::channel::<Option<Bytes>>(None);

    // Telemetry control: subscribed flag + clamped rate, as a watch so the
    // pump sleeps until a subscription instead of polling.
    let (tele_cfg_tx, mut tele_cfg_rx) = watch::channel(TelemetryState {
        subscribed: false,
        rate_hz: 30,
    });
    let max_rate_hz = sup.cfg().server.telemetry_max_hz.max(1);
    let default_rate_hz = sup.cfg().server.telemetry_default_hz.max(1);
    let mut client_telemetry_subscribed = false;

    // Task: telemetry pump -> latest-value slot.
    let tele_state = state.clone();
    let telemetry_task = tokio::spawn(async move {
        loop {
            let cfg = *tele_cfg_rx.borrow_and_update();
            if !cfg.subscribed {
                if tele_cfg_rx.changed().await.is_err() {
                    break;
                }
                continue;
            }
            let period = Duration::from_secs_f32(1.0 / cfg.rate_hz.max(1) as f32);
            tokio::time::sleep(period).await;
            if !tele_cfg_rx.borrow().subscribed {
                continue;
            }
            // conn_id rides along so the arbitration flags inside the
            // drive state are computed for *this* client ("you").
            let frame = build_telemetry(
                &tele_state.sup,
                &tele_state.imu,
                tele_state.imu_present,
                &tele_state.policy_io,
                &tele_state.vision,
                &tele_state.model,
                &tele_state.voice,
                conn_id,
            );
            if tele_tx.is_closed() {
                break;
            }
            // Overwrites a frame the writer hasn't sent yet: on a slow
            // link the client gets the newest state, not a backlog.
            tele_tx.send_replace(Some(encode(&telemetry_envelope(frame))));
        }
    });

    // Task: forward supervisor events (mode change, e-stop latched).
    let tx_events = reply_tx.clone();
    let mut event_rx = sup.subscribe();
    let event_task = tokio::spawn(async move {
        loop {
            // A lagged receiver (writer stalled long enough to overrun the
            // broadcast) skips the lost events and keeps forwarding; only a
            // closed supervisor ends the task. Telemetry carries the current
            // mode and E-STOP state, so a skipped push is recovered there.
            let ev = match event_rx.recv().await {
                Ok(ev) => ev,
                Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                    debug!(conn_id, skipped = n, "supervisor event stream lagged");
                    continue;
                }
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
            };
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
                if tx_events.send(encode(&msg)).await.is_err() {
                    break;
                }
            }
        }
    });

    // Task: WS writer. Replies/events first, then the newest telemetry.
    let writer_task = tokio::spawn(async move {
        use futures::SinkExt;
        while let Some(bytes) = next_outbound(&mut reply_rx, &mut tele_rx).await {
            if let Err(e) = sink.send(Message::Binary(bytes.to_vec())).await {
                debug!(error = %e, "ws send error; closing");
                break;
            }
        }
    });

    // Reader loop: decode each frame once, then dispatch.
    use futures::StreamExt;
    while let Some(frame) = stream.next().await {
        match frame {
            Ok(Message::Binary(bytes)) => {
                let req = match <proto::ClientRuntimeMessage as bebop_proto::Message>::decode(
                    bytes.as_ref(),
                ) {
                    Ok(req) => req,
                    Err(e) => {
                        let reply = error_response(0, format!("decode error: {e}"));
                        if !queue_reply(&reply_tx, encode(&reply), false).await {
                            break;
                        }
                        continue;
                    }
                };
                // Connection-local side effects (telemetry subscription).
                match &req.payload {
                    Some(proto::client_runtime_message::Payload::SubscribeTelemetry(s)) => {
                        let rate_hz = if s.rate_hz == 0 {
                            default_rate_hz
                        } else {
                            s.rate_hz.min(max_rate_hz)
                        };
                        tele_cfg_tx.send_replace(TelemetryState {
                            subscribed: true,
                            rate_hz,
                        });
                        if !client_telemetry_subscribed {
                            sup.inc_telemetry_subscribers();
                            client_telemetry_subscribed = true;
                        }
                    }
                    Some(proto::client_runtime_message::Payload::UnsubscribeTelemetry(_)) => {
                        tele_cfg_tx.send_modify(|t| t.subscribed = false);
                        if client_telemetry_subscribed {
                            sup.dec_telemetry_subscribers();
                            client_telemetry_subscribed = false;
                        }
                    }
                    _ => {}
                }
                let is_twist = matches!(
                    req.payload,
                    Some(proto::client_runtime_message::Payload::SetVelocityCommand(
                        _
                    ))
                );
                if let Some(reply) = handle_client_message(&state, conn_id, req) {
                    if !queue_reply(&reply_tx, encode(&reply), is_twist).await {
                        break;
                    }
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

    // Release the seat and the telemetry-subscriber count first: nothing
    // in the flush below depends on them, and a stalled writer must not
    // delay the link-loss stop or keep other operators out of the seat.
    if client_telemetry_subscribed {
        sup.dec_telemetry_subscribers();
    }
    // Operator-link-loss stop: if this connection held the drive
    // assignment, its twist is now orphaned — the stop gesture that
    // would have ended the drive died with the socket. Release the
    // assignment and zero the twist so the chassis halts (motors stay
    // armed/balancing; recovery is automatic once any client sends
    // fresh commands). Viewers and rejected would-be drivers touched
    // nothing, so their disconnect is a no-op here (it only forgets the
    // connection's `ClientHello` name).
    sup.operator_disconnected(conn_id);

    // Stop the producers so the writer sees both channels close, then let
    // it flush what is already queued (bounded: a dead peer can't hold
    // this connection's cleanup hostage).
    telemetry_task.abort();
    event_task.abort();
    drop(reply_tx);
    let _ = telemetry_task.await;
    let _ = event_task.await;
    let mut writer_task = writer_task;
    if tokio::time::timeout(Duration::from_secs(2), &mut writer_task)
        .await
        .is_err()
    {
        writer_task.abort();
    }
    info!(conn_id, "ws client disconnected");
}

/// The writer's next outbound frame: a queued reply/event if there is
/// one (biased), else the newest telemetry frame once it changes. `None`
/// once the reply queue is closed (the connection is over); a closed
/// telemetry slot just leaves replies.
async fn next_outbound(
    replies: &mut mpsc::Receiver<Bytes>,
    telemetry: &mut watch::Receiver<Option<Bytes>>,
) -> Option<Bytes> {
    loop {
        tokio::select! {
            biased;
            reply = replies.recv() => return reply,
            changed = telemetry.changed() => {
                if changed.is_err() {
                    return replies.recv().await;
                }
                if let Some(frame) = telemetry.borrow_and_update().clone() {
                    return Some(frame);
                }
            }
        }
    }
}

/// Queue a reply for this connection's writer task; false once the
/// connection is gone.
///
/// The reply queue no longer carries telemetry, so it is normally empty.
/// A twist's reply (only ever a rejection, which the next twist repeats)
/// is still dropped when the queue is full: waiting would park this
/// reader, and the app streams twists without waiting for replies
/// (bebop-app `twistStream.ts`), so stops and E-STOP would pile up unread
/// behind it. Other replies wait: their callers await them, and they are
/// rare.
async fn queue_reply(tx: &mpsc::Sender<Bytes>, reply: Bytes, droppable: bool) -> bool {
    if droppable {
        !matches!(
            tx.try_send(reply),
            Err(mpsc::error::TrySendError::Closed(_))
        )
    } else {
        tx.send(reply).await.is_ok()
    }
}

#[cfg(test)]
mod reply_queue_tests {
    use super::*;

    fn frame(id: u8) -> Bytes {
        Bytes::from(vec![id])
    }

    #[tokio::test]
    async fn a_twist_reply_never_waits_for_a_full_queue() {
        let (tx, mut rx) = mpsc::channel::<Bytes>(1);
        assert!(queue_reply(&tx, frame(1), true).await);
        // Queue full: a twist reply is dropped at once instead of parking
        // the reader (this would hang if it awaited a slot).
        let res = tokio::time::timeout(
            std::time::Duration::from_millis(200),
            queue_reply(&tx, frame(2), true),
        )
        .await;
        assert_eq!(res.ok(), Some(true));
        assert_eq!(rx.recv().await, Some(frame(1)));
        assert!(
            rx.try_recv().is_err(),
            "the dropped reply must not be queued"
        );
    }

    #[tokio::test]
    async fn other_replies_wait_and_a_closed_queue_stops_the_reader() {
        let (tx, mut rx) = mpsc::channel::<Bytes>(1);
        assert!(queue_reply(&tx, frame(1), false).await);
        let pending = queue_reply(&tx, frame(2), false);
        tokio::pin!(pending);
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(50), &mut pending)
                .await
                .is_err(),
            "a non-twist reply waits for a slot"
        );
        assert_eq!(rx.recv().await, Some(frame(1)));
        assert!(pending.await);
        drop(rx);
        assert!(!queue_reply(&tx, frame(3), true).await);
        assert!(!queue_reply(&tx, frame(4), false).await);
    }

    #[tokio::test]
    async fn telemetry_keeps_only_the_newest_frame() {
        let (_reply_tx, mut reply_rx) = mpsc::channel::<Bytes>(4);
        let (tele_tx, mut tele_rx) = watch::channel::<Option<Bytes>>(None);
        // Writer stalled while three frames were produced: only the last
        // one is ever sent, then nothing until the next frame.
        for id in 1..=3 {
            tele_tx.send_replace(Some(frame(id)));
        }
        assert_eq!(
            next_outbound(&mut reply_rx, &mut tele_rx).await,
            Some(frame(3))
        );
        let idle = tokio::time::timeout(
            std::time::Duration::from_millis(50),
            next_outbound(&mut reply_rx, &mut tele_rx),
        )
        .await;
        assert!(idle.is_err(), "a sent telemetry frame is not resent");
        tele_tx.send_replace(Some(frame(4)));
        assert_eq!(
            next_outbound(&mut reply_rx, &mut tele_rx).await,
            Some(frame(4))
        );
    }

    #[tokio::test]
    async fn replies_go_out_before_pending_telemetry() {
        let (reply_tx, mut reply_rx) = mpsc::channel::<Bytes>(4);
        let (tele_tx, mut tele_rx) = watch::channel::<Option<Bytes>>(None);
        tele_tx.send_replace(Some(frame(100)));
        reply_tx.send(frame(1)).await.unwrap();
        reply_tx.send(frame(2)).await.unwrap();
        assert_eq!(
            next_outbound(&mut reply_rx, &mut tele_rx).await,
            Some(frame(1))
        );
        assert_eq!(
            next_outbound(&mut reply_rx, &mut tele_rx).await,
            Some(frame(2))
        );
        assert_eq!(
            next_outbound(&mut reply_rx, &mut tele_rx).await,
            Some(frame(100))
        );
    }

    #[tokio::test]
    async fn writer_ends_when_replies_close_even_without_telemetry_pump() {
        let (reply_tx, mut reply_rx) = mpsc::channel::<Bytes>(4);
        let (tele_tx, mut tele_rx) = watch::channel::<Option<Bytes>>(None);
        drop(tele_tx);
        reply_tx.send(frame(7)).await.unwrap();
        drop(reply_tx);
        assert_eq!(
            next_outbound(&mut reply_rx, &mut tele_rx).await,
            Some(frame(7))
        );
        assert_eq!(next_outbound(&mut reply_rx, &mut tele_rx).await, None);
    }
}

#[cfg(test)]
mod capture_tests {
    use super::*;

    #[test]
    fn safe_capture_path_rejects_traversal_and_non_mcap() {
        let dir = std::env::temp_dir();
        assert!(safe_capture_path(&dir, "../evil.mcap").is_err());
        assert!(safe_capture_path(&dir, "a/b.mcap").is_err());
        assert!(safe_capture_path(&dir, "a\\b.mcap").is_err());
        assert!(safe_capture_path(&dir, "notes.txt").is_err());
        assert!(safe_capture_path(&dir, "").is_err());
        assert!(safe_capture_path(&dir, "missing.mcap").is_err());
    }

    #[test]
    fn safe_capture_path_accepts_existing_mcap() {
        let dir = std::env::temp_dir().join(format!("bebop_ws_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("system_x.mcap");
        std::fs::write(&path, b"x").unwrap();
        assert_eq!(safe_capture_path(&dir, "system_x.mcap").unwrap(), path);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
