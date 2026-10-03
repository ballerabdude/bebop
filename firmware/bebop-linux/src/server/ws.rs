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

use crate::capture_control::CaptureControl;
use crate::imu::ImuShared;
use crate::nav_goal::NavGoalShared;
use crate::policy_control::PolicyControlShared;
use crate::policy_io::PolicyIoShared;
use crate::safety::{Supervisor, SupervisorEvent};
use crate::server::handlers::{encode, handle_client_message};
use crate::server::telemetry::{build_telemetry, telemetry_envelope};
use crate::vision::VisionShared;
use anyhow::Result;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Path as AxumPath, State};
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Redirect};
use axum::routing::{get, post};
use axum::Json;
use axum::Router;
use bebop_proto::runtime::v1 as proto;
use serde::Serialize;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, UNIX_EPOCH};
use tokio::sync::broadcast;
use tokio::sync::mpsc;
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
    /// Operator navigation goal (navd, plan §8). Written by the
    /// SetNavigationGoal handler; changes broadcast to all clients.
    pub nav_goal: Arc<NavGoalShared>,
    /// bebop-vision service control + status (see [`crate::vision`]).
    pub vision: VisionShared,
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
        .route("/captures/rerun/:name", get(convert_capture))
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
/// footer. The app calls this before downloading an `active` capture.
/// Waits (bounded) for the writers to open a fresh segment.
async fn finalize_captures(State(state): State<AppState>) -> impl IntoResponse {
    let control = &state.captures;
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
    Json(serde_json::json!({ "ok": true }))
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

/// Convert an MCAP to a Rerun `.rrd` (app id + dashboard) on the robot,
/// then redirect to the download URL. Cached: a fresh `.rrd` is served
/// without re-running the converter.
async fn convert_capture(
    State(state): State<AppState>,
    AxumPath(name): AxumPath<String>,
) -> impl IntoResponse {
    let mcap = match safe_capture_path(&state.capture_dir, &name) {
        Ok(p) => p,
        Err(e) => return (StatusCode::BAD_REQUEST, e).into_response(),
    };
    let stem = mcap
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("capture")
        .to_string();
    let rrd = mcap.with_file_name(format!("{stem}.rrd"));

    if !rrd_is_fresh(&mcap, &rrd) {
        let converter = state.rerun_converter.clone();
        if !converter.exists() {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                "Rerun conversion not set up on this robot (run scripts/setup-rerun.sh)",
            )
                .into_response();
        }
        let mcap_for_job = mcap.clone();
        let rrd_for_job = rrd.clone();
        let dir_for_job = state.capture_dir.clone();
        let run = tokio::task::spawn_blocking(move || {
            prune_rrds(&dir_for_job, RRD_BUDGET_BYTES, &rrd_for_job);
            std::process::Command::new(&converter)
                .arg(&mcap_for_job)
                .arg(&rrd_for_job)
                .output()
                .map_err(|e| format!("spawn {}: {e}", converter.display()))
        })
        .await;
        match run {
            Ok(Ok(out)) if out.status.success() => {}
            Ok(Ok(out)) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!(
                        "converter exited {}: {}",
                        out.status,
                        String::from_utf8_lossy(&out.stderr).trim()
                    ),
                )
                    .into_response();
            }
            Ok(Err(e)) => return (StatusCode::INTERNAL_SERVER_ERROR, e).into_response(),
            Err(e) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("conversion task failed: {e}"),
                )
                    .into_response();
            }
        }
    }

    let location = format!("/captures/dl/{stem}.rrd");
    Redirect::temporary(&location).into_response()
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
        nav_goal,
        vision,
        capture_dir: _,
        captures: _,
        rerun_converter: _,
    } = state;
    // Per-connection identity for operator arbitration. Monotonic so a
    // reconnecting client never inherits a stale assignment; never 0
    // (0 reads as "no connection" in the per-connection telemetry
    // flags).
    let conn_id = WS_CONN_COUNTER.fetch_add(1, Ordering::SeqCst) + 1;
    info!(conn_id, "ws client connected");
    let (mut sink, mut stream) = socket.split();
    let (tx, mut rx) = mpsc::channel::<proto::ServerRuntimeMessage>(256);

    // Post-connect flush: the current navigation goal, so late joiners
    // (e.g. a restarting navd process) see the active goal without a
    // snapshot round-trip.
    let _ = tx
        .send(proto::ServerRuntimeMessage {
            request_id: 0,
            payload: Some(proto::server_runtime_message::Payload::NavGoal(
                nav_goal.get().to_proto(),
            )),
        })
        .await;

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
    let policy_io_tele = policy_io.clone();
    let vision_tele = vision.clone();
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
            // conn_id rides along so the arbitration flags inside the
            // drive state are computed for *this* client ("you").
            let frame = build_telemetry(
                &sup_tele,
                &imu_tele,
                imu_present,
                &policy_io_tele,
                &vision_tele,
                conn_id,
            );
            let env = telemetry_envelope(frame);
            if tx_tele.send(env).await.is_err() {
                break;
            }
        }
    });

    // Task: nav-goal change pump. Every client sees goal changes (the
    // navd goal-drive process consumes them; the app mirrors state).
    let tx_goal = tx.clone();
    let mut goal_rx = nav_goal.subscribe();
    let goal_task = tokio::spawn(async move {
        loop {
            match goal_rx.recv().await {
                Ok(goal) => {
                    let msg = proto::ServerRuntimeMessage {
                        request_id: 0,
                        payload: Some(proto::server_runtime_message::Payload::NavGoal(
                            goal.to_proto(),
                        )),
                    };
                    if tx_goal.send(msg).await.is_err() {
                        break;
                    }
                }
                Err(broadcast::error::RecvError::Lagged(n)) => {
                    debug!(skipped = n, "nav-goal broadcast lagged");
                }
                Err(broadcast::error::RecvError::Closed) => break,
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
                    &nav_goal,
                    &vision,
                    conn_id,
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
    goal_task.abort();
    event_task.abort();
    if client_telemetry_subscribed {
        sup.dec_telemetry_subscribers();
    }
    // Operator-link-loss stop: if this connection held the drive
    // assignment, its twist is now orphaned — the stop gesture that
    // would have ended the drive died with the socket. Release the
    // assignment and zero the twist so the chassis halts (motors stay
    // armed/balancing; recovery is automatic once any client sends
    // fresh commands). Viewers and rejected would-be drivers touched
    // nothing, so their disconnect is a no-op here.
    sup.operator_disconnected(conn_id);
    info!(conn_id, "ws client disconnected");
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
