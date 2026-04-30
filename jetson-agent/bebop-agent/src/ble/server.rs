//! BLE GATT server implementation on top of BlueZ via the `bluer` crate.
//!
//! This file owns the event loop: it sets up the adapter, publishes our
//! custom service, and wires read/write/notify callbacks to the shared
//! [`AppState`].
//!
//! The happy path looks like:
//!
//! 1. `serve()` is called by `main`.
//! 2. Adapter is powered on; legacy pairing / classic discoverability is off.
//! 3. We register the Bebop primary service with three characteristics
//!    (`request`, `response`, `status`) using UUIDs from [`super::uuids`].
//! 4. Advertising is started with the configured local name.
//! 5. Incoming writes on the `request` characteristic are reassembled via
//!    [`super::framing::Reassembler`], decoded into
//!    [`bebop_proto::v1::ClientRequest`], dispatched via
//!    [`super::dispatcher::handle`], and the response is encoded + fragmented
//!    + pushed through the `response` characteristic's Notify channel.
//! 6. The `status` characteristic is a periodic snapshot the phone subscribes
//!    to so its UI stays live without polling.
//!
//! Concurrency notes:
//!
//! * BlueZ multiplexes multiple ATT connections on a single `bluer`
//!   characteristic IO socket. We treat the latest accepted reader/writer as
//!   "the" client; for our use case (one phone provisioning one robot) this
//!   is sufficient. A future improvement could keep a map keyed by
//!   `(adapter, address)` and dispatch concurrently.

use std::time::Duration;

use anyhow::{Context, Result};
use bebop_proto::{
    v1::{agent_response, AgentResponse, ClientRequest, ResponseStatus},
    Message,
};
use bluer::{
    adv::Advertisement,
    gatt::{
        local::{
            characteristic_control, Application, Characteristic, CharacteristicControlEvent,
            CharacteristicNotify, CharacteristicNotifyMethod, CharacteristicRead,
            CharacteristicWrite, CharacteristicWriteMethod, ReqError, Service,
        },
        CharacteristicReader, CharacteristicWriter,
    },
    Adapter, Session,
};
use futures::{future, pin_mut, FutureExt, StreamExt};
use prost::bytes::Bytes;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tracing::{debug, info, warn};

use crate::state::AppState;

use super::{
    dispatcher,
    framing::{self, Reassembler},
};

/// How often to push a status snapshot to subscribed clients.
const STATUS_NOTIFY_INTERVAL: Duration = Duration::from_secs(5);

/// Conservative ceiling for a single ATT notification payload. The negotiated
/// MTU is reported by the IO socket; we subtract the 4-byte framing header
/// from it (and a few bytes of ATT overhead) to derive how much protobuf
/// payload fits in one frame. Falls back to ~20 bytes if MTU is unavailable.
fn payload_per_frame(mtu: usize) -> usize {
    // ATT overhead is 3 bytes for the notification opcode/handle; bluer's
    // reported `mtu()` already excludes that, but we leave a small cushion
    // to be safe across stacks.
    mtu.saturating_sub(framing::HEADER_LEN + 3).max(16)
}

pub async fn serve(state: AppState) -> Result<()> {
    let session = Session::new().await.context("connect to bluez")?;
    let adapter = acquire_adapter(&session, state.config().await.ble.adapter.as_deref()).await?;
    adapter
        .set_powered(true)
        .await
        .context("power on adapter")?;
    adapter
        .set_discoverable(false)
        .await
        .context("disable classic discoverability")?;

    let adapter_address = adapter
        .address()
        .await
        .context("read adapter address")?;
    info!(
        adapter = %adapter.name(),
        address = %adapter_address,
        "BLE adapter ready"
    );

    // Build per-characteristic control handles. Request and response use the
    // low-overhead IO transport (raw socket per ATT op); status uses the
    // callback transport since it's just a small periodic blob.
    let (req_control, req_handle) = characteristic_control();
    let (resp_control, resp_handle) = characteristic_control();

    let app = Application {
        services: vec![Service {
            uuid: super::SERVICE_UUID,
            primary: true,
            characteristics: vec![
                request_characteristic(req_handle),
                response_characteristic(resp_handle),
                status_characteristic(state.clone()),
            ],
            ..Default::default()
        }],
        ..Default::default()
    };

    let app_handle = adapter
        .serve_gatt_application(app)
        .await
        .context("register GATT application")?;
    info!(service = %super::SERVICE_UUID, "BLE GATT application registered");

    // Start advertising once the GATT app is live so scanners that connect
    // immediately don't race against service registration.
    let cfg = state.config().await;
    let local_name = cfg.ble.local_name.clone();
    let adv = Advertisement {
        service_uuids: [super::SERVICE_UUID].into_iter().collect(),
        discoverable: Some(true),
        local_name: Some(local_name.clone()),
        ..Default::default()
    };
    let adv_handle = adapter
        .advertise(adv)
        .await
        .context("start advertising")?;
    info!(name = %local_name, "BLE advertising started");

    // Pump the request/response IO loop forever.
    let result = run_io_loop(state, req_control, resp_control).await;

    // Tear down explicitly so the OS releases the adapter cleanly even if
    // the IO loop returned early.
    drop(adv_handle);
    drop(app_handle);

    result
}

/// Long-running event loop: accepts the latest BLE reader/writer, reassembles
/// incoming protobuf messages, dispatches them, and writes back fragmented
/// responses.
async fn run_io_loop(
    state: AppState,
    req_control: bluer::gatt::local::CharacteristicControl,
    resp_control: bluer::gatt::local::CharacteristicControl,
) -> Result<()> {
    pin_mut!(req_control);
    pin_mut!(resp_control);

    let mut reader: Option<CharacteristicReader> = None;
    let mut writer: Option<CharacteristicWriter> = None;
    let mut reassembler = Reassembler::new();
    // Sized to the negotiated MTU at accept-time.
    let mut read_buf: Vec<u8> = Vec::new();

    loop {
        tokio::select! {
            // A new client opened the request characteristic for writing.
            evt = req_control.next() => {
                match evt {
                    Some(CharacteristicControlEvent::Write(req)) => {
                        let mtu = req.mtu();
                        info!(mtu, addr = %req.device_address(), "request channel opened");
                        match req.accept() {
                            Ok(r) => {
                                read_buf = vec![0u8; mtu.max(64)];
                                reader = Some(r);
                                reassembler = Reassembler::new();
                            }
                            Err(e) => warn!(error = %e, "failed to accept request IO"),
                        }
                    }
                    Some(CharacteristicControlEvent::Notify(_)) => {
                        // Request char shouldn't get notify subscriptions.
                        warn!("unexpected notify subscription on request characteristic");
                    }
                    None => {
                        warn!("request control stream ended");
                        return Ok(());
                    }
                }
            }

            // A new client subscribed to response notifications.
            evt = resp_control.next() => {
                match evt {
                    Some(CharacteristicControlEvent::Notify(notifier)) => {
                        info!(mtu = notifier.mtu(), "response channel opened (notify)");
                        writer = Some(notifier);
                    }
                    Some(CharacteristicControlEvent::Write(req)) => {
                        warn!(addr = %req.device_address(), "ignoring write on response characteristic");
                        req.reject(ReqError::NotSupported);
                    }
                    None => {
                        warn!("response control stream ended");
                        return Ok(());
                    }
                }
            }

            // Bytes available on the request characteristic.
            n = async {
                match reader.as_mut() {
                    Some(r) => r.read(&mut read_buf).await,
                    None => future::pending().await,
                }
            } => {
                match n {
                    Ok(0) => {
                        debug!("request reader EOF");
                        reader = None;
                        reassembler = Reassembler::new();
                    }
                    Ok(n) => {
                        let frame = &read_buf[..n];
                        if let Err(e) = handle_inbound_frame(
                            &state,
                            frame,
                            &mut reassembler,
                            writer.as_mut(),
                        ).await {
                            warn!(error = %e, "error handling inbound frame");
                            // Reset reassembly state so a corrupt fragment
                            // doesn't poison future messages.
                            reassembler = Reassembler::new();
                        }
                    }
                    Err(e) => {
                        warn!(error = %e, "request reader error; dropping client");
                        reader = None;
                        reassembler = Reassembler::new();
                    }
                }
            }
        }
    }
}

/// Push one inbound BLE frame into the reassembler. If a complete message
/// emerges, decode -> dispatch -> encode -> fragment -> write back.
async fn handle_inbound_frame(
    state: &AppState,
    frame: &[u8],
    reassembler: &mut Reassembler,
    writer: Option<&mut CharacteristicWriter>,
) -> Result<()> {
    let Some(message_bytes) = reassembler.push(frame)? else {
        // Mid-fragment, nothing to do yet.
        return Ok(());
    };

    let req = match ClientRequest::decode(Bytes::from(message_bytes)) {
        Ok(r) => r,
        Err(e) => {
            warn!(error = %e, "failed to decode ClientRequest");
            // Best-effort: still send back an error frame so the phone knows.
            let resp = AgentResponse {
                request_id: 0,
                status: ResponseStatus::Error as i32,
                message: format!("decode error: {e}"),
                payload: None,
            };
            return send_response(resp, writer).await;
        }
    };

    debug!(request_id = req.request_id, "dispatching request");
    let response = dispatcher::handle(state, req).await;
    send_response(response, writer).await
}

/// Encode + fragment + write a response, if a notifier is currently attached.
async fn send_response(
    response: AgentResponse,
    writer: Option<&mut CharacteristicWriter>,
) -> Result<()> {
    let Some(writer) = writer else {
        warn!(
            request_id = response.request_id,
            "no active response notifier; dropping response"
        );
        return Ok(());
    };

    let mut buf = Vec::with_capacity(response.encoded_len());
    response.encode(&mut buf).context("encode AgentResponse")?;

    let frames = framing::encode(&buf, payload_per_frame(writer.mtu()));
    for frame in frames {
        writer
            .write_all(&frame)
            .await
            .context("notify response frame")?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Characteristic factories
// ---------------------------------------------------------------------------

fn request_characteristic(
    handle: bluer::gatt::local::CharacteristicControlHandle,
) -> Characteristic {
    Characteristic {
        uuid: super::CHAR_REQUEST_UUID,
        write: Some(CharacteristicWrite {
            write: true,
            write_without_response: true,
            method: CharacteristicWriteMethod::Io,
            ..Default::default()
        }),
        control_handle: handle,
        ..Default::default()
    }
}

fn response_characteristic(
    handle: bluer::gatt::local::CharacteristicControlHandle,
) -> Characteristic {
    Characteristic {
        uuid: super::CHAR_RESPONSE_UUID,
        notify: Some(CharacteristicNotify {
            notify: true,
            method: CharacteristicNotifyMethod::Io,
            ..Default::default()
        }),
        control_handle: handle,
        ..Default::default()
    }
}

fn status_characteristic(state: AppState) -> Characteristic {
    let read_state = state.clone();
    let notify_state = state;

    Characteristic {
        uuid: super::CHAR_STATUS_UUID,
        read: Some(CharacteristicRead {
            read: true,
            fun: Box::new(move |_req| {
                let s = read_state.clone();
                async move {
                    match status_snapshot(&s).await {
                        Ok(bytes) => Ok(bytes),
                        Err(e) => {
                            warn!(error = %e, "status read failed");
                            Err(ReqError::Failed)
                        }
                    }
                }
                .boxed()
            }),
            ..Default::default()
        }),
        notify: Some(CharacteristicNotify {
            notify: true,
            method: CharacteristicNotifyMethod::Fun(Box::new(move |notifier| {
                let s = notify_state.clone();
                async move {
                    tokio::spawn(status_notify_loop(s, notifier));
                }
                .boxed()
            })),
            ..Default::default()
        }),
        ..Default::default()
    }
}

/// Periodic push loop: every `STATUS_NOTIFY_INTERVAL`, send the latest
/// app/wifi/ota status snapshot. Exits cleanly when the subscriber goes away.
async fn status_notify_loop(
    state: AppState,
    mut notifier: bluer::gatt::local::CharacteristicNotifier,
) {
    let mut ticker = tokio::time::interval(STATUS_NOTIFY_INTERVAL);
    // Fire once immediately so subscribers see fresh data on connect.
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

    loop {
        ticker.tick().await;
        if notifier.is_stopped() {
            debug!("status notify session stopped by peer");
            break;
        }
        match status_snapshot(&state).await {
            Ok(bytes) => {
                if let Err(e) = notifier.notify(bytes).await {
                    debug!(error = %e, "status notify ended");
                    break;
                }
            }
            Err(e) => warn!(error = %e, "failed to build status snapshot"),
        }
    }
}

/// Build the on-the-wire status snapshot. Currently we surface the
/// `AppStatus` proto wrapped in an `AgentResponse` envelope (with
/// `request_id == 0` to indicate "unsolicited push") so the mobile app can
/// reuse its existing decoder. Wi-Fi and OTA snapshots are pushed via the
/// notify loop on subsequent ticks; for the read-on-demand path we always
/// return the `AppStatus` view since it changes most often.
async fn status_snapshot(state: &AppState) -> Result<Vec<u8>> {
    use bebop_proto::v1::{AppStatus, ResponseStatus};

    let app = state.app_status().await;
    let cfg = state.config().await;
    let payload = AppStatus {
        app_name: cfg.app.name,
        image: app.image,
        image_digest: app.image_digest,
        state: app_lifecycle_to_proto(app.state) as i32,
        container_id: app.container_id,
        started_at_unix: app.started_at_unix,
        restart_count: app.restart_count,
    };
    let envelope = AgentResponse {
        request_id: 0,
        status: ResponseStatus::Ok as i32,
        message: String::new(),
        payload: Some(agent_response::Payload::AppStatus(payload)),
    };
    let mut buf = Vec::with_capacity(envelope.encoded_len());
    envelope.encode(&mut buf).context("encode status snapshot")?;
    Ok(buf)
}

fn app_lifecycle_to_proto(s: crate::state::AppLifecycle) -> bebop_proto::v1::AppState {
    use crate::state::AppLifecycle as L;
    use bebop_proto::v1::AppState as P;
    match s {
        L::Stopped => P::Stopped,
        L::Starting => P::Starting,
        L::Running => P::Running,
        L::Crashed => P::Crashed,
        L::Updating => P::Updating,
    }
}

async fn acquire_adapter(session: &Session, preferred: Option<&str>) -> Result<Adapter> {
    if let Some(name) = preferred {
        return session
            .adapter(name)
            .with_context(|| format!("acquire adapter {name}"));
    }
    let default = session
        .default_adapter()
        .await
        .context("no default BLE adapter available")?;
    Ok(default)
}
