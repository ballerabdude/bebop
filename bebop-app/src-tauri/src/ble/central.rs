//! Native BLE central built on `btleplug`.
//!
//! Wire format mirrors `bebop-app/src/ble/webBluetoothTransport.ts`
//! and the agent in `jetson-agent/bebop-agent/src/ble/server.rs`:
//!   * scan for peripherals advertising the Bebop service UUID
//!   * connect, discover services + characteristics
//!   * subscribe to the response characteristic for notifications
//!   * exchange `bebop.v1.ClientRequest` / `bebop.v1.AgentResponse`
//!     protobuf messages, framed by `super::framing`
//!
//! All Bebop knowledge stops here; the rest of the Rust side just calls
//! `BleManager::request(...)` with a `ClientRequest::payload` arm and
//! gets back a fully-decoded `AgentResponse`.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use bebop_proto::v1::{client_request, AgentResponse, ClientRequest, ResponseStatus};
use btleplug::api::{Central, Manager as _, Peripheral as _, ScanFilter, WriteType};
use btleplug::platform::{Adapter, Manager, Peripheral, PeripheralId};
use futures::StreamExt;
use prost::Message;
use serde::Serialize;
use tokio::sync::{oneshot, Mutex};
use uuid::{uuid, Uuid};

use super::framing::{encode, Reassembler};

const SERVICE_UUID: Uuid = uuid!("b3b0b000-0b3b-4f9b-9b3b-b3b0b3b0b3b0");
const CHAR_REQUEST_UUID: Uuid = uuid!("b3b0b001-0b3b-4f9b-9b3b-b3b0b3b0b3b0");
const CHAR_RESPONSE_UUID: Uuid = uuid!("b3b0b002-0b3b-4f9b-9b3b-b3b0b3b0b3b0");
const CHAR_STATUS_UUID: Uuid = uuid!("b3b0b003-0b3b-4f9b-9b3b-b3b0b3b0b3b0");

/// Conservative default well below the typical negotiated ATT MTU.
const MAX_PAYLOAD: usize = 128;
const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(15);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

#[derive(Debug, Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct DiscoveredRobot {
    pub id: String,
    pub name: String,
    pub rssi: i32,
}

type PendingMap = Arc<Mutex<HashMap<u32, oneshot::Sender<AgentResponse>>>>;

struct Connection {
    peripheral: Peripheral,
    request_char: btleplug::api::Characteristic,
    next_request_id: u32,
    pending: PendingMap,
    notify_task: tokio::task::JoinHandle<()>,
}

/// Process-wide BLE state. Held by Tauri as a managed state.
#[derive(Default)]
pub struct BleManager {
    adapter: Mutex<Option<Adapter>>,
    discovered: Mutex<HashMap<String, PeripheralId>>,
    connection: Mutex<Option<Connection>>,
}

impl BleManager {
    async fn adapter(&self) -> Result<Adapter, String> {
        let mut guard = self.adapter.lock().await;
        if let Some(a) = guard.as_ref() {
            return Ok(a.clone());
        }
        let manager = Manager::new().await.map_err(|e| e.to_string())?;
        let adapters = manager.adapters().await.map_err(|e| e.to_string())?;
        let adapter = adapters
            .into_iter()
            .next()
            .ok_or_else(|| "no Bluetooth adapter found".to_string())?;
        *guard = Some(adapter.clone());
        Ok(adapter)
    }

    pub async fn scan(&self, timeout_ms: u32) -> Result<Vec<DiscoveredRobot>, String> {
        let adapter = self.adapter().await?;
        adapter
            .start_scan(ScanFilter {
                services: vec![SERVICE_UUID],
            })
            .await
            .map_err(|e| e.to_string())?;

        let dwell = Duration::from_millis(timeout_ms.max(500) as u64);
        tokio::time::sleep(dwell).await;
        let _ = adapter.stop_scan().await;

        let peripherals = adapter.peripherals().await.map_err(|e| e.to_string())?;
        let mut out = Vec::new();
        let mut map = self.discovered.lock().await;
        map.clear();
        for p in peripherals {
            let Some(props) = p.properties().await.map_err(|e| e.to_string())? else {
                continue;
            };
            if !props.services.contains(&SERVICE_UUID) {
                continue;
            }
            let id = p.id().to_string();
            map.insert(id.clone(), p.id());
            out.push(DiscoveredRobot {
                id,
                name: props
                    .local_name
                    .unwrap_or_else(|| "Unknown Bebop".to_string()),
                rssi: props.rssi.map(i32::from).unwrap_or(0),
            });
        }
        Ok(out)
    }

    pub async fn connect(&self, robot_id: String) -> Result<(), String> {
        let adapter = self.adapter().await?;
        let pid = {
            let map = self.discovered.lock().await;
            map.get(&robot_id).cloned()
        }
        .ok_or_else(|| "unknown robot id; scan first".to_string())?;

        let peripheral = adapter.peripheral(&pid).await.map_err(|e| e.to_string())?;
        peripheral
            .connect_with_timeout(CONNECT_TIMEOUT)
            .await
            .map_err(|e| e.to_string())?;
        peripheral
            .discover_services()
            .await
            .map_err(|e| e.to_string())?;

        let chars = peripheral.characteristics();
        let request_char = chars
            .iter()
            .find(|c| c.uuid == CHAR_REQUEST_UUID)
            .cloned()
            .ok_or_else(|| "request characteristic missing".to_string())?;
        let response_char = chars
            .iter()
            .find(|c| c.uuid == CHAR_RESPONSE_UUID)
            .cloned()
            .ok_or_else(|| "response characteristic missing".to_string())?;
        let status_char = chars.iter().find(|c| c.uuid == CHAR_STATUS_UUID).cloned();

        peripheral
            .subscribe(&response_char)
            .await
            .map_err(|e| e.to_string())?;
        if let Some(c) = status_char.as_ref() {
            // Status notifications are advisory; not all devices expose them.
            let _ = peripheral.subscribe(c).await;
        }

        // Open the notification stream BEFORE returning so the very first
        // request after connect() is guaranteed to have a listener. In
        // btleplug, `notifications()` returns a fresh receiver on a
        // per-peripheral broadcast channel — notifications that arrive
        // before the receiver exists are not buffered, which would let
        // the response to the first request go to /dev/null and hang the
        // caller until its timeout.
        let stream = peripheral
            .notifications()
            .await
            .map_err(|e| e.to_string())?;

        let pending: PendingMap = Arc::new(Mutex::new(HashMap::new()));
        let notify_task = spawn_notification_loop(stream, pending.clone());

        let mut guard = self.connection.lock().await;
        if let Some(prev) = guard.take() {
            prev.notify_task.abort();
            let _ = prev.peripheral.disconnect().await;
        }
        *guard = Some(Connection {
            peripheral,
            request_char,
            next_request_id: 1,
            pending,
            notify_task,
        });
        Ok(())
    }

    pub async fn disconnect(&self) -> Result<(), String> {
        let mut guard = self.connection.lock().await;
        if let Some(conn) = guard.take() {
            conn.notify_task.abort();
            let _ = conn.peripheral.disconnect().await;
        }
        Ok(())
    }

    #[allow(dead_code)]
    pub async fn is_connected(&self) -> bool {
        let g = self.connection.lock().await;
        match g.as_ref() {
            Some(c) => c.peripheral.is_connected().await.unwrap_or(false),
            None => false,
        }
    }

    /// Send a `ClientRequest` payload and wait for the matching response.
    ///
    /// `request_id` is injected automatically. Returns the full
    /// `AgentResponse` so callers can inspect both the status and the
    /// payload oneof. If the agent answered with a non-OK status, this
    /// returns `Err(message)` (mirrors the TS transport's behaviour).
    pub async fn request(&self, payload: client_request::Payload) -> Result<AgentResponse, String> {
        self.request_with_timeout(payload, DEFAULT_REQUEST_TIMEOUT)
            .await
    }

    /// Like [`request`], but with a caller-supplied timeout. Useful for
    /// agent calls that legitimately take longer than the default (e.g.
    /// `nmcli wifi list --rescan yes` can take 20+ seconds in busy RF
    /// environments).
    pub async fn request_with_timeout(
        &self,
        payload: client_request::Payload,
        timeout: Duration,
    ) -> Result<AgentResponse, String> {
        let (request_id, peripheral, request_char, pending) = {
            let mut guard = self.connection.lock().await;
            let conn = guard.as_mut().ok_or_else(|| "not connected".to_string())?;
            let id = conn.next_request_id;
            conn.next_request_id = conn.next_request_id.wrapping_add(1).max(1);
            (
                id,
                conn.peripheral.clone(),
                conn.request_char.clone(),
                conn.pending.clone(),
            )
        };

        let req = ClientRequest {
            request_id,
            payload: Some(payload),
        };
        let bytes = req.encode_to_vec();

        let (tx, rx) = oneshot::channel();
        pending.lock().await.insert(request_id, tx);

        for frame in encode(&bytes, MAX_PAYLOAD) {
            if let Err(e) = peripheral
                .write(&request_char, &frame, WriteType::WithoutResponse)
                .await
            {
                pending.lock().await.remove(&request_id);
                return Err(e.to_string());
            }
        }

        let resp = match tokio::time::timeout(timeout, rx).await {
            Ok(Ok(resp)) => resp,
            Ok(Err(_)) => return Err("response channel closed".into()),
            Err(_) => {
                pending.lock().await.remove(&request_id);
                return Err("request timed out".into());
            }
        };

        if resp.status == ResponseStatus::Ok as i32 {
            Ok(resp)
        } else {
            let label = ResponseStatus::try_from(resp.status)
                .map(|s| format!("{s:?}"))
                .unwrap_or_else(|_| resp.status.to_string());
            let msg = if resp.message.is_empty() {
                format!("agent returned status {label}")
            } else {
                resp.message.clone()
            };
            Err(msg)
        }
    }
}

fn spawn_notification_loop<S>(stream: S, pending: PendingMap) -> tokio::task::JoinHandle<()>
where
    S: futures::Stream<Item = btleplug::api::ValueNotification> + Send + Unpin + 'static,
{
    tokio::spawn(async move {
        let mut reassembler = Reassembler::new();
        let mut stream = stream;
        while let Some(notification) = stream.next().await {
            if notification.uuid != CHAR_RESPONSE_UUID {
                // Status push; not routed through the request/response map.
                continue;
            }
            let complete = match reassembler.push(&notification.value) {
                Ok(Some(c)) => c,
                Ok(None) => continue,
                Err(_) => {
                    reassembler = Reassembler::new();
                    continue;
                }
            };
            let resp = match AgentResponse::decode(complete.as_slice()) {
                Ok(r) => r,
                Err(_) => continue,
            };
            if resp.request_id == 0 {
                // Unsolicited push (e.g. status snapshot). Drop here; the
                // status characteristic delivers these to UI consumers.
                continue;
            }
            if let Some(tx) = pending.lock().await.remove(&resp.request_id) {
                let _ = tx.send(resp);
            }
        }
    })
}
