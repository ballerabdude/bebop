//! Tauri commands fronting a native BLE central.
//!
//! `central::BleManager` owns the actual `btleplug` connection and is held
//! by Tauri as managed state. Each command here is a thin wrapper that:
//!   * builds a `bebop_proto::v1::ClientRequest` payload variant,
//!   * sends it via `BleManager::request`,
//!   * extracts the matching `AgentResponse` `oneof` arm, and
//!   * converts the prost types into camelCase serde DTOs the frontend
//!     can consume directly.
//!
//! The wire is protobuf end-to-end (TS app ↔ this layer ↔ Jetson agent);
//! these structs only exist to give the JS side a stable, typed shape.

mod central;
mod framing;

pub use central::BleManager;

use bebop_proto::v1::{
    agent_response, client_request, AppCommand as ProtoAppCommand, AppState as ProtoAppState,
    ControlAppRequest, GetAppStatusRequest, GetDeviceInfoRequest, GetOtaStatusRequest,
    GetRobotConfigRequest, GetWifiStatusRequest, OtaState as ProtoOtaState,
    RobotConfig as ProtoRobotConfig, ScanWifiRequest, SetRobotConfigRequest,
    SetWifiCredentialsRequest, TriggerOtaRequest,
};
use serde::{Deserialize, Serialize};
use tauri::State;

use central::DiscoveredRobot as CentralDiscoveredRobot;

// ---- DTOs (camelCase so the frontend can consume them as-is) --------------

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DiscoveredRobot {
    pub id: String,
    pub name: String,
    pub rssi: i32,
}

impl From<CentralDiscoveredRobot> for DiscoveredRobot {
    fn from(r: CentralDiscoveredRobot) -> Self {
        Self {
            id: r.id,
            name: r.name,
            rssi: r.rssi,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DeviceInfo {
    pub serial_number: String,
    pub model: String,
    pub agent_version: String,
    pub jetpack_version: String,
    pub hostname: String,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WifiNetwork {
    pub ssid: String,
    pub signal_dbm: i32,
    pub security: String,
    pub saved: bool,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WifiStatus {
    pub connected: bool,
    pub ssid: String,
    pub ip_address: String,
    pub signal_dbm: i32,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RobotConfig {
    pub robot_name: String,
    pub owner_id: String,
    pub timezone: String,
    pub extra: std::collections::HashMap<String, String>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AppStatus {
    pub app_name: String,
    pub image: String,
    pub image_digest: String,
    pub state: String,
    pub container_id: String,
    pub started_at_unix: i64,
    pub restart_count: i32,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OtaStatus {
    pub state: String,
    pub current_image: String,
    pub target_image: String,
    pub progress_percent: u32,
    pub error: String,
}

// ---- helpers --------------------------------------------------------------

fn expect_payload<T, F>(
    resp: bebop_proto::v1::AgentResponse,
    expected: &'static str,
    extract: F,
) -> Result<T, String>
where
    F: FnOnce(agent_response::Payload) -> Option<T>,
{
    let case_name = match resp.payload.as_ref() {
        Some(agent_response::Payload::DeviceInfo(_)) => "deviceInfo",
        Some(agent_response::Payload::WifiScanResult(_)) => "wifiScanResult",
        Some(agent_response::Payload::WifiStatus(_)) => "wifiStatus",
        Some(agent_response::Payload::RobotConfig(_)) => "robotConfig",
        Some(agent_response::Payload::AppStatus(_)) => "appStatus",
        Some(agent_response::Payload::OtaStatus(_)) => "otaStatus",
        None => "<none>",
    };
    let payload = resp
        .payload
        .ok_or_else(|| format!("agent returned no payload (expected {expected})"))?;
    extract(payload).ok_or_else(|| {
        format!("agent returned unexpected payload: {case_name} (expected {expected})")
    })
}

fn app_state_label(s: i32) -> String {
    match ProtoAppState::try_from(s).unwrap_or(ProtoAppState::Unspecified) {
        ProtoAppState::Stopped => "STOPPED",
        ProtoAppState::Starting => "STARTING",
        ProtoAppState::Running => "RUNNING",
        ProtoAppState::Crashed => "CRASHED",
        ProtoAppState::Updating => "UPDATING",
        ProtoAppState::Unspecified => "UNSPECIFIED",
    }
    .to_string()
}

fn ota_state_label(s: i32) -> String {
    match ProtoOtaState::try_from(s).unwrap_or(ProtoOtaState::Unspecified) {
        ProtoOtaState::Idle => "IDLE",
        ProtoOtaState::Checking => "CHECKING",
        ProtoOtaState::Downloading => "DOWNLOADING",
        ProtoOtaState::Applying => "APPLYING",
        ProtoOtaState::Success => "SUCCESS",
        ProtoOtaState::Failed => "FAILED",
        ProtoOtaState::Unspecified => "UNSPECIFIED",
    }
    .to_string()
}

fn app_command_from_label(c: &str) -> ProtoAppCommand {
    match c {
        "START" => ProtoAppCommand::Start,
        "STOP" => ProtoAppCommand::Stop,
        "RESTART" => ProtoAppCommand::Restart,
        _ => ProtoAppCommand::Unspecified,
    }
}

// ---- commands -------------------------------------------------------------

#[tauri::command]
pub async fn ble_scan(
    state: State<'_, BleManager>,
    timeout_ms: u32,
) -> Result<Vec<DiscoveredRobot>, String> {
    Ok(state
        .scan(timeout_ms)
        .await?
        .into_iter()
        .map(Into::into)
        .collect())
}

#[tauri::command]
pub async fn ble_connect(state: State<'_, BleManager>, robot_id: String) -> Result<(), String> {
    state.connect(robot_id).await
}

#[tauri::command]
pub async fn ble_disconnect(state: State<'_, BleManager>) -> Result<(), String> {
    state.disconnect().await
}

#[tauri::command]
pub async fn ble_get_device_info(state: State<'_, BleManager>) -> Result<DeviceInfo, String> {
    let resp = state
        .request(client_request::Payload::GetDeviceInfo(
            GetDeviceInfoRequest {},
        ))
        .await?;
    let info = expect_payload(resp, "deviceInfo", |p| match p {
        agent_response::Payload::DeviceInfo(v) => Some(v),
        _ => None,
    })?;
    Ok(DeviceInfo {
        serial_number: info.serial_number,
        model: info.model,
        agent_version: info.agent_version,
        jetpack_version: info.jetpack_version,
        hostname: info.hostname,
    })
}

#[tauri::command]
pub async fn ble_scan_wifi(state: State<'_, BleManager>) -> Result<Vec<WifiNetwork>, String> {
    // nmcli runs a fresh radio scan (`--rescan yes`) which routinely takes
    // 10-25s in busy environments. Give it more headroom than the default.
    let resp = state
        .request_with_timeout(
            client_request::Payload::ScanWifi(ScanWifiRequest {}),
            std::time::Duration::from_secs(30),
        )
        .await?;
    let result = expect_payload(resp, "wifiScanResult", |p| match p {
        agent_response::Payload::WifiScanResult(v) => Some(v),
        _ => None,
    })?;
    Ok(result
        .networks
        .into_iter()
        .map(|n| WifiNetwork {
            ssid: n.ssid,
            signal_dbm: n.signal_dbm,
            security: n.security,
            saved: n.saved,
        })
        .collect())
}

#[tauri::command]
pub async fn ble_set_wifi_credentials(
    state: State<'_, BleManager>,
    ssid: String,
    password: String,
    hidden: bool,
) -> Result<(), String> {
    state
        .request(client_request::Payload::SetWifiCredentials(
            SetWifiCredentialsRequest {
                ssid,
                password,
                hidden,
            },
        ))
        .await?;
    Ok(())
}

#[tauri::command]
pub async fn ble_get_wifi_status(state: State<'_, BleManager>) -> Result<WifiStatus, String> {
    let resp = state
        .request(client_request::Payload::GetWifiStatus(
            GetWifiStatusRequest {},
        ))
        .await?;
    let s = expect_payload(resp, "wifiStatus", |p| match p {
        agent_response::Payload::WifiStatus(v) => Some(v),
        _ => None,
    })?;
    Ok(WifiStatus {
        connected: s.connected,
        ssid: s.ssid,
        ip_address: s.ip_address,
        signal_dbm: s.signal_dbm,
    })
}

#[tauri::command]
pub async fn ble_get_robot_config(state: State<'_, BleManager>) -> Result<RobotConfig, String> {
    let resp = state
        .request(client_request::Payload::GetRobotConfig(
            GetRobotConfigRequest {},
        ))
        .await?;
    let c = expect_payload(resp, "robotConfig", |p| match p {
        agent_response::Payload::RobotConfig(v) => Some(v),
        _ => None,
    })?;
    Ok(RobotConfig {
        robot_name: c.robot_name,
        owner_id: c.owner_id,
        timezone: c.timezone,
        extra: c.extra,
    })
}

#[tauri::command]
pub async fn ble_set_robot_config(
    state: State<'_, BleManager>,
    config: RobotConfig,
) -> Result<(), String> {
    state
        .request(client_request::Payload::SetRobotConfig(
            SetRobotConfigRequest {
                config: Some(ProtoRobotConfig {
                    robot_name: config.robot_name,
                    owner_id: config.owner_id,
                    timezone: config.timezone,
                    extra: config.extra,
                }),
            },
        ))
        .await?;
    Ok(())
}

#[tauri::command]
pub async fn ble_get_app_status(state: State<'_, BleManager>) -> Result<AppStatus, String> {
    let resp = state
        .request(client_request::Payload::GetAppStatus(
            GetAppStatusRequest {},
        ))
        .await?;
    let s = expect_payload(resp, "appStatus", |p| match p {
        agent_response::Payload::AppStatus(v) => Some(v),
        _ => None,
    })?;
    Ok(AppStatus {
        app_name: s.app_name,
        image: s.image,
        image_digest: s.image_digest,
        state: app_state_label(s.state),
        container_id: s.container_id,
        started_at_unix: s.started_at_unix,
        restart_count: s.restart_count,
    })
}

#[tauri::command]
pub async fn ble_control_app(
    state: State<'_, BleManager>,
    app_name: String,
    command: String,
) -> Result<(), String> {
    state
        .request(client_request::Payload::ControlApp(ControlAppRequest {
            app_name,
            command: app_command_from_label(&command) as i32,
        }))
        .await?;
    Ok(())
}

#[tauri::command]
pub async fn ble_trigger_ota(
    state: State<'_, BleManager>,
    target_image: String,
) -> Result<(), String> {
    state
        .request(client_request::Payload::TriggerOta(TriggerOtaRequest {
            target_image,
        }))
        .await?;
    Ok(())
}

#[tauri::command]
pub async fn ble_get_ota_status(state: State<'_, BleManager>) -> Result<OtaStatus, String> {
    let resp = state
        .request(client_request::Payload::GetOtaStatus(
            GetOtaStatusRequest {},
        ))
        .await?;
    let s = expect_payload(resp, "otaStatus", |p| match p {
        agent_response::Payload::OtaStatus(v) => Some(v),
        _ => None,
    })?;
    Ok(OtaStatus {
        state: ota_state_label(s.state),
        current_image: s.current_image,
        target_image: s.target_image,
        progress_percent: s.progress_percent,
        error: s.error,
    })
}
