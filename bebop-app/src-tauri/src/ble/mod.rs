//! Tauri commands fronting a native BLE central.
//!
//! `central::BleManager` owns the actual `btleplug` connection and is held
//! by Tauri as managed state. Each command here is a thin wrapper that:
//!   * builds the JSON request envelope expected by the agent (mirrors the
//!     shape used by the TypeScript `WebBluetoothTransport`),
//!   * sends it via `BleManager::request`,
//!   * extracts the typed payload from the response.
//!
//! All DTOs use camelCase so the frontend can consume them directly.

mod central;
mod framing;

pub use central::BleManager;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
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

fn obj(value: Value) -> serde_json::Map<String, Value> {
    match value {
        Value::Object(m) => m,
        _ => serde_json::Map::new(),
    }
}

fn extract<T: for<'de> Deserialize<'de>>(resp: &Value, field: &str) -> Result<T, String> {
    let v = resp
        .get(field)
        .cloned()
        .ok_or_else(|| format!("response missing `{field}`: {resp}"))?;
    serde_json::from_value(v).map_err(|e| format!("decoding `{field}`: {e}"))
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
pub async fn ble_connect(
    state: State<'_, BleManager>,
    robot_id: String,
) -> Result<(), String> {
    state.connect(robot_id).await
}

#[tauri::command]
pub async fn ble_disconnect(state: State<'_, BleManager>) -> Result<(), String> {
    state.disconnect().await
}

#[tauri::command]
pub async fn ble_get_device_info(
    state: State<'_, BleManager>,
) -> Result<DeviceInfo, String> {
    let resp = state.request(obj(json!({ "getDeviceInfo": {} }))).await?;
    extract(&resp, "deviceInfo")
}

#[tauri::command]
pub async fn ble_scan_wifi(
    state: State<'_, BleManager>,
) -> Result<Vec<WifiNetwork>, String> {
    let resp = state.request(obj(json!({ "scanWifi": {} }))).await?;
    let networks = resp
        .get("wifiScanResult")
        .and_then(|v| v.get("networks"))
        .cloned()
        .unwrap_or_else(|| Value::Array(Vec::new()));
    serde_json::from_value(networks).map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn ble_set_wifi_credentials(
    state: State<'_, BleManager>,
    ssid: String,
    password: String,
    hidden: bool,
) -> Result<(), String> {
    state
        .request(obj(json!({
            "setWifiCredentials": { "ssid": ssid, "password": password, "hidden": hidden },
        })))
        .await?;
    Ok(())
}

#[tauri::command]
pub async fn ble_get_wifi_status(
    state: State<'_, BleManager>,
) -> Result<WifiStatus, String> {
    let resp = state.request(obj(json!({ "getWifiStatus": {} }))).await?;
    extract(&resp, "wifiStatus")
}

#[tauri::command]
pub async fn ble_get_robot_config(
    state: State<'_, BleManager>,
) -> Result<RobotConfig, String> {
    let resp = state.request(obj(json!({ "getRobotConfig": {} }))).await?;
    extract(&resp, "robotConfig")
}

#[tauri::command]
pub async fn ble_set_robot_config(
    state: State<'_, BleManager>,
    config: RobotConfig,
) -> Result<(), String> {
    state
        .request(obj(json!({ "setRobotConfig": { "config": config } })))
        .await?;
    Ok(())
}

#[tauri::command]
pub async fn ble_get_app_status(
    state: State<'_, BleManager>,
) -> Result<AppStatus, String> {
    let resp = state.request(obj(json!({ "getAppStatus": {} }))).await?;
    extract(&resp, "appStatus")
}

#[tauri::command]
pub async fn ble_control_app(
    state: State<'_, BleManager>,
    app_name: String,
    command: String,
) -> Result<(), String> {
    state
        .request(obj(json!({
            "controlApp": { "appName": app_name, "command": command },
        })))
        .await?;
    Ok(())
}

#[tauri::command]
pub async fn ble_trigger_ota(
    state: State<'_, BleManager>,
    target_image: String,
) -> Result<(), String> {
    state
        .request(obj(json!({ "triggerOta": { "targetImage": target_image } })))
        .await?;
    Ok(())
}

#[tauri::command]
pub async fn ble_get_ota_status(
    state: State<'_, BleManager>,
) -> Result<OtaStatus, String> {
    let resp = state.request(obj(json!({ "getOtaStatus": {} }))).await?;
    extract(&resp, "otaStatus")
}
