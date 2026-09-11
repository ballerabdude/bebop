//! Translates decoded `ClientRequest` protobufs into agent actions and
//! produces `AgentResponse` protobufs to send back.

use bebop_proto::v1::{
    agent_response, client_request, AgentResponse, ClientRequest, DeviceInfo, NetworkConfig,
    ResponseStatus, RobotConfig, WifiScanResult, WifiStatus,
};

use crate::config::{self, AgentConfig};
use crate::error::AgentError;
use crate::state::AppState;
use crate::{ap, wifi, AGENT_VERSION};

/// Top-level dispatch.
pub async fn handle(state: &AppState, req: ClientRequest) -> AgentResponse {
    let request_id = req.request_id;
    let Some(payload) = req.payload else {
        return err_response(request_id, ResponseStatus::Error, "missing payload");
    };

    match payload {
        client_request::Payload::GetDeviceInfo(_) => device_info(state, request_id).await,
        client_request::Payload::ScanWifi(_) => scan_wifi(state, request_id).await,
        client_request::Payload::SetWifiCredentials(req) => {
            set_wifi(state, request_id, req.ssid, req.password, req.hidden).await
        }
        client_request::Payload::GetWifiStatus(_) => wifi_status(state, request_id).await,
        client_request::Payload::SetRobotConfig(req) => {
            set_robot_config(state, request_id, req.config).await
        }
        client_request::Payload::GetRobotConfig(_) => get_robot_config(state, request_id).await,
        client_request::Payload::GetNetworkConfig(_) => get_network_config(state, request_id).await,
        client_request::Payload::SetNetworkConfig(req) => {
            set_network_config(state, request_id, req.config).await
        }
    }
}

async fn device_info(state: &AppState, request_id: u32) -> AgentResponse {
    let cfg = state.config().await;
    let info = DeviceInfo {
        serial_number: serial_number(),
        model: "bebop-v1".into(),
        agent_version: AGENT_VERSION.into(),
        jetpack_version: jetpack_version().unwrap_or_default(),
        hostname: cfg.robot_name,
    };
    ok_response(request_id, agent_response::Payload::DeviceInfo(info))
}

async fn scan_wifi(state: &AppState, request_id: u32) -> AgentResponse {
    match wifi::scan(state).await {
        Ok(networks) => {
            let result = WifiScanResult {
                networks: networks.into_iter().map(Into::into).collect(),
            };
            ok_response(request_id, agent_response::Payload::WifiScanResult(result))
        }
        Err(e) => err_response(request_id, ResponseStatus::Error, &e.to_string()),
    }
}

/// Persist the requested credentials, reply immediately, then (in the
/// background) drop the SoftAP and join the target network.
///
/// The reply must be sent *before* the radio switches — the app is reachable
/// over the SoftAP, which has to come down for the client join to succeed.
/// The app polls `getWifiStatus` after the user reconnects to their own
/// network.
async fn set_wifi(
    state: &AppState,
    request_id: u32,
    ssid: String,
    password: String,
    hidden: bool,
) -> AgentResponse {
    if ssid.trim().is_empty() {
        return err_response(request_id, ResponseStatus::Error, "missing ssid");
    }

    // Suppress the AP supervisor while the join is in flight so it doesn't
    // race the connection by re-raising the hotspot.
    state
        .update_ap_status(|s| {
            s.connecting = true;
            s.last_error = None;
        })
        .await;

    let task_state = state.clone();
    let ssid_task = ssid.clone();
    tokio::spawn(async move {
        let _ = ap::lower_now(&task_state).await;
        match wifi::connect(&task_state, &ssid_task, &password, hidden).await {
            Ok(status) => {
                tracing::info!(ssid = %ssid_task, connected = status.connected, "wifi join finished");
            }
            Err(e) => {
                tracing::warn!(error = %e, ssid = %ssid_task, "wifi join failed");
                task_state
                    .update_ap_status(|s| s.last_error = Some(e.to_string()))
                    .await;
            }
        }
        task_state.update_ap_status(|s| s.connecting = false).await;
    });

    ok_response_with_message(
        request_id,
        &format!("connecting to {ssid}; robot will drop the setup network"),
    )
}

async fn wifi_status(state: &AppState, request_id: u32) -> AgentResponse {
    let s = state.wifi_status().await;
    ok_response(
        request_id,
        agent_response::Payload::WifiStatus(WifiStatus {
            connected: s.connected,
            ssid: s.ssid,
            ip_address: s.ip_address,
            signal_dbm: s.signal_dbm,
        }),
    )
}

async fn set_robot_config(
    state: &AppState,
    request_id: u32,
    cfg: Option<RobotConfig>,
) -> AgentResponse {
    let Some(cfg) = cfg else {
        return err_response(request_id, ResponseStatus::Error, "missing config");
    };
    let robot_name = cfg.robot_name.clone();
    if let Err(e) = mutate_and_persist(state, |c| {
        if !robot_name.is_empty() {
            c.robot_name = robot_name;
        }
    })
    .await
    {
        return err_response(request_id, ResponseStatus::Error, &e.to_string());
    }
    ok_response(request_id, agent_response::Payload::RobotConfig(cfg))
}

async fn get_robot_config(state: &AppState, request_id: u32) -> AgentResponse {
    let cfg = state.config().await;
    let resp = RobotConfig {
        robot_name: cfg.robot_name,
        owner_id: String::new(),
        timezone: String::new(),
        extra: Default::default(),
    };
    ok_response(request_id, agent_response::Payload::RobotConfig(resp))
}

async fn get_network_config(state: &AppState, request_id: u32) -> AgentResponse {
    let cfg = state.config().await;
    let ap_status = state.ap_status().await;
    let ap_ssid = cfg.network.ap_ssid();
    ok_response(
        request_id,
        agent_response::Payload::NetworkConfig(NetworkConfig {
            mode: cfg.network.mode,
            ap_ssid,
            ap_address: ap_status.address,
        }),
    )
}

async fn set_network_config(
    state: &AppState,
    request_id: u32,
    cfg: Option<NetworkConfig>,
) -> AgentResponse {
    let Some(cfg) = cfg else {
        return err_response(request_id, ResponseStatus::Error, "missing config");
    };
    let mode = cfg.mode.to_ascii_lowercase();
    if !matches!(mode.as_str(), "auto" | "client" | "ap") {
        return err_response(
            request_id,
            ResponseStatus::Error,
            "mode must be one of auto|client|ap",
        );
    }
    if let Err(e) = mutate_and_persist(state, |c| {
        c.network.mode = mode.clone();
    })
    .await
    {
        return err_response(request_id, ResponseStatus::Error, &e.to_string());
    }
    get_network_config(state, request_id).await
}

// ---------------------------------------------------------------------------
// helpers

fn ok_response(request_id: u32, payload: agent_response::Payload) -> AgentResponse {
    AgentResponse {
        request_id,
        status: ResponseStatus::Ok as i32,
        message: String::new(),
        payload: Some(payload),
    }
}

fn ok_response_with_message(request_id: u32, msg: &str) -> AgentResponse {
    AgentResponse {
        request_id,
        status: ResponseStatus::Ok as i32,
        message: msg.into(),
        payload: None,
    }
}

fn err_response(request_id: u32, status: ResponseStatus, msg: &str) -> AgentResponse {
    AgentResponse {
        request_id,
        status: status as i32,
        message: msg.into(),
        payload: None,
    }
}

fn serial_number() -> String {
    std::fs::read_to_string("/proc/device-tree/serial-number")
        .ok()
        .map(|s| s.trim_end_matches('\0').trim().to_owned())
        .unwrap_or_else(|| "unknown".into())
}

fn jetpack_version() -> Option<String> {
    std::fs::read_to_string("/etc/nv_tegra_release")
        .ok()
        .map(|s| s.lines().next().unwrap_or_default().to_owned())
}

/// Clone the current config, apply `f`, persist to disk, then swap the
/// in-memory copy on success. Keeps the live config and the on-disk file
/// in lockstep so a crash mid-write can't leave them disagreeing.
async fn mutate_and_persist<F>(state: &AppState, f: F) -> Result<(), AgentError>
where
    F: FnOnce(&mut AgentConfig),
{
    let mut next = state.config().await;
    f(&mut next);
    let path = config::config_path();
    config::save(&next, &path).map_err(|e| AgentError::Config(e.to_string()))?;
    state.update_config(|c| *c = next).await;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use bebop_proto::v1::{client_request, GetDeviceInfoRequest};

    #[tokio::test]
    async fn device_info_round_trips() {
        let state = AppState::new(
            AgentConfig::load()
                .unwrap_or_else(|_| toml::from_str("robot_name = \"test\"").unwrap()),
        )
        .await
        .unwrap();
        let req = ClientRequest {
            request_id: 7,
            payload: Some(client_request::Payload::GetDeviceInfo(
                GetDeviceInfoRequest {},
            )),
        };
        let resp = handle(&state, req).await;
        assert_eq!(resp.request_id, 7);
        assert_eq!(resp.status, ResponseStatus::Ok as i32);
    }

    #[tokio::test]
    async fn invalid_network_mode_rejected() {
        let state = AppState::new(toml::from_str("robot_name = \"test\"").unwrap())
            .await
            .unwrap();
        let req = ClientRequest {
            request_id: 1,
            payload: Some(client_request::Payload::SetNetworkConfig(
                bebop_proto::v1::SetNetworkConfigRequest {
                    config: Some(NetworkConfig {
                        mode: "bogus".into(),
                        ..Default::default()
                    }),
                },
            )),
        };
        let resp = handle(&state, req).await;
        assert_eq!(resp.status, ResponseStatus::Error as i32);
    }
}
