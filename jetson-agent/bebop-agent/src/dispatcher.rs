//! Translates decoded `ClientRequest` protobufs into agent actions and
//! produces `AgentResponse` protobufs to send back.

use bebop_proto::v1::{
    agent_response, client_request, AgentResponse, ClientRequest, DeviceInfo, NetworkConfig,
    ResponseStatus, RobotConfig, WifiScanResult, WifiStatus,
};

use crate::config::AgentConfig;
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

/// Wi-Fi join.
///
/// In Hosted Network mode (`ap`) the credentials are **saved but not
/// applied** — the hotspot stays up. The robot joins the network only after
/// the button switches it to Known Network (`client`), which is also when
/// the saved profile autoconnects. In `client` mode we join immediately.
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
    let hosts_ap = state.config().await.network.hosts_ap();

    if hosts_ap {
        match wifi::save_credentials(state, &ssid, &password, hidden).await {
            Ok(status) => ok_response_with_message(
                request_id,
                &format!("Saved {ssid}. Long-press the button to switch to Known Network."),
                agent_response::Payload::WifiStatus(status.into()),
            ),
            Err(e) => err_response(request_id, ResponseStatus::Error, &e.to_string()),
        }
    } else {
        match wifi::connect(state, &ssid, &password, hidden).await {
            Ok(status) => ok_response(
                request_id,
                agent_response::Payload::WifiStatus(status.into()),
            ),
            Err(e) => err_response(request_id, ResponseStatus::Error, &e.to_string()),
        }
    }
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
    ok_response(
        request_id,
        agent_response::Payload::NetworkConfig(NetworkConfig {
            mode: cfg.network.mode,
            ap_ssid: cfg.network.ap_ssid,
            // Never leak the passphrase; empty means "unchanged" on write.
            ap_password: String::new(),
            ap_band: cfg.network.ap_band,
            ap_address: ap_status.address,
        }),
    )
}

/// Update Hosted Network settings. `mode` is **ignored** — it is owned by
/// the physical button. An empty `ap_password` keeps the existing one.
async fn set_network_config(
    state: &AppState,
    request_id: u32,
    cfg: Option<NetworkConfig>,
) -> AgentResponse {
    let Some(cfg) = cfg else {
        return err_response(request_id, ResponseStatus::Error, "missing config");
    };
    let band = cfg.ap_band.trim().to_owned();
    if !band.is_empty() && band != "2.4" && band != "5" {
        return err_response(
            request_id,
            ResponseStatus::Error,
            "ap_band must be \"2.4\" or \"5\"",
        );
    }

    let mut ap_changed = false;
    if let Err(e) = mutate_and_persist(state, |c| {
        if !cfg.ap_ssid.trim().is_empty() && cfg.ap_ssid != c.network.ap_ssid {
            c.network.ap_ssid = cfg.ap_ssid.clone();
            ap_changed = true;
        }
        if !band.is_empty() && band != c.network.ap_band {
            c.network.ap_band = band.clone();
            ap_changed = true;
        }
        if !cfg.ap_password.is_empty() {
            c.network.ap_password = cfg.ap_password.clone();
            ap_changed = true;
        }
    })
    .await
    {
        return err_response(request_id, ResponseStatus::Error, &e.to_string());
    }

    // Re-raise the hotspot so new settings take effect promptly. This drops
    // any connected client briefly (single radio).
    if ap_changed && state.config().await.network.hosts_ap() {
        ap::reconfigure(state).await;
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

fn ok_response_with_message(
    request_id: u32,
    msg: &str,
    payload: agent_response::Payload,
) -> AgentResponse {
    AgentResponse {
        request_id,
        status: ResponseStatus::Ok as i32,
        message: msg.into(),
        payload: Some(payload),
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
    state.update_config(f).await;
    state
        .persist_config()
        .await
        .map_err(|e| AgentError::Config(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use bebop_proto::v1::{client_request, GetDeviceInfoRequest};

    async fn test_state() -> AppState {
        let cfg: AgentConfig = toml::from_str("robot_name = \"test\"").unwrap();
        AppState::new(cfg).await.unwrap()
    }

    #[tokio::test]
    async fn device_info_round_trips() {
        let state = test_state().await;
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
    async fn invalid_band_rejected() {
        let state = test_state().await;
        let req = ClientRequest {
            request_id: 1,
            payload: Some(client_request::Payload::SetNetworkConfig(
                bebop_proto::v1::SetNetworkConfigRequest {
                    config: Some(NetworkConfig {
                        ap_band: "6".into(),
                        ..Default::default()
                    }),
                },
            )),
        };
        let resp = handle(&state, req).await;
        assert_eq!(resp.status, ResponseStatus::Error as i32);
    }
}
