//! Translates decoded `ClientRequest` protobufs into agent actions and
//! produces `AgentResponse` protobufs to send back.

use bebop_proto::v1::{
    agent_response, client_request, AgentResponse, AppCommand, AppStatus, ClientRequest,
    DeviceInfo, OtaStatus, ResponseStatus, RobotConfig, WifiScanResult, WifiStatus,
};

use crate::config::{self, AgentConfig};
use crate::error::AgentError;
use crate::{containers, ota, state::AppState, wifi, AGENT_VERSION};

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
        client_request::Payload::GetAppStatus(_) => app_status(state, request_id).await,
        client_request::Payload::ControlApp(req) => {
            let cmd = req.command();
            control_app(state, request_id, req.app_name, cmd).await
        }
        client_request::Payload::TriggerOta(req) => {
            trigger_ota(state, request_id, req.target_image).await
        }
        client_request::Payload::GetOtaStatus(_) => ota_status(state, request_id).await,
        client_request::Payload::SetAppImage(req) => {
            set_app_image(state, request_id, req.image).await
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

async fn set_wifi(
    state: &AppState,
    request_id: u32,
    ssid: String,
    password: String,
    hidden: bool,
) -> AgentResponse {
    match wifi::connect(state, &ssid, &password, hidden).await {
        Ok(status) => ok_response(
            request_id,
            agent_response::Payload::WifiStatus(status.into()),
        ),
        Err(e) => err_response(request_id, ResponseStatus::Error, &e.to_string()),
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

async fn app_status(state: &AppState, request_id: u32) -> AgentResponse {
    let s = state.app_status().await;
    let cfg = state.config().await;
    let msg = AppStatus {
        app_name: cfg.app.name,
        image: s.image,
        image_digest: s.image_digest,
        state: app_state_to_proto(s.state) as i32,
        container_id: s.container_id,
        started_at_unix: s.started_at_unix,
        restart_count: s.restart_count,
    };
    ok_response(request_id, agent_response::Payload::AppStatus(msg))
}

async fn set_app_image(state: &AppState, request_id: u32, image: String) -> AgentResponse {
    let next = if image.is_empty() { None } else { Some(image) };
    if let Err(e) = mutate_and_persist(state, |c| {
        c.app.image = next;
    })
    .await
    {
        return err_response(request_id, ResponseStatus::Error, &e.to_string());
    }
    // Return the live AppStatus so the caller can render the change
    // immediately. The configured image now reflects the new value, but
    // the running container is untouched until ControlApp{RESTART} is
    // invoked.
    app_status(state, request_id).await
}

async fn control_app(
    state: &AppState,
    request_id: u32,
    _app_name: String,
    cmd: AppCommand,
) -> AgentResponse {
    let res = match cmd {
        AppCommand::Start => containers::start(state).await,
        AppCommand::Stop => containers::stop(state).await,
        AppCommand::Restart => containers::restart(state).await,
        AppCommand::Unspecified => Err(crate::error::AgentError::InvalidRequest(
            "unspecified command".into(),
        )),
    };
    match res {
        Ok(()) => ok_response_with_message(request_id, "ok"),
        Err(e) => err_response(request_id, ResponseStatus::Error, &e.to_string()),
    }
}

async fn trigger_ota(state: &AppState, request_id: u32, target_image: String) -> AgentResponse {
    match ota::trigger(
        state,
        if target_image.is_empty() {
            None
        } else {
            Some(target_image)
        },
    )
    .await
    {
        Ok(()) => ok_response_with_message(request_id, "ota started"),
        Err(e) => err_response(request_id, ResponseStatus::Error, &e.to_string()),
    }
}

async fn ota_status(state: &AppState, request_id: u32) -> AgentResponse {
    let s = state.ota_status().await;
    let msg = OtaStatus {
        state: ota_state_to_proto(s.state) as i32,
        current_image: s.current_image,
        target_image: s.target_image,
        progress_percent: s.progress_percent,
        error: s.error.unwrap_or_default(),
    };
    ok_response(request_id, agent_response::Payload::OtaStatus(msg))
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

fn app_state_to_proto(s: crate::state::AppLifecycle) -> bebop_proto::v1::AppState {
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

fn ota_state_to_proto(s: crate::state::OtaLifecycle) -> bebop_proto::v1::OtaState {
    use crate::state::OtaLifecycle as L;
    use bebop_proto::v1::OtaState as P;
    match s {
        L::Idle => P::Idle,
        L::Checking => P::Checking,
        L::Downloading => P::Downloading,
        L::Applying => P::Applying,
        L::Success => P::Success,
        L::Failed => P::Failed,
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
