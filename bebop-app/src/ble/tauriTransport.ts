import { invoke } from "@tauri-apps/api/core";

import type { BebopTransport } from "./transport";
import type {
  AppStatus,
  DeviceInfo,
  DiscoveredRobot,
  OtaStatus,
  RobotConfig,
  WifiNetwork,
  WifiStatus,
} from "./types";

/// Transport that delegates all BLE + protobuf work to the Rust side of
/// the Tauri app. The Rust commands are currently stubs (see
/// `src-tauri/src/ble.rs`) — the real implementation will plug into a
/// native BLE central stack on each platform.
export class TauriTransport implements BebopTransport {
  private connected = false;

  async scan(timeoutMs: number): Promise<DiscoveredRobot[]> {
    return await invoke<DiscoveredRobot[]>("ble_scan", { timeoutMs });
  }

  async connect(robotId: string): Promise<void> {
    await invoke("ble_connect", { robotId });
    this.connected = true;
  }

  async disconnect(): Promise<void> {
    await invoke("ble_disconnect");
    this.connected = false;
  }

  isConnected(): boolean {
    return this.connected;
  }

  async getDeviceInfo(): Promise<DeviceInfo> {
    return await invoke<DeviceInfo>("ble_get_device_info");
  }

  async scanWifi(): Promise<WifiNetwork[]> {
    return await invoke<WifiNetwork[]>("ble_scan_wifi");
  }

  async setWifiCredentials(
    ssid: string,
    password: string,
    hidden: boolean,
  ): Promise<void> {
    await invoke("ble_set_wifi_credentials", { ssid, password, hidden });
  }

  async getWifiStatus(): Promise<WifiStatus> {
    return await invoke<WifiStatus>("ble_get_wifi_status");
  }

  async getRobotConfig(): Promise<RobotConfig> {
    return await invoke<RobotConfig>("ble_get_robot_config");
  }

  async setRobotConfig(config: RobotConfig): Promise<void> {
    await invoke("ble_set_robot_config", { config });
  }

  async getAppStatus(): Promise<AppStatus> {
    return await invoke<AppStatus>("ble_get_app_status");
  }

  async controlApp(
    appName: string,
    command: "START" | "STOP" | "RESTART",
  ): Promise<void> {
    await invoke("ble_control_app", { appName, command });
  }

  async triggerOta(targetImage?: string): Promise<void> {
    await invoke("ble_trigger_ota", { targetImage: targetImage ?? "" });
  }

  async getOtaStatus(): Promise<OtaStatus> {
    return await invoke<OtaStatus>("ble_get_ota_status");
  }
}
