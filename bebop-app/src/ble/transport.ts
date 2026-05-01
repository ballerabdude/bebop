import type {
  AppStatus,
  DeviceInfo,
  DiscoveredRobot,
  OtaStatus,
  RobotConfig,
  WifiNetwork,
  WifiStatus,
} from "./types";

/// The transport used by the setup wizard to talk to a Bebop robot.
///
/// Implementations hide the details of BLE central scanning, connection
/// management, and protobuf framing. The UI layer should only deal with
/// these high-level RPC calls.
export interface BebopTransport {
  /// List robots the user has already permitted/paired. MUST NOT trigger
  /// any OS-level picker — safe to call on mount.
  scan(timeoutMs: number): Promise<DiscoveredRobot[]>;
  /// Prompt the user to pick a robot (Web Bluetooth picker on web,
  /// platform picker on Tauri). MUST be called from a user gesture.
  pickDevice(): Promise<DiscoveredRobot | null>;
  connect(robotId: string): Promise<void>;
  disconnect(): Promise<void>;
  isConnected(): boolean;

  getDeviceInfo(): Promise<DeviceInfo>;

  scanWifi(): Promise<WifiNetwork[]>;
  setWifiCredentials(
    ssid: string,
    password: string,
    hidden: boolean,
  ): Promise<void>;
  getWifiStatus(): Promise<WifiStatus>;

  getRobotConfig(): Promise<RobotConfig>;
  setRobotConfig(config: RobotConfig): Promise<void>;

  getAppStatus(): Promise<AppStatus>;
  controlApp(
    appName: string,
    command: "START" | "STOP" | "RESTART",
  ): Promise<void>;

  triggerOta(targetImage?: string): Promise<void>;
  getOtaStatus(): Promise<OtaStatus>;
}
