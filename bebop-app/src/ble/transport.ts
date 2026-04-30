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
  scan(timeoutMs: number): Promise<DiscoveredRobot[]>;
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
