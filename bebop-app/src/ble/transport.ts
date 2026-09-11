import type {
  DeviceInfo,
  NetworkConfig,
  RobotConfig,
  WifiNetwork,
  WifiStatus,
} from "./types";

/// Transport used by the setup wizard to talk to a robot's provisioning
/// server. The only implementation is `SetupTransport`, which speaks
/// protobuf over a binary WebSocket to `bebop-agent` on port 9091 — over
/// the LAN once the robot is on Wi-Fi, or directly over its setup hotspot
/// during first-time provisioning.
export interface BebopTransport {
  /// Open the underlying WebSocket. Resolves once the handshake completes.
  connect(): Promise<void>;
  disconnect(): void;
  isConnected(): boolean;

  getDeviceInfo(): Promise<DeviceInfo>;

  scanWifi(): Promise<WifiNetwork[]>;
  setWifiCredentials(
    ssid: string,
    password: string,
    hidden: boolean,
  ): Promise<WifiStatus>;
  getWifiStatus(): Promise<WifiStatus>;

  getRobotConfig(): Promise<RobotConfig>;
  setRobotConfig(config: RobotConfig): Promise<void>;

  getNetworkConfig(): Promise<NetworkConfig>;
  setNetworkConfig(config: NetworkConfig): Promise<NetworkConfig>;
}
