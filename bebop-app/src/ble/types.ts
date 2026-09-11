// Plain TypeScript mirrors of the provisioned setup messages defined in
// `bebop-proto/proto/bebop.proto`.

export interface DeviceInfo {
  serialNumber: string;
  model: string;
  agentVersion: string;
  jetpackVersion: string;
  hostname: string;
}

export interface WifiNetwork {
  ssid: string;
  signalDbm: number;
  security: string;
  saved: boolean;
}

export interface WifiStatus {
  connected: boolean;
  ssid: string;
  ipAddress: string;
  signalDbm: number;
}

export interface RobotConfig {
  robotName: string;
  ownerId: string;
  timezone: string;
  extra: Record<string, string>;
}

/// Wi-Fi provisioning mode for the robot.
///
/// - `auto`   — join a known network; fall back to the setup hotspot.
/// - `client` — only ever act as a Wi-Fi client.
/// - `ap`     — always host the setup hotspot.
export type NetworkMode = "auto" | "client" | "ap";

export interface NetworkConfig {
  mode: NetworkMode;
  /// SSID of the robot's setup hotspot (read-only).
  apSsid: string;
  /// `host:port` to reach the setup server while the hotspot is up.
  apAddress: string;
}
