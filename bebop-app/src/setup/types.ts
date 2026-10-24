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

/// Network mode. A two-way switch with no automatic fallback, changed only
/// by the physical button long-press.
///
/// - `client` — join a saved network ("Known Network").
/// - `ap`     — host the setup hotspot ("Hosted Network").
export type NetworkMode = "client" | "ap";

export type ApBand = "2.4" | "5";

export interface NetworkConfig {
  /// Read-only: owned by the physical button.
  mode: NetworkMode;
  /// SSID of the robot's Hosted Network hotspot.
  apSsid: string;
  /// Write-only: always empty in responses. Leave empty when saving to keep
  /// the existing passphrase.
  apPassword: string;
  /// Hotspot band.
  apBand: ApBand;
  /// `host:port` to reach the setup server while hosting (read-only).
  apAddress: string;
}
