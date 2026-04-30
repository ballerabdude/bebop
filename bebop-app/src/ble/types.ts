// Plain TypeScript mirrors of the protobuf messages defined in
// `bebop-proto/proto/bebop.proto`. Until we wire up a generated protobuf
// client, we treat the agent responses as already-decoded JSON — the real
// transport will handle encode/decode internally.

export type AppState =
  | "UNSPECIFIED"
  | "STOPPED"
  | "STARTING"
  | "RUNNING"
  | "CRASHED"
  | "UPDATING";

export type OtaState =
  | "UNSPECIFIED"
  | "IDLE"
  | "CHECKING"
  | "DOWNLOADING"
  | "APPLYING"
  | "SUCCESS"
  | "FAILED";

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

export interface AppStatus {
  appName: string;
  image: string;
  imageDigest: string;
  state: AppState;
  containerId: string;
  startedAtUnix: number;
  restartCount: number;
}

export interface OtaStatus {
  state: OtaState;
  currentImage: string;
  targetImage: string;
  progressPercent: number;
  error: string;
}

export interface DiscoveredRobot {
  id: string; // platform-specific peripheral id
  name: string;
  rssi: number;
}
