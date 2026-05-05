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

/// One Bluetooth device returned by `scanControllers`. `kind` is
/// `"gamepad"` when the agent's Class-of-Device / UUID / name
/// heuristics match, `"unknown"` otherwise — the UI hides
/// non-gamepads behind a "Show all" toggle.
export interface DiscoveredController {
  mac: string;
  name: string;
  rssi: number;
  paired: boolean;
  connected: boolean;
  kind: "gamepad" | "unknown";
}

/// Live status of the agent's Bluetooth-controller subsystem. Mirrors
/// `bebop.v1.ControllerStatus`. `armed` is true iff the deadman is
/// held AND no e-stop is latched — i.e. velocity commands are
/// flowing to bebop-linux.
export interface ControllerStatus {
  enabled: boolean;
  pairedMac: string;
  deviceName: string;
  connected: boolean;
  armed: boolean;
  estopLatched: boolean;
  lastEventUnixMs: number;
  targetAddr: string;
}
