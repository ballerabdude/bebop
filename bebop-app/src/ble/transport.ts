import type {
  AppStatus,
  ControllerStatus,
  DeviceInfo,
  DiscoveredController,
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
  /// Persist a new container image to the agent's on-disk config. The
  /// running container is NOT restarted automatically; call `controlApp`
  /// with `"RESTART"` to apply. Empty string clears the configured image.
  setAppImage(image: string): Promise<AppStatus>;

  triggerOta(targetImage?: string): Promise<void>;
  getOtaStatus(): Promise<OtaStatus>;

  /// Discover nearby Bluetooth devices (gamepads + others). The agent
  /// runs `bluetoothctl scan` for `timeoutMs` and returns everything it
  /// saw, with `kind == "gamepad"` for likely controllers. Default
  /// timeout (when 0 is passed) is 8000 ms on the agent side.
  scanControllers(timeoutMs: number): Promise<DiscoveredController[]>;
  /// Pair, trust, connect, and persist `mac` as the active controller.
  /// On success the agent immediately starts forwarding teleop input.
  pairController(mac: string): Promise<ControllerStatus>;
  /// Forget `mac` (or the currently paired controller if `mac` is
  /// empty). Resolves with the post-unpair status.
  unpairController(mac: string): Promise<ControllerStatus>;
  getControllerStatus(): Promise<ControllerStatus>;
}
