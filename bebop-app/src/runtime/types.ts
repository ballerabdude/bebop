// Plain-TS shapes consumed by the UI. Mirror the protobuf types from
// `src/proto/bebop_runtime_pb.ts` but use string mode names and friendlier
// field names so React components don't have to import the proto module.

export type RuntimeMode = "UNSPECIFIED" | "IDLE" | "DIAL_IN";

export interface MotorView {
  jointName: string;
  canInterface: string;
  motorId: number;
  model: string;
  armed: boolean;
  feedbackStale: boolean;
  faultBits: number;
  position: number;
  velocity: number;
  torque: number;
  temperature: number;
  /// Most recent commanded position target (post-clamp / post-slew).
  /// Only meaningful while `armed`; reset to the live position on each
  /// new arm.
  target: number;
  posMin: number;
  posMax: number;
  velMax: number;
  tauMax: number;
  tempMax: number;
}

export interface BusView {
  canInterface: string;
  state: string;
  healthy: boolean;
}

/// One ODrive wheel telemetry view. Positions/velocities are already in
/// robot frame (the per-wheel `direction` sign applied), so a positive
/// velocity is "forward" for both wheels.
export interface WheelView {
  name: string;
  canInterface: string;
  nodeId: number;
  armed: boolean;
  feedbackStale: boolean;
  positionReceived: boolean;
  errorCode: number;
  position: number; // cumulative rad, unbounded
  velocity: number; // rad/s
  targetVelocity: number; // rad/s, post clamp + slew
  velMax: number; // rad/s
  /// Raw ODrive AxisState from the heartbeat; 1 = IDLE, 8 = CLOSED_LOOP,
  /// 3..=12 minus 8 = calibration states. Lets the UI flag "armed but
  /// axis still IDLE" — the silent-enable-fail mode of a lost encoder
  /// calibration.
  axisState: number;
}

/// Differential-drive summary: the operator twist in effect plus the
/// wheel-encoder odometry pose. `present` is false without a `drive:` block.
export interface DriveView {
  present: boolean;
  cmdLinearX: number; // m/s
  cmdAngularZ: number; // rad/s
  /// True when the robot's operator-link watchdog zeroed a stale twist
  /// (no fresh SetVelocityCommand within the server's
  /// `operator_timeout_ms` while a non-zero twist was active, e.g. this
  /// app dropped off Wi-Fi mid-drive). Cleared by the next fresh
  /// command; show a "link lost — motion halted" pill while set.
  operatorStale: boolean;
  /// Operator arbitration: at most one connected client ("active
  /// operator") may send non-zero drive commands — the assignment
  /// follows input (first non-zero command claims it; it's lost on
  /// disconnect, staleness, E-STOP, or after the server's
  /// `operator_grace_ms` without input). Stops are always accepted
  /// from any client.
  /// `hasActiveOperator`: some client currently holds a fresh
  /// assignment. `youAreActiveOperator`: it's *this* connection (frames
  /// are per-connection, so "you" is well-defined).
  hasActiveOperator: boolean;
  youAreActiveOperator: boolean;
  /// Name the seat holder announced via `ClientHello` ("" when none,
  /// or when the holder never sent one / on older firmware).
  activeOperatorName: string;
  /// The seat holder declared itself autonomous in `ClientHello` (the
  /// voice agent). A person's twist or stop preempts it at once. Always
  /// false on firmware older than the PREEMPT change.
  activeOperatorAutonomous: boolean;
  odomX: number; // m
  odomY: number; // m
  odomTheta: number; // rad
}

/// Power-board telemetry view. Mirrors the firmware's `PowerStats`
/// proto with friendlier field names and a `present` flag the UI can
/// use to decide whether to render the power card at all.
///
/// All numeric fields are 0 when the firmware hasn't received a status
/// response yet (`statusReceived = false`); check that before drawing
/// e.g. a state-of-charge bar. `stateOfChargePct < 0` is the explicit
/// "unknown" sentinel — render as "—".
export interface PowerView {
  present: boolean;
  canInterface: string;
  powerId: number;
  firmwareVersion: string;

  statusReceived: boolean;
  statusStale: boolean;
  lastStatusAgeMs: number;

  batteryVoltageV: number;
  motorVoltageV: number;
  boardTemperatureC: number;

  faultBits: number;
  faultDescription: string;
  rail12vOn: boolean;
  softStartOn: boolean;
  motorRailOn: boolean;
  rail24vOn: boolean;

  currentAlA: number;
  currentArA: number;
  currentLlA: number;
  currentLrA: number;
  totalMotorCurrentA: number;

  batteryCells: number;
  packFullVoltageV: number;
  packEmptyVoltageV: number;
  /// Linear-interp state-of-charge in percent (0..100), or `< 0` for
  /// "unknown" (no battery reading yet, or out-of-range pack voltage).
  stateOfChargePct: number;
}

/// Orientation telemetry view. Mirrors the firmware's `ImuStats` proto.
/// Always present in the view layer; `present === false` when the
/// firmware has no `imu:` block configured — the UI should hide the
/// orientation card in that case. `received === false` until the
/// BNO085 produces its first usable quaternion.
///
/// `quaternion` is `[qx, qy, qz, qw]` in Hamilton (XYZW) order, body
/// frame. Identity orientation is `(0, 0, 0, 1)`.
export interface ImuView {
  present: boolean;
  received: boolean;
  stale: boolean;
  lastUpdateAgeMs: number;
  quaternion: [number, number, number, number];
}

/// State of the on-robot voice (speech-to-speech) service
/// (`bebop-voice.service`). `present === false` means the unit isn't
/// installed on this robot; the UI should hide the voice card. `running` is
/// true while the unit is systemd-`active` (the Python service is up).
/// Audio does not travel over the runtime WS — the voice page talks to the
/// service's own WebSocket on :9093 (see docs/voice.md).
export interface VoiceView {
  present: boolean;
  running: boolean;
  state: string;
  detail: string;
  service: string;
}

export interface RuntimeSnapshot {
  hostUnixMs: number;
  mode: RuntimeMode;
  estopLatched: boolean;
  estopReason: string;
  motors: MotorView[];
  buses: BusView[];
  /// ODrive wheels (empty without a `wheels:` block).
  wheels: WheelView[];
  /// Differential-drive summary; `drive.present === false` without a
  /// `drive:` block.
  drive: DriveView;
  /// Always present in the view layer; `power.present === false` when
  /// the firmware has no `power:` block configured.
  power: PowerView;
  /// Always present in the view layer; `imu.present === false` when
  /// the firmware has no `imu:` block configured.
  imu: ImuView;
  /// bebop-vision service state. `vision.present === false` when the
  /// `bebop-vision.service` unit isn't installed — the UI hides the
  /// vision control card in that case.
  vision: VisionView;
  /// Model-weight provisioning state. `model.present === false` when the
  /// download template unit isn't installed — the UI hides the model card.
  model: ModelView;
  /// Voice (speech-to-speech) service state. `voice.present === false` when
  /// `bebop-voice.service` isn't installed — the UI hides the voice card.
  voice: VoiceView;
}

/// State of the Python bebop-vision service (`bebop-vision.service`).
/// `present === false` means the unit isn't installed on this robot; the
/// UI should hide the vision control card. `running` is true while the
/// unit is systemd-`active` (the recorder process is up). `state` is the
/// raw systemd ActiveState for transition rendering; `detail` carries the
/// SubState or the last systemctl error.
export interface VisionView {
  present: boolean;
  running: boolean;
  state: string;
  detail: string;
  service: string;
}

/// One catalog model and its live provisioning status. `kind === "hf"`
/// entries are downloadable (from Hugging Face); `kind === "local"` entries
/// were trained on the robot or copied from a workstation, so `ready` just
/// reports presence.
export interface ModelEntryView {
  id: string;
  name: string;
  description: string;
  kind: string;
  /// What the model is for (e.g. "segmentation", "voice").
  purpose: string;
  repo: string;
  gated: boolean;
  bytesTotal: number;
  ready: boolean;
  /// "idle" | "downloading" | "ready" | "failed" | "unauthorized"
  state: string;
  detail: string;
  bytesDownloaded: number;
}

/// Model-weight provisioning state (Hugging Face token + per-model status).
/// `present === false` means the download unit isn't installed; the UI hides
/// the provisioning card in that case.
export interface ModelView {
  tokenSet: boolean;
  present: boolean;
  diskFreeBytes: number;
  detail: string;
  models: ModelEntryView[];
}
