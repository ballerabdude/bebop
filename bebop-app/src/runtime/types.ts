// Plain-TS shapes consumed by the UI. Mirror the protobuf types from
// `src/proto/bebop_runtime_pb.ts` but use string mode names and friendlier
// field names so React components don't have to import the proto module.

export type RuntimeMode = "UNSPECIFIED" | "IDLE" | "DIAL_IN" | "RUN_POLICY";

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
  /// new arm. Drives the dial-in slider's "what we asked for" marker.
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
/// BNO08x produces its first usable quaternion.
///
/// `quaternion` is `[qx, qy, qz, qw]` in Hamilton (XYZW) order to match
/// the policy-side convention. Identity orientation is `(0, 0, 0, 1)`.
export interface ImuView {
  present: boolean;
  received: boolean;
  stale: boolean;
  lastUpdateAgeMs: number;
  quaternion: [number, number, number, number];
  headingAccuracyRad: number;
}

export interface RuntimeSnapshot {
  hostUnixMs: number;
  mode: RuntimeMode;
  estopLatched: boolean;
  estopReason: string;
  motors: MotorView[];
  buses: BusView[];
  /// Always present in the view layer; `power.present === false` when
  /// the firmware has no `power:` block configured.
  power: PowerView;
  /// Always present in the view layer; `imu.present === false` when
  /// the firmware has no `imu:` block configured.
  imu: ImuView;
}
