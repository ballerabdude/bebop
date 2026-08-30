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
/// wheel-encoder odometry pose. `present` is false on the humanoid.
export interface DriveView {
  present: boolean;
  cmdLinearX: number; // m/s
  cmdAngularZ: number; // rad/s
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

/// Camera gimbal telemetry view. Mirrors the firmware's `CameraState`
/// proto. Always present in the view layer; `present === false` when the
/// firmware has no `video:` block configured — the UI should hide the
/// PTZ controls. Pose is the *actual* read-back pose (the gimbal
/// occasionally applies its own horizon compensation, so never assume
/// the commanded target).
export interface CameraView {
  present: boolean;
  /// Degrees, raw UVC pan/tilt units. The proto doc claims
  /// "pan + = right", but on the OBSBOT Tiny 2 measured hardware it
  /// is the opposite: increasing UVC pan turns the camera LEFT (and
  /// the read-back follows). `useCameraPtz` negates pan commands at
  /// the single place they're generated so every input path (pad,
  /// I/J/K/L keys, the gamepad's right stick) drives the camera the
  /// way the operator pushes; these read-back values therefore match
  /// what was commanded, in the camera's own space. Tilt is
  /// unquirky (+ = up). OBSBOT Tiny 2 range: pan ±130° / ±90°.
  panDeg: number;
  tiltDeg: number;
  /// True while the actual pose differs from the last commanded target
  /// by more than ~1 degree on either axis (gimbal still slewing).
  moving: boolean;
}

/// Navigable-path runner summary (telemetry). Mirrors the firmware's
/// `NavState` proto. `present === false` when the robot has no `nav:`
/// block or the model failed to load — the video overlay should hide
/// its toggle then. `received === false` until the first mask.
export interface NavView {
  present: boolean;
  received: boolean;
  /// Measured mask rate (Hz).
  maskHz: number;
  /// Execution provider serving the model ("cuda" / "cpu").
  provider: string;
  /// Camera frame sequence number the latest mask was computed from.
  seq: number;
  /// Camera capture timestamp (µs since epoch; matches the
  /// `X-Timestamp-Us` part header on the video stream).
  tsUs: number;
  /// Per-class fractions of the label map, 0..1.
  fracBlocked: number;
  fracNavigable: number;
  fracCaution: number;
}

/// One pushed navigable-path mask grid (~10 Hz, subscribe-gated).
/// Mirrors the firmware's `NavMaskFrame` proto. `grid` is one label
/// byte per cell, row-major: 0 = blocked, 1 = navigable, 2 = caution;
/// `width` × `height` cells cover the full camera frame.
export interface NavMaskView {
  seq: number;
  tsUs: number;
  width: number;
  height: number;
  grid: Uint8Array;
  fracBlocked: number;
  fracNavigable: number;
  fracCaution: number;
  maskHz: number;
  provider: string;
}

export interface RuntimeSnapshot {
  hostUnixMs: number;
  mode: RuntimeMode;
  estopLatched: boolean;
  estopReason: string;
  motors: MotorView[];
  buses: BusView[];
  /// ODrive wheels (empty on the legged humanoid).
  wheels: WheelView[];
  /// Differential-drive summary; `drive.present === false` on the humanoid.
  drive: DriveView;
  /// Always present in the view layer; `power.present === false` when
  /// the firmware has no `power:` block configured.
  power: PowerView;
  /// Always present in the view layer; `imu.present === false` when
  /// the firmware has no `imu:` block configured.
  imu: ImuView;
  /// Camera gimbal state; `camera.present === false` when the firmware
  /// has no `video:` block configured (no PTZ controls then).
  camera: CameraView;
  /// Navigable-path runner summary; `nav.present === false` when the
  /// firmware has no `nav:` block or the model failed to load (the
  /// video overlay toggle hides then).
  nav: NavView;
  /// Policy observation/action I/O. `policyIo.present === false` when
  /// `policy.onnx` failed to load at boot.
  policyIo: PolicyIoView;
}

/// Policy I/O view. Mirrors the scalar capture/lifecycle fields of the
/// firmware's `PolicyIoStats` proto. The live observation/action/kp/kd
/// vectors are intentionally NOT surfaced here anymore — operators
/// inspect those post-hoc in Foxglove via the downloaded MCAP, so
/// carrying them per-frame through the view layer was pure overhead.
export interface PolicyIoView {
  present: boolean;
  /** Operator has enabled dry-run: RUN_POLICY still infers + publishes +
   *  captures, but no PD commands reach the motors. */
  dryRun: boolean;
  /** Robot is currently writing MCAP capture samples. Distinct from
   *  "operator requested capture" — only `true` once the writer thread
   *  has actually opened the file on disk. */
  captureActive: boolean;
  /** Absolute path of the active capture file on the robot, or "" when
   *  no capture is open. `.mcap` extension. */
  capturePath: string;
  /** Sample count appended to the active capture file (0 when none). */
  captureRows: number;
  /** Cumulative samples the tick thread tried to enqueue but the writer
   *  thread's bounded channel was full for. Monotonic across the process
   *  lifetime; 0 means the writer is keeping up. UIs should flag a
   *  non-zero value as data loss. */
  captureDropped: number;
}
