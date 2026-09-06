// Runtime WebSocket transport.
//
// Connects to ws://<robot_ip>:<port>/ws and exchanges binary protobuf
// frames (`ClientRuntimeMessage` / `ServerRuntimeMessage`).
//
// Two distinct flow shapes the API has to serve:
//
//  - **Request/response**: arm / disarm / set mode / E-STOP. We tag each
//    outgoing message with a monotonic `requestId` and resolve a pending
//    Promise when the matching `Ack` / `Error` / `Snapshot` arrives.
//  - **Push streams**: telemetry (frequent), mode changes, E-STOP latches.
//    Listeners registered via `onTelemetry()` / `onModeChanged()` etc.
//    receive every matching frame.
//
// The runtime API is intentionally separate from the BLE control surface
// in `src/ble/`; the BLE transport is for one-shot setup over Bluetooth,
// while this one is the high-bandwidth IP path used after Wi-Fi config.

import { create, fromBinary, toBinary } from "@bufbuild/protobuf";
import {
  ClientRuntimeMessageSchema,
  EmergencyStopSchema,
  GetSnapshotSchema,
  Mode,
  ResetEStopSchema,
  ServerRuntimeMessageSchema,
  SetAllMotorsEnabledSchema,
  SetMechanicalZeroSchema,
  SetMechanicalZeroAllSchema,
  SetModeSchema,
  SetMotorEnabledSchema,
  SetMotorTargetSchema,
  SetPolicyDryRunSchema,
  SetVelocityCommandSchema,
  SetWheelEnabledSchema,
  SetAllWheelsEnabledSchema,
  ResetOdometrySchema,
  CalibrateWheelSchema,
  SubscribeTelemetrySchema,
  UnsubscribeTelemetrySchema,
  type ClientRuntimeMessage,
  type Snapshot,
  type ServerRuntimeMessage,
  type TelemetryFrame,
  type MotorState as ProtoMotorState,
  type BusEntry as ProtoBusEntry,
  type PowerStats as ProtoPowerStats,
  type ImuStats as ProtoImuStats,
  type PolicyIoStats as ProtoPolicyIoStats,
  type WheelState as ProtoWheelState,
  type DriveState as ProtoDriveState,
} from "../proto/bebop_runtime_pb";
import type {
  BusView,
  DriveView,
  ImuView,
  MotorView,
  PolicyIoView,
  PowerView,
  RuntimeMode,
  RuntimeSnapshot,
  WheelView,
} from "./types";

const DEFAULT_PORT = 9090;
const ACK_TIMEOUT_MS = 5_000;
/// How long to wait between auto-reconnect attempts after an unintended
/// socket close (server reboot, Wi-Fi flap, etc.). Matches the
/// firmware's ~5 s telemetry-pump period so the operator sees recovery
/// within a couple of frames; short enough that a brief outage is
/// barely noticeable but long enough to avoid pegging the CPU /
/// network if the robot is genuinely down.
const RECONNECT_DELAY_MS = 5_000;

type PendingResolver = (msg: ServerRuntimeMessage) => void;
type TelemetryListener = (snapshot: RuntimeSnapshot) => void;
type EStopListener = (reason: string) => void;
type ModeListener = (mode: RuntimeMode) => void;
/// Nav-mask pushes are high-rate (~10 Hz × 14 KB) and are consumed by
/// the video overlay's draw loop directly — they deliberately do NOT
/// flow through the `RuntimeSnapshot` state machine like telemetry.

/// Lifecycle of the underlying WebSocket exposed to consumers.
///
/// - `disconnected` — no socket, and we're not trying to open one.
///   Set on initial construction and after `disconnect()`.
/// - `connecting` — a `new WebSocket(...)` is in flight (initial open
///   OR an auto-reconnect attempt after an unintended close).
/// - `connected` — `onopen` fired and the socket is OPEN.
export type RuntimeConnectionState = "disconnected" | "connecting" | "connected";

type ConnectionStateListener = (state: RuntimeConnectionState) => void;

type ClientPayload = NonNullable<ClientRuntimeMessage["payload"]>;

export class RuntimeTransport {
  private ws: WebSocket | null = null;
  private nextRequestId = 1;
  private pending = new Map<number, PendingResolver>();
  private telemetryListeners = new Set<TelemetryListener>();
  private estopListeners = new Set<EStopListener>();
  private modeListeners = new Set<ModeListener>();
  private connectionStateListeners = new Set<ConnectionStateListener>();

  /// Last (host, port) the caller asked us to open. Stashed so the
  /// auto-reconnect timer knows where to dial when the socket drops.
  /// Cleared by `disconnect()`.
  private endpoint: { host: string; port: number } | null = null;
  /// True between `connect()` and `disconnect()`. When false, an
  /// `onclose` is treated as final and does NOT schedule a reconnect.
  /// Without this flag a `disconnect()` racing with a server close
  /// would still spawn a reconnect loop that the caller can't cancel.
  private wantConnected = false;
  /// `setTimeout` handle for the pending reconnect attempt, if any.
  /// Cleared on `disconnect()` and whenever the reconnect actually
  /// fires (so a fresh failure can schedule a new one).
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  /// Last subscribed telemetry rate, so we can transparently resume
  /// the subscription after an auto-reconnect. `null` means the
  /// caller has not subscribed (or has explicitly unsubscribed) and
  /// we should NOT re-issue `SubscribeTelemetry` on reconnect.
  private subscribedRateHz: number | null = null;
  /// Last subscribed nav-mask rate, so an auto-reconnect transparently
  /// resumes the overlay stream (same pattern as telemetry).
  private connectionState: RuntimeConnectionState = "disconnected";

  /** Open the socket and resolve once we get the `open` event.
   *
   *  `this.ws` is set immediately, *before* `onopen` fires, so a
   *  `disconnect()` called during the `CONNECTING` window (e.g. React
   *  StrictMode's effect cleanup, or a user navigating away mid-handshake)
   *  can still close the in-flight socket. Without this guarantee the
   *  WebSocket would silently become a zombie on the server side until
   *  the server tried to write to it (broadcasting a ModeChanged event,
   *  for example), at which point it would fail with "Sending after
   *  closing is not allowed".
   *
   *  Once `connect()` succeeds, the transport latches `wantConnected =
   *  true` and will auto-reconnect every `RECONNECT_DELAY_MS` after
   *  any subsequent close (server reboot, brief Wi-Fi outage, etc.)
   *  until `disconnect()` is called. Telemetry subscriptions are
   *  re-established automatically on each reconnect so the operator
   *  UI continues to receive frames without re-mounting. Consumers
   *  can subscribe to `onConnectionStateChange` to render a
   *  "Reconnecting…" indicator while the link is down.
   */
  connect(host: string, port: number = DEFAULT_PORT): Promise<void> {
    this.endpoint = { host, port };
    this.wantConnected = true;
    // Any pending reconnect timer is now obsolete — the caller is
    // asking us to open RIGHT NOW.
    if (this.reconnectTimer !== null) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    return this.openSocket();
  }

  /// Lower-level open used by both the initial `connect()` call and the
  /// auto-reconnect timer. Splits the host/port read out of `endpoint`
  /// so the reconnect path doesn't need to re-thread those args
  /// through callers.
  private openSocket(): Promise<void> {
    const endpoint = this.endpoint;
    if (!endpoint) {
      return Promise.reject(new Error("RuntimeTransport: no endpoint set"));
    }
    // Idempotent: if a socket is already OPEN we trust it. Callers can
    // share a cached transport (see runtime/cache.ts) and not have to
    // remember whether they were the ones who first connected it.
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      return Promise.resolve();
    }
    // If a socket is mid-handshake, return its open/error promise rather
    // than spawning a parallel one. We don't keep a Promise<void> handle
    // around for this, so just close the in-flight one and start fresh —
    // simpler than tracking per-instance state and rare in practice.
    if (this.ws && this.ws.readyState === WebSocket.CONNECTING) {
      try {
        this.ws.close();
      } catch {
        /* ignore */
      }
      this.ws = null;
    }
    this.setConnectionState("connecting");
    return new Promise((resolve, reject) => {
      const url = `ws://${endpoint.host}:${endpoint.port}/ws`;
      const ws = new WebSocket(url);
      ws.binaryType = "arraybuffer";
      this.ws = ws;
      let settled = false;

      ws.onopen = () => {
        if (settled) return;
        settled = true;
        // If disconnect() fired between `new WebSocket` and `onopen`,
        // `this.ws` was nulled out. Close the socket we just opened
        // (otherwise the server-side connection sticks around) and
        // reject so the caller doesn't think it's connected.
        if (this.ws !== ws) {
          try {
            ws.close();
          } catch {
            /* ignore */
          }
          reject(new Error("disconnected during connect"));
          return;
        }
        this.setConnectionState("connected");
        // Resume telemetry subscription if the caller had one before
        // the drop. Fire-and-forget — a failure here just means the
        // next close will trigger another reconnect, and the operator
        // already sees the "reconnecting" pulse if anything goes
        // wrong with subscription itself.
        const rate = this.subscribedRateHz;
        if (rate !== null) {
          void this.subscribeTelemetry(rate).catch(() => {
            /* surfaced via reconnect loop */
          });
        }
        resolve();
      };
      ws.onerror = () => {
        if (settled) return;
        settled = true;
        reject(new Error(`WebSocket error connecting to ${url}`));
      };
      ws.onclose = () => {
        if (this.ws === ws) {
          // Server-side close, or our own close() racing the open.
          this.ws = null;
          this.failAllPending("WS closed");
        }
        if (!settled) {
          settled = true;
          reject(new Error("WebSocket closed before open"));
        }
        // Schedule an auto-reconnect if the caller still wants us
        // online. `disconnect()` clears `wantConnected` first, so a
        // deliberate teardown never spawns a reconnect.
        this.scheduleReconnect();
      };
      ws.onmessage = (ev) => this.onMessage(ev);
    });
  }

  /// Arm the reconnect timer (if not already armed and the caller
  /// still wants to stay connected). Cleared on `disconnect()` and
  /// whenever the timer actually fires.
  private scheduleReconnect(): void {
    if (!this.wantConnected) {
      this.setConnectionState("disconnected");
      return;
    }
    if (this.reconnectTimer !== null) return;
    this.setConnectionState("connecting");
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      if (!this.wantConnected) return;
      void this.openSocket().catch(() => {
        // openSocket() will already have triggered `onclose` which
        // schedules the next attempt; nothing more to do here.
      });
    }, RECONNECT_DELAY_MS);
  }

  disconnect(): void {
    // Stop the auto-reconnect loop FIRST so a close-event onslaught
    // doesn't immediately re-arm it.
    this.wantConnected = false;
    if (this.reconnectTimer !== null) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    this.subscribedRateHz = null;
    this.endpoint = null;
    const ws = this.ws;
    this.ws = null;
    // Fail in-flight requests synchronously so awaiters get a useful
    // error immediately, not after the close handshake completes.
    this.failAllPending("WS disconnected by client");
    if (ws) {
      try {
        ws.close();
      } catch {
        /* ignore */
      }
    }
    this.setConnectionState("disconnected");
  }

  /// Current state of the underlying WebSocket. Cheap to call; the
  /// transport tracks the state internally rather than peeking at
  /// `WebSocket.readyState` so consumers see "connecting" during the
  /// auto-reconnect wait too (when there's no socket object yet).
  getConnectionState(): RuntimeConnectionState {
    return this.connectionState;
  }

  /// Register a callback fired whenever the connection state changes.
  /// Returns an unsubscribe function. The current state is delivered
  /// synchronously on subscription so the consumer can render the
  /// initial pill without an extra `getConnectionState()` call.
  onConnectionStateChange(cb: ConnectionStateListener): () => void {
    this.connectionStateListeners.add(cb);
    cb(this.connectionState);
    return () => this.connectionStateListeners.delete(cb);
  }

  private setConnectionState(next: RuntimeConnectionState): void {
    if (this.connectionState === next) return;
    this.connectionState = next;
    for (const cb of this.connectionStateListeners) {
      try {
        cb(next);
      } catch {
        /* ignore listener errors */
      }
    }
  }

  private failAllPending(message: string): void {
    if (this.pending.size === 0) return;
    for (const [id, fn] of this.pending) {
      fn({
        $typeName: "bebop.runtime.v1.ServerRuntimeMessage",
        requestId: id,
        payload: {
          case: "error",
          value: { $typeName: "bebop.runtime.v1.Error", message },
        },
      } as ServerRuntimeMessage);
    }
    this.pending.clear();
  }

  isConnected(): boolean {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
  }

  // -------------------------------------------------------------- listeners
  onTelemetry(cb: TelemetryListener): () => void {
    this.telemetryListeners.add(cb);
    return () => this.telemetryListeners.delete(cb);
  }
  onEStopLatched(cb: EStopListener): () => void {
    this.estopListeners.add(cb);
    return () => this.estopListeners.delete(cb);
  }
  onModeChanged(cb: ModeListener): () => void {
    this.modeListeners.add(cb);
    return () => this.modeListeners.delete(cb);
  }

  // -------------------------------------------------------------- requests
  async getSnapshot(): Promise<RuntimeSnapshot> {
    const msg = await this.request({
      case: "getSnapshot",
      value: create(GetSnapshotSchema, {}),
    });
    if (msg.payload.case !== "snapshot") {
      throw new Error(asErrorMessage(msg, "expected Snapshot"));
    }
    return snapshotFromProto(msg.payload.value);
  }

  async subscribeTelemetry(rateHz = 30): Promise<void> {
    await this.requestAck({
      case: "subscribeTelemetry",
      value: create(SubscribeTelemetrySchema, { rateHz }),
    });
    // Remember the rate so an auto-reconnect can transparently
    // re-subscribe without the caller having to listen for
    // connection-state changes.
    this.subscribedRateHz = rateHz;
  }

  async unsubscribeTelemetry(): Promise<void> {
    // Clear the remembered rate FIRST: if the unsubscribe RPC fails
    // we still don't want a future reconnect to silently re-arm the
    // subscription the caller explicitly tore down.
    this.subscribedRateHz = null;
    await this.requestAck({
      case: "unsubscribeTelemetry",
      value: create(UnsubscribeTelemetrySchema, {}),
    });
  }

  /// Subscribe to the pushed nav-mask frames (the video overlay feed).
  /// Rate hint is an upper bound — the firmware only pushes new masks,
  /// so a rate above the model's own inference rate just idles.

  async setMotorEnabled(jointName: string, enabled: boolean): Promise<void> {
    await this.requestAck({
      case: "setMotorEnabled",
      value: create(SetMotorEnabledSchema, { jointName, enabled }),
    });
  }

  async setAllMotorsEnabled(enabled: boolean): Promise<void> {
    await this.requestAck({
      case: "setAllMotorsEnabled",
      value: create(SetAllMotorsEnabledSchema, { enabled }),
    });
  }

  /// Command the supervisor's hold-target for one armed motor. Only
  /// effective in DIAL_IN mode and when the motor is armed; the firmware
  /// rejects with an error otherwise. The supervisor's slew limiter
  /// converts an instant target jump into a controlled per-tick move,
  /// so the UI is free to send rapid drag updates.
  async setMotorTarget(jointName: string, positionRad: number): Promise<void> {
    await this.requestAck({
      case: "setMotorTarget",
      value: create(SetMotorTargetSchema, { jointName, positionRad }),
    });
  }

  /// Re-zero the joint's mechanical origin to its current physical
  /// position. Sends Robstride SET_ZERO (CMD 0x06), which the motor
  /// commits to flash. Firmware refuses unless the joint is *disarmed*,
  /// not E-STOPed, and on a healthy CAN bus. Caller should confirm with
  /// the operator before invoking — this overwrites the motor's stored
  /// origin and cannot be undone except by re-zeroing again at a
  /// different physical position.
  async setMechanicalZero(jointName: string): Promise<void> {
    await this.requestAck({
      case: "setMechanicalZero",
      value: create(SetMechanicalZeroSchema, { jointName }),
    });
  }

  /// Batch re-zero of every actuator for the guided zero-calibration
  /// flow (robot in the reference pose — face-flat, legs straight — with
  /// every joint disarmed). The firmware checks preconditions for ALL
  /// joints before touching any motor, then reports a per-joint summary
  /// including post-zero verification failures (a motor that ignored
  /// SET_ZERO, or a joint not at the reference pose). Returns the
  /// firmware's summary message so the UI can show it verbatim.
  async setMechanicalZeroAll(): Promise<string> {
    const reply = await this.request({
      case: "setMechanicalZeroAll",
      value: create(SetMechanicalZeroAllSchema, {}),
    });
    if (reply.payload.case === "error") {
      throw new Error(reply.payload.value.message || "runtime error");
    }
    if (reply.payload.case !== "ack") {
      throw new Error(`expected Ack, got ${String(reply.payload.case)}`);
    }
    if (!reply.payload.value.ok) {
      throw new Error(reply.payload.value.message || "request failed");
    }
    return reply.payload.value.message;
  }

  async setMode(mode: RuntimeMode): Promise<void> {
    await this.requestAck({
      case: "setMode",
      value: create(SetModeSchema, { mode: modeToProto(mode) }),
    });
  }

  async emergencyStop(reason: string = "operator"): Promise<void> {
    await this.requestAck({
      case: "emergencyStop",
      value: create(EmergencyStopSchema, { reason }),
    });
  }

  async resetEStop(): Promise<void> {
    await this.requestAck({
      case: "resetEstop",
      value: create(ResetEStopSchema, {}),
    });
  }

  /// Toggle the policy dry-run flag. While enabled, RUN_POLICY still
  /// infers, publishes telemetry, and writes MCAP captures, but no PD
  /// commands reach the motors. Persists across mode transitions; the
  /// UI is responsible for showing the operator that it's on.
  async setPolicyDryRun(enabled: boolean): Promise<void> {
    await this.requestAck({
      case: "setPolicyDryRun",
      value: create(SetPolicyDryRunSchema, { enabled }),
    });
  }

  // -------------------------------------------------------------- drive

  /// Drive a twist (body frame) into the differential-drive chassis.
  /// `linearX` (m/s forward) and `angularZ` (rad/s yaw, + left). The
  /// firmware clamps + slew-limits and converts to per-wheel velocity;
  /// send (0, 0) to stop. No-op on the humanoid (no `drive:` block).
  async setVelocityCommand(linearX: number, angularZ: number): Promise<void> {
    await this.requestAck({
      case: "setVelocityCommand",
      value: create(SetVelocityCommandSchema, {
        linearX,
        angularZ,
      }),
    });
  }

  /// Arm/disarm a single ODrive wheel. `wheelName` matches a `wheels:[]`
  /// key; arming sets velocity mode + closed-loop control.
  async setWheelEnabled(wheelName: string, enabled: boolean): Promise<void> {
    await this.requestAck({
      case: "setWheelEnabled",
      value: create(SetWheelEnabledSchema, { wheelName, enabled }),
    });
  }

  /// Arm/disarm every configured wheel at once.
  async setAllWheelsEnabled(enabled: boolean): Promise<void> {
    await this.requestAck({
      case: "setAllWheelsEnabled",
      value: create(SetAllWheelsEnabledSchema, { enabled }),
    });
  }

  /// Run the ODrive FULL_CALIBRATION_SEQUENCE on one wheel (~20-30 s spin).
  /// The axis must be disarmed first. Recovers a lost encoder calibration —
  /// the CAN result is NOT saved to the S1's NVM, so it must be re-run
  /// after a power cycle (or persisted once via `odrivetool` over USB).
  async calibrateWheel(wheelName: string): Promise<void> {
    await this.requestAck({
      case: "calibrateWheel",
      value: create(CalibrateWheelSchema, { wheelName }),
    });
  }

  // -------------------------------------------------------------- camera

  /// Command the camera gimbal to an absolute pan/tilt pose in degrees
  /// (pan + = right, tilt + = up; 0/0 = power-on center). The firmware
  /// clamps to the camera's UVC limits (OBSBOT Tiny 2: pan ±130°, tilt
  /// ±90°) and is deliberately NOT mode-gated — PTZ look-around is safe
  /// in any mode (dataset recording included). The settled pose arrives
  /// via telemetry (`camera` field of the snapshot views); a 30° move
  /// takes ~0.5-0.7 s, so relative jogging should read the latest pose
  /// from a snapshot rather than tracking locally.

  // -------------------------------------------------------------- misc

  /// Reset the wheel-encoder odometry pose to the origin.
  async resetOdometry(): Promise<void> {
    await this.requestAck({
      case: "resetOdometry",
      value: create(ResetOdometrySchema, {}),
    });
  }

  // -------------------------------------------------------------- internals
  private async requestAck(payload: ClientPayload): Promise<void> {
    const reply = await this.request(payload);
    if (reply.payload.case === "error") {
      throw new Error(reply.payload.value.message || "runtime error");
    }
    if (reply.payload.case !== "ack") {
      throw new Error(`expected Ack, got ${String(reply.payload.case)}`);
    }
    if (!reply.payload.value.ok) {
      throw new Error(reply.payload.value.message || "request failed");
    }
  }

  private request(payload: ClientPayload): Promise<ServerRuntimeMessage> {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      return Promise.reject(new Error("WebSocket not connected"));
    }
    const requestId = this.nextRequestId++;
    const msg = create(ClientRuntimeMessageSchema, {
      requestId,
      payload,
    });
    const bytes = toBinary(ClientRuntimeMessageSchema, msg);
    this.ws.send(bytes);

    return new Promise<ServerRuntimeMessage>((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(requestId);
        reject(new Error(`timeout waiting for response (id=${requestId})`));
      }, ACK_TIMEOUT_MS);
      this.pending.set(requestId, (m) => {
        clearTimeout(timer);
        resolve(m);
      });
    });
  }

  private onMessage(ev: MessageEvent): void {
    if (!(ev.data instanceof ArrayBuffer)) return;
    const bytes = new Uint8Array(ev.data);
    // A corrupt/partial frame decoding off the wire shouldn't tear down
    // the whole WS receive path. Drop quietly and let the next frame
    // recover — mirrors the wsAgentTransport's handling. The firmware's
    // telemetry pump is resilient to a missed frame.
    let msg: ServerRuntimeMessage;
    try {
      msg = fromBinary(ServerRuntimeMessageSchema, bytes);
    } catch {
      return;
    }
    const id = msg.requestId;

    // Solicited responses.
    if (id !== 0 && this.pending.has(id)) {
      const fn = this.pending.get(id)!;
      this.pending.delete(id);
      fn(msg);
      return;
    }

    // Async push events.
    switch (msg.payload.case) {
      case "telemetry":
        for (const cb of this.telemetryListeners) {
          cb(snapshotFromProto(msg.payload.value));
        }
        break;
      case "estopLatched":
        for (const cb of this.estopListeners) cb(msg.payload.value.reason);
        break;
      case "modeChanged":
        for (const cb of this.modeListeners) cb(modeFromProto(msg.payload.value.mode));
        break;
      case "snapshot":
      case "ack":
      case "error":
      case "busStatus":
      case undefined:
        // Either an unsolicited push variant we don't subscribe to yet,
        // or a stray response without a pending entry. Drop quietly.
        break;
    }
  }
}

// --------------------------------------------------------- proto <-> view
function snapshotFromProto(s: Snapshot | TelemetryFrame): RuntimeSnapshot {
  return {
    hostUnixMs: Number(s.hostUnixMs),
    mode: modeFromProto(s.mode),
    estopLatched: s.estopLatched,
    estopReason: s.estopReason,
    motors: s.motors.map(motorFromProto),
    buses: s.buses.map(busFromProto),
    wheels: s.wheels.map(wheelFromProto),
    drive: driveFromProto(s.drive),
    power: powerFromProto(s.power),
    imu: imuFromProto(s.imu),
    policyIo: policyIoFromProto(s.policyIo),
  };
}

function wheelFromProto(w: ProtoWheelState): WheelView {
  return {
    name: w.name,
    canInterface: w.canInterface,
    nodeId: w.nodeId,
    armed: w.armed,
    feedbackStale: w.feedbackStale,
    positionReceived: w.positionReceived,
    errorCode: w.errorCode,
    position: w.positionRad,
    velocity: w.velocityRadS,
    targetVelocity: w.targetVelocityRadS,
    velMax: w.velMax,
    axisState: w.axisState,
  };
}

function driveFromProto(d: ProtoDriveState | undefined): DriveView {
  if (!d) {
    return EMPTY_DRIVE_VIEW;
  }
  return {
    present: d.present,
    cmdLinearX: d.cmdLinearX,
    cmdAngularZ: d.cmdAngularZ,
    operatorStale: d.operatorStale,
    hasActiveOperator: d.hasActiveOperator,
    youAreActiveOperator: d.youAreActiveOperator,
    odomX: d.odomX,
    odomY: d.odomY,
    odomTheta: d.odomTheta,
  };
}

const EMPTY_DRIVE_VIEW: DriveView = {
  present: false,
  cmdLinearX: 0,
  cmdAngularZ: 0,
  operatorStale: false,
  hasActiveOperator: false,
  youAreActiveOperator: false,
  odomX: 0,
  odomY: 0,
  odomTheta: 0,
};

function imuFromProto(p: ProtoImuStats | undefined): ImuView {
  // Older firmware (or a decode where the field is absent) collapses to
  // "no IMU configured" — `present=false` hides the orientation card in
  // the operator UI.
  if (!p) {
    return EMPTY_IMU_VIEW;
  }
  return {
    present: p.present,
    received: p.received,
    stale: p.stale,
    lastUpdateAgeMs: p.lastUpdateAgeMs,
    quaternion: [p.quaternionX, p.quaternionY, p.quaternionZ, p.quaternionW],
    headingAccuracyRad: p.headingAccuracyRad,
  };
}

const EMPTY_IMU_VIEW: ImuView = {
  present: false,
  received: false,
  stale: true,
  lastUpdateAgeMs: 0,
  quaternion: [0, 0, 0, 1],
  headingAccuracyRad: 0,
};

function policyIoFromProto(p: ProtoPolicyIoStats | undefined): PolicyIoView {
  if (!p) {
    return EMPTY_POLICY_IO_VIEW;
  }
  // Only the scalar capture/lifecycle fields are surfaced to the view
  // layer — the observation/action/kp/kd vectors are inspected post-hoc
  // in Foxglove via the downloaded MCAP, so we intentionally don't copy
  // them per-frame here.
  return {
    present: p.present,
    dryRun: p.dryRun,
    captureActive: p.captureActive,
    capturePath: p.capturePath,
    captureRows: Number(p.captureRows),
    captureDropped: Number(p.captureDropped),
  };
}

const EMPTY_POLICY_IO_VIEW: PolicyIoView = {
  present: false,
  dryRun: false,
  captureActive: false,
  capturePath: "",
  captureRows: 0,
  captureDropped: 0,
};

function powerFromProto(p: ProtoPowerStats | undefined): PowerView {
  // Older firmware (or a transient decode where the field is absent)
  // collapses to "no power board configured" — present=false hides the
  // power card entirely on the operator UI.
  if (!p) {
    return EMPTY_POWER_VIEW;
  }
  return {
    present: p.present,
    canInterface: p.canInterface,
    powerId: p.powerId,
    firmwareVersion: p.firmwareVersion,
    statusReceived: p.statusReceived,
    statusStale: p.statusStale,
    lastStatusAgeMs: p.lastStatusAgeMs,
    batteryVoltageV: p.batteryVoltageV,
    motorVoltageV: p.motorVoltageV,
    boardTemperatureC: p.boardTemperatureC,
    faultBits: p.faultBits,
    faultDescription: p.faultDescription,
    rail12vOn: p.rail12vOn,
    softStartOn: p.softStartOn,
    motorRailOn: p.motorRailOn,
    rail24vOn: p.rail24vOn,
    currentAlA: p.currentAlA,
    currentArA: p.currentArA,
    currentLlA: p.currentLlA,
    currentLrA: p.currentLrA,
    totalMotorCurrentA: p.totalMotorCurrentA,
    batteryCells: p.batteryCells,
    packFullVoltageV: p.packFullVoltageV,
    packEmptyVoltageV: p.packEmptyVoltageV,
    stateOfChargePct: p.stateOfChargePct,
  };
}

const EMPTY_POWER_VIEW: PowerView = {
  present: false,
  canInterface: "",
  powerId: 0,
  firmwareVersion: "",
  statusReceived: false,
  statusStale: false,
  lastStatusAgeMs: 0,
  batteryVoltageV: 0,
  motorVoltageV: 0,
  boardTemperatureC: 0,
  faultBits: 0,
  faultDescription: "",
  rail12vOn: false,
  softStartOn: false,
  motorRailOn: false,
  rail24vOn: false,
  currentAlA: 0,
  currentArA: 0,
  currentLlA: 0,
  currentLrA: 0,
  totalMotorCurrentA: 0,
  batteryCells: 0,
  packFullVoltageV: 0,
  packEmptyVoltageV: 0,
  stateOfChargePct: -1,
};

function motorFromProto(m: ProtoMotorState): MotorView {
  return {
    jointName: m.jointName,
    canInterface: m.canInterface,
    motorId: m.motorId,
    model: m.model,
    armed: m.armed,
    feedbackStale: m.feedbackStale,
    faultBits: m.faultBits,
    position: m.positionRad,
    velocity: m.velocityRadS,
    torque: m.torqueNm,
    temperature: m.temperatureC,
    target: m.targetPositionRad,
    posMin: m.posMinRad,
    posMax: m.posMaxRad,
    velMax: m.velMax,
    tauMax: m.tauMax,
    tempMax: m.tempMax,
  };
}

function busFromProto(b: ProtoBusEntry): BusView {
  return {
    canInterface: b.canInterface,
    state: b.state,
    healthy: b.healthy,
  };
}

function modeFromProto(m: Mode): RuntimeMode {
  switch (m) {
    case Mode.IDLE:
      return "IDLE";
    case Mode.DIAL_IN:
      return "DIAL_IN";
    case Mode.RUN_POLICY:
      return "RUN_POLICY";
    case Mode.UNSPECIFIED:
    default:
      return "UNSPECIFIED";
  }
}

function modeToProto(m: RuntimeMode): Mode {
  switch (m) {
    case "IDLE":
      return Mode.IDLE;
    case "DIAL_IN":
      return Mode.DIAL_IN;
    case "RUN_POLICY":
      return Mode.RUN_POLICY;
    case "UNSPECIFIED":
    default:
      return Mode.UNSPECIFIED;
  }
}

function asErrorMessage(msg: ServerRuntimeMessage, fallback: string): string {
  if (msg.payload.case === "error") {
    return msg.payload.value.message || fallback;
  }
  return fallback;
}
