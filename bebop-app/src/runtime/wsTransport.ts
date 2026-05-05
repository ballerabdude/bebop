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
  SetModeSchema,
  SetMotorEnabledSchema,
  SetMotorTargetSchema,
  SubscribeTelemetrySchema,
  UnsubscribeTelemetrySchema,
  type ClientRuntimeMessage,
  type Snapshot,
  type ServerRuntimeMessage,
  type TelemetryFrame,
  type MotorState as ProtoMotorState,
  type BusEntry as ProtoBusEntry,
  type PowerStats as ProtoPowerStats,
} from "../proto/bebop_runtime_pb";
import type {
  BusView,
  MotorView,
  PowerView,
  RuntimeMode,
  RuntimeSnapshot,
} from "./types";

const DEFAULT_PORT = 9090;
const ACK_TIMEOUT_MS = 5_000;

type PendingResolver = (msg: ServerRuntimeMessage) => void;
type TelemetryListener = (snapshot: RuntimeSnapshot) => void;
type EStopListener = (reason: string) => void;
type ModeListener = (mode: RuntimeMode) => void;

type ClientPayload = NonNullable<ClientRuntimeMessage["payload"]>;

export class RuntimeTransport {
  private ws: WebSocket | null = null;
  private nextRequestId = 1;
  private pending = new Map<number, PendingResolver>();
  private telemetryListeners = new Set<TelemetryListener>();
  private estopListeners = new Set<EStopListener>();
  private modeListeners = new Set<ModeListener>();

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
   */
  connect(host: string, port: number = DEFAULT_PORT): Promise<void> {
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
    return new Promise((resolve, reject) => {
      const url = `ws://${host}:${port}/ws`;
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
      };
      ws.onmessage = (ev) => this.onMessage(ev);
    });
  }

  disconnect(): void {
    const ws = this.ws;
    if (!ws) return;
    this.ws = null;
    // Fail in-flight requests synchronously so awaiters get a useful
    // error immediately, not after the close handshake completes.
    this.failAllPending("WS disconnected by client");
    try {
      ws.close();
    } catch {
      /* ignore */
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
  }

  async unsubscribeTelemetry(): Promise<void> {
    await this.requestAck({
      case: "unsubscribeTelemetry",
      value: create(UnsubscribeTelemetrySchema, {}),
    });
  }

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
    const msg = fromBinary(ServerRuntimeMessageSchema, bytes);
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
    power: powerFromProto(s.power),
  };
}

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
