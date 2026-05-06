// WebSocket-based BebopTransport that talks to bebop-agent's network
// control surface (jetson-agent/bebop-agent/src/ws.rs) instead of going
// through BLE.
//
// Why this exists
// ===============
// The "Connect by IP" path skips Bluetooth entirely (workstation on the
// LAN talking to a robot already on Wi-Fi, or a browser without
// Web Bluetooth support). That path can reach `bebop-linux` for motors,
// but the controller pairing API lives in `bebop-agent` — historically
// only reachable over BLE. The agent now exposes the same
// `bebop.v1.ClientRequest` / `AgentResponse` envelope over a binary
// WebSocket on port 9091 by default; this transport speaks that surface.
//
// The implementation is intentionally minimal: the IP-only flow only
// needs the controller methods, so the BLE-specific ones (scan / connect /
// disconnect) and the provisioning ones (Wi-Fi, robot config, container
// management, OTA) throw a clear "not supported over WS" error. Add
// the corresponding request paths here when the IP-only path grows new
// screens that need them — the agent's dispatcher already serves them
// all (see `bebop-agent/src/ble/dispatcher.rs`).
//
// Lives next to the BLE transports because it implements the same
// `BebopTransport` interface; the folder name is slightly anachronistic
// at this point but keeping all transports together is easier to reason
// about than scattering them across `/ble/` and a hypothetical `/agent/`.

import { create, fromBinary, toBinary } from "@bufbuild/protobuf";

import {
  AgentResponseSchema,
  ClientRequestSchema,
  GetControllerStatusRequestSchema,
  PairControllerRequestSchema,
  ResponseStatus,
  ScanControllersRequestSchema,
  UnpairControllerRequestSchema,
  type AgentResponse,
  type ClientRequest,
  type ControllerStatus as ProtoControllerStatus,
  type DiscoveredController as ProtoDiscoveredController,
} from "../proto/bebop_pb";
import type { BebopTransport } from "./transport";
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

/// Default port the agent's WS control surface listens on. Matches the
/// `default_net_bind_addr` constant in `bebop-agent/src/config/mod.rs`.
/// The runtime server (`bebop-linux`) runs on 9090, so the agent picks
/// the next port.
export const DEFAULT_AGENT_PORT = 9091;

/// Per-request timeout in milliseconds. The agent's controller scan can
/// take up to ~8s by default; we cap a bit above that to leave headroom
/// for the WS round-trip.
const REQUEST_TIMEOUT_MS = 12_000;

type ClientPayload = NonNullable<ClientRequest["payload"]>;
type AgentPayload = NonNullable<AgentResponse["payload"]>;

interface PendingResolver {
  resolve: (msg: AgentResponse) => void;
  reject: (err: Error) => void;
  timer: ReturnType<typeof setTimeout>;
}

/// BebopTransport implementation that proxies to bebop-agent's WS
/// control surface. Reuses the same protobuf envelope BLE uses.
export class WsAgentTransport implements BebopTransport {
  private ws: WebSocket | null = null;
  private nextRequestId = 1;
  private pending = new Map<number, PendingResolver>();

  constructor(
    private readonly host: string,
    private readonly port: number = DEFAULT_AGENT_PORT,
  ) {}

  // -------------------------------------------------------------- lifecycle

  /// Open the WS and resolve when the handshake completes. Mirrors the
  /// runtime transport's connect() so the screens that use this don't
  /// have to special-case WS state. Idempotent: a no-op when already
  /// OPEN, restarts the socket if mid-handshake.
  connectWs(): Promise<void> {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      return Promise.resolve();
    }
    if (this.ws && this.ws.readyState === WebSocket.CONNECTING) {
      try {
        this.ws.close();
      } catch {
        /* ignore */
      }
      this.ws = null;
    }
    return new Promise((resolve, reject) => {
      const url = `ws://${this.host}:${this.port}/ws`;
      const ws = new WebSocket(url);
      ws.binaryType = "arraybuffer";
      this.ws = ws;
      let settled = false;

      ws.onopen = () => {
        if (settled) return;
        settled = true;
        // If disconnect raced the handshake, throw away the socket we
        // just opened (otherwise the agent-side connection lingers).
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

  disconnectWs(): void {
    const ws = this.ws;
    if (!ws) return;
    this.ws = null;
    this.failAllPending("WS disconnected by client");
    try {
      ws.close();
    } catch {
      /* ignore */
    }
  }

  /// True iff the underlying WS is OPEN. Reflects the network surface,
  /// not the BLE connection state required by `BebopTransport.isConnected`.
  isWsConnected(): boolean {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
  }

  private failAllPending(message: string): void {
    if (this.pending.size === 0) return;
    for (const [, p] of this.pending) {
      clearTimeout(p.timer);
      p.reject(new Error(message));
    }
    this.pending.clear();
  }

  // -------------------------------------------------------------- request

  private async request(payload: ClientPayload): Promise<AgentPayload> {
    const reply = await this.requestRaw(payload);
    if (reply.status !== ResponseStatus.OK) {
      const msg = reply.message || "agent rejected request";
      throw new Error(msg);
    }
    if (!reply.payload || reply.payload.case === undefined) {
      throw new Error(reply.message || "agent returned empty payload");
    }
    return reply.payload;
  }

  /// Returns the raw `AgentResponse` (including non-OK ones) so callers
  /// that need to inspect `message` on success can do so.
  private requestRaw(payload: ClientPayload): Promise<AgentResponse> {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      return Promise.reject(new Error("agent WS not connected"));
    }
    const requestId = this.nextRequestId++;
    const msg = create(ClientRequestSchema, {
      requestId,
      payload,
    });
    const bytes = toBinary(ClientRequestSchema, msg);
    this.ws.send(bytes);

    return new Promise<AgentResponse>((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(requestId);
        reject(new Error(`timeout waiting for agent response (id=${requestId})`));
      }, REQUEST_TIMEOUT_MS);
      this.pending.set(requestId, { resolve, reject, timer });
    });
  }

  private onMessage(ev: MessageEvent): void {
    if (!(ev.data instanceof ArrayBuffer)) return;
    const bytes = new Uint8Array(ev.data);
    let msg: AgentResponse;
    try {
      msg = fromBinary(AgentResponseSchema, bytes);
    } catch {
      // Drop malformed frames quietly; the agent won't normally send
      // unsolicited messages, so a decode failure is almost always
      // recoverable on the next frame.
      return;
    }
    const id = msg.requestId;
    const pending = this.pending.get(id);
    if (!pending) {
      // Unsolicited / stale response. Ignore: the request that asked
      // for it has already resolved or timed out.
      return;
    }
    this.pending.delete(id);
    clearTimeout(pending.timer);
    pending.resolve(msg);
  }

  // -------------------------------------------------------------- BLE methods
  //
  // These are required by the BebopTransport interface but don't make
  // sense over WS. The WS path always lands directly on the IP-only
  // flow, which never asks the transport to scan / pick / connect.

  scan(_timeoutMs: number): Promise<DiscoveredRobot[]> {
    return Promise.resolve([]);
  }
  pickDevice(): Promise<DiscoveredRobot | null> {
    return Promise.resolve(null);
  }
  async connect(_robotId: string): Promise<void> {
    throw new Error("BLE connect is not supported on the WS agent transport");
  }
  async disconnect(): Promise<void> {
    /* nothing to disconnect at the BLE layer; ws lifecycle is
     * managed via connectWs/disconnectWs. */
  }
  isConnected(): boolean {
    return this.isWsConnected();
  }

  // -------------------------------------------------------------- not yet wired
  //
  // Provisioning surfaces the agent supports but no IP-only screen
  // currently calls. Wire them up here when a screen needs them
  // (the dispatcher already implements the agent side).

  async getDeviceInfo(): Promise<DeviceInfo> {
    throw notSupported("getDeviceInfo");
  }
  async scanWifi(): Promise<WifiNetwork[]> {
    throw notSupported("scanWifi");
  }
  async setWifiCredentials(): Promise<void> {
    throw notSupported("setWifiCredentials");
  }
  async getWifiStatus(): Promise<WifiStatus> {
    throw notSupported("getWifiStatus");
  }
  async getRobotConfig(): Promise<RobotConfig> {
    throw notSupported("getRobotConfig");
  }
  async setRobotConfig(): Promise<void> {
    throw notSupported("setRobotConfig");
  }
  async getAppStatus(): Promise<AppStatus> {
    throw notSupported("getAppStatus");
  }
  async controlApp(): Promise<void> {
    throw notSupported("controlApp");
  }
  async setAppImage(): Promise<AppStatus> {
    throw notSupported("setAppImage");
  }
  async triggerOta(): Promise<void> {
    throw notSupported("triggerOta");
  }
  async getOtaStatus(): Promise<OtaStatus> {
    throw notSupported("getOtaStatus");
  }

  // -------------------------------------------------------------- controllers

  async scanControllers(timeoutMs: number): Promise<DiscoveredController[]> {
    const payload = await this.request({
      case: "scanControllers",
      value: create(ScanControllersRequestSchema, { timeoutMs }),
    });
    if (payload.case !== "controllerScanResult") {
      throw new Error(`expected ControllerScanResult, got ${String(payload.case)}`);
    }
    return payload.value.devices.map(controllerFromProto);
  }

  async pairController(mac: string): Promise<ControllerStatus> {
    const payload = await this.request({
      case: "pairController",
      value: create(PairControllerRequestSchema, { mac }),
    });
    if (payload.case !== "controllerStatus") {
      throw new Error(`expected ControllerStatus, got ${String(payload.case)}`);
    }
    return statusFromProto(payload.value);
  }

  async unpairController(mac: string): Promise<ControllerStatus> {
    const payload = await this.request({
      case: "unpairController",
      value: create(UnpairControllerRequestSchema, { mac }),
    });
    if (payload.case !== "controllerStatus") {
      throw new Error(`expected ControllerStatus, got ${String(payload.case)}`);
    }
    return statusFromProto(payload.value);
  }

  async getControllerStatus(): Promise<ControllerStatus> {
    const payload = await this.request({
      case: "getControllerStatus",
      value: create(GetControllerStatusRequestSchema, {}),
    });
    if (payload.case !== "controllerStatus") {
      throw new Error(`expected ControllerStatus, got ${String(payload.case)}`);
    }
    return statusFromProto(payload.value);
  }
}

// ---------------------------------------------------------------------------
// proto <-> view helpers
// ---------------------------------------------------------------------------

function controllerFromProto(d: ProtoDiscoveredController): DiscoveredController {
  return {
    mac: d.mac,
    name: d.name,
    rssi: d.rssi,
    paired: d.paired,
    connected: d.connected,
    kind: d.kind === "gamepad" ? "gamepad" : "unknown",
  };
}

function statusFromProto(s: ProtoControllerStatus): ControllerStatus {
  return {
    enabled: s.enabled,
    pairedMac: s.pairedMac,
    deviceName: s.deviceName,
    connected: s.connected,
    armed: s.armed,
    estopLatched: s.estopLatched,
    // proto int64 → bigint; UI deals in number (ms-since-epoch fits
    // comfortably in a JS number for the next ~285k years).
    lastEventUnixMs: Number(s.lastEventUnixMs),
    targetAddr: s.targetAddr,
  };
}

function notSupported(name: string): Error {
  return new Error(`${name} is not implemented on the WS agent transport`);
}
