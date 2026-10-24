// SetupTransport — protobuf-over-WebSocket client for `bebop-agent`'s
// provisioning server (`jetson-agent/bebop-agent/src/server.rs`, :9091).
//
// Two ways the app reaches it:
//   * Over the LAN once the robot is on Wi-Fi (host = robot IP).
//   * Directly over the robot's setup hotspot during first-time setup
//     (host = the hotspot gateway, 192.168.42.1).
//
// The socket opens lazily on the first request and re-opens on the next
// request after a drop (e.g. the robot switching networks), so callers never
// manage its lifecycle beyond `disconnect()`.
//
// `disconnect()` is not terminal: the next request opens a fresh socket. That
// matters under React StrictMode, which runs an effect's cleanup and then the
// effect again on the same memoized transport. A disconnect while a socket is
// still CONNECTING abandons that attempt (its callers get "disconnected") but
// never hands the dead attempt to later callers. Before this, the second
// mount awaited the first mount's aborted socket and Settings › Network
// showed "WebSocket error connecting to ws://…:9091/ws" against a healthy
// agent.
//
// The runtime/operator surface (motors, telemetry, video) is separate; see
// `src/robot/endpoints.ts`.

import { create, fromBinary, toBinary } from "@bufbuild/protobuf";

import {
  AgentResponseSchema,
  ClientRequestSchema,
  GetDeviceInfoRequestSchema,
  GetNetworkConfigRequestSchema,
  GetRobotConfigRequestSchema,
  GetWifiStatusRequestSchema,
  ResponseStatus,
  ScanWifiRequestSchema,
  SetNetworkConfigRequestSchema,
  SetRobotConfigRequestSchema,
  SetWifiCredentialsRequestSchema,
  type AgentResponse,
  type ClientRequest,
  type NetworkConfig as ProtoNetworkConfig,
} from "../proto/bebop_pb.ts";
import { endpoints } from "../robot/endpoints.ts";
import type { BebopTransport, WifiJoinResult } from "./transport.ts";
import type {
  ApBand,
  DeviceInfo,
  NetworkConfig,
  NetworkMode,
  RobotConfig,
  WifiNetwork,
  WifiStatus,
} from "./types.ts";

const REQUEST_TIMEOUT_MS = 12_000;

type ClientPayload = NonNullable<ClientRequest["payload"]>;
type AgentPayload = NonNullable<AgentResponse["payload"]>;

interface PendingResolver {
  resolve: (msg: AgentResponse) => void;
  reject: (err: Error) => void;
  timer: ReturnType<typeof setTimeout>;
}

export class SetupTransport implements BebopTransport {
  private ws: WebSocket | null = null;
  private nextRequestId = 1;
  private pending = new Map<number, PendingResolver>();

  private opening: Promise<void> | null = null;
  /// Bumped by `disconnect()`; an open attempt from an older generation
  /// never becomes the live socket.
  private generation = 0;

  constructor(readonly host: string) {}

  // -------------------------------------------------------------- lifecycle

  connect(): Promise<void> {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      return Promise.resolve();
    }
    if (this.opening) return this.opening;
    const opening = this.open().finally(() => {
      if (this.opening === opening) this.opening = null;
    });
    this.opening = opening;
    return opening;
  }

  private open(): Promise<void> {
    const generation = this.generation;
    return new Promise((resolve, reject) => {
      const url = endpoints(this.host).setupWs;
      const ws = new WebSocket(url);
      ws.binaryType = "arraybuffer";
      this.ws = ws;
      let settled = false;
      const stale = () => generation !== this.generation || this.ws !== ws;

      ws.onopen = () => {
        if (settled) return;
        settled = true;
        if (stale()) {
          try {
            ws.close();
          } catch {
            /* ignore */
          }
          reject(new Error("setup connection closed by the client"));
          return;
        }
        resolve();
      };
      ws.onerror = () => {
        if (settled) return;
        settled = true;
        reject(
          new Error(
            stale()
              ? "setup connection closed by the client"
              : `WebSocket error connecting to ${url}`,
          ),
        );
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
      ws.onmessage = (ev) => {
        if (this.ws === ws) this.onMessage(ev);
      };
    });
  }

  /// Close the socket and fail in-flight requests. Not terminal: the next
  /// request opens a new socket.
  disconnect(): void {
    this.generation += 1;
    this.opening = null;
    const ws = this.ws;
    this.ws = null;
    this.failAllPending("WS disconnected by client");
    if (!ws) return;
    try {
      ws.close();
    } catch {
      /* ignore */
    }
  }

  isConnected(): boolean {
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
      throw new Error(reply.message || "agent rejected request");
    }
    if (!reply.payload || reply.payload.case === undefined) {
      // Requests that succeed with a message only (e.g. wifi join kickoff)
      // have no payload; callers that need one handle this individually.
      throw new Error(reply.message || "agent returned empty payload");
    }
    return reply.payload;
  }

  private async requestRaw(payload: ClientPayload): Promise<AgentResponse> {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      await this.connect();
    }
    const ws = this.ws;
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      throw new Error("setup connection is not open");
    }
    const requestId = this.nextRequestId++;
    const msg = create(ClientRequestSchema, { requestId, payload });
    const bytes = toBinary(ClientRequestSchema, msg);
    ws.send(bytes);

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
    let msg: AgentResponse;
    try {
      msg = fromBinary(AgentResponseSchema, new Uint8Array(ev.data));
    } catch {
      return;
    }
    const pending = this.pending.get(msg.requestId);
    if (!pending) return;
    this.pending.delete(msg.requestId);
    clearTimeout(pending.timer);
    pending.resolve(msg);
  }

  // -------------------------------------------------------------- surface

  async getDeviceInfo(): Promise<DeviceInfo> {
    const payload = await this.request({
      case: "getDeviceInfo",
      value: create(GetDeviceInfoRequestSchema, {}),
    });
    if (payload.case !== "deviceInfo") {
      throw new Error(`expected DeviceInfo, got ${String(payload.case)}`);
    }
    const d = payload.value;
    return {
      serialNumber: d.serialNumber,
      model: d.model,
      agentVersion: d.agentVersion,
      jetpackVersion: d.jetpackVersion,
      hostname: d.hostname,
    };
  }

  async scanWifi(): Promise<WifiNetwork[]> {
    const payload = await this.request({
      case: "scanWifi",
      value: create(ScanWifiRequestSchema, {}),
    });
    if (payload.case !== "wifiScanResult") {
      throw new Error(`expected WifiScanResult, got ${String(payload.case)}`);
    }
    return payload.value.networks.map((n) => ({
      ssid: n.ssid,
      signalDbm: n.signalDbm,
      security: n.security,
      saved: n.saved,
    }));
  }

  async setWifiCredentials(
    ssid: string,
    password: string,
    hidden: boolean,
  ): Promise<WifiJoinResult> {
    // In Hosted Network mode the agent saves the profile without applying
    // it (so the hotspot stays up); in Known Network mode it joins now.
    const reply = await this.requestRaw({
      case: "setWifiCredentials",
      value: create(SetWifiCredentialsRequestSchema, {
        ssid,
        password,
        hidden,
      }),
    });
    if (reply.status !== ResponseStatus.OK) {
      throw new Error(reply.message || "agent rejected Wi-Fi credentials");
    }
    const status: WifiStatus =
      reply.payload?.case === "wifiStatus"
        ? {
            connected: reply.payload.value.connected,
            ssid: reply.payload.value.ssid,
            ipAddress: reply.payload.value.ipAddress,
            signalDbm: reply.payload.value.signalDbm,
          }
        : { connected: false, ssid, ipAddress: "", signalDbm: 0 };
    return { status, message: reply.message };
  }

  async getWifiStatus(): Promise<WifiStatus> {
    const payload = await this.request({
      case: "getWifiStatus",
      value: create(GetWifiStatusRequestSchema, {}),
    });
    if (payload.case !== "wifiStatus") {
      throw new Error(`expected WifiStatus, got ${String(payload.case)}`);
    }
    const w = payload.value;
    return {
      connected: w.connected,
      ssid: w.ssid,
      ipAddress: w.ipAddress,
      signalDbm: w.signalDbm,
    };
  }

  async getRobotConfig(): Promise<RobotConfig> {
    const payload = await this.request({
      case: "getRobotConfig",
      value: create(GetRobotConfigRequestSchema, {}),
    });
    if (payload.case !== "robotConfig") {
      throw new Error(`expected RobotConfig, got ${String(payload.case)}`);
    }
    const c = payload.value;
    return {
      robotName: c.robotName,
      ownerId: c.ownerId,
      timezone: c.timezone,
      extra: { ...c.extra },
    };
  }

  async setRobotConfig(config: RobotConfig): Promise<void> {
    await this.request({
      case: "setRobotConfig",
      value: create(SetRobotConfigRequestSchema, {
        config: {
          robotName: config.robotName,
          ownerId: config.ownerId,
          timezone: config.timezone,
          extra: config.extra,
        },
      }),
    });
  }

  async getNetworkConfig(): Promise<NetworkConfig> {
    const payload = await this.request({
      case: "getNetworkConfig",
      value: create(GetNetworkConfigRequestSchema, {}),
    });
    if (payload.case !== "networkConfig") {
      throw new Error(`expected NetworkConfig, got ${String(payload.case)}`);
    }
    return networkFromProto(payload.value);
  }

  async setNetworkConfig(config: NetworkConfig): Promise<NetworkConfig> {
    // `mode` is button-owned and ignored by the agent; send only the
    // Hosted Network settings.
    const payload = await this.request({
      case: "setNetworkConfig",
      value: create(SetNetworkConfigRequestSchema, {
        config: {
          apSsid: config.apSsid,
          apPassword: config.apPassword,
          apBand: config.apBand,
        },
      }),
    });
    if (payload.case !== "networkConfig") {
      throw new Error(`expected NetworkConfig, got ${String(payload.case)}`);
    }
    return networkFromProto(payload.value);
  }
}

function networkFromProto(c: ProtoNetworkConfig): NetworkConfig {
  const mode = c.mode as NetworkMode;
  const apBand = c.apBand === "5" ? "5" : "2.4";
  return {
    mode: mode === "client" ? "client" : "ap",
    apSsid: c.apSsid,
    apPassword: "",
    apBand: apBand as ApBand,
    apAddress: c.apAddress,
  };
}
