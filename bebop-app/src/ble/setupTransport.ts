// SetupTransport — protobuf-over-WebSocket client for `bebop-agent`'s
// provisioning server (`jetson-agent/bebop-agent/src/server.rs`).
//
// Two ways the app reaches it:
//   * Over the LAN once the robot is on Wi-Fi (host = robot IP:9091).
//   * Directly over the robot's setup hotspot during first-time setup
//     (host = the hotspot gateway, e.g. 192.168.42.1:9091).
//
// The runtime/operator surface (motors, telemetry, video) is separate and
// lives on `bebop-linux` at :9090/:9092.

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
} from "../proto/bebop_pb";
import type { BebopTransport, WifiJoinResult } from "./transport";
import type {
  ApBand,
  DeviceInfo,
  NetworkConfig,
  NetworkMode,
  RobotConfig,
  WifiNetwork,
  WifiStatus,
} from "./types";

/// Default port for the agent's setup server. The firmware runtime server
/// (`bebop-linux`) runs on 9090.
export const DEFAULT_SETUP_PORT = 9091;

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

  constructor(
    private readonly host: string,
    private readonly port: number = DEFAULT_SETUP_PORT,
  ) {}

  // -------------------------------------------------------------- lifecycle

  connect(): Promise<void> {
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

  disconnect(): void {
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

  private requestRaw(payload: ClientPayload): Promise<AgentResponse> {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      return Promise.reject(new Error("setup connection is not open"));
    }
    const requestId = this.nextRequestId++;
    const msg = create(ClientRequestSchema, { requestId, payload });
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
