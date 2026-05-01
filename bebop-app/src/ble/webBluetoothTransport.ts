import { create, fromBinary, toBinary } from "@bufbuild/protobuf";

import type { BebopTransport } from "./transport";
import type {
  AppState as AppStateLabel,
  AppStatus,
  DeviceInfo,
  DiscoveredRobot,
  OtaState as OtaStateLabel,
  OtaStatus,
  RobotConfig,
  WifiNetwork,
  WifiStatus,
} from "./types";
import {
  CHAR_REQUEST_UUID,
  CHAR_RESPONSE_UUID,
  CHAR_STATUS_UUID,
  SERVICE_UUID,
  encodeFrames,
  Reassembler,
} from "./protocol";
import {
  AgentResponseSchema,
  AppCommand,
  AppState,
  ClientRequestSchema,
  ControlAppRequestSchema,
  GetAppStatusRequestSchema,
  GetDeviceInfoRequestSchema,
  GetOtaStatusRequestSchema,
  GetRobotConfigRequestSchema,
  GetWifiStatusRequestSchema,
  OtaState,
  RobotConfigSchema,
  ResponseStatus,
  ScanWifiRequestSchema,
  SetAppImageRequestSchema,
  SetRobotConfigRequestSchema,
  SetWifiCredentialsRequestSchema,
  TriggerOtaRequestSchema,
  type AgentResponse,
  type ClientRequest,
} from "../proto/bebop_pb";

const MAX_PAYLOAD = 128; // safe default below typical ATT MTU

/// Transport using the Web Bluetooth API (Chrome, Edge, Opera on desktop).
/// Wire format: protobuf-encoded `ClientRequest` / `AgentResponse` from
/// `jetson-agent/bebop-proto/proto/bebop.proto`, framed by `./protocol.ts`.
/// The agent on the Jetson speaks the same encoding via `prost`.
export class WebBluetoothTransport implements BebopTransport {
  private device: BluetoothDevice | null = null;
  private server: BluetoothRemoteGATTServer | null = null;
  private requestChar: BluetoothRemoteGATTCharacteristic | null = null;
  private responseChar: BluetoothRemoteGATTCharacteristic | null = null;
  private statusChar: BluetoothRemoteGATTCharacteristic | null = null;
  private reassembler = new Reassembler();
  private nextRequestId = 1;
  // Cache BluetoothDevice handles by id so connect() can reuse the object
  // returned from a prior scan() (or getDevices()) instead of triggering
  // the OS pairing picker a second time.
  private knownDevices = new Map<string, BluetoothDevice>();
  // Web Bluetooth allows only one GATT operation in flight per device. If
  // two requests fire concurrently (e.g. dashboard's `Promise.all`), the
  // second write rejects with `NetworkError: GATT operation already in
  // progress`. Funnel all writes through this serial chain so each
  // request's frames are sent atomically. Responses are demultiplexed by
  // `request_id` via `pending`, so concurrent awaiters still work.
  private writeChain: Promise<void> = Promise.resolve();
  private pending = new Map<
    number,
    {
      resolve: (resp: AgentResponse) => void;
      reject: (err: Error) => void;
    }
  >();

  /// List previously-permitted devices via Web Bluetooth's
  /// `navigator.bluetooth.getDevices()`. This NEVER opens the OS picker,
  /// so it's safe to call on mount (and safe under React.StrictMode's
  /// double-invocation of effects).
  async scan(_timeoutMs: number): Promise<DiscoveredRobot[]> {
    if (typeof navigator.bluetooth.getDevices !== "function") {
      // Older browsers or contexts without permission backend support;
      // we can't enumerate without a picker, so return empty and let the
      // UI prompt the user to call pickDevice() instead.
      return [];
    }
    try {
      const known = await navigator.bluetooth.getDevices();
      return known.map((d) => {
        this.knownDevices.set(d.id, d);
        return {
          id: d.id,
          name: d.name ?? "Unknown Bebop",
          rssi: 0,
        };
      });
    } catch {
      return [];
    }
  }

  /// Open the OS pairing picker. Must be called from a user gesture
  /// (Web Bluetooth requires it, and otherwise StrictMode-style
  /// remounts would re-trigger the picker without the user asking).
  async pickDevice(): Promise<DiscoveredRobot | null> {
    try {
      const device = await navigator.bluetooth.requestDevice({
        filters: [{ services: [SERVICE_UUID] }],
        optionalServices: [SERVICE_UUID],
      });
      this.knownDevices.set(device.id, device);
      return {
        id: device.id,
        name: device.name ?? "Unknown Bebop",
        rssi: 0, // Web Bluetooth doesn't expose RSSI in requestDevice
      };
    } catch (e) {
      if (
        e instanceof DOMException &&
        (e.name === "NotFoundError" || e.name === "AbortError")
      ) {
        return null; // user cancelled
      }
      throw e;
    }
  }

  async connect(robotId: string): Promise<void> {
    let device =
      this.knownDevices.get(robotId) ??
      (this.device?.id === robotId ? this.device : null);

    if (!device) {
      // No cached handle for this id — last resort, prompt for it. This
      // path is only hit if scan() was bypassed (e.g. caller passed a
      // stale id from a previous session).
      device = await navigator.bluetooth.requestDevice({
        filters: [{ services: [SERVICE_UUID] }],
        optionalServices: [SERVICE_UUID],
      });
      this.knownDevices.set(device.id, device);
    }

    this.device = device;
    this.server = await device.gatt!.connect();

    const service = await this.server.getPrimaryService(SERVICE_UUID);
    this.requestChar = await service.getCharacteristic(CHAR_REQUEST_UUID);
    this.responseChar = await service.getCharacteristic(CHAR_RESPONSE_UUID);
    this.statusChar = await service.getCharacteristic(CHAR_STATUS_UUID);

    await this.responseChar.startNotifications();
    this.responseChar.addEventListener(
      "characteristicvaluechanged",
      (evt: Event) => {
        const target = evt.target as BluetoothRemoteGATTCharacteristic;
        if (!target.value) return;
        const frame = new Uint8Array(target.value.buffer);
        this.handleNotification(frame);
      },
    );

    await this.statusChar.startNotifications();
  }

  async disconnect(): Promise<void> {
    this.pending.forEach((p) => p.reject(new Error("disconnected")));
    this.pending.clear();
    if (this.server?.connected) {
      this.server.disconnect();
    }
    this.device = null;
    this.server = null;
    this.requestChar = null;
    this.responseChar = null;
    this.statusChar = null;
    this.reassembler = new Reassembler();
  }

  isConnected(): boolean {
    return this.server?.connected ?? false;
  }

  async getDeviceInfo(): Promise<DeviceInfo> {
    const resp = await this.sendRequest({
      case: "getDeviceInfo",
      value: create(GetDeviceInfoRequestSchema),
    });
    const info = expectPayload(resp, "deviceInfo");
    return {
      serialNumber: info.serialNumber,
      model: info.model,
      agentVersion: info.agentVersion,
      jetpackVersion: info.jetpackVersion,
      hostname: info.hostname,
    };
  }

  async scanWifi(): Promise<WifiNetwork[]> {
    // nmcli runs a fresh radio scan (`--rescan yes`) which routinely takes
    // 10-25s in busy environments. Give it more headroom than the default.
    const resp = await this.sendRequest(
      {
        case: "scanWifi",
        value: create(ScanWifiRequestSchema),
      },
      { timeoutMs: 30_000 },
    );
    const result = expectPayload(resp, "wifiScanResult");
    return result.networks.map((n) => ({
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
  ): Promise<void> {
    await this.sendRequest({
      case: "setWifiCredentials",
      value: create(SetWifiCredentialsRequestSchema, {
        ssid,
        password,
        hidden,
      }),
    });
  }

  async getWifiStatus(): Promise<WifiStatus> {
    const resp = await this.sendRequest({
      case: "getWifiStatus",
      value: create(GetWifiStatusRequestSchema),
    });
    const s = expectPayload(resp, "wifiStatus");
    return {
      connected: s.connected,
      ssid: s.ssid,
      ipAddress: s.ipAddress,
      signalDbm: s.signalDbm,
    };
  }

  async getRobotConfig(): Promise<RobotConfig> {
    const resp = await this.sendRequest({
      case: "getRobotConfig",
      value: create(GetRobotConfigRequestSchema),
    });
    const c = expectPayload(resp, "robotConfig");
    return {
      robotName: c.robotName,
      ownerId: c.ownerId,
      timezone: c.timezone,
      extra: { ...c.extra },
    };
  }

  async setRobotConfig(config: RobotConfig): Promise<void> {
    await this.sendRequest({
      case: "setRobotConfig",
      value: create(SetRobotConfigRequestSchema, {
        config: create(RobotConfigSchema, {
          robotName: config.robotName,
          ownerId: config.ownerId,
          timezone: config.timezone,
          extra: { ...config.extra },
        }),
      }),
    });
  }

  async getAppStatus(): Promise<AppStatus> {
    const resp = await this.sendRequest({
      case: "getAppStatus",
      value: create(GetAppStatusRequestSchema),
    });
    const s = expectPayload(resp, "appStatus");
    return {
      appName: s.appName,
      image: s.image,
      imageDigest: s.imageDigest,
      state: appStateLabel(s.state),
      containerId: s.containerId,
      // protobuf int64 → bigint; UI consumes as number (seconds since epoch fits)
      startedAtUnix: Number(s.startedAtUnix),
      restartCount: s.restartCount,
    };
  }

  async controlApp(
    appName: string,
    command: "START" | "STOP" | "RESTART",
  ): Promise<void> {
    await this.sendRequest({
      case: "controlApp",
      value: create(ControlAppRequestSchema, {
        appName,
        command: appCommandFromLabel(command),
      }),
    });
  }

  async setAppImage(image: string): Promise<AppStatus> {
    const resp = await this.sendRequest({
      case: "setAppImage",
      value: create(SetAppImageRequestSchema, { image }),
    });
    const s = expectPayload(resp, "appStatus");
    return {
      appName: s.appName,
      image: s.image,
      imageDigest: s.imageDigest,
      state: appStateLabel(s.state),
      containerId: s.containerId,
      startedAtUnix: Number(s.startedAtUnix),
      restartCount: s.restartCount,
    };
  }

  async triggerOta(targetImage?: string): Promise<void> {
    await this.sendRequest({
      case: "triggerOta",
      value: create(TriggerOtaRequestSchema, {
        targetImage: targetImage ?? "",
      }),
    });
  }

  async getOtaStatus(): Promise<OtaStatus> {
    const resp = await this.sendRequest({
      case: "getOtaStatus",
      value: create(GetOtaStatusRequestSchema),
    });
    const s = expectPayload(resp, "otaStatus");
    return {
      state: otaStateLabel(s.state),
      currentImage: s.currentImage,
      targetImage: s.targetImage,
      progressPercent: s.progressPercent,
      error: s.error,
    };
  }

  // ---- internal helpers ------------------------------------------------

  private async sendRequest(
    payload: ClientRequest["payload"],
    options: { timeoutMs?: number } = {},
  ): Promise<AgentResponse> {
    if (!this.requestChar) throw new Error("not connected");

    const requestId = this.nextRequestId++;
    const req = create(ClientRequestSchema, { requestId, payload });
    const encoded = toBinary(ClientRequestSchema, req);
    const frames = encodeFrames(encoded, MAX_PAYLOAD);

    // Web Bluetooth wants a plain ArrayBuffer/BufferSource, so copy into
    // a fresh one rather than passing the view directly. Run the whole
    // frame burst inside `writeChain` so concurrent sendRequest() callers
    // don't trip the "GATT operation already in progress" error.
    const requestChar = this.requestChar;
    const send = this.writeChain.then(async () => {
      for (const frame of frames) {
        const buf = new ArrayBuffer(frame.byteLength);
        new Uint8Array(buf).set(frame);
        await requestChar.writeValueWithoutResponse(buf);
      }
    });
    // The chain itself must never reject, otherwise every subsequent
    // request would inherit the failure.
    this.writeChain = send.catch(() => {});
    await send;

    const timeoutMs = options.timeoutMs ?? 15_000;
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        this.pending.delete(requestId);
        reject(new Error("request timed out"));
      }, timeoutMs);

      this.pending.set(requestId, {
        resolve: (resp) => {
          clearTimeout(timeout);
          if (resp.status === ResponseStatus.OK) {
            resolve(resp);
          } else {
            reject(
              new Error(
                resp.message ||
                  `agent returned status ${ResponseStatus[resp.status] ?? resp.status}`,
              ),
            );
          }
        },
        reject: (err) => {
          clearTimeout(timeout);
          reject(err);
        },
      });
    });
  }

  private handleNotification(frame: Uint8Array): void {
    let complete: Uint8Array | null;
    try {
      complete = this.reassembler.push(frame);
    } catch {
      this.reassembler = new Reassembler();
      return;
    }
    if (!complete) return;

    let resp: AgentResponse;
    try {
      resp = fromBinary(AgentResponseSchema, complete);
    } catch {
      // malformed frame — drop silently; the timeout will surface to the caller.
      return;
    }
    const pending = this.pending.get(resp.requestId);
    if (!pending) return; // request_id == 0 = unsolicited status push, ignore here
    this.pending.delete(resp.requestId);
    pending.resolve(resp);
  }
}

// ---- payload helpers ---------------------------------------------------

type PayloadOf<C extends NonNullable<AgentResponse["payload"]>["case"]> = Extract<
  NonNullable<AgentResponse["payload"]>,
  { case: C }
>["value"];

/// Pull the expected `oneof` arm out of an `AgentResponse`, throwing a
/// readable error if the agent answered with a different payload (or none
/// at all). `sendRequest` already filters out non-OK responses, so reaching
/// this helper means status was OK; this just defends against a schema
/// mismatch between agent and app.
function expectPayload<C extends NonNullable<AgentResponse["payload"]>["case"]>(
  resp: AgentResponse,
  expected: C,
): PayloadOf<C> {
  if (resp.payload?.case !== expected) {
    throw new Error(
      `agent returned unexpected payload: ${resp.payload?.case ?? "<none>"} (expected ${expected})`,
    );
  }
  return resp.payload.value as PayloadOf<C>;
}

function appStateLabel(s: AppState): AppStateLabel {
  switch (s) {
    case AppState.STOPPED:
      return "STOPPED";
    case AppState.STARTING:
      return "STARTING";
    case AppState.RUNNING:
      return "RUNNING";
    case AppState.CRASHED:
      return "CRASHED";
    case AppState.UPDATING:
      return "UPDATING";
    case AppState.UNSPECIFIED:
    default:
      return "UNSPECIFIED";
  }
}

function otaStateLabel(s: OtaState): OtaStateLabel {
  switch (s) {
    case OtaState.IDLE:
      return "IDLE";
    case OtaState.CHECKING:
      return "CHECKING";
    case OtaState.DOWNLOADING:
      return "DOWNLOADING";
    case OtaState.APPLYING:
      return "APPLYING";
    case OtaState.SUCCESS:
      return "SUCCESS";
    case OtaState.FAILED:
      return "FAILED";
    case OtaState.UNSPECIFIED:
    default:
      return "UNSPECIFIED";
  }
}

function appCommandFromLabel(c: "START" | "STOP" | "RESTART"): AppCommand {
  switch (c) {
    case "START":
      return AppCommand.START;
    case "STOP":
      return AppCommand.STOP;
    case "RESTART":
      return AppCommand.RESTART;
  }
}
