import type { BebopTransport } from "./transport";
import type {
  AppStatus,
  DeviceInfo,
  DiscoveredRobot,
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

const MAX_PAYLOAD = 128; // safe default below typical ATT MTU

/// Transport using the Web Bluetooth API (Chrome, Edge, Opera on desktop).
/// This lets us iterate on the wizard flow in a browser without Tauri.
export class WebBluetoothTransport implements BebopTransport {
  private device: BluetoothDevice | null = null;
  private server: BluetoothRemoteGATTServer | null = null;
  private requestChar: BluetoothRemoteGATTCharacteristic | null = null;
  private responseChar: BluetoothRemoteGATTCharacteristic | null = null;
  private statusChar: BluetoothRemoteGATTCharacteristic | null = null;
  private reassembler = new Reassembler();
  private nextRequestId = 1;
  private pending = new Map<
    number,
    { resolve: (value: Uint8Array) => void; reject: (err: Error) => void }
  >();

  async scan(_timeoutMs: number): Promise<DiscoveredRobot[]> {
    try {
      const device = await navigator.bluetooth.requestDevice({
        filters: [{ services: [SERVICE_UUID] }],
        optionalServices: [SERVICE_UUID],
      });
      // Web Bluetooth scan is a picker — we only get one device at a time.
      // Return it as a single-element list so the UI works the same way.
      return [
        {
          id: device.id,
          name: device.name ?? "Unknown Bebop",
          rssi: 0, // Web Bluetooth doesn't expose RSSI in requestDevice
        },
      ];
    } catch (e) {
      if (
        e instanceof DOMException &&
        (e.name === "NotFoundError" || e.name === "AbortError")
      ) {
        return []; // user cancelled
      }
      throw e;
    }
  }

  async connect(robotId: string): Promise<void> {
    // Re-request the device — Web Bluetooth requires a user gesture for
    // requestDevice, so scan() already did that. Here we just connect.
    // If we already have a cached device from scan, use it.
    if (this.device && this.device.id === robotId) {
      this.server = await this.device.gatt!.connect();
    } else {
      // Fallback: try to get the device from the chooser again
      const device = await navigator.bluetooth.requestDevice({
        filters: [{ services: [SERVICE_UUID] }],
        optionalServices: [SERVICE_UUID],
      });
      this.device = device;
      this.server = await device.gatt!.connect();
    }

    const service = await this.server.getPrimaryService(SERVICE_UUID);
    this.requestChar = await service.getCharacteristic(CHAR_REQUEST_UUID);
    this.responseChar = await service.getCharacteristic(CHAR_RESPONSE_UUID);
    this.statusChar = await service.getCharacteristic(CHAR_STATUS_UUID);

    // Listen for notifications on the response characteristic
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

    // Also listen on status
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
    const resp = await this.sendRequest({ getDeviceInfo: {} });
    return resp.deviceInfo as DeviceInfo;
  }

  async scanWifi(): Promise<WifiNetwork[]> {
    const resp = await this.sendRequest({ scanWifi: {} });
    const result = resp.wifiScanResult as
      | { networks?: WifiNetwork[] }
      | undefined;
    return result?.networks ?? [];
  }

  async setWifiCredentials(
    ssid: string,
    password: string,
    hidden: boolean,
  ): Promise<void> {
    await this.sendRequest({ setWifiCredentials: { ssid, password, hidden } });
  }

  async getWifiStatus(): Promise<WifiStatus> {
    const resp = await this.sendRequest({ getWifiStatus: {} });
    return resp.wifiStatus as WifiStatus;
  }

  async getRobotConfig(): Promise<RobotConfig> {
    const resp = await this.sendRequest({ getRobotConfig: {} });
    return resp.robotConfig as RobotConfig;
  }

  async setRobotConfig(config: RobotConfig): Promise<void> {
    await this.sendRequest({ setRobotConfig: { config } });
  }

  async getAppStatus(): Promise<AppStatus> {
    const resp = await this.sendRequest({ getAppStatus: {} });
    return resp.appStatus as AppStatus;
  }

  async controlApp(
    appName: string,
    command: "START" | "STOP" | "RESTART",
  ): Promise<void> {
    await this.sendRequest({ controlApp: { appName, command } });
  }

  async triggerOta(targetImage?: string): Promise<void> {
    await this.sendRequest({ triggerOta: { targetImage: targetImage ?? "" } });
  }

  async getOtaStatus(): Promise<OtaStatus> {
    const resp = await this.sendRequest({ getOtaStatus: {} });
    return resp.otaStatus as OtaStatus;
  }

  // ---- internal helpers ------------------------------------------------

  private async sendRequest(
    payload: Record<string, unknown>,
  ): Promise<Record<string, unknown>> {
    if (!this.requestChar) throw new Error("not connected");

    const requestId = this.nextRequestId++;
    const json = JSON.stringify({ requestId, ...payload });
    const encoded = new TextEncoder().encode(json);
    const frames = encodeFrames(encoded, MAX_PAYLOAD);

    // Write each frame. Web Bluetooth wants a plain ArrayBuffer/BufferSource,
    // so copy into a fresh one rather than passing the view directly.
    for (const frame of frames) {
      const buf = new ArrayBuffer(frame.byteLength);
      new Uint8Array(buf).set(frame);
      await this.requestChar.writeValueWithoutResponse(buf);
    }

    // Wait for the matching response notification
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        this.pending.delete(requestId);
        reject(new Error("request timed out"));
      }, 15_000);

      this.pending.set(requestId, {
        resolve: (data: Uint8Array) => {
          clearTimeout(timeout);
          const json = new TextDecoder().decode(data);
          resolve(JSON.parse(json));
        },
        reject: (err: Error) => {
          clearTimeout(timeout);
          reject(err);
        },
      });
    });
  }

  private handleNotification(frame: Uint8Array): void {
    try {
      const complete = this.reassembler.push(frame);
      if (!complete) return;

      const json = new TextDecoder().decode(complete);
      const msg = JSON.parse(json);
      const requestId = msg.requestId as number;
      const pending = this.pending.get(requestId);
      if (pending) {
        this.pending.delete(requestId);
        pending.resolve(complete);
      }
    } catch {
      // ignore malformed frames
    }
  }
}
