import { TauriTransport } from "./tauriTransport";
import { WebBluetoothTransport } from "./webBluetoothTransport";
import type { BebopTransport } from "./transport";

export type { BebopTransport } from "./transport";
export * from "./types";

function isTauri(): boolean {
  return typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
}

function hasWebBluetooth(): boolean {
  return typeof navigator !== "undefined" && "bluetooth" in navigator;
}

/// Whether the current environment can talk to a robot at all.
export function bluetoothSupported(): boolean {
  return isTauri() || hasWebBluetooth();
}

/// Pick the best available transport for the current environment.
///
/// Inside the Tauri shell we delegate to the native BLE bridge; in a
/// plain browser we fall back to the Web Bluetooth API. There is no
/// user-facing toggle — the runtime picks whatever is supported.
export function createTransport(): BebopTransport {
  if (isTauri()) return new TauriTransport();
  if (hasWebBluetooth()) return new WebBluetoothTransport();
  throw new Error(
    "Bluetooth is not available in this environment. Use the Bebop desktop/mobile app, or open this page in a Web Bluetooth-capable browser (Chrome, Edge).",
  );
}
