// Route helpers. The router is a hash router (`#/...`), so these paths work
// under the Vite dev server, any static file server and `tauri://`.

export type RobotSection =
  | "drive"
  | "autonomy"
  | "autonomy/memory"
  | "settings"
  | `diagnostics/${DiagnosticsTab}`;

export const DIAGNOSTICS_TABS = [
  "motors",
  "power-imu",
  "captures",
  "services",
  "cameras",
] as const;
export type DiagnosticsTab = (typeof DIAGNOSTICS_TABS)[number];

export function robotPath(host: string, section: RobotSection = "drive"): string {
  return `/r/${encodeURIComponent(host)}/${section}`;
}

export function setupPath(step: "connect" | "wifi" | "identity", host?: string): string {
  return host ? `/setup/${step}?host=${encodeURIComponent(host)}` : `/setup/${step}`;
}
