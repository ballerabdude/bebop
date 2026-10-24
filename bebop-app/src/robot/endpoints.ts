// Single source of truth for every robot service the app talks to.
//
// Every port and URL in the app comes from here; screens never spell a port
// or a scheme. Scheme selection (ws/wss, http/https) follows the page, see
// `runtime/urls.ts`, so the same build works over plain HTTP on the LAN and
// behind a TLS proxy.
//
//   9090 runtime  bebop-linux firmware: WS /ws (protobuf), /healthz, /captures*
//   9091 setup    bebop-agent provisioning: WS /ws (protobuf), /healthz
//   9092 video    bebop-vision videoserver: WHEP POST /whep, DELETE <Location>
//   9093 voice    bebop-voice: WS /voice, HTTP /healthz, /config
//   9094 models   bebop-models supervisor: HTTP /models, /healthz
//   9095 places   place memory (inside bebop-vision): HTTP /places, /whereami
//   9096 slam     live SLAM structure (only with --live-slam): HTTP /structure

import { httpUrl, wsUrl } from "../runtime/urls.ts";

export const PORTS = {
  runtime: 9090,
  setup: 9091,
  video: 9092,
  voice: 9093,
  models: 9094,
  places: 9095,
  slam: 9096,
} as const;

export type ServiceName = keyof typeof PORTS;

/// Camera streams served by the videoserver (WHEP `?stream=`).
export const VIDEO_STREAMS: { id: string; label: string }[] = [
  { id: "color_near", label: "Color" },
  { id: "depth_near", label: "Depth" },
  { id: "color_far", label: "Far" },
  { id: "depth_far", label: "Far depth" },
];

/// The robot's address while it hosts its own network (AP mode).
export const HOTSPOT_HOST = "192.168.42.1";

export interface RobotEndpoints {
  host: string;
  /// Firmware runtime WebSocket (protobuf).
  runtimeWs: string;
  /// Provisioning WebSocket on bebop-agent (protobuf).
  setupWs: string;
  /// WHEP offer endpoint for one camera stream.
  whep: (stream: string) => string;
  /// Absolute URL of a WHEP session resource from the POST's `Location`
  /// header (relative or absolute), for the teardown DELETE.
  whepResource: (location: string) => string;
  /// Voice WebSocket; `query` is appended verbatim (without the leading `?`).
  voiceWs: (query?: string) => string;
  voiceHttp: (path: string) => string;
  models: (path?: string) => string;
  places: (path?: string) => string;
  slam: (path?: string) => string;
  /// Firmware capture listing and downloads (`/captures`, `/captures/dl/..`).
  captures: (path?: string) => string;
  /// `/healthz` of one service.
  healthz: (service: ServiceName) => string;
}

function join(base: string, path: string): string {
  if (!path) return base;
  return path.startsWith("/") ? `${base}${path}` : `${base}/${path}`;
}

/// Strip a scheme, path, or port the user may have pasted into a host field.
export function normalizeHost(raw: string): string {
  let h = raw.trim();
  h = h.replace(/^[a-z]+:\/\//i, "");
  h = h.replace(/\/.*$/, "");
  // Bracketed IPv6 keeps its colons; otherwise drop a trailing :port.
  if (!h.startsWith("[")) h = h.replace(/:\d+$/, "");
  return h.toLowerCase();
}

export function endpoints(host: string): RobotEndpoints {
  const whepBase = httpUrl(host, PORTS.video, "/whep");
  return {
    host,
    runtimeWs: wsUrl(host, PORTS.runtime, "/ws"),
    setupWs: wsUrl(host, PORTS.setup, "/ws"),
    whep: (stream) => `${whepBase}?stream=${encodeURIComponent(stream)}`,
    whepResource: (location) => {
      try {
        return new URL(location, whepBase).toString();
      } catch {
        return join(httpUrl(host, PORTS.video), location);
      }
    },
    voiceWs: (query) =>
      wsUrl(host, PORTS.voice, "/voice") + (query ? `?${query}` : ""),
    voiceHttp: (path) => httpUrl(host, PORTS.voice, path),
    models: (path = "") => join(httpUrl(host, PORTS.models), path),
    places: (path = "") => join(httpUrl(host, PORTS.places), path),
    slam: (path = "") => join(httpUrl(host, PORTS.slam), path),
    captures: (path = "") =>
      join(httpUrl(host, PORTS.runtime, "/captures"), path),
    healthz: (service) => httpUrl(host, PORTS[service], "/healthz"),
  };
}
