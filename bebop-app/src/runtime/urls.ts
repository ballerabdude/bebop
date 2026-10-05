/// Scheme-aware robot endpoints.
///
/// The app is served either over plain HTTP on the LAN or over HTTPS (a TLS
/// reverse proxy / Tailscale). When the page is HTTPS, links to the robot's
/// (still-cleartext) services must be HTTPS/WSS too, or the browser blocks
/// them as mixed content. Everything is keyed off `window.location.protocol`
/// so the same build works both ways with no configuration:
///
///   page http://…  -> ws://host:port,   http://host:port
///   page https://… -> wss://host:port,  https://host:port
///
/// `tauri://` / `capacitor://` (native shells) are treated as insecure here,
/// since the robot services are cleartext; native shells also expose the
/// microphone anyway.

export function pageIsSecure(): boolean {
  return typeof window !== "undefined" && window.location.protocol === "https:";
}

export function wsUrl(host: string, port: number, path = ""): string {
  const scheme = pageIsSecure() ? "wss" : "ws";
  return `${scheme}://${host}:${port}${norm(path)}`;
}

export function httpUrl(host: string, port: number, path = ""): string {
  const scheme = pageIsSecure() ? "https" : "http";
  return `${scheme}://${host}:${port}${norm(path)}`;
}

function norm(path: string): string {
  if (!path) return "";
  return path.startsWith("/") ? path : `/${path}`;
}
