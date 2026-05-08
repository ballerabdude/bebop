import { useEffect, useState } from "react";

import { Banner, Button, Card, Field, Spinner } from "../components/ui";

const STORAGE_KEY = "bebop.connectByIp";
const DEFAULT_PORT = 9090;

interface StoredEndpoint {
  ip: string;
  port: number;
}

interface ConnectByIpProps {
  /** Called once we've successfully reached the runtime server. */
  onConnected: (ip: string, port: number) => void;
  /** Optional cancel — only present when BLE is available, so the user
   *  can fall back to the BLE setup wizard. */
  onCancel?: () => void;
  /** Pre-fill from auto-detected (e.g. via BLE WifiStatus) IP. */
  prefillIp?: string;
}

/** Manual entry point: type the robot's IP, hit Connect, jump to the
 *  motor bench. Useful when:
 *    - the browser doesn't support Web Bluetooth (Firefox / mobile Safari)
 *    - you've already paired before and just want to manage motors
 *    - you're running the operator app on a workstation that talks to a
 *      robot on the same LAN
 *
 *  We probe the runtime server's `GET /healthz` endpoint as a connection
 *  pre-flight: it's fast, doesn't speak protobuf, and surfaces clear
 *  errors (DNS / unreachable / wrong port) without leaving WS state hanging.
 */
export function ConnectByIpScreen({
  onConnected,
  onCancel,
  prefillIp,
}: ConnectByIpProps) {
  const [ip, setIp] = useState<string>(prefillIp ?? "");
  const [port, setPort] = useState<number>(DEFAULT_PORT);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Restore last-used endpoint on mount.
  useEffect(() => {
    if (prefillIp) return;
    try {
      const raw = window.localStorage.getItem(STORAGE_KEY);
      if (!raw) return;
      const parsed = JSON.parse(raw) as StoredEndpoint;
      if (typeof parsed.ip === "string") setIp(parsed.ip);
      if (typeof parsed.port === "number") setPort(parsed.port);
    } catch {
      /* ignore corrupt storage */
    }
  }, [prefillIp]);

  async function connect(e?: React.FormEvent) {
    e?.preventDefault();
    setError(null);
    const trimmed = ip.trim();
    if (!trimmed) {
      setError("Enter the robot's IP address.");
      return;
    }
    // `0.0.0.0` is a bind-side wildcard, not a client destination. If you
    // see it printed in `bebop-linux`'s "starting WS runtime server"
    // log line, you want `127.0.0.1` (same machine) or the LAN IP.
    if (trimmed === "0.0.0.0") {
      setError(
        "0.0.0.0 is the server's bind address, not a destination. " +
          "Use 127.0.0.1 (or localhost) if bebop-linux is on this machine, " +
          "otherwise use the robot's LAN IP.",
      );
      return;
    }
    setBusy(true);
    try {
      // Pre-flight via HTTP; fast and gives clear errors. AbortController
      // gives us a 4-second timeout which is more useful than the browser's
      // default minutes-long fetch timeout. We deliberately don't open a
      // probe WebSocket here: that would race with MotorBenchScreen's own
      // WS connect and (under React StrictMode dev double-mount) leave a
      // stale socket the firmware would later trip over when broadcasting
      // ModeChanged. /healthz is on the same listener as /ws — if the HTTP
      // route answers, the WS route is reachable too. Any genuine WS
      // failure (firmware proto mismatch, etc.) surfaces in the motor
      // screen's connect path via a clear error banner.
      const ctrl = new AbortController();
      const t = setTimeout(() => ctrl.abort(), 4_000);
      try {
        const res = await fetch(`http://${ip}:${port}/healthz`, {
          signal: ctrl.signal,
        });
        if (!res.ok) {
          throw new Error(`server replied ${res.status}`);
        }
      } finally {
        clearTimeout(t);
      }

      // Persist for next visit.
      try {
        window.localStorage.setItem(
          STORAGE_KEY,
          JSON.stringify({ ip: ip.trim(), port }),
        );
      } catch {
        /* localStorage may be disabled */
      }

      onConnected(ip.trim(), port);
    } catch (err) {
      const message =
        err instanceof Error
          ? err.name === "AbortError"
            ? "Connection timed out — check the IP, port, and that bebop-linux is running."
            : err.message
          : String(err);
      setError(message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <form
      onSubmit={connect}
      className="flex flex-col flex-1 justify-center items-stretch gap-5 max-w-md mx-auto w-full"
    >
      <div className="text-center mb-2">
        <div className="text-[40px] mb-2" aria-hidden>
          🛰️
        </div>
        <h1 className="text-xl font-semibold">Connect by IP</h1>
        <p className="text-sm text-text-dim mt-1.5 leading-relaxed">
          Skip Bluetooth setup. Enter the address of a robot already on
          your network.
        </p>
      </div>

      {error ? <Banner tone="error">{error}</Banner> : null}

      <Card>
        <div className="flex flex-col gap-3 py-2">
          <Field
            label="Robot IP or hostname"
            hint="Use the LAN IP (e.g. 192.168.1.42) or a hostname like bebop.local."
          >
            <input
              autoFocus
              // Default text keyboard on mobile so hostnames like
              // "bebop.local" are typeable; "url" hints the keyboard
              // toward `.` and `/` which is handy for both IPv4 dots
              // and DNS labels. We deliberately don't use
              // `inputMode="decimal"` here — it locks iOS / Android to
              // a numeric keypad with no letters.
              inputMode="url"
              type="text"
              autoComplete="off"
              autoCapitalize="none"
              autoCorrect="off"
              spellCheck={false}
              value={ip}
              onChange={(e) => setIp(e.target.value)}
              placeholder="192.168.1.42 or bebop.local"
              className="w-full bg-bg-elev-2 border border-border rounded-[var(--radius-card)] px-3 py-3 text-text outline-none focus:border-accent text-base"
            />
          </Field>
          <Field label="Port" hint="The runtime server defaults to 9090.">
            <input
              type="number"
              inputMode="numeric"
              min={1}
              max={65535}
              value={port}
              onChange={(e) =>
                setPort(parseInt(e.target.value || "0", 10) || DEFAULT_PORT)
              }
              className="w-full bg-bg-elev-2 border border-border rounded-[var(--radius-card)] px-3 py-3 text-text outline-none focus:border-accent text-base"
            />
          </Field>
        </div>
      </Card>

      <Button type="submit" loading={busy}>
        {busy ? "Connecting…" : "Connect"}
      </Button>

      {onCancel ? (
        <Button variant="ghost" type="button" onClick={onCancel} disabled={busy}>
          Use Bluetooth setup instead
        </Button>
      ) : null}

      {busy ? (
        <div className="flex items-center justify-center gap-2 text-text-dim text-xs">
          <Spinner />
          <span>
            Probing <code>{ip}:{port}</code>…
          </span>
        </div>
      ) : null}
    </form>
  );
}
