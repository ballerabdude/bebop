import { useEffect, useState } from "react";

import { Banner, Button, Card, Field, Spinner } from "../components/ui";

const STORAGE_KEY = "bebop.connectByIp";

interface StoredEndpoint {
  ip: string;
  port: number;
}

interface ConnectByIpProps {
  /** Called once we've successfully reached the server's `/healthz`. */
  onConnected: (ip: string, port: number) => void;
  /** Optional cancel. */
  onCancel?: () => void;
  /** Pre-fill the address field (e.g. a previously detected IP). */
  prefillIp?: string;
  /** Heading shown above the form. */
  heading?: string;
  /** One-line explanation under the heading. */
  description?: string;
  /** Label for the submit button (default "Connect"). */
  submitLabel?: string;
  /** Port to use when there's nothing stored. */
  defaultPort?: number;
  /** Address to use when there's nothing stored. */
  defaultIp?: string;
}

/// Manual connection entry point. Probes `GET /healthz` as a pre-flight so
/// DNS / unreachable / wrong-port errors surface clearly before we open a
/// WebSocket. Used both for the provisioning server (bebop-agent, :9091)
/// and the runtime controls server (bebop-linux, :9090).
export function ConnectByIpScreen({
  onConnected,
  onCancel,
  prefillIp,
  heading = "Connect by IP",
  description = "Enter the address of a robot.",
  submitLabel = "Connect",
  defaultPort = 9090,
  defaultIp = "",
}: ConnectByIpProps) {
  const [ip, setIp] = useState<string>(prefillIp ?? defaultIp);
  const [port, setPort] = useState<number>(defaultPort);
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
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [prefillIp]);

  async function connect(e?: React.FormEvent) {
    e?.preventDefault();
    setError(null);
    const trimmed = ip.trim();
    if (!trimmed) {
      setError("Enter the robot's IP address or hostname.");
      return;
    }
    if (trimmed === "0.0.0.0") {
      setError(
        "0.0.0.0 is a server bind address, not a destination. Use the " +
          "robot's address (e.g. 192.168.42.1 over its hotspot).",
      );
      return;
    }
    setBusy(true);
    try {
      const ctrl = new AbortController();
      const t = setTimeout(() => ctrl.abort(), 4_000);
      try {
        const res = await fetch(`http://${trimmed}:${port}/healthz`, {
          signal: ctrl.signal,
        });
        if (!res.ok) {
          throw new Error(`server replied ${res.status}`);
        }
      } finally {
        clearTimeout(t);
      }

      try {
        window.localStorage.setItem(
          STORAGE_KEY,
          JSON.stringify({ ip: trimmed, port }),
        );
      } catch {
        /* localStorage may be disabled */
      }

      onConnected(trimmed, port);
    } catch (err) {
      const message =
        err instanceof Error
          ? err.name === "AbortError"
            ? "Connection timed out — check the address, port, and that the robot is powered on."
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
        <h1 className="text-xl font-semibold">{heading}</h1>
        <p className="text-sm text-text-dim mt-1.5 leading-relaxed">
          {description}
        </p>
      </div>

      {error ? <Banner tone="error">{error}</Banner> : null}

      <Card>
        <div className="flex flex-col gap-3 py-2">
          <Field
            label="Robot address"
            hint="Use the hotspot gateway (192.168.42.1) or the robot's LAN IP."
          >
            <input
              autoFocus
              inputMode="url"
              type="text"
              autoComplete="off"
              autoCapitalize="none"
              autoCorrect="off"
              spellCheck={false}
              value={ip}
              onChange={(e) => setIp(e.target.value)}
              placeholder="192.168.42.1 or bebop.local"
              className="w-full bg-bg-elev-2 border border-border rounded-[var(--radius-card)] px-3 py-3 text-text outline-none focus:border-accent text-base"
            />
          </Field>
          <Field
            label="Port"
            hint={
              defaultPort === 9091
                ? "The setup server defaults to 9091."
                : "The runtime server defaults to 9090."
            }
          >
            <input
              type="number"
              inputMode="numeric"
              min={1}
              max={65535}
              value={port}
              onChange={(e) =>
                setPort(parseInt(e.target.value || "0", 10) || defaultPort)
              }
              className="w-full bg-bg-elev-2 border border-border rounded-[var(--radius-card)] px-3 py-3 text-text outline-none focus:border-accent text-base"
            />
          </Field>
        </div>
      </Card>

      <Button type="submit" loading={busy}>
        {busy ? "Connecting…" : submitLabel}
      </Button>

      {onCancel ? (
        <Button variant="ghost" type="button" onClick={onCancel} disabled={busy}>
          Back
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
