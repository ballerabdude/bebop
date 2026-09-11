import { useCallback, useEffect, useState } from "react";

import type {
  BebopTransport,
  DeviceInfo,
  NetworkConfig,
  NetworkMode,
  WifiStatus,
} from "../ble";
import { Banner, Button, Card, Spinner } from "../components/ui";

interface DashboardProps {
  transport: BebopTransport;
  /** Reports the robot's LAN IP as soon as it's known, so the operator
   *  screens can reach the runtime server. */
  onIp: (ip: string) => void;
  onReconfigure: () => void;
  onDisconnect: () => void;
  onOpenMotors: () => void;
  onOpenTeleop: () => void;
}

const MODE_OPTIONS: { id: NetworkMode; label: string; hint: string }[] = [
  { id: "auto", label: "Auto", hint: "Join a known network; hotspot fallback." },
  { id: "client", label: "Client", hint: "Wi-Fi client only; never host a hotspot." },
  { id: "ap", label: "Hotspot", hint: "Always host the setup hotspot." },
];

/// Live dashboard shown after setup. Stays connected to the provisioning
/// server so the user can monitor Wi-Fi and change the network mode.
export function DashboardScreen({
  transport,
  onIp,
  onReconfigure,
  onDisconnect,
  onOpenMotors,
  onOpenTeleop,
}: DashboardProps) {
  const [info, setInfo] = useState<DeviceInfo | null>(null);
  const [wifi, setWifi] = useState<WifiStatus | null>(null);
  const [net, setNet] = useState<NetworkConfig | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busyMode, setBusyMode] = useState<NetworkMode | null>(null);

  const refresh = useCallback(async () => {
    try {
      const [w, n] = await Promise.all([
        transport.getWifiStatus(),
        transport.getNetworkConfig(),
      ]);
      setWifi(w);
      setNet(n);
      if (w.ipAddress) onIp(w.ipAddress);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [transport, onIp]);

  useEffect(() => {
    void (async () => {
      try {
        setInfo(await transport.getDeviceInfo());
      } catch {
        /* device info optional */
      }
      await refresh();
    })();
    const id = setInterval(refresh, 5_000);
    return () => clearInterval(id);
  }, [refresh]);

  async function chooseMode(mode: NetworkMode) {
    setError(null);
    setBusyMode(mode);
    try {
      setNet(await transport.setNetworkConfig({ mode, apSsid: "", apAddress: "" }));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusyMode(null);
    }
  }

  const reachable = wifi?.connected && wifi.ipAddress;

  return (
    <div className="flex flex-col flex-1 gap-4">
      <div className="text-center mt-3">
        <div
          className="w-14 h-14 mx-auto mb-3 rounded-full bg-success/15 text-success flex items-center justify-center text-[28px] font-bold"
          aria-hidden
        >
          ✓
        </div>
        <h2 className="text-2xl font-bold">Robot is online</h2>
        {info ? (
          <p className="text-text-dim text-sm mt-1">
            {info.model} · agent v{info.agentVersion}
          </p>
        ) : null}
      </div>

      {error ? <Banner tone="error">{error}</Banner> : null}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Card>
          <div className="flex items-center justify-between py-1">
            <div>
              <div className="text-xs text-text-dim uppercase tracking-wider mb-1">
                Wi-Fi
              </div>
              <div className="font-semibold">
                {wifi?.connected ? wifi.ssid : "Not connected"}
              </div>
              {wifi?.connected && wifi.ipAddress ? (
                <div className="text-text-dim text-[13px] mt-0.5 font-mono">
                  {wifi.ipAddress}
                </div>
              ) : null}
            </div>
            <div
              className={`w-2.5 h-2.5 rounded-full ${
                wifi?.connected ? "bg-success" : "bg-text-dim/40"
              }`}
            />
          </div>
        </Card>

        <Card>
          <div className="py-1">
            <div className="text-xs text-text-dim uppercase tracking-wider mb-1">
              Device
            </div>
            {info ? (
              <div className="text-[13px] text-text-dim">
                <div className="font-mono truncate" title={info.serialNumber}>
                  SN {info.serialNumber}
                </div>
                <div className="font-mono truncate">host: {info.hostname}</div>
              </div>
            ) : (
              <div className="flex items-center gap-2 text-text-dim text-sm">
                <Spinner />
                Loading…
              </div>
            )}
          </div>
        </Card>
      </div>

      <Card>
        <div className="py-2 flex flex-col gap-3">
          <div className="text-xs text-text-dim uppercase tracking-wider">
            Network mode
          </div>
          <div className="grid grid-cols-3 gap-2">
            {MODE_OPTIONS.map((o) => (
              <button
                key={o.id}
                type="button"
                title={o.hint}
                disabled={busyMode !== null}
                onClick={() => chooseMode(o.id)}
                className={`rounded-[var(--radius-card)] border px-3 py-2 text-sm font-semibold transition-colors disabled:opacity-50 ${
                  net?.mode === o.id
                    ? "border-accent bg-accent/15 text-accent"
                    : "border-border bg-bg-elev text-text-dim hover:text-text"
                }`}
              >
                {busyMode === o.id ? "…" : o.label}
              </button>
            ))}
          </div>
          {net && net.mode !== "client" ? (
            <div className="text-[12px] text-text-dim">
              Setup hotspot:{" "}
              <span className="font-mono text-text">{net.apSsid}</span>
              {net.apAddress ? (
                <>
                  {" "}
                  at <span className="font-mono text-text">{net.apAddress}</span>
                </>
              ) : null}
              . Password: <span className="font-mono text-text">bebopbebop</span>
            </div>
          ) : null}
        </div>
      </Card>

      <div className="mt-auto pt-4 flex flex-col gap-3">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
          <Button onClick={onOpenTeleop} disabled={!reachable}>
            Drive robot
          </Button>
          <Button variant="secondary" onClick={onOpenMotors} disabled={!reachable}>
            Open motor bench
          </Button>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
          <Button variant="secondary" onClick={onReconfigure}>
            Change Wi-Fi network
          </Button>
          <Button variant="ghost" onClick={onDisconnect}>
            Disconnect &amp; start over
          </Button>
        </div>
      </div>
    </div>
  );
}
