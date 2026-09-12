import { useCallback, useEffect, useState } from "react";

import type {
  ApBand,
  BebopTransport,
  DeviceInfo,
  NetworkConfig,
  WifiStatus,
} from "../ble";
import { Banner, Button, Card, Field } from "../components/ui";

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

/// Live dashboard shown after setup. Stays connected to the provisioning
/// server so the user can monitor Wi-Fi and edit the Hosted Network.
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
  const [savingAp, setSavingAp] = useState(false);

  // Hosted Network edit draft.
  const [apSsid, setApSsid] = useState("");
  const [apPassword, setApPassword] = useState("");
  const [apBand, setApBand] = useState<ApBand>("2.4");

  const refresh = useCallback(async () => {
    try {
      const [w, n] = await Promise.all([
        transport.getWifiStatus(),
        transport.getNetworkConfig(),
      ]);
      setWifi(w);
      setNet(n);
      setApSsid((prev) => (prev === "" ? n.apSsid : prev));
      setApBand(n.apBand);
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

  async function saveHosted() {
    setError(null);
    setSavingAp(true);
    try {
      const next = await transport.setNetworkConfig({
        mode: net?.mode ?? "ap",
        apSsid: apSsid.trim(),
        apPassword,
        apBand,
        apAddress: net?.apAddress ?? "",
      });
      setNet(next);
      setApPassword("");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSavingAp(false);
    }
  }

  const hosting = net?.mode === "ap";
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
              Mode
            </div>
            <div className="font-semibold">
              {net ? (hosting ? "Hosted Network" : "Known Network") : "—"}
            </div>
            <div className="text-text-dim text-[12px] mt-0.5">
              Long-press the mode button to switch.
            </div>
          </div>
        </Card>
      </div>

      <Card>
        <div className="py-2 flex flex-col gap-3">
          <div className="text-xs text-text-dim uppercase tracking-wider">
            Hosted Network
          </div>
          <Field label="Network name (SSID)">
            <input
              type="text"
              autoComplete="off"
              autoCapitalize="off"
              autoCorrect="off"
              spellCheck={false}
              value={apSsid}
              onChange={(e) => setApSsid(e.currentTarget.value)}
              placeholder="Bebop-XXXX"
              className="w-full bg-bg-elev-2 border border-border rounded-[var(--radius-card)] px-3 py-2.5 text-text outline-none focus:border-accent"
            />
          </Field>
          <Field
            label="Password"
            hint="Leave blank to keep the current password (min 8 characters)."
          >
            <input
              type="password"
              autoComplete="off"
              value={apPassword}
              onChange={(e) => setApPassword(e.currentTarget.value)}
              placeholder="••••••••"
              className="w-full bg-bg-elev-2 border border-border rounded-[var(--radius-card)] px-3 py-2.5 text-text outline-none focus:border-accent"
            />
          </Field>
          <Field label="Band">
            <div className="flex gap-2">
              {(["2.4", "5"] as ApBand[]).map((b) => (
                <button
                  key={b}
                  type="button"
                  onClick={() => setApBand(b)}
                  className={`flex-1 rounded-[var(--radius-card)] border px-3 py-2 text-sm font-semibold transition-colors ${
                    apBand === b
                      ? "border-accent bg-accent/15 text-accent"
                      : "border-border bg-bg-elev text-text-dim hover:text-text"
                  }`}
                >
                  {b} GHz
                </button>
              ))}
            </div>
          </Field>
          {hosting ? (
            <p className="text-[12px] text-text-dim leading-relaxed">
              Saving while hosting briefly drops connected devices; reconnect
              with the new credentials.
            </p>
          ) : null}
          <Button
            variant="secondary"
            onClick={saveHosted}
            loading={savingAp}
            disabled={apSsid.trim().length === 0 || (apPassword !== "" && apPassword.length < 8)}
          >
            Save Hosted Network
          </Button>
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
