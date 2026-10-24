// Settings › Network: Wi-Fi status and the Hosted Network (hotspot) editor,
// served by bebop-agent (:9091). The network mode itself is owned by the
// physical button and is read-only here.

import { useCallback, useEffect, useMemo, useState } from "react";
import { Link } from "react-router";

import {
  Banner,
  Button,
  Card,
  CardHeader,
  Field,
  INPUT_CLASS,
  SegmentedControl,
  StatusPill,
} from "../../components/ui";
import { setupPath } from "../../app/paths";
import { useRobot } from "../../robot/session";
import { SetupTransport } from "../../setup";
import type { ApBand, DeviceInfo, NetworkConfig, WifiStatus } from "../../setup";

const POLL_MS = 5_000;

export function NetworkSettings() {
  const { host } = useRobot();
  const transport = useMemo(() => new SetupTransport(host), [host]);
  useEffect(() => () => transport.disconnect(), [transport]);

  const [info, setInfo] = useState<DeviceInfo | null>(null);
  const [wifi, setWifi] = useState<WifiStatus | null>(null);
  const [net, setNet] = useState<NetworkConfig | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [apSsid, setApSsid] = useState("");
  const [apPassword, setApPassword] = useState("");
  const [apBand, setApBand] = useState<ApBand>("2.4");

  const refresh = useCallback(async () => {
    try {
      const [w, n] = await Promise.all([transport.getWifiStatus(), transport.getNetworkConfig()]);
      setWifi(w);
      setNet(n);
      setApSsid((prev) => (prev === "" ? n.apSsid : prev));
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [transport]);

  useEffect(() => {
    let cancelled = false;
    void transport.getDeviceInfo().then(
      (i) => {
        if (!cancelled) setInfo(i);
      },
      () => {},
    );
    void refresh();
    const id = window.setInterval(() => void refresh(), POLL_MS);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [transport, refresh]);

  // Seed the band from the robot once it is known.
  const robotBand = net?.apBand;
  useEffect(() => {
    if (robotBand) setApBand(robotBand);
  }, [robotBand]);

  async function saveHosted() {
    setError(null);
    setSaving(true);
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
      setSaving(false);
    }
  }

  const hosting = net?.mode === "ap";

  return (
    <Card className="flex flex-col gap-4">
      <CardHeader
        eyebrow="Network"
        title={info ? `${info.model} · agent v${info.agentVersion}` : "Robot network"}
        actions={
          <Link
            to={setupPath("wifi", host)}
            className="inline-flex h-control-sm items-center rounded-card border border-border px-3 text-label font-semibold text-text hover:bg-bg-elev-2 pointer-coarse:h-touch"
          >
            Change Wi-Fi
          </Link>
        }
      />
      {error ? (
        <Banner tone="error" title="Setup agent unreachable">
          {error}
        </Banner>
      ) : null}

      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        <div className="rounded-card border border-border bg-bg-elev-2/40 px-3 py-2.5">
          <div className="text-caption uppercase tracking-wider text-text-dim">Wi-Fi</div>
          <div className="mt-0.5 flex items-center gap-2 font-semibold">
            {wifi?.connected ? wifi.ssid : "Not connected"}
            <StatusPill tone={wifi?.connected ? "ok" : "dim"} dot>
              {wifi?.connected ? "joined" : "off"}
            </StatusPill>
          </div>
          {wifi?.connected && wifi.ipAddress ? (
            <div className="mt-0.5 font-mono text-body-sm text-text-dim">{wifi.ipAddress}</div>
          ) : null}
        </div>
        <div className="rounded-card border border-border bg-bg-elev-2/40 px-3 py-2.5">
          <div className="text-caption uppercase tracking-wider text-text-dim">Mode</div>
          <div className="mt-0.5 font-semibold">
            {net ? (hosting ? "Hosted Network" : "Known Network") : "—"}
          </div>
          <div className="mt-0.5 text-label text-text-dim">
            Hold the robot's mode button for 5 s to switch.
          </div>
        </div>
      </div>

      <div className="flex flex-col gap-3">
        <div className="text-caption font-semibold uppercase tracking-wider text-text-dim">
          Hosted Network
        </div>
        <Field label="Network name (SSID)">
          <input
            className={INPUT_CLASS}
            autoComplete="off"
            autoCapitalize="off"
            autoCorrect="off"
            spellCheck={false}
            value={apSsid}
            onChange={(e) => setApSsid(e.currentTarget.value)}
            placeholder="Bebop-XXXX"
          />
        </Field>
        <Field label="Password" hint="Leave blank to keep the current password (min 8 characters).">
          <input
            className={INPUT_CLASS}
            type="password"
            autoComplete="off"
            value={apPassword}
            onChange={(e) => setApPassword(e.currentTarget.value)}
            placeholder="••••••••"
          />
        </Field>
        <Field label="Band">
          <SegmentedControl<ApBand>
            label="Hotspot band"
            value={apBand}
            onChange={setApBand}
            options={[
              { value: "2.4", label: "2.4 GHz" },
              { value: "5", label: "5 GHz" },
            ]}
            className="w-full sm:w-auto"
          />
        </Field>
        {hosting ? (
          <p className="text-label leading-relaxed text-text-dim">
            Saving while hosting briefly drops connected devices; reconnect with
            the new credentials.
          </p>
        ) : null}
        <Button
          variant="secondary"
          onClick={() => void saveHosted()}
          loading={saving}
          disabled={
            !net || apSsid.trim().length === 0 || (apPassword !== "" && apPassword.length < 8)
          }
          className="self-start"
        >
          Save Hosted Network
        </Button>
      </div>
    </Card>
  );
}
