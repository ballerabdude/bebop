// Settings › Network: Wi-Fi status and the Hosted Network (hotspot) editor,
// served by bebop-agent (:9091). The network mode itself is owned by the
// physical button and is read-only here.

import { useEffect, useRef, useState } from "react";
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
import { useSetupTransport } from "../../setup";
import type { ApBand, DeviceInfo, NetworkConfig, WifiStatus } from "../../setup";

const POLL_MS = 5_000;
/// Consecutive failed polls before the "unreachable" banner shows, so one
/// dropped request (Wi-Fi hiccup, agent restart) doesn't flash an error.
const FAILURES_BEFORE_ERROR = 2;

/// "# R39 (release), REVISION: 2.1, GCID: …" (/etc/nv_tegra_release) ->
/// "L4T R39.2.1"; anything else passes through.
function l4tVersion(raw: string): string {
  const m = /R(\d+)\b.*?REVISION:\s*([\d.]+)/.exec(raw);
  return m ? `L4T R${m[1]}.${m[2]}` : raw;
}

export function NetworkSettings() {
  const { host } = useRobot();
  const transport = useSetupTransport(host);

  const [info, setInfo] = useState<DeviceInfo | null>(null);
  const [wifi, setWifi] = useState<WifiStatus | null>(null);
  const [net, setNet] = useState<NetworkConfig | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [apSsid, setApSsid] = useState("");
  const [apPassword, setApPassword] = useState("");
  const [apBand, setApBand] = useState<ApBand>("2.4");

  const infoRef = useRef<DeviceInfo | null>(null);
  infoRef.current = info;

  // Poll Wi-Fi + network config. Every result is dropped once this effect
  // run is cleaned up (StrictMode re-run, host change, unmount), so a request
  // aborted by the cleanup can never paint an error over a live connection.
  useEffect(() => {
    let alive = true;
    let failures = 0;
    let timer = 0;
    const poll = async () => {
      try {
        const [w, n] = await Promise.all([
          transport.getWifiStatus(),
          transport.getNetworkConfig(),
        ]);
        if (!alive) return;
        failures = 0;
        setWifi(w);
        setNet(n);
        setApSsid((prev) => (prev === "" ? n.apSsid : prev));
        setError(null);
        if (!infoRef.current) {
          const i = await transport.getDeviceInfo().catch(() => null);
          if (alive && i) setInfo(i);
        }
      } catch (e) {
        if (!alive) return;
        failures += 1;
        if (failures >= FAILURES_BEFORE_ERROR) {
          setError(e instanceof Error ? e.message : String(e));
        }
      } finally {
        if (alive) timer = window.setTimeout(() => void poll(), failures ? 2_000 : POLL_MS);
      }
    };
    void poll();
    return () => {
      alive = false;
      window.clearTimeout(timer);
    };
  }, [transport]);

  // Seed the band from the robot once it is known.
  const robotBand = net?.apBand;
  useEffect(() => {
    if (robotBand) setApBand(robotBand);
  }, [robotBand]);

  async function saveHosted() {
    setSaveError(null);
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
      setSaveError(e instanceof Error ? e.message : String(e));
    } finally {
      setSaving(false);
    }
  }

  const hosting = net?.mode === "ap";
  const loading = !net && !error;
  const passwordTooShort = apPassword !== "" && apPassword.length < 8;

  return (
    <div className="flex flex-col gap-4">
      {error ? (
        <Banner tone="error" title="Setup agent unreachable">
          {error}. bebop-agent serves Wi-Fi and hotspot settings on :9091; this
          page retries every few seconds.
        </Banner>
      ) : null}

      <Card className="flex flex-col gap-4">
        <CardHeader
          eyebrow="Connection"
          title={info ? `${info.hostname || "Robot"} · ${info.model}` : loading ? "Connecting to the setup agent…" : "Robot network"}
          description={info ? `bebop-agent v${info.agentVersion}${info.jetpackVersion ? ` · ${l4tVersion(info.jetpackVersion)}` : ""}` : undefined}
          actions={
            <Link
              to={setupPath("wifi", host)}
              className="inline-flex h-control-sm items-center rounded-card border border-border px-3 text-label font-semibold text-text hover:bg-bg-elev-2 pointer-coarse:h-touch"
            >
              Change Wi-Fi
            </Link>
          }
        />
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
          <div className="rounded-card border border-border bg-bg-elev-2/40 px-3 py-2.5">
            <div className="text-caption uppercase tracking-wider text-text-dim">Wi-Fi</div>
            <div className="mt-0.5 flex min-w-0 flex-wrap items-center gap-2 font-semibold">
              <span className="min-w-0 truncate">
                {wifi ? (wifi.connected ? wifi.ssid : "Not connected") : "—"}
              </span>
              {wifi ? (
                <StatusPill tone={wifi.connected ? "ok" : "dim"} dot>
                  {wifi.connected ? "joined" : "off"}
                </StatusPill>
              ) : null}
            </div>
            {wifi?.connected && wifi.ipAddress ? (
              <div className="mt-0.5 font-mono text-body-sm text-text-dim">
                {wifi.ipAddress}
                {wifi.signalDbm ? ` · ${wifi.signalDbm} dBm` : ""}
              </div>
            ) : null}
          </div>
          <div className="rounded-card border border-border bg-bg-elev-2/40 px-3 py-2.5">
            <div className="text-caption uppercase tracking-wider text-text-dim">Mode</div>
            <div className="mt-0.5 font-semibold">
              {net ? (hosting ? "Hosted Network (hotspot)" : "Known Network (Wi-Fi client)") : "—"}
            </div>
            <div className="mt-0.5 text-label leading-snug text-text-dim">
              Hold the robot&apos;s mode button for 5 s to switch. The app can&apos;t change it.
            </div>
          </div>
        </div>
      </Card>

      <Card className="flex flex-col gap-3">
        <CardHeader
          eyebrow="Hosted Network"
          title="Robot hotspot"
          description={
            net?.apAddress
              ? `While hosting, the robot is at ${net.apAddress.split(":")[0]}.`
              : "The network the robot hosts in Hosted mode (first-time setup, no Wi-Fi around)."
          }
        />
        <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
          <Field label="Network name (SSID)">
            <input
              className={INPUT_CLASS}
              autoComplete="off"
              autoCapitalize="off"
              autoCorrect="off"
              spellCheck={false}
              value={apSsid}
              disabled={!net}
              onChange={(e) => setApSsid(e.currentTarget.value)}
              placeholder="Bebop-XXXX"
            />
          </Field>
          <Field
            label="Password"
            hint={passwordTooShort ? "At least 8 characters." : "Leave blank to keep the current password."}
          >
            <input
              className={INPUT_CLASS}
              type="password"
              autoComplete="new-password"
              value={apPassword}
              disabled={!net}
              onChange={(e) => setApPassword(e.currentTarget.value)}
              placeholder="unchanged"
            />
          </Field>
        </div>
        <Field label="Band">
          <SegmentedControl<ApBand>
            label="Hotspot band"
            value={apBand}
            onChange={setApBand}
            options={[
              { value: "2.4", label: "2.4 GHz" },
              { value: "5", label: "5 GHz" },
            ]}
            className="w-full sm:w-auto sm:self-start"
          />
        </Field>
        {hosting ? (
          <Banner tone="warning">
            You are probably connected through this hotspot: saving restarts
            it, so reconnect with the new credentials.
          </Banner>
        ) : null}
        {saveError ? <Banner tone="error" title="Not saved">{saveError}</Banner> : null}
        <div className="flex flex-wrap items-center gap-3">
          <Button
            variant="secondary"
            onClick={() => void saveHosted()}
            loading={saving}
            disabled={!net || apSsid.trim().length === 0 || passwordTooShort}
            className="w-full sm:w-auto"
          >
            Save hotspot settings
          </Button>
        </div>
      </Card>
    </div>
  );
}
