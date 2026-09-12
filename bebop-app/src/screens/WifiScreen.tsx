import { useEffect, useState } from "react";

import type { BebopTransport, WifiNetwork, WifiStatus } from "../ble";
import { Banner, Button, Card, Field, Spinner } from "../components/ui";

export function WifiScreen({
  transport,
  onDone,
}: {
  transport: BebopTransport;
  onDone: (status: WifiStatus) => void;
}) {
  const [currentStatus, setCurrentStatus] = useState<WifiStatus | null>(null);
  const [networks, setNetworks] = useState<WifiNetwork[]>([]);
  const [selected, setSelected] = useState<WifiNetwork | null>(null);
  const [manual, setManual] = useState(false);
  const [manualSsid, setManualSsid] = useState("");
  const [password, setPassword] = useState("");
  const [scanning, setScanning] = useState(false);
  const [joining, setJoining] = useState(false);
  const [loadingStatus, setLoadingStatus] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [scanNote, setScanNote] = useState<string | null>(null);
  // Set when credentials were saved but not applied (Hosted Network mode).
  const [savedMessage, setSavedMessage] = useState<string | null>(null);

  async function fetchStatus() {
    try {
      setCurrentStatus(await transport.getWifiStatus());
    } catch {
      /* pre-connect agents may not answer; ignore */
    } finally {
      setLoadingStatus(false);
    }
  }

  async function scan() {
    setError(null);
    setScanNote(null);
    setScanning(true);
    try {
      const list = await transport.scanWifi();
      const byKey = new Map<string, WifiNetwork>();
      for (const n of list) {
        if (!n.ssid) continue;
        const key = `${n.ssid}\x00${n.security}`;
        const prev = byKey.get(key);
        if (!prev || n.signalDbm > prev.signalDbm) {
          byKey.set(key, { ...n, saved: n.saved || prev?.saved || false });
        } else if (n.saved && !prev.saved) {
          byKey.set(key, { ...prev, saved: true });
        }
      }
      const deduped = Array.from(byKey.values()).sort(
        (a, b) => b.signalDbm - a.signalDbm,
      );
      setNetworks(deduped);
      if (deduped.length === 0) {
        setScanNote(
          "No networks found. While the setup hotspot is active the robot may " +
            "be unable to scan — enter your network name manually below.",
        );
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setScanNote("Scan failed. Enter your network name manually below.");
    } finally {
      setScanning(false);
    }
  }

  useEffect(() => {
    void Promise.all([fetchStatus(), scan()]);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function join(ssid: string) {
    setError(null);
    setJoining(true);
    try {
      const { status, message } = await transport.setWifiCredentials(
        ssid,
        password,
        false,
      );
      if (status.connected) {
        onDone(status);
        return;
      }
      if (message) {
        // Hosted Network mode: saved, not applied (the hotspot stays up).
        setSavedMessage(message);
        setJoining(false);
        return;
      }
      setError(
        `Could not join ${ssid}. Check the password and try again.`,
      );
      setJoining(false);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setJoining(false);
    }
  }

  // Saved-but-not-applied panel (Hosted Network provisioning).
  if (savedMessage) {
    return (
      <div className="flex flex-col flex-1 gap-4">
        <h2 className="text-2xl font-bold mt-2">Wi-Fi saved</h2>
        <Banner tone="info">{savedMessage}</Banner>
        <p className="text-text-dim leading-relaxed">
          The robot keeps hosting its network until you long-press the mode
          button. It will join the saved network when it switches to Known
          Network.
        </p>
        <div className="mt-auto pt-4">
          <Button
            onClick={() =>
              onDone({ connected: false, ssid: "", ipAddress: "", signalDbm: 0 })
            }
          >
            Done
          </Button>
        </div>
      </div>
    );
  }

  if (selected) {
    const needsPassword = selected.security !== "OPEN";
    return (
      <div className="flex flex-col flex-1 gap-4">
        <h2 className="text-2xl font-bold mt-2">Join {selected.ssid}</h2>
        <p className="text-text-dim leading-relaxed">
          {joining
            ? "The robot is joining the network and will drop the setup hotspot. Your phone may briefly lose its connection."
            : needsPassword
              ? "Enter the Wi-Fi password. The robot will use this network going forward."
              : "This is an open network. Tap Join to connect."}
        </p>

        {error ? <Banner tone="error">{error}</Banner> : null}

        {needsPassword && !joining ? (
          <Field label="Password">
            <input
              type="password"
              autoFocus
              autoComplete="off"
              value={password}
              onChange={(e) => setPassword(e.currentTarget.value)}
              placeholder="••••••••"
              className="bg-bg-elev border border-border rounded-[var(--radius-card)] px-3.5 py-3 text-text outline-none focus:border-accent transition-colors duration-120"
            />
          </Field>
        ) : null}

        <div className="mt-auto pt-4 flex flex-row gap-3">
          <Button
            variant="secondary"
            onClick={() => {
              setSelected(null);
              setPassword("");
            }}
            disabled={joining}
            className="flex-1"
          >
            Back
          </Button>
          <Button
            onClick={() => join(selected.ssid)}
            loading={joining}
            disabled={needsPassword && password.length === 0}
            className="flex-1"
          >
            Join
          </Button>
        </div>
      </div>
    );
  }

  if (manual) {
    return (
      <div className="flex flex-col flex-1 gap-4">
        <h2 className="text-2xl font-bold mt-2">Enter network manually</h2>
        <p className="text-text-dim leading-relaxed">
          Type the exact network name (SSID) the robot should join.
        </p>

        {error ? <Banner tone="error">{error}</Banner> : null}

        <Field label="Network name (SSID)">
          <input
            autoFocus
            autoComplete="off"
            value={manualSsid}
            onChange={(e) => setManualSsid(e.currentTarget.value)}
            placeholder="MyHomeWiFi"
            className="bg-bg-elev border border-border rounded-[var(--radius-card)] px-3.5 py-3 text-text outline-none focus:border-accent"
          />
        </Field>
        <Field label="Password" hint="Leave blank for an open network.">
          <input
            type="password"
            autoComplete="off"
            value={password}
            onChange={(e) => setPassword(e.currentTarget.value)}
            placeholder="••••••••"
            className="bg-bg-elev border border-border rounded-[var(--radius-card)] px-3.5 py-3 text-text outline-none focus:border-accent"
          />
        </Field>

        <div className="mt-auto pt-4 flex flex-row gap-3">
          <Button
            variant="secondary"
            onClick={() => setManual(false)}
            disabled={joining}
            className="flex-1"
          >
            Back
          </Button>
          <Button
            onClick={() => join(manualSsid.trim())}
            loading={joining}
            disabled={manualSsid.trim().length === 0}
            className="flex-1"
          >
            Join
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col flex-1 gap-4">
      <h2 className="text-2xl font-bold mt-2">Choose a Wi-Fi network</h2>
      <p className="text-text-dim leading-relaxed">
        Your robot needs Wi-Fi to run its application.
      </p>

      {loadingStatus ? (
        <div className="flex justify-center py-4">
          <Spinner />
        </div>
      ) : currentStatus?.connected ? (
        <Card>
          <div className="flex items-center justify-between">
            <div>
              <div className="text-xs text-text-dim uppercase tracking-wider mb-0.5">
                Currently connected
              </div>
              <div className="font-semibold">{currentStatus.ssid}</div>
              <div className="text-text-dim text-[13px]">
                {currentStatus.ipAddress || "no IP"}
              </div>
            </div>
            <div
              className="w-2 h-2 rounded-full bg-success shrink-0"
              aria-label="connected"
            />
          </div>
        </Card>
      ) : null}

      {error ? <Banner tone="error">{error}</Banner> : null}
      {scanNote ? <Banner tone="info">{scanNote}</Banner> : null}

      <ul className="flex flex-col gap-2 list-none m-0 p-0">
        {networks.map((n) => {
          const isCurrent =
            currentStatus?.connected && n.ssid === currentStatus.ssid;
          return (
            <li
              key={`${n.ssid}\x00${n.security}`}
              className={`border rounded-[var(--radius-card)] overflow-hidden ${
                isCurrent ? "bg-accent/10 border-accent/40" : "bg-bg-elev border-border"
              }`}
            >
              <button
                className="flex w-full items-center justify-between px-4 py-3.5 bg-transparent border-0 text-left cursor-pointer hover:bg-bg-elev-2"
                onClick={() => {
                  if (isCurrent && currentStatus) {
                    onDone(currentStatus);
                    return;
                  }
                  setSelected(n);
                }}
              >
                <div>
                  <div className="font-semibold flex items-center gap-2">
                    {n.ssid}
                    {isCurrent ? (
                      <span className="text-[11px] font-normal text-accent bg-accent/15 px-1.5 py-0.5 rounded-full">
                        connected
                      </span>
                    ) : null}
                  </div>
                  <div className="text-text-dim text-[13px] mt-0.5">
                    {n.security} · {n.signalDbm} dBm
                  </div>
                </div>
                <span className="text-text-dim text-[22px] leading-none" aria-hidden>
                  ›
                </span>
              </button>
            </li>
          );
        })}
        {!scanning && networks.length === 0 ? (
          <li className="text-text-dim py-4 text-center text-sm">
            No networks found.
          </li>
        ) : null}
      </ul>

      <div className="mt-auto pt-4 flex flex-col gap-3">
        {currentStatus?.connected ? (
          <Button onClick={() => onDone(currentStatus)}>
            Continue with {currentStatus.ssid}
          </Button>
        ) : null}
        <Button variant="secondary" onClick={scan} loading={scanning}>
          {scanning ? "Scanning…" : "Rescan"}
        </Button>
        <Button variant="ghost" onClick={() => setManual(true)}>
          Enter network manually
        </Button>
      </div>
    </div>
  );
}
