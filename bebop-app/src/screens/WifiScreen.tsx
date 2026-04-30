import { useEffect, useState } from "react";

import type { BebopTransport, WifiNetwork, WifiStatus } from "../ble";
import { Banner, Button, Field } from "../components/ui";

export function WifiScreen({
  transport,
  onDone,
}: {
  transport: BebopTransport;
  onDone: (status: WifiStatus) => void;
}) {
  const [networks, setNetworks] = useState<WifiNetwork[]>([]);
  const [selected, setSelected] = useState<WifiNetwork | null>(null);
  const [password, setPassword] = useState("");
  const [scanning, setScanning] = useState(false);
  const [joining, setJoining] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function scan() {
    setError(null);
    setScanning(true);
    try {
      const list = await transport.scanWifi();
      list.sort((a, b) => b.signalDbm - a.signalDbm);
      setNetworks(list);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setScanning(false);
    }
  }

  useEffect(() => {
    void scan();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function join() {
    if (!selected) return;
    setError(null);
    setJoining(true);
    try {
      await transport.setWifiCredentials(selected.ssid, password, false);
      // Give the agent a moment, then read status.
      const status = await transport.getWifiStatus();
      if (!status.connected) {
        throw new Error(
          "Robot reported Wi-Fi not connected. Double-check the password.",
        );
      }
      onDone(status);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setJoining(false);
    }
  }

  if (selected) {
    const needsPassword = selected.security !== "OPEN";
    return (
      <div className="flex flex-col flex-1 gap-4">
        <h2 className="text-2xl font-bold mt-2">Join {selected.ssid}</h2>
        <p className="text-text-dim leading-relaxed">
          {needsPassword
            ? "Enter the Wi-Fi password. The robot will use this network going forward."
            : "This is an open network. Tap Join to connect."}
        </p>

        {error ? <Banner tone="error">{error}</Banner> : null}

        {needsPassword ? (
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
            onClick={join}
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

  return (
    <div className="flex flex-col flex-1 gap-4">
      <h2 className="text-2xl font-bold mt-2">Choose a Wi-Fi network</h2>
      <p className="text-text-dim leading-relaxed">
        Your robot needs Wi-Fi to download updates and run its application.
      </p>

      {error ? <Banner tone="error">{error}</Banner> : null}

      <ul className="flex flex-col gap-2 list-none m-0 p-0">
        {networks.map((n) => (
          <li
            key={n.ssid}
            className="bg-bg-elev border border-border rounded-[var(--radius-card)] overflow-hidden"
          >
            <button
              className="flex w-full items-center justify-between px-4 py-3.5 bg-transparent border-0 text-left cursor-pointer hover:bg-bg-elev-2"
              onClick={() => setSelected(n)}
            >
              <div>
                <div className="font-semibold">{n.ssid}</div>
                <div className="text-text-dim text-[13px] mt-0.5">
                  {n.security} · {n.signalDbm} dBm
                </div>
              </div>
              <span
                className="text-text-dim text-[22px] leading-none"
                aria-hidden
              >
                ›
              </span>
            </button>
          </li>
        ))}
        {!scanning && networks.length === 0 ? (
          <li className="text-text-dim py-4 text-center text-sm">
            No networks found.
          </li>
        ) : null}
      </ul>

      <div className="mt-auto pt-4 flex flex-col gap-3">
        <Button variant="secondary" onClick={scan} loading={scanning}>
          {scanning ? "Scanning…" : "Rescan"}
        </Button>
      </div>
    </div>
  );
}
