import { useEffect, useState } from "react";

import type { BebopTransport, DiscoveredRobot } from "../ble";
import { Banner, Button, Spinner } from "../components/ui";

export function ScanScreen({
  transport,
  onConnected,
}: {
  transport: BebopTransport;
  onConnected: (robot: DiscoveredRobot) => void;
}) {
  const [scanning, setScanning] = useState(false);
  const [robots, setRobots] = useState<DiscoveredRobot[]>([]);
  const [connectingId, setConnectingId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function scan() {
    setError(null);
    setScanning(true);
    try {
      const found = await transport.scan(5_000);
      setRobots(found);
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

  async function connect(robot: DiscoveredRobot) {
    setError(null);
    setConnectingId(robot.id);
    try {
      await transport.connect(robot.id);
      onConnected(robot);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setConnectingId(null);
    }
  }

  return (
    <div className="flex flex-col flex-1 gap-4">
      <h2 className="text-2xl font-bold mt-2">Find your robot</h2>
      <p className="text-text-dim leading-relaxed">
        Tap the robot you want to configure. The serial number is on the sticker
        underneath your Bebop.
      </p>

      {error ? <Banner tone="error">{error}</Banner> : null}

      <ul className="flex flex-col gap-2 list-none m-0 p-0">
        {robots.map((r) => {
          const bars = signalBars(r.rssi);
          const isConnecting = connectingId === r.id;
          return (
            <li
              key={r.id}
              className="bg-bg-elev border border-border rounded-[var(--radius-card)] overflow-hidden"
            >
              <button
                className="flex w-full items-center justify-between px-4 py-3.5 bg-transparent border-0 text-left cursor-pointer hover:bg-bg-elev-2 disabled:opacity-55 disabled:cursor-not-allowed"
                onClick={() => connect(r)}
                disabled={connectingId !== null}
              >
                <div>
                  <div className="font-semibold">{r.name}</div>
                  <div className="text-text-dim text-[13px] mt-0.5">{r.id}</div>
                </div>
                <div className="flex items-center gap-2.5 text-text-dim text-sm">
                  <span
                    className="font-mono tracking-wider text-accent"
                    aria-label={`${bars} of 4 bars`}
                  >
                    {"▂▃▅▇".slice(0, bars)}
                  </span>
                  {isConnecting ? <Spinner /> : null}
                </div>
              </button>
            </li>
          );
        })}
        {!scanning && robots.length === 0 ? (
          <li className="text-text-dim py-4 text-center text-sm">
            No robots found nearby.
          </li>
        ) : null}
      </ul>

      <div className="mt-auto pt-4 flex flex-col gap-3">
        <Button variant="secondary" onClick={scan} loading={scanning}>
          {scanning ? "Scanning…" : "Scan again"}
        </Button>
      </div>
    </div>
  );
}

function signalBars(rssi: number): number {
  if (rssi >= -55) return 4;
  if (rssi >= -65) return 3;
  if (rssi >= -75) return 2;
  if (rssi >= -85) return 1;
  return 0;
}
