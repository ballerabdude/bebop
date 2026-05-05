import { useEffect, useRef, useState } from "react";

import type {
  BebopTransport,
  ControllerStatus,
  DiscoveredController,
} from "../ble";
import { Banner, Button, Card, Spinner } from "../components/ui";

interface ControllersScreenProps {
  transport: BebopTransport;
  /// Called when the user taps "Back to dashboard". The screen is
  /// reachable from the dashboard so we always have somewhere to
  /// return to.
  onDone: () => void;
}

const SCAN_TIMEOUT_MS = 8_000;
const STATUS_POLL_MS = 1_000;

/// Pair a Bluetooth gamepad with the robot, then verify the
/// deadman + e-stop wiring before driving anything heavy.
///
/// The agent owns BlueZ; this screen only orchestrates a
/// scan / pair / unpair RPC sequence and renders the live
/// `ControllerStatus` it gets back so the user can see the deadman
/// engage in real time.
export function ControllersScreen({
  transport,
  onDone,
}: ControllersScreenProps) {
  const [status, setStatus] = useState<ControllerStatus | null>(null);
  const [devices, setDevices] = useState<DiscoveredController[]>([]);
  const [scanning, setScanning] = useState(false);
  const [showAll, setShowAll] = useState(false);
  const [pairingMac, setPairingMac] = useState<string | null>(null);
  const [unpairing, setUnpairing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loadingStatus, setLoadingStatus] = useState(true);
  // Track the latest status fetch so a stale request firing after a
  // pair/unpair doesn't clobber the fresh state.
  const lastStatusReq = useRef(0);

  async function fetchStatus() {
    const seq = ++lastStatusReq.current;
    try {
      const s = await transport.getControllerStatus();
      if (seq === lastStatusReq.current) {
        setStatus(s);
      }
    } catch {
      // Older agents won't have getControllerStatus; treat as
      // "no controller subsystem" instead of an error.
    } finally {
      setLoadingStatus(false);
    }
  }

  async function scan() {
    setError(null);
    setScanning(true);
    try {
      const list = await transport.scanControllers(SCAN_TIMEOUT_MS);
      // Sort: gamepads first, then by signal strength descending.
      const sorted = [...list].sort((a, b) => {
        if (a.kind !== b.kind) return a.kind === "gamepad" ? -1 : 1;
        return b.rssi - a.rssi;
      });
      setDevices(sorted);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setScanning(false);
    }
  }

  async function pair(mac: string) {
    setError(null);
    setPairingMac(mac);
    try {
      const next = await transport.pairController(mac);
      setStatus(next);
      // Clear the discovered-list selection cue. Re-scan in the
      // background so the new device's `paired/connected` flags
      // refresh in case the user wants to compare.
      void scan();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setPairingMac(null);
    }
  }

  async function unpair() {
    if (!status?.pairedMac) return;
    setError(null);
    setUnpairing(true);
    try {
      const next = await transport.unpairController(status.pairedMac);
      setStatus(next);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setUnpairing(false);
    }
  }

  // Initial load: status first (cheap), then a fresh scan.
  useEffect(() => {
    void (async () => {
      await fetchStatus();
      void scan();
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Poll status while the screen is open so the armed / e-stop /
  // connected pills update live as the user tests the controller.
  useEffect(() => {
    const id = setInterval(fetchStatus, STATUS_POLL_MS);
    return () => clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const visibleDevices = showAll
    ? devices
    : devices.filter((d) => d.kind === "gamepad");

  const paired = status && status.pairedMac.length > 0;

  return (
    <div className="flex flex-col flex-1 gap-4">
      <h2 className="text-2xl font-bold mt-2">Bluetooth controller</h2>
      <p className="text-text-dim leading-relaxed">
        Pair a gamepad to drive Bebop manually. Hold the right trigger
        as a deadman; press Circle (or B) to e-stop and Cross (or A)
        to re-arm.
      </p>

      {error ? <Banner tone="error">{error}</Banner> : null}

      {/* Currently paired controller --------------------------------- */}
      {loadingStatus ? (
        <div className="flex justify-center py-4">
          <Spinner />
        </div>
      ) : paired ? (
        <Card>
          <div className="py-1 flex flex-col gap-2">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-xs text-text-dim uppercase tracking-wider mb-0.5">
                  Paired controller
                </div>
                <div className="font-semibold">
                  {status?.deviceName || "Unnamed controller"}
                </div>
                <div className="text-text-dim text-[13px] font-mono">
                  {status?.pairedMac}
                </div>
              </div>
              <div className="flex flex-col items-end gap-1">
                <StatusPill
                  label={status?.connected ? "connected" : "offline"}
                  tone={status?.connected ? "ok" : "muted"}
                />
                <StatusPill
                  label={
                    status?.estopLatched
                      ? "e-stop"
                      : status?.armed
                        ? "armed"
                        : "idle"
                  }
                  tone={
                    status?.estopLatched
                      ? "danger"
                      : status?.armed
                        ? "ok"
                        : "muted"
                  }
                />
              </div>
            </div>
            <div className="text-text-dim text-[12px]">
              Forwarding to {status?.targetAddr}
            </div>
            <div className="mt-1">
              <Button
                variant="secondary"
                onClick={unpair}
                loading={unpairing}
                className="py-2!"
              >
                Unpair
              </Button>
            </div>
          </div>
        </Card>
      ) : (
        <Card>
          <div className="py-2 text-text-dim text-sm">
            No controller paired. Pick one below to bind it to this robot.
          </div>
        </Card>
      )}

      {/* Discovered devices list ------------------------------------- */}
      <div className="flex items-center justify-between">
        <div className="text-xs text-text-dim uppercase tracking-wider">
          Nearby Bluetooth devices
        </div>
        <label className="flex items-center gap-2 text-[13px] text-text-dim cursor-pointer select-none">
          <input
            type="checkbox"
            checked={showAll}
            onChange={(e) => setShowAll(e.currentTarget.checked)}
            className="accent-accent"
          />
          Show all
        </label>
      </div>

      <ul className="flex flex-col gap-2 list-none m-0 p-0">
        {visibleDevices.map((d) => {
          const isCurrent =
            paired && d.mac.toLowerCase() === status?.pairedMac.toLowerCase();
          const pairing = pairingMac === d.mac;
          return (
            <li
              key={d.mac}
              className={`border rounded-[var(--radius-card)] overflow-hidden ${
                isCurrent
                  ? "bg-accent/10 border-accent/40"
                  : "bg-bg-elev border-border"
              }`}
            >
              <div className="flex w-full items-center justify-between px-4 py-3.5 gap-3">
                <div className="min-w-0">
                  <div className="font-semibold flex items-center gap-2">
                    <span className="truncate">
                      {d.name || "Unnamed device"}
                    </span>
                    {d.kind === "gamepad" ? (
                      <span className="text-[11px] font-normal text-accent bg-accent/15 px-1.5 py-0.5 rounded-full shrink-0">
                        gamepad
                      </span>
                    ) : null}
                    {isCurrent ? (
                      <span className="text-[11px] font-normal text-success bg-success/15 px-1.5 py-0.5 rounded-full shrink-0">
                        active
                      </span>
                    ) : null}
                  </div>
                  <div className="text-text-dim text-[13px] mt-0.5 font-mono">
                    {d.mac}
                    {d.rssi !== 0 ? ` · ${d.rssi} dBm` : ""}
                  </div>
                </div>
                {!isCurrent ? (
                  <Button
                    variant="secondary"
                    onClick={() => pair(d.mac)}
                    loading={pairing}
                    disabled={pairingMac !== null}
                    className="py-2! text-sm! shrink-0"
                  >
                    Pair
                  </Button>
                ) : null}
              </div>
            </li>
          );
        })}
        {!scanning && visibleDevices.length === 0 ? (
          <li className="text-text-dim py-4 text-center text-sm">
            {devices.length === 0
              ? "No devices found. Put your controller in pairing mode and rescan."
              : "No gamepads found. Toggle “Show all” to see other devices."}
          </li>
        ) : null}
      </ul>

      <div className="mt-auto pt-4 flex flex-col gap-3">
        <Button variant="secondary" onClick={scan} loading={scanning}>
          {scanning ? "Scanning…" : "Rescan"}
        </Button>
        <Button variant="ghost" onClick={onDone}>
          Back to dashboard
        </Button>
      </div>
    </div>
  );
}

function StatusPill({
  label,
  tone,
}: {
  label: string;
  tone: "ok" | "muted" | "danger";
}) {
  const toneClasses: Record<typeof tone, string> = {
    ok: "bg-success/15 text-success",
    muted: "bg-bg-elev-2 text-text-dim",
    danger: "bg-danger/12 text-[#ffb5b8]",
  };
  return (
    <span
      className={`text-[11px] font-semibold uppercase tracking-wider px-2 py-0.5 rounded-full ${toneClasses[tone]}`}
    >
      {label}
    </span>
  );
}
