import { useCallback, useEffect, useState } from "react";

import type {
  AppStatus,
  BebopTransport,
  DeviceInfo,
  OtaStatus,
  WifiStatus,
} from "../ble";
import { Banner, Button, Card, Field, Spinner } from "../components/ui";

interface DashboardProps {
  transport: BebopTransport;
  wifi: WifiStatus;
  onReconfigure: () => void;
  onDisconnect: () => void;
  onOpenMotors: () => void;
}

/// Live dashboard shown after initial setup succeeds. Stays connected via
/// BLE so the user can monitor the robot and adjust settings.
export function DashboardScreen({
  transport,
  wifi,
  onReconfigure,
  onDisconnect,
  onOpenMotors,
}: DashboardProps) {
  const [info, setInfo] = useState<DeviceInfo | null>(null);
  const [app, setApp] = useState<AppStatus | null>(null);
  const [ota, setOta] = useState<OtaStatus | null>(null);
  const [currentWifi, setCurrentWifi] = useState<WifiStatus>(wifi);
  const [error, setError] = useState<string | null>(null);
  const [otaRunning, setOtaRunning] = useState(false);
  // App-container management: image edit + start/stop/restart.
  const [editingImage, setEditingImage] = useState(false);
  const [imageDraft, setImageDraft] = useState("");
  const [savingImage, setSavingImage] = useState(false);
  // True after a successful image save, until a Restart actually applies
  // it to the running container. Cleared whenever the running container's
  // image matches the configured one.
  const [pendingRestart, setPendingRestart] = useState(false);
  const [appAction, setAppAction] = useState<
    "START" | "STOP" | "RESTART" | null
  >(null);

  const refresh = useCallback(async () => {
    try {
      const [a, o, w] = await Promise.all([
        transport.getAppStatus(),
        transport.getOtaStatus(),
        transport.getWifiStatus(),
      ]);
      setApp(a);
      setOta(o);
      setCurrentWifi(w);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [transport]);

  useEffect(() => {
    void (async () => {
      try {
        const i = await transport.getDeviceInfo();
        setInfo(i);
      } catch {
        /* device info optional */
      }
      await refresh();
    })();

    const id = setInterval(refresh, 5_000);
    return () => clearInterval(id);
  }, [refresh]);

  async function checkForUpdate() {
    setError(null);
    setOtaRunning(true);
    try {
      await transport.triggerOta();
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setOtaRunning(false);
    }
  }

  function startEditingImage() {
    setError(null);
    setImageDraft(app?.image ?? "");
    setEditingImage(true);
  }

  async function saveImage() {
    setError(null);
    setSavingImage(true);
    try {
      const next = await transport.setAppImage(imageDraft.trim());
      setApp(next);
      setEditingImage(false);
      // The agent persisted the new image, but the running container
      // hasn't been swapped yet. Surface a hint until the user restarts.
      if (next.image !== imageDraft.trim()) {
        setPendingRestart(true);
      } else {
        // No change after save (e.g. cleared while already empty).
        setPendingRestart(false);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSavingImage(false);
    }
  }

  async function runAppAction(action: "START" | "STOP" | "RESTART") {
    if (!app) return;
    setError(null);
    setAppAction(action);
    try {
      await transport.controlApp(app.appName || "", action);
      await refresh();
      // A successful restart applies the configured image; the hint can
      // go away. A bare Start may also have done so.
      if (action === "RESTART" || action === "START") {
        setPendingRestart(false);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setAppAction(null);
    }
  }

  return (
    <div className="flex flex-col flex-1 gap-4">
      <div className="text-center mt-3">
        <div
          className="w-14 h-14 mx-auto mb-3 rounded-full bg-success/15 text-success flex items-center justify-center text-[28px] font-bold"
          aria-hidden
        >
          ✓
        </div>
        <h2 className="text-2xl font-bold">Bebop is online</h2>
        {info ? (
          <p className="text-text-dim text-sm mt-1">
            {info.model} · agent v{info.agentVersion}
          </p>
        ) : null}
      </div>

      {error ? <Banner tone="error">{error}</Banner> : null}

      {/* Status cards. On desktop, Wi-Fi and OTA sit side by side above
          the wider Robot Application card so the dashboard fills the
          available width without forcing a long vertical scroll. */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Wi-Fi card */}
        <Card>
          <div className="flex items-center justify-between py-1">
            <div>
              <div className="text-xs text-text-dim uppercase tracking-wider mb-1">
                Wi-Fi
              </div>
              <div className="font-semibold">
                {currentWifi.connected ? currentWifi.ssid : "Not connected"}
              </div>
              {currentWifi.connected && currentWifi.ipAddress ? (
                <div className="text-text-dim text-[13px] mt-0.5">
                  {currentWifi.ipAddress} · {currentWifi.signalDbm} dBm
                </div>
              ) : null}
            </div>
            <div>
              <div
                className={`w-2.5 h-2.5 rounded-full ${
                  currentWifi.connected ? "bg-success" : "bg-text-dim/40"
                }`}
              />
            </div>
          </div>
        </Card>

        {/* OTA card */}
        <Card>
          <div className="py-1">
            <div className="text-xs text-text-dim uppercase tracking-wider mb-1">
              System update
            </div>
            {ota ? (
              <div className="flex items-center justify-between">
                <div className="font-semibold">{ota.state}</div>
                {ota.progressPercent > 0 && ota.progressPercent < 100 ? (
                  <span className="text-sm text-accent">
                    {ota.progressPercent}%
                  </span>
                ) : null}
              </div>
            ) : (
              <div className="flex items-center gap-2 text-text-dim text-sm">
                <Spinner />
                Loading…
              </div>
            )}
            {ota?.error ? (
              <div className="text-[13px] text-[#ffb5b8] mt-0.5 truncate">
                {ota.error}
              </div>
            ) : null}
          </div>
        </Card>
      </div>

      {/* Robot app card */}
      <Card>
        <div className="py-1 flex flex-col gap-3">
          <div className="text-xs text-text-dim uppercase tracking-wider">
            Robot application
          </div>
          {app ? (
            <>
              <div className="flex items-center justify-between">
                <div className="font-semibold">{app.appName || "—"}</div>
                <span
                  className={`text-xs font-semibold px-2 py-0.5 rounded-full ${
                    app.state === "RUNNING"
                      ? "bg-success/15 text-success"
                      : app.state === "CRASHED"
                        ? "bg-danger/12 text-[#ffb5b8]"
                        : "bg-bg-elev-2 text-text-dim"
                  }`}
                >
                  {app.state}
                </span>
              </div>

              {editingImage ? (
                <Field
                  label="Container image"
                  hint="Persists to /etc/bebop/agent.toml. Restart the container to apply."
                >
                  <input
                    type="text"
                    autoFocus
                    spellCheck={false}
                    autoCapitalize="off"
                    autoCorrect="off"
                    value={imageDraft}
                    onChange={(e) => setImageDraft(e.currentTarget.value)}
                    placeholder="registry/bebop-app:1.2.3"
                    className="bg-bg-elev-2 border border-border rounded-[var(--radius-card)] px-3 py-2.5 text-text font-mono text-[13px] outline-none focus:border-accent transition-colors duration-120"
                  />
                  <div className="flex flex-row gap-2 mt-2">
                    <Button
                      variant="secondary"
                      onClick={() => {
                        setEditingImage(false);
                        setImageDraft("");
                      }}
                      disabled={savingImage}
                      className="flex-1 py-2.5!"
                    >
                      Cancel
                    </Button>
                    <Button
                      onClick={saveImage}
                      loading={savingImage}
                      className="flex-1 py-2.5!"
                    >
                      Save
                    </Button>
                  </div>
                </Field>
              ) : (
                <div className="flex items-start justify-between gap-3">
                  <div className="text-text-dim text-[13px] font-mono truncate min-w-0">
                    {app.image || (
                      <span className="italic text-text-dim/70">
                        no image configured
                      </span>
                    )}
                  </div>
                  <button
                    type="button"
                    onClick={startEditingImage}
                    className="text-accent text-[13px] font-semibold cursor-pointer bg-transparent border-0 px-0 hover:underline shrink-0"
                  >
                    Edit
                  </button>
                </div>
              )}

              {pendingRestart && !editingImage ? (
                <Banner tone="info">
                  Image saved. Restart the container to apply.
                </Banner>
              ) : null}

              {!editingImage ? (
                <div className="flex flex-row gap-2">
                  <Button
                    variant="secondary"
                    onClick={() => runAppAction("START")}
                    loading={appAction === "START"}
                    disabled={
                      appAction !== null ||
                      app.state === "RUNNING" ||
                      app.state === "STARTING" ||
                      !app.image
                    }
                    className="flex-1 py-2.5! text-sm!"
                  >
                    Start
                  </Button>
                  <Button
                    variant="secondary"
                    onClick={() => runAppAction("STOP")}
                    loading={appAction === "STOP"}
                    disabled={appAction !== null || app.state === "STOPPED"}
                    className="flex-1 py-2.5! text-sm!"
                  >
                    Stop
                  </Button>
                  <Button
                    variant="secondary"
                    onClick={() => runAppAction("RESTART")}
                    loading={appAction === "RESTART"}
                    disabled={appAction !== null || !app.image}
                    className="flex-1 py-2.5! text-sm!"
                  >
                    Restart
                  </Button>
                </div>
              ) : null}
            </>
          ) : (
            <div className="flex items-center gap-2 text-text-dim text-sm">
              <Spinner />
              Loading…
            </div>
          )}
        </div>
      </Card>

      <div className="mt-auto pt-4 flex flex-col gap-3">
        <Button
          onClick={onOpenMotors}
          disabled={!currentWifi.connected || !currentWifi.ipAddress}
        >
          Open motor bench
        </Button>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
          <Button
            variant="secondary"
            onClick={checkForUpdate}
            loading={otaRunning}
          >
            Check for updates
          </Button>
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
