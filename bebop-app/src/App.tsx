import { useEffect, useMemo, useRef, useState } from "react";

import { bluetoothSupported, createTransport } from "./ble";
import type { DiscoveredRobot, WifiStatus } from "./ble";
import { Spinner } from "./components/ui";
import { ConfigScreen } from "./screens/ConfigScreen";
import { DashboardScreen } from "./screens/DashboardScreen";
import { ScanScreen } from "./screens/ScanScreen";
import { WelcomeScreen } from "./screens/WelcomeScreen";
import { WifiScreen } from "./screens/WifiScreen";
import "./App.css";

type Step =
  | "welcome"
  | "scan"
  | "wifi"
  | "wifi-reconfig"
  | "config"
  | "dashboard";

const SETUP_ORDER: Step[] = ["welcome", "scan", "wifi", "config", "dashboard"];

/** Is this step part of first-time setup? Controls progress bar visibility. */
function isSetupStep(s: Step): boolean {
  return SETUP_ORDER.includes(s);
}

function App() {
  const supported = bluetoothSupported();
  const transport = useMemo(
    () => (supported ? createTransport() : null),
    [supported],
  );
  const [step, setStep] = useState<Step>("welcome");
  const [, setRobot] = useState<DiscoveredRobot | null>(null);
  const [wifi, setWifi] = useState<WifiStatus | null>(null);
  // While true, we're trying to silently re-attach to a previously-paired
  // robot. The browser drops the GATT link on every refresh, but the OS
  // pairing is preserved, so we can rebuild the session transparently.
  const [resuming, setResuming] = useState<boolean>(supported);
  const resumeStarted = useRef(false);

  const setupIdx = SETUP_ORDER.indexOf(step);
  const progress =
    setupIdx >= 0 ? ((setupIdx + 1) / SETUP_ORDER.length) * 100 : 0;

  async function reset() {
    if (!transport) return;
    try {
      if (transport.isConnected()) await transport.disconnect();
    } catch {
      /* ignore */
    }
    setRobot(null);
    setWifi(null);
    setStep("welcome");
  }

  // Auto-resume: on first mount, if there's exactly one already-permitted
  // robot, try to reconnect to it and skip directly to the dashboard (or
  // the Wi-Fi step if setup was never finished). Falls back to the
  // welcome flow if anything goes wrong — we never want auto-reconnect
  // to block the user from the manual path.
  useEffect(() => {
    if (!transport) return;
    // Guard against React.StrictMode's intentional double-invocation of
    // effects in dev — otherwise we'd race two `gatt.connect()` calls.
    if (resumeStarted.current) return;
    resumeStarted.current = true;

    void (async () => {
      try {
        const known = await transport.scan(2_000);
        if (known.length !== 1) return;
        const robot = known[0];
        await transport.connect(robot.id);
        const status = await transport.getWifiStatus();
        setRobot(robot);
        if (status.connected) {
          setWifi(status);
          setStep("dashboard");
        } else {
          // Robot is back but never finished Wi-Fi setup; drop into the
          // Wi-Fi step rather than the dashboard.
          setStep("wifi");
        }
      } catch {
        // Robot out of range, agent not running, or no remembered device
        // — leave the user at the welcome screen.
      } finally {
        setResuming(false);
      }
    })();
  }, [transport]);

  if (!transport) {
    return (
      <main className="flex flex-col min-h-full max-w-[520px] mx-auto font-sans px-5 py-8">
        <div className="rounded-card border border-border bg-bg-elev p-6 text-text">
          <h1 className="text-lg font-semibold mb-2">Bluetooth unavailable</h1>
          <p className="text-sm text-text-dim leading-relaxed">
            This browser doesn&rsquo;t support Web Bluetooth. Install the Bebop
            desktop / mobile app, or open this page in Chrome or Edge on a
            device with Bluetooth.
          </p>
        </div>
      </main>
    );
  }

  return (
    <main className="flex flex-col min-h-full max-w-[520px] mx-auto font-sans">
      <header className="px-5 pt-5 pb-3">
        <div className="flex items-center justify-between mb-2.5 gap-3">
          <div className="text-sm tracking-[0.08em] uppercase text-text-dim">
            {step === "dashboard" || step === "wifi-reconfig"
              ? "Bebop"
              : "Bebop · Setup"}
          </div>
        </div>
        {!resuming && isSetupStep(step) ? (
          <div className="h-1 bg-bg-elev rounded-full overflow-hidden">
            <div
              className="h-full bg-accent transition-[width] duration-200 ease"
              style={{ width: `${progress}%` }}
            />
          </div>
        ) : null}
      </header>

      <section className="flex-1 px-5 pt-2 pb-6 flex flex-col">
        {resuming ? (
          <div className="flex-1 flex flex-col items-center justify-center gap-3 text-text-dim">
            <Spinner />
            <div className="text-sm">Looking for your robot…</div>
          </div>
        ) : null}

        {!resuming && step === "welcome" ? (
          <WelcomeScreen onStart={() => setStep("scan")} />
        ) : null}

        {!resuming && step === "scan" ? (
          <ScanScreen
            transport={transport}
            onConnected={(r) => {
              setRobot(r);
              setStep("wifi");
            }}
          />
        ) : null}

        {!resuming && (step === "wifi" || step === "wifi-reconfig") ? (
          <WifiScreen
            transport={transport}
            onDone={(status) => {
              setWifi(status);
              // If coming from the dashboard (reconfig), go back there.
              // Otherwise continue the setup flow.
              if (step === "wifi-reconfig") {
                setStep("dashboard");
              } else {
                setStep("config");
              }
            }}
          />
        ) : null}

        {!resuming && step === "config" ? (
          <ConfigScreen
            transport={transport}
            onDone={() => setStep("dashboard")}
          />
        ) : null}

        {!resuming && step === "dashboard" && wifi ? (
          <DashboardScreen
            transport={transport}
            wifi={wifi}
            onReconfigure={() => setStep("wifi-reconfig")}
            onDisconnect={reset}
          />
        ) : null}
      </section>
    </main>
  );
}

export default App;
