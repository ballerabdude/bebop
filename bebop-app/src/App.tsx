import { useEffect, useMemo, useRef, useState } from "react";

import { bluetoothSupported, createTransport } from "./ble";
import type { DiscoveredRobot, WifiStatus } from "./ble";
import { Spinner } from "./components/ui";
import { ConfigScreen } from "./screens/ConfigScreen";
import { ConnectByIpScreen } from "./screens/ConnectByIpScreen";
import { ControllersScreen } from "./screens/ControllersScreen";
import { DashboardScreen } from "./screens/DashboardScreen";
import { MotorBenchScreen } from "./screens/MotorBenchScreen";
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
  | "dashboard"
  | "motors"
  | "controllers"
  // IP-only path: skip BLE entirely. `direct-motors` is the same screen
  // as `motors` but with a back button that returns to the IP form rather
  // than to the (nonexistent) BLE dashboard.
  | "connect-ip"
  | "direct-motors";

const SETUP_ORDER: Step[] = ["welcome", "scan", "wifi", "config", "dashboard"];

function isSetupStep(s: Step): boolean {
  return SETUP_ORDER.includes(s);
}

// Per-step content width. Setup wizard is purposely narrow (mobile-first
// reading width); operating screens get more room so motor names don't
// truncate and the toolbar can lay out horizontally on desktop.
function containerWidth(step: Step): string {
  if (step === "motors" || step === "direct-motors") return "max-w-6xl";
  if (step === "dashboard") return "max-w-3xl";
  return "max-w-[520px]";
}

// Step shown when the user opens "Bluetooth controllers" from the
// dashboard. Same width as the wifi reconfig path for visual parity.

function App() {
  const supported = bluetoothSupported();
  const transport = useMemo(
    () => (supported ? createTransport() : null),
    [supported],
  );
  // Initial step depends on whether BLE is available: with BLE, start at
  // welcome (BLE setup wizard). Without BLE, jump straight to the IP form.
  const [step, setStep] = useState<Step>(supported ? "welcome" : "connect-ip");
  const [, setRobot] = useState<DiscoveredRobot | null>(null);
  const [wifi, setWifi] = useState<WifiStatus | null>(null);
  // Direct-IP endpoint, populated by ConnectByIpScreen when the user
  // chooses the no-BLE path.
  const [directIp, setDirectIp] = useState<{ ip: string; port: number } | null>(
    null,
  );
  // While true, we're trying to silently re-attach to a previously-paired
  // robot. Only meaningful when BLE is supported.
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
  // robot, try to reconnect to it and skip directly to the dashboard.
  useEffect(() => {
    if (!transport) return;
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
          setStep("wifi");
        }
      } catch {
        /* leave at welcome */
      } finally {
        setResuming(false);
      }
    })();
  }, [transport]);

  const width = containerWidth(step);

  return (
    <main className={`flex flex-col min-h-full mx-auto font-sans w-full ${width}`}>
      <header className="px-5 pt-5 pb-3 sm:px-6 sm:pt-6">
        <div className="flex items-center justify-between mb-2.5 gap-3">
          <div className="text-sm tracking-[0.08em] uppercase text-text-dim">
            {step === "dashboard" ||
            step === "wifi-reconfig" ||
            step === "motors" ||
            step === "direct-motors" ||
            step === "controllers"
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

      <section className="flex-1 px-5 pt-2 pb-6 sm:px-6 flex flex-col">
        {resuming ? (
          <div className="flex-1 flex flex-col items-center justify-center gap-3 text-text-dim">
            <Spinner />
            <div className="text-sm">Looking for your robot…</div>
          </div>
        ) : null}

        {!resuming && step === "welcome" ? (
          <WelcomeScreen
            onStart={() => setStep("scan")}
            onConnectByIp={() => setStep("connect-ip")}
          />
        ) : null}

        {!resuming && step === "connect-ip" ? (
          <ConnectByIpScreen
            // If we already discovered the robot's IP via BLE, prefill it
            // so re-connecting is one tap.
            prefillIp={wifi?.ipAddress}
            onConnected={(ip, port) => {
              setDirectIp({ ip, port });
              setStep("direct-motors");
            }}
            onCancel={supported ? () => setStep("welcome") : undefined}
          />
        ) : null}

        {!resuming && step === "scan" && transport ? (
          <ScanScreen
            transport={transport}
            onConnected={(r) => {
              setRobot(r);
              setStep("wifi");
            }}
          />
        ) : null}

        {!resuming &&
        (step === "wifi" || step === "wifi-reconfig") &&
        transport ? (
          <WifiScreen
            transport={transport}
            onDone={(status) => {
              setWifi(status);
              if (step === "wifi-reconfig") {
                setStep("dashboard");
              } else {
                setStep("config");
              }
            }}
          />
        ) : null}

        {!resuming && step === "config" && transport ? (
          <ConfigScreen
            transport={transport}
            onDone={() => setStep("dashboard")}
          />
        ) : null}

        {!resuming && step === "dashboard" && wifi && transport ? (
          <DashboardScreen
            transport={transport}
            wifi={wifi}
            onReconfigure={() => setStep("wifi-reconfig")}
            onDisconnect={reset}
            onOpenMotors={() => setStep("motors")}
            onOpenControllers={() => setStep("controllers")}
          />
        ) : null}

        {!resuming && step === "controllers" && transport ? (
          <ControllersScreen
            transport={transport}
            onDone={() => setStep("dashboard")}
          />
        ) : null}

        {!resuming && step === "motors" && wifi?.ipAddress ? (
          <MotorBenchScreen
            robotIp={wifi.ipAddress}
            onBack={() => setStep("dashboard")}
          />
        ) : null}

        {!resuming && step === "direct-motors" && directIp ? (
          <MotorBenchScreen
            robotIp={directIp.ip}
            runtimePort={directIp.port}
            onBack={() => setStep("connect-ip")}
          />
        ) : null}
      </section>
    </main>
  );
}

export default App;
