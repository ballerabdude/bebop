import { useMemo, useState } from "react";

import { bluetoothSupported, createTransport } from "./ble";
import type { DiscoveredRobot, WifiStatus } from "./ble";
import { ConfigScreen } from "./screens/ConfigScreen";
import { DoneScreen } from "./screens/DoneScreen";
import { ScanScreen } from "./screens/ScanScreen";
import { WelcomeScreen } from "./screens/WelcomeScreen";
import { WifiScreen } from "./screens/WifiScreen";
import "./App.css";

type Step = "welcome" | "scan" | "wifi" | "config" | "done";

const STEP_ORDER: Step[] = ["welcome", "scan", "wifi", "config", "done"];

function App() {
  const supported = bluetoothSupported();
  const transport = useMemo(
    () => (supported ? createTransport() : null),
    [supported],
  );
  const [step, setStep] = useState<Step>("welcome");
  const [, setRobot] = useState<DiscoveredRobot | null>(null);
  const [wifi, setWifi] = useState<WifiStatus | null>(null);

  const progress = ((STEP_ORDER.indexOf(step) + 1) / STEP_ORDER.length) * 100;

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
            Bebop · Setup
          </div>
        </div>
        <div className="h-1 bg-bg-elev rounded-full overflow-hidden">
          <div
            className="h-full bg-accent transition-[width] duration-200 ease"
            style={{ width: `${progress}%` }}
          />
        </div>
      </header>

      <section className="flex-1 px-5 pt-2 pb-6 flex flex-col">
        {step === "welcome" ? (
          <WelcomeScreen onStart={() => setStep("scan")} />
        ) : null}

        {step === "scan" ? (
          <ScanScreen
            transport={transport}
            onConnected={(r) => {
              setRobot(r);
              setStep("wifi");
            }}
          />
        ) : null}

        {step === "wifi" ? (
          <WifiScreen
            transport={transport}
            onDone={(status) => {
              setWifi(status);
              setStep("config");
            }}
          />
        ) : null}

        {step === "config" ? (
          <ConfigScreen transport={transport} onDone={() => setStep("done")} />
        ) : null}

        {step === "done" && wifi ? (
          <DoneScreen transport={transport} wifi={wifi} onStartOver={reset} />
        ) : null}
      </section>
    </main>
  );
}

export default App;
