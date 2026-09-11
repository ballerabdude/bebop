import { useMemo, useState } from "react";

import { SetupTransport } from "./ble";
import { ConfigScreen } from "./screens/ConfigScreen";
import { ConnectByIpScreen } from "./screens/ConnectByIpScreen";
import { DashboardScreen } from "./screens/DashboardScreen";
import { MotorBenchScreen } from "./screens/MotorBenchScreen";
import { TeleopScreen } from "./screens/TeleopScreen";
import { VideoScreen } from "./screens/VideoScreen";
import { WelcomeScreen } from "./screens/WelcomeScreen";
import { WifiScreen } from "./screens/WifiScreen";
import "./App.css";

type Step =
  | "welcome"
  // Provisioning path: talk to bebop-agent's setup server (:9091), over the
  // robot's setup hotspot or the LAN.
  | "setup-address"
  | "wifi"
  | "config"
  | "dashboard"
  // Operator path: talk to bebop-linux's runtime server (:9090) on a robot
  // that is already on the network.
  | "connect-ip"
  | "direct-motors"
  | "direct-teleop"
  | "direct-video"
  | "motors"
  | "teleop"
  | "video";

function containerWidth(step: Step): string {
  if (step === "motors" || step === "direct-motors") return "max-w-6xl";
  if (step === "teleop" || step === "direct-teleop") return "max-w-6xl";
  if (step === "dashboard") return "max-w-3xl";
  return "max-w-[520px]";
}

function isSetupStep(s: Step): boolean {
  return (
    s === "welcome" ||
    s === "setup-address" ||
    s === "wifi" ||
    s === "config"
  );
}

function App() {
  const [step, setStep] = useState<Step>("welcome");

  // Setup surface (bebop-agent :9091), created once the user gives an address.
  const [setupEndpoint, setSetupEndpoint] = useState<{
    host: string;
    port: number;
  } | null>(null);
  const transport = useMemo(
    () =>
      setupEndpoint
        ? new SetupTransport(setupEndpoint.host, setupEndpoint.port)
        : null,
    [setupEndpoint],
  );

  // Operator surface (bebop-linux :9090) on a robot already on the network.
  const [directIp, setDirectIp] = useState<{ ip: string; port: number } | null>(
    null,
  );
  const [teleopReturn, setTeleopReturn] = useState<Step>("dashboard");

  // Robot IP learned once Wi-Fi is up (used to open the operator screens).
  const [robotIp, setRobotIp] = useState<string>("");

  const width = containerWidth(step);

  function reset() {
    transport?.disconnect();
    setSetupEndpoint(null);
    setRobotIp("");
    setDirectIp(null);
    setStep("welcome");
  }

  return (
    <main className={`flex flex-col min-h-full mx-auto font-sans w-full ${width}`}>
      <header className="px-4 pt-4 pb-3 sm:px-6 sm:pt-6">
        <div className="flex items-center justify-between mb-2.5 gap-3">
          <div className="text-sm tracking-[0.08em] uppercase text-text-dim">
            {isSetupStep(step) ? "Bebop · Setup" : "Bebop"}
          </div>
        </div>
      </header>

      <section className="flex-1 px-4 pt-2 pb-6 sm:px-6 flex flex-col">
        {step === "welcome" ? (
          <WelcomeScreen
            onStart={() => setStep("setup-address")}
            onConnectByIp={() => setStep("connect-ip")}
          />
        ) : null}

        {step === "setup-address" ? (
          <ConnectByIpScreen
            heading="Connect to robot setup"
            description="Join the robot's setup hotspot (Bebop-XXXX) and leave the address as-is, or enter the robot's address on your network."
            submitLabel="Connect"
            defaultPort={9091}
            defaultIp="192.168.42.1"
            onConnected={(ip, port) => {
              setSetupEndpoint({ host: ip, port });
              setStep("wifi");
            }}
            onCancel={() => setStep("welcome")}
          />
        ) : null}

        {step === "connect-ip" ? (
          <ConnectByIpScreen
            heading="Connect to controls"
            description="Enter the address of a robot that is already on your network."
            submitLabel="Connect"
            defaultPort={9090}
            onConnected={(ip, port) => {
              setDirectIp({ ip, port });
              setStep("direct-motors");
            }}
            onCancel={() => setStep("welcome")}
          />
        ) : null}

        {step === "wifi" && transport ? (
          <WifiScreen
            transport={transport}
            onDone={(status) => {
              setRobotIp(status.ipAddress || "");
              if (status.connected && status.ipAddress) {
                setStep("config");
              } else {
                // The setup hotspot dropped while the robot joined the new
                // network. Send the user to the operator connect screen so
                // they can reach the robot at its new address.
                setStep("connect-ip");
              }
            }}
          />
        ) : null}

        {step === "config" && transport ? (
          <ConfigScreen transport={transport} onDone={() => setStep("dashboard")} />
        ) : null}

        {step === "dashboard" && transport ? (
          <DashboardScreen
            transport={transport}
            onIp={setRobotIp}
            onReconfigure={() => setStep("wifi")}
            onDisconnect={reset}
            onOpenMotors={() => setStep("motors")}
            onOpenTeleop={() => {
              setTeleopReturn("dashboard");
              setStep("teleop");
            }}
          />
        ) : null}

        {step === "motors" && robotIp ? (
          <MotorBenchScreen
            robotIp={robotIp}
            onBack={() => setStep("dashboard")}
            onOpenVideo={() => setStep("video")}
            onOpenTeleop={() => {
              setTeleopReturn("motors");
              setStep("teleop");
            }}
          />
        ) : null}

        {step === "direct-motors" && directIp ? (
          <MotorBenchScreen
            robotIp={directIp.ip}
            runtimePort={directIp.port}
            onBack={() => setStep("connect-ip")}
            onOpenVideo={() => setStep("direct-video")}
            onOpenTeleop={() => {
              setTeleopReturn("direct-motors");
              setStep("direct-teleop");
            }}
          />
        ) : null}

        {step === "video" && robotIp ? (
          <VideoScreen
            robotIp={robotIp}
            onBack={() => setStep("motors")}
            backLabel="Back to motor bench"
          />
        ) : null}

        {step === "direct-video" && directIp ? (
          <VideoScreen
            robotIp={directIp.ip}
            runtimePort={directIp.port}
            onBack={() => setStep("direct-motors")}
            backLabel="Back to motor bench"
          />
        ) : null}

        {step === "teleop" && robotIp ? (
          <TeleopScreen
            robotIp={robotIp}
            onBack={() => setStep(teleopReturn)}
            backLabel={
              teleopReturn === "motors" ? "Back to motor bench" : "Back to dashboard"
            }
          />
        ) : null}

        {step === "direct-teleop" && directIp ? (
          <TeleopScreen
            robotIp={directIp.ip}
            runtimePort={directIp.port}
            onBack={() => setStep(teleopReturn)}
            backLabel={
              teleopReturn === "direct-motors"
                ? "Back to motor bench"
                : "Back to dashboard"
            }
          />
        ) : null}
      </section>
    </main>
  );
}

export default App;
