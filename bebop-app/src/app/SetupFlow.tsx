// `#/setup/*` — first-time provisioning, outside the operator shell:
//
//   connect   reach bebop-agent (:9091), over the hotspot or the LAN
//   wifi      join a network (or skip and keep the current one)
//   identity  name the robot, then hand off to its Drive page
//
// The setup host rides in `?host=` so a reload resumes the same step. In
// Hosted (AP) mode the robot stays at 192.168.42.1 and the operator routes
// work against that address directly; nothing is gated on client Wi-Fi.

import { useState } from "react";
import { Link, Navigate, Route, Routes, useNavigate, useSearchParams } from "react-router";

import { Banner, Button, Card, Field, INPUT_CLASS } from "../components/ui";
import { HOTSPOT_HOST, normalizeHost, PORTS } from "../robot/endpoints";
import { ConfigScreen } from "../screens/ConfigScreen";
import { WifiScreen } from "../screens/WifiScreen";
import { useSetupTransport } from "../setup";
import { robotPath, setupPath } from "./paths";
import { probe, rememberRobot } from "./robots";

const STEPS = ["connect", "wifi", "identity"] as const;
type Step = (typeof STEPS)[number];

function StepHeader({ step }: { step: Step }) {
  const index = STEPS.indexOf(step);
  return (
    <header className="flex items-center justify-between gap-3">
      <div>
        <div className="text-label uppercase tracking-[0.08em] text-text-dim">
          Bebop · Setup
        </div>
        <ol className="mt-1 flex items-center gap-1.5" aria-label="Setup progress">
          {STEPS.map((s, i) => (
            <li
              key={s}
              aria-current={i === index ? "step" : undefined}
              className={`h-1.5 w-8 rounded-full ${i <= index ? "bg-accent" : "bg-border"}`}
            >
              <span className="sr-only">{s}</span>
            </li>
          ))}
        </ol>
      </div>
      <Link
        to="/"
        className="inline-flex h-control-sm items-center rounded-card px-3 text-sm text-text-dim hover:text-text pointer-coarse:h-touch"
      >
        Cancel
      </Link>
    </header>
  );
}

function ConnectStep() {
  const navigate = useNavigate();
  const [params] = useSearchParams();
  const [host, setHost] = useState(params.get("host") ?? HOTSPOT_HOST);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    const h = normalizeHost(host);
    setError(null);
    if (!h || h === "0.0.0.0") {
      setError(`Enter the robot's address (${HOTSPOT_HOST} over its hotspot).`);
      return;
    }
    setBusy(true);
    const ok = await probe(h, "setup");
    setBusy(false);
    if (!ok) {
      setError(
        `No setup server answered at ${h}:${PORTS.setup}. Join the robot's Bebop-XXXX hotspot (or the same network as the robot) and try again.`,
      );
      return;
    }
    navigate(setupPath("wifi", h));
  }

  return (
    <form onSubmit={submit} className="flex flex-col gap-4">
      <h1 className="text-2xl font-bold">Connect to the robot</h1>
      <p className="leading-relaxed text-text-dim">
        Power on the robot. If it isn't on Wi-Fi yet it hosts a{" "}
        <strong>Bebop-XXXX</strong> hotspot: join it from this device's Wi-Fi
        settings and keep the address below. If it is already on your
        network, enter its address instead.
      </p>
      {error ? <Banner tone="error">{error}</Banner> : null}
      <Card>
        <Field label="Robot address" hint={`Hotspot gateway ${HOTSPOT_HOST}, or the robot's LAN IP / hostname.`}>
          <input
            className={INPUT_CLASS}
            autoFocus
            inputMode="url"
            autoComplete="off"
            autoCapitalize="none"
            autoCorrect="off"
            spellCheck={false}
            value={host}
            onChange={(e) => setHost(e.currentTarget.value)}
          />
        </Field>
      </Card>
      <Button type="submit" size="lg" loading={busy}>
        {busy ? "Checking…" : "Connect"}
      </Button>
    </form>
  );
}

function WifiStep() {
  const navigate = useNavigate();
  const [params] = useSearchParams();
  const host = params.get("host");
  const transport = useSetupTransport(host);
  if (!host || !transport) return <Navigate to={setupPath("connect")} replace />;
  const next = (ip: string) =>
    navigate(
      `${setupPath("identity", host)}${ip ? `&ip=${encodeURIComponent(ip)}` : ""}`,
    );
  return (
    <div className="flex flex-1 flex-col gap-3">
      <WifiScreen transport={transport} onDone={(status) => next(status.ipAddress)} />
      <Button variant="ghost" onClick={() => next("")}>
        Skip: keep the current network
      </Button>
    </div>
  );
}

function IdentityStep() {
  const navigate = useNavigate();
  const [params] = useSearchParams();
  const host = params.get("host");
  const ip = params.get("ip");
  const transport = useSetupTransport(host);
  if (!host || !transport) return <Navigate to={setupPath("connect")} replace />;
  // Where the operator UI lives after setup: the robot's new LAN address if
  // it joined a network, else the address we provisioned it on (the
  // hotspot gateway in Hosted mode).
  const target = ip || host;
  return (
    <div className="flex flex-1 flex-col gap-3">
      <ConfigScreen
        transport={transport}
        submitLabel="Save and open controls"
        onDone={(config) => {
          rememberRobot(target, config.robotName.trim() || undefined);
          navigate(robotPath(target, "drive"));
        }}
      />
      {ip && ip !== host ? (
        <p className="text-label leading-relaxed text-text-dim">
          The robot joined the network as <code className="text-text">{ip}</code>.
          Switch this device to the same network if the controls don't load.
        </p>
      ) : null}
    </div>
  );
}

function StepFrame({ step, children }: { step: Step; children: React.ReactNode }) {
  return (
    <div className="mx-auto flex min-h-full w-full max-w-[520px] flex-col gap-5 px-4 pt-[max(1.5rem,env(safe-area-inset-top))] pb-[max(1.5rem,env(safe-area-inset-bottom))] sm:px-6">
      <StepHeader step={step} />
      <div className="flex flex-1 flex-col">{children}</div>
    </div>
  );
}

export default function SetupFlow() {
  return (
    <Routes>
      <Route path="connect" element={<StepFrame step="connect"><ConnectStep /></StepFrame>} />
      <Route path="wifi" element={<StepFrame step="wifi"><WifiStep /></StepFrame>} />
      <Route path="identity" element={<StepFrame step="identity"><IdentityStep /></StepFrame>} />
      <Route path="*" element={<Navigate to="connect" replace />} />
    </Routes>
  );
}
