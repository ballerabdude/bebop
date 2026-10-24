// `#/` — pick a robot. Remembered robots are probed on :9090 (runtime) and
// :9091 (setup) in parallel so the operator sees at a glance whether a robot
// is drivable, only provisionable, or offline.

import { useState } from "react";
import { Link, useNavigate } from "react-router";

import {
  Banner,
  Button,
  Card,
  Field,
  INPUT_CLASS,
  StatusDot,
} from "../components/ui";
import type { Tone } from "../components/ui";
import { HOTSPOT_HOST, normalizeHost, PORTS } from "../robot/endpoints";
import { robotPath, setupPath } from "./paths";
import {
  forgetRobot,
  probe,
  rememberRobot,
  useRememberedRobots,
  useRobotProbe,
} from "./robots";
import type { ProbeResult, RememberedRobot } from "./robots";

function tone(p: ProbeResult): Tone {
  return p === "up" ? "ok" : p === "down" ? "err" : "dim";
}

function RobotRow({ robot, nonce }: { robot: RememberedRobot; nonce: number }) {
  const navigate = useNavigate();
  const { runtime, setup } = useRobotProbe(robot.host, nonce);
  const status =
    runtime === "up"
      ? "Online"
      : setup === "up"
        ? "Setup only (runtime not answering)"
        : runtime === "checking" || setup === "checking"
          ? "Checking…"
          : "Offline";
  return (
    <li className="flex flex-col gap-3 rounded-card border border-border bg-bg-elev px-4 py-3 sm:flex-row sm:items-center">
      <div className="min-w-0 flex-1">
        <div className="truncate font-semibold">{robot.name || robot.host}</div>
        <div className="mt-0.5 flex flex-wrap items-center gap-x-3 gap-y-1 text-label text-text-dim">
          {robot.name ? <span className="font-mono">{robot.host}</span> : null}
          <span className="inline-flex items-center gap-1.5">
            <StatusDot tone={tone(runtime)} label={`runtime ${runtime}`} pulse={runtime === "checking"} />
            runtime
          </span>
          <span className="inline-flex items-center gap-1.5">
            <StatusDot tone={tone(setup)} label={`setup ${setup}`} pulse={setup === "checking"} />
            setup
          </span>
          <span>{status}</span>
        </div>
      </div>
      <div className="flex gap-2">
        <Button
          className="flex-1 sm:flex-none"
          onClick={() => {
            rememberRobot(robot.host);
            navigate(robotPath(robot.host));
          }}
        >
          Open
        </Button>
        <Button
          variant="secondary"
          className="flex-1 sm:flex-none"
          disabled={setup !== "up"}
          onClick={() => navigate(setupPath("wifi", robot.host))}
        >
          Setup
        </Button>
        <Button
          variant="ghost"
          aria-label={`Forget ${robot.name || robot.host}`}
          onClick={() => forgetRobot(robot.host)}
        >
          Forget
        </Button>
      </div>
    </li>
  );
}

export function RobotPicker() {
  const navigate = useNavigate();
  const robots = useRememberedRobots();
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [nonce, setNonce] = useState(0);

  async function add(e: React.FormEvent) {
    e.preventDefault();
    const host = normalizeHost(draft);
    setError(null);
    if (!host) {
      setError("Enter the robot's IP address or hostname.");
      return;
    }
    if (host === "0.0.0.0") {
      setError(
        `0.0.0.0 is a bind address, not a destination. Over the robot's hotspot use ${HOTSPOT_HOST}.`,
      );
      return;
    }
    setBusy(true);
    const [runtime, setup] = await Promise.all([
      probe(host, "runtime"),
      probe(host, "setup"),
    ]);
    setBusy(false);
    if (!runtime && !setup) {
      setError(
        `No answer from ${host} on :${PORTS.runtime} or :${PORTS.setup}. Check the address, that the robot is on, and that this device is on the same network (or the robot's Bebop-XXXX hotspot).`,
      );
      return;
    }
    rememberRobot(host);
    setDraft("");
    navigate(runtime ? robotPath(host) : setupPath("wifi", host));
  }

  return (
    <div className="mx-auto flex w-full max-w-2xl flex-col gap-5 px-4 pt-[max(1.5rem,env(safe-area-inset-top))] pb-[max(1.5rem,env(safe-area-inset-bottom))] sm:px-6">
      <header className="flex items-center justify-between gap-3">
        <div>
          <div className="text-label uppercase tracking-[0.08em] text-text-dim">Bebop</div>
          <h1 className="text-2xl font-bold">Robots</h1>
        </div>
        <Button variant="ghost" onClick={() => setNonce((n) => n + 1)}>
          Recheck
        </Button>
      </header>

      {robots.length > 0 ? (
        <ul className="m-0 flex list-none flex-col gap-2 p-0">
          {robots.map((r) => (
            <RobotRow key={r.host} robot={r} nonce={nonce} />
          ))}
        </ul>
      ) : (
        <Card>
          <p className="text-sm leading-relaxed text-text-dim">
            No robots yet. Add one by address below, or set up a new robot over
            its hotspot.
          </p>
        </Card>
      )}

      <Card>
        <form onSubmit={add} className="flex flex-col gap-3">
          <Field
            label="Add a robot"
            hint={`LAN IP or hostname (e.g. bebop.local). Over the robot's own hotspot it is ${HOTSPOT_HOST}.`}
          >
            <div className="flex flex-col gap-2 sm:flex-row">
              <input
                className={INPUT_CLASS}
                inputMode="url"
                autoComplete="off"
                autoCapitalize="none"
                autoCorrect="off"
                spellCheck={false}
                placeholder="192.168.1.42 or bebop.local"
                value={draft}
                onChange={(e) => setDraft(e.currentTarget.value)}
              />
              <Button type="submit" loading={busy}>
                {busy ? "Checking…" : "Connect"}
              </Button>
            </div>
          </Field>
          {error ? <Banner tone="error">{error}</Banner> : null}
        </form>
      </Card>

      <Card className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <div className="font-semibold">Set up a new robot</div>
          <p className="mt-0.5 text-label leading-relaxed text-text-dim">
            Power it on and join its <strong>Bebop-XXXX</strong> hotspot from
            this device's Wi-Fi settings first.
          </p>
        </div>
        <Link
          to={setupPath("connect")}
          className="inline-flex h-control-md shrink-0 items-center justify-center rounded-card bg-bg-elev-2 px-4 text-sm font-semibold text-text hover:bg-border"
        >
          Start setup
        </Link>
      </Card>
    </div>
  );
}
