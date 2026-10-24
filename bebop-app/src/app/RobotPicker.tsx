// `#/` — pick a robot. Remembered robots (per browser) are probed on :9090
// (runtime) and :9091 (setup) in parallel, and again every 10 s, so the
// operator sees at a glance whether a robot is drivable, only provisionable,
// or offline. Robots can be renamed (a local label) or removed from the list.

import { useState } from "react";
import { Link, useNavigate } from "react-router";

import {
  Banner,
  Button,
  Card,
  Field,
  INPUT_CLASS,
  StatusDot,
  StatusPill,
  useConfirm,
} from "../components/ui";
import type { ConfirmOptions, Tone } from "../components/ui";
import { HOTSPOT_HOST, normalizeHost, PORTS } from "../robot/endpoints";
import { robotPath, setupPath } from "./paths";
import {
  forgetRobot,
  probe,
  rememberRobot,
  renameRobot,
  useRememberedRobots,
  useRobotProbe,
} from "./robots";
import type { ProbeResult, RememberedRobot } from "./robots";

const REPROBE_MS = 10_000;

function summary(runtime: ProbeResult, setup: ProbeResult): { tone: Tone; label: string; hint: string } {
  if (runtime === "up") return { tone: "ok", label: "Online", hint: "Runtime and controls reachable" };
  if (setup === "up") return { tone: "warn", label: "Setup only", hint: "Setup agent answers; the runtime doesn't" };
  if (runtime === "checking" || setup === "checking") return { tone: "dim", label: "Checking", hint: "Probing :9090 and :9091" };
  return { tone: "err", label: "Offline", hint: "No answer on :9090 or :9091" };
}

function RobotRow({
  robot,
  nonce,
  confirm,
}: {
  robot: RememberedRobot;
  nonce: number;
  confirm: (o: ConfirmOptions) => Promise<boolean>;
}) {
  const navigate = useNavigate();
  const { runtime, setup } = useRobotProbe(robot.host, nonce, REPROBE_MS);
  const st = summary(runtime, setup);
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(robot.name ?? "");
  const title = robot.name || robot.host;

  return (
    <li className="flex flex-col gap-3 rounded-card border border-border bg-bg-elev px-4 py-3">
      <div className="flex items-start gap-3">
        <span className="mt-1.5">
          <StatusDot tone={st.tone} pulse={st.label === "Checking"} label={st.label} />
        </span>
        <div className="min-w-0 flex-1">
          {editing ? (
            <form
              className="flex gap-2"
              onSubmit={(e) => {
                e.preventDefault();
                renameRobot(robot.host, draft);
                setEditing(false);
              }}
            >
              <input
                className={INPUT_CLASS}
                autoFocus
                value={draft}
                maxLength={40}
                placeholder={robot.host}
                aria-label={`Name for ${robot.host}`}
                onChange={(e) => setDraft(e.currentTarget.value)}
                onKeyDown={(e) => {
                  if (e.key === "Escape") setEditing(false);
                }}
              />
              <Button type="submit" size="sm">
                Save
              </Button>
            </form>
          ) : (
            <div className="truncate text-base font-semibold">{title}</div>
          )}
          <div className="mt-0.5 flex flex-wrap items-center gap-x-2 gap-y-1 text-label text-text-dim">
            {robot.name ? <span className="font-mono">{robot.host}</span> : null}
            <StatusPill tone={st.tone}>{st.label}</StatusPill>
            <span>{st.hint}</span>
          </div>
        </div>
      </div>
      <div className="grid grid-cols-3 gap-2 sm:flex sm:flex-wrap">
        <Button
          className="col-span-3 sm:min-w-28"
          variant={runtime === "up" ? "primary" : "secondary"}
          onClick={() => {
            rememberRobot(robot.host);
            navigate(robotPath(robot.host));
          }}
        >
          Open
        </Button>
        <Button
          variant="ghost"
          className="border border-border"
          disabled={setup !== "up"}
          onClick={() => navigate(setupPath("wifi", robot.host))}
        >
          Wi-Fi
        </Button>
        <Button
          variant="ghost"
          onClick={() => {
            setDraft(robot.name ?? "");
            setEditing((v) => !v);
          }}
        >
          {editing ? "Cancel" : "Rename"}
        </Button>
        <Button
          variant="ghost"
          className="text-danger-fg sm:ml-auto"
          onClick={async () => {
            const ok = await confirm({
              title: `Remove ${title}?`,
              body: "It's only removed from this device's list; the robot is not changed. Add it again by address any time.",
              confirmLabel: "Remove",
              tone: "danger",
            });
            if (ok) forgetRobot(robot.host);
          }}
        >
          Remove
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
  const { confirm, dialog } = useConfirm();

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
      {dialog}
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
            <RobotRow key={r.host} robot={r} nonce={nonce} confirm={confirm} />
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
