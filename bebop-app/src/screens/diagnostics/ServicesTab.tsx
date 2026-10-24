// Diagnostics › Services: reachability of every robot service, the systemd
// units the firmware controls (vision, voice) and the runtime restart.

import { useCallback, useEffect, useState } from "react";

import {
  Banner,
  Button,
  Card,
  CardHeader,
  StatusDot,
  StatusPill,
  useConfirm,
} from "../../components/ui";
import type { Tone } from "../../components/ui";
import { PORTS } from "../../robot/endpoints";
import type { ServiceName } from "../../robot/endpoints";
import { useConnection, useRobot, useRuntimeAction, useTelemetry } from "../../robot/session";

const SERVICES: { id: ServiceName; label: string; what: string }[] = [
  { id: "runtime", label: "Runtime", what: "bebop-linux: drive, telemetry, captures" },
  { id: "setup", label: "Setup agent", what: "bebop-agent: Wi-Fi and hotspot" },
  { id: "video", label: "Video", what: "bebop-vision WHEP streams" },
  { id: "voice", label: "Voice", what: "bebop-voice: agent + speech" },
  { id: "models", label: "Models", what: "bebop-models GPU supervisor" },
  { id: "places", label: "Places", what: "place memory (in bebop-vision)" },
  { id: "slam", label: "Live SLAM", what: "only with --live-slam" },
];

type Probe = "up" | "down" | "checking";

/// Reachability only: `no-cors` resolves for any HTTP answer even when the
/// service sends no CORS headers (the models supervisor), and rejects when
/// nothing listens.
async function reachable(url: string): Promise<boolean> {
  try {
    await fetch(url, {
      mode: "no-cors",
      cache: "no-store",
      signal: AbortSignal.timeout(2500),
    });
    return true;
  } catch {
    return false;
  }
}

function ServiceHealth() {
  const { endpoints } = useRobot();
  const [state, setState] = useState<Record<string, Probe>>({});
  const check = useCallback(() => {
    setState(Object.fromEntries(SERVICES.map((s) => [s.id, "checking" as Probe])));
    for (const s of SERVICES) {
      void reachable(endpoints.healthz(s.id)).then((ok) =>
        setState((prev) => ({ ...prev, [s.id]: ok ? "up" : "down" })),
      );
    }
  }, [endpoints]);
  useEffect(() => {
    check();
    const id = window.setInterval(check, 10_000);
    return () => window.clearInterval(id);
  }, [check]);
  const tone = (p: Probe | undefined): Tone =>
    p === "up" ? "ok" : p === "down" ? "err" : "dim";
  return (
    <Card className="space-y-2">
      <CardHeader
        eyebrow="Services"
        title="Reachability"
        actions={
          <Button variant="secondary" size="sm" onClick={check}>
            Recheck
          </Button>
        }
      />
      <ul className="m-0 grid list-none grid-cols-1 gap-1 p-0 sm:grid-cols-2">
        {SERVICES.map((s) => (
          <li key={s.id} className="flex items-center gap-2.5 rounded-input px-1 py-1.5">
            <StatusDot
              tone={tone(state[s.id])}
              pulse={state[s.id] === "checking"}
              label={`${s.label} ${state[s.id] ?? "unknown"}`}
            />
            <div className="min-w-0">
              <div className="text-sm font-medium">
                {s.label} <span className="font-mono text-caption text-text-dim">:{PORTS[s.id]}</span>
              </div>
              <div className="truncate text-caption text-text-dim">{s.what}</div>
            </div>
          </li>
        ))}
      </ul>
    </Card>
  );
}

function UnitCard({
  name,
  unit,
  onToggle,
  busy,
}: {
  name: string;
  unit: { present: boolean; running: boolean; state: string; detail: string; service: string };
  onToggle: () => void;
  busy: boolean;
}) {
  return (
    <Card className="space-y-3">
      <CardHeader
        eyebrow={name}
        title={<span className="font-mono">{unit.service || "—"}</span>}
        actions={
          <StatusPill tone={unit.running ? "ok" : unit.state === "failed" ? "err" : "dim"} dot>
            {unit.running ? "running" : unit.state || "stopped"}
          </StatusPill>
        }
      />
      {unit.detail ? (
        <div className="break-words font-mono text-caption text-text-dim">{unit.detail}</div>
      ) : null}
      <Button
        variant={unit.running ? "secondary" : "success"}
        size="sm"
        onClick={onToggle}
        loading={busy}
        disabled={!unit.present}
      >
        {unit.running ? `Stop ${name.toLowerCase()}` : `Start ${name.toLowerCase()}`}
      </Button>
    </Card>
  );
}

export function ServicesTab() {
  const link = useConnection();
  const units = useTelemetry((s) => ({ vision: s?.vision ?? null, voice: s?.voice ?? null }), 5);
  const { busy, error, run } = useRuntimeAction();
  const { confirm, dialog } = useConfirm();

  async function restartRuntime() {
    const ok = await confirm({
      title: "Restart the robot runtime?",
      confirmLabel: "Restart",
      tone: "danger",
      body: (
        <>
          <p>
            <span className="font-mono text-text">bebop-linux.service</span> stops
            and starts again. Telemetry goes stale for a few seconds and the
            link drops, then reconnects on its own.
          </p>
          <p>
            Use it when the runtime is wedged, most often a dead CAN socket
            after the USB-CAN hub was unplugged. The firmware refuses while
            anything is armed.
          </p>
        </>
      ),
    });
    if (ok) await run("restart", (t) => t.restartRuntime());
  }

  return (
    <div className="flex flex-col gap-3">
      {error ? <Banner tone="error">{error}</Banner> : null}
      <ServiceHealth />
      <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
        {units.vision?.present ? (
          <UnitCard
            name="Vision"
            unit={units.vision}
            busy={busy === "vision"}
            onToggle={() => {
              const next = !units.vision!.running;
              void run("vision", (t) => t.setVisionEnabled(next));
            }}
          />
        ) : null}
        {units.voice?.present ? (
          <UnitCard
            name="Voice"
            unit={units.voice}
            busy={busy === "voice"}
            onToggle={() => {
              const next = !units.voice!.running;
              void run("voice", (t) => t.setVoiceEnabled(next));
            }}
          />
        ) : null}
      </div>
      <Card className="space-y-3">
        <CardHeader
          eyebrow="Runtime"
          title={<span className="font-mono">bebop-linux.service</span>}
          description="Restart the firmware runtime when it is wedged. Disarm first."
          actions={
            <StatusPill tone={link.state === "connected" ? "ok" : "warn"} dot>
              {link.state}
            </StatusPill>
          }
        />
        <Button
          variant="danger"
          size="sm"
          onClick={() => void restartRuntime()}
          loading={busy === "restart"}
          disabled={!!busy || link.state === "disconnected"}
        >
          Restart runtime
        </Button>
      </Card>
      {dialog}
    </div>
  );
}
