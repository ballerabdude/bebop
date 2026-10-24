// Settings › Models: the `bebop-models` supervisor (:9094,
// bebop-vision/bebop_vision/models_supervisor.py). Shows what is resident
// against the GPU budget and loads/unloads Edge-LLM backends. Loading one
// evicts ready residents until it fits, so this is also the switch between,
// say, the voice model and a future VLA.
//
// Unlike the old ModelSupervisorCard it never hides itself: an unreachable
// supervisor is shown as such (with the CORS hint for old robot builds).

import { useCallback, useEffect, useState } from "react";

import {
  Banner,
  Button,
  Card,
  CardHeader,
  StatusPill,
  useConfirm,
} from "../../components/ui";
import type { Tone } from "../../components/ui";
import { PORTS } from "../../robot/endpoints";
import { useRobot } from "../../robot/session";
import { predictEvictions } from "./modelEvictions";

interface BackendView {
  id: string;
  purpose: string;
  kind: "edgellm" | "external" | "tracked" | string;
  state: string;
  detail: string;
  footprint_gb: number;
  base_url: string | null;
}

interface SupervisorHealth {
  ok: boolean;
  budget_gb: number;
  resident_gb: number;
  mem?: Record<string, number>;
}

const POLL_MS = 3_000;

function stateTone(state: string): Tone {
  if (state === "ready" || state === "tracked") return "ok";
  if (state === "starting" || state === "building" || state === "loading") return "warn";
  if (state === "failed" || state === "error") return "err";
  return "dim";
}

export function ModelSupervisorPanel() {
  const { endpoints } = useRobot();
  const [backends, setBackends] = useState<BackendView[]>([]);
  const [health, setHealth] = useState<SupervisorHealth | null>(null);
  const [reachable, setReachable] = useState<boolean | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const [modelsRes, healthRes] = await Promise.all([
        fetch(endpoints.models("/models"), { cache: "no-store", signal: AbortSignal.timeout(2500) }),
        fetch(endpoints.models("/healthz"), { cache: "no-store", signal: AbortSignal.timeout(2500) }),
      ]);
      if (!modelsRes.ok || !healthRes.ok) {
        throw new Error(`HTTP ${modelsRes.ok ? healthRes.status : modelsRes.status}`);
      }
      const models = (await modelsRes.json()) as { backends?: BackendView[] };
      setBackends(models.backends ?? []);
      setHealth((await healthRes.json()) as SupervisorHealth);
      setReachable(true);
    } catch {
      setReachable(false);
    }
  }, [endpoints]);

  useEffect(() => {
    let alive = true;
    let timer = 0;
    const loop = async () => {
      await refresh();
      if (alive) timer = window.setTimeout(() => void loop(), POLL_MS);
    };
    void loop();
    return () => {
      alive = false;
      window.clearTimeout(timer);
    };
  }, [refresh]);

  const act = useCallback(
    async (id: string, op: "load" | "unload") => {
      setBusy(id);
      setError(null);
      try {
        // `load` can block a long time (an Edge-LLM backend's first start
        // builds its TensorRT engines), so no request timeout here.
        const res = await fetch(endpoints.models(`/models/${encodeURIComponent(id)}/${op}`), {
          method: "POST",
        });
        const body = (await res.json().catch(() => ({}))) as { error?: string };
        if (!res.ok) throw new Error(body.error ?? `HTTP ${res.status}`);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setBusy(null);
        void refresh();
      }
    },
    [endpoints, refresh],
  );

  const { confirm, dialog } = useConfirm();

  /// Load, but first name any running backend the supervisor would stop to
  /// fit the budget (a voice turn in progress would end).
  const load = useCallback(
    async (id: string) => {
      const evicted = health
        ? predictEvictions(backends, id, health.budget_gb, health.mem?.MemAvailable)
        : [];
      if (evicted.length > 0) {
        const names = evicted.map((b) => b.id).join(", ");
        const ok = await confirm({
          title: `Load ${id}?`,
          body: (
            <p>
              It doesn't fit next to what is running, so the supervisor will
              unload <strong className="text-text">{names}</strong> first. Anything
              using {evicted.length > 1 ? "them" : "it"} (a voice conversation,
              for example) stops.
            </p>
          ),
          confirmLabel: `Unload ${evicted.length > 1 ? `${evicted.length} models` : names} and load`,
        });
        if (!ok) return;
      }
      await act(id, "load");
    },
    [health, backends, confirm, act],
  );

  const resident = health?.resident_gb ?? 0;
  const budget = health?.budget_gb ?? 0;
  const pct = budget > 0 ? Math.min(100, (resident / budget) * 100) : 0;

  return (
    <Card className="flex flex-col gap-3">
      <CardHeader
        eyebrow="GPU budget"
        title="Model supervisor"
        description="Every generative model runs as a TensorRT Edge-LLM backend. Loading one evicts others until it fits the budget."
        actions={
          <StatusPill
            tone={reachable === null ? "dim" : reachable ? "ok" : "err"}
            dot
            pulse={reachable === null}
          >
            {reachable === null ? "checking" : reachable ? "online" : "unreachable"}
          </StatusPill>
        }
      />

      {reachable === false ? (
        <Banner tone="warning" title={`bebop-models isn't answering on :${PORTS.models}`}>
          The service may be stopped or not installed
          (<code>install-jetson.sh --setup-models</code>). Robot builds from
          before 2026-10-23 also lack CORS on this port, which the browser
          reports the same way.
        </Banner>
      ) : null}

      {reachable ? (
        <>
          <div className="flex flex-col gap-1.5">
            <div className="flex items-baseline justify-between text-label text-text-dim">
              <span>Resident</span>
              <span className="font-mono tabular-nums text-text">
                {resident.toFixed(1)} / {budget.toFixed(0)} GB
              </span>
            </div>
            <div
              className="h-2 overflow-hidden rounded-full bg-bg-elev-2"
              role="meter"
              aria-label="GPU memory resident"
              aria-valuemin={0}
              aria-valuemax={budget}
              aria-valuenow={resident}
            >
              <div
                className={pct > 90 ? "h-full bg-warning" : "h-full bg-accent"}
                style={{ width: `${pct}%` }}
              />
            </div>
          </div>

          {backends.length === 0 ? (
            <p className="text-sm text-text-dim">No backends registered.</p>
          ) : (
            <ul className="m-0 flex list-none flex-col gap-2 p-0">
              {backends.map((b) => {
                const tracked = b.kind === "tracked";
                const active = b.state === "ready" || b.state === "tracked";
                return (
                  <li
                    key={b.id}
                    className="flex flex-col gap-2 rounded-card border border-border bg-bg-elev-2/40 px-3 py-2.5 sm:flex-row sm:items-center"
                  >
                    <div className="min-w-0 flex-1">
                      <div className="flex flex-wrap items-center gap-2">
                        <span className="truncate font-semibold">{b.id}</span>
                        <StatusPill tone={stateTone(b.state)}>{b.state}</StatusPill>
                        <span className="text-label text-text-dim">
                          {b.purpose || b.kind} · {b.footprint_gb.toFixed(0)} GB
                        </span>
                      </div>
                      {b.detail || b.base_url ? (
                        <div className="mt-0.5 truncate font-mono text-label text-text-dim">
                          {b.detail || b.base_url}
                        </div>
                      ) : null}
                    </div>
                    {tracked ? (
                      <span className="text-label text-text-dim">managed by its service</span>
                    ) : active ? (
                      <Button
                        variant="secondary"
                        size="sm"
                        loading={busy === b.id}
                        disabled={busy !== null}
                        onClick={() => void act(b.id, "unload")}
                      >
                        Unload
                      </Button>
                    ) : (
                      <Button
                        variant="primary"
                        size="sm"
                        loading={busy === b.id}
                        disabled={busy !== null}
                        onClick={() => void load(b.id)}
                      >
                        Load
                      </Button>
                    )}
                  </li>
                );
              })}
            </ul>
          )}
        </>
      ) : null}

      {error ? <Banner tone="error" title="Request failed">{error}</Banner> : null}
      {dialog}
    </Card>
  );
}
