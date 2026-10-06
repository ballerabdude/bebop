import { useCallback, useEffect, useState } from "react";

import { httpUrl } from "../runtime/urls";
import { Banner, Button, Card } from "./ui";

/// A model backend as reported by `bebop-models` (:9094).
interface BackendView {
  id: string;
  purpose: string;
  kind: "container" | "external" | "tracked" | string;
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

const SUPERVISOR_PORT = 9094;
const POLL_MS = 3000;

/// GPU-budget + model-lifecycle view backed by the `bebop-models` supervisor
/// (`bebop-vision/bebop_vision/models_supervisor.py`). Shows what is resident
/// against the GPU budget and lets the operator load/unload backends. Loading a
/// backend evicts ready residents until it fits, so this is also the switch
/// between, say, the voice model and a future VLA model.
///
/// Renders nothing if the supervisor isn't reachable, so it stays out of the
/// way on robots that don't run it.
export function ModelSupervisorCard({ robotIp }: { robotIp: string }) {
  const [backends, setBackends] = useState<BackendView[]>([]);
  const [health, setHealth] = useState<SupervisorHealth | null>(null);
  const [reachable, setReachable] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    if (!robotIp) return;
    try {
      const [modelsRes, healthRes] = await Promise.all([
        fetch(httpUrl(robotIp, SUPERVISOR_PORT, "/models"), {
          signal: AbortSignal.timeout(2500),
        }),
        fetch(httpUrl(robotIp, SUPERVISOR_PORT, "/healthz"), {
          signal: AbortSignal.timeout(2500),
        }),
      ]);
      if (!modelsRes.ok || !healthRes.ok) throw new Error("bad response");
      const models = (await modelsRes.json()) as { backends: BackendView[] };
      setBackends(models.backends ?? []);
      setHealth((await healthRes.json()) as SupervisorHealth);
      setReachable(true);
      setError(null);
    } catch (e) {
      setReachable(false);
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [robotIp]);

  useEffect(() => {
    void refresh();
    const id = window.setInterval(() => void refresh(), POLL_MS);
    return () => window.clearInterval(id);
  }, [refresh]);

  const act = useCallback(
    async (id: string, op: "load" | "unload") => {
      setBusy(id);
      setError(null);
      try {
        // `load` blocks until the backend is ready (cold start can be long),
        // so don't apply a request timeout here.
        const res = await fetch(
          httpUrl(robotIp, SUPERVISOR_PORT, `/models/${id}/${op}`),
          { method: "POST" },
        );
        const body = (await res.json()) as { error?: string };
        if (!res.ok) throw new Error(body.error ?? `HTTP ${res.status}`);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setBusy(null);
        void refresh();
      }
    },
    [robotIp, refresh],
  );

  if (!reachable) return null;

  const resident = health?.resident_gb ?? 0;
  const budget = health?.budget_gb ?? 0;

  return (
    <Card>
      <div className="flex items-baseline justify-between">
        <h3 className="text-lg font-semibold">GPU budget</h3>
        <span className="text-sm text-text-dim">
          {resident.toFixed(1)} / {budget.toFixed(1)} GB resident
        </span>
      </div>
      <p className="mt-1 text-sm text-text-dim leading-relaxed">
        The supervisor keeps one large model resident at a time. Loading a
        backend evicts others until it fits the budget.
      </p>

      <div className="mt-3 flex flex-col gap-2">
        {backends.map((b) => {
          const tracked = b.kind === "tracked";
          const active = b.state === "ready" || b.state === "tracked";
          return (
            <div
              key={b.id}
              className="flex items-center justify-between gap-3 rounded border border-border px-3 py-2"
            >
              <div className="min-w-0">
                <div className="flex items-center gap-2">
                  <span className="font-medium truncate">{b.id}</span>
                  <span className="text-xs rounded bg-bg-elev-2 px-1.5 py-0.5 text-text-dim">
                    {b.purpose || b.kind}
                  </span>
                  <span
                    className={`text-xs ${active ? "text-accent" : "text-text-dim"}`}
                  >
                    {b.state}
                  </span>
                </div>
                {(b.detail || b.base_url) && (
                  <div className="text-xs text-text-dim truncate">
                    {b.detail || b.base_url}
                  </div>
                )}
              </div>
              <div className="flex shrink-0 items-center gap-2">
                <span className="text-xs text-text-dim">
                  {b.footprint_gb.toFixed(0)} GB
                </span>
                {!tracked && (
                  <Button
                    variant={active ? "ghost" : "secondary"}
                    disabled={busy === b.id}
                    onClick={() => void act(b.id, active ? "unload" : "load")}
                  >
                    {busy === b.id ? "…" : active ? "Unload" : "Load"}
                  </Button>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {error && <Banner tone="error">{error}</Banner>}
    </Card>
  );
}
