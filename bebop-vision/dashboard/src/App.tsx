import { useEffect, useState } from "react";
import { api, type ModelInfo, type SessionInfo } from "./api/client";
import ModelValidationPage from "./pages/ModelValidationPage";
import PipelinePage from "./pages/PipelinePage";
import TickReviewPage from "./pages/TickReviewPage";
import ShortcutsModal from "./components/ShortcutsModal";
import { Badge, Kbd, Toasts, cn } from "./components/ui";

type Tab = "review" | "validate" | "pipeline";

export default function App() {
  const [tab, setTab] = useState<Tab>("review");
  const [sessions, setSessions] = useState<SessionInfo[]>([]);
  const [session, setSession] = useState<string | null>(null);
  const [filter, setFilter] = useState("");
  const [refreshKey, setRefreshKey] = useState(0);
  const [lastStamp, setLastStamp] = useState("");
  const [model, setModel] = useState<ModelInfo | null>(null);
  const [showKeys, setShowKeys] = useState(false);

  useEffect(() => {
    api.sessions().then((r) => {
      setSessions(r.sessions);
      if (!session && r.sessions.length) setSession(r.sessions[0].name);
    }).catch(() => setSessions([]));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [refreshKey]);

  useEffect(() => {
    api.model().then(setModel).catch(() => setModel(null));
  }, []);

  // global "?" opens the shortcuts modal
  useEffect(() => {
    const h = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement | null)?.tagName;
      if (tag && ["SELECT", "INPUT", "TEXTAREA"].includes(tag)) return;
      if (e.key === "?") setShowKeys(true);
    };
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, []);

  const shown = sessions.filter((s) =>
    s.name.toLowerCase().includes(filter.toLowerCase()),
  );

  return (
    <div className="flex h-screen">
      {/* sidebar */}
      <aside className="flex w-64 shrink-0 flex-col border-r border-white/[0.06] bg-zinc-900/40">
        <div className="flex items-center gap-2 px-4 pb-2 pt-4">
          <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-indigo-600 font-bold text-white">
            b
          </span>
          <div>
            <div className="text-sm font-semibold text-zinc-100">navd review</div>
            <div className="text-[10px] text-zinc-500">dataset + model QA</div>
          </div>
        </div>

        <nav className="mt-2 space-y-0.5 px-2">
          {(
            [
              ["review", "Tick review", "Review and correct teacher labels"],
              ["validate", "Model validation", "Replay the ONNX vs teacher"],
              ["pipeline", "Pipeline", "How data flows to the model"],
            ] as const
          ).map(([k, label, title]) => (
            <button
              key={k}
              title={title}
              onClick={() => setTab(k)}
              className={cn(
                "w-full rounded-lg px-3 py-2 text-left text-[13px] font-medium transition-colors",
                tab === k
                  ? "bg-zinc-800 text-white"
                  : "text-zinc-400 hover:bg-zinc-800/60 hover:text-zinc-200",
              )}
            >
              {label}
            </button>
          ))}
        </nav>

        <div className="mt-5 min-h-0 flex-1 overflow-y-auto px-2">
          <div className="flex items-center justify-between px-2 pb-1.5">
            <span className="text-[10px] font-semibold uppercase tracking-wider text-zinc-600">
              sessions
            </span>
            <span className="text-[10px] text-zinc-600">{shown.length}</span>
          </div>
          <input
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            placeholder="filter…"
            className="mb-2 w-full rounded-lg border border-white/10 bg-zinc-800/80 px-2.5 py-1.5 text-[11px] text-zinc-300 placeholder:text-zinc-600"
          />
          <div className="space-y-1 pb-4">
            {shown.map((s) => (
              <button
                key={s.name}
                onClick={() => setSession(s.name)}
                className={cn(
                  "w-full rounded-lg px-2.5 py-2 text-left transition-colors",
                  session === s.name
                    ? "bg-indigo-950/60 ring-1 ring-inset ring-indigo-800"
                    : "hover:bg-zinc-800/60",
                )}
              >
                <div className="flex items-center justify-between gap-2">
                  <span className="truncate font-mono text-[11px] text-zinc-300">
                    {s.name.replace("navd_session_", "")}
                  </span>
                  <span className="shrink-0 text-[10px] text-zinc-600">{s.ticks}</span>
                </div>
                <div className="mt-1.5">
                  <div className="h-1 w-full overflow-hidden rounded-full bg-zinc-800">
                    <div
                      className="h-full rounded-full bg-emerald-500"
                      style={{ width: `${s.ticks ? (100 * s.hand) / s.ticks : 0}%` }}
                    />
                  </div>
                </div>
                <div className="mt-1 flex items-center gap-1.5 text-[10px] text-zinc-600">
                  <span>{s.hand} hand</span>
                  {s.legacy && <Badge tone="amber">legacy</Badge>}
                </div>
              </button>
            ))}
            {!shown.length && (
              <p className="px-2 py-4 text-[11px] text-zinc-600">
                No sessions under datasets/navd-v0.
              </p>
            )}
          </div>
        </div>

        <footer className="border-t border-white/[0.06] p-3">
          <button
            onClick={() => setTab("validate")}
            className="w-full rounded-lg bg-zinc-900 px-3 py-2 text-left ring-1 ring-inset ring-white/[0.06] hover:bg-zinc-800"
          >
            <div className="text-[10px] uppercase tracking-wider text-zinc-600">model</div>
            {model?.loaded ? (
              <div className="mt-0.5 truncate text-[11px] text-emerald-400">
                ● {model.path?.split("/").pop()} [{model.providers?.[0]?.replace("ExecutionProvider", "")}]
              </div>
            ) : (
              <div className="mt-0.5 text-[11px] text-amber-400">
                ○ not loaded — open validation to load
              </div>
            )}
          </button>
          <div className="mt-2 flex items-center justify-between text-[10px] text-zinc-600">
            <span>
              shortcuts <Kbd>?</Kbd>
            </span>
            <span>:8099</span>
          </div>
        </footer>
      </aside>

      {/* main */}
      <main className="min-w-0 flex-1 overflow-hidden">
        {tab === "review" && (
          <TickReviewPage
            session={session}
            onHandChange={() => setRefreshKey((k) => k + 1)}
            onStampViewed={setLastStamp}
          />
        )}
        {tab === "validate" && (
          <ModelValidationPage session={session} lastStamp={lastStamp} />
        )}
        {tab === "pipeline" && <PipelinePage />}
      </main>

      <ShortcutsModal open={showKeys} onClose={() => setShowKeys(false)} />
      <Toasts />
    </div>
  );
}
