import { useCallback, useEffect, useRef, useState } from "react";
import {
  api,
  type Grid,
  type ModelInfo,
  type SweepResults,
  type SweepStatus,
  type TickPayload,
  type TickSummary,
  type ValidateTick,
} from "../api/client";
import { heatColor } from "../api/grid";
import BevCanvas from "../components/BevCanvas";
import TickFilmstrip from "../components/TickFilmstrip";
import {
  Badge,
  Button,
  Card,
  Img,
  Segmented,
  Spinner,
  cn,
  toast,
} from "../components/ui";

type Show = "model" | "label" | "agree" | "disagree" | "prob";

/** Pause between ticks while playing, by speed setting. Inference itself
 * (~4 ms GPU / ~200 ms CPU) is on top of this. */
const PAUSE_MS = { 1: 1600, 2: 800, 4: 400 } as const;
type Speed = keyof typeof PAUSE_MS;

/** Model validation: replay the exported ONNX over recorded ticks with the
 * exact runtime preprocessing path. Play/pause/step through the session
 * like a video — every tick shown is evaluated against the effective
 * teacher (hand > fused). Gate-rejected ticks still render: the reason
 * the drive node would hold is the thing being debugged. */
export default function ModelValidationPage({
  session,
  lastStamp,
}: {
  session: string | null;
  lastStamp: string;
}) {
  const [model, setModel] = useState<ModelInfo | null>(null);
  const [modelPath, setModelPath] = useState("weights/navd.onnx");
  const [loadingModel, setLoadingModel] = useState(false);

  const [ticks, setTicks] = useState<TickSummary[]>([]);
  const [idx, setIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState<Speed>(1);
  const [vt, setVt] = useState<ValidateTick | null>(null);
  const [running, setRunning] = useState(false);
  const [probCh, setProbCh] = useState(1);
  const [show, setShow] = useState<Show>("model");
  const [err, setErr] = useState("");
  const [payload, setPayload] = useState<TickPayload | null>(null);
  const mediaSeq = useRef(0);

  const [results, setResults] = useState<SweepResults | null>(null);
  const [resultsFiles, setResultsFiles] = useState<string[]>([]);
  const [sweep, setSweep] = useState<SweepStatus | null>(null);

  const reqSeq = useRef(0);

  useEffect(() => {
    api.model().then(setModel).catch(() => setModel(null));
  }, []);

  // session -> ticks; land on the tick last viewed in review if known
  useEffect(() => {
    if (!session) return;
    setPlaying(false);
    setVt(null);
    api
      .ticks(session)
      .then((t) => {
        const list = Array.isArray(t) ? t : [];
        setTicks(list);
        const hit = lastStamp ? list.findIndex((x) => x.stamp === lastStamp) : -1;
        setIdx(hit >= 0 ? hit : 0);
      })
      .catch((e) => toast(`tick list failed: ${e}`, "error"));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [session]);

  // evaluate the current tick whenever it changes
  useEffect(() => {
    if (!session || !ticks[idx]) return;
    const my = ++reqSeq.current;
    setRunning(true);
    setErr("");
    api
      .validateTick(session, ticks[idx].stamp)
      .then((v) => {
        if (my !== reqSeq.current) return;
        setVt(v);
        setErr("");
      })
      .catch((e) => {
        if (my !== reqSeq.current) return;
        setVt(null);
        setErr(String(e));
      })
      .finally(() => {
        if (my === reqSeq.current) setRunning(false);
      });
  }, [session, idx, ticks]);

  // media for the current tick (grids skipped — heavy, not needed here)
  useEffect(() => {
    if (!session || !ticks[idx]) return;
    const my = ++mediaSeq.current;
    setPayload(null);
    api
      .tick(session, ticks[idx].stamp, { grids: false })
      .then((p) => {
        if (my === mediaSeq.current) setPayload(p);
      })
      .catch(() => void 0);
  }, [session, idx, ticks]);

  // playback loop: advance while playing, stop at the end
  useEffect(() => {
    if (!playing) return;
    if (idx >= ticks.length - 1) {
      setPlaying(false);
      return;
    }
    const t = setTimeout(() => {
      setIdx((i) => Math.min(i + 1, ticks.length - 1));
    }, PAUSE_MS[speed]);
    return () => clearTimeout(t);
  }, [playing, idx, ticks, speed]);

  const togglePlay = useCallback(() => {
    if (!ticks.length) return;
    if (!playing && idx >= ticks.length - 1) setIdx(0); // replay from start
    setPlaying((p) => !p);
  }, [playing, idx, ticks.length]);

  const skip = useCallback(
    (d: number) => {
      setPlaying(false);
      setIdx((i) => Math.min(Math.max(i + d, 0), ticks.length - 1));
    },
    [ticks.length],
  );

  const jumpStamp = useCallback(
    (stamp: string) => {
      const i = ticks.findIndex((t) => t.stamp === stamp);
      if (i >= 0) {
        setPlaying(false);
        setIdx(i);
      }
    },
    [ticks],
  );

  const poll = useCallback(async () => {
    if (!session) return null;
    const s = await api.sweepStatus(session);
    setSweep(s);
    if (s.results_files) setResultsFiles(s.results_files);
    return s;
  }, [session]);

  useEffect(() => {
    if (!session) return;
    void poll();
  }, [session, poll]);

  const runSweep = async () => {
    if (!session) return;
    try {
      await api.runSweep(session, modelPath);
      toast("session sweep started", "info");
      // poll until done
      const iv = setInterval(async () => {
        try {
          const s = await poll();
          if (s && s.running === false) clearInterval(iv);
        } catch {
          clearInterval(iv);
        }
      }, 1000);
    } catch (e) {
      setErr(String(e));
    }
  };

  const loadResults = async (file: string) => {
    if (!session) return;
    try {
      setResults(await api.sweepResults(session, file));
    } catch (e) {
      setErr(String(e));
    }
  };

  const loadModel = async () => {
    setLoadingModel(true);
    try {
      const t0 = performance.now();
      await api.loadModel(modelPath);
      setModel(await api.model());
      toast(
        `model loaded (${((performance.now() - t0) / 1000).toFixed(1)}s, GPU warmup paid)`,
        "success",
      );
    } catch (e) {
      toast(`load failed: ${e}`, "error");
    } finally {
      setLoadingModel(false);
    }
  };

  const cur = ticks[idx];

  // space / arrows transport (when not typing)
  useEffect(() => {
    const h = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement | null)?.tagName;
      if (tag && ["SELECT", "INPUT", "TEXTAREA"].includes(tag)) return;
      if (e.key === " ") {
        e.preventDefault();
        togglePlay();
      } else if (e.key === "ArrowRight") {
        e.preventDefault();
        skip(1);
      } else if (e.key === "ArrowLeft") {
        e.preventDefault();
        skip(-1);
      }
    };
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, [togglePlay, skip]);

  if (!session) {
    return (
      <Empty
        title="No session selected"
        body="Pick a session in the sidebar — it is shared between review and validation."
      />
    );
  }

  return (
    <div className="mx-auto max-w-6xl space-y-4 p-4">
      {err && (
        <div className="rounded-lg bg-red-950/70 px-3.5 py-2.5 text-xs text-red-300 ring-1 ring-inset ring-red-800">
          {err}
        </div>
      )}

      {/* model loader */}
      <Card
        title="model"
        actions={
          model?.loaded ? (
            <Badge tone={model.providers?.[0] === "CUDAExecutionProvider" ? "green" : "amber"}>
              {model.providers?.[0]?.replace("ExecutionProvider", "") ?? "?"}
            </Badge>
          ) : (
            <Badge tone="amber">not loaded</Badge>
          )
        }
      >
        <div className="flex flex-wrap items-center gap-2">
          <input
            value={modelPath}
            onChange={(e) => setModelPath(e.target.value)}
            className="w-72 rounded-lg border border-white/10 bg-zinc-800 px-2.5 py-1.5 font-mono text-xs text-zinc-200"
            placeholder="weights/navd.onnx"
          />
          <Button variant="primary" disabled={loadingModel} onClick={() => void loadModel()}>
            {loadingModel ? "loading…" : "load"}
          </Button>
          {model?.loaded && (
            <span className="truncate font-mono text-[11px] text-zinc-500">{model.path}</span>
          )}
        </div>
      </Card>

      {/* player */}
      <Card
        title="replay"
        actions={
          vt ? (
            <Segmented
              size="sm"
              value={show}
              onChange={setShow}
              options={[
                { value: "model", label: "model" },
                { value: "label", label: "label" },
                { value: "disagree", label: "disagree" },
                { value: "prob", label: "P(navigable)" },
              ]}
            />
          ) : null
        }
      >
        {/* transport bar */}
        <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
          <div className="flex items-center gap-1">
            <Button onClick={() => skip(-1)} disabled={!ticks.length} title="previous tick (←)">
              ⏮
            </Button>
            <Button
              variant="primary"
              onClick={togglePlay}
              disabled={!ticks.length || !model?.loaded}
              title="play / pause (space)"
              className="!px-3"
            >
              {playing ? "❚❚ pause" : "▶ play"}
            </Button>
            <Button onClick={() => skip(1)} disabled={!ticks.length} title="next tick (→)">
              ⏭
            </Button>
          </div>
          <span className="font-mono text-xs text-zinc-400">
            {ticks.length ? `${idx + 1} / ${ticks.length}` : "—"}
            {running && <Spinner className="ml-2 inline align-[-2px]" />}
          </span>
          <Segmented
            size="sm"
            value={speed}
            onChange={(v) => setSpeed(v)}
            options={[
              { value: 1 as Speed, label: "0.5×" },
              { value: 2 as Speed, label: "1×" },
              { value: 4 as Speed, label: "2×" },
            ]}
          />
          {cur && (
            <span className="ml-auto truncate font-mono text-[11px] text-zinc-600">
              {cur.stamp}
            </span>
          )}
        </div>

        <div className="mt-2.5">
          <TickFilmstrip ticks={ticks} idx={idx} onSelect={(i) => { setPlaying(false); setIdx(i); }} />
        </div>

        {vt && (
          <div className="mt-4 grid grid-cols-1 gap-4 xl:grid-cols-[280px_auto_1fr]">
            {/* camera views */}
            <div className="space-y-2.5">
              <div className="text-[11px] uppercase tracking-wider text-zinc-500">
                what the robot saw
              </div>
              <Img b64={payload?.color_near} mime="image/jpeg" label="near color" />
              <Img b64={payload?.depth_near} mime="image/png" label="depth near" />
              <Img b64={payload?.color_far} mime="image/jpeg" label="far color" />
              <Img b64={payload?.depth_far} mime="image/png" label="depth far" />
            </div>

            {/* grid view */}
            <div className="space-y-2">
              {show === "prob" ? (
                <ProbHeat grid={vt.prob[probCh]} cell={10} />
              ) : (
                <BevCanvas
                  grid={
                    show === "model"
                      ? vt.model
                      : show === "label"
                        ? (vt.label ?? vt.model)
                        : vt.model
                  }
                  diffGrid={show === "disagree" ? (vt.agreement ?? null) : null}
                  cellPx={10}
                />
              )}
              {show === "disagree" && (
                <p className="max-w-[600px] text-[11px] leading-relaxed text-zinc-500">
                  White outlines: model ≠ label. Fill color shows the{" "}
                  <b>model's</b> class at the disagreeing cell (red = blocked,
                  green = navigable, amber = caution).
                </p>
              )}
              {show === "prob" && (
                <>
                  <Segmented
                    size="sm"
                    value={probCh}
                    onChange={setProbCh}
                    options={[
                      { value: 0, label: "P(blocked)" },
                      { value: 1, label: "P(navigable)" },
                      { value: 2, label: "P(caution)" },
                    ]}
                  />
                  <div className="h-2 w-[600px] rounded bg-gradient-to-r from-blue-500 via-emerald-500 to-red-500" />
                </>
              )}
            </div>

            {/* metrics column */}
            <div className="space-y-3">
              <div className="grid grid-cols-2 gap-2.5">
                <Metric
                  label="agreement"
                  value={vt.agreement_pct !== undefined ? `${vt.agreement_pct}%` : "—"}
                  tone={vt.agreement_pct === undefined || vt.agreement_pct < 80 ? "amber" : "green"}
                />
                <Metric label="frac navigable" value={vt.frac_navigable.toFixed(2)} />
                <Metric label="latency" value={`${vt.latency_ms} ms`} />
                <Metric label="label key" value={vt.label_key ?? "—"} />
              </div>
              {vt.gate_rejected && (
                <div className="rounded-lg bg-red-950/70 px-3 py-2 text-xs text-red-300 ring-1 ring-inset ring-red-800">
                  <b>RUNTIME WOULD HOLD.</b> {vt.gate_reason}
                </div>
              )}
              <div>
                <div className="mb-1.5 text-[11px] uppercase tracking-wider text-zinc-500">
                  goal raster the model received
                </div>
                <GoalRaster grid={vt.goal_raster} />
              </div>
            </div>
          </div>
        )}
        {!vt && !running && model?.loaded && (
          <p className="mt-3 text-xs text-zinc-600">
            Press play — every tick is run through the model with the robot's
            exact preprocessing and scored against its teacher label.
          </p>
        )}
        {!model?.loaded && (
          <p className="mt-3 text-xs text-zinc-600">
            Load a model above to start replaying.
          </p>
        )}
      </Card>

      {/* session sweep */}
      <Card
        title="session sweep"
        actions={
          sweep?.running ? (
            <span className="flex items-center gap-2 text-[11px] text-amber-300">
              <Spinner /> {sweep.done}/{sweep.total ?? "…"}
            </span>
          ) : null
        }
      >
        <div className="flex flex-wrap items-center gap-2">
          <Button
            variant="primary"
            onClick={() => void runSweep()}
            disabled={!model?.loaded || !!sweep?.running}
          >
            validate whole session
          </Button>
          {sweep?.error && <span className="text-xs text-red-400">{sweep.error}</span>}
          {resultsFiles.map((f) => (
            <Button key={f} onClick={() => void loadResults(f)}>
              {f.replace("validation_", "").replace(".json", "")}
            </Button>
          ))}
        </div>

        {results ? (
          <div className="mt-4 space-y-4">
            <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
              <Metric
                label="mean agreement"
                value={`${(results.mean_agreement * 100).toFixed(1)}%`}
                tone={results.mean_agreement > 0.8 ? "green" : "amber"}
              />
              <Metric label="ticks scored" value={`${results.ticks_scored}/${results.ticks_total}`} />
              <Metric
                label="runtime-rejected"
                value={String(results.gate_rejected.length)}
                tone={results.gate_rejected.length ? "red" : "green"}
              />
              <Metric
                label="not scoreable"
                value={String(results.failures.length)}
                tone={results.failures.length ? "amber" : "green"}
              />
            </div>

            <div className="flex flex-wrap gap-6">
              <div>
                <div className="mb-1.5 text-[11px] uppercase tracking-wider text-zinc-500">
                  per-class IoU
                </div>
                <div className="w-52 space-y-2">
                  {["blocked", "navigable", "caution"].map((n, i) => (
                    <div key={n}>
                      <div className="flex justify-between text-[11px] text-zinc-400">
                        <span>{n}</span>
                        <span className="font-mono">
                          {(results.per_class_iou[i] * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="h-1.5 overflow-hidden rounded-full bg-zinc-800">
                        <div
                          className="h-full rounded-full"
                          style={{
                            width: `${results.per_class_iou[i] * 100}%`,
                            background: ["#d32f2f", "#2ea043", "#ffa500"][i],
                          }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
              <div>
                <div className="mb-1.5 text-[11px] uppercase tracking-wider text-zinc-500">
                  confusion — rows label, cols model
                </div>
                <Confusion m={results.confusion_label_x_model} />
              </div>
            </div>

            {results.gate_rejected.length > 0 && (
              <Collapsible
                tone="amber"
                title={`${results.gate_rejected.length} ticks the runtime would REJECT`}
              >
                {results.gate_rejected.map((f) => (
                  <div key={f.stamp} className="font-mono text-[11px]">
                    {f.stamp} — {f.reason}
                  </div>
                ))}
              </Collapsible>
            )}
            {results.failures.length > 0 && (
              <Collapsible tone="red" title={`${results.failures.length} ticks not scoreable`}>
                {results.failures.map((f) => (
                  <div key={f.stamp} className="font-mono text-[11px]">
                    {f.stamp} — {f.reason}
                  </div>
                ))}
              </Collapsible>
            )}

            <div>
              <div className="mb-1.5 text-[11px] uppercase tracking-wider text-zinc-500">
                worst ticks — click to replay
              </div>
              <div className="max-h-64 overflow-y-auto rounded-lg ring-1 ring-inset ring-white/[0.06]">
                <table className="w-full font-mono text-[11px]">
                  <thead className="sticky top-0 bg-zinc-800 text-zinc-400">
                    <tr>
                      <th className="px-3 py-1.5 text-left font-medium">stamp</th>
                      <th className="px-3 py-1.5 text-right font-medium">agreement</th>
                      <th className="px-3 py-1.5 text-right font-medium">frac_nav</th>
                      <th className="px-3 py-1.5 text-right font-medium">latency</th>
                    </tr>
                  </thead>
                  <tbody>
                    {results.worst.map((w) => (
                      <tr
                        key={w.stamp}
                        className="cursor-pointer border-t border-white/[0.04] hover:bg-zinc-800/60"
                        onClick={() => jumpStamp(w.stamp)}
                      >
                        <td className="px-3 py-1.5 text-zinc-300">{w.stamp}</td>
                        <td className="px-3 py-1.5 text-right">
                          {(w.agreement * 100).toFixed(1)}%
                        </td>
                        <td className="px-3 py-1.5 text-right">{w.frac_navigable}</td>
                        <td className="px-3 py-1.5 text-right text-zinc-500">
                          {w.latency_ms} ms
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        ) : (
          <p className="mt-3 text-xs text-zinc-600">
            Runs the model over every tick in the session and aggregates
            agreement vs the effective label (hand &gt; fused). Results are
            persisted per model and reload on the next visit.
          </p>
        )}
      </Card>
    </div>
  );
}

function Empty({ title, body }: { title: string; body: string }) {
  return (
    <div className="flex h-full items-center justify-center p-10">
      <div className="max-w-md rounded-xl border border-dashed border-zinc-700 p-8 text-center">
        <h2 className="text-sm font-semibold text-zinc-300">{title}</h2>
        <p className="mt-2 text-xs text-zinc-500">{body}</p>
      </div>
    </div>
  );
}

function Metric({
  label,
  value,
  tone = "zinc",
}: {
  label: string;
  value: string;
  tone?: "zinc" | "green" | "amber" | "red";
}) {
  return (
    <div className="rounded-lg bg-zinc-800/50 px-3 py-2.5 ring-1 ring-inset ring-white/[0.06]">
      <div className="text-[10px] uppercase tracking-wider text-zinc-500">{label}</div>
      <div
        className={cn(
          "mt-0.5 text-lg font-semibold tabular-nums",
          tone === "zinc" && "text-zinc-200",
          tone === "green" && "text-emerald-400",
          tone === "amber" && "text-amber-400",
          tone === "red" && "text-red-400",
        )}
      >
        {value}
      </div>
    </div>
  );
}

function Collapsible({
  tone,
  title,
  children,
}: {
  tone: "amber" | "red";
  title: string;
  children: React.ReactNode;
}) {
  return (
    <details className="rounded-lg bg-zinc-800/40 px-3 py-2 ring-1 ring-inset ring-white/[0.06]">
      <summary
        className={cn(
          "cursor-pointer text-[11px] font-medium",
          tone === "amber" ? "text-amber-300" : "text-red-300",
        )}
      >
        {title}
      </summary>
      <div className="mt-2 max-h-40 space-y-1 overflow-y-auto">{children}</div>
    </details>
  );
}

function Confusion({ m }: { m: number[][] }) {
  const tot = m.flat().reduce((a, b) => a + b, 0) || 1;
  return (
    <table className="font-mono text-[11px]">
      <tbody>
        {m.map((row, i) => (
          <tr key={i}>
            {row.map((v, j) => {
              const f = v / tot;
              return (
                <td
                  key={j}
                  className="border border-white/[0.06] px-2.5 py-1 text-right tabular-nums"
                  style={{
                    background:
                      i === j
                        ? `rgba(16,185,129,${0.12 + f})`
                        : `rgba(239,68,68,${f})`,
                  }}
                  title={`label ${i} → model ${j}: ${v}`}
                >
                  {v}
                </td>
              );
            })}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function ProbHeat({ grid, cell = 10 }: { grid: Grid; cell?: number }) {
  const n = grid.length;
  return (
    <canvas
      width={n * cell}
      height={n * cell}
      style={{ imageRendering: "pixelated" }}
      className="rounded-lg ring-1 ring-inset ring-white/[0.08]"
      ref={(cv) => {
        if (!cv) return;
        const cx = cv.getContext("2d")!;
        for (let r = 0; r < n; r++) {
          for (let c = 0; c < n; c++) {
            cx.fillStyle = heatColor(grid[r][c] / 255);
            cx.fillRect((n - 1 - c) * cell, r * cell, cell, cell);
          }
        }
      }}
    />
  );
}

function GoalRaster({ grid }: { grid: Grid }) {
  const cell = 5;
  const n = grid.length;
  return (
    <canvas
      width={n * cell}
      height={n * cell}
      style={{ imageRendering: "pixelated" }}
      className="rounded-lg ring-1 ring-inset ring-white/[0.08]"
      ref={(cv) => {
        if (!cv) return;
        const cx = cv.getContext("2d")!;
        cx.fillStyle = "#0a0d10";
        cx.fillRect(0, 0, n * cell, n * cell);
        for (let r = 0; r < n; r++) {
          for (let c = 0; c < n; c++) {
            const v = grid[r][c];
            if (v > 0) {
              cx.fillStyle = `rgba(129,140,248,${v})`;
              cx.fillRect((n - 1 - c) * cell, r * cell, cell, cell);
            }
          }
        }
      }}
    />
  );
}
