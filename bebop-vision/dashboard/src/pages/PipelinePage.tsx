import { useCallback, useEffect, useState } from "react";
import { api, type SessionPipeline, type StageSam } from "../api/client";
import { Badge, Button, Card, Spinner } from "../components/ui";

const STAGES = [
  ["1", "record", "raw MCAP (datasets/sessions)"],
  ["2", "extract", "color/ depth/ labels/ manifest"],
  ["3", "sam", "sam_floor{,_far} masks per camera"],
  ["4", "fuse", "SAM floor gated by depth → fused"],
  ["5", "review", "hand corrections (additive)"],
] as const;

interface Props {
  session: string | null;
  onOpenReview: (name: string) => void;
}

/** Live pipeline artifact browser: which stage produced what, per
 * session. */
export default function PipelinePage({ session, onOpenReview }: Props) {
  const [rows, setRows] = useState<SessionPipeline[] | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const load = useCallback(() => {
    api
      .pipeline()
      .then((r) => {
        setRows(r.sessions);
        setErr(null);
      })
      .catch((e) => setErr(String(e)));
  }, []);

  useEffect(() => load(), [load]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="h-full overflow-auto">
      <div className="mx-auto max-w-6xl space-y-4 p-4">
        <Card
          title="sessions through the pipeline"
          bodyClass="p-0"
          actions={
            <Button onClick={load} title="rescan artifacts">
              refresh
            </Button>
          }
        >
          <div className="overflow-x-auto">
            <table className="w-full text-left text-[11px]">
              <thead>
                <tr className="border-b border-white/[0.06] text-zinc-500">
                  <th className="px-3.5 py-2 font-medium">session</th>
                  {STAGES.map(([n, name, title]) => (
                    <th key={n} className="px-3 py-2 font-medium" title={title}>
                      <span className="mr-1 font-mono text-zinc-600">{n}</span>
                      {name}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows === null && (
                  <tr>
                    <td colSpan={8} className="px-3.5 py-6">
                      <div className="flex items-center gap-2 text-zinc-500">
                        <Spinner /> scanning artifacts…
                      </div>
                    </td>
                  </tr>
                )}
                {rows !== null && !rows.length && (
                  <tr>
                    <td colSpan={8} className="px-3.5 py-6 text-zinc-500">
                      No sessions under datasets/navd-v0 — extract an MCAP first.
                    </td>
                  </tr>
                )}
                {rows?.map((s) => (
                  <tr
                    key={s.name}
                    className="border-b border-white/[0.04] last:border-0 hover:bg-white/[0.02]"
                  >
                    <td className="px-3.5 py-2.5">
                      <button
                        className="font-mono text-zinc-300 hover:text-indigo-300"
                        title="open in tick review"
                        onClick={() => onOpenReview(s.name)}
                      >
                        {s.name.replace("navd_session_", "")}
                      </button>
                      {s.name === session && (
                        <span className="ml-1.5 text-[10px] text-indigo-400">●</span>
                      )}
                    </td>
                    <td className="px-3 py-2.5">
                      {s.record.present ? (
                        <Badge tone="green">{s.record.size_mb} MB</Badge>
                      ) : (
                        <Badge tone="red">no mcap</Badge>
                      )}
                    </td>
                    <td className="px-3 py-2.5 font-mono text-zinc-400">
                      {s.extract.ticks}
                      <span className="text-zinc-600">{s.extract.manifest ? " ✓mf" : ""}</span>
                    </td>
                    <td className="px-3 py-2.5">
                      <SamCell label="n" st={s.sam.near} />
                      <SamCell label="f" st={s.sam.far} />
                    </td>
                    <td className="px-3 py-2.5">
                      <MiniBar done={s.fuse.fused} total={s.fuse.total} />
                    </td>
                    <td className="px-3 py-2.5">
                      <MiniBar done={s.review.hand} total={s.ticks} />
                    </td>
                    <td className="px-3 py-2.5 text-zinc-500">↓ artifacts</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {err && (
            <div className="border-t border-white/[0.06] px-3.5 py-2 text-[11px] text-red-400">
              pipeline scan failed: {err}
            </div>
          )}
        </Card>

        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <Card title="classes">
            <ul className="space-y-2 text-xs">
              {[
                ["0", "blocked", "obstacle in the 0.03–0.30 m band; conservative full band when depth is missing (glass/dark)"],
                ["1", "navigable", "SAM floor confirmed by the depth landing within 0.25 m of the flat-floor prediction"],
                ["2", "caution", "unconfirmed floor, conflicts, out-of-FOV — planning-blocked but not 'solid'"],
              ].map(([n, name, d]) => (
                <li key={n} className="flex gap-2.5">
                  <span
                    className="mt-0.5 h-3 w-3 shrink-0 rounded-[3px]"
                    style={{ background: ["#d32f2f", "#2ea043", "#ffa500"][Number(n)] }}
                  />
                  <span>
                    <b>{name}</b> <span className="text-zinc-500">({d})</span>
                  </span>
                </li>
              ))}
            </ul>
          </Card>
          <Card title="grid geometry">
            <ul className="space-y-1.5 text-xs text-zinc-400">
              <li>60×60 cells · 3×3 m · 5 cm/cell</li>
              <li>row 0 = far edge (canvas top) · row 59 = at the robot</li>
              <li>grid col 0 = robot right → renders at canvas right</li>
              <li className="text-zinc-500">
                same convention everywhere: server overlay, filmstrip, runtime planner
              </li>
            </ul>
          </Card>
        </div>
      </div>
    </div>
  );
}

function MiniBar({ done, total }: { done: number; total: number }) {
  const pct = total ? Math.round((100 * done) / total) : 0;
  return (
    <div className="flex items-center gap-1.5">
      <div className="h-1.5 w-14 overflow-hidden rounded-full bg-zinc-800">
        <div
          className={`h-full rounded-full ${pct === 100 ? "bg-emerald-500" : "bg-amber-500"}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="font-mono text-[10px] text-zinc-500">
        {done}/{total}
      </span>
    </div>
  );
}

function SamCell({ label, st }: { label: string; st: StageSam }) {
  const pct = st.coverage != null ? `${Math.round(st.coverage * 100)}%` : "—";
  return (
    <div className="flex items-center gap-1.5" title={`SAM ${label === "n" ? "near" : "far"}`}>
      <span className="w-3 text-zinc-600">{label}</span>
      <MiniBar done={st.done} total={st.total} />
      <span className="font-mono text-[10px] text-zinc-600" title="mean mask coverage (sampled)">
        {pct}
      </span>
    </div>
  );
}
