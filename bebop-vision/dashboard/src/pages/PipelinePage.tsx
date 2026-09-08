import { Card } from "../components/ui";

/** Pipeline orientation: how data flows, what the classes mean, and what
 * the model actually sees. The reading an engineer needs on day one. */
export default function PipelinePage() {
  return (
    <div className="mx-auto max-w-4xl space-y-4 p-4">
      <Card title="pipeline">
        <ol className="space-y-1">
          {[
            ["Record", "main.py --record-navd → MCAP sessions (color JPEG, lossless depth, cmd_vel/odom/goal, /bev_teacher bookkeeping)"],
            ["Extract", "tools/mcap_extract.py → datasets/navd-v0/<session>/"],
            ["SAM floor pass", "tools/sam_floor_label.py — SAM 3.1, floor/carpet/rug/ground concepts, both cameras"],
            ["Fuse", "tools/fuse_navd_labels.py — SAM floor gated by measured depth → fused 60×60 teacher"],
            ["Human review", "this dashboard — corrections save additively as hand"],
            ["Train + export", "train_navd.py · tools/export_navd_onnx.py (parity gate ≥ 0.99)"],
            ["Runtime", "bebop_vision/navd_runtime.py — model-only, grid is the seam"],
          ].map(([t, d], i) => (
            <li key={t} className="flex gap-3 py-1">
              <span className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-zinc-800 font-mono text-[10px] text-zinc-400 ring-1 ring-inset ring-white/10">
                {i + 1}
              </span>
              <span className="text-xs leading-relaxed text-zinc-400">
                <b className="text-zinc-200">{t}</b> — {d}
              </span>
            </li>
          ))}
        </ol>
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
                  <b>{name}</b>{" "}
                  <span className="text-zinc-500">({d})</span>
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
            <li className="text-zinc-500">same convention everywhere: server overlay, filmstrip, runtime planner</li>
          </ul>
        </Card>
      </div>
      <Card title="model inputs — what the network actually sees">
        <div className="grid grid-cols-2 gap-3 text-xs text-zinc-400 md:grid-cols-4">
          {[
            ["depth_near", "meters, clip [0.3, 6.0], 0 = invalid"],
            ["depth_far", "same"],
            ["color (near)", "ImageNet-normalized RGB"],
            ["goal raster", "60×60 fan toward the goal bearing"],
          ].map(([k, v]) => (
            <div key={k} className="rounded-lg bg-zinc-800/50 px-3 py-2.5 ring-1 ring-inset ring-white/[0.06]">
              <div className="font-mono text-[11px] text-indigo-300">{k}</div>
              <div className="mt-1 text-[11px] leading-snug text-zinc-500">{v}</div>
            </div>
          ))}
        </div>
        <p className="mt-3 text-[11px] text-zinc-600">
          All preprocessing lives in <code className="text-zinc-400">navd_pre.py</code>, shared
          byte-identical by training, export, runtime — and this dashboard.
        </p>
      </Card>
    </div>
  );
}
