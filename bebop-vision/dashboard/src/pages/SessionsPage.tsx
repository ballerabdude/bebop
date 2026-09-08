import { api } from "../api/client";
import { CLS_HEX, CLS_NAMES, GRID } from "../api/grid";

/** About / pipeline reference: how data flows and what the classes mean —
 * the orientation material an engineer needs on day one. */
export default function SessionsPage() {
  void api; // typed client is exercised elsewhere; page stays dependency-free
  return (
    <div className="max-w-4xl space-y-4 p-6 text-sm leading-relaxed">
      <h1 className="text-lg font-bold">navd data &amp; model pipeline</h1>

      <section className="rounded border border-[#39434f] bg-[#1b2129] p-4">
        <h2 className="font-bold">Pipeline (collection → labels → training → runtime)</h2>
        <ol className="list-decimal space-y-1 pl-6 text-[#9fb0c0]">
          <li>
            <b>Record</b> — <code>main.py --record-navd</code> writes one MCAP
            per session: near/far color JPEG, lossless depth PNG, cmd_vel /
            odom / goal, geometric <code>/bev_teacher</code> (bookkeeping only).
          </li>
          <li>
            <b>Extract</b> — <code>tools/mcap_extract.py</code> →
            datasets/navd-v0/&lt;session&gt;/ (color/, color_far/, depth/,
            labels/, manifest.jsonl).
          </li>
          <li>
            <b>SAM floor pass</b> — <code>tools/sam_floor_label.py</code>: SAM
            3.1 with floor/carpet/rug/ground concepts over both cameras.
          </li>
          <li>
            <b>Fuse → teacher labels</b> — <code>tools/fuse_navd_labels.py</code>:
            SAM floor masks gated by measured depth (ray landing within 0.25 m
            of the flat-floor prediction = navigable; non-floor with valid
            depth = blocked band; no return = conservative full-band blocked;
            everything else = caution). Writes the <code>fused</code> grid.
          </li>
          <li>
            <b>Human review</b> — this dashboard. Corrections save as{" "}
            <code>hand</code> in labels/*.npz; training prefers hand over
            fused; teacher keys are never modified.
          </li>
          <li>
            <b>Train</b> — <code>train_navd.py</code> (NavdUNet, 6-ch stem).
            <b>Export</b> — <code>tools/export_navd_onnx.py</code> (parity gate
            ≥ 0.99 vs torch).
          </li>
          <li>
            <b>Runtime</b> — <code>bebop_vision/navd_runtime.py</code>: ONNX on
            raw frames, grid is the seam the planner drives on. Model-only, no
            fallback; <code>frac_navigable</code> ∈ [0.05, 0.95] else hold.
          </li>
        </ol>
      </section>

      <section className="rounded border border-[#39434f] bg-[#1b2129] p-4">
        <h2 className="font-bold">Classes</h2>
        <ul className="space-y-1 pl-6">
          {CLS_NAMES.map((n, i) => (
            <li key={n} className="flex items-center gap-2">
              <span
                className="inline-block h-3 w-3 rounded-sm"
                style={{ background: CLS_HEX[i] }}
              />
              <b>{i}</b> — {n}
            </li>
          ))}
        </ul>
        <p className="mt-2 text-[#9fb0c0]">
          Grid: {GRID}×{GRID} cells, 3×3 m @ 5 cm. Row 0 = far edge (canvas
          top); grid col 0 = robot right (canvas right).
        </p>
      </section>

      <section className="rounded border border-[#39434f] bg-[#1b2129] p-4">
        <h2 className="font-bold">Model inputs (what it actually sees)</h2>
        <p className="text-[#9fb0c0]">
          depth_near + depth_far (meters, clipped [0.3, 6.0], 0=invalid), near
          color (ImageNet-norm), goal fan raster. All preprocessed by{" "}
          <code>navd_pre.py</code> — the dashboard's validation tab runs the
          exact same code path, so what you see is what the robot saw.
        </p>
      </section>
    </div>
  );
}
