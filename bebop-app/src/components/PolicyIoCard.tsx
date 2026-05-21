import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import type { PolicyIoView } from "../runtime";

const OBS_GROUPS: { title: string; start: number; end: number; unit?: string }[] = [
  { title: "Base lin vel", start: 0, end: 3, unit: "m/s" },
  { title: "Base ang vel", start: 3, end: 6, unit: "rad/s" },
  { title: "Projected gravity", start: 6, end: 9 },
  { title: "Joint pos rel", start: 9, end: 17, unit: "rad" },
  { title: "Joint vel", start: 17, end: 25, unit: "rad/s" },
  { title: "Last action", start: 25, end: 49 },
  { title: "Cmd vel", start: 49, end: 52, unit: "m/s, rad/s" },
];

const AXIS_LABELS = ["x", "y", "z", "w"];

/// Window options (samples kept per signal). Telemetry runs ~30 Hz so
/// 240 samples ≈ 8 s, 600 samples ≈ 20 s.
const HISTORY_OPTIONS = [
  { label: "5s", samples: 150 },
  { label: "15s", samples: 450 },
  { label: "30s", samples: 900 },
];

function shortJointName(full: string): string {
  return full
    .replace("_joint", "")
    .replace("hip_abduction_", "hip_")
    .replace("femur_", "fem_")
    .replace("shin_", "sh_")
    .replace("foot_", "ft_");
}

function fmt(v: number, digits = 3): string {
  if (!Number.isFinite(v)) return "—";
  return v.toFixed(digits);
}

// --------------------------------------------------------------------------
// History buffer
// --------------------------------------------------------------------------

/// Per-signal ring buffer. Keys map to fields on `PolicyIoView`.
interface PolicyIoHistory {
  observation: number[][];
  rawAction: number[][];
  positionTargetsRad: number[][];
  kp: number[][];
  kd: number[][];
  /// True iff we have at least one sample buffered.
  ready: boolean;
}

/// Track the last N samples per scalar signal across telemetry updates.
///
/// Resets when transitioning from inactive → active (e.g. entering
/// RunPolicy after a pause) so we never plot a stale frozen window. The
/// buffer size grows with `capacity`; shrinking trims from the head.
///
/// The returned `clear` callback wipes every buffer in place — useful for
/// the "clear graph" toolbar action so the operator can start a fresh
/// run without bouncing the policy out of RunPolicy.
function usePolicyIoHistory(
  policyIo: PolicyIoView | undefined,
  capacity: number,
): { history: PolicyIoHistory; clear: () => void } {
  const histRef = useRef<PolicyIoHistory & { wasActive: boolean }>({
    observation: [],
    rawAction: [],
    positionTargetsRad: [],
    kp: [],
    kd: [],
    ready: false,
    wasActive: false,
  });
  /// Force a re-render after pushing — telemetry triggers one too, but
  /// the capacity slider and the explicit clear button have no parent
  /// state change to ride on.
  const [, setTick] = useState(0);

  useEffect(() => {
    if (!policyIo) return;
    const h = histRef.current;

    if (policyIo.active && !h.wasActive) {
      h.observation = policyIo.observation.map(() => []);
      h.rawAction = policyIo.rawAction.map(() => []);
      h.positionTargetsRad = policyIo.positionTargetsRad.map(() => []);
      h.kp = policyIo.kp.map(() => []);
      h.kd = policyIo.kd.map(() => []);
      h.ready = false;
    }
    h.wasActive = policyIo.active;

    if (!policyIo.active) return;

    const ensure = (target: number[][], n: number) => {
      while (target.length < n) target.push([]);
    };
    ensure(h.observation, policyIo.observation.length);
    ensure(h.rawAction, policyIo.rawAction.length);
    ensure(h.positionTargetsRad, policyIo.positionTargetsRad.length);
    ensure(h.kp, policyIo.kp.length);
    ensure(h.kd, policyIo.kd.length);

    const push = (target: number[][], values: number[]) => {
      for (let i = 0; i < values.length; i++) {
        const arr = target[i];
        arr.push(values[i]);
        while (arr.length > capacity) arr.shift();
      }
    };
    push(h.observation, policyIo.observation);
    push(h.rawAction, policyIo.rawAction);
    push(h.positionTargetsRad, policyIo.positionTargetsRad);
    push(h.kp, policyIo.kp);
    push(h.kd, policyIo.kd);

    if (policyIo.observation.length > 0 || policyIo.rawAction.length > 0) {
      h.ready = true;
    }
    setTick((t) => (t + 1) | 0);
  }, [policyIo, capacity]);

  useEffect(() => {
    const h = histRef.current;
    const trim = (arr: number[][]) => {
      for (let i = 0; i < arr.length; i++) {
        while (arr[i].length > capacity) arr[i].shift();
      }
    };
    trim(h.observation);
    trim(h.rawAction);
    trim(h.positionTargetsRad);
    trim(h.kp);
    trim(h.kd);
  }, [capacity]);

  const clear = useCallback(() => {
    const h = histRef.current;
    // Truncate in place so the SeriesGroup `histories.slice(...)`
    // references still point at the same inner arrays (now empty).
    const wipe = (arr: number[][]) => {
      for (let i = 0; i < arr.length; i++) {
        arr[i].length = 0;
      }
    };
    wipe(h.observation);
    wipe(h.rawAction);
    wipe(h.positionTargetsRad);
    wipe(h.kp);
    wipe(h.kd);
    h.ready = false;
    setTick((t) => (t + 1) | 0);
  }, []);

  return { history: histRef.current, clear };
}

// --------------------------------------------------------------------------
// Sparkline
// --------------------------------------------------------------------------

interface Range {
  min: number;
  max: number;
}

/// Compute a shared y-range across every signal in a group so the
/// sparklines are visually comparable. Auto-pads by 10 % and always
/// includes 0 in the range so the dashed zero-line is meaningful.
function groupRange(seriesList: number[][], explicit?: Range): Range {
  if (explicit) return explicit;
  let mn = Number.POSITIVE_INFINITY;
  let mx = Number.NEGATIVE_INFINITY;
  for (const series of seriesList) {
    for (const v of series) {
      if (Number.isFinite(v)) {
        if (v < mn) mn = v;
        if (v > mx) mx = v;
      }
    }
  }
  if (!Number.isFinite(mn) || !Number.isFinite(mx)) {
    return { min: -1, max: 1 };
  }
  if (mn === mx) {
    return { min: mn - 0.5, max: mx + 0.5 };
  }
  const pad = (mx - mn) * 0.1;
  return {
    min: Math.min(mn - pad, 0),
    max: Math.max(mx + pad, 0),
  };
}

function Sparkline({
  data,
  range,
  strokeClass = "stroke-text-dim",
  fillClass,
}: {
  data: number[];
  range: Range;
  strokeClass?: string;
  /// Optional gradient under the line. Pair with a CSS class that sets
  /// `fill` to a low-opacity hue matching the stroke.
  fillClass?: string;
}) {
  if (data.length < 2) {
    return (
      <div className="flex-1 min-w-[64px] h-5 rounded bg-bg-elev-2/40" aria-hidden />
    );
  }
  const span = range.max - range.min || 1;
  const w = 200;
  const h = 32;
  const step = w / (data.length - 1);

  let path = "";
  for (let i = 0; i < data.length; i++) {
    const x = i * step;
    const v = Math.max(range.min, Math.min(range.max, data[i]));
    const y = h - ((v - range.min) / span) * h;
    path += i === 0 ? `M${x.toFixed(2)},${y.toFixed(2)}` : ` L${x.toFixed(2)},${y.toFixed(2)}`;
  }

  const fillPath = fillClass ? `${path} L${w},${h} L0,${h} Z` : null;

  let zeroLineY: number | null = null;
  if (range.min < 0 && range.max > 0) {
    zeroLineY = h - ((0 - range.min) / span) * h;
  }

  // Highlight the most recent sample so the eye can latch on.
  const lastX = (data.length - 1) * step;
  const lastV = Math.max(range.min, Math.min(range.max, data[data.length - 1]));
  const lastY = h - ((lastV - range.min) / span) * h;

  return (
    <svg
      viewBox={`0 0 ${w} ${h}`}
      preserveAspectRatio="none"
      className="flex-1 min-w-[64px] h-5"
    >
      {zeroLineY !== null ? (
        <line
          x1="0"
          y1={zeroLineY}
          x2={w}
          y2={zeroLineY}
          className="stroke-border"
          strokeWidth="0.5"
          strokeDasharray="3,3"
          vectorEffect="non-scaling-stroke"
        />
      ) : null}
      {fillPath ? <path d={fillPath} className={fillClass} /> : null}
      <path
        d={path}
        fill="none"
        className={strokeClass}
        strokeWidth="1.25"
        strokeLinejoin="round"
        strokeLinecap="round"
        vectorEffect="non-scaling-stroke"
      />
      <circle
        cx={lastX}
        cy={lastY}
        r="1.5"
        className={strokeClass.replace("stroke-", "fill-")}
        vectorEffect="non-scaling-stroke"
      />
    </svg>
  );
}

// --------------------------------------------------------------------------
// Series rows
// --------------------------------------------------------------------------

function SeriesRow({
  label,
  history,
  range,
  latest,
  accent = false,
  labelWidth = "w-16",
  valueWidth = "w-14",
}: {
  label: string;
  history: number[];
  range: Range;
  latest: number;
  accent?: boolean;
  labelWidth?: string;
  valueWidth?: string;
}) {
  const strokeClass = accent ? "stroke-accent" : "stroke-text-dim";
  const fillClass = accent ? "fill-accent/10" : "fill-text-dim/10";
  return (
    <div className="flex items-center gap-2 text-[11px] tabular-nums">
      <span
        className={`${labelWidth} shrink-0 truncate text-text-dim`}
        title={label}
      >
        {label}
      </span>
      <Sparkline
        data={history}
        range={range}
        strokeClass={strokeClass}
        fillClass={fillClass}
      />
      <span className={`${valueWidth} shrink-0 text-right text-text`}>
        {fmt(latest)}
      </span>
    </div>
  );
}

function SeriesGroup({
  title,
  unit,
  histories,
  latestValues,
  labels,
  rangeOverride,
  accent = false,
}: {
  title: string;
  unit?: string;
  histories: number[][];
  latestValues: number[];
  labels: string[];
  rangeOverride?: Range;
  accent?: boolean;
}) {
  const range = useMemo(
    () => groupRange(histories, rangeOverride),
    [histories, rangeOverride],
  );
  if (histories.length === 0) return null;
  return (
    <div className="space-y-1">
      <div className="flex items-baseline justify-between gap-2">
        <div className="text-[10px] uppercase tracking-wider text-text-dim">
          {title}
          {unit ? (
            <span className="normal-case tracking-normal text-text-dim/70">
              {" "}
              ({unit})
            </span>
          ) : null}
        </div>
        <div className="text-[9px] tabular-nums text-text-dim/70">
          y: [{fmt(range.min, 2)}, {fmt(range.max, 2)}]
        </div>
      </div>
      <div className="space-y-0.5">
        {histories.map((series, i) => (
          <SeriesRow
            key={`${title}-${labels[i] ?? i}`}
            label={labels[i] ?? `[${i}]`}
            history={series}
            range={range}
            latest={latestValues[i]}
            accent={accent}
          />
        ))}
      </div>
    </div>
  );
}

// --------------------------------------------------------------------------
// Main card
// --------------------------------------------------------------------------

export function PolicyIoCard({ policyIo }: { policyIo: PolicyIoView }) {
  const [windowIdx, setWindowIdx] = useState(0);
  const capacity = HISTORY_OPTIONS[windowIdx].samples;
  const { history, clear } = usePolicyIoHistory(policyIo, capacity);

  /// Remember the joint-name list across deactivations. The firmware
  /// sends `joint_names` every tick while active but the array is empty
  /// after stop(), so we'd lose the labels mid-screenshot otherwise.
  const lastNamesRef = useRef<string[]>([]);
  if (policyIo.jointNames.length > 0) {
    lastNamesRef.current = policyIo.jointNames;
  }

  if (!policyIo.present) return null;

  const jointNames =
    lastNamesRef.current.length > 0
      ? lastNamesRef.current
      : ["AL", "AR", "FL", "FR", "SL", "SR", "FTL", "FTR"];
  const shortNames = jointNames.map(shortJointName);

  /// When the policy is active, the live snapshot drives "latest"
  /// numbers in the value column. When it's inactive (RunPolicy
  /// stopped) the firmware zeros those fields, so fall back to the
  /// last buffered sample so the column matches what the sparkline
  /// shows on its right edge.
  const tailOf = (series: number[][]): number[] =>
    series.map((arr) => (arr.length > 0 ? arr[arr.length - 1] : 0));

  const obs = policyIo.active && policyIo.observation.length > 0
    ? policyIo.observation
    : tailOf(history.observation);
  const raw = policyIo.active && policyIo.rawAction.length > 0
    ? policyIo.rawAction
    : tailOf(history.rawAction);
  const positionTargets = policyIo.active
    ? policyIo.positionTargetsRad
    : tailOf(history.positionTargetsRad);
  const kpLatest = policyIo.active ? policyIo.kp : tailOf(history.kp);
  const kdLatest = policyIo.active ? policyIo.kd : tailOf(history.kd);

  return (
    <div className="rounded-[var(--radius-card)] border border-border bg-bg-elev px-3.5 py-3 space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <div className="text-[11px] uppercase tracking-wider text-text-dim">
            Policy I/O
          </div>
          <div className="text-[13px] text-text font-semibold mt-0.5">
            ONNX observation → action (history)
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-1.5">
          <StatusPill
            label={
              policyIo.active
                ? "Active"
                : history.ready
                ? "Frozen"
                : "Idle"
            }
            tone={
              policyIo.active ? "success" : history.ready ? "warn" : "dim"
            }
            pulse={policyIo.active}
          />
          {policyIo.active ? (
            <StatusPill
              label={policyIo.imuLive ? "IMU live" : "IMU synthetic"}
              tone={policyIo.imuLive ? "success" : "warn"}
            />
          ) : null}
          <div
            className="inline-flex rounded-full border border-border bg-bg-elev-2 p-0.5"
            role="group"
            aria-label="History window"
          >
            {HISTORY_OPTIONS.map((opt, i) => (
              <button
                key={opt.label}
                type="button"
                onClick={() => setWindowIdx(i)}
                className={`px-2 py-0.5 text-[10px] font-medium rounded-full transition-colors ${
                  i === windowIdx
                    ? "bg-accent text-white"
                    : "text-text-dim hover:text-text"
                }`}
              >
                {opt.label}
              </button>
            ))}
          </div>
          <button
            type="button"
            onClick={clear}
            disabled={!history.ready}
            className="inline-flex items-center gap-1 rounded-full border border-border bg-bg-elev-2 px-2 py-0.5 text-[10px] font-medium text-text-dim hover:text-text hover:border-text-dim/40 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            title="Wipe the sparkline buffers without stopping the policy"
          >
            <svg
              viewBox="0 0 16 16"
              className="w-3 h-3"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.5"
              aria-hidden
            >
              <path d="M3 5h10M6 5V3.5A.5.5 0 0 1 6.5 3h3a.5.5 0 0 1 .5.5V5M5 5l.7 7.5a1 1 0 0 0 1 .9h2.6a1 1 0 0 0 1-.9L11 5" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
            Clear
          </button>
        </div>
      </div>

      {!history.ready ? (
        <p className="text-[12px] text-text-dim italic">
          {policyIo.active
            ? "Waiting for first policy tick…"
            : "Waiting for Run Policy mode — sparklines start accumulating at ~30 Hz once the inference loop is running."}
        </p>
      ) : (
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-x-6 gap-y-4">
          <section className="space-y-3 min-w-0">
            <h3 className="text-[12px] font-semibold text-accent">
              Observations (input)
            </h3>
            {OBS_GROUPS.map((g) => {
              const count = g.end - g.start;
              const histories = history.observation.slice(g.start, g.end);
              const latest = obs.slice(g.start, g.end);

              let labels: string[];
              if (count <= 3) {
                labels = AXIS_LABELS.slice(0, count);
              } else if (count === 8) {
                labels = shortNames;
              } else if (count === 24) {
                labels = Array.from({ length: 24 }, (_, i) => {
                  const channel = i < 8 ? "pos" : i < 16 ? "kp" : "kd";
                  const joint = shortNames[i % 8] ?? `j${i % 8}`;
                  return `${joint} ${channel}`;
                });
              } else {
                labels = Array.from({ length: count }, (_, i) => `[${g.start + i}]`);
              }

              // Bound `last_action` and `cmd_vel` to the policy's known
              // valid range so spikes are calibrated against [-1, 1].
              let rangeOverride: Range | undefined;
              if (g.title === "Last action") {
                rangeOverride = { min: -1, max: 1 };
              }

              return (
                <SeriesGroup
                  key={g.title}
                  title={g.title}
                  unit={g.unit}
                  histories={histories}
                  latestValues={latest}
                  labels={labels}
                  rangeOverride={rangeOverride}
                />
              );
            })}
          </section>

          <section className="space-y-3 min-w-0">
            <h3 className="text-[12px] font-semibold text-success">
              Actions (output)
            </h3>
            <SeriesGroup
              title="Raw position"
              histories={history.rawAction.slice(0, 8)}
              latestValues={raw.slice(0, 8)}
              labels={shortNames}
              rangeOverride={{ min: -1, max: 1 }}
              accent
            />
            <SeriesGroup
              title="Raw kp"
              histories={history.rawAction.slice(8, 16)}
              latestValues={raw.slice(8, 16)}
              labels={shortNames}
              rangeOverride={{ min: -1, max: 1 }}
              accent
            />
            <SeriesGroup
              title="Raw kd"
              histories={history.rawAction.slice(16, 24)}
              latestValues={raw.slice(16, 24)}
              labels={shortNames}
              rangeOverride={{ min: -1, max: 1 }}
              accent
            />

            <div className="pt-1 border-t border-border/60 space-y-2">
              <div className="text-[10px] uppercase tracking-wider text-text-dim">
                Decoded MIT commands
              </div>
              <SeriesGroup
                title="Position targets"
                unit="rad"
                histories={history.positionTargetsRad}
                latestValues={positionTargets}
                labels={shortNames}
              />
              <SeriesGroup
                title="Kp"
                histories={history.kp}
                latestValues={kpLatest}
                labels={shortNames}
              />
              <SeriesGroup
                title="Kd"
                histories={history.kd}
                latestValues={kdLatest}
                labels={shortNames}
              />
            </div>
          </section>
        </div>
      )}
    </div>
  );
}

function StatusPill({
  label,
  tone,
  pulse = false,
}: {
  label: string;
  tone: "success" | "warn" | "dim";
  pulse?: boolean;
}) {
  const styles = {
    success: "border-success/40 bg-success/10 text-success",
    warn: "border-yellow-500/40 bg-yellow-500/10 text-yellow-700 dark:text-yellow-300",
    dim: "border-border bg-bg-elev-2 text-text-dim",
  }[tone];

  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-[10px] font-medium ${styles}`}
    >
      {pulse ? (
        <span
          className="inline-block w-1.5 h-1.5 rounded-full bg-success animate-pulse"
          aria-hidden
        />
      ) : null}
      {label}
    </span>
  );
}
