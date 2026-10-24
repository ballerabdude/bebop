// What the operator needs to see while the agent works: where the robot
// thinks it is, battery, mode, who holds the drive seat, E-STOP, and a
// small live camera.

import { VideoFeed } from "../../components/VideoFeed";
import { cn, StatusDot } from "../../components/ui";
import type { Tone } from "../../components/ui";
import { usePlaces, useRobot, useTelemetry } from "../../robot/session";
import type { RuntimeSnapshot } from "../../runtime/types";
import { CameraIcon, PinIcon } from "./icons";
import { formatMatchScore } from "../memory/places";

export interface RobotStatusSlice {
  loaded: boolean;
  socKnown: boolean;
  soc: number;
  volts: number;
  powerPresent: boolean;
  mode: string;
  driver: "you" | "other" | "none";
  driverName: string;
  estop: boolean;
  estopReason: string;
  visionRunning: boolean;
  visionPresent: boolean;
}

export function selectRobotStatus(s: RuntimeSnapshot | null): RobotStatusSlice {
  const p = s?.power;
  const d = s?.drive;
  return {
    loaded: !!s,
    powerPresent: !!p?.present,
    socKnown: !!p?.present && !!p.statusReceived && p.stateOfChargePct >= 0,
    soc: Math.round(p?.stateOfChargePct ?? -1),
    volts: Math.round((p?.batteryVoltageV ?? 0) * 10) / 10,
    mode: s?.mode ?? "",
    driver: d?.youAreActiveOperator ? "you" : d?.hasActiveOperator ? "other" : "none",
    driverName: d?.activeOperatorName ?? "",
    estop: !!s?.estopLatched,
    estopReason: s?.estopReason ?? "",
    visionRunning: !!s?.vision.running,
    visionPresent: !!s?.vision.present,
  };
}

const MODE_LABEL: Record<string, string> = {
  UNSPECIFIED: "Unknown",
  IDLE: "Idle",
  DIAL_IN: "Armed",
  RUN_POLICY: "Policy",
};

export function placeLabel(where: ReturnType<typeof usePlaces>["whereami"], reachable: boolean | null): {
  text: string;
  tone: Tone;
  hint: string;
} {
  if (reachable === false) return { text: "Places offline", tone: "dim", hint: "Place memory isn't reachable." };
  if (!where) return { text: "Locating…", tone: "dim", hint: "" };
  const place = where.place as { display_name?: string; name?: string } | null | undefined;
  const name = place?.display_name || place?.name || "";
  if (where.matched && name) {
    const score = typeof where.score === "number" ? ` · score ${formatMatchScore(where.score)}` : "";
    return { text: name, tone: "ok", hint: `Recognised place${score}` };
  }
  // `top` entries are `{id, name, score}` (older services nested a `place`).
  const top = (where.top?.[0] ?? null) as { name?: string; place?: { name?: string } } | null;
  const best = top?.name || top?.place?.name || "";
  return {
    text: "Unknown place",
    tone: "dim",
    hint: best ? `Closest: ${best}` : String(where.reason ?? "Not a remembered place"),
  };
}

function Item({
  label,
  value,
  tone,
  hint,
}: {
  label: string;
  value: string;
  tone: Tone;
  hint?: string;
}) {
  return (
    <div className="flex min-w-0 flex-col gap-0.5" title={hint || undefined}>
      <span className="text-micro font-semibold uppercase tracking-wider text-text-dim">{label}</span>
      <span className="flex min-w-0 items-center gap-1.5 text-body-sm font-semibold text-text">
        <StatusDot tone={tone} label={`${label}: ${value}`} />
        <span className="truncate">{value}</span>
      </span>
    </div>
  );
}

export function useRobotStatus() {
  const s = useTelemetry(selectRobotStatus, 4);
  const places = usePlaces();
  const place = placeLabel(places.whereami, places.reachable);
  const battery = s.socKnown
    ? { text: `${s.soc}%`, tone: (s.soc < 25 ? "err" : s.soc < 60 ? "warn" : "ok") as Tone }
    : s.powerPresent && s.volts > 0
      ? { text: `${s.volts.toFixed(1)} V`, tone: "dim" as Tone }
      : { text: "—", tone: "dim" as Tone };
  const driving =
    s.driver === "none"
      ? { text: "Nobody", tone: "dim" as Tone }
      : s.driver === "you"
        ? { text: "You", tone: "ok" as Tone }
        : { text: s.driverName || "Another operator", tone: (s.driverName === "voice" ? "info" : "warn") as Tone };
  return { s, place, battery, driving };
}

/// 2×2 (or 4-across) grid of the status items.
export function StatusGrid({
  status,
  className,
}: {
  status: ReturnType<typeof useRobotStatus>;
  className?: string;
}) {
  const { s, place, battery, driving } = status;
  return (
    <div className={cn("grid grid-cols-2 gap-x-4 gap-y-3", className)}>
      <div className="col-span-2 flex min-w-0 items-center gap-2" title={place.hint || undefined}>
        <span
          className={cn(
            "inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-full",
            place.tone === "ok" ? "bg-success/15 text-success" : "bg-bg-elev-2 text-text-dim",
          )}
        >
          <PinIcon size={16} />
        </span>
        <div className="min-w-0">
          <div className="text-micro font-semibold uppercase tracking-wider text-text-dim">Where am I</div>
          <div className="truncate text-sm font-semibold text-text">{place.text}</div>
          {place.hint ? <div className="truncate text-caption text-text-dim">{place.hint}</div> : null}
        </div>
      </div>
      <Item label="Battery" value={battery.text} tone={battery.tone} />
      <Item
        label="Mode"
        value={s.estop ? "E-STOP" : MODE_LABEL[s.mode] ?? (s.mode || "—")}
        tone={s.estop ? "err" : s.mode === "DIAL_IN" ? "warn" : "dim"}
        hint={s.estop ? s.estopReason : undefined}
      />
      <Item label="Driving" value={driving.text} tone={driving.tone} />
      <Item
        label="Camera"
        value={!s.visionPresent ? "Not installed" : s.visionRunning ? "Vision on" : "Off"}
        tone={s.visionRunning ? "ok" : "dim"}
      />
    </div>
  );
}

/// The small live camera tile (color_near). Shows a calm placeholder when
/// the vision service is off.
export function CameraTile({
  running,
  className,
  maxHeight,
}: {
  running: boolean;
  className?: string;
  maxHeight?: string;
}) {
  const { endpoints } = useRobot();
  if (!running) {
    return (
      <div
        className={cn(
          "flex aspect-video w-full flex-col items-center justify-center gap-1.5 rounded-input border border-dashed border-border bg-bg/60 text-label text-text-dim",
          className,
        )}
        style={maxHeight ? { maxHeight } : undefined}
      >
        <CameraIcon size={20} />
        Camera is off
        <span className="text-caption">Start vision from Diagnostics › Services.</span>
      </div>
    );
  }
  return (
    <VideoFeed
      whepUrl={endpoints.whep("color_near")}
      whepResource={endpoints.whepResource}
      reconnectKey={0}
      className={cn("w-full overflow-hidden rounded-input border border-border", className)}
      maxHeight={maxHeight}
    >
      <span className="pointer-events-none absolute left-2 top-2 z-10 inline-flex items-center gap-1.5 rounded bg-black/60 px-1.5 py-0.5 text-caption font-semibold text-white/90">
        <span className="h-1.5 w-1.5 rounded-full bg-danger" aria-hidden />
        Front camera
      </span>
    </VideoFeed>
  );
}
