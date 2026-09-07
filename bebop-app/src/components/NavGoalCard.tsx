/// Navigation-goal card (plan §8): operator sets a heading offset and/or
/// an odom waypoint for the navd goal-drive pipeline. The goal rides the
/// runtime WS (SetNavigationGoal → firmware broadcast → the navd process's
/// goal slot); the card mirrors the authoritative state from the
/// firmware's NavigationGoalState push, so goals issued by *any* client
/// (this card, stdin, another tool) all show up here.
import { Button } from "./ui";
import type { NavGoalUpdate } from "../runtime";

interface NavGoalCardProps {
  activeGoal: NavGoalUpdate | null;
  headingDeg: number;
  onHeadingDeg: (deg: number) => void;
  distanceM: number;
  onDistanceM: (m: number) => void;
  onSend: (goal?: {
    headingRad?: number;
    pointOdom?: { x: number; y: number };
  }) => Promise<void>;
  busy: boolean;
  estopLatched: boolean;
}

function goalText(goal: NavGoalUpdate | null): string {
  if (goal === null) return "no active goal";
  if (goal.kind === "heading")
    return `heading ${(goal.headingRad * (180 / Math.PI)).toFixed(0)}°`;
  return `point (${goal.x.toFixed(2)}, ${goal.y.toFixed(2)}) m`;
}

export function NavGoalCard({
  activeGoal,
  headingDeg,
  onHeadingDeg,
  distanceM,
  onDistanceM,
  onSend,
  busy,
  estopLatched,
}: NavGoalCardProps) {
  return (
    <div className="rounded-[var(--radius-card)] border border-border bg-bg-elev px-3.5 py-3 space-y-3">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div className="min-w-0">
          <div className="text-[11px] uppercase tracking-wider text-text-dim">
            Navigate
          </div>
          <div
            className={`text-[13px] font-semibold mt-0.5 ${
              activeGoal ? "text-accent" : "text-text"
            }`}
          >
            {goalText(activeGoal)}
          </div>
        </div>
        <Button
          variant="ghost"
          disabled={busy || activeGoal === null}
          className="py-1.5! px-2! text-[12px]!"
          onClick={() => void onSend()}
        >
          Clear
        </Button>
      </div>

      <label className="block space-y-1">
        <span className="flex justify-between text-xs text-text-dim">
          <span>Heading offset</span>
          <span className="text-text font-medium">
            {headingDeg > 0 ? `+${headingDeg}° left` : headingDeg < 0 ? `${headingDeg}° right` : "straight"}
          </span>
        </span>
        <input
          type="range"
          min={-90}
          max={90}
          step={5}
          value={headingDeg}
          onChange={(e) => onHeadingDeg(Number(e.target.value))}
          className="w-full accent-[var(--color-accent)]"
        />
      </label>

      <label className="block space-y-1">
        <span className="flex justify-between text-xs text-text-dim">
          <span>Distance</span>
          <span className="text-text font-medium">{distanceM.toFixed(1)} m</span>
        </span>
        <input
          type="range"
          min={0.5}
          max={3}
          step={0.25}
          value={distanceM}
          onChange={(e) => onDistanceM(Number(e.target.value))}
          className="w-full accent-[var(--color-accent)]"
        />
      </label>

      <Button
        disabled={busy || estopLatched}
        onClick={() =>
          void onSend({
            headingRad: (headingDeg * Math.PI) / 180,
            pointOdom: undefined,
          })
        }
        className="w-full"
      >
        {busy ? "Sending…" : "Go (heading hold)"}
      </Button>
      <Button
        variant="secondary"
        disabled={busy || estopLatched}
        onClick={() =>
          void onSend({
            pointOdom: { x: 0, y: 0 },
          })
        }
        className="w-full"
        title="Waypoint goal — odom coordinates are computed by the robot from the requested heading + distance"
      >
        Go (waypoint)
      </Button>
      {estopLatched ? (
        <span className="block text-xs text-danger">
          E-STOP latched — reset it before sending goals.
        </span>
      ) : null}
    </div>
  );
}
