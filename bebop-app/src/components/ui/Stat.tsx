import type { ReactNode } from "react";

import { cn } from "./cn";

/// Label-over-value readout ("Battery / 78 %"). `hint` is shown as a
/// caption under the value, so it works on touch (no hover tooltip).
export function Stat({
  label,
  value,
  hint,
  className,
}: {
  label: ReactNode;
  value: ReactNode;
  hint?: ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("flex min-w-0 flex-col leading-tight", className)}>
      <span className="text-micro uppercase tracking-wider text-text-dim">{label}</span>
      <span className="font-semibold tabular-nums text-text">{value}</span>
      {hint ? <span className="text-caption text-text-dim">{hint}</span> : null}
    </div>
  );
}
