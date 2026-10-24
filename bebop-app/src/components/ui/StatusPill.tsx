import type { ReactNode } from "react";

import { cn } from "./cn";

export type Tone = "ok" | "warn" | "err" | "info" | "dim";

export const TONE_PILL: Record<Tone, string> = {
  ok: "bg-success/15 text-success border-success/40",
  warn: "bg-warning/15 text-warning-fg border-warning/40",
  err: "bg-danger/12 text-danger-fg border-danger/40",
  info: "bg-info/12 text-info-fg border-info/40",
  dim: "bg-bg-elev-2 text-text-dim border-border",
};

export const TONE_DOT: Record<Tone, string> = {
  ok: "bg-success",
  warn: "bg-warning",
  err: "bg-danger",
  info: "bg-info",
  dim: "bg-text-dim/40",
};

/// Compact status chip. `pulse` animates the dot (connecting, building).
export function StatusPill({
  tone = "dim",
  children,
  dot = false,
  pulse = false,
  className,
  title,
}: {
  tone?: Tone;
  children: ReactNode;
  dot?: boolean;
  pulse?: boolean;
  className?: string;
  title?: string;
}) {
  return (
    <span
      title={title}
      className={cn(
        "inline-flex items-center gap-1.5 whitespace-nowrap rounded-full border px-2 py-0.5 text-micro font-semibold uppercase tracking-wider",
        TONE_PILL[tone],
        className,
      )}
    >
      {dot ? (
        <span
          className={cn("h-1.5 w-1.5 rounded-full", TONE_DOT[tone], pulse && "animate-pulse")}
          aria-hidden
        />
      ) : null}
      {children}
    </span>
  );
}

/// Bare status dot with an accessible label.
export function StatusDot({
  tone,
  label,
  pulse = false,
  className,
}: {
  tone: Tone;
  label: string;
  pulse?: boolean;
  className?: string;
}) {
  return (
    <span
      role="img"
      aria-label={label}
      title={label}
      className={cn(
        "inline-block h-2 w-2 shrink-0 rounded-full",
        TONE_DOT[tone],
        pulse && "animate-pulse",
        className,
      )}
    />
  );
}
