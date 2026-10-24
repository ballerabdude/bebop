import type { ReactNode } from "react";

import { cn } from "./cn";

export type BannerTone = "info" | "error" | "success" | "warning";

const TONE: Record<BannerTone, string> = {
  info: "bg-info/10 text-info-fg border-info/25",
  error: "bg-danger/12 text-danger-fg border-danger/30",
  success: "bg-success/12 text-success-fg border-success/30",
  warning: "bg-warning/10 text-warning-fg border-warning/30",
};

/// Inline message. `title` is bold on top; `action` sits on the right (or
/// below on narrow screens).
export function Banner({
  tone = "info",
  title,
  action,
  children,
  className,
}: {
  tone?: BannerTone;
  title?: ReactNode;
  action?: ReactNode;
  children?: ReactNode;
  className?: string;
}) {
  return (
    <div
      role={tone === "error" ? "alert" : "status"}
      className={cn(
        "flex flex-col gap-2 rounded-card border px-3.5 py-3 text-sm sm:flex-row sm:items-start sm:justify-between",
        TONE[tone],
        className,
      )}
    >
      <div className="min-w-0">
        {title ? <div className="mb-0.5 font-semibold">{title}</div> : null}
        {children ? (
          <div className={title ? "text-label leading-relaxed" : undefined}>{children}</div>
        ) : null}
      </div>
      {action ? <div className="shrink-0">{action}</div> : null}
    </div>
  );
}
