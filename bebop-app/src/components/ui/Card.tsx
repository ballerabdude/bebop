import type { HTMLAttributes, ReactNode } from "react";

import { cn } from "./cn";

export function Card({
  className,
  children,
  ...rest
}: HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn(
        "min-w-0 rounded-card border border-border bg-bg-elev px-4 py-3",
        className,
      )}
      {...rest}
    >
      {children}
    </div>
  );
}

/// Eyebrow + title + optional actions on the right; wraps on phones.
export function CardHeader({
  eyebrow,
  title,
  description,
  actions,
  className,
}: {
  eyebrow?: ReactNode;
  title?: ReactNode;
  description?: ReactNode;
  actions?: ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("flex flex-wrap items-start justify-between gap-2", className)}>
      <div className="min-w-0">
        {eyebrow ? (
          <div className="text-caption font-semibold uppercase tracking-wider text-text-dim">
            {eyebrow}
          </div>
        ) : null}
        {title ? (
          <div className="mt-0.5 text-body-sm font-semibold text-text">{title}</div>
        ) : null}
        {description ? (
          <p className="mt-1 max-w-prose text-label leading-relaxed text-text-dim">
            {description}
          </p>
        ) : null}
      </div>
      {actions ? (
        <div className="flex max-w-full flex-wrap items-center gap-2">{actions}</div>
      ) : null}
    </div>
  );
}
