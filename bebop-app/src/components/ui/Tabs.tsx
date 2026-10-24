import type { ReactNode } from "react";
import { NavLink } from "react-router";

import { cn } from "./cn";

export interface TabItem {
  id: string;
  label: ReactNode;
  /// Route-backed tab (NavLink). Omit for local-state tabs.
  to?: string;
}

const TAB =
  "inline-flex h-control-sm shrink-0 items-center rounded-full px-3.5 text-sm font-semibold transition-colors pointer-coarse:h-touch";

/// Horizontal tab strip; scrolls sideways on phones instead of wrapping.
/// With `to` on every item the tabs are links (the URL owns the state);
/// otherwise pass `value` + `onChange`.
export function Tabs({
  items,
  value,
  onChange,
  label,
  className,
}: {
  items: TabItem[];
  value?: string;
  onChange?: (id: string) => void;
  label: string;
  className?: string;
}) {
  return (
    <nav
      aria-label={label}
      className={cn(
        "-mx-4 flex gap-1.5 overflow-x-auto px-4 pb-1 [scrollbar-width:none] sm:mx-0 sm:px-0",
        className,
      )}
    >
      {items.map((t) =>
        t.to ? (
          <NavLink
            key={t.id}
            to={t.to}
            end
            className={({ isActive }) =>
              cn(
                TAB,
                isActive
                  ? "bg-accent/15 text-accent"
                  : "text-text-dim hover:bg-bg-elev-2 hover:text-text",
              )
            }
          >
            {t.label}
          </NavLink>
        ) : (
          <button
            key={t.id}
            type="button"
            aria-pressed={value === t.id}
            onClick={() => onChange?.(t.id)}
            className={cn(
              TAB,
              "cursor-pointer",
              value === t.id
                ? "bg-accent/15 text-accent"
                : "text-text-dim hover:bg-bg-elev-2 hover:text-text",
            )}
          >
            {t.label}
          </button>
        ),
      )}
    </nav>
  );
}
