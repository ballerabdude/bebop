import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import type { ReactNode } from "react";
import { NavLink, useLocation } from "react-router";

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
  const ref = useRef<HTMLElement>(null);
  const { pathname } = useLocation();
  // Which edges have hidden tabs beyond them, for the fade hint.
  const [edges, setEdges] = useState({ left: false, right: false });
  const measure = useCallback(() => {
    const el = ref.current;
    if (!el) return;
    const left = el.scrollLeft > 2;
    const right = el.scrollLeft + el.clientWidth < el.scrollWidth - 2;
    setEdges((e) => (e.left === left && e.right === right ? e : { left, right }));
  }, []);
  // Keep the selected tab in view on phones (e.g. a deep link to the last tab).
  useLayoutEffect(() => {
    const el = ref.current;
    const active = el?.querySelector<HTMLElement>('[aria-current="page"], [aria-pressed="true"]');
    if (el && active && el.scrollWidth > el.clientWidth) {
      const a = active.offsetLeft - el.offsetLeft;
      if (a < el.scrollLeft || a + active.offsetWidth > el.scrollLeft + el.clientWidth) {
        el.scrollLeft = Math.max(0, a - (el.clientWidth - active.offsetWidth) / 2);
      }
    }
    measure();
  }, [pathname, value, measure]);
  useEffect(() => {
    window.addEventListener("resize", measure);
    return () => window.removeEventListener("resize", measure);
  }, [measure]);
  const fade =
    edges.left && edges.right
      ? "[mask-image:linear-gradient(to_right,transparent,black_28px,black_calc(100%-28px),transparent)]"
      : edges.right
        ? "[mask-image:linear-gradient(to_right,black_calc(100%-36px),transparent)]"
        : edges.left
          ? "[mask-image:linear-gradient(to_right,transparent,black_36px)]"
          : "";
  return (
    <nav
      ref={ref}
      aria-label={label}
      onScroll={measure}
      className={cn(
        "-mx-4 flex gap-1.5 overflow-x-auto px-4 pb-1 [scrollbar-width:none] sm:mx-0 sm:px-0",
        fade,
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
