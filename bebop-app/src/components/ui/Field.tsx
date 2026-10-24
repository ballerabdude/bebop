import type { ReactNode } from "react";

/// Shared input styling (text inputs, selects, textareas).
export const INPUT_CLASS =
  "w-full min-w-0 h-control-md rounded-input border border-border bg-bg-elev-2 px-3 text-base text-text outline-none transition-colors focus:border-accent sm:text-sm disabled:opacity-60";
export const TEXTAREA_CLASS =
  "w-full min-w-0 min-h-24 rounded-input border border-border bg-bg-elev-2 px-3 py-2.5 text-base text-text outline-none transition-colors focus:border-accent sm:text-sm";

export function Field({
  label,
  children,
  hint,
}: {
  label: string;
  children: ReactNode;
  hint?: ReactNode;
}) {
  return (
    <label className="flex min-w-0 flex-col gap-1.5">
      <span className="text-body-sm text-text-dim">{label}</span>
      {children}
      {hint ? <span className="text-caption leading-snug text-text-dim">{hint}</span> : null}
    </label>
  );
}
