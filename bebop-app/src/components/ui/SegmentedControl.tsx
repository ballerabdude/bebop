import { cn } from "./cn";

export interface SegmentOption<T extends string> {
  value: T;
  label: string;
  disabled?: boolean;
}

/// Mutually exclusive choice rendered as a pill row (radio group).
export function SegmentedControl<T extends string>({
  value,
  options,
  onChange,
  disabled = false,
  label,
  className,
  size = "md",
}: {
  value: T;
  options: SegmentOption<T>[];
  onChange: (v: T) => void;
  disabled?: boolean;
  label: string;
  className?: string;
  size?: "sm" | "md";
}) {
  return (
    <div
      role="radiogroup"
      aria-label={label}
      className={cn("inline-flex rounded-card bg-bg-elev-2 p-0.5", className)}
    >
      {options.map((o) => {
        const active = o.value === value;
        return (
          <button
            key={o.value}
            type="button"
            role="radio"
            aria-checked={active}
            disabled={disabled || o.disabled}
            onClick={() => onChange(o.value)}
            className={cn(
              "flex-1 rounded-[calc(var(--radius-card)-3px)] px-3 font-semibold transition-colors disabled:cursor-not-allowed disabled:opacity-50 cursor-pointer",
              size === "sm"
                ? "h-8 text-label pointer-coarse:h-10"
                : "h-control-sm text-sm pointer-coarse:h-10",
              active ? "bg-accent text-white shadow-sm" : "text-text-dim hover:text-text",
            )}
          >
            {o.label}
          </button>
        );
      })}
    </div>
  );
}
