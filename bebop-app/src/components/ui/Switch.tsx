import { cn } from "./cn";

/// Accessible on/off switch. The hit area is 44 px tall on touch even
/// though the track is drawn smaller.
export function Switch({
  checked,
  onChange,
  label,
  disabled = false,
  className,
}: {
  checked: boolean;
  onChange: (next: boolean) => void;
  /// Accessible name (visually hidden when no visible label is next to it).
  label: string;
  disabled?: boolean;
  className?: string;
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      aria-label={label}
      disabled={disabled}
      onClick={() => onChange(!checked)}
      className={cn(
        "group relative inline-flex h-control-sm w-14 shrink-0 items-center justify-center disabled:cursor-not-allowed disabled:opacity-50 cursor-pointer pointer-coarse:h-touch",
        className,
      )}
    >
      <span
        className={cn(
          "relative h-6 w-11 rounded-full border transition-colors",
          checked ? "border-success bg-success" : "border-border bg-bg-elev-2",
        )}
        aria-hidden
      >
        <span
          className={cn(
            "absolute top-0.5 left-0.5 h-[18px] w-[18px] rounded-full bg-white shadow transition-transform",
            checked ? "translate-x-5" : "translate-x-0",
          )}
        />
      </span>
    </button>
  );
}
