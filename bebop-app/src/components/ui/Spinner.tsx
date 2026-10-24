import { cn } from "./cn";

export function Spinner({ large = false, className }: { large?: boolean; className?: string }) {
  return (
    <span
      className={cn(
        "inline-block shrink-0 animate-spin rounded-full border-current/25 border-t-current",
        large ? "h-9 w-9 border-[3px] text-accent" : "h-4 w-4 border-2",
        className,
      )}
      aria-hidden
    />
  );
}
