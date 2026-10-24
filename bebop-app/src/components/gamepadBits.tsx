// Small presentational pieces shared by the gamepad UI (`GamepadDrive`):
// the controller glyph, a chord + label hint, and a friendly pad name.
// Moved out of the retired legged dial-in bridge (`GamepadDriver.tsx`).

export function Hint({ chord, label }: { chord: string; label: string }) {
  return (
    <span className="inline-flex items-center gap-1 whitespace-nowrap">
      <kbd className="text-[10px] font-mono px-1 py-0.5 rounded border border-border bg-bg-elev-2 text-text">
        {chord}
      </kbd>
      <span>{label}</span>
    </span>
  );
}

export function ControllerIcon() {
  return (
    <svg
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
    >
      <path d="M6 12h4M8 10v4" />
      <circle cx="15" cy="11" r="0.8" fill="currentColor" />
      <circle cx="17" cy="13" r="0.8" fill="currentColor" />
      <path d="M17.32 5H6.68a4 4 0 0 0-3.978 3.59L2 13.5A2.5 2.5 0 0 0 4.5 16h.5a2 2 0 0 0 1.789-1.106L7.62 13.5h8.76l.831 1.394A2 2 0 0 0 19 16h.5a2.5 2.5 0 0 0 2.5-2.5l-.702-4.91A4 4 0 0 0 17.32 5Z" />
    </svg>
  );
}

/// Best-effort: reduce the verbose `Gamepad.id` ("8BitDo Pro 2 (STANDARD
/// GAMEPAD Vendor: 2dc8 Product: 6003)") down to just the human name.
/// Browsers vary in how they format this; we strip any trailing
/// parenthesised section, which catches Chromium and Safari.
export function prettifyGamepadId(id: string): string {
  if (!id) return "Gamepad";
  const open = id.indexOf("(");
  return (open > 0 ? id.slice(0, open) : id).trim();
}
