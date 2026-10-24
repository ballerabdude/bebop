// The global E-STOP. Always on screen inside the robot shell: top-right on
// tablets/desktop, floating bottom-right above the tab bar on phones.
//
// It latches optimistically (the button flips to LATCHED the instant it is
// pressed) and follows telemetry afterwards. Space triggers it whenever the
// operator isn't typing. Reset only appears while latched. A press that
// never reached the robot (socket down) turns the button into a pulsing
// "NOT SENT" retry with the reason beside it, until a press succeeds.

import { useEffect } from "react";

import { Button, cn } from "../components/ui";
import { useEStop } from "../robot/session";

function isTypingTarget(t: EventTarget | null): boolean {
  if (!(t instanceof HTMLElement)) return false;
  if (t.isContentEditable) return true;
  const tag = t.tagName;
  if (tag === "TEXTAREA" || tag === "SELECT") return true;
  if (tag === "INPUT") {
    const type = (t as HTMLInputElement).type;
    return !["checkbox", "radio", "button", "submit", "range", "reset"].includes(type);
  }
  return false;
}

export function useSpaceEStop(trigger: () => void): void {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.code !== "Space" && e.key !== " ") return;
      if (e.repeat || e.ctrlKey || e.metaKey || e.altKey) return;
      if (isTypingTarget(e.target)) return;
      e.preventDefault();
      trigger();
    };
    window.addEventListener("keydown", onKey, { capture: true });
    return () => window.removeEventListener("keydown", onKey, { capture: true });
  }, [trigger]);
}

export function EStopButton({ className }: { className?: string }) {
  const estop = useEStop();
  const { trigger } = estop;
  useSpaceEStop(() => void trigger("operator"));
  const notSent = estop.errorKind === "estop" && !estop.latched && estop.busy !== "estop";
  return (
    <div className={cn("flex items-center gap-2", className)}>
      {estop.error ? (
        <span
          role="alert"
          title={estop.error}
          className="max-w-[11rem] rounded-input border border-danger/60 bg-danger/15 px-2 py-1 text-caption font-semibold leading-tight text-danger-fg"
        >
          {estop.errorKind === "estop" ? "E-STOP NOT SENT" : "Reset failed"}
          <span className="block truncate font-normal text-text-dim">{estop.error}</span>
        </span>
      ) : null}
      {estop.latched ? (
        <Button
          variant="secondary"
          size="lg"
          onClick={() => void estop.reset()}
          loading={estop.busy === "reset"}
          disabled={!estop.confirmed && estop.busy === "estop"}
          title={estop.reason || "Reset the E-STOP latch"}
          className="h-control-lg"
        >
          Reset
        </Button>
      ) : null}
      <button
        type="button"
        onClick={() => void trigger("operator")}
        aria-label={
          estop.latched
            ? "E-STOP latched"
            : notSent
              ? "E-STOP not sent, press to retry (Space)"
              : "Emergency stop (Space)"
        }
        aria-pressed={estop.latched}
        className={cn(
          "inline-flex h-control-lg min-w-[56px] shrink-0 cursor-pointer select-none items-center justify-center rounded-card px-4 text-sm font-extrabold tracking-wider text-white shadow-lg ring-2 ring-danger/40 transition-colors active:scale-[0.97]",
          estop.latched
            ? "bg-danger-press ring-danger-fg/60"
            : notSent
              ? "animate-pulse bg-danger ring-4 ring-danger-fg hover:bg-danger-press"
              : "bg-danger hover:bg-danger-press",
        )}
      >
        {estop.latched
          ? estop.confirmed
            ? "STOPPED"
            : "STOPPING"
          : notSent
            ? "RETRY E-STOP"
            : "E-STOP"}
      </button>
    </div>
  );
}
