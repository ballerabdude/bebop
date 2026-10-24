// The global E-STOP. Always on screen inside the robot shell, never on top
// of page content:
//
//   phones   (<768)   `EStopDock`: an in-flow strip between the page and the
//                      tab bar. E-STOP sits bottom-right under the thumb;
//                      the latch reason and a small Reset sit beside it.
//   tablets+ (>=768)  `EStopBar`: E-STOP at the right end of the status bar;
//                      `EStopStrip`, a full-width line under the status bar
//                      while latched (or after a failed send), carries the
//                      reason and Reset.
//   short    (<=500 px tall, i.e. landscape phones, which get the tablet
//             layout) no strip: the reason and Reset sit inline in the
//             status bar beside the button, so Drive keeps its height.
//
// It latches optimistically (the button flips to STOPPING the instant it is
// pressed) and follows telemetry afterwards. Space triggers it whenever the
// operator isn't typing. Reset only appears while latched and is drawn as a
// quiet secondary control so it can't be mistaken for the stop. A press that
// never reached the robot (socket down) turns the button into a pulsing
// "RETRY E-STOP" with the reason beside it, until a press succeeds.
//
// Exactly one of the two layouts is visible at a time (CSS breakpoints), and
// the Space hotkey is registered once by the shell via `useSpaceEStop`.

import { useEffect } from "react";
import type { Ref } from "react";

import { Button, cn } from "../components/ui";
import { useEStop } from "../robot/session";
import type { EStopState } from "../robot/session";
import { humanizeEStopReason } from "../robot/estopReason";

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

/// Registers the Space hotkey once for the whole shell.
export function EStopHotkey() {
  const { trigger } = useEStop();
  useSpaceEStop(() => void trigger("operator"));
  return null;
}

type EStop = ReturnType<typeof useEStop>;

function notSent(e: EStopState): boolean {
  return e.errorKind === "estop" && !e.latched && e.busy !== "estop";
}

/// "motor fault: DC bus over-voltage" style reason, or a fallback while the
/// latch is only optimistic.
function latchText(e: EStopState): string {
  if (e.reason) return humanizeEStopReason(e.reason);
  return e.confirmed ? "Latched by the firmware" : "Stop sent, waiting for the robot";
}

function StopButton({ estop, className }: { estop: EStop; className?: string }) {
  const failed = notSent(estop);
  return (
    <button
      type="button"
      onClick={() => void estop.trigger("operator")}
      aria-label={
        estop.latched
          ? `E-STOP latched${estop.reason ? `: ${estop.reason}` : ""}. Press again to re-send.`
          : failed
            ? "E-STOP not sent, press to retry (Space)"
            : "Emergency stop (Space)"
      }
      aria-pressed={estop.latched}
      className={cn(
        "inline-flex h-control-lg min-w-[56px] [@media(max-height:500px)]:h-11 shrink-0 cursor-pointer select-none items-center justify-center gap-2 rounded-card px-4 text-sm font-extrabold tracking-wider text-white shadow-lg ring-2 transition-colors active:scale-[0.97]",
        estop.latched
          ? "bg-danger/25 text-danger-fg ring-danger/70 hover:bg-danger/35"
          : failed
            ? "animate-pulse bg-danger ring-4 ring-danger-fg hover:bg-danger-press"
            : "bg-danger ring-danger/40 hover:bg-danger-press",
        className,
      )}
    >
      <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden className="shrink-0">
        <path
          d="M8.2 2h7.6L22 8.2v7.6L15.8 22H8.2L2 15.8V8.2z"
          fill="none"
          stroke="currentColor"
          strokeWidth="2.2"
          strokeLinejoin="round"
        />
      </svg>
      {estop.latched
        ? estop.confirmed
          ? "STOPPED"
          : "STOPPING"
        : failed
          ? "RETRY"
          : "E-STOP"}
    </button>
  );
}

function ResetButton({ estop, className }: { estop: EStop; className?: string }) {
  return (
    <Button
      variant="ghost"
      size="sm"
      onClick={() => void estop.reset()}
      loading={estop.busy === "reset"}
      disabled={!estop.confirmed && estop.busy === "estop"}
      aria-label="Reset the E-STOP latch"
      className={cn("border border-border text-text-dim", className)}
    >
      Reset
    </Button>
  );
}

/// Status text beside the button: the latch reason, a send failure, or a
/// short hint while everything is fine.
function StatusText({ estop, compact }: { estop: EStop; compact: boolean }) {
  if (!compact && (estop.error || estop.latched)) {
    const failed = !!estop.error;
    return (
      <div role={failed ? "alert" : "status"} className="flex min-w-0 items-baseline gap-2 text-sm">
        <span className="shrink-0 text-label font-bold uppercase tracking-wider text-danger-fg">
          {failed
            ? estop.errorKind === "estop"
              ? "E-STOP not sent"
              : "Reset failed"
            : estop.confirmed
              ? "E-STOP latched"
              : "Stopping"}
        </span>
        <span className="min-w-0 truncate text-text">
          {failed ? estop.error : latchText(estop)}
        </span>
      </div>
    );
  }
  if (estop.error) {
    return (
      <div role="alert" className="min-w-0 leading-tight">
        <div className="text-label font-bold uppercase tracking-wider text-danger-fg">
          {estop.errorKind === "estop" ? "E-STOP not sent" : "Reset failed"}
        </div>
        <div className="line-clamp-2 text-label text-text-dim">{estop.error}</div>
      </div>
    );
  }
  if (estop.latched) {
    return (
      <div role="status" className="min-w-0 leading-tight">
        <div className="text-label font-bold uppercase tracking-wider text-danger-fg">
          {estop.confirmed ? "E-STOP latched" : "Stopping"}
        </div>
        <div className={cn("text-label text-text", compact ? "line-clamp-2" : "truncate")}>
          {latchText(estop)}
        </div>
      </div>
    );
  }
  return compact ? (
    <div className="min-w-0 text-label leading-tight text-text-dim">
      Emergency stop
      <span className="block">Halts every motor at once.</span>
    </div>
  ) : null;
}

/// Phone layout (md:hidden). Normally an in-flow strip above the tab bar.
/// `floating` (the full-bleed Drive page) pins it bottom-right above any
/// page overlay instead (z-40, so fullscreen video at z-35 can't cover it).
/// Its footprint changes with state (about 150 x 70 px idle, about
/// 240 x 112 px latched with the reason box), so pages pad against the
/// measured `--estop-float-w` / `--estop-float-h` the shell publishes on
/// <html> (distance from the right / bottom viewport edge, safe area
/// included) rather than a fixed number.
export function EStopDock({
  className,
  floating = false,
  rootRef,
}: {
  className?: string;
  floating?: boolean;
  rootRef?: Ref<HTMLDivElement>;
}) {
  const estop = useEStop();
  if (floating) {
    return (
      <div
        ref={rootRef}
        data-estop-float=""
        className={cn(
          "fixed right-3 bottom-[max(1rem,env(safe-area-inset-bottom))] z-40 flex max-w-[calc(100vw-1.5rem)] flex-col items-end gap-2 md:hidden",
          className,
        )}
      >
        {estop.latched || estop.error ? (
          <div className="max-w-[16rem] rounded-input border border-danger/50 bg-bg/90 px-2.5 py-1.5 shadow-lg backdrop-blur-md">
            <StatusText estop={estop} compact />
          </div>
        ) : null}
        <div className="flex items-center gap-2">
          {estop.latched ? <ResetButton estop={estop} className="bg-bg/90 backdrop-blur-md" /> : null}
          <StopButton estop={estop} />
        </div>
      </div>
    );
  }
  return (
    <div
      ref={rootRef}
      className={cn(
        "flex shrink-0 items-center gap-2 border-t px-3 py-2 md:hidden",
        estop.latched || estop.error
          ? "border-danger/40 bg-danger/10"
          : "border-border bg-bg-elev/80",
        className,
      )}
    >
      <div className="min-w-0 flex-1">
        <StatusText estop={estop} compact />
      </div>
      {estop.latched ? <ResetButton estop={estop} /> : null}
      <StopButton estop={estop} className="min-w-[132px]" />
    </div>
  );
}

/// Tablet/desktop: the button at the right end of the status bar. On short
/// viewports (landscape phones) the latch reason and Reset ride along here
/// instead of in `EStopStrip`.
export function EStopBar({ className }: { className?: string }) {
  const estop = useEStop();
  const showReason = estop.latched || !!estop.error;
  return (
    <div className={cn("hidden items-center gap-2 md:flex", className)}>
      {showReason ? (
        <span
          role={estop.error ? "alert" : "status"}
          className="hidden max-w-[19rem] min-w-0 truncate text-label font-semibold text-danger-fg [@media(max-height:500px)]:block"
        >
          {estop.error
            ? `${estop.errorKind === "estop" ? "E-STOP not sent" : "Reset failed"}: ${estop.error}`
            : latchText(estop)}
        </span>
      ) : null}
      {estop.latched ? (
        <ResetButton estop={estop} className="hidden [@media(max-height:500px)]:inline-flex" />
      ) : null}
      <StopButton estop={estop} />
    </div>
  );
}

/// Tablet/desktop: reason + Reset under the status bar while latched or
/// after a failed send. Renders nothing otherwise.
export function EStopStrip({ className }: { className?: string }) {
  const estop = useEStop();
  if (!estop.latched && !estop.error) return null;
  return (
    <div
      className={cn(
        "hidden items-center gap-3 border-b border-danger/40 bg-danger/10 px-4 py-1.5 md:[@media(min-height:501px)]:flex",
        className,
      )}
    >
      <div className="min-w-0 flex-1">
        <StatusText estop={estop} compact={false} />
      </div>
      {estop.latched ? (
        <span className="hidden text-label text-text-dim lg:inline">Reset only when the robot is safe.</span>
      ) : null}
      {estop.latched ? <ResetButton estop={estop} /> : null}
    </div>
  );
}
