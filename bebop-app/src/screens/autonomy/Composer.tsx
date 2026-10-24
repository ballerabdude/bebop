// Text box, hold-to-talk mic, "Speak replies" and "New conversation".
//
// Enter sends, Shift+Enter breaks the line, Esc stops the running task.
// The mic is hold-to-talk; a quick tap latches it instead (tap again to
// send), which also covers the permission prompt stealing the pointer.

import { useEffect, useRef, useState } from "react";
import type { PointerEvent as ReactPointerEvent } from "react";

import { cn, Switch } from "../../components/ui";
import { MicIcon, MicOffIcon, PlusIcon, SendIcon, StopIcon } from "./icons";

const TAP_MS = 300;

export interface ComposerProps {
  /// The agent socket is open.
  online: boolean;
  /// Something is in flight; sending is replaced by Stop.
  busy: boolean;
  /// Live mic capture.
  listening: boolean;
  /// Why the mic can't be used, or null.
  micBlocked: string | null;
  speak: boolean;
  onSpeakChange: (v: boolean) => void;
  onSend: (text: string) => boolean;
  onStop: () => void;
  onMicDown: () => void;
  onMicUp: () => void;
  onNewConversation: () => void;
  canReset: boolean;
  draft: string;
  onDraftChange: (v: string) => void;
  placeholder?: string;
}

export function Composer(p: ComposerProps) {
  const taRef = useRef<HTMLTextAreaElement>(null);
  const pressAt = useRef(0);
  const [latched, setLatched] = useState(false);

  // Grow with the text up to ~5 lines.
  useEffect(() => {
    const ta = taRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = `${Math.min(ta.scrollHeight, 132)}px`;
  }, [p.draft]);

  useEffect(() => {
    if (!p.listening) setLatched(false);
  }, [p.listening]);

  const canSend = p.online && !p.busy && p.draft.trim().length > 0;
  const micDisabled = !!p.micBlocked || !p.online || (p.busy && !p.listening);

  function submit() {
    if (!canSend) return;
    if (p.onSend(p.draft)) p.onDraftChange("");
  }

  function micDown(e: ReactPointerEvent<HTMLButtonElement>) {
    if (micDisabled || e.button !== 0) return;
    e.preventDefault();
    try {
      e.currentTarget.setPointerCapture(e.pointerId);
    } catch {
      /* ignore */
    }
    if (latched) {
      // Second tap of tap-to-talk: send.
      setLatched(false);
      p.onMicUp();
      pressAt.current = 0;
      return;
    }
    pressAt.current = performance.now();
    p.onMicDown();
  }

  function micUp() {
    if (!pressAt.current) return;
    const held = performance.now() - pressAt.current;
    pressAt.current = 0;
    if (held < TAP_MS) {
      setLatched(true); // keep listening until the next tap
      return;
    }
    p.onMicUp();
  }

  const micLabel = p.micBlocked
    ? p.micBlocked
    : p.listening
      ? latched
        ? "Listening. Tap to send."
        : "Listening. Release to send."
      : "Hold to talk (or tap to start, tap again to send)";

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-end gap-2">
        <div
          className={cn(
            "flex min-w-0 flex-1 items-end rounded-[22px] border bg-bg-elev-2 pl-4 pr-1.5 transition-colors focus-within:border-accent",
            p.online ? "border-border" : "border-border/60 opacity-80",
          )}
        >
          <textarea
            ref={taRef}
            rows={1}
            value={p.draft}
            onChange={(e) => p.onDraftChange(e.currentTarget.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
                e.preventDefault();
                submit();
              } else if (e.key === "Escape" && p.busy) {
                e.preventDefault();
                p.onStop();
              }
            }}
            placeholder={p.placeholder ?? "Ask the robot…"}
            aria-label="Message to the robot"
            enterKeyHint="send"
            // The pill's border shows focus; the global :focus-visible ring
            // (unlayered CSS, beats utilities) would draw a box inside it.
            style={{ outline: "none" }}
            className="max-h-[132px] min-h-[44px] w-full focus-visible:outline-none min-w-0 resize-none bg-transparent py-[11px] text-base leading-[22px] text-text outline-none placeholder:text-text-dim/70 sm:text-sm sm:leading-[22px]"
          />
          {p.busy ? (
            <button
              type="button"
              onClick={p.onStop}
              aria-label="Stop task"
              title="Stop task (Esc)"
              className="mb-[3px] inline-flex h-[38px] w-[38px] shrink-0 cursor-pointer items-center justify-center rounded-full bg-text text-bg hover:bg-text/85 pointer-coarse:h-touch pointer-coarse:w-touch pointer-coarse:mb-0"
            >
              <StopIcon size={16} />
            </button>
          ) : (
            <button
              type="button"
              onClick={submit}
              disabled={!canSend}
              aria-label="Send"
              title="Send (Enter)"
              className="mb-[3px] inline-flex h-[38px] w-[38px] shrink-0 cursor-pointer items-center justify-center rounded-full bg-accent text-white transition-opacity hover:bg-accent-press disabled:cursor-not-allowed disabled:opacity-35 pointer-coarse:h-touch pointer-coarse:w-touch pointer-coarse:mb-0"
            >
              <SendIcon size={18} />
            </button>
          )}
        </div>
        <button
          type="button"
          aria-label={micLabel}
          title={micLabel}
          aria-pressed={p.listening}
          disabled={micDisabled}
          onPointerDown={micDown}
          onPointerUp={micUp}
          onPointerCancel={micUp}
          onContextMenu={(e) => e.preventDefault()}
          onKeyDown={(e) => {
            // Keyboard users toggle with Enter (Space is the E-STOP key).
            if (e.key === "Enter" && !micDisabled) {
              e.preventDefault();
              if (p.listening) p.onMicUp();
              else {
                setLatched(true);
                p.onMicDown();
              }
            }
          }}
          className={cn(
            "relative inline-flex h-14 w-14 shrink-0 touch-none select-none items-center justify-center rounded-full border transition-colors",
            p.listening
              ? "border-danger bg-danger text-white shadow-[0_0_0_6px_rgba(255,90,95,0.18)]"
              : "border-border bg-bg-elev-2 text-text hover:bg-bg-elev",
            micDisabled ? "cursor-not-allowed opacity-45" : "cursor-pointer",
          )}
        >
          {p.micBlocked ? <MicOffIcon size={22} /> : <MicIcon size={22} />}
          {p.listening ? (
            <span className="absolute inset-0 animate-ping rounded-full border-2 border-danger/60" aria-hidden />
          ) : null}
        </button>
      </div>
      <div className="flex flex-wrap items-center justify-between gap-x-3 gap-y-1">
        <label className="flex min-h-touch cursor-pointer items-center gap-0.5 text-body-sm text-text-dim md:min-h-0">
          <Switch label="Speak replies" checked={p.speak} onChange={p.onSpeakChange} className="-ml-1.5" />
          Speak replies
        </label>
        <p
          className={cn(
            "order-last w-full text-caption leading-snug sm:order-none sm:w-auto sm:flex-1 sm:text-center",
            p.listening ? "font-semibold text-danger-fg" : "text-text-dim",
            !p.micBlocked && !p.listening && "pointer-coarse:hidden",
          )}
          aria-live="polite"
        >
          {p.micBlocked
            ? p.micBlocked
            : p.listening
              ? latched
                ? "Listening… tap the mic to send"
                : "Listening… release to send"
              : "Hold the mic to talk · Enter to send"}
        </p>
        <button
          type="button"
          onClick={p.onNewConversation}
          disabled={!p.canReset}
          className="inline-flex h-control-sm items-center gap-1.5 rounded-card px-2.5 text-label font-semibold text-text-dim hover:bg-bg-elev-2 hover:text-text disabled:cursor-not-allowed disabled:opacity-45 pointer-coarse:h-touch cursor-pointer"
        >
          <PlusIcon size={15} />
          New conversation
        </button>
      </div>
    </div>
  );
}
