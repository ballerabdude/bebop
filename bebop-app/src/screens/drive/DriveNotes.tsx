// Drive-note capture on the Drive page: REC indicator, the sticky intent
// and the event markers (hotkeys 1-4). Writes go to the videoserver's
// /intent and /event routes (`robot/driveNotes.ts`); /event answers 409
// when no segment is recording and the toast says the marker was lost.

import { useEffect, useRef, useState } from "react";

import { Button, cn, INPUT_CLASS } from "../../components/ui";
import { isEditableTarget } from "../../input";
import {
  describeNotesError,
  DRIVE_EVENTS,
  INTENT_PRESETS,
  intentIsEmpty,
} from "../../robot/driveNotes";
import type { DriveNotesState } from "./useDriveNotes";
import type { ToastTone } from "./Toasts";
import { FlagIcon } from "./icons";

type Notify = (tone: ToastTone, text: string) => void;

export function RecChip({ notes, className }: { notes: DriveNotesState; className?: string }) {
  const rec = notes.notes?.recording ?? false;
  const unknown = notes.reachable !== true;
  return (
    <span
      role="status"
      title={
        unknown
          ? "Recorder state unknown (vision unreachable)"
          : rec
            ? "A drive segment is recording; markers are saved"
            : "Not recording. Recording starts when the wheels are armed in drive mode"
      }
      className={cn(
        "inline-flex h-touch shrink-0 items-center gap-1.5 rounded-full border px-3 text-caption font-bold tracking-wider backdrop-blur-md",
        className,
        rec
          ? "border-danger/70 bg-danger/25 text-white"
          : "border-white/15 bg-black/55 text-white/65",
      )}
    >
      <span
        className={cn(
          "h-2.5 w-2.5 rounded-full",
          rec ? "animate-pulse bg-danger" : unknown ? "border border-white/40" : "bg-white/35",
        )}
        aria-hidden
      />
      {rec ? "REC" : unknown ? "REC ?" : "NOT REC"}
    </span>
  );
}

export function IntentChip({
  notes,
  open,
  onToggle,
  className,
}: {
  notes: DriveNotesState;
  open: boolean;
  onToggle: () => void;
  className?: string;
}) {
  const intent = notes.notes?.intent ?? null;
  const empty = intentIsEmpty(intent);
  const text = empty
    ? "No intent"
    : [intent!.text, intent!.to ? `→ ${intent!.to}` : ""].filter(Boolean).join(" ");
  return (
    <button
      type="button"
      onClick={onToggle}
      aria-expanded={open}
      aria-label={`Intent: ${text}. Change`}
      className={cn(
        "flex h-touch min-w-0 max-w-[14rem] items-center gap-1.5 rounded-full border px-3 text-caption font-semibold backdrop-blur-md",
        empty
          ? "border-white/15 bg-black/55 text-white/70 hover:bg-black/70"
          : "border-accent/50 bg-accent/20 text-info-fg hover:bg-accent/30",
        className,
      )}
    >
      <FlagIcon />
      <span className="truncate">{text}</span>
    </button>
  );
}

/// Intent picker body (shown in a page panel).
export function IntentPanel({
  notes,
  notify,
  onDone,
}: {
  notes: DriveNotesState;
  notify: Notify;
  onDone: () => void;
}) {
  const [custom, setCustom] = useState("");
  const intent = notes.notes?.intent ?? null;
  const empty = intentIsEmpty(intent);
  const busy = notes.busy === "intent";

  const apply = async (value: string | null) => {
    const r = value === null ? await notes.clearIntent() : await notes.setIntent(value);
    if (r.ok) {
      notify("ok", value === null ? "Intent cleared" : `Intent: ${value}`);
      setCustom("");
      onDone();
    } else {
      notify("err", describeNotesError(r));
    }
  };

  return (
    <>
      <p className="mb-3 text-label leading-relaxed text-text-dim">
        What this drive is about. It is written into every recorded segment until you change it.
      </p>
      <div className="grid grid-cols-3 gap-1.5">
        {INTENT_PRESETS.map((p) => {
          const active = !empty && intent!.text === p.text;
          return (
            <button
              key={p.text}
              type="button"
              disabled={busy}
              onClick={() => void apply(p.text)}
              aria-pressed={active}
              className={cn(
                "h-touch rounded-input border px-2 text-label font-semibold transition-colors disabled:opacity-50",
                active
                  ? "border-accent bg-accent text-white"
                  : "border-border bg-bg-elev-2 text-text hover:border-text-dim/50",
              )}
            >
              {p.label}
            </button>
          );
        })}
        <button
          type="button"
          disabled={busy || empty}
          onClick={() => void apply(null)}
          className="h-touch rounded-input border border-border px-2 text-label font-semibold text-text-dim hover:text-text disabled:opacity-40"
        >
          Clear
        </button>
      </div>
      <form
        className="mt-2 flex gap-1.5"
        onSubmit={(e) => {
          e.preventDefault();
          if (custom.trim()) void apply(custom.trim());
        }}
      >
        <input
          value={custom}
          onChange={(e) => setCustom(e.target.value)}
          placeholder="Custom, e.g. to the kitchen"
          maxLength={500}
          className={cn(INPUT_CLASS, "h-touch")}
          aria-label="Custom intent"
        />
        <Button type="submit" size="sm" className="h-touch" disabled={busy || !custom.trim()}>
          Set
        </Button>
      </form>
    </>
  );
}

/// Sends one marker and toasts the outcome.
function sendMarker(notes: DriveNotesState, notify: Notify, kind: string, label: string) {
  void notes.mark(kind).then((r) => {
    if (r.ok) notify("ok", `${label} marked`);
    else notify(r.kind === "not-recording" ? "warn" : "err", describeNotesError(r));
  });
}

/// The 1-4 marker hotkeys. Registered once by the page, so they work
/// whether or not the buttons are on screen (phones keep them in a panel).
export function useMarkerHotkeys(notes: DriveNotesState, notify: Notify) {
  const ref = useRef({ notes, notify });
  ref.current = { notes, notify };
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.repeat || e.ctrlKey || e.metaKey || e.altKey) return;
      if (isEditableTarget(e.target)) return;
      const ev = DRIVE_EVENTS.find((d) => d.hotkey === e.key);
      if (!ev) return;
      e.preventDefault();
      sendMarker(ref.current.notes, ref.current.notify, ev.kind, ev.label);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
}

/// Toolbar button for small landscape phones: opens the markers panel and
/// carries the REC state.
export function MarkButton({
  notes,
  open,
  onToggle,
}: {
  notes: DriveNotesState;
  open: boolean;
  onToggle: () => void;
}) {
  const rec = notes.notes?.recording ?? false;
  return (
    <button
      type="button"
      onClick={onToggle}
      aria-expanded={open}
      aria-label={`Markers and intent (${rec ? "recording" : "not recording"})`}
      className={cn(
        "flex h-touch shrink-0 items-center gap-1.5 rounded-full border px-3 text-caption font-bold tracking-wider backdrop-blur-md",
        rec
          ? "border-danger/70 bg-danger/25 text-white"
          : "border-white/15 bg-black/55 text-white/80 hover:bg-black/70",
      )}
    >
      <span
        className={cn("h-2.5 w-2.5 rounded-full", rec ? "animate-pulse bg-danger" : "bg-white/35")}
        aria-hidden
      />
      {rec ? "REC" : "Mark"}
    </button>
  );
}

/// Marker buttons (the 1-4 hotkeys live in `useMarkerHotkeys`).
export function EventButtons({
  notes,
  notify,
  className,
  layout = "row",
}: {
  notes: DriveNotesState;
  notify: Notify;
  className?: string;
  /// row: 4 across; column: stacked; grid: 2 x 2.
  layout?: "row" | "column" | "grid";
}) {
  const vertical = layout === "column";
  const rec = notes.notes?.recording ?? false;
  const markRef = useRef<(kind: string, label: string) => void>(() => {});
  const [flash, setFlash] = useState<string | null>(null);

  markRef.current = (kind, label) => {
    setFlash(kind);
    window.setTimeout(() => setFlash((f) => (f === kind ? null : f)), 350);
    sendMarker(notes, notify, kind, label);
  };

  return (
    <div
      role="group"
      aria-label="Drive markers"
      className={cn(
        layout === "column" ? "flex flex-col gap-1.5" : layout === "grid" ? "grid grid-cols-2 gap-1.5" : "grid grid-cols-4 gap-1.5",
        className,
      )}
    >
      {DRIVE_EVENTS.map((ev) => (
        <button
          key={ev.kind}
          type="button"
          onClick={() => markRef.current(ev.kind, ev.label)}
          title={`${ev.hint} (key ${ev.hotkey})${rec ? "" : " — not recording"}`}
          className={cn(
            "flex h-touch min-w-0 items-center justify-center gap-1.5 rounded-input border px-2.5 text-label font-semibold backdrop-blur-md transition-colors",
            flash === ev.kind
              ? "border-accent bg-accent text-white"
              : rec
                ? "border-white/20 bg-black/60 text-white hover:bg-black/75"
                : "border-white/10 bg-black/45 text-white/60 hover:bg-black/65",
            vertical && "justify-between",
          )}
        >
          <span className="truncate">{ev.label}</span>
          <kbd className="hidden rounded border border-white/20 px-1 font-mono text-micro text-white/60 pointer-fine:inline">
            {ev.hotkey}
          </kbd>
        </button>
      ))}
    </div>
  );
}
