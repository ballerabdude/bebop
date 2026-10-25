// Drive-note capture on the Drive page: REC indicator, the sticky intent
// with its runs (task, target, run id, end outcome) and the event markers
// (hotkeys 1-4). Writes go to the videoserver's
// /intent and /event routes (`robot/driveNotes.ts`); /event answers 409
// when no segment is recording and the toast says the marker was lost.

import { useEffect, useId, useRef, useState } from "react";

import { Button, cn, Field, INPUT_CLASS } from "../../components/ui";
import { isEditableTarget } from "../../input";
import {
  describeNotesError,
  DRIVE_EVENTS,
  INTENT_PRESETS,
  intentIsEmpty,
  intentLabel,
  MAX_TEXT,
  OUTCOMES,
  outcomeLabel,
  runOutcome,
  runTag,
  stopText,
  TASKS,
  taskSpec,
  UNFINISHED,
} from "../../robot/driveNotes";
import type { DriveIntent } from "../../robot/driveNotes";
import { usePlaces } from "../../robot/session";
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
  const text = empty ? "No intent" : intentLabel(intent!);
  const tag = empty ? "" : runTag(intent!.runId);
  return (
    <button
      type="button"
      onClick={onToggle}
      aria-expanded={open}
      aria-label={`Intent: ${text}${tag ? `, run ${tag}` : ""}. Change`}
      className={cn(
        "flex h-touch min-w-0 max-w-[16rem] items-center gap-1.5 rounded-full border px-3 text-caption font-semibold backdrop-blur-md",
        empty
          ? "border-white/15 bg-black/55 text-white/70 hover:bg-black/70"
          : "border-accent/50 bg-accent/20 text-info-fg hover:bg-accent/30",
        className,
      )}
    >
      <FlagIcon />
      <span className="truncate">{text}</span>
      {tag ? <span className="shrink-0 font-mono text-micro opacity-70">{tag}</span> : null}
    </button>
  );
}

interface IntentForm {
  task: string;
  target: string;
  text: string;
  from: string;
  to: string;
}

const EMPTY_FORM: IntentForm = { task: "", target: "", text: "", from: "", to: "" };
const FORM_KEYS = ["task", "target", "text", "from", "to"] as const;
const PRESET_TEXTS: readonly string[] = INTENT_PRESETS.map((p) => p.text);

function formOf(intent: DriveIntent | null): IntentForm {
  if (!intent) return EMPTY_FORM;
  const { task, target, text, from, to } = intent;
  return { task, target, text, from, to };
}

interface FormOrigin {
  runId: string;
  version: number;
  form: IntentForm;
}

/// What the form was seeded from: the run id and version identify it, the
/// form is the baseline that Update diffs against.
function originOf(intent: DriveIntent | null): FormOrigin {
  const empty = intentIsEmpty(intent);
  return {
    runId: empty ? "" : intent!.runId,
    version: intent?.version ?? 0,
    form: formOf(empty ? null : intent),
  };
}

function formIsEmpty(f: IntentForm): boolean {
  return FORM_KEYS.every((k) => !f[k].trim());
}

/// Local HH:MM of an epoch-ns stamp ("" for 0).
function clockOf(ns: number): string {
  if (!ns) return "";
  return new Date(ns / 1e6).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

/// The end buttons of the run card; unfinished gets its own "Close" row.
const END_BUTTONS = OUTCOMES.filter((o) => o.outcome !== UNFINISHED);

const SECTION_CLASS = "mb-1.5 text-caption font-semibold uppercase tracking-wider text-text-dim";

/// Intent picker body (shown in a page panel): the open run with its end
/// buttons, the task / target / free-text form, Continue / Repeat / New and
/// the recent runs.
///
/// Continue keeps the current run id: it closes the panel unchanged, or,
/// when the form was edited, updates the open run in place (mode
/// "continue"). Repeat starts a fresh run with the same meaning (the open
/// run, or the last one when none is open). New starts a fresh run from the
/// form.
export function IntentPanel({
  notes,
  notify,
  onDone,
}: {
  notes: DriveNotesState;
  notify: Notify;
  onDone: () => void;
}) {
  const intent = notes.notes?.intent ?? null;
  const recent = notes.notes?.recent ?? [];
  const empty = intentIsEmpty(intent);
  const runId = empty ? "" : intent!.runId;
  const busy = notes.busy === "intent";
  // The form is seeded from the intent and diffed against that seed (the
  // origin), not against the live intent: a poll or another viewer that
  // changes the intent while the panel is open must not turn every field
  // that differs into an edit. While the form is untouched it follows the
  // live intent (re-seeded on a new run id or version); once edited it
  // stays as typed.
  const [origin, setOrigin] = useState(() => originOf(intent));
  const [form, setForm] = useState<IntentForm>(() => origin.form);
  const dirty = FORM_KEYS.some((k) => form[k].trim() !== origin.form[k]);
  const liveVersion = intent?.version ?? 0;
  useEffect(() => {
    if (origin.runId === runId && origin.version === liveVersion) return;
    if (dirty) return;
    const next = originOf(intent);
    setOrigin(next);
    setForm(next.form);
  }, [runId, liveVersion, dirty]);
  // The open run is no longer the one the form was opened on (it ended, or
  // another viewer started a new one): Update would write onto a run the
  // operator did not see.
  const stale = origin.runId !== runId;
  const spec = taskSpec(form.task);
  const repeatId = runId || recent[0]?.runId || "";
  const places = usePlaces();
  const placeNames = [...new Set(places.places.map((p) => p.name).filter(Boolean))];
  const listId = useId();

  const set = (patch: Partial<IntentForm>) => setForm((f) => ({ ...f, ...patch }));
  const pickTask = (task: string) =>
    setForm((f) => {
      // Keep the target only when the new task takes the same kind of
      // target (and, for a closed set, the value is one of its options):
      // a hidden or reinterpreted target would be sent without the operator
      // seeing it.
      const prev = taskSpec(f.task);
      const next = taskSpec(task);
      const keep =
        !!next &&
        next.target !== "none" &&
        next.target === prev?.target &&
        (next.target !== "choice" || !!next.options?.some((o) => o.value === f.target));
      return { ...f, task, target: keep ? f.target : "" };
    });
  const pickPreset = (value: string) =>
    setForm((f) => ({
      ...f,
      task: "maneuver",
      target: value,
      // Older readers of /intent look at the text: keep it in step unless the
      // operator wrote their own.
      text: !f.text.trim() || PRESET_TEXTS.includes(f.text.trim()) ? value : f.text,
    }));

  const onContinue = async () => {
    if (!runId || !dirty) {
      onDone();
      return;
    }
    if (stale) {
      notify("err", `The run changed while you were editing (now ${runTag(runId)}): use New, or reopen the panel`);
      return;
    }
    const changed: Partial<IntentForm> = {};
    for (const k of FORM_KEYS) if (form[k].trim() !== origin.form[k]) changed[k] = form[k];
    const r = await notes.setIntent(changed, "continue");
    if (r.ok) {
      notify("ok", `Run ${runTag(runId)} updated: ${r.value ? intentLabel(r.value) : intentLabel(form)}`);
      onDone();
    } else {
      notify("err", describeNotesError(r));
    }
  };

  const onNew = async () => {
    if (formIsEmpty(form)) return;
    const r = await notes.setIntent(form, "new");
    if (r.ok) {
      const tag = r.value ? runTag(r.value.runId) : "";
      notify("ok", `New run${tag ? ` ${tag}` : ""}: ${r.value ? intentLabel(r.value) : intentLabel(form)}`);
      onDone();
    } else {
      notify("err", describeNotesError(r));
    }
  };

  const onRepeat = async (id: string) => {
    const r = await notes.repeatRun(id);
    if (r.ok) {
      const tag = r.value ? runTag(r.value.runId) : "";
      notify("ok", `Repeating ${runTag(id)} as ${tag || "a new run"}`);
      onDone();
    } else {
      notify("err", describeNotesError(r));
    }
  };

  const onEnd = async (outcome: string, ends: boolean) => {
    const r = await notes.endRun(outcome);
    if (!r.ok) {
      notify("err", describeNotesError(r));
      return;
    }
    const tag = runTag(r.value.run?.runId ?? runId);
    const where = r.value.eventRecorded ? "" : " (not recording: no marker in a file)";
    if (outcome === UNFINISHED) {
      notify("ok", `Run ${tag} closed: ${outcomeLabel(outcome)}${where}`);
    } else if (ends) {
      notify("ok", `Run ${tag} ended: ${outcomeLabel(outcome)}${where}`);
    } else {
      notify("ok", `Interruption marked on run ${tag}${where}`);
      onDone();
    }
  };

  const onClear = async () => {
    const r = await notes.clearIntent();
    if (r.ok) {
      notify("ok", "Intent cleared");
      // Empty baseline; version -1 lets the next poll re-seed it.
      setOrigin({ runId: "", version: -1, form: EMPTY_FORM });
      setForm(EMPTY_FORM);
    } else {
      notify("err", describeNotesError(r));
    }
  };

  const chip = (active: boolean) =>
    cn(
      "h-touch min-w-0 rounded-input border px-2 text-label font-semibold transition-colors disabled:opacity-50",
      active ? "border-accent bg-accent text-white" : "border-border bg-bg-elev-2 text-text hover:border-text-dim/50",
    );

  const pastRuns = recent.filter((r) => r.runId !== runId);
  const openRow = runId ? recent.find((r) => r.runId === runId) : undefined;
  const openStop = stopText(openRow?.lastStop ?? intent?.lastStop ?? null, Date.now() * 1e6);

  return (
    <>
      <p className="mb-3 text-label leading-relaxed text-text-dim">
        What this drive is about. It is written into every recorded segment until you change it; each attempt
        is its own run.
      </p>

      {empty ? null : (
        <section className="mb-3 rounded-input border border-accent/40 bg-accent/10 p-2.5" aria-label="Current run">
          <div className="mb-2 flex min-w-0 items-baseline gap-2">
            <span className="min-w-0 flex-1 truncate text-sm font-semibold text-text">{intentLabel(intent!)}</span>
            {runId ? (
              <span className="shrink-0 font-mono text-caption text-text-dim">
                {runTag(runId)}
                {intent!.runStartedNs ? ` · ${clockOf(intent!.runStartedNs)}` : ""}
              </span>
            ) : null}
          </div>
          <div className="grid grid-cols-2 gap-1.5 min-[400px]:grid-cols-4" role="group" aria-label="End run">
            {END_BUTTONS.map((o) => (
              <button
                key={o.outcome}
                type="button"
                disabled={busy || !runId}
                onClick={() => void onEnd(o.outcome, o.ends)}
                title={o.ends ? `End the run: ${o.label.toLowerCase()}` : "Mark an interruption; the run continues"}
                className={chip(false)}
              >
                {o.label}
              </button>
            ))}
          </div>
          {openStop ? <p className="mt-1.5 text-caption leading-snug text-text-dim">{openStop}</p> : null}
          <div className="mt-1.5 grid grid-cols-2 gap-1.5">
            <button
              type="button"
              disabled={busy || !runId}
              onClick={() => void onEnd(UNFINISHED, true)}
              title="Close the run without an ending; it is kept as a label"
              className="h-touch rounded-input px-2 text-label font-semibold text-text-dim hover:text-text disabled:opacity-40"
            >
              Close ({outcomeLabel(UNFINISHED).toLowerCase()})
            </button>
            <button
              type="button"
              disabled={busy}
              onClick={() => void onClear()}
              className="h-touch rounded-input px-2 text-label font-semibold text-text-dim hover:text-text disabled:opacity-40"
            >
              Clear intent
            </button>
          </div>
        </section>
      )}

      <form
        className="flex flex-col gap-2.5"
        onSubmit={(e) => {
          e.preventDefault();
          // Enter fires the highlighted primary action: Update on an edited
          // open run (so one attempt is not split into two runs), else New.
          if (runId && dirty) void onContinue();
          else void onNew();
        }}
      >
        <div>
          <div className={SECTION_CLASS}>Maneuver</div>
          <div className="grid grid-cols-3 gap-1.5 min-[400px]:grid-cols-5">
            {INTENT_PRESETS.map((p) => (
              <button
                key={p.text}
                type="button"
                disabled={busy}
                onClick={() => pickPreset(p.text)}
                aria-pressed={form.task === "maneuver" && form.target === p.text}
                className={chip(form.task === "maneuver" && form.target === p.text)}
              >
                {p.label}
              </button>
            ))}
          </div>
        </div>

        <Field label="Task">
          <select
            value={form.task}
            onChange={(e) => pickTask(e.target.value)}
            className={cn(INPUT_CLASS, "h-touch")}
            aria-label="Task"
          >
            <option value="">None (free text)</option>
            {TASKS.map((t) => (
              <option key={t.task} value={t.task}>
                {t.label}
              </option>
            ))}
          </select>
        </Field>

        {spec && spec.target !== "none" && spec.task !== "maneuver" ? (
          spec.target === "choice" ? (
            <div role="group" aria-label="Target">
              <div className={SECTION_CLASS}>Target</div>
              <div className="grid grid-cols-3 gap-1.5">
                {spec.options!.map((o) => (
                  <button
                    key={o.value}
                    type="button"
                    disabled={busy}
                    onClick={() => set({ target: o.value })}
                    aria-pressed={form.target === o.value}
                    className={chip(form.target === o.value)}
                  >
                    {o.label}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <Field label={spec.target === "place" ? "Place" : "Target"}>
              <input
                value={form.target}
                onChange={(e) => set({ target: e.target.value })}
                placeholder={spec.placeholder}
                maxLength={MAX_TEXT}
                list={spec.target === "place" ? listId : undefined}
                className={cn(INPUT_CLASS, "h-touch")}
              />
            </Field>
          )
        ) : null}

        <Field label="Note">
          <input
            value={form.text}
            onChange={(e) => set({ text: e.target.value })}
            placeholder="Free text, e.g. to the kitchen"
            maxLength={MAX_TEXT}
            className={cn(INPUT_CLASS, "h-touch")}
          />
        </Field>

        <div className="grid grid-cols-2 gap-1.5">
          <Field label="From">
            <input
              value={form.from}
              onChange={(e) => set({ from: e.target.value })}
              placeholder="Place"
              maxLength={MAX_TEXT}
              list={listId}
              className={cn(INPUT_CLASS, "h-touch")}
            />
          </Field>
          <Field label="To">
            <input
              value={form.to}
              onChange={(e) => set({ to: e.target.value })}
              placeholder="Place"
              maxLength={MAX_TEXT}
              list={listId}
              className={cn(INPUT_CLASS, "h-touch")}
            />
          </Field>
        </div>
        <datalist id={listId}>
          {placeNames.map((n) => (
            <option key={n} value={n} />
          ))}
        </datalist>

        <div className="grid grid-cols-3 gap-1.5">
          <Button
            type="button"
            size="sm"
            variant={dirty && runId ? "primary" : "secondary"}
            className="h-touch"
            disabled={busy}
            onClick={() => void onContinue()}
            title={
              runId
                ? dirty
                  ? `Update run ${runTag(runId)} with these fields`
                  : `Keep run ${runTag(runId)} unchanged`
                : "Close"
            }
          >
            {dirty && runId ? "Update" : "Continue"}
          </Button>
          <Button
            type="button"
            size="sm"
            variant="secondary"
            className="h-touch"
            disabled={busy || !repeatId}
            onClick={() => void onRepeat(repeatId)}
            title={repeatId ? `Start a new run like ${runTag(repeatId)}` : "No run to repeat"}
          >
            Repeat
          </Button>
          <Button
            type="button"
            size="sm"
            variant={dirty && !runId ? "primary" : "secondary"}
            className="h-touch"
            disabled={busy || formIsEmpty(form)}
            onClick={() => void onNew()}
            title="Start a new run from these fields"
          >
            New
          </Button>
        </div>
        {runId ? (
          <p className="text-caption leading-snug text-text-dim">
            {dirty
              ? `Update keeps run ${runTag(runId)}; New starts a separate run.`
              : `Continue keeps run ${runTag(runId)}; Repeat starts a new attempt at the same task.`}
          </p>
        ) : null}
      </form>

      {pastRuns.length ? (
        <section className="mt-3" aria-label="Recent runs">
          <div className={SECTION_CLASS}>Recent runs</div>
          <ul className="flex flex-col gap-1">
            {pastRuns.map((r) => (
              <li
                key={r.runId}
                className="flex min-w-0 items-center gap-2 rounded-input border border-border bg-bg-elev-2 py-1 pl-2.5 pr-1"
              >
                <div className="min-w-0 flex-1">
                  <div className="truncate text-label font-semibold text-text">{intentLabel(r)}</div>
                  <div className="truncate font-mono text-micro text-text-dim">
                    {[
                      runTag(r.runId),
                      clockOf(r.startedNs),
                      r.endedNs ? outcomeLabel(runOutcome(r)) : "open",
                      r.interruptions ? `${r.interruptions} interrupted` : "",
                      r.stops ? `${r.stops} ${r.stops === 1 ? "stop" : "stops"}` : "",
                      r.resumes ? `${r.resumes} resumed` : "",
                    ]
                      .filter(Boolean)
                      .join(" · ")}
                  </div>
                </div>
                <Button
                  type="button"
                  size="sm"
                  variant="ghost"
                  className="h-touch shrink-0"
                  disabled={busy}
                  onClick={() => void onRepeat(r.runId)}
                  aria-label={`Repeat run ${runTag(r.runId)}`}
                >
                  Repeat
                </Button>
              </li>
            ))}
          </ul>
        </section>
      ) : null}
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
