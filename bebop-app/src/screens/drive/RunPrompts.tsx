// Run prompts around arming and disarming on the Drive page.
//
// The recorder never guesses what a break in a run was. Before the wheels
// arm with a run still open, the operator says whether this drive continues
// it (POST /intent {"resume": true}) or whether it closes unfinished
// ({"end": "unfinished"}), with an optional note either way; Cancel arms
// nothing. After "Stop driving" (the normal stop: every wheel disarmed) the
// end-of-run sheet offers the endings, or keeps the run open so the next
// start asks again. Both labels go into the run for training.

import { useState } from "react";

import { Button, cn, Dialog, Field, INPUT_CLASS } from "../../components/ui";
import {
  describeNotesError,
  MAX_TEXT,
  OUTCOMES,
  outcomeLabel,
  resumePrompt,
  runTag,
  taskLabel,
  UNFINISHED,
} from "../../robot/driveNotes";
import type { RecentRun } from "../../robot/driveNotes";
import type { ToastTone } from "./Toasts";
import type { DriveNotesState } from "./useDriveNotes";

type Notify = (tone: ToastTone, text: string, ms?: number) => void;

/// Local HH:MM of an epoch-ns stamp.
function clockOf(ns: number): string {
  return new Date(ns / 1e6).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function NoteField({ value, onChange, id }: { value: string; onChange: (v: string) => void; id?: string }) {
  return (
    <Field label="Anything to note?">
      <input
        id={id}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder="Optional, e.g. moved the chair, battery swap"
        maxLength={MAX_TEXT}
        className={cn(INPUT_CLASS, "h-touch")}
      />
    </Field>
  );
}

/// "Run still open" modal shown before arming. `onArm` runs after Continue
/// or Close it (whether or not the note call succeeded: the drive itself
/// never depends on the vision process); Cancel and Esc only close.
export function ResumeRunDialog({
  run,
  notes,
  notify,
  onArm,
  onClose,
}: {
  run: RecentRun;
  notes: DriveNotesState;
  notify: Notify;
  onArm: () => void;
  onClose: () => void;
}) {
  const [note, setNote] = useState("");
  const [busy, setBusy] = useState<"resume" | "close" | null>(null);
  const prompt = resumePrompt(run, Date.now() * 1e6, clockOf);
  const tag = runTag(run.runId);

  const decide = async (choice: "resume" | "close") => {
    setBusy(choice);
    try {
      if (choice === "resume") {
        const r = await notes.resumeRun(note);
        if (r.ok) notify("ok", `Continuing run ${tag}`);
        else notify("warn", `Continue not saved (${describeNotesError(r)}); driving anyway`, 5000);
      } else {
        const r = await notes.endRun(UNFINISHED, note);
        if (r.ok) notify("ok", `Run ${tag} closed: ${outcomeLabel(UNFINISHED)}`);
        else notify("warn", `Close not saved (${describeNotesError(r)}); driving anyway`, 5000);
      }
    } finally {
      setBusy(null);
    }
    onClose();
    onArm();
  };

  return (
    <Dialog
      open
      onClose={() => {
        if (!busy) onClose();
      }}
      title={prompt.title}
      footer={
        // Phones stack the footer reversed: Cancel lands bottom-most,
        // nearest the E-STOP dock, as the Dialog contract asks.
        <>
          <Button variant="secondary" disabled={!!busy} onClick={onClose}>
            Cancel
          </Button>
          <Button
            variant="secondary"
            disabled={!!busy}
            loading={busy === "close"}
            onClick={() => void decide("close")}
            title={`Close run ${tag} as ${outcomeLabel(UNFINISHED).toLowerCase()}, then drive`}
          >
            Close it
          </Button>
          <Button
            variant="primary"
            data-autofocus
            disabled={!!busy}
            loading={busy === "resume"}
            onClick={() => void decide("resume")}
            title={`Continue run ${tag}, then drive`}
          >
            Continue
          </Button>
        </>
      }
    >
      <div className="space-y-1">
        {prompt.lines.map((l) => (
          <p key={l} className="text-text">
            {l}
          </p>
        ))}
        {prompt.stop ? <p>{prompt.stop}</p> : null}
        <p className="font-mono text-caption">{tag}</p>
      </div>
      <p>Continue this run, or close it unfinished and drive without one?</p>
      <form
        onSubmit={(e) => {
          e.preventDefault();
          if (!busy) void decide("resume");
        }}
      >
        <NoteField value={note} onChange={setNote} />
      </form>
    </Dialog>
  );
}

/// The end buttons of the sheet (unfinished is "Keep it open" declined
/// later, interrupted is not an ending).
const SHEET_OUTCOMES = OUTCOMES.filter((o) => o.ends && o.outcome !== UNFINISHED);

/// End-of-run sheet shown after "Stop driving" disarmed the wheels while a
/// run is open. Keep it open (or Esc) leaves the run as it is.
export function EndRunSheet({
  run,
  notes,
  notify,
  onClose,
}: {
  run: RecentRun;
  notes: DriveNotesState;
  notify: Notify;
  onClose: () => void;
}) {
  const [note, setNote] = useState("");
  const [busy, setBusy] = useState<string | null>(null);
  const tag = runTag(run.runId);
  const head = taskLabel(run) || run.text || tag;

  /// A failed end keeps the sheet open with the note typed so far, so the
  /// operator can retry (or Keep it open) instead of re-entering it.
  const end = async (outcome: string) => {
    setBusy(outcome);
    let done = false;
    try {
      const r = await notes.endRun(outcome, note);
      // No run open any more (closed elsewhere): nothing left to retry.
      done = r.ok || r.kind === "no-run";
      if (r.ok) notify("ok", `Run ${tag} ended: ${outcomeLabel(outcome)}`);
      else if (done) notify("warn", describeNotesError(r), 5000);
      else notify("err", `Not saved (${describeNotesError(r)}): try again or keep it open`, 5000);
    } finally {
      setBusy(null);
    }
    if (done) onClose();
  };

  return (
    <Dialog
      open
      onClose={() => {
        if (!busy) onClose();
      }}
      title={`Wheels disarmed. How did ${head} go?`}
      footer={
        <>
          <Button variant="secondary" disabled={!!busy} onClick={onClose} title="The next start asks again">
            Keep it open
          </Button>
          <div className="grid grid-cols-3 gap-2 sm:flex">
            {SHEET_OUTCOMES.map((o) => (
              <Button
                key={o.outcome}
                variant="primary"
                disabled={!!busy}
                loading={busy === o.outcome}
                onClick={() => void end(o.outcome)}
              >
                {o.label}
              </Button>
            ))}
          </div>
        </>
      }
    >
      <p className="font-mono text-caption">{tag}</p>
      <NoteField value={note} onChange={setNote} />
    </Dialog>
  );
}
