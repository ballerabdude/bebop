// The open-run question asked before any arm that starts a drive.
//
// Shared by the Drive page and Diagnostics › Motors, so every path that ends
// with all wheels armed (Start driving, Arm all, the pad's L3, one wheel
// toggle after another) asks whether a run left open continues or closes
// unfinished. The recorder starts a segment as soon as every wheel is
// armed, so asking only on the Drive page's Start button would let a drive
// slip into an open run unasked.

import { useCallback, useRef, useState } from "react";
import type { ReactNode } from "react";

import { openRun } from "../../robot/driveNotes";
import type { RecentRun } from "../../robot/driveNotes";
import { ResumeRunDialog } from "./RunPrompts";
import type { ToastTone } from "./Toasts";
import type { DriveNotesState } from "./useDriveNotes";

type Notify = (tone: ToastTone, text: string, ms?: number) => void;

export interface RunStartGuard {
  /// Arms through `arm` once the open-run question is settled.
  requestStart: (arm: () => void) => Promise<void>;
  /// GET /notes is in flight.
  checking: boolean;
  /// The "Run still open" dialog is showing.
  prompting: boolean;
  /// The dialog element (null when hidden); render it once in the page.
  dialog: ReactNode;
}

/// No run open arms at once; an open run always asks first; an unreachable
/// /notes arms anyway (driving never depends on the vision process) and
/// says the run state is unknown.
export function useRunStartGuard(notes: DriveNotesState, notify: Notify): RunStartGuard {
  const [resumeFor, setResumeFor] = useState<{ run: RecentRun; arm: () => void } | null>(null);
  const [checking, setChecking] = useState(false);
  const checkingRef = useRef(false);
  const checkNotes = notes.check;

  const requestStart = useCallback(
    async (arm: () => void) => {
      if (checkingRef.current) return;
      checkingRef.current = true;
      setChecking(true);
      try {
        const r = await checkNotes();
        if (!r.ok) {
          notify("warn", "Run state unknown (vision unreachable): starting without the run check", 5000);
          arm();
          return;
        }
        const open = openRun(r.value);
        if (open) setResumeFor({ run: open, arm });
        else arm();
      } finally {
        checkingRef.current = false;
        setChecking(false);
      }
    },
    [checkNotes, notify],
  );

  const dialog = resumeFor ? (
    <ResumeRunDialog
      key={resumeFor.run.runId}
      run={resumeFor.run}
      notes={notes}
      notify={notify}
      onArm={resumeFor.arm}
      onClose={() => setResumeFor(null)}
    />
  ) : null;

  return { requestStart, checking, prompting: resumeFor !== null, dialog };
}
