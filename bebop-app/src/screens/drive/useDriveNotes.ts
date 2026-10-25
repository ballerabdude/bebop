// React binding for the drive-note client (`robot/driveNotes.ts`): polls
// GET /notes for the REC state, the sticky intent and the recent runs, and
// wraps the writes (intent and run calls share the "intent" busy label).
// `check` fetches /notes now, for decisions that must not act on a poll up
// to 2 s old (is a run open before arming?).

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { createDriveNotes, notesBase } from "../../robot/driveNotes";
import type {
  DriveEvent,
  DriveIntent,
  EndRunResult,
  IntentFields,
  IntentMode,
  NotesResult,
  NotesSnapshot,
  ResumeResult,
} from "../../robot/driveNotes";

const POLL_MS = 2000;

export interface DriveNotesState {
  /// null until the first answer; false when /notes is unreachable.
  reachable: boolean | null;
  notes: NotesSnapshot | null;
  busy: string | null;
  mark: (kind: string, note?: string) => Promise<NotesResult<DriveEvent>>;
  setIntent: (fields: IntentFields, mode?: IntentMode) => Promise<NotesResult<DriveIntent | null>>;
  clearIntent: () => Promise<NotesResult<DriveIntent | null>>;
  repeatRun: (runId: string) => Promise<NotesResult<DriveIntent | null>>;
  endRun: (outcome: string, note?: string) => Promise<NotesResult<EndRunResult>>;
  resumeRun: (note?: string) => Promise<NotesResult<ResumeResult>>;
  /// GET /notes now (also updates `notes` / `reachable`).
  check: () => Promise<NotesResult<NotesSnapshot>>;
  refresh: () => Promise<void>;
}

export function useDriveNotes(host: string, enabled = true): DriveNotesState {
  const client = useMemo(() => createDriveNotes(notesBase(host)), [host]);
  const [notes, setNotes] = useState<NotesSnapshot | null>(null);
  const [reachable, setReachable] = useState<boolean | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const alive = useRef(true);

  const check = useCallback(async () => {
    const r = await client.getNotes();
    if (alive.current) {
      if (r.ok) {
        setNotes(r.value);
        setReachable(true);
      } else {
        setReachable(false);
      }
    }
    return r;
  }, [client]);
  const refresh = useCallback(async () => {
    await check();
  }, [check]);

  useEffect(() => {
    alive.current = true;
    if (!enabled) return;
    void refresh();
    const t = window.setInterval(() => {
      if (!document.hidden) void refresh();
    }, POLL_MS);
    return () => {
      alive.current = false;
      window.clearInterval(t);
    };
  }, [enabled, refresh]);

  const wrap = useCallback(
    async <T,>(label: string, fn: () => Promise<NotesResult<T>>): Promise<NotesResult<T>> => {
      setBusy(label);
      try {
        const r = await fn();
        if (!r.ok && r.kind === "not-recording" && alive.current) {
          setNotes((n) => (n ? { ...n, recording: false } : n));
        }
        void refresh();
        return r;
      } finally {
        if (alive.current) setBusy(null);
      }
    },
    [refresh],
  );

  return {
    reachable,
    notes,
    busy,
    refresh,
    check,
    mark: (kind, note = "") => wrap(`event:${kind}`, () => client.addEvent(kind, note)),
    setIntent: (fields, mode = "new") => wrap("intent", () => client.setIntent(fields, mode)),
    clearIntent: () => wrap("intent", () => client.clearIntent()),
    repeatRun: (runId) => wrap("intent", () => client.repeatRun(runId)),
    endRun: (outcome, note = "") => wrap("intent", () => client.endRun(outcome, note)),
    resumeRun: (note = "") => wrap("intent", () => client.resumeRun(note)),
  };
}
