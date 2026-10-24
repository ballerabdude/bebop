// React binding for the drive-note client (`robot/driveNotes.ts`): polls
// GET /notes for the REC state and the sticky intent, and wraps the writes.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { createDriveNotes, notesBase } from "../../robot/driveNotes";
import type { DriveEvent, DriveIntent, NotesResult, NotesSnapshot } from "../../robot/driveNotes";

const POLL_MS = 2000;

export interface DriveNotesState {
  /// null until the first answer; false when /notes is unreachable.
  reachable: boolean | null;
  notes: NotesSnapshot | null;
  busy: string | null;
  mark: (kind: string, note?: string) => Promise<NotesResult<DriveEvent>>;
  setIntent: (text: string) => Promise<NotesResult<DriveIntent>>;
  clearIntent: () => Promise<NotesResult<DriveIntent>>;
  refresh: () => Promise<void>;
}

export function useDriveNotes(host: string, enabled = true): DriveNotesState {
  const client = useMemo(() => createDriveNotes(notesBase(host)), [host]);
  const [notes, setNotes] = useState<NotesSnapshot | null>(null);
  const [reachable, setReachable] = useState<boolean | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const alive = useRef(true);

  const refresh = useCallback(async () => {
    const r = await client.getNotes();
    if (!alive.current) return;
    if (r.ok) {
      setNotes(r.value);
      setReachable(true);
    } else {
      setReachable(false);
    }
  }, [client]);

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
    mark: (kind, note = "") => wrap(`event:${kind}`, () => client.addEvent(kind, note)),
    setIntent: (text) => wrap("intent", () => client.setIntent(text)),
    clearIntent: () => wrap("intent", () => client.clearIntent()),
  };
}
