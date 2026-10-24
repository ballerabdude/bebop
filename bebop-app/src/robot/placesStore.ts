// Shared poller for the places service (:9095): the confirmed place list,
// the pending (scout-captured, not yet imported) queue and the current
// recognition (`/whereami`). The Memory screens and the shell's
// status dot all read it, so the robot sees one poll loop per app instead of
// one per screen. Polling runs only while at least one consumer is mounted,
// and the pending queue (large: keyframes + facts per entry) is fetched only
// while a consumer asked for it (`acquire({ pending: true })`).

export const PLACES_POLL_MS = 15_000;
export const WHEREAMI_POLL_MS = 3_000;
export const HEALTH_POLL_MS = 10_000;
const REQUEST_TIMEOUT_MS = 4_000;

export interface PlaceSummary {
  id: string;
  name: string;
  [key: string]: unknown;
}

export interface WhereAmI {
  matched?: boolean;
  place?: PlaceSummary | null;
  score?: number;
  margin?: number;
  min_score?: number;
  reason?: string;
  top?: { id: string; name: string; score: number }[];
  [key: string]: unknown;
}

export interface PlacesState {
  places: PlaceSummary[];
  /// Pending places waiting for review/import (`GET /places/pending`).
  pending: PlaceSummary[];
  whereami: WhereAmI | null;
  /// Places service reachable on the last /places request (null = unknown).
  reachable: boolean | null;
  error: string | null;
  loading: boolean;
  updatedAt: number | null;
}

const INITIAL: PlacesState = {
  places: [],
  pending: [],
  whereami: null,
  reachable: null,
  error: null,
  loading: false,
  updatedAt: null,
};

async function readError(res: Response): Promise<string> {
  try {
    const body = await res.json();
    if (body && typeof body.error === "string") return body.error;
  } catch {
    /* not JSON */
  }
  return `${res.status} ${res.statusText}`;
}

export class PlacesStore {
  private state: PlacesState = INITIAL;
  private listeners = new Set<() => void>();
  private consumers = 0;
  private pendingConsumers = 0;
  private placesTimer: ReturnType<typeof setInterval> | null = null;
  private whereTimer: ReturnType<typeof setInterval> | null = null;
  private healthConsumers = 0;
  private healthTimer: ReturnType<typeof setInterval> | null = null;

  private readonly url: (path: string) => string;

  constructor(url: (path: string) => string) {
    this.url = url;
  }

  subscribe = (cb: () => void): (() => void) => {
    this.listeners.add(cb);
    return () => this.listeners.delete(cb);
  };

  getSnapshot = (): PlacesState => this.state;

  acquire(opts: { pending?: boolean } = {}): () => void {
    const wantsPending = !!opts.pending;
    this.consumers += 1;
    if (wantsPending) this.pendingConsumers += 1;
    if (this.consumers === 1 || this.placesTimer === null) {
      this.stopPolling();
      this.startPolling();
    } else if (wantsPending && this.pendingConsumers === 1) {
      // Polling already runs without the queue: fetch it now, not in 15 s.
      void this.loadPlaces();
    }
    let released = false;
    return () => {
      if (released) return;
      released = true;
      this.consumers = Math.max(0, this.consumers - 1);
      if (wantsPending) this.pendingConsumers = Math.max(0, this.pendingConsumers - 1);
      if (this.consumers === 0) this.stopPolling();
    };
  }

  /// Keep a slow /healthz probe alive (the shell's status dot). It only
  /// updates `reachable`, so it is cheap enough to run on every route.
  acquireHealth(): () => void {
    this.healthConsumers += 1;
    if (this.healthTimer === null) {
      void this.probeHealth();
      this.healthTimer = setInterval(() => void this.probeHealth(), HEALTH_POLL_MS);
    }
    let released = false;
    return () => {
      if (released) return;
      released = true;
      this.healthConsumers = Math.max(0, this.healthConsumers - 1);
      if (this.healthConsumers === 0) this.stopHealth();
    };
  }

  private stopHealth(): void {
    if (this.healthTimer !== null) clearInterval(this.healthTimer);
    this.healthTimer = null;
  }

  private async probeHealth(): Promise<void> {
    let ok = false;
    try {
      const res = await fetch(this.url("/healthz"), {
        cache: "no-store",
        signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
      });
      ok = res.ok;
    } catch {
      ok = false;
    }
    if (ok !== this.state.reachable) this.set({ reachable: ok });
  }

  /// Re-fetch both resources now (after a create/rename/delete).
  refresh = async (): Promise<void> => {
    await Promise.all([this.loadPlaces(), this.loadWhere()]);
  };

  /// Stop polling (session teardown). Consumers re-acquiring restart it.
  dispose(): void {
    this.stopPolling();
    this.stopHealth();
  }

  private set(patch: Partial<PlacesState>): void {
    this.state = { ...this.state, ...patch };
    for (const cb of this.listeners) cb();
  }

  private startPolling(): void {
    void this.refresh();
    this.placesTimer = setInterval(() => void this.loadPlaces(), PLACES_POLL_MS);
    this.whereTimer = setInterval(() => void this.loadWhere(), WHEREAMI_POLL_MS);
  }

  private stopPolling(): void {
    if (this.placesTimer !== null) clearInterval(this.placesTimer);
    if (this.whereTimer !== null) clearInterval(this.whereTimer);
    this.placesTimer = null;
    this.whereTimer = null;
  }

  private async loadPlaces(): Promise<void> {
    this.set({ loading: true });
    try {
      const opts = { cache: "no-store" as const, signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS) };
      const withPending = this.pendingConsumers > 0;
      const [res, pendingRes] = await Promise.all([
        fetch(this.url("/places"), opts),
        withPending ? fetch(this.url("/places/pending"), opts).catch(() => null) : null,
      ]);
      if (!res.ok) throw new Error(await readError(res));
      const body = (await res.json()) as { places?: PlaceSummary[] };
      // A failed/garbled pending fetch keeps the last queue: /places itself
      // answered, so the service is reachable.
      let pending: PlaceSummary[] = this.state.pending;
      if (pendingRes?.ok) {
        try {
          const pb = (await pendingRes.json()) as { places?: PlaceSummary[] };
          pending = Array.isArray(pb?.places) ? pb.places : [];
        } catch {
          /* non-JSON 200: keep the previous queue */
        }
      }
      this.set({
        places: Array.isArray(body.places) ? body.places : [],
        pending,
        reachable: true,
        error: null,
        loading: false,
        updatedAt: Date.now(),
      });
    } catch (e) {
      this.set({
        reachable: false,
        error: e instanceof Error ? e.message : String(e),
        loading: false,
      });
    }
  }

  private async loadWhere(): Promise<void> {
    if (typeof document !== "undefined" && document.hidden) return;
    try {
      const res = await fetch(this.url("/whereami"), {
        cache: "no-store",
        signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
      });
      if (!res.ok) return;
      this.set({ whereami: (await res.json()) as WhereAmI });
    } catch {
      /* transient; /places surfaces hard errors */
    }
  }
}
