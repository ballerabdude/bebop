// Ref-counted telemetry + liveness store for one robot session.
//
// Every screen (and the shell's status bar) reads runtime telemetry through
// this store instead of wiring its own listeners on the transport:
//
//   * Rate: each consumer asks for a rate; the firmware subscription runs at
//     the max of the active consumers and is torn down when the last one
//     leaves (no more "subscribe at 5 Hz and never unsubscribe" leaks).
//   * Coalescing: frames are stashed and flushed to React once per animation
//     frame (`schedule`), so a 30 Hz pump or a burst after a stall never
//     cascades into a render backlog.
//   * Liveness: while a subscription is active, no frame for `degradedMs`
//     marks the link `degraded`; no frame for `resetMs` force-reconnects the
//     socket (a half-dead link after a Wi-Fi roam never fires `onclose`).
//
// Pure TS with injected timers so it runs under `node --test`.

import type { RuntimeConnectionState } from "../runtime/wsTransport.ts";
import type { RuntimeSnapshot } from "../runtime/types.ts";

export const DEGRADED_AFTER_MS = 1_000;
export const RESET_AFTER_MS = 2_500;
export const LIVENESS_TICK_MS = 250;

export type LinkState = "connecting" | "connected" | "degraded" | "disconnected";

/// The slice of `RuntimeTransport` the store needs.
export interface TelemetrySource {
  connect(host: string, port: number): Promise<void>;
  disconnect(): void;
  forceReconnect(): void;
  getSnapshot(): Promise<RuntimeSnapshot>;
  subscribeTelemetry(rateHz: number): Promise<void>;
  unsubscribeTelemetry(): Promise<void>;
  onTelemetry(cb: (s: RuntimeSnapshot) => void): () => void;
  onEStopLatched(cb: (reason: string) => void): () => void;
  onConnectionStateChange(cb: (s: RuntimeConnectionState) => void): () => void;
  isConnected(): boolean;
}

export interface StoreDeps {
  now: () => number;
  /// Defer a flush (requestAnimationFrame in the browser).
  schedule: (fn: () => void) => unknown;
  cancelSchedule: (handle: unknown) => void;
  setInterval: (fn: () => void, ms: number) => unknown;
  clearInterval: (handle: unknown) => void;
}

export interface LinkSnapshot {
  state: LinkState;
  /// ms since the last telemetry frame; null before the first one.
  lastFrameAt: number | null;
}

export class TelemetryStore {
  private snapshot: RuntimeSnapshot | null = null;
  private pending: RuntimeSnapshot | null = null;
  private flushHandle: unknown = null;
  private listeners = new Set<() => void>();
  private linkListeners = new Set<() => void>();
  private consumers = new Map<number, number>();
  private nextConsumer = 1;
  private subscribedHz: number | null = null;
  private socketState: RuntimeConnectionState = "disconnected";
  private link: LinkSnapshot = { state: "disconnected", lastFrameAt: null };
  private lastRxAt: number | null = null;
  private subscribedAt: number | null = null;
  private offs: Array<() => void> = [];
  private livenessHandle: unknown = null;
  private running = false;

  private readonly source: TelemetrySource;
  private readonly host: string;
  private readonly port: number;
  private readonly deps: StoreDeps;

  constructor(source: TelemetrySource, host: string, port: number, deps: StoreDeps) {
    this.source = source;
    this.host = host;
    this.port = port;
    this.deps = deps;
  }

  // ------------------------------------------------------------- lifecycle
  start(): void {
    if (this.running) return;
    this.running = true;
    const src = this.source;
    this.offs.push(
      src.onTelemetry((s) => this.onFrame(s)),
      src.onEStopLatched(() => {
        void src.getSnapshot().then(
          (s) => this.onFrame(s),
          () => {},
        );
      }),
      src.onConnectionStateChange((state) => this.onSocketState(state)),
    );
    this.livenessHandle = this.deps.setInterval(
      () => this.checkLiveness(),
      LIVENESS_TICK_MS,
    );
    void src.connect(this.host, this.port).catch(() => {
      /* the transport retries with backoff; link state shows it */
    });
  }

  stop(): void {
    if (!this.running) return;
    this.running = false;
    for (const off of this.offs) off();
    this.offs = [];
    if (this.livenessHandle !== null) this.deps.clearInterval(this.livenessHandle);
    this.livenessHandle = null;
    if (this.flushHandle !== null) this.deps.cancelSchedule(this.flushHandle);
    this.flushHandle = null;
    this.pending = null;
    this.subscribedHz = null;
    this.source.disconnect();
    this.setLink("disconnected");
  }

  // ------------------------------------------------------------- consumers
  /// Register a consumer wanting telemetry at `hz`. Returns its release.
  acquire(hz: number): () => void {
    const id = this.nextConsumer++;
    this.consumers.set(id, Math.max(1, Math.round(hz)));
    this.syncRate();
    let released = false;
    return () => {
      if (released) return;
      released = true;
      this.consumers.delete(id);
      this.syncRate();
    };
  }

  /// Highest rate any live consumer asked for; null with no consumers.
  wantedHz(): number | null {
    let max = 0;
    for (const hz of this.consumers.values()) max = Math.max(max, hz);
    return max > 0 ? max : null;
  }

  activeRateHz(): number | null {
    return this.subscribedHz;
  }

  // ---------------------------------------------------- external store API
  subscribe = (cb: () => void): (() => void) => {
    this.listeners.add(cb);
    return () => this.listeners.delete(cb);
  };

  getSnapshot = (): RuntimeSnapshot | null => this.snapshot;

  subscribeLink = (cb: () => void): (() => void) => {
    this.linkListeners.add(cb);
    return () => this.linkListeners.delete(cb);
  };

  getLink = (): LinkSnapshot => this.link;

  /// Apply a snapshot right away (e.g. after a command), skipping the
  /// coalescing delay.
  push(s: RuntimeSnapshot): void {
    this.lastRxAt = this.deps.now();
    this.pending = null;
    this.commit(s);
  }

  // ------------------------------------------------------------- internals
  private onFrame(s: RuntimeSnapshot): void {
    this.lastRxAt = this.deps.now();
    if (this.link.state === "degraded") this.setLink("connected");
    this.pending = s;
    if (this.flushHandle === null) {
      this.flushHandle = this.deps.schedule(() => {
        this.flushHandle = null;
        const next = this.pending;
        this.pending = null;
        if (next) this.commit(next);
      });
    }
  }

  private commit(s: RuntimeSnapshot): void {
    this.snapshot = s;
    for (const cb of this.listeners) cb();
  }

  private onSocketState(state: RuntimeConnectionState): void {
    this.socketState = state;
    if (state === "connected") {
      this.setLink("connected");
      // A fresh socket has no subscription; re-issue at the wanted rate and
      // anchor the liveness clock so the first frame has time to arrive.
      this.subscribedHz = null;
      this.subscribedAt = this.deps.now();
      this.syncRate();
      void this.source.getSnapshot().then(
        (s) => this.push(s),
        () => {},
      );
    } else {
      this.subscribedHz = null;
      this.setLink(state === "connecting" ? "connecting" : "disconnected");
    }
  }

  private syncRate(): void {
    if (!this.running || this.socketState !== "connected") return;
    const want = this.wantedHz();
    if (want === this.subscribedHz) return;
    this.subscribedHz = want;
    if (want === null) {
      void this.source.unsubscribeTelemetry().catch(() => {});
    } else {
      this.subscribedAt = this.deps.now();
      void this.source.subscribeTelemetry(want).catch(() => {});
    }
  }

  private checkLiveness(): void {
    if (this.socketState !== "connected" || this.subscribedHz === null) return;
    const now = this.deps.now();
    const since = Math.max(this.lastRxAt ?? 0, this.subscribedAt ?? 0);
    const silent = now - since;
    if (silent > RESET_AFTER_MS) {
      this.subscribedAt = now;
      this.lastRxAt = null;
      this.setLink("connecting");
      this.source.forceReconnect();
    } else if (silent > DEGRADED_AFTER_MS) {
      this.setLink("degraded");
    }
  }

  private setLink(state: LinkState): void {
    const lastFrameAt = this.lastRxAt;
    if (this.link.state === state && this.link.lastFrameAt === lastFrameAt) return;
    this.link = { state, lastFrameAt };
    for (const cb of this.linkListeners) cb();
  }
}
