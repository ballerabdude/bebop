// Optimistic E-STOP latch for the shell's global E-STOP button.
//
// Latches the instant it is pressed, then follows telemetry: the firmware's
// `estopLatched` confirms it. If the request fails, the optimistic latch is
// dropped at once and `error` stays set (the button shows "NOT SENT") until a
// later press reaches the robot, so a stop that never left the browser can't
// look like one that did. If the request succeeded but telemetry never
// confirms, the optimistic latch clears after ESTOP_OPTIMISTIC_MS.
//
// Pure TS with injected clock/timers so the node test runner can drive it.

export const ESTOP_OPTIMISTIC_MS = 3_000;

export interface EStopState {
  /// Optimistic or confirmed latch.
  latched: boolean;
  /// Telemetry confirms the latch.
  confirmed: boolean;
  reason: string;
  /// A request is in flight.
  busy: "estop" | "reset" | null;
  /// Last request failure; for "estop" the stop did NOT reach the robot.
  error: string | null;
  /// Which request the error belongs to.
  errorKind: "estop" | "reset" | null;
}

export interface EStopTransport {
  emergencyStop(reason: string): Promise<unknown>;
  resetEStop(): Promise<unknown>;
}

export interface EStopTelemetry<S> {
  getSnapshot(): S | null;
  /// Fetch and apply a fresh snapshot (after a reset).
  refresh(): Promise<void>;
}

export interface EStopDeps {
  now: () => number;
  setTimeout: (fn: () => void, ms: number) => unknown;
}

type Frame = { estopLatched?: boolean; estopReason?: string } | null;

function message(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

export class EStopStore<S extends Frame = Frame> {
  private state: EStopState = {
    latched: false,
    confirmed: false,
    reason: "",
    busy: null,
    error: null,
    errorKind: null,
  };
  private optimisticUntil = 0;
  private listeners = new Set<() => void>();

  private readonly transport: EStopTransport;
  private readonly telemetry: EStopTelemetry<S>;
  private readonly deps: EStopDeps;

  constructor(
    transport: EStopTransport,
    telemetry: EStopTelemetry<S>,
    deps: EStopDeps,
  ) {
    this.transport = transport;
    this.telemetry = telemetry;
    this.deps = deps;
  }

  subscribe = (cb: () => void): (() => void) => {
    this.listeners.add(cb);
    return () => {
      this.listeners.delete(cb);
    };
  };
  getSnapshot = (): EStopState => this.state;

  private set(patch: Partial<EStopState>): void {
    this.state = { ...this.state, ...patch };
    for (const cb of this.listeners) cb();
  }

  /// Fold a telemetry frame in.
  observe(s: S | null): void {
    const confirmed = !!s?.estopLatched;
    const optimistic = this.deps.now() < this.optimisticUntil;
    const latched = confirmed || (optimistic && this.state.latched);
    const reason = s?.estopReason ?? "";
    if (
      confirmed !== this.state.confirmed ||
      latched !== this.state.latched ||
      reason !== this.state.reason
    ) {
      this.set({ confirmed, latched, reason });
    }
  }

  trigger = async (reason = "operator"): Promise<void> => {
    this.optimisticUntil = this.deps.now() + ESTOP_OPTIMISTIC_MS;
    this.set({ latched: true, busy: "estop" });
    try {
      await this.transport.emergencyStop(reason);
      this.set({ busy: null, error: null, errorKind: null });
      this.deps.setTimeout(
        () => this.observe(this.telemetry.getSnapshot()),
        ESTOP_OPTIMISTIC_MS + 50,
      );
    } catch (e) {
      // Not sent: drop the optimistic latch now so the button reads as
      // failed rather than "STOPPING".
      this.optimisticUntil = 0;
      this.set({ busy: null, error: message(e), errorKind: "estop" });
      this.observe(this.telemetry.getSnapshot());
    }
  };

  reset = async (): Promise<void> => {
    this.optimisticUntil = 0;
    this.set({ busy: "reset", error: null, errorKind: null });
    try {
      await this.transport.resetEStop();
      await this.telemetry.refresh();
      this.set({ busy: null });
    } catch (e) {
      this.set({ busy: null, error: message(e), errorKind: "reset" });
    }
    this.observe(this.telemetry.getSnapshot());
  };
}
