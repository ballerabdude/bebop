// RobotProvider: one live session per robot host.
//
// Owns the single runtime transport (firmware WS :9090), the ref-counted
// telemetry store with liveness, the shared places poller and the optimistic
// E-STOP latch. Screens use the hooks below and never touch the transport
// cache, ports, or telemetry subscriptions themselves.
//
// Mount it with `key={host}` so switching robots tears the old session down
// (socket closed, timers cleared) before the new one connects.

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  useSyncExternalStore,
} from "react";
import type { ReactNode } from "react";

import { getOrCreateRuntimeTransport } from "../runtime/cache";
import type { RuntimeTransport } from "../runtime/wsTransport";
import type { RuntimeSnapshot } from "../runtime/types";
import { endpoints, PORTS } from "./endpoints";
import type { RobotEndpoints } from "./endpoints";
import { EStopStore } from "./estopStore";
import type { EStopState } from "./estopStore";
import { PlacesStore } from "./placesStore";
import type { PlacesState } from "./placesStore";
import { TelemetryStore } from "./telemetryStore";
import type { LinkSnapshot } from "./telemetryStore";

interface RobotSession {
  host: string;
  endpoints: RobotEndpoints;
  transport: RuntimeTransport;
  telemetry: TelemetryStore;
  places: PlacesStore;
  estop: EStopStore<RuntimeSnapshot>;
}

export type { EStopState };

const RobotContext = createContext<RobotSession | null>(null);

// ---------------------------------------------------------------- provider

export function RobotProvider({
  host,
  children,
}: {
  host: string;
  children: ReactNode;
}) {
  const session = useMemo<RobotSession>(() => {
    const transport = getOrCreateRuntimeTransport(host, PORTS.runtime);
    const ep = endpoints(host);
    const telemetry = new TelemetryStore(transport, host, PORTS.runtime, {
      now: () => performance.now(),
      schedule: (fn) => requestAnimationFrame(fn),
      cancelSchedule: (h) => cancelAnimationFrame(h as number),
      setInterval: (fn, ms) => window.setInterval(fn, ms),
      clearInterval: (h) => window.clearInterval(h as number),
    });
    return {
      host,
      endpoints: ep,
      transport,
      telemetry,
      places: new PlacesStore((path) => ep.places(path)),
      estop: new EStopStore<RuntimeSnapshot>(
        transport,
        {
          getSnapshot: () => telemetry.getSnapshot(),
          refresh: async () => telemetry.push(await transport.getSnapshot()),
        },
        {
          now: () => Date.now(),
          setTimeout: (fn, ms) => window.setTimeout(fn, ms),
        },
      ),
    };
  }, [host]);

  useEffect(() => {
    const { telemetry, estop, transport, places } = session;
    telemetry.start();
    const off = telemetry.subscribe(() => estop.observe(telemetry.getSnapshot()));
    return () => {
      off();
      // Leaving the robot while this client holds the drive seat: send one
      // stop before the socket closes (the firmware also halts on the
      // disconnect, this just makes it immediate).
      if (telemetry.getSnapshot()?.drive.youAreActiveOperator) {
        transport.sendVelocityCommand(0, 0);
      }
      telemetry.stop();
      places.dispose();
    };
  }, [session]);

  return (
    <RobotContext.Provider value={session}>{children}</RobotContext.Provider>
  );
}

function useSession(): RobotSession {
  const s = useContext(RobotContext);
  if (!s) throw new Error("robot hooks must be used inside <RobotProvider>");
  return s;
}

/// The current host and its service URLs.
export function useRobot(): { host: string; endpoints: RobotEndpoints } {
  const s = useSession();
  return { host: s.host, endpoints: s.endpoints };
}

/// Transport for commands (arm, mode, E-STOP, model downloads, ...).
export function useRuntime(): RuntimeTransport {
  return useSession().transport;
}

/// Runtime link state: connecting / connected / degraded / disconnected.
export function useConnection(): LinkSnapshot {
  const { telemetry } = useSession();
  return useSyncExternalStore(telemetry.subscribeLink, telemetry.getLink);
}

function shallowEqual(a: unknown, b: unknown): boolean {
  if (Object.is(a, b)) return true;
  if (
    typeof a !== "object" ||
    typeof b !== "object" ||
    a === null ||
    b === null ||
    Array.isArray(a) !== Array.isArray(b)
  ) {
    return false;
  }
  const ka = Object.keys(a as object);
  const kb = Object.keys(b as object);
  if (ka.length !== kb.length) return false;
  for (const k of ka) {
    if (
      !Object.is(
        (a as Record<string, unknown>)[k],
        (b as Record<string, unknown>)[k],
      )
    ) {
      return false;
    }
  }
  return true;
}

/// Subscribe to telemetry at `hz` (the firmware streams at the max rate of
/// all mounted consumers) and select a slice of the latest snapshot. The
/// component re-renders only when the selected value changes (shallow).
/// Pass `select = (s) => s` for the whole snapshot.
export function useTelemetry<T>(
  select: (s: RuntimeSnapshot | null) => T,
  hz = 10,
): T {
  const { telemetry } = useSession();
  useEffect(() => telemetry.acquire(hz), [telemetry, hz]);
  const selectRef = useRef(select);
  selectRef.current = select;
  const cache = useRef<{ src: RuntimeSnapshot | null; value: T } | null>(null);
  const getSelected = useCallback(() => {
    const src = telemetry.getSnapshot();
    const prev = cache.current;
    if (prev && prev.src === src) return prev.value;
    const value = selectRef.current(src);
    if (prev && shallowEqual(prev.value, value)) {
      cache.current = { src, value: prev.value };
      return prev.value;
    }
    cache.current = { src, value };
    return value;
  }, [telemetry]);
  return useSyncExternalStore(telemetry.subscribe, getSelected);
}

/// Apply a fresh snapshot immediately (after a command round-trip).
export function useRefreshSnapshot(): () => Promise<void> {
  const { transport, telemetry } = useSession();
  return useCallback(async () => {
    telemetry.push(await transport.getSnapshot());
  }, [transport, telemetry]);
}

/// Shared /places + /whereami poller (polls while any consumer is mounted).
/// `enabled = false` reads the last values without keeping the poll alive.
/// `pending` also polls the pending-place queue (Memory only: it is large).
export function usePlaces(
  enabled = true,
  { pending = false }: { pending?: boolean } = {},
): PlacesState & {
  refresh: () => Promise<void>;
} {
  const { places } = useSession();
  useEffect(
    () => (enabled ? places.acquire({ pending }) : undefined),
    [places, enabled, pending],
  );
  const state = useSyncExternalStore(places.subscribe, places.getSnapshot);
  return { ...state, refresh: places.refresh };
}

/// Places service (:9095) reachability from a slow /healthz probe, for the
/// shell's status dot (null until the first probe answers).
export function usePlacesReachable(): boolean | null {
  const { places } = useSession();
  useEffect(() => places.acquireHealth(), [places]);
  return useSyncExternalStore(places.subscribe, () => places.getSnapshot().reachable);
}

/// Global E-STOP (optimistic latch, confirmed from telemetry).
export function useEStop(): EStopState & {
  trigger: (reason?: string) => Promise<void>;
  reset: () => Promise<void>;
} {
  const { estop } = useSession();
  const state = useSyncExternalStore(estop.subscribe, estop.getSnapshot);
  return { ...state, trigger: estop.trigger, reset: estop.reset };
}

/// Run a runtime command with a busy label and an error message, then pull
/// a fresh snapshot so the UI reflects the result without waiting a frame.
export function useRuntimeAction(): {
  busy: string | null;
  error: string | null;
  setError: (e: string | null) => void;
  run: (label: string, fn: (t: RuntimeTransport) => Promise<unknown>) => Promise<void>;
} {
  const { transport, telemetry } = useSession();
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const run = useCallback(
    async (label: string, fn: (t: RuntimeTransport) => Promise<unknown>) => {
      setBusy(label);
      setError(null);
      try {
        await fn(transport);
        telemetry.push(await transport.getSnapshot());
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setBusy(null);
      }
    },
    [transport, telemetry],
  );
  return { busy, error, setError, run };
}
