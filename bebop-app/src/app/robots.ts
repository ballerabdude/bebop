// Remembered robots (per browser) and the /healthz reachability probe.
//
// One list for every surface: the picker, the setup wizard (which adds the
// robot it just provisioned) and the shell (which records the last visit).
// It replaces the old ConnectByIp storage key that the setup and operator
// forms shared, where one overwrote the other's port.

import { useCallback, useEffect, useState } from "react";

import { endpoints, normalizeHost } from "../robot/endpoints";
import type { ServiceName } from "../robot/endpoints";

const STORAGE_KEY = "bebop.robots.v1";
const LEGACY_KEY = "bebop.connectByIp";
const PROBE_TIMEOUT_MS = 2_500;

export interface RememberedRobot {
  host: string;
  /// Friendly name (robot_name from setup), optional.
  name?: string;
  lastUsedAt: number;
}

function read(): RememberedRobot[] {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw) as RememberedRobot[];
      if (Array.isArray(parsed)) {
        return parsed.filter((r) => r && typeof r.host === "string" && r.host);
      }
    }
    // One-time migration from the old single-endpoint key.
    const legacy = window.localStorage.getItem(LEGACY_KEY);
    if (legacy) {
      const { ip } = JSON.parse(legacy) as { ip?: string };
      if (typeof ip === "string" && ip) {
        return [{ host: normalizeHost(ip), lastUsedAt: Date.now() }];
      }
    }
  } catch {
    /* storage disabled or corrupt */
  }
  return [];
}

function write(list: RememberedRobot[]): void {
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(list));
  } catch {
    /* storage disabled: the list just isn't remembered */
  }
  window.dispatchEvent(new Event("bebop-robots"));
}

export function rememberRobot(host: string, name?: string): void {
  const h = normalizeHost(host);
  if (!h) return;
  const list = read();
  const prev = list.find((r) => r.host === h);
  const next: RememberedRobot = {
    host: h,
    name: name ?? prev?.name,
    lastUsedAt: Date.now(),
  };
  write([next, ...list.filter((r) => r.host !== h)].slice(0, 20));
}

/// Set or clear (empty string) a robot's friendly name.
export function renameRobot(host: string, name: string): void {
  const list = read();
  write(
    list.map((r) => (r.host === host ? { ...r, name: name.trim() || undefined } : r)),
  );
}

export function forgetRobot(host: string): void {
  write(read().filter((r) => r.host !== host));
}

export function useRememberedRobots(): RememberedRobot[] {
  const [list, setList] = useState<RememberedRobot[]>(() => read());
  useEffect(() => {
    const sync = () => setList(read());
    window.addEventListener("bebop-robots", sync);
    window.addEventListener("storage", sync);
    return () => {
      window.removeEventListener("bebop-robots", sync);
      window.removeEventListener("storage", sync);
    };
  }, []);
  return list;
}

export type ProbeResult = "up" | "down" | "checking";

export async function probe(host: string, service: ServiceName): Promise<boolean> {
  try {
    const res = await fetch(endpoints(host).healthz(service), {
      cache: "no-store",
      signal: AbortSignal.timeout(PROBE_TIMEOUT_MS),
    });
    return res.ok;
  } catch {
    return false;
  }
}

/// Probe the runtime (:9090) and setup (:9091) health endpoints in
/// parallel. Re-probes when `host` or `nonce` changes, and every
/// `everyMs` (0 = never) while the page is visible.
export function useRobotProbe(
  host: string,
  nonce = 0,
  everyMs = 0,
): { runtime: ProbeResult; setup: ProbeResult; recheck: () => void } {
  const [runtime, setRuntime] = useState<ProbeResult>("checking");
  const [setup, setSetup] = useState<ProbeResult>("checking");
  const [local, setLocal] = useState(0);
  useEffect(() => {
    if (!host) return;
    let cancelled = false;
    setRuntime("checking");
    setSetup("checking");
    const run = () => {
      void probe(host, "runtime").then((ok) => {
        if (!cancelled) setRuntime(ok ? "up" : "down");
      });
      void probe(host, "setup").then((ok) => {
        if (!cancelled) setSetup(ok ? "up" : "down");
      });
    };
    run();
    // Background re-probes keep the last answer on screen (no "checking"
    // flicker) and skip hidden tabs.
    const id = everyMs > 0
      ? window.setInterval(() => {
          if (!document.hidden) run();
        }, everyMs)
      : 0;
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [host, nonce, local, everyMs]);
  const recheck = useCallback(() => setLocal((n) => n + 1), []);
  return { runtime, setup, recheck };
}
