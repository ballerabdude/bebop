// Poller for Memory › Place graph (`/places/graph`). Observe-only.

import { useEffect, useState } from "react";

import { isPlaceGraph, normalizePlaceGraph } from "./mapView";
import type { PlaceGraph } from "./mapView";
import type { PlacesUrl } from "./places";

export const GRAPH_POLL_MS = 15000;
const REQUEST_TIMEOUT_MS = 4000;

function timeoutMessage(e: unknown, what: string): string {
  return (e as { name?: string }).name === "AbortError"
    ? `${what} request timed out (4 s).`
    : e instanceof Error
      ? e.message
      : String(e);
}

export interface PlaceGraphState {
  graph: PlaceGraph | null;
  error: string | null;
  receivedAt: number;
}

export function usePlaceGraph(api: PlacesUrl): PlaceGraphState {
  const [state, setState] = useState<PlaceGraphState>({
    graph: null,
    error: null,
    receivedAt: 0,
  });

  useEffect(() => {
    let cancelled = false;
    let timer = 0;
    let ctrl: AbortController | null = null;

    const poll = async () => {
      ctrl = new AbortController();
      const timeout = window.setTimeout(() => ctrl?.abort(), REQUEST_TIMEOUT_MS);
      try {
        const res = await fetch(api("/places/graph"), { cache: "no-store", signal: ctrl.signal });
        if (!res.ok) throw new Error(`places service returned ${res.status} ${res.statusText}`);
        const body: unknown = await res.json();
        if (!isPlaceGraph(body)) throw new Error("places service returned an unexpected graph payload");
        if (!cancelled) {
          setState({ graph: normalizePlaceGraph(body), error: null, receivedAt: performance.now() });
        }
      } catch (e) {
        if (!cancelled) setState((s) => ({ ...s, error: timeoutMessage(e, "Place-graph") }));
      } finally {
        window.clearTimeout(timeout);
        if (!cancelled) timer = window.setTimeout(poll, GRAPH_POLL_MS);
      }
    };

    void poll();
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
      ctrl?.abort();
    };
  }, [api]);

  return state;
}
