// Pollers for Memory › Map: the selected place's stored local map
// (`/places/{id}/map`, back-off by payload size) and the place graph
// (`/places/graph`). Observe-only; a 404 is an explicit blank state.

import { startTransition, useEffect, useState } from "react";

import {
  isPlaceGraph,
  isPlaceMapPayload,
  normalizePlaceGraph,
  normalizePlaceMap,
  placeMapAgeSeconds,
  placeMapMissingKind,
} from "./mapView";
import type { PlaceGraph, PlaceMap, PlaceMapMissingKind } from "./mapView";
import type { PlacesUrl } from "./places";

export const MIN_POLL_MS = 2000;
const LARGE_POLL_MS = 5000;
const HUGE_POLL_MS = 10000;
export const GRAPH_POLL_MS = 15000;
export const REQUEST_TIMEOUT_MS = 4000;
const LARGE_BYTES = 1_500_000;
const HUGE_BYTES = 2_800_000;

export function placePath(placeId: string, suffix: string): string {
  return `/places/${encodeURIComponent(placeId)}${suffix}`;
}

function timeoutMessage(e: unknown, what: string): string {
  return (e as { name?: string }).name === "AbortError"
    ? `${what} request timed out (4 s).`
    : e instanceof Error
      ? e.message
      : String(e);
}

export interface PlaceMapState {
  map: PlaceMap | null;
  missing: PlaceMapMissingKind | null;
  error: string | null;
  loading: boolean;
  /// performance.now() of the last answer (0 = none yet).
  receivedAt: number;
  /// Age of the stored record when it was fetched (s).
  storedAge: number | null;
  bytes: number;
  pollMs: number;
}

const BLANK: PlaceMapState = {
  map: null,
  missing: null,
  error: null,
  loading: false,
  receivedAt: 0,
  storedAge: null,
  bytes: 0,
  pollMs: MIN_POLL_MS,
};

/// Poll one place's stored local map. A place switch starts from a blank
/// slate: the previous place's map is never shown while the new request is
/// in flight. `retryKey` changes force an immediate re-poll.
export function usePlaceMap(
  api: PlacesUrl,
  placeId: string | null,
  paused: boolean,
  retryKey: number,
): [PlaceMapState, (patch: Partial<PlaceMapState>) => void] {
  const [state, setState] = useState<PlaceMapState>(BLANK);
  const patch = (p: Partial<PlaceMapState>) => setState((s) => ({ ...s, ...p }));

  useEffect(() => {
    setState({ ...BLANK, loading: !!placeId });
  }, [placeId]);

  useEffect(() => {
    if (paused || !placeId) return;
    let cancelled = false;
    let timer = 0;
    let pollMs = MIN_POLL_MS;
    let ctrl: AbortController | null = null;

    const poll = async () => {
      ctrl = new AbortController();
      const timeout = window.setTimeout(() => ctrl?.abort(), REQUEST_TIMEOUT_MS);
      try {
        const res = await fetch(api(placePath(placeId, "/map")), {
          cache: "no-store",
          signal: ctrl.signal,
        });
        if (res.status === 404) {
          // "No map recorded for this place" vs "this service build predates
          // the endpoint": ask the sibling graph endpoint.
          let graphStatus: number | null = null;
          try {
            graphStatus = (await fetch(api("/places/graph"), { cache: "no-store" })).status;
          } catch {
            graphStatus = null;
          }
          if (!cancelled) {
            setState({
              ...BLANK,
              missing: placeMapMissingKind(404, graphStatus),
              receivedAt: performance.now(),
            });
          }
          return;
        }
        if (!res.ok) throw new Error(`places service returned ${res.status} ${res.statusText}`);
        const text = await res.text();
        const len = Number(res.headers.get("content-length"));
        const bytes = Number.isFinite(len) && len > 0 ? len : text.length;
        pollMs = bytes >= HUGE_BYTES ? HUGE_POLL_MS : bytes >= LARGE_BYTES ? LARGE_POLL_MS : MIN_POLL_MS;
        const body: unknown = JSON.parse(text);
        if (!isPlaceMapPayload(body)) {
          throw new Error("places service returned an unexpected place-map payload");
        }
        const map = normalizePlaceMap(body);
        if (!cancelled) {
          const storedAge = placeMapAgeSeconds(map.updated_at, Date.now());
          startTransition(() =>
            setState({
              map,
              missing: null,
              error: null,
              loading: false,
              receivedAt: performance.now(),
              storedAge,
              bytes,
              pollMs,
            }),
          );
        }
      } catch (e) {
        if (!cancelled) patch({ loading: false, error: timeoutMessage(e, "Place-map") });
      } finally {
        window.clearTimeout(timeout);
        if (!cancelled) timer = window.setTimeout(poll, pollMs);
      }
    };

    void poll();
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
      ctrl?.abort();
    };
  }, [api, placeId, paused, retryKey]);

  return [state, patch];
}

export interface PlaceGraphState {
  graph: PlaceGraph | null;
  missing: boolean;
  error: string | null;
  receivedAt: number;
}

export function usePlaceGraph(api: PlacesUrl, paused: boolean, retryKey: number): PlaceGraphState {
  const [state, setState] = useState<PlaceGraphState>({
    graph: null,
    missing: false,
    error: null,
    receivedAt: 0,
  });

  useEffect(() => {
    if (paused) return;
    let cancelled = false;
    let timer = 0;
    let ctrl: AbortController | null = null;

    const poll = async () => {
      ctrl = new AbortController();
      const timeout = window.setTimeout(() => ctrl?.abort(), REQUEST_TIMEOUT_MS);
      try {
        const res = await fetch(api("/places/graph"), { cache: "no-store", signal: ctrl.signal });
        if (res.status === 404) {
          if (!cancelled) setState({ graph: null, missing: true, error: null, receivedAt: performance.now() });
          return;
        }
        if (!res.ok) throw new Error(`places service returned ${res.status} ${res.statusText}`);
        const body: unknown = await res.json();
        if (!isPlaceGraph(body)) throw new Error("places service returned an unexpected graph payload");
        if (!cancelled) {
          setState({ graph: normalizePlaceGraph(body), missing: false, error: null, receivedAt: performance.now() });
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
  }, [api, paused, retryKey]);

  return state;
}
