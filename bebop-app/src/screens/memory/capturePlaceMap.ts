// "Capture this place": store the live SLAM structure snapshot (:9096,
// only with --live-slam) as the selected place's local map. Every step that
// could store a wrong or unusable map refuses instead, and says why.

import { isStructureSnapshot, normalizeStructure, placeCaptureGate } from "./mapView";
import { placePath, REQUEST_TIMEOUT_MS } from "./mapPolling";
import { readError } from "./places";
import type { PlacesUrl, WhereAmI } from "./places";

const POST_TIMEOUT_MS = 15000;

export interface CaptureResult {
  ok: boolean;
  note: string;
  /// The places service has no map-ingest endpoint.
  endpointMissing?: boolean;
}

async function fetchWithTimeout(url: string, ms: number, init?: RequestInit): Promise<Response> {
  const ctrl = new AbortController();
  const t = window.setTimeout(() => ctrl.abort(), ms);
  try {
    return await fetch(url, { cache: "no-store", ...init, signal: ctrl.signal });
  } finally {
    window.clearTimeout(t);
  }
}

function why(e: unknown, what: string, s: number): string {
  return (e as { name?: string }).name === "AbortError"
    ? `${what} timed out (${s} s)`
    : e instanceof Error
      ? e.message
      : String(e);
}

export async function capturePlaceMap(opts: {
  api: PlacesUrl;
  slam: (path: string) => string;
  placeId: string;
  placeName: string;
}): Promise<CaptureResult> {
  const { api, slam, placeId, placeName } = opts;

  // 1. Re-confirm recognition at capture time: a map from another room would
  // persist a wrong anchor and corrupt the place graph.
  let where: WhereAmI;
  try {
    const res = await fetchWithTimeout(api("/whereami"), REQUEST_TIMEOUT_MS);
    if (!res.ok) throw new Error(`whereami returned ${res.status} ${res.statusText}`);
    where = (await res.json()) as WhereAmI;
  } catch (e) {
    return {
      ok: false,
      note: `Could not confirm the robot's current place (${why(e, "whereami", 4)}). Nothing was captured.`,
    };
  }
  const gate = placeCaptureGate(placeId, placeName, where);
  if (!gate.allowed) return { ok: false, note: `${gate.reason} Nothing was captured.` };

  // 2. Read the live structure snapshot afresh, never a cached payload.
  let text = "";
  try {
    const res = await fetchWithTimeout(slam("/structure"), REQUEST_TIMEOUT_MS);
    if (res.status === 404 || res.status === 503) {
      return {
        ok: false,
        note: "The live structure feed is unavailable (live SLAM off or the vision service is restarting). Nothing was captured.",
      };
    }
    if (!res.ok) throw new Error(`structure feed returned ${res.status} ${res.statusText}`);
    text = await res.text();
  } catch (e) {
    return {
      ok: false,
      note: `Can't read the live structure feed (${why(e, "structure request", 4)}). Live SLAM may be off. Nothing was captured.`,
    };
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    return { ok: false, note: "The live structure feed returned invalid JSON. Nothing was captured." };
  }
  if (!isStructureSnapshot(parsed)) {
    return { ok: false, note: "The live structure feed returned an unexpected payload. Nothing was captured." };
  }

  // 3. A submap without a gravity estimate can't be anchored.
  if (normalizeStructure(parsed).gravity.valid !== true) {
    return {
      ok: false,
      note: "Gravity isn't estimated yet here, so the submap can't be anchored. Wait for a floor/gravity estimate and capture again. Nothing was captured.",
    };
  }

  // 4. POST the snapshot verbatim.
  try {
    const res = await fetchWithTimeout(api(placePath(placeId, "/map")), POST_TIMEOUT_MS, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: text,
    });
    if (res.status === 404) {
      return {
        ok: false,
        endpointMissing: true,
        note: "This places-service build has no map-ingest endpoint; nothing was captured.",
      };
    }
    if (!res.ok) throw new Error(await readError(res));
    let revision: number | null = null;
    try {
      const body: unknown = await res.json();
      const raw = body && typeof body === "object" && "map" in body ? (body as { map?: unknown }).map : body;
      if (raw && typeof raw === "object" && "revision" in raw) {
        const r = Number((raw as { revision?: unknown }).revision);
        if (Number.isFinite(r)) revision = r;
      }
    } catch {
      /* stored; the revision is only a nicety */
    }
    return {
      ok: true,
      note: `Captured ${placeName}${revision != null ? ` · revision ${revision}` : ""}. Refreshing the stored map…`,
    };
  } catch (e) {
    return { ok: false, note: `Capture failed: ${why(e, "map ingest", 15)}. Nothing was stored.` };
  }
}
