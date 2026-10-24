// Place-memory types and REST calls for the places service (:9095, inside
// bebop-vision). Reads go through the session's shared poller (`usePlaces`);
// these are the writes and per-place reads the Memory screens make.

export interface Fact {
  text: string;
  source: string;
  created_at: number;
}

export interface Keyframe {
  path: string;
  role: string;
  stamp_us: number;
}

export interface BestMatch {
  id: string;
  name: string;
  score: number | null;
}

export interface Place {
  id: string;
  name: string;
  display_name?: string;
  status?: string;
  source?: string;
  seen_count?: number;
  suggested_name?: string | null;
  suggested_facts?: string[];
  best_match?: BestMatch | null;
  created_at: number;
  updated_at: number;
  last_seen_at: number | null;
  facts: Fact[];
  keyframes: Keyframe[];
}

export interface WhereAmI {
  matched?: boolean;
  place?: { id: string; name: string } | null;
  score?: number;
  margin?: number;
  min_score?: number;
  reason?: string;
  top?: { id: string; name: string; score: number }[];
}

export type PlacesUrl = (path: string) => string;

/// Score at or above which a pending place offers "Merge into <match>".
export const MERGE_SCORE = 0.8;

/// Match score as shown to the operator: truncated (not rounded) to two
/// decimals, so a 0.7992 reads "0.79" and never contradicts the 0.80 merge
/// threshold it fails.
export function formatMatchScore(score: number): string {
  return (Math.floor(score * 100 + 1e-9) / 100).toFixed(2);
}

export function placeTitle(p: Place): string {
  return p.display_name || p.name;
}

export function plural(n: number, word: string): string {
  return `${n} ${word}${n === 1 ? "" : "s"}`;
}

export function keyframeUrl(api: PlacesUrl, placeId: string, path: string): string {
  const file = path.split("/").pop() ?? path;
  return api(`/places/${encodeURIComponent(placeId)}/keyframes/${encodeURIComponent(file)}`);
}

/// Seconds-since-epoch to a short local "Oct 21, 14:03".
export function shortTime(epochS: number | null | undefined): string {
  if (!epochS) return "—";
  return new Date(epochS * 1000).toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

async function readError(res: Response): Promise<string> {
  try {
    const body = await res.json();
    if (body && typeof body.error === "string") return body.error;
  } catch {
    /* not JSON */
  }
  return `${res.status} ${res.statusText}`;
}

async function send<T = Record<string, unknown>>(
  url: string,
  method: "POST" | "DELETE",
  body?: unknown,
): Promise<T> {
  const res = await fetch(url, {
    method,
    headers: body === undefined ? undefined : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!res.ok) throw new Error(await readError(res));
  return (await res.json().catch(() => ({}))) as T;
}

const placeUrl = (api: PlacesUrl, id: string, suffix = "") =>
  api(`/places/${encodeURIComponent(id)}${suffix}`);

export function teachPlace(api: PlacesUrl, name: string) {
  return send<{ place: Place; views: number; descriptors: number }>(api("/places"), "POST", { name });
}

export function addFact(api: PlacesUrl, id: string, text: string) {
  return send(placeUrl(api, id, "/facts"), "POST", { text });
}

export function captureViews(api: PlacesUrl, id: string, seconds: number) {
  return send<{ keyframes: number; views_total: number }>(
    placeUrl(api, id, "/views"),
    "POST",
    { seconds, interval_s: 1.5 },
  );
}

export function deletePlace(api: PlacesUrl, id: string) {
  return send(placeUrl(api, id), "DELETE");
}

export interface SuggestResult {
  suggested_name?: string;
  facts?: string[];
  conflict?: { name: string } | null;
  error?: string;
}

export async function suggestName(api: PlacesUrl, id: string): Promise<SuggestResult> {
  const body = await send<SuggestResult>(
    api(`/places/pending/${encodeURIComponent(id)}/suggest`),
    "POST",
    {},
  );
  if (body.error) throw new Error(body.error);
  return body;
}

export interface ImportResult {
  imported?: number;
  merged?: number;
  failed?: number;
  results?: { id: string; status?: string; detail?: string; error?: string }[];
}

export function importPending(
  api: PlacesUrl,
  ids: string[],
  opts: { names?: Record<string, string>; mergeInto?: Record<string, string> },
) {
  return send<ImportResult>(api("/places/pending/import"), "POST", {
    ids,
    ...(opts.names ? { names: opts.names } : {}),
    ...(opts.mergeInto ? { merge_into: opts.mergeInto } : {}),
  });
}
