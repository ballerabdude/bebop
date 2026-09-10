/** Typed API client. All stamps are STRINGS end to end — 19-digit
 * stamp_ns exceeds Number.MAX_SAFE_INTEGER (the legacy tool hit this). */

export interface SessionInfo {
  name: string;
  ticks: number;
  hand: number;
  legacy: boolean;
}

export interface TickSummary {
  stamp: string;
  stamp_ns: number;
  hand: boolean;
  f0: number;
  f1: number;
  f2: number;
  disagree_cells: number;
  unconfirmed_cells: number;
}

export type Grid = number[][]; // 60x60 uint8

export interface TickPayload {
  name: string;
  stamp: string;
  stamp_ns: number;
  legacy: boolean;
  color_near: string | null;
  color_far: string | null;
  depth_near: string | null;
  depth_far: string | null;
  grids: Partial<Record<string, Grid>>;
  manifest: {
    stamp_ns: number;
    cmd_vel?: { vx: number; wz: number };
    odom?: { x: number; y: number; theta: number };
    goal?: { type: string; heading_rad?: number; x?: number; y?: number };
  } | null;
}

export interface OverlayPayload {
  role: string;
  src: string;
  overlay: string | null;
  blend: string | null;
}

export type SamMode = "off" | "sam" | "gate";

export interface SamOverlayPayload {
  role: string;
  mode: "sam" | "gate";
  overlay: string | null;
  blend: string | null;
  floor_frac: number;
  gate: { floor_px: number; blocked_px: number } | null;
}

export interface StageRecord {
  present: boolean;
  size_mb: number | null;
  mtime: number | null;
}

export interface StageSam {
  done: number;
  total: number;
  complete: boolean;
  coverage: number | null;
}

export interface SessionPipeline {
  name: string;
  ticks: number;
  record: StageRecord;
  extract: { ticks: number; manifest: boolean };
  sam: { near: StageSam; far: StageSam };
  fuse: { fused: number; total: number; complete: boolean };
  review: { hand: number };
}

export interface TrainRun {
  name: string;
  epochs: number;
  best_val_miou: number | null;
  best_ious: number[] | null;
  mtime: number;
}

export interface TrainArtifacts {
  runs: TrainRun[];
  onnx: { name: string; size_mb: number; mtime: number }[];
  checkpoints: { name: string; files: string[] }[];
}


export interface GridTexture {
  png: string | null; // 60x60 RGB PNG b64
  coverage: number;
  error?: string;
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(path, init);
  if (!r.ok) {
    let detail = `${r.status}`;
    try {
      detail = (await r.json()).detail ?? detail;
    } catch {
      /* non-JSON error */
    }
    throw new Error(detail);
  }
  return r.json() as Promise<T>;
}

export const api = {
  sessions: () => req<{ sessions: SessionInfo[] }>("/api/sessions"),
  ticks: (s: string) => req<TickSummary[]>(`/api/session/${s}/ticks`),
  tick: (s: string, stamp: string, opts?: { grids?: boolean }) =>
    req<TickPayload>(
      `/api/tick/${s}/${stamp}${opts?.grids === false ? "?grids=false" : ""}`,
    ),
  gridTex: (s: string, stamp: string) =>
    req<GridTexture>(`/api/gridtex/${s}/${stamp}`),
  overlay: (s: string, stamp: string, q: { role: string; src: string; alpha: number }) =>
    req<OverlayPayload>(
      `/api/overlay/${s}/${stamp}?role=${q.role}&src=${q.src}&alpha=${q.alpha}`,
    ),
  samOverlay: (s: string, stamp: string, q: { role: "near" | "far"; mode: "sam" | "gate" }) =>
    req<SamOverlayPayload>(
      `/api/sam/${s}/${stamp}?role=${q.role}&mode=${q.mode}`,
    ),
  pipeline: () => req<{ sessions: SessionPipeline[] }>("/api/pipeline"),
  saveHand: (s: string, stamp: string, grid: Grid) =>
    req<{ ok: boolean }>(`/api/tick/${s}/${stamp}/hand`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ grid }),
    }),
  clearHand: (s: string, stamp: string) =>
    req<{ ok: boolean }>(`/api/tick/${s}/${stamp}/hand`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ clear: true }),
    }),
};
