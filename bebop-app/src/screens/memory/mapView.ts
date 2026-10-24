/// Pure geometry/state helpers for the Map screen. No DOM, no fake data.

export interface MapPose {
  t_ns?: number;
  translation: number[];
  rotation: number[];
  frame_id?: string;
  segment?: number;
  age_s?: number | null;
}

export interface MapStatus {
  session_id?: string;
  map_error?: string | null;
  tracking_state?: string;
  pose_valid?: boolean;
  pose?: MapPose | null;
  segment?: number;
  frames_seen?: number;
  frames_processed?: number;
  frames_tracked?: number;
  frames_lost?: number;
  frames_rejected?: number;
  last_error?: string | null;
  pair_skew_ms?: number | null;
  depth_scale_mm?: number | null;
  backend?: string;
  imu?: { enabled?: boolean; reason?: string };
  [key: string]: unknown;
}

export function clamp(n: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, n));
}

/// Human-readable body size for the poll/payload readout. A real stored place
/// map is a few KB, so never round it away to "0.0 MB".
export function formatPayloadBytes(bytes: number | null | undefined): string {
  if (bytes == null || !Number.isFinite(bytes) || bytes <= 0) return "—";
  if (bytes < 1000) return `${Math.round(bytes)} B`;
  if (bytes < 10_000) return `${(bytes / 1000).toFixed(1)} KB`;
  if (bytes < 1_000_000) return `${Math.round(bytes / 1000)} KB`;
  return `${(bytes / 1_000_000).toFixed(2)} MB`;
}

export function formatAge(seconds: number | null | undefined): string {
  if (seconds == null || !Number.isFinite(seconds) || seconds < 0) return "—";
  if (seconds < 1) return `${seconds.toFixed(2)} s`;
  if (seconds < 60) return `${seconds.toFixed(1)} s`;
  const whole = Math.floor(seconds);
  const d = Math.floor(whole / 86_400);
  const h = Math.floor((whole % 86_400) / 3_600);
  const m = Math.floor((whole % 3_600) / 60);
  if (whole < 3_600) return `${m}m ${whole % 60}s`;
  if (whole < 86_400) return `${h}h ${m}m`;
  return `${d}d ${h}h`;
}

// ---------------------------------------------------------------------------
// 2D floor-plan structure feed (GET /structure)
// ---------------------------------------------------------------------------

export interface PlanGravity {
  valid: boolean;
  /** Gravity "up" in the camera-optical frame; null until estimated. */
  up: number[] | null;
  roll_deg: number | null;
  pitch_deg: number | null;
  inlier_ratio: number | null;
}

export interface PlanFloor {
  valid: boolean;
  /** Floor normal in the camera-optical frame; null until estimated. */
  normal: number[] | null;
  offset_m: number | null;
  camera_height_m: number | null;
  inliers: number | null;
}

export interface PlanGrid {
  /** Map-frame min corner of the grid, in metres. */
  origin: [number, number];
  resolution_m: number;
  width: number;
  height: number;
  occupied_count: number;
  free_count: number;
  encoding: string;
  occupied_rle: string;
  free_rle: string;
}

export interface PlanPose {
  valid: boolean;
  x: number;
  y: number;
  yaw_deg: number;
  t_ns: number;
  segment: number;
}

export interface StructureSnapshot {
  ok: boolean;
  service: string;
  session_id: string;
  /** Null before a map frame exists (pre-floor-support payloads). */
  frame_id: string | null;
  pose_frame: string | null;
  structural: boolean;
  persistent: boolean;
  loop_corrected: boolean;
  gravity_source: string;
  gravity: PlanGravity;
  floor: PlanFloor;
  grid: PlanGrid;
  walls: number[][];
  wall_cells: number[][];
  path: number[][];
  pose: PlanPose;
  trajectory: unknown;
  status: MapStatus;
  revision: number;
  updated_age_s: number | null;
}

export function isStructureSnapshot(value: unknown): value is StructureSnapshot {
  if (!value || typeof value !== "object") return false;
  const v = value as Record<string, unknown>;
  return (
    typeof v.session_id === "string" &&
    (v.frame_id == null || typeof v.frame_id === "string") &&
    (v.pose_frame == null || typeof v.pose_frame === "string") &&
    typeof v.revision === "number" &&
    !!v.gravity && typeof v.gravity === "object" &&
    !!v.floor && typeof v.floor === "object" &&
    !!v.grid && typeof v.grid === "object" &&
    Array.isArray(v.walls) &&
    Array.isArray(v.path) &&
    !!v.pose && typeof v.pose === "object" &&
    !!v.status && typeof v.status === "object"
  );
}

export function normalizeStructure(value: StructureSnapshot): StructureSnapshot {
  const finiteTuple = (p: unknown, n: number) =>
    Array.isArray(p) && p.length === n && p.every(Number.isFinite);
  const vec3 = (v: unknown): number[] | null => {
    if (
      !Array.isArray(v) ||
      v.length !== 3 ||
      !v.every((x) => typeof x === "number" && Number.isFinite(x))
    ) {
      return null;
    }
    return v.map(Number);
  };
  const finiteOrNull = (n: unknown): number | null =>
    typeof n === "number" && Number.isFinite(n) ? n : null;
  const walls = value.walls.filter((w) => finiteTuple(w, 4));
  const wallCells = Array.isArray(value.wall_cells)
    ? value.wall_cells.filter((c) => finiteTuple(c, 2))
    : [];
  const path = value.path.filter((p) => finiteTuple(p, 2));
  const grid = value.grid;
  return {
    ...value,
    // Pre-floor payloads report null frames/gravity/floor; keep those null
    // rather than inventing zeros, so the UI can say "not yet estimated".
    frame_id: typeof value.frame_id === "string" ? value.frame_id : null,
    pose_frame: typeof value.pose_frame === "string" ? value.pose_frame : null,
    structural: value.structural === true,
    persistent: value.persistent === true,
    loop_corrected: value.loop_corrected === true,
    gravity_source:
      typeof value.gravity_source === "string" ? value.gravity_source : "",
    gravity: {
      ...value.gravity,
      valid: value.gravity.valid === true,
      up: vec3(value.gravity.up),
      roll_deg: finiteOrNull(value.gravity.roll_deg),
      pitch_deg: finiteOrNull(value.gravity.pitch_deg),
      inlier_ratio: finiteOrNull(value.gravity.inlier_ratio),
    },
    floor: {
      ...value.floor,
      valid: value.floor.valid === true,
      normal: vec3(value.floor.normal),
      offset_m: finiteOrNull(value.floor.offset_m),
      camera_height_m: finiteOrNull(value.floor.camera_height_m),
      inliers: finiteOrNull(value.floor.inliers),
    },
    walls,
    wall_cells: wallCells,
    path,
    grid: {
      ...grid,
      origin:
        finiteTuple(grid.origin, 2)
          ? [Number(grid.origin[0]), Number(grid.origin[1])]
          : [0, 0],
      resolution_m: Number.isFinite(grid.resolution_m) ? grid.resolution_m : 0,
      width: Number.isFinite(grid.width) ? grid.width : 0,
      height: Number.isFinite(grid.height) ? grid.height : 0,
      occupied_count: Number.isFinite(grid.occupied_count)
        ? grid.occupied_count
        : 0,
      free_count: Number.isFinite(grid.free_count) ? grid.free_count : 0,
      encoding: typeof grid.encoding === "string" ? grid.encoding : "",
      occupied_rle: typeof grid.occupied_rle === "string" ? grid.occupied_rle : "",
      free_rle: typeof grid.free_rle === "string" ? grid.free_rle : "",
    },
    pose: {
      ...value.pose,
      valid: value.pose.valid === true,
      x: Number.isFinite(value.pose.x) ? value.pose.x : 0,
      y: Number.isFinite(value.pose.y) ? value.pose.y : 0,
      yaw_deg: Number.isFinite(value.pose.yaw_deg) ? value.pose.yaw_deg : 0,
      t_ns: Number.isFinite(value.pose.t_ns) ? value.pose.t_ns : 0,
      segment: Number.isFinite(value.pose.segment) ? value.pose.segment : 0,
    },
    updated_age_s:
      value.updated_age_s == null || Number.isFinite(value.updated_age_s)
        ? value.updated_age_s
        : null,
  };
}

export interface PlanView {
  centerX: number;
  centerY: number;
  scale: number;
  width: number;
  height: number;
}

export interface PlanBounds {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
}

/// Map-frame metres -> canvas pixels. Map +X is right, map +Y is up (screen
/// -Y), so a yaw of 0 points right and +90 deg points toward the top.
export function worldToScreen(
  x: number,
  y: number,
  view: PlanView,
): [number, number] {
  return [
    view.width / 2 + (x - view.centerX) * view.scale,
    view.height / 2 - (y - view.centerY) * view.scale,
  ];
}

export function screenToWorld(
  sx: number,
  sy: number,
  view: PlanView,
): [number, number] {
  return [
    view.centerX + (sx - view.width / 2) / view.scale,
    view.centerY - (sy - view.height / 2) / view.scale,
  ];
}

/// True when the snapshot carries anything that can be drawn as a plan.
export function hasStructure(s: StructureSnapshot | null | undefined): boolean {
  if (!s) return false;
  if (s.walls.length > 0 || s.wall_cells.length > 0 || s.path.length > 0)
    return true;
  return (
    s.grid.width > 0 &&
    s.grid.height > 0 &&
    s.grid.free_count + s.grid.occupied_count > 0
  );
}

export function planBounds(
  walls: number[][],
  path: number[][],
  grid?: PlanGrid | null,
): PlanBounds | null {
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  let n = 0;
  const add = (x: number, y: number) => {
    if (!Number.isFinite(x) || !Number.isFinite(y)) return;
    minX = Math.min(minX, x);
    minY = Math.min(minY, y);
    maxX = Math.max(maxX, x);
    maxY = Math.max(maxY, y);
    n += 1;
  };
  for (const w of walls) {
    if (w.length >= 4) {
      add(w[0], w[1]);
      add(w[2], w[3]);
    }
  }
  for (const p of path) {
    if (p.length >= 2) add(p[0], p[1]);
  }
  const g = gridBounds(grid);
  if (g) {
    add(g.minX, g.minY);
    add(g.maxX, g.maxY);
  }
  return n ? { minX, minY, maxX, maxY } : null;
}

/// Axis-aligned footprint of the occupancy grid, in map metres.
export function gridBounds(grid: PlanGrid | null | undefined): PlanBounds | null {
  if (
    !grid ||
    !Number.isFinite(grid.width) ||
    !Number.isFinite(grid.height) ||
    !Number.isFinite(grid.resolution_m) ||
    grid.width <= 0 ||
    grid.height <= 0 ||
    grid.resolution_m <= 0
  ) {
    return null;
  }
  const [ox, oy] = grid.origin;
  if (!Number.isFinite(ox) || !Number.isFinite(oy)) return null;
  return {
    minX: ox,
    minY: oy,
    maxX: ox + grid.width * grid.resolution_m,
    maxY: oy + grid.height * grid.resolution_m,
  };
}

export function fitPlanView(
  bounds: PlanBounds | null,
  width: number,
  height: number,
  padding = 28,
): PlanView {
  if (!bounds || width <= 0 || height <= 0) {
    return { centerX: 0, centerY: 0, scale: 40, width, height };
  }
  const spanX = Math.max(bounds.maxX - bounds.minX, 0.5);
  const spanY = Math.max(bounds.maxY - bounds.minY, 0.5);
  const scale = Math.max(
    0.5,
    Math.min((width - 2 * padding) / spanX, (height - 2 * padding) / spanY),
  );
  return {
    centerX: (bounds.minX + bounds.maxX) / 2,
    centerY: (bounds.minY + bounds.maxY) / 2,
    scale,
    width,
    height,
  };
}

/// Map-frame yaw (degrees, CCW from +X) -> screen unit heading (y down).
export function screenHeading(yawDeg: number): [number, number] {
  const a = (yawDeg * Math.PI) / 180;
  return [Math.cos(a), -Math.sin(a)];
}

/// Zoom about a screen anchor (cursor / pinch midpoint): the world point under
/// the anchor stays under the anchor. Scale is clamped to [min, max].
export function zoomPlanView(
  view: PlanView,
  factor: number,
  anchorSx: number,
  anchorSy: number,
  minScale = 0.5,
  maxScale = 500,
): PlanView {
  if (!Number.isFinite(factor) || factor <= 0) return view;
  const scale = clamp(view.scale * factor, minScale, maxScale);
  if (!Number.isFinite(scale) || scale <= 0) return view;
  const [wx, wy] = screenToWorld(anchorSx, anchorSy, view);
  return {
    centerX: wx - (anchorSx - view.width / 2) / scale,
    centerY: wy + (anchorSy - view.height / 2) / scale,
    scale,
    width: view.width,
    height: view.height,
  };
}

/// Drag-to-pan: a screen-space drag delta moves the map with the finger, so
/// the map centre moves the opposite way in world space.
export function panPlanView(
  view: PlanView,
  dxPx: number,
  dyPx: number,
): PlanView {
  if (!Number.isFinite(dxPx) || !Number.isFinite(dyPx) || view.scale <= 0) {
    return view;
  }
  return {
    ...view,
    centerX: view.centerX - dxPx / view.scale,
    centerY: view.centerY + dyPx / view.scale,
  };
}

/// Largest "nice" length (metres) that fits in `maxPx` at the current scale.
export function scaleBarMeters(scale: number, maxPx: number): number {
  const options = [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100];
  let best = options[0];
  if (!Number.isFinite(scale) || scale <= 0 || !Number.isFinite(maxPx)) {
    return best;
  }
  for (const m of options) if (m * scale <= maxPx) best = m;
  return best;
}

function decodeBase64(text: string): Uint8Array | null {
  try {
    const bin = atob(text);
    const out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i += 1) out[i] = bin.charCodeAt(i);
    return out;
  } catch {
    return null;
  }
}

function readU16LE(bytes: Uint8Array, at: number): number {
  return bytes[at] | (bytes[at + 1] << 8);
}

/// Canonical backend format: a flat array of little-endian uint16 run lengths
/// with no value field. Runs alternate 0,1,0,1,... and always start with a
/// 0-run (the encoder emits a zero-length leading run when cell 0 is set), so
/// total bytes = runs * 2 and the sum of lengths must be exactly the cell count.
function expandRunLengths(
  bytes: Uint8Array,
  expected: number,
): Uint8Array | null {
  if (bytes.length === 0 || bytes.length % 2 !== 0) return null;
  const out = new Uint8Array(expected);
  let at = 0;
  let value: 0 | 1 = 0;
  for (let i = 0; i < bytes.length; i += 2) {
    const count = readU16LE(bytes, i);
    if (at + count > expected) return null;
    out.fill(value, at, at + count);
    at += count;
    value = value === 0 ? 1 : 0;
  }
  return at === expected ? out : null;
}

/// Older/defensive format: little-endian uint16 (value, count) pairs.
/// The backend encodes row-major runs and always starts with a 0-run, so the
/// first run's value must be 0; a zero-count first run is accepted because the
/// encoder emits one when cell 0 is occupied.
function expandU16Runs(
  bytes: Uint8Array,
  valueFirst: boolean,
  expected: number,
): Uint8Array | null {
  if (bytes.length === 0 || bytes.length % 4 !== 0) return null;
  const out = new Uint8Array(expected);
  let at = 0;
  for (let i = 0; i < bytes.length; i += 4) {
    const a = readU16LE(bytes, i);
    const b = readU16LE(bytes, i + 2);
    const value = valueFirst ? a : b;
    const count = valueFirst ? b : a;
    if (value > 1) return null;
    if (i === 0 && value !== 0) return null;
    if (at + count > expected) return null;
    out.fill(value, at, at + count);
    at += count;
  }
  return at === expected ? out : null;
}

function sameCells(a: Uint8Array, b: Uint8Array): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i += 1) if (a[i] !== b[i]) return false;
  return true;
}

/// All plausible decodes of one layer, best first. The canonical wire format is
/// a flat list of uint16 run lengths (`expandRunLengths`); the older
/// `(value,count)` pair layouts stay as fallbacks for safety.
/// `decodeGridMask` picks the candidate that matches the reported cell count,
/// so a wrong interpretation cannot silently produce a fake layer.
function rleCandidates(encoded: string, expectedCells: number): Uint8Array[] {
  const bytes = decodeBase64(encoded);
  if (!bytes) return [];
  const out: Uint8Array[] = [];
  const add = (cells: Uint8Array | null) => {
    if (!cells) return;
    if (out.some((seen) => sameCells(seen, cells))) return;
    out.push(cells);
  };
  add(expandRunLengths(bytes, expectedCells));
  add(expandU16Runs(bytes, true, expectedCells));
  add(expandU16Runs(bytes, false, expectedCells));
  return out;
}

/// Decode the structure grid's base64 RLE into one byte per cell (0/1).
/// Payloads are little-endian uint16 run lengths (see `expandRunLengths`),
/// with the legacy pair layouts as fallbacks. Returns null for malformed input
/// rather than inventing cells; use `decodeGridMask` when the reported
/// `free_count`/`occupied_count` is available to disambiguate.
export function decodeRleBase64(
  encoded: string,
  expectedCells: number,
): Uint8Array | null {
  if (!Number.isFinite(expectedCells) || expectedCells < 0) return null;
  if (expectedCells === 0) return new Uint8Array(0);
  if (!encoded) return null;
  return rleCandidates(encoded, expectedCells)[0] ?? null;
}

export interface GridMask {
  /** One byte per cell (0/1), row-major from `origin`. */
  cells: Uint8Array;
  /** Which byte value marks a cell as inside this layer. */
  value: 0 | 1;
  width: number;
  height: number;
  origin: [number, number];
  resolution_m: number;
}

/// Decode one grid layer (`free_rle` or `occupied_rle`) and cross-check the
/// decoded cell count against the snapshot's own `free_count`/`occupied_count`.
/// `rleCandidates` can offer more than one pair order, and this check decides
/// which one is plausible. Returns null instead of guessing when nothing
/// validates (including non-`rle_base64` encodings).
export function decodeGridMask(
  grid: PlanGrid | null | undefined,
  layer: "free" | "occupied",
): GridMask | null {
  if (!grid) return null;
  if (
    typeof grid.encoding === "string" &&
    grid.encoding.length > 0 &&
    grid.encoding !== "rle_base64"
  ) {
    return null;
  }
  const width = Math.floor(grid.width);
  const height = Math.floor(grid.height);
  const total = width * height;
  if (!Number.isFinite(total) || total <= 0) return null;
  const encoded = layer === "free" ? grid.free_rle : grid.occupied_rle;
  const reported = layer === "free" ? grid.free_count : grid.occupied_count;
  if (typeof encoded !== "string" || encoded.length === 0) return null;
  for (const cells of rleCandidates(encoded, total)) {
    let ones = 0;
    for (let i = 0; i < cells.length; i += 1) ones += cells[i];
    let value: 0 | 1;
    if (ones === reported) value = 1;
    else if (total - ones === reported) value = 0;
    else continue;
    return {
      cells,
      value,
      width,
      height,
      origin: [grid.origin[0], grid.origin[1]],
      resolution_m: grid.resolution_m,
    };
  }
  return null;
}

export interface GridRun {
  /** Row index in the grid. */
  iy: number;
  /** Half-open cell range [x0, x1) in that row. */
  x0: number;
  x1: number;
}

/// Merge each grid row into horizontal runs, so drawing needs one rectangle
/// per run instead of one per cell. Stops after `limit` runs (the render
/// primitive budget), and never allocates the whole grid twice.
export function gridRuns(
  mask: GridMask | null | undefined,
  limit = Number.POSITIVE_INFINITY,
): GridRun[] {
  const out: GridRun[] = [];
  if (!mask || mask.width <= 0 || mask.height <= 0) return out;
  const max = Number.isFinite(limit)
    ? Math.max(0, Math.floor(limit))
    : Number.POSITIVE_INFINITY;
  if (max === 0) return out;
  outer: for (let iy = 0; iy < mask.height; iy += 1) {
    let run = -1;
    const row = iy * mask.width;
    for (let ix = 0; ix <= mask.width; ix += 1) {
      const on =
        ix < mask.width && mask.cells[row + ix] === mask.value;
      if (on) {
        if (run < 0) run = ix;
      } else if (run >= 0) {
        out.push({ iy, x0: run, x1: ix });
        run = -1;
        if (out.length >= max) break outer;
      }
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Per-place local map (GET /places/{id}/map) and place graph
// (GET /places/graph) on the Places service (:9095)
//
// Local-map coordinates are metres in the place's own anchor-relative frame
// (z = floor plane). Nothing here synthesizes walls, cells, or routes: a
// payload that does not parse is rejected so the UI can say "no map yet".
// ---------------------------------------------------------------------------

export interface PlaceMapAnchor {
  x: number;
  y: number;
  yaw_deg: number;
}

export interface PlaceMapGravity {
  /** `gravity.valid` when the backend sends it; null means "not reported". */
  valid: boolean | null;
  up: number[] | null;
  roll_deg: number | null;
  pitch_deg: number | null;
}

export interface PlaceMap {
  place_id: string;
  /** Anchor marker in the place frame; null when the payload omits it. */
  anchor: PlaceMapAnchor | null;
  /** Grid min corner, metres in the place frame. */
  origin: [number, number];
  resolution_m: number;
  width: number;
  height: number;
  occupied_rle: string;
  free_rle: string;
  /** The stored grid reports these; prefer them over recomputing counts. */
  free_count: number | null;
  occupied_count: number | null;
  /** Grid encoding (the live service sends `rle_base64`). */
  encoding: string;
  walls: number[][];
  gravity: PlaceMapGravity;
  camera_height_m: number | null;
  revision: number;
  updated_at: number | null;
}

function finiteOrNull(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function stringOr(value: unknown, fallback = ""): string {
  return typeof value === "string" ? value : fallback;
}

/// The live service wraps the body as `{"map": {...}}`; the flat variant puts
/// the same fields at the top level. Unwrap once so both parse identically.
export function unwrapPlaceMap(value: unknown): Record<string, unknown> | null {
  if (!value || typeof value !== "object") return null;
  const v = value as Record<string, unknown>;
  if (v.map && typeof v.map === "object" && !Array.isArray(v.map)) {
    return v.map as Record<string, unknown>;
  }
  return v;
}

/// The stored grid fields live under `grid` in the live payload; the flat
/// fallback keeps them at the top level. Never mix the two: pick one source.
function placeMapGridSource(
  raw: Record<string, unknown>,
): Record<string, unknown> {
  const grid = raw.grid;
  if (grid && typeof grid === "object" && !Array.isArray(grid)) {
    return grid as Record<string, unknown>;
  }
  return raw;
}

export function isPlaceMapPayload(value: unknown): value is Record<string, unknown> {
  const raw = unwrapPlaceMap(value);
  if (!raw || typeof raw.place_id !== "string") return false;
  const grid = placeMapGridSource(raw);
  return (
    Number.isFinite(grid.width) &&
    Number.isFinite(grid.height) &&
    Number.isFinite(grid.resolution_m)
  );
}

export function normalizePlaceMap(value: unknown): PlaceMap {
  const raw = unwrapPlaceMap(value) ?? {};
  const grid = placeMapGridSource(raw);
  const pair = (v: unknown): [number, number] | null =>
    Array.isArray(v) && v.length === 2 && v.every(Number.isFinite)
      ? [Number(v[0]), Number(v[1])]
      : null;
  const vec3 = (v: unknown): number[] | null =>
    Array.isArray(v) && v.length === 3 && v.every(Number.isFinite)
      ? v.map(Number)
      : null;
  const anchorObj =
    raw.anchor && typeof raw.anchor === "object"
      ? (raw.anchor as Record<string, unknown>)
      : {};
  const origin = pair(grid.origin) ?? [
    Number.isFinite(grid.origin_x) ? Number(grid.origin_x) : 0,
    Number.isFinite(grid.origin_y) ? Number(grid.origin_y) : 0,
  ];
  // Live shape: anchor {x,y,yaw_deg}. Flat fallback: anchor_x/anchor_y/
  // anchor_yaw_deg. Never invent an anchor when all three are absent.
  const ax = finiteOrNull(anchorObj.x) ?? finiteOrNull(raw.anchor_x);
  const ay = finiteOrNull(anchorObj.y) ?? finiteOrNull(raw.anchor_y);
  const yaw =
    finiteOrNull(anchorObj.yaw_deg) ?? finiteOrNull(raw.anchor_yaw_deg);
  const gravity =
    raw.gravity && typeof raw.gravity === "object"
      ? (raw.gravity as Record<string, unknown>)
      : {};
  const walls = Array.isArray(raw.walls)
    ? raw.walls.filter(
        (w): w is number[] =>
          Array.isArray(w) && w.length === 4 && w.every(Number.isFinite),
      )
    : [];
  return {
    place_id: typeof raw.place_id === "string" ? raw.place_id : "",
    anchor:
      ax != null && ay != null && yaw != null
        ? { x: ax, y: ay, yaw_deg: yaw }
        : null,
    origin,
    resolution_m: Number(grid.resolution_m),
    width: Math.max(0, Math.floor(Number(grid.width))),
    height: Math.max(0, Math.floor(Number(grid.height))),
    occupied_rle: stringOr(grid.occupied_rle),
    free_rle: stringOr(grid.free_rle),
    // The live payload carries these; decodeGridMask uses them to pick the
    // correct RLE interpretation instead of guessing.
    free_count: finiteOrNull(grid.free_count) ?? finiteOrNull(raw.free_count),
    occupied_count:
      finiteOrNull(grid.occupied_count) ?? finiteOrNull(raw.occupied_count),
    encoding: stringOr(grid.encoding, "rle_base64"),
    walls,
    gravity: {
      valid:
        typeof gravity.valid === "boolean" ? gravity.valid : null,
      up: vec3(gravity.up),
      roll_deg: finiteOrNull(gravity.roll_deg),
      pitch_deg: finiteOrNull(gravity.pitch_deg),
    },
    camera_height_m: finiteOrNull(raw.camera_height_m),
    revision: Number.isFinite(raw.revision) ? Number(raw.revision) : 0,
    updated_at: finiteOrNull(raw.updated_at),
  };
}

/// The stored per-place layers use the same row-major RLE as /structure but
/// the contract does not include free/occupied counts. Adapt to `PlanGrid`
/// (counts stay null) so callers can share the bounds/fit helpers.
export function placeMapGrid(map: PlaceMap | null | undefined): PlanGrid | null {
  if (
    !map ||
    !(map.width > 0) ||
    !(map.height > 0) ||
    !(map.resolution_m > 0) ||
    !Number.isFinite(map.origin[0]) ||
    !Number.isFinite(map.origin[1])
  ) {
    return null;
  }
  return {
    origin: [map.origin[0], map.origin[1]],
    resolution_m: map.resolution_m,
    width: map.width,
    height: map.height,
    occupied_count: map.occupied_count ?? 0,
    free_count: map.free_count ?? 0,
    encoding: map.encoding,
    occupied_rle: map.occupied_rle,
    free_rle: map.free_rle,
  };
}

/// Decode one stored per-place layer. When the payload carries a cell count,
/// cross-check it through `decodeGridMask` (and reject a mismatch). Otherwise
/// decode the canonical uint16 run-length vector directly; without a count
/// there is nothing to disambiguate, so the layer is used only if it decodes
/// to exactly width*height cells. Returns null instead of inventing a layer.
export function decodePlaceLayer(
  map: PlaceMap | null | undefined,
  layer: "free" | "occupied",
): GridMask | null {
  const grid = placeMapGrid(map);
  if (!grid || !map) return null;
  const reported = layer === "free" ? map.free_count : map.occupied_count;
  if (reported != null) return decodeGridMask(grid, layer);
  const encoded = layer === "free" ? map.free_rle : map.occupied_rle;
  if (!encoded) return null;
  const total = grid.width * grid.height;
  const cells = decodeRleBase64(encoded, total);
  if (!cells) return null;
  return {
    cells,
    value: 1,
    width: grid.width,
    height: grid.height,
    origin: [grid.origin[0], grid.origin[1]],
    resolution_m: grid.resolution_m,
  };
}

/// A place has a drawable map only when the payload actually carries walls or
/// a non-empty occupancy grid. An anchor alone is not a map.
export function hasPlaceMapContent(map: PlaceMap | null | undefined): boolean {
  if (!map) return false;
  if (map.walls.length > 0) return true;
  return (
    (map.free_rle.length > 0 || map.occupied_rle.length > 0) &&
    placeMapGrid(map) != null
  );
}

/// Fit bounds for a place: walls + grid footprint + the anchor marker.
export function placeMapBounds(
  map: PlaceMap | null | undefined,
): PlanBounds | null {
  if (!map) return null;
  const anchor = map.anchor;
  return planBounds(
    map.walls,
    anchor ? [[anchor.x, anchor.y]] : [],
    placeMapGrid(map),
  );
}

/// Age of a stored place map. `updated_at` may be epoch seconds (contract) or
/// epoch milliseconds (defensive); negatives and clock skew clamp to 0.
export function placeMapAgeSeconds(
  updatedAt: number | null | undefined,
  nowMs: number,
): number | null {
  if (
    updatedAt == null ||
    !Number.isFinite(updatedAt) ||
    !Number.isFinite(nowMs)
  ) {
    return null;
  }
  const atMs = Math.abs(updatedAt) > 1e11 ? updatedAt : updatedAt * 1000;
  return Math.max(0, (nowMs - atMs) / 1000);
}

/// Stored-map gravity. `gravity.valid` is authoritative when the backend sends
/// it; an older payload that only carries up/tilt fields is treated as
/// estimated. No map at all is not a gravity claim either way.
export function placeMapHasGravity(map: PlaceMap | null | undefined): boolean {
  if (!map) return false;
  if (map.gravity.valid != null) return map.gravity.valid;
  return (
    map.gravity.up != null ||
    map.gravity.roll_deg != null ||
    map.gravity.pitch_deg != null
  );
}

/// Stat/legend copy that never confuses "no map stored" with "stored map has
/// no gravity estimate".
export function placeMapGravityLabel(
  map: PlaceMap | null | undefined,
): string {
  if (!map) return "no map captured yet";
  if (!placeMapHasGravity(map)) return "gravity not estimated";
  if (!map.gravity.up) return "estimated";
  return map.gravity.up.map((v) => v.toFixed(2)).join(", ");
}

/// Honest one-line legend for the per-place view. The app never has a loop
/// correction or metric-ground-truth claim for a stored place map; with no
/// stored map there is nothing to describe beyond that absence.
export function placeMapLegend(
  placeName: string,
  map: PlaceMap | null | undefined,
): string {
  const name = placeName.trim() || "unknown place";
  if (!map) return `local map for ${name} · no map captured yet`;
  return [
    `local map for ${name}`,
    placeMapHasGravity(map)
      ? "gravity estimated from floor"
      : "gravity not estimated yet",
    "not loop-corrected",
  ].join(" · ");
}

export interface PlaceRecognition {
  matched?: boolean;
  place?: { id?: string; name?: string } | null;
}

export interface PlaceCaptureGate {
  allowed: boolean;
  /** Clear operator-facing reason when `allowed` is false. */
  reason: string | null;
}

/// Capture may only store the selected place's map while the robot is
/// recognized there. Storing it anywhere else would persist a wrong anchor.
export function placeCaptureGate(
  selectedId: string | null,
  selectedName: string,
  recognition: PlaceRecognition | null | undefined,
  paused = false,
): PlaceCaptureGate {
  const recognized =
    recognition?.matched && recognition.place ? recognition.place : null;
  const recognizedId =
    recognized && typeof recognized.id === "string" ? recognized.id : null;
  const recognizedName =
    recognized && typeof recognized.name === "string" && recognized.name
      ? recognized.name
      : recognizedId;
  if (paused) {
    return { allowed: false, reason: "Resume polling before capturing a map." };
  }
  if (!selectedId) {
    return { allowed: false, reason: "Select a place first." };
  }
  if (recognizedId === selectedId) {
    return { allowed: true, reason: null };
  }
  if (recognizedId) {
    return {
      allowed: false,
      reason: `Capture is disabled: the robot is recognized as "${recognizedName}", not "${selectedName}". Select "${recognizedName}" to capture its map.`,
    };
  }
  return {
    allowed: false,
    reason:
      "Capture is disabled: the robot is not recognized at any taught place right now.",
  };
}

/// 404 on `/places/{id}/map` means either "this place has no map captured" or
/// "this service build predates the endpoint". A 404 from the sibling graph
/// endpoint is the only honest signal available for the latter.
export type PlaceMapMissingKind = "no-map" | "endpoint-missing";

export function placeMapMissingKind(
  mapStatus: number | null,
  graphStatus: number | null,
): PlaceMapMissingKind {
  if (mapStatus === 404 && graphStatus === 404) return "endpoint-missing";
  return "no-map";
}

// --- place graph ---------------------------------------------------------

export interface PlaceGraphNode {
  id: string;
  name: string;
}

export interface PlaceGraphEdge {
  from: string;
  to: string;
  traversals: number;
  successes: number;
  median_duration_s: number | null;
  last_success_at: number | null;
}

export interface PlaceGraph {
  nodes: PlaceGraphNode[];
  edges: PlaceGraphEdge[];
}

export function isPlaceGraph(value: unknown): value is Record<string, unknown> {
  if (!value || typeof value !== "object") return false;
  const v = value as Record<string, unknown>;
  return Array.isArray(v.nodes) && Array.isArray(v.edges);
}

/// Defensive normalize: keep only nodes with a non-empty id, dedupe ids, and
/// drop edges whose endpoints are not in the node list. Counts floor at 0.
export function normalizePlaceGraph(value: unknown): PlaceGraph {
  const v =
    value && typeof value === "object" ? (value as Record<string, unknown>) : {};
  const nodes: PlaceGraphNode[] = [];
  const seen = new Set<string>();
  for (const raw of Array.isArray(v.nodes) ? v.nodes : []) {
    if (!raw || typeof raw !== "object") continue;
    const r = raw as Record<string, unknown>;
    if (typeof r.id !== "string" || r.id.length === 0 || seen.has(r.id)) continue;
    seen.add(r.id);
    nodes.push({
      id: r.id,
      name:
        typeof r.name === "string" && r.name.length > 0 ? r.name : r.id,
    });
  }
  const edges: PlaceGraphEdge[] = [];
  for (const raw of Array.isArray(v.edges) ? v.edges : []) {
    if (!raw || typeof raw !== "object") continue;
    const r = raw as Record<string, unknown>;
    if (typeof r.from !== "string" || typeof r.to !== "string") continue;
    if (!seen.has(r.from) || !seen.has(r.to)) continue;
    const traversals = Math.max(
      0,
      Math.floor(finiteOrNull(r.traversals) ?? 0),
    );
    const successes = Math.min(
      traversals,
      Math.max(0, Math.floor(finiteOrNull(r.successes) ?? 0)),
    );
    edges.push({
      from: r.from,
      to: r.to,
      traversals,
      successes,
      median_duration_s: finiteOrNull(r.median_duration_s),
      last_success_at: finiteOrNull(r.last_success_at),
    });
  }
  return { nodes, edges };
}

export interface GraphNodeLayout {
  id: string;
  name: string;
  x: number;
  y: number;
  degree: number;
}

export interface GraphEdgeLayout {
  from: string;
  to: string;
  x0: number;
  y0: number;
  x1: number;
  y1: number;
  labelX: number;
  labelY: number;
  edge: PlaceGraphEdge;
}

export interface GraphLayout {
  nodes: GraphNodeLayout[];
  edges: GraphEdgeLayout[];
  /** Nodes/edges omitted by the render caps. */
  hiddenNodes: number;
  hiddenEdges: number;
}

/// Deterministic circular layout — readable, not geographic. Nodes are ordered
/// by the payload's node order; edges are straight lines between them. The
/// caller renders the counts, so this is honest about being a route graph
/// rather than a measured map.
export function layoutPlaceGraph(
  graph: PlaceGraph,
  width: number,
  height: number,
  padding = 48,
  maxNodes = 256,
  maxEdges = 2048,
): GraphLayout {
  const w = Number.isFinite(width) && width > 0 ? width : 0;
  const h = Number.isFinite(height) && height > 0 ? height : 0;
  const nodeCap = Math.max(0, Math.floor(maxNodes));
  const edgeCap = Math.max(0, Math.floor(maxEdges));
  const nodes = graph.nodes.slice(0, nodeCap);
  const included = new Set(nodes.map((n) => n.id));
  const allEdges = graph.edges.filter(
    (e) => included.has(e.from) && included.has(e.to) && e.from !== e.to,
  );
  const edges = allEdges.slice(0, edgeCap);

  const degree = new Map<string, number>();
  for (const e of allEdges) {
    degree.set(e.from, (degree.get(e.from) ?? 0) + 1);
    degree.set(e.to, (degree.get(e.to) ?? 0) + 1);
  }

  const cx = w / 2;
  const cy = h / 2;
  const radius = Math.max(
    0,
    Math.min(w, h) / 2 - Math.max(0, padding),
  );
  const positions = new Map<string, { x: number; y: number }>();
  const laidOut = nodes.map((node, i) => {
    let x = cx;
    let y = cy;
    if (nodes.length === 2) {
      x = cx + (i === 0 ? -radius : radius);
    } else if (nodes.length > 1) {
      const a = -Math.PI / 2 + (2 * Math.PI * i) / nodes.length;
      x = cx + Math.cos(a) * radius;
      y = cy + Math.sin(a) * radius;
    }
    positions.set(node.id, { x, y });
    return {
      id: node.id,
      name: node.name,
      x,
      y,
      degree: degree.get(node.id) ?? 0,
    };
  });

  const offsets = new Map<string, number>();
  const edgeLayouts: GraphEdgeLayout[] = [];
  for (const edge of edges) {
    const a = positions.get(edge.from);
    const b = positions.get(edge.to);
    if (!a || !b) continue;
    const key = [edge.from, edge.to].sort().join("\u0000");
    const n = offsets.get(key) ?? 0;
    offsets.set(key, n + 1);
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    const len = Math.hypot(dx, dy) || 1;
    // Fan reciprocal/multi edges apart so their labels don't stack.
    const shift = (n % 2 === 0 ? 1 : -1) * Math.ceil(n / 2) * 14;
    const nx = (-dy / len) * shift;
    const ny = (dx / len) * shift;
    edgeLayouts.push({
      from: edge.from,
      to: edge.to,
      x0: a.x + nx,
      y0: a.y + ny,
      x1: b.x + nx,
      y1: b.y + ny,
      labelX: (a.x + b.x) / 2 + nx,
      labelY: (a.y + b.y) / 2 + ny - 4,
      edge,
    });
  }

  return {
    nodes: laidOut,
    edges: edgeLayouts,
    hiddenNodes: Math.max(0, graph.nodes.length - nodes.length),
    // Self-loop edges are not drawable routes; everything else that is not in
    // `edges` (node cap, edge cap) counts as hidden for the caption.
    hiddenEdges: Math.max(
      0,
      graph.edges.filter((e) => e.from !== e.to).length - edges.length,
    ),
  };
}

/// Compact edge label: recorded traversal count and how many succeeded.
export function graphEdgeLabel(edge: PlaceGraphEdge): string {
  const duration =
    edge.median_duration_s != null && Number.isFinite(edge.median_duration_s)
      ? ` · ${edge.median_duration_s.toFixed(1)} s`
      : "";
  return `${edge.traversals}× · ${edge.successes} ok${duration}`;
}
