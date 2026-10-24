// Client-side mirror of `Supervisor.load()` in
// bebop-vision/bebop_vision/models_supervisor.py: which resident backends a
// load would stop to fit the GPU budget. Settings › Models uses it to name
// them before the operator taps Load, since an eviction can end a voice turn.

export interface EvictionBackend {
  id: string;
  kind: string;
  state: string;
  footprint_gb: number;
}

/// Backends (in supervisor order) that loading `targetId` would stop.
/// `memAvailableGb` (healthz `mem.MemAvailable`) adds the free-memory check
/// the supervisor also applies; omit it when unknown.
export function predictEvictions(
  backends: EvictionBackend[],
  targetId: string,
  budgetGb: number,
  memAvailableGb?: number,
): EvictionBackend[] {
  const target = backends.find((b) => b.id === targetId);
  if (!target || ["ready", "starting", "tracked"].includes(target.state)) return [];
  const needed = target.footprint_gb;
  const live = new Set(backends.filter((b) => b.state === "ready" || b.state === "tracked").map((b) => b.id));
  let free = memAvailableGb;
  const resident = () =>
    backends.filter((b) => live.has(b.id)).reduce((sum, b) => sum + b.footprint_gb, 0);
  const evicted: EvictionBackend[] = [];
  for (const other of backends) {
    if (other.id === targetId || other.kind === "tracked") continue;
    if (other.state !== "ready" && other.state !== "starting") continue;
    if (resident() + needed <= budgetGb && (free === undefined || free >= needed)) break;
    evicted.push(other);
    if (live.delete(other.id) && free !== undefined) free += other.footprint_gb;
  }
  return evicted;
}
