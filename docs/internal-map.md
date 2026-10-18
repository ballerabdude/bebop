# internal map — per-place local maps and the place graph

Status: **backend slice landed (2026-10)**. The single global metric map is
retired as a product. Each taught place can own a local, gravity-aligned
submap, and places are linked by a graph of *routes that actually worked*.
This is the map a person carries, not a survey: local detail plus landmark
recognition, with no claim of global metric consistency. Nothing here is a
safety or navigation map yet.

Live SLAM's pose and structure feed are described in [`slam.md`](slam.md).

## 1. Goal and rationale

Human navigation mixes two kinds of knowledge:

| Knowledge | What it is | Cost / failure mode |
|---|---|---|
| **Route knowledge** | landmark-conditioned action sequences: "turn left at the fridge, then the second door" | cheap, drift-tolerant, brittle when a landmark changes |
| **Survey knowledge** | landmarks and their spatial relations, built slowly and locally detailed | expensive, locally accurate, not globally consistent |

Nobody carries a globally consistent metric map in their head. Landmark
recognition is what resets accumulated error: seeing the kitchen tells you
where you are even if dead reckoning drifted. The backend follows that model:

- **one local submap per place**, anchored at the capture pose;
- **a place graph** whose edges are remembered routes with success counts;
- **DINOv3 place recognition** as the eventual re-anchoring mechanism.

A drifting local patch therefore cannot corrupt the global layout, because
there is no global layout. A renamed room cannot invalidate geometry, because
geometry belongs to the place.

## 2. Structure feed (the submap source)

`GET /structure` on the live-SLAM service (`:9096`, schema in
[`slam.md`](slam.md)) produces a gravity-aligned, per-session occupancy grid
from RGB-D depth plus visual odometry. Key properties:

- gravity is **estimated from the floor plane**, never from an IMU;
- free space is carved along rays to every depth hit, including floor hits;
- the **occupied layer is the obstacle layer**: desks, chairs, boxes and
  people are all rendered;
- **walls are a narrower subset**: a candidate must clear a modest height
  floor (~0.8 m, just above desk height), then its connected component must
  be elongated and thin and must bound observed free space. The height floor
  only removes low clutter; shape + boundary do the wall-vs-furniture
  discrimination. A compact obstacle is occupied but produces no wall line.
  Details and the known "wall seen from both sides" limitation are in
  `slam.md`;
- wall line segments are fitted only from accepted wall cells, with segments
  shorter than 0.8 m dropped;
- there is **no loop closure, relocalization, or persistence across
  sessions**. The snapshot is a local patch, and that is exactly what the
  places store consumes.

## 3. Per-place local maps

`places/store.py` stores at most one current submap per place in
`place_maps`:

| Column | Meaning |
|---|---|
| `place_id` | FK to `places`, primary key (one row per place, upsert) |
| `anchor_x`, `anchor_y` | capture pose in the vision `map` frame, metres |
| `anchor_yaw_deg` | capture heading in that frame, degrees CCW from +x |
| `origin_x`, `origin_y` | near corner of the accumulated union bbox in the **same** vision frame, metres |
| `resolution_m` | cell size (0.05 m in practice); a re-capture is rebinned to the stored resolution |
| `width`, `height` | cell counts of the accumulated union bbox |
| `occupied_rle`, `free_rle` | row-major base64 uint16 RLE of the unioned occupied/free evidence |
| `walls_json` | accumulated `[[x0,y0,x1,y1], ...]` segments in the same frame (duplicates dropped) |
| `gravity_json` | `{valid, up, roll_deg, pitch_deg, [inlier_ratio]}` (camera-optical up) |
| `camera_height_m` | camera height above the estimated floor |
| `revision` | 1 on first capture, +1 on every re-capture |
| `created_at`, `updated_at` | timestamps; `created_at` survives refinement |

### The anchor and the frame

The submap is **not** rotated or translated into a synthetic "place frame".
It is a local patch captured in one session's gravity-aligned vision `map`
frame, so the raster, wall lines and anchor are mutually consistent exactly as
the structure service produced them. The anchor records where the robot was
on the **first** capture; it is pinned there and is never moved by a later
capture. Anchor-relative coordinates are:

```
p_local = Rz(-anchor_yaw_deg) · (p_frame - [anchor_x, anchor_y])
```

Consequences, stated honestly:

- Two captures from **different sessions** have unrelated `map` frames until
  relocalization exists. The accumulator assumes the incoming cells share the
  stored frame (true for captures within one session).
- Re-capturing a place **accumulates evidence**: the stored and incoming
  occupied/free cell sets are unioned, walls are unioned with near-duplicates
  dropped, and the result is rebinned into a union bounding box. Cells are
  placed by their capture-frame centre
  (`origin + (i + 0.5) * resolution_m`), so captures with different
  `origin`/`width`/`height` add evidence instead of overwriting each other.
  Counts never shrink; an empty or weak snapshot cannot erase a better map.
  `revision` increments on every capture and `created_at` survives.
- **Reset is explicit.** `POST /places/{id}/map` with `{"replace": true}`
  deliberately starts the place over: the row becomes exactly the new patch,
  anchor included. Merge is the default.
- A capture for place B only ever writes B's row. Place isolation is the
  primary key.

### Coordinate-system rule

Submap coordinates come **only** from the structure snapshot's visual pose
and grid. The legacy `places.odom_x/odom_y/odom_theta` columns (wheel
odometry captured at teach time) are retained for display/compatibility and
are **never read** by the map or graph code. A test asserts that an anchor is
the visual pose even when the place carries deliberately wrong odometry.

### API (`:9095`)

| Method | Path | Behaviour |
|---|---|---|
| `POST` | `/places/{id}/map` | accumulate a `/structure` payload (or `{"structure": ..., "pose": ..., "replace": true}`); 201 on first capture, 200 on refinement. Merge is the default; `replace` resets the patch |
| `GET` | `/places/{id}/map` | the stored submap; 404 when the place is unknown or has never been mapped |
| `GET` | `/places/graph` | nodes (places, `has_map`, `map_revision`) + edges |
| `POST` | `/places/{from}/edges/{to}` | record one traversal (`success`, optional `duration_s`, optional `guide_path`) |

Malformed payloads are rejected with 400 rather than stored: missing/invalid
grid, wrong encoding, RLE that does not cover exactly `width * height` cells,
non-finite geometry, missing gravity/floor, invalid pose, or malformed walls.
Unknown places are 404.

## 4. Place graph — route knowledge

`place_edges` is deliberately not a metric transform. An edge means "you got
from A to B by this route". It carries:

| Column | Meaning |
|---|---|
| `from_place`, `to_place` | FKs; composite primary key (unique directed edge) |
| `traversals` | attempts recorded |
| `successes` | attempts where arrival was confirmed |
| `last_attempt_at`, `last_success_at` | timestamps |
| `median_duration_s` | median over the last ≤100 successful durations supplied |
| `guide_path_json` | `[[x,y], ...]` in the **from-place's local frame** |
| `durations_json` | internal auxiliary list used to maintain the median exactly |
| `created_at`, `updated_at` | timestamps |

Semantics:

- a failure counts as an attempt, never as a success;
- a success without a duration counts, but is not a timing sample;
- a guide path is recorded only for successful traversals (a failed route is
  not a guide);
- `success_rate = successes / traversals` is computed on read;
- the graph is directed: A→B and B→A are separate edges;
- deleting a place cascades its submap and both directions of every incident
  edge.

Caller-supplied guide paths are stored verbatim and must already be expressed
in the from-place's local frame (origin at its anchor, +x along its heading).
The service does not reinterpret them, because the caller may have composed
them from a local planner or a corrected path.

## 5. What exists vs. what is missing

Exists:

- live camera-optical pose with explicit tracking/staleness semantics;
- gravity-estimated, gravity-aligned structure with free-space carving;
- **obstacle layer** plus a filtered **wall layer** (compact furniture is
  occupied but never a wall);
- per-place local submaps persisted in SQLite, with anchor, revision and
  RLE/wall geometry;
- a persistent place graph with traversal/success counters, median duration,
  and optional guide paths;
- persistent semantic place memory with DINOv3 appearance recognition;
- tests for revision refinement, place isolation, RLE round-trip, graph
  serialization, success semantics, malformed-payload rejection, and the
  no-wheel-odometry rule.

Missing, in the order that unblocks the most:

1. **Relocalization / loop closure.** Recognition can name a place, but it
   does not yet recover a metric relationship between a new session's frame
   and a stored submap. This is what would let DINOv3 reset accumulated error.
2. **Anchor persistence across sessions in a usable sense.** The anchor row is
   persisted, but its frame is session-local. Without (1) it is a record of
   where the capture happened, not a coordinate another session can drive to.
3. **Real traversal recording.** The endpoint and counters exist, but nothing
   in the drive loop (`--goal-drive`) posts a traversal yet, and guide paths
   are caller-supplied and unverified. Edges are therefore empty or
   test-authored on a fresh robot.
4. **Submap refinement within a session.** Re-capture replaces geometry and
   bumps the revision; it does not merge two patches that share a frame
   (nor average repeated observations).
5. **Route planning over the graph.** The graph stores experience; there is
   no A→B planner or "unknown place → known place" policy consuming it.
6. **LLM vision labeling** of rooms and wall features, attached to places.

## 6. Honest summary

Today the robot can build a per-session floor plan with obstacles and real
walls, store a local gravity-aligned submap per named place, and remember
which routes between places were attempted and which succeeded. It cannot yet
relocalize a new session into a stored submap, merge captures, or navigate
between remembered places from the graph. Those are the next milestones, and
each is listed above rather than assumed done.
