import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import {
  decodeGridMask,
  decodePlaceLayer,
  decodeRleBase64,
  fitPlanView,
  formatAge,
  formatPayloadBytes,
  graphEdgeLabel,
  gridBounds,
  gridRuns,
  hasPlaceMapContent,
  hasStructure,
  isPlaceGraph,
  isPlaceMapPayload,
  isStructureSnapshot,
  layoutPlaceGraph,
  normalizePlaceGraph,
  normalizePlaceMap,
  normalizeStructure,
  panPlanView,
  placeCaptureGate,
  placeMapAgeSeconds,
  placeMapBounds,
  placeMapGravityLabel,
  placeMapHasGravity,
  placeMapLegend,
  placeMapMissingKind,
  planBounds,
  scaleBarMeters,
  screenHeading,
  screenToWorld,
  worldToScreen,
  zoomPlanView,
} from "../src/screens/memory/mapView.ts";

// Canonical backend format: uint16 run lengths only, alternating from a 0-run.
const u16Lengths = (lengths) => {
  const buf = Buffer.alloc(lengths.length * 2);
  lengths.forEach((n, i) => buf.writeUInt16LE(n, i * 2));
  return buf.toString("base64");
};
// Legacy/defensive format: little-endian uint16 (value,count) pairs.
const u16Runs = (runs, valueFirst = true) => {
  const buf = Buffer.alloc(runs.length * 4);
  runs.forEach(([value, count], i) => {
    buf.writeUInt16LE(valueFirst ? value : count, i * 4);
    buf.writeUInt16LE(valueFirst ? count : value, i * 4 + 2);
  });
  return buf.toString("base64");
};

test("plan world<->screen projection round-trips with pan and zoom", () => {
  const view = { centerX: 2, centerY: 3, scale: 10, width: 200, height: 100 };
  assert.deepEqual(worldToScreen(2, 3, view), [100, 50]);
  assert.deepEqual(worldToScreen(3, 3, view), [110, 50]);
  assert.deepEqual(worldToScreen(2, 4, view), [100, 40]);
  for (const [x, y] of [[0, 0], [2.5, 3.25], [-7, 11]]) {
    const [sx, sy] = worldToScreen(x, y, view);
    const [rx, ry] = screenToWorld(sx, sy, view);
    assert.ok(Math.abs(rx - x) < 1e-9 && Math.abs(ry - y) < 1e-9);
  }
});

test("fit plan view covers walls and path with padding", () => {
  const bounds = planBounds(
    [[0, 0, 4, 0], [0, 0, 0, 3]],
    [[0, 0], [1, 1]],
  );
  assert.deepEqual(bounds, { minX: 0, minY: 0, maxX: 4, maxY: 3 });
  const view = fitPlanView(bounds, 400, 300, 20);
  assert.equal(view.centerX, 2);
  assert.equal(view.centerY, 1.5);
  assert.ok(Math.abs(view.scale - Math.min(360 / 4, 260 / 3)) < 1e-9);
  assert.deepEqual(fitPlanView(null, 400, 300).scale, 40);
});

test("map yaw maps to a screen heading with screen-Y down", () => {
  const near = (a, b) => Math.abs(a - b) < 1e-9;
  const h0 = screenHeading(0);
  const h90 = screenHeading(90);
  const h180 = screenHeading(180);
  assert.ok(near(h0[0], 1) && near(h0[1], 0));
  assert.ok(near(h90[0], 0) && near(h90[1], -1));
  assert.ok(near(h180[0], -1) && near(h180[1], 0));
});

test("RLE base64 decodes the backend's uint16 run-length vector", () => {
  // Real encoder-produced vector from the robot: 4x4, runs
  // 1,2,3,1,1,4,4 alternating from a 0-run. The payload is 14 bytes, i.e. not
  // a multiple of 4, which the old pair-only decoder rejected outright.
  const encoded = "AQACAAMAAQABAAQABAA=";
  const cells = [0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0];
  assert.equal(Buffer.from(encoded, "base64").length, 14);
  assert.deepEqual([...decodeRleBase64(encoded, 16)], cells);
  assert.equal(cells.reduce((a, b) => a + b, 0), 7);
  // The grid layer is accepted only because it matches the reported count.
  const grid = {
    origin: [0, 0], resolution_m: 0.05, width: 4, height: 4,
    free_count: 7, occupied_count: 9, encoding: "rle_base64",
    free_rle: encoded, occupied_rle: "",
  };
  const mask = decodeGridMask(grid, "free");
  assert.ok(mask);
  assert.equal(mask.value, 1);
  assert.deepEqual([...mask.cells], cells);
  assert.equal(decodeGridMask({ ...grid, free_count: 8 }, "free"), null);
});

test("all-free grids and grids without wall lines render cleanly", () => {
  const grid = {
    origin: [0, 0], resolution_m: 0.5, width: 4, height: 3,
    free_count: 12, occupied_count: 0, encoding: "rle_base64",
    free_rle: u16Lengths([0, 12]), occupied_rle: "",
  };
  const free = decodeGridMask(grid, "free");
  assert.ok(free);
  // One merged run per row, no walls or wall cells required.
  assert.deepEqual(gridRuns(free), [
    { iy: 0, x0: 0, x1: 4 },
    { iy: 1, x0: 0, x1: 4 },
    { iy: 2, x0: 0, x1: 4 },
  ]);
  assert.deepEqual(planBounds([], [], grid), {
    minX: 0, minY: 0, maxX: 2, maxY: 1.5,
  });
  assert.equal(decodeGridMask(grid, "occupied"), null);
});

test("RLE base64 still decodes legacy (value,count) pairs", () => {
  // 8 cells: 0 0 1 1 1 0 1 1 -> (value,count) runs.
  const runs = [[0, 2], [1, 3], [0, 1], [1, 2]];
  const cells = [0, 0, 1, 1, 1, 0, 1, 1];
  assert.deepEqual([...decodeRleBase64(u16Runs(runs), 8)], cells);
  // The older (count,value) pair order still decodes.
  assert.deepEqual([...decodeRleBase64(u16Runs(runs, false), 8)], cells);
  // The encoder emits a zero-length leading 0-run when cell 0 is occupied.
  assert.deepEqual(
    [...decodeRleBase64(u16Runs([[0, 0], [1, 8]]), 8)],
    Array(8).fill(1),
  );
  // Counts are uint16, not uint8: a 300-cell run survives.
  const wide = decodeRleBase64(u16Runs([[0, 300], [1, 2]]), 302);
  assert.equal(wide.length, 302);
  assert.equal(wide[299], 0);
  assert.equal(wide[300], 1);
  assert.equal(wide[301], 1);
  // Must start with a 0-run, match the cell count, and be valid base64.
  assert.equal(decodeRleBase64(u16Runs([[1, 2], [0, 6]]), 8), null);
  assert.equal(decodeRleBase64("", 8), null);
  assert.equal(decodeRleBase64(u16Runs(runs), 7), null);
  assert.equal(decodeRleBase64("not-base64!!", 8), null);
});

test("structure fixture validates and normalizes defensive arrays", () => {
  const fixture = {
    ok: true,
    service: "live-slam",
    session_id: "session-1",
    frame_id: "map",
    pose_frame: "map_from_camera_optical",
    structural: true,
    persistent: false,
    loop_corrected: false,
    gravity_source: "estimated_from_floor",
    gravity: { valid: true, up: [0, 0, 1], roll_deg: 1, pitch_deg: -2, inlier_ratio: 0.8 },
    floor: { valid: true, normal: [0, 0, 1], offset_m: 0, camera_height_m: 1.3, inliers: 100 },
    grid: {
      origin: [0, 0], resolution_m: 0.1, width: 4, height: 2,
      occupied_count: 2, free_count: 6, encoding: "rle_base64",
      occupied_rle: u16Runs([[0, 1], [1, 1], [0, 3], [1, 1], [0, 2]]),
      free_rle: u16Runs([[0, 1], [1, 3], [0, 1], [1, 3]]),
    },
    walls: [[0, 0, 1, 0], [0, 0, 0, 1, 99]],
    wall_cells: [[0, 0], [99]],
    path: [[0, 0], [1, 1], [2]],
    pose: { valid: true, x: 0, y: 0, yaw_deg: 45, t_ns: 1, segment: 0 },
    trajectory: {},
    status: { tracking_state: "tracking" },
    revision: 1,
    updated_age_s: 0.1,
  };
  assert.equal(isStructureSnapshot(fixture), true);
  const n = normalizeStructure(fixture);
  assert.equal(n.walls.length, 1);
  assert.equal(n.wall_cells.length, 1);
  assert.equal(n.path.length, 2);
  assert.deepEqual(n.grid.origin, [0, 0]);
  // Real uint16 payloads decode to the reported free/occupied cell counts.
  const free = decodeGridMask(n.grid, "free");
  const occupied = decodeGridMask(n.grid, "occupied");
  assert.ok(free && occupied);
  assert.equal(free.value, 1);
  assert.equal(occupied.value, 1);
  assert.deepEqual([...occupied.cells], [0, 1, 0, 0, 0, 1, 0, 0]);
});

test("pre-floor payloads validate and keep missing estimates null", () => {
  const preFloor = {
    ok: true,
    service: "live-slam",
    session_id: "session-1",
    frame_id: null,
    pose_frame: null,
    structural: true,
    persistent: false,
    loop_corrected: false,
    gravity_source: "none",
    gravity: { valid: false, up: null, roll_deg: null, pitch_deg: null, inlier_ratio: null },
    floor: { valid: false, normal: null, offset_m: null, camera_height_m: null, inliers: null },
    grid: {
      origin: [0, 0], resolution_m: 0.05, width: 0, height: 0,
      occupied_count: 0, free_count: 0, encoding: "rle_base64",
      occupied_rle: "", free_rle: "",
    },
    walls: [],
    wall_cells: [],
    path: [],
    pose: { valid: false, x: 0, y: 0, yaw_deg: 0, t_ns: 0, segment: 0 },
    trajectory: {},
    status: { tracking_state: "starting" },
    revision: 0,
    updated_age_s: null,
  };
  assert.equal(isStructureSnapshot(preFloor), true);
  const n = normalizeStructure(preFloor);
  assert.equal(n.frame_id, null);
  assert.equal(n.pose_frame, null);
  assert.equal(n.gravity.valid, false);
  assert.equal(n.gravity.up, null);
  assert.equal(n.gravity.roll_deg, null);
  assert.equal(n.gravity.pitch_deg, null);
  assert.equal(n.gravity.inlier_ratio, null);
  assert.equal(n.floor.camera_height_m, null);
  assert.equal(n.floor.inliers, null);
  assert.equal(n.loop_corrected, false);
  assert.equal(n.grid.width, 0);
  assert.equal(decodeGridMask(n.grid, "free"), null);
  assert.equal(decodeGridMask(n.grid, "occupied"), null);
  assert.equal(hasStructure(n), false);
});

test("grid bounds extend the plan fit beyond walls and path", () => {
  const grid = {
    origin: [-1, -2], resolution_m: 0.5, width: 8, height: 4,
    occupied_count: 0, free_count: 0, encoding: "rle_base64",
    occupied_rle: "", free_rle: "",
  };
  assert.deepEqual(gridBounds(grid), { minX: -1, minY: -2, maxX: 3, maxY: 0 });
  assert.equal(gridBounds({ ...grid, width: 0 }), null);
  // Walls/path alone stay authoritative when no grid exists.
  assert.deepEqual(planBounds([[0, 0, 1, 0]], [], null), {
    minX: 0, minY: 0, maxX: 1, maxY: 0,
  });
  // The grid footprint is included so Fit shows the whole surveyed area.
  assert.deepEqual(planBounds([[0, 0, 1, 0]], [], grid), {
    minX: -1, minY: -2, maxX: 3, maxY: 0,
  });
  const view = fitPlanView(planBounds([], [], grid), 400, 300, 20);
  assert.equal(view.centerX, 1);
  assert.equal(view.centerY, -1);
});

test("grid layers decode and merge into row runs", () => {
  // 4x2 free grid: row0 = 0 1 1 1, row1 = 1 1 0 0.
  const freeCells = [0, 1, 1, 1, 1, 1, 0, 0];
  const encoded = u16Runs([[0, 1], [1, 5], [0, 2]]);
  const grid = {
    origin: [10, 20], resolution_m: 0.1, width: 4, height: 2,
    occupied_count: 0, free_count: 5, encoding: "rle_base64",
    occupied_rle: "", free_rle: encoded,
  };
  const mask = decodeGridMask(grid, "free");
  assert.ok(mask);
  assert.equal(mask.value, 1);
  assert.deepEqual([...mask.cells], freeCells);
  assert.deepEqual(gridRuns(mask), [
    { iy: 0, x0: 1, x1: 4 },
    { iy: 1, x0: 0, x1: 2 },
  ]);
  // A budget stops mid-grid rather than building every run.
  assert.equal(gridRuns(mask, 1).length, 1);
  assert.equal(gridRuns(mask, 0).length, 0);
  // A count that matches neither the ones nor the zeros is rejected, not guessed.
  assert.equal(decodeGridMask({ ...grid, free_count: 4 }, "free"), null);
  assert.equal(decodeGridMask({ ...grid, encoding: "raw" }, "free"), null);
  assert.equal(decodeGridMask(grid, "occupied"), null);
});

test("grid layer accepts runs where the marker value is 0", () => {
  // 6 of 8 cells are 0 and free_count says 6, so the layer marks free cells
  // with 0 rather than 1.
  const encoded = u16Runs([[0, 6], [1, 2]]);
  const grid = {
    origin: [0, 0], resolution_m: 0.1, width: 4, height: 2,
    occupied_count: 2, free_count: 6, encoding: "rle_base64",
    occupied_rle: "", free_rle: encoded,
  };
  const mask = decodeGridMask(grid, "free");
  assert.ok(mask);
  assert.equal(mask.value, 0);
  assert.deepEqual(gridRuns(mask), [
    { iy: 0, x0: 0, x1: 4 },
    { iy: 1, x0: 0, x1: 2 },
  ]);
});

test("zoom keeps the world point under the anchor fixed", () => {
  const view = { centerX: 2, centerY: -1, scale: 20, width: 400, height: 300 };
  for (const [ax, ay] of [[0, 0], [200, 150], [377, 42]]) {
    const before = screenToWorld(ax, ay, view);
    const zoomed = zoomPlanView(view, 1.7, ax, ay);
    const after = worldToScreen(before[0], before[1], zoomed);
    assert.ok(Math.abs(after[0] - ax) < 1e-9);
    assert.ok(Math.abs(after[1] - ay) < 1e-9);
    assert.equal(zoomed.width, view.width);
    assert.equal(zoomed.height, view.height);
  }
  assert.equal(zoomPlanView(view, 1e9, 0, 0).scale, 500);
  assert.equal(zoomPlanView(view, 1e-9, 0, 0).scale, 0.5);
  assert.equal(zoomPlanView(view, 0, 0, 0), view);
  assert.equal(zoomPlanView(view, NaN, 0, 0), view);
});

test("pan moves the map with the drag and leaves scale untouched", () => {
  const view = { centerX: 0, centerY: 0, scale: 50, width: 400, height: 300 };
  const panned = panPlanView(view, 25, -10);
  assert.equal(panned.centerX, -0.5);
  assert.equal(panned.centerY, -0.2);
  assert.equal(panned.scale, 50);
  assert.equal(panPlanView(view, NaN, 0), view);
  // Dragging right must move drawn content right on screen.
  const [sx0] = worldToScreen(0, 0, view);
  const [sx1] = worldToScreen(0, 0, panned);
  assert.ok(sx1 > sx0);
});

test("scale bar picks the largest nice length that fits", () => {
  assert.equal(scaleBarMeters(50, 200), 2);
  assert.equal(scaleBarMeters(1, 200), 100);
  assert.equal(scaleBarMeters(0.5, 30), 50);
  assert.equal(scaleBarMeters(0, 200), 0.05);
  assert.equal(scaleBarMeters(NaN, 200), 0.05);
});

// --- per-place local map -------------------------------------------------

const placeMap = (overrides = {}) => ({
  place_id: "kitchen",
  anchor_x: 0.5,
  anchor_y: -0.25,
  anchor_yaw_deg: 90,
  origin_x: -1,
  origin_y: -1,
  resolution_m: 0.5,
  width: 3,
  height: 2,
  occupied_rle: u16Lengths([2, 1, 1, 2]),
  free_rle: u16Lengths([0, 2, 1, 1, 2]),
  walls: [[-1, -1, 0.5, -1]],
  gravity: { up: [0, 0, 1], roll_deg: 0.4, pitch_deg: -0.2 },
  camera_height_m: 1.31,
  revision: 7,
  updated_at: 1_700_000_000,
  ...overrides,
});

// Exact live response captured from
// GET http://bebop.local:9095/places/office-room-330f8a/map (2026-10-17).
// The service wraps the map in an envelope and nests anchor/grid.
const nestedPlaceMapFixture = JSON.parse(
  readFileSync(
    new URL("./fixtures/place-map-nested.json", import.meta.url),
    "utf8",
  ),
);

test("nested {map:{anchor,grid}} payload is accepted and draws the real capture", () => {
  assert.equal(isPlaceMapPayload(nestedPlaceMapFixture), true);
  const m = normalizePlaceMap(nestedPlaceMapFixture);
  assert.equal(m.place_id, "office-room-330f8a");
  assert.equal(m.revision, 1);
  assert.deepEqual(m.anchor, { x: 0.0025, y: -0.0018, yaw_deg: -0.137 });
  assert.deepEqual(m.origin, [0, -2.3]);
  assert.equal(m.resolution_m, 0.05);
  assert.equal(m.width, 80);
  assert.equal(m.height, 68);
  assert.equal(m.encoding, "rle_base64");
  // Counts come from the nested grid, not recomputed.
  assert.equal(m.free_count, 2195);
  assert.equal(m.occupied_count, 83);
  assert.equal(m.walls.length, 2);
  assert.deepEqual(m.walls[0], [1.575, 0.725, 2.375, 0.725]);
  assert.equal(m.gravity.valid, true);
  assert.deepEqual(m.gravity.up, [-0.008106, -0.983419, -0.181168]);
  assert.equal(m.gravity.roll_deg, -0.472);
  assert.equal(m.gravity.pitch_deg, 10.438);
  assert.equal(m.camera_height_m, 1.3218);
  assert.equal(hasPlaceMapContent(m), true);

  // Both stored layers decode against the reported counts.
  const free = decodePlaceLayer(m, "free");
  const occupied = decodePlaceLayer(m, "occupied");
  assert.ok(free && occupied);
  assert.equal(free.width, 80);
  assert.equal(free.height, 68);
  const count = (mask) => {
    let n = 0;
    for (const cell of mask.cells) if (cell === mask.value) n += 1;
    return n;
  };
  assert.equal(count(free), 2195);
  assert.equal(count(occupied), 83);
  assert.equal(free.cells.length, 80 * 68);

  // Grid footprint dominates Fit: 80*0.05 x 68*0.05 from origin [0,-2.3].
  const bounds = placeMapBounds(m);
  assert.equal(bounds.minX, 0);
  assert.equal(bounds.minY, -2.3);
  assert.equal(bounds.maxX, 4);
  assert.ok(Math.abs(bounds.maxY - 1.1) < 1e-12);
  assert.equal(placeMapLegend("office room", m),
    "local map for office room · gravity estimated from floor · not loop-corrected");
});

test("place-map payload validates and normalizes the anchor/grid/origin", () => {
  const raw = placeMap();
  assert.equal(isPlaceMapPayload(raw), true);
  const m = normalizePlaceMap(raw);
  assert.equal(m.place_id, "kitchen");
  assert.deepEqual(m.anchor, { x: 0.5, y: -0.25, yaw_deg: 90 });
  assert.deepEqual(m.origin, [-1, -1]);
  assert.deepEqual(m.gravity, {
    valid: null,
    up: [0, 0, 1],
    roll_deg: 0.4,
    pitch_deg: -0.2,
  });
  assert.equal(m.revision, 7);
  // A payload without an anchor must not invent one at 0,0.
  const noAnchor = normalizePlaceMap(
    placeMap({ anchor_x: null, anchor_y: null, anchor_yaw_deg: null }),
  );
  assert.equal(noAnchor.anchor, null);
  // Contract fields are required for a usable map payload.
  assert.equal(isPlaceMapPayload({ ...raw, width: undefined }), false);
  assert.equal(isPlaceMapPayload({ ...raw, place_id: 3 }), false);
  assert.equal(isPlaceMapPayload(null), false);
});

test("place layers decode from stored RLE without reported cell counts", () => {
  const m = normalizePlaceMap(placeMap());
  const free = decodePlaceLayer(m, "free");
  const occupied = decodePlaceLayer(m, "occupied");
  assert.ok(free && occupied);
  assert.equal(free.value, 1);
  assert.deepEqual([...free.cells], [1, 1, 0, 1, 0, 0]);
  assert.deepEqual([...occupied.cells], [0, 0, 1, 0, 1, 1]);
  assert.deepEqual(gridRuns(free), [
    { iy: 0, x0: 0, x1: 2 },
    { iy: 1, x0: 0, x1: 1 },
  ]);
  // Missing/short layers are rejected, never filled in.
  assert.ok(decodePlaceLayer(m, "free"));
  assert.equal(decodePlaceLayer(normalizePlaceMap(placeMap({ free_rle: "" })), "free"), null);
  assert.equal(
    decodePlaceLayer(normalizePlaceMap(placeMap({ free_rle: u16Lengths([0, 2]) })), "free"),
    null,
  );
});

test("place layers still honour a reported count when a backend sends one", () => {
  const m = normalizePlaceMap(
    placeMap({ free_count: 3, occupied_count: 3 }),
  );
  assert.ok(decodePlaceLayer(m, "free"));
  assert.ok(decodePlaceLayer(m, "occupied"));
  // A count that contradicts the stored RLE rejects the layer instead of
  // drawing a guessed interpretation.
  const bad = normalizePlaceMap(placeMap({ free_count: 4 }));
  assert.equal(decodePlaceLayer(bad, "free"), null);
});

test("a place map needs walls or an occupancy grid, not just an anchor", () => {
  assert.equal(hasPlaceMapContent(null), false);
  assert.equal(hasPlaceMapContent(normalizePlaceMap(placeMap())), true);
  assert.equal(
    hasPlaceMapContent(normalizePlaceMap(placeMap({ walls: [], free_rle: "", occupied_rle: "" }))),
    false,
  );
  // Walls alone (few or no grid layers) are still a drawable local map.
  assert.equal(
    hasPlaceMapContent(normalizePlaceMap(placeMap({ free_rle: "", occupied_rle: "" }))),
    true,
  );
});

test("place-map bounds include the grid, walls, and anchor", () => {
  const m = normalizePlaceMap(placeMap());
  // Grid: origin (-1,-1), 3x2 @ 0.5 -> max (0.5, 0); wall reaches x=0.5.
  assert.deepEqual(placeMapBounds(m), {
    minX: -1, minY: -1, maxX: 0.5, maxY: 0,
  });
  const farAnchor = normalizePlaceMap(placeMap({ anchor_x: 4, anchor_y: 3 }));
  assert.deepEqual(placeMapBounds(farAnchor), {
    minX: -1, minY: -1, maxX: 4, maxY: 3,
  });
  assert.equal(placeMapBounds(null), null);
  const view = fitPlanView(placeMapBounds(farAnchor), 400, 300, 20);
  assert.ok(view.scale > 0);
});

test("capture is only allowed at the selected, currently recognized place", () => {
  const where = (id, name) => ({ matched: true, place: { id, name } });
  // Selected == recognized: allowed.
  assert.deepEqual(placeCaptureGate("office", "office room", where("office", "office room")), {
    allowed: true,
    reason: null,
  });
  // Recognized elsewhere: disabled, and the reason names the real place.
  const elsewhere = placeCaptureGate(
    "office",
    "office room",
    where("kitchen", "kitchen"),
  );
  assert.equal(elsewhere.allowed, false);
  assert.match(elsewhere.reason, /recognized as "kitchen", not "office room"/);
  assert.match(elsewhere.reason, /Select "kitchen"/);
  // Unknown / no recognition / no selection.
  assert.match(
    placeCaptureGate("office", "office room", { matched: false }).reason,
    /not recognized at any taught place/,
  );
  assert.match(
    placeCaptureGate("office", "office room", null).reason,
    /not recognized/,
  );
  assert.equal(
    placeCaptureGate(null, "this place", where("office", "office room")).reason,
    "Select a place first.",
  );
  // Paused wins over recognition.
  assert.equal(
    placeCaptureGate("office", "office room", where("office", "office room"), true)
      .reason,
    "Resume polling before capturing a map.",
  );
  // A place with no name still identifies by id.
  assert.equal(
    placeCaptureGate("office", "office room", { matched: true, place: { id: "office" } })
      .allowed,
    true,
  );
});

test("payload sizes are shown in B/KB, never rounded away to 0.0 MB", () => {
  assert.equal(formatPayloadBytes(null), "—");
  assert.equal(formatPayloadBytes(0), "—");
  assert.equal(formatPayloadBytes(NaN), "—");
  assert.equal(formatPayloadBytes(512), "512 B");
  assert.equal(formatPayloadBytes(1852), "1.9 KB");
  assert.equal(formatPayloadBytes(999_999), "1000 KB");
  assert.equal(formatPayloadBytes(1_500_000), "1.50 MB");
});

test("stored map age accepts seconds or milliseconds and clamps skew", () => {
  const nowMs = 1_700_000_060_000;
  assert.equal(placeMapAgeSeconds(1_700_000_000, nowMs), 60);
  assert.equal(placeMapAgeSeconds(1_700_000_000_000, nowMs), 60);
  assert.equal(placeMapAgeSeconds(1_700_000_100, nowMs), 0);
  assert.equal(placeMapAgeSeconds(null, nowMs), null);
  assert.equal(placeMapAgeSeconds(NaN, nowMs), null);
  assert.equal(placeMapAgeSeconds(1, NaN), null);
});

test("place-map legend is explicit about gravity and loop correction", () => {
  const m = normalizePlaceMap(placeMap());
  assert.equal(
    placeMapLegend("kitchen", m),
    "local map for kitchen · gravity estimated from floor · not loop-corrected",
  );
  assert.equal(
    placeMapLegend("  ", normalizePlaceMap(placeMap({ gravity: {} }))),
    "local map for unknown place · gravity not estimated yet · not loop-corrected",
  );
  // No stored map at all is not a gravity claim.
  assert.equal(
    placeMapLegend("kitchen", null),
    "local map for kitchen · no map captured yet",
  );
});

test("gravity copy distinguishes no map from an unestimated stored map", () => {
  assert.equal(placeMapGravityLabel(null), "no map captured yet");
  const stored = normalizePlaceMap(placeMap());
  assert.equal(placeMapHasGravity(stored), true);
  assert.equal(placeMapGravityLabel(stored), "0.00, 0.00, 1.00");
  // Inferred from payload fields when `gravity.valid` is absent.
  assert.equal(
    placeMapGravityLabel(normalizePlaceMap(placeMap({ gravity: { up: null } }))),
    "gravity not estimated",
  );
  assert.equal(
    placeMapHasGravity(normalizePlaceMap(placeMap({ gravity: { up: null } }))),
    false,
  );
  // An explicit `gravity.valid` is authoritative in both directions.
  const invalid = normalizePlaceMap(
    placeMap({ gravity: { valid: false, up: [0, 0, 1], roll_deg: 1, pitch_deg: 2 } }),
  );
  assert.equal(placeMapHasGravity(invalid), false);
  assert.equal(placeMapGravityLabel(invalid), "gravity not estimated");
  const valid = normalizePlaceMap(placeMap({ gravity: { valid: true, up: null } }));
  assert.equal(placeMapHasGravity(valid), true);
  assert.equal(placeMapGravityLabel(valid), "estimated");
});

test("404 classification distinguishes a missing endpoint from a missing map", () => {
  assert.equal(placeMapMissingKind(404, 404), "endpoint-missing");
  assert.equal(placeMapMissingKind(404, 200), "no-map");
  assert.equal(placeMapMissingKind(404, null), "no-map");
  assert.equal(placeMapMissingKind(200, 404), "no-map");
});

// --- place graph ---------------------------------------------------------

const graphFixture = () => ({
  nodes: [
    { id: "kitchen", name: "Kitchen" },
    { id: "hall", name: "Hall" },
    { id: "lab", name: "Lab" },
  ],
  edges: [
    { from: "kitchen", to: "hall", traversals: 4, successes: 3, median_duration_s: 12.34, last_success_at: 1 },
    { from: "hall", to: "lab", traversals: 2, successes: 2, median_duration_s: null, last_success_at: null },
  ],
});

test("place graph validates and drops dangling/duplicate nodes", () => {
  assert.equal(isPlaceGraph(graphFixture()), true);
  assert.equal(isPlaceGraph({ nodes: {} }), false);
  assert.equal(isPlaceGraph(null), false);
  const g = normalizePlaceGraph({
    nodes: [
      { id: "a", name: "A" },
      { id: "a", name: "A again" },
      { id: "", name: "empty" },
      { id: "b" },
      null,
    ],
    edges: [
      { from: "a", to: "b", traversals: 3.9, successes: 9 },
      { from: "a", to: "missing", traversals: 1, successes: 1 },
      { from: 1, to: "b" },
    ],
  });
  assert.deepEqual(g.nodes, [{ id: "a", name: "A" }, { id: "b", name: "b" }]);
  assert.equal(g.edges.length, 1);
  assert.equal(g.edges[0].traversals, 3);
  // Successes cannot exceed traversals.
  assert.equal(g.edges[0].successes, 3);
  assert.equal(g.edges[0].median_duration_s, null);
  assert.deepEqual(normalizePlaceGraph(null), { nodes: [], edges: [] });
});

test("graph layout is deterministic, centered, and capped", () => {
  const g = normalizePlaceGraph(graphFixture());
  const layout = layoutPlaceGraph(g, 400, 300, 40);
  assert.equal(layout.nodes.length, 3);
  // First node starts at the top of the circle; all three are equidistant
  // from the centre.
  const cx = 200;
  const cy = 150;
  const radius = 300 / 2 - 40;
  assert.ok(Math.abs(layout.nodes[0].x - cx) < 1e-9);
  assert.ok(Math.abs(layout.nodes[0].y - (cy - radius)) < 1e-9);
  for (const n of layout.nodes) {
    assert.ok(Math.abs(Math.hypot(n.x - cx, n.y - cy) - radius) < 1e-9);
  }
  assert.equal(layout.edges.length, 2);
  const edge = layout.edges[0];
  assert.equal(edge.from, "kitchen");
  assert.equal(edge.to, "hall");
  assert.ok(Number.isFinite(edge.labelX) && Number.isFinite(edge.labelY));
  // Deterministic between calls.
  assert.deepEqual(layoutPlaceGraph(g, 400, 300, 40), layout);
  // A single node sits at the centre; an empty graph draws nothing.
  const single = layoutPlaceGraph(
    normalizePlaceGraph({ nodes: [{ id: "a", name: "A" }], edges: [] }),
    400,
    300,
  );
  assert.deepEqual(
    { x: single.nodes[0].x, y: single.nodes[0].y },
    { x: 200, y: 150 },
  );
  assert.equal(layoutPlaceGraph(normalizePlaceGraph(null), 400, 300).nodes.length, 0);
  // Render caps report what was omitted.
  const capped = layoutPlaceGraph(g, 400, 300, 40, 2, 1);
  assert.equal(capped.nodes.length, 2);
  assert.equal(capped.hiddenNodes, 1);
  assert.equal(capped.edges.length, 1);
  // Self edges are not drawable routes and are dropped.
  const self = layoutPlaceGraph(
    normalizePlaceGraph({
      nodes: [{ id: "a", name: "A" }],
      edges: [{ from: "a", to: "a", traversals: 1, successes: 1 }],
    }),
    400,
    300,
  );
  assert.equal(self.edges.length, 0);
});

test("graph edge labels carry recorded counts, not distances", () => {
  const g = normalizePlaceGraph(graphFixture());
  assert.equal(graphEdgeLabel(g.edges[0]), "4× · 3 ok · 12.3 s");
  assert.equal(graphEdgeLabel(g.edges[1]), "2× · 2 ok");
});

test("hasStructure is false until the plan carries drawable data", () => {
  const base = {
    walls: [], wall_cells: [], path: [],
    grid: { width: 0, height: 0, free_count: 0, occupied_count: 0 },
  };
  assert.equal(hasStructure(null), false);
  assert.equal(hasStructure(base), false);
  assert.equal(hasStructure({ ...base, walls: [[0, 0, 1, 0]] }), true);
  assert.equal(hasStructure({ ...base, path: [[0, 0]] }), true);
  assert.equal(hasStructure({ ...base, wall_cells: [[2, 3]] }), true);
  assert.equal(
    hasStructure({ ...base, grid: { width: 4, height: 4, free_count: 3, occupied_count: 0 } }),
    true,
  );
});

test("formatAge scales units from seconds to days", () => {
  assert.equal(formatAge(null), "—");
  assert.equal(formatAge(0.25), "0.25 s");
  assert.equal(formatAge(12.34), "12.3 s");
  assert.equal(formatAge(125), "2m 5s");
  assert.equal(formatAge(3599.9), "59m 59s");
  assert.equal(formatAge(3600), "1h 0m");
  assert.equal(formatAge(8821 * 60 + 44), "6d 3h");
  assert.equal(formatAge(86_399), "23h 59m");
});
