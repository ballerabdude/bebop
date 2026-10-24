import assert from "node:assert/strict";
import test from "node:test";
import {
  formatAge,
  graphEdgeLabel,
  isPlaceGraph,
  layoutPlaceGraph,
  normalizePlaceGraph,
} from "../src/screens/memory/mapView.ts";

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
