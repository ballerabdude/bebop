/** Grid geometry + class conventions shared with the server
 * (tools/dataset_dashboard.py, bebop_vision/bev.py):
 * 60x60 cells, 3x3 m @ 5 cm; row 0 = far edge (canvas TOP), grid col 0 =
 * robot right (canvas RIGHT, hence the column mirror). */

export const GRID = 60;
export const N_CELLS = GRID * GRID;

import type { Grid } from "./client";

/** blocked / navigable / caution — same hex as the legacy dashboard. */
export const CLS_HEX = ["#d32f2f", "#2ea043", "#ffa500"] as const;
export const CLS_NAMES = ["blocked", "navigable", "caution"] as const;
export const BG_HEX = "#0a0d10";

/** Grid (r, c) -> canvas col. Mirror so robot-right draws at canvas right. */
export const gridToCanvasCol = (c: number) => GRID - 1 - c;

/** Canvas (x, y) in [0,60) -> grid indices [row, col]; inverse of draw(). */
export function canvasToGrid(y: number, x: number): [number, number] {
  const cx = Math.floor(x);
  const cy = Math.floor(y);
  if (cx < 0 || cy < 0 || cx >= GRID || cy >= GRID) return [-1, -1];
  return [cy, gridToCanvasCol(cx)];
}

/** Paint a square brush of size `size` centered at grid (r0, c0). */
export function brushCells(
  grid: Grid,
  r0: number,
  c0: number,
  size: number,
  value: number,
): void {
  const h = Math.floor(size / 2);
  for (let r = r0 - h; r <= r0 + h; r++) {
    for (let c = c0 - h; c <= c0 + h; c++) {
      if (r < 0 || c < 0 || r >= GRID || c >= GRID) continue;
      grid[r][c] = value;
    }
  }
}

export function classCounts(grid: Grid): [number, number, number] {
  const c: [number, number, number] = [0, 0, 0];
  for (const row of grid) for (const v of row) c[v]++;
  return c;
}

export function cloneGrid(g: Grid): Grid {
  return g.map((r) => r.slice());
}

/** Distinct-color scale for probability heatmaps (0..1) — viridis-ish. */
export function heatColor(v: number): string {
  const x = Math.min(1, Math.max(0, v));
  const r = Math.round(255 * Math.min(1, Math.max(0, 1.6 * x - 0.1)));
  const g = Math.round(255 * Math.min(1, Math.max(0, 1.4 - Math.abs(x - 0.5) * 2.6)));
  const b = Math.round(255 * Math.min(1, Math.max(0, 0.9 - 1.8 * x)));
  return `rgb(${r},${g},${b})`;
}
