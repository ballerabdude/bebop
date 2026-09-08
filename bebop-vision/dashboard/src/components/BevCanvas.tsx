import { useCallback, useEffect, useRef, useState } from "react";
import {
  brushCells,
  canvasToGrid,
  gridToCanvasCol,
} from "../api/grid";
import type { Grid } from "../api/client";
import { cn } from "./ui";

export interface HoverCell {
  r: number;
  c: number;
  value: number;
}

interface Props {
  grid: Grid | null;
  /** Optional second grid rendered as thin outline where it differs. */
  diffGrid?: Grid | null;
  cellPx?: number;
  editable?: boolean;
  brush?: number;
  cls?: number;
  /** 60x60 RGB PNG data URL — the camera image projected onto the grid.
   * Drawn under the class colors (mirrored to the camera alignment). */
  underlay?: string | null;
  /** Opacity of the class-color fill; 1 = solid (default), lower shows
   * the underlay through. */
  classAlpha?: number;
  onChange?: (g: Grid) => void;
  onDirty?: () => void;
  onHoverCell?: (cell: HoverCell | null) => void;
  className?: string;
}

/** Camera-aligned 60x60 paint canvas. Grid col 0 (robot right) renders at
 * canvas RIGHT; hover shows a brush footprint preview + emits the cell
 * under the cursor. */
export default function BevCanvas({
  grid,
  diffGrid,
  cellPx = 8,
  editable = false,
  brush = 1,
  cls = 0,
  underlay = null,
  classAlpha = 1,
  onChange,
  onDirty,
  onHoverCell,
  className = "",
}: Props) {
  const ref = useRef<HTMLCanvasElement>(null);
  const painting = useRef(false);
  const [hover, setHover] = useState<{ r: number; c: number } | null>(null);
  const [texImg, setTexImg] = useState<HTMLImageElement | null>(null);
  const size = grid ? grid.length * cellPx : 0;

  // decode the underlay texture once per URL
  useEffect(() => {
    if (!underlay) {
      setTexImg(null);
      return;
    }
    const im = new Image();
    im.onload = () => setTexImg(im);
    im.src = underlay;
  }, [underlay]);

  const draw = useCallback(() => {
    const cv = ref.current;
    if (!cv || !grid) return;
    const cx = cv.getContext("2d");
    if (!cx) return;
    const n = grid.length;
    cx.clearRect(0, 0, cv.width, cv.height);
    if (texImg) {
      // mirror horizontally: texture pixel col c belongs at canvas col
      // GRID-1-c, same as the class cells
      cx.save();
      cx.translate(cv.width, 0);
      cx.scale(-1, 1);
      cx.imageSmoothingEnabled = false;
      cx.drawImage(texImg, 0, 0, cv.width, cv.height);
      cx.restore();
    }
    for (let r = 0; r < n; r++) {
      for (let c = 0; c < n; c++) {
        cx.fillStyle = ["#d32f2f", "#2ea043", "#ffa500"][grid[r][c]] ?? "#0a0d10";
        cx.globalAlpha = texImg ? classAlpha : 1;
        cx.fillRect(gridToCanvasCol(c) * cellPx, r * cellPx, cellPx, cellPx);
      }
    }
    cx.globalAlpha = 1;
    if (diffGrid) {
      cx.strokeStyle = texImg && classAlpha < 0.9
        ? "rgba(255,255,255,0.95)"
        : "rgba(255,255,255,0.85)";
      cx.lineWidth = 1;
      for (let r = 0; r < n; r++) {
        for (let c = 0; c < n; c++) {
          if (diffGrid[r][c] !== grid[r][c]) {
            cx.strokeRect(
              gridToCanvasCol(c) * cellPx + 0.5,
              r * cellPx + 0.5,
              cellPx - 1,
              cellPx - 1,
            );
          }
        }
      }
    }
  }, [grid, diffGrid, cellPx, texImg, classAlpha]);

  useEffect(draw, [draw]);

  const cellFromEvent = (e: React.MouseEvent): [number, number] => {
    const rect = ref.current!.getBoundingClientRect();
    return canvasToGrid(
      ((e.clientY - rect.top) / rect.height) * (grid?.length ?? 60),
      ((e.clientX - rect.left) / rect.width) * (grid?.length ?? 60),
    );
  };

  const paint = (e: React.MouseEvent) => {
    if (!editable || !grid || !onChange) return;
    const [r, c] = cellFromEvent(e);
    if (r < 0) return;
    const next = grid.map((row) => row.slice());
    brushCells(next, r, c, brush, cls);
    onChange(next);
    onDirty?.();
  };

  return (
    <div
      className={cn(
        "relative inline-block overflow-hidden rounded-lg ring-1 ring-inset ring-white/[0.08]",
        editable && "cursor-crosshair",
        className,
      )}
      onMouseMove={(e) => {
        if (!grid) return;
        const [r, c] = cellFromEvent(e);
        setHover(r >= 0 ? { r, c } : null);
        onHoverCell?.(r >= 0 ? { r, c, value: grid[r][c] } : null);
      }}
      onMouseLeave={() => {
        setHover(null);
        onHoverCell?.(null);
      }}
      onMouseUp={() => {
        painting.current = false;
      }}
    >
      <canvas
        ref={ref}
        width={size}
        height={size}
        style={{ imageRendering: "pixelated", display: "block" }}
        onMouseDown={(e) => {
          if (!editable) return;
          painting.current = true;
          paint(e);
        }}
        onMouseMove={(e) => {
          if (painting.current) paint(e);
        }}
      />
      {hover && grid && (
        <div
          className="pointer-events-none absolute rounded-[2px] border border-white/90 shadow-[0_0_0_1px_rgba(0,0,0,0.6)]"
          style={{
            left: gridToCanvasCol(hover.c) * cellPx,
            top: hover.r * cellPx,
            width: cellPx,
            height: cellPx,
          }}
        />
      )}
      {hover && grid && editable && brush > 1 && (
        <div
          className="pointer-events-none absolute rounded-[2px] border border-dashed border-white/50"
          style={{
            left: (gridToCanvasCol(hover.c) - Math.floor(brush / 2)) * cellPx,
            top: (hover.r - Math.floor(brush / 2)) * cellPx,
            width: brush * cellPx,
            height: brush * cellPx,
          }}
        />
      )}
    </div>
  );
}
