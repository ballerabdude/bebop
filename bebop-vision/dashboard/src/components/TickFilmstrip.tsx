import { useEffect, useRef } from "react";
import type { TickSummary } from "../api/client";

/** Horizontal strip of every tick in the session. Each tick renders its
 * class balance as a tiny stacked bar; markers flag hand-reviewed and
 * disagree ticks so a reviewer can scan a whole session at a glance. */
export default function TickFilmstrip({
  ticks,
  idx,
  onSelect,
}: {
  ticks: TickSummary[];
  idx: number;
  onSelect: (i: number) => void;
}) {
  const boxRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = boxRef.current?.querySelector(`[data-i="${idx}"]`);
    el?.scrollIntoView({ block: "nearest", inline: "nearest" });
  }, [idx]);

  if (!ticks.length) return null;
  return (
    <div
      ref={boxRef}
      className="flex gap-[3px] overflow-x-auto rounded-xl bg-zinc-900 px-2.5 py-2.5 ring-1 ring-inset ring-white/[0.06]"
    >
      {ticks.map((t, i) => {
        const tot = t.f0 + t.f1 + t.f2 || 1;
        return (
          <button
            key={t.stamp}
            data-i={i}
            title={`${t.stamp}\nblocked ${Math.round(100 * t.f0)}% · navigable ${Math.round(100 * t.f1)}% · caution ${Math.round(100 * t.f2)}%${t.hand ? "\nhand-reviewed" : ""}${t.disagree_cells ? `\n${t.disagree_cells} disagree cells` : ""}`}
            onClick={() => onSelect(i)}
            className={`group relative flex h-11 w-[7px] shrink-0 flex-col justify-end overflow-hidden rounded-[3px] ring-offset-2 ring-offset-zinc-900 transition-all ${
              i === idx
                ? "ring-2 ring-indigo-400"
                : "opacity-70 hover:opacity-100"
            }`}
          >
            <div style={{ height: `${(100 * t.f0) / tot}%`, background: "#d32f2f" }} />
            <div style={{ height: `${(100 * t.f1) / tot}%`, background: "#2ea043" }} />
            <div style={{ height: `${(100 * t.f2) / tot}%`, background: "#ffa500" }} />
            {t.hand && (
              <span className="absolute inset-x-0 top-0 h-[2px] bg-emerald-400" />
            )}
            {t.disagree_cells > 0 && (
              <span className="absolute inset-x-0 top-[3px] h-[2px] bg-red-400/80" />
            )}
          </button>
        );
      })}
    </div>
  );
}
