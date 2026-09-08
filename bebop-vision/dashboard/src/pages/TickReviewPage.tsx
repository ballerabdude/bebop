import { useCallback, useEffect, useRef, useState } from "react";
import {
  api,
  type Grid,
  type OverlayPayload,
  type TickPayload,
  type TickSummary,
} from "../api/client";
import { classCounts, cloneGrid } from "../api/grid";
import BevCanvas from "../components/BevCanvas";
import TickFilmstrip from "../components/TickFilmstrip";
import {
  Badge,
  Button,
  Card,
  ClassLegend,
  Img,
  Segmented,
  Spinner,
  toast,
} from "../components/ui";

type Src = "auto" | "hand" | "fused";

const MINING_KEYS = [
  "disagree",
  "unconfirmed",
  "sem_near",
  "sem_far",
  "floor_near",
  "floor_far",
  "teacher",
] as const;

const CLS_HEX = ["#d32f2f", "#2ea043", "#ffa500"];
const CLS_NAMES = ["blocked", "navigable", "caution"];

interface Props {
  session: string | null;
  onHandChange: () => void;
  onStampViewed: (stamp: string) => void;
}

/** Tick review workspace: filmstrip navigator + camera views + hand-paint
 * BEV canvas. Edits are client-side until Save; the server stores `hand`
 * and never touches teacher keys. */
export default function TickReviewPage({ session, onHandChange, onStampViewed }: Props) {
  const [ticks, setTicks] = useState<TickSummary[]>([]);
  const [idx, setIdx] = useState(0);
  const [payload, setPayload] = useState<TickPayload | null>(null);
  const [loading, setLoading] = useState(false);
  const [grid, setGrid] = useState<Grid | null>(null);
  const [dirty, setDirty] = useState(false);
  const [undoStack, setUndoStack] = useState<Grid[]>([]);
  const [brush, setBrush] = useState(1);
  const [cls, setCls] = useState(0);
  const [src, setSrc] = useState<Src>("auto");
  const [miningKey, setMiningKey] = useState<string | null>(null);
  const [overlayOn, setOverlayOn] = useState(false);
  const [alpha, setAlpha] = useState(0.5);
  const [overlay, setOverlay] = useState<OverlayPayload | null>(null);
  const [hover, setHover] = useState<{ r: number; c: number; value: number } | null>(null);
  const [tex, setTex] = useState<string | null>(null);
  const [texOn, setTexOn] = useState(true);
  const [paintAlpha, setPaintAlpha] = useState(0.55);
  const reqSeq = useRef(0);

  const stamp = ticks[idx]?.stamp ?? null;
  const g = payload?.grids ?? {};
  const editable = !!payload && !payload.legacy && !!(g.fused || g.hand);
  const hasHand = !!g.hand;

  // session -> ticks
  useEffect(() => {
    if (!session) return;
    const my = ++reqSeq.current;
    setLoading(true);
    setPayload(null);
    setGrid(null);
    setIdx(0);
    setDirty(false);
    setUndoStack([]);
    setSrc("auto");
    api
      .ticks(session)
      .then((t) => {
        if (my !== reqSeq.current) return;
        setTicks(Array.isArray(t) ? t : []);
        setLoading(false);
      })
      .catch((e) => {
        if (my === reqSeq.current) {
          setLoading(false);
          toast(`session load failed: ${e}`, "error");
        }
      });
  }, [session]);

  // stamp -> tick payload
  useEffect(() => {
    if (!session || !stamp) return;
    const my = ++reqSeq.current;
    setLoading(true);
    setDirty(false);
    setUndoStack([]);
    setOverlay(null);
    setOverlayOn(false);
    setTex(null);
    api
      .tick(session, stamp)
      .then((p) => {
        if (my !== reqSeq.current) return;
        setPayload(p);
        setGrid(cloneGrid(p.grids.hand ?? p.grids.fused ?? p.grids.teacher ?? []));
        setLoading(false);
        onStampViewed(p.stamp);
      })
      .catch(() => {
        if (my === reqSeq.current) setLoading(false);
      });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [session, stamp]);

  // label overlay (near camera) + projected-image underlay for the grid
  useEffect(() => {
    if (!overlayOn || !session || !stamp) return;
    api
      .overlay(session, stamp, { role: "near", src: src === "hand" ? "hand" : "auto", alpha })
      .then(setOverlay)
      .catch(() => setOverlay(null));
  }, [overlayOn, session, stamp, src, alpha]);

  useEffect(() => {
    if (!session || !stamp || !texOn) return;
    const my = reqSeq.current;
    api
      .gridTex(session, stamp)
      .then((t) => {
        if (t.png) setTex(`data:image/png;base64,${t.png}`);
      })
      .catch(() => void 0);
    void my;
  }, [session, stamp, texOn]);

  const nav = useCallback(
    (d: number) => {
      if (!ticks.length) return;
      const next = Math.min(Math.max(idx + d, 0), ticks.length - 1);
      if (next === idx) return;
      if (dirty && !window.confirm("Unsaved hand edits — discard?")) return;
      setIdx(next);
    },
    [ticks, idx, dirty],
  );

  const jump = useCallback(
    (kind: "disagree" | "caution" | "unreviewed" | "first") => {
      if (!ticks.length) return;
      if (kind === "first") return setIdx(0);
      for (let k = 1; k <= ticks.length; k++) {
        const i = (idx + k) % ticks.length;
        const t = ticks[i];
        const hit =
          (kind === "disagree" && t.disagree_cells > 0) ||
          (kind === "caution" && t.f2 > 0.35 && !t.hand) ||
          (kind === "unreviewed" && !t.hand);
        if (hit) return setIdx(i);
      }
      toast(`no tick matches jump: ${kind}`);
    },
    [ticks, idx],
  );

  const reloadTick = useCallback(async () => {
    if (!session || !stamp) return;
    const p = await api.tick(session, stamp);
    setPayload(p);
    setGrid(cloneGrid(p.grids.hand ?? p.grids.fused ?? p.grids.teacher ?? []));
  }, [session, stamp]);

  const save = useCallback(async () => {
    if (!session || !stamp || !grid || !editable) return;
    try {
      await api.saveHand(session, stamp, grid);
      setDirty(false);
      setTicks((ts) => ts.map((t, i) => (i === idx ? { ...t, hand: true } : t)));
      onHandChange();
      toast("hand grid saved — teacher keys untouched", "success");
    } catch (e) {
      toast(`save failed: ${e}`, "error");
    }
  }, [session, stamp, grid, editable, idx, onHandChange]);

  const clearHand = useCallback(async () => {
    if (!session || !stamp) return;
    try {
      await api.clearHand(session, stamp);
      setDirty(false);
      setTicks((ts) => ts.map((t, i) => (i === idx ? { ...t, hand: false } : t)));
      onHandChange();
      const p = await api.tick(session, stamp);
      setPayload(p);
      setGrid(cloneGrid(p.grids.fused ?? p.grids.teacher ?? []));
      toast("hand grid cleared — back to teacher", "success");
    } catch (e) {
      toast(`clear failed: ${e}`, "error");
    }
  }, [session, stamp, idx, onHandChange]);

  const undo = useCallback(() => {
    if (!undoStack.length) return;
    if (grid) setUndoStack((s) => [...s.slice(0, -1)]);
    setGrid(cloneGrid(undoStack[undoStack.length - 1]));
    setDirty(true);
  }, [undoStack]);

  // keyboard (legacy-compatible)
  useEffect(() => {
    const h = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement | null)?.tagName;
      if (tag && ["SELECT", "INPUT", "TEXTAREA"].includes(tag)) return;
      switch (e.key) {
        case "ArrowRight":
          e.preventDefault();
          nav(1);
          break;
        case "ArrowLeft":
          e.preventDefault();
          nav(-1);
          break;
        case "1":
          setCls(0);
          break;
        case "2":
          setCls(1);
          break;
        case "3":
          setCls(2);
          break;
        case "[":
          setBrush((b) => Math.max(1, b - 1));
          break;
        case "]":
          setBrush((b) => Math.min(5, b + 1));
          break;
        case "u":
          undo();
          break;
        case "s":
          void save();
          break;
        case "r":
          if (!dirty || window.confirm("Revert unsaved hand edits?")) {
            setDirty(false);
            void reloadTick();
          }
          break;
        case "t":
          setSrc((v) => (v === "auto" ? "hand" : v === "hand" ? "fused" : "auto"));
          break;
        case "o":
          setOverlayOn((v) => !v);
          break;
        case "d":
          jump("disagree");
          break;
        case "c":
          jump("caution");
          break;
        case "n":
          jump("unreviewed");
          break;
        case "f":
          jump("first");
          break;
      }
    };
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, [nav, jump, save, undo, dirty, reloadTick]);

  const fusedCounts = g.fused ? classCounts(g.fused) : null;
  const handCounts = g.hand ? classCounts(g.hand) : null;
  const cellCount = (k: string) =>
    g[k] ? (g[k] as Grid).flat().filter((v) => v > 0).length : 0;
  const m = payload?.manifest;

  return (
    <div className="flex h-full flex-col">
      {/* toolbar */}
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2 border-b border-white/[0.06] bg-zinc-900/60 px-4 py-2.5">
        <div className="flex items-center gap-1">
          <Button onClick={() => nav(-1)} disabled={!ticks.length} title="previous (←)">
            ←
          </Button>
          <span className="min-w-16 text-center font-mono text-xs text-zinc-400">
            {ticks.length ? `${idx + 1}/${ticks.length}` : "—"}
          </span>
          <Button onClick={() => nav(1)} disabled={!ticks.length} title="next (→)">
            →
          </Button>
        </div>
        <div className="flex gap-1">
          <Button onClick={() => jump("disagree")} disabled={!ticks.length} title="next disagree tick (d)">
            disagree
          </Button>
          <Button onClick={() => jump("caution")} disabled={!ticks.length} title="next high-caution unreviewed (c)">
            caution
          </Button>
          <Button onClick={() => jump("unreviewed")} disabled={!ticks.length} title="next unreviewed (n)">
            unreviewed
          </Button>
        </div>
        <div className="ml-auto flex flex-wrap items-center gap-x-4 gap-y-2">
          <span className="flex items-center gap-1.5 text-[11px] text-zinc-500">
            grid
            <Segmented
              size="sm"
              value={src}
              onChange={setSrc}
              options={[
                { value: "auto", label: "auto", title: "hand if present, else fused" },
                { value: "hand", label: "hand", title: "human corrections" },
                { value: "fused", label: "fused", title: "SAM+depth teacher" },
              ]}
            />
          </span>
          <select
            value={miningKey ?? ""}
            onChange={(e) => setMiningKey(e.target.value || null)}
            className="rounded-lg border border-white/10 bg-zinc-800 px-2 py-1.5 text-[11px] text-zinc-300"
          >
            <option value="">mining: none</option>
            {MINING_KEYS.map((k) => (
              <option key={k} value={k}>
                mining: {k}
              </option>
            ))}
          </select>
          <span className="flex items-center gap-2 text-[11px] text-zinc-500">
            overlay
            <Button active={overlayOn} onClick={() => setOverlayOn((v) => !v)} title="toggle (o)">
              {overlayOn ? "on" : "off"}
            </Button>
            <input
              type="range"
              min={0}
              max={100}
              value={alpha * 100}
              onChange={(e) => setAlpha(Number(e.target.value) / 100)}
              className="w-20 accent-indigo-500"
            />
          </span>
          {payload?.legacy && <Badge tone="amber">legacy — view only</Badge>}
          {hasHand && <Badge tone="green">hand-reviewed</Badge>}
        </div>
      </div>

      {/* filmstrip */}
      <div className="px-4 pt-3">
        <TickFilmstrip ticks={ticks} idx={idx} onSelect={setIdx} />
      </div>

      {/* workspace */}
      <div className="flex min-h-0 flex-1 gap-4 overflow-auto p-4">
        {/* media column */}
        <div className="flex w-[400px] shrink-0 flex-col gap-3">
          <Card
            title="near color"
            bodyClass="p-2"
            actions={
              overlayOn && overlay ? <Badge tone="indigo">{overlay.src}</Badge> : null
            }
          >
            <Img
              b64={overlayOn && overlay?.blend ? overlay.blend : payload?.color_near}
              mime="image/jpeg"
            />
          </Card>
          <div className="grid grid-cols-2 gap-3">
            <Card title="depth near" bodyClass="p-2">
              <Img b64={payload?.depth_near} mime="image/png" />
            </Card>
            <Card title="far color" bodyClass="p-2">
              <Img b64={payload?.color_far} mime="image/jpeg" />
            </Card>
          </div>
          <Card title="depth far" bodyClass="p-2">
            <Img b64={payload?.depth_far} mime="image/png" />
          </Card>
        </div>

        {/* editor column */}
        <div className="flex min-w-0 flex-1 flex-col gap-3">
          <Card
            title="BEV label editor"
            bodyClass="flex flex-col items-center gap-2 p-4"
            actions={
              <>
                <Segmented
                  size="sm"
                  value={cls}
                  onChange={setCls}
                  options={[
                    { value: 0, label: <ClsDot i={0} /> },
                    { value: 1, label: <ClsDot i={1} /> },
                    { value: 2, label: <ClsDot i={2} /> },
                  ]}
                />
                <span className="flex items-center gap-1 text-[11px] text-zinc-500">
                  brush
                  <Button onClick={() => setBrush((b) => Math.max(1, b - 1))}>−</Button>
                  <b className="w-4 text-center font-mono text-zinc-300">{brush}</b>
                  <Button onClick={() => setBrush((b) => Math.min(5, b + 1))}>+</Button>
                </span>
                <Button onClick={undo} disabled={!undoStack.length || !editable}>
                  undo
                </Button>
                <Button onClick={() => void reloadTick()}>revert</Button>
                <Button
                  variant="primary"
                  onClick={() => void save()}
                  disabled={!editable || !dirty}
                  className={dirty ? "ring-2 ring-amber-400/60" : ""}
                >
                  {dirty ? "save •" : "save"}
                </Button>
                <Button onClick={() => void clearHand()} disabled={!hasHand}>
                  clear hand
                </Button>
              </>
            }
          >
            {loading && !payload ? (
              <div className="flex h-[486px] items-center justify-center">
                <Spinner className="h-6 w-6" />
              </div>
            ) : (
              <BevCanvas
                grid={grid}
                diffGrid={miningKey ? ((g[miningKey] as Grid | undefined) ?? null) : null}
                editable={editable}
                brush={brush}
                cls={cls}
                underlay={texOn ? tex : null}
                classAlpha={texOn ? paintAlpha : 1}
                onChange={(ng) => {
                  if (grid) setUndoStack((s) => [...s.slice(-199), cloneGrid(grid)]);
                  setGrid(ng);
                  setDirty(true);
                }}
                onHoverCell={setHover}
              />
            )}
            <div className="flex w-full items-center justify-between text-[11px] text-zinc-500">
              <span>
                {hover
                  ? `cell r${hover.r} c${hover.c} · ${CLS_NAMES[hover.value]}`
                  : "hover the grid for cell info"}
              </span>
              <ClassLegend />
            </div>
            <div className="flex w-full items-center gap-3 text-[11px] text-zinc-500">
              <span>image underlay</span>
              <Button active={texOn} onClick={() => setTexOn((v) => !v)}>
                {texOn ? "on" : "off"}
              </Button>
              <input
                type="range"
                min={10}
                max={100}
                value={paintAlpha * 100}
                onChange={(e) => setPaintAlpha(Number(e.target.value) / 100)}
                className="w-24 accent-indigo-500"
                title="label-color opacity over the projected image"
              />
              <span className="text-zinc-600">
                paint α {Math.round(paintAlpha * 100)}% — the camera image
                projected onto the grid (tall objects smear far-ward)
              </span>
            </div>
            {miningKey && (
              <div className="w-full text-[11px] text-zinc-500">
                white outlines = <b className="text-zinc-300">{miningKey}</b> &gt; 0
              </div>
            )}
          </Card>

          <Card title="label balance" bodyClass="space-y-2.5">
            <StatRow
              label="fused"
              counts={fusedCounts}
              note={hasHand ? "superseded by hand" : "training target"}
            />
            <StatRow label="hand" counts={handCounts} note={hasHand ? "preferred by training" : null} />
            <div className="flex flex-wrap gap-x-5 gap-y-1 pt-1 font-mono text-[11px] text-zinc-500">
              <span>
                disagree <b className="text-zinc-300">{cellCount("disagree")}</b>
              </span>
              <span>
                unconfirmed <b className="text-zinc-300">{cellCount("unconfirmed")}</b>
              </span>
            </div>
          </Card>

          {m && (
            <Card title="tick context">
              <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 font-mono text-[11px]">
                <dt className="text-zinc-600">cmd_vel</dt>
                <dd className="text-zinc-300">
                  vx {m.cmd_vel?.vx?.toFixed(3) ?? "—"} · wz {m.cmd_vel?.wz?.toFixed(3) ?? "—"}
                </dd>
                <dt className="text-zinc-600">odom</dt>
                <dd className="text-zinc-300">
                  x {m.odom?.x?.toFixed(2) ?? "—"} · y {m.odom?.y?.toFixed(2) ?? "—"} · θ{" "}
                  {m.odom?.theta?.toFixed(2) ?? "—"}
                </dd>
                <dt className="text-zinc-600">goal</dt>
                <dd className="text-zinc-300">
                  {m.goal?.type === "heading"
                    ? `heading ${Math.round(((m.goal.heading_rad ?? 0) * 180) / Math.PI)}°`
                    : m.goal?.type === "point"
                      ? `xy (${m.goal.x?.toFixed(2)}, ${m.goal.y?.toFixed(2)})`
                      : "none"}
                </dd>
              </dl>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}

function ClsDot({ i }: { i: number }) {
  return (
    <span className="inline-flex items-center gap-1.5">
      <span className="inline-block h-2.5 w-2.5 rounded-[3px]" style={{ background: CLS_HEX[i] }} />
      {CLS_NAMES[i]}
    </span>
  );
}

function StatRow({
  label,
  counts,
  note,
}: {
  label: string;
  counts: [number, number, number] | null;
  note: string | null;
}) {
  const tot = counts ? counts[0] + counts[1] + counts[2] || 1 : 1;
  return (
    <div className="flex items-center gap-3">
      <span className="w-12 text-right text-[11px] text-zinc-500">{label}</span>
      <div className="flex h-2 flex-1 overflow-hidden rounded-full bg-zinc-800">
        {counts &&
          counts.map((v, i) => (
            <div
              key={i}
              style={{ width: `${(100 * v) / tot}%`, background: CLS_HEX[i] }}
            />
          ))}
      </div>
      <span className="w-44 text-right font-mono text-[10px] text-zinc-500">
        {counts
          ? counts.map((v, i) => `${CLS_NAMES[i][0]} ${Math.round((100 * v) / tot)}%`).join(" · ")
          : "—"}
        {note && <span className="ml-2 text-zinc-600">({note})</span>}
      </span>
    </div>
  );
}
