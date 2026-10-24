// The conversation: operator bubbles (typed or spoken), the robot's streamed
// reply, and a card per tool call with its result and motion progress.

import { useLayoutEffect, useRef, useState } from "react";
import type { ReactNode } from "react";

import { cn, Spinner } from "../../components/ui";
import { assistantText, motionFraction, summarizeArgs } from "../../voice/agentTurns";
import type { ToolItem, Turn } from "../../voice/agentTurns";
import {
  CheckIcon,
  CrossIcon,
  KeyboardIcon,
  MicIcon,
  MoveIcon,
  SpeakerIcon,
  ToolIcon,
} from "./icons";

const TOOL_LABEL: Record<string, string> = {
  where_am_i: "Where am I",
  list_places: "List places",
  place_facts: "Place facts",
  remember_place: "Remember place",
  add_place_fact: "Add place fact",
  rename_place: "Rename place",
  forget_place: "Forget place",
  robot_status: "Robot status",
  note_drive_intent: "Note drive intent",
  mark_drive_event: "Mark drive event",
  move_relative: "Move",
};

function toolLabel(name: string): string {
  return (
    TOOL_LABEL[name] ??
    name.replace(/_/g, " ").replace(/^./, (c) => c.toUpperCase())
  );
}

function formatMs(ms: number | undefined): string {
  if (ms === undefined) return "";
  return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1)} s`;
}

function formatTime(at: number): string {
  try {
    return new Date(at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  } catch {
    return "";
  }
}

function ThinkingDots({ label }: { label: string }) {
  return (
    <div className="flex items-center gap-2 py-1 text-body-sm text-text-dim" role="status">
      <span className="flex gap-1" aria-hidden>
        {[0, 1, 2].map((i) => (
          <span
            key={i}
            className="h-1.5 w-1.5 animate-pulse rounded-full bg-text-dim"
            style={{ animationDelay: `${i * 180}ms` }}
          />
        ))}
      </span>
      {label}
    </div>
  );
}

function MotionBar({ tool }: { tool: ToolItem }) {
  const p = tool.progress;
  if (!p) return null;
  const frac = motionFraction(p);
  const parts: string[] = [];
  if (p.totalM) parts.push(`${Math.abs(p.doneM).toFixed(2)} / ${Math.abs(p.totalM).toFixed(2)} m`);
  if (p.totalDeg) parts.push(`${Math.round(Math.abs(p.doneDeg))} / ${Math.round(Math.abs(p.totalDeg))}°`);
  const running = tool.ok === undefined;
  return (
    <div className="mt-2">
      <div
        className="h-2 overflow-hidden rounded-full bg-bg-elev-2"
        role="progressbar"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(frac * 100)}
        aria-label="Move progress"
      >
        <div
          className={cn(
            "h-full rounded-full transition-[width] duration-200",
            running ? "bg-warning" : tool.ok ? "bg-success" : "bg-danger",
          )}
          style={{ width: `${Math.max(3, frac * 100)}%` }}
        />
      </div>
      <div className="mt-1 flex justify-between text-caption tabular-nums text-text-dim">
        <span>{parts.join(" · ")}</span>
        <span>{Math.round(frac * 100)}%</span>
      </div>
    </div>
  );
}

function ToolCard({ tool }: { tool: ToolItem }) {
  const [expanded, setExpanded] = useState(false);
  const running = tool.ok === undefined;
  const motion = tool.name === "move_relative";
  const args = summarizeArgs(tool.args, 90);
  const long = (tool.result?.length ?? 0) > 180;
  return (
    <div
      className={cn(
        "rounded-input border bg-bg/60 px-3 py-2.5",
        running
          ? motion
            ? "border-warning/50"
            : "border-info/40"
          : tool.ok
            ? "border-border"
            : "border-danger/40",
      )}
    >
      <div className="flex items-center gap-2">
        <span
          className={cn(
            "inline-flex h-6 w-6 shrink-0 items-center justify-center rounded-full",
            running
              ? "bg-info/15 text-info-fg"
              : tool.ok
                ? "bg-success/15 text-success"
                : "bg-danger/15 text-danger-fg",
          )}
        >
          {running ? <Spinner className="h-3.5 w-3.5" /> : tool.ok ? <CheckIcon size={14} /> : <CrossIcon size={14} />}
        </span>
        <span className="flex min-w-0 flex-1 items-center gap-1.5 text-body-sm font-semibold text-text">
          <span className="shrink-0 text-text-dim">{motion ? <MoveIcon size={14} /> : <ToolIcon size={14} />}</span>
          <span className="truncate">{toolLabel(tool.name)}</span>
        </span>
        <span className="shrink-0 text-caption tabular-nums text-text-dim">
          {running ? (motion ? "moving…" : "running…") : formatMs(tool.ms)}
        </span>
      </div>
      {args ? (
        <div className="mt-1 truncate pl-8 font-mono text-caption text-text-dim" title={args}>
          {args}
        </div>
      ) : null}
      {motion ? <div className="pl-8"><MotionBar tool={tool} /></div> : null}
      {tool.result ? (
        <div className="mt-1.5 pl-8">
          <p
            className={cn(
              "whitespace-pre-wrap break-words text-label leading-relaxed",
              tool.ok ? "text-text-dim" : "text-danger-fg",
              long && !expanded && "line-clamp-3",
            )}
          >
            {tool.result}
          </p>
          {long ? (
            <button
              type="button"
              onClick={() => setExpanded((v) => !v)}
              className="mt-0.5 -ml-1 inline-flex min-h-8 items-center rounded px-1 text-caption font-semibold text-accent hover:underline pointer-coarse:min-h-touch"
            >
              {expanded ? "Show less" : "Show more"}
            </button>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}

function SourceTag({ source }: { source: Turn["source"] }) {
  return (
    <span className="inline-flex items-center gap-1 text-micro font-semibold uppercase tracking-wider text-text-dim">
      {source === "voice" ? <MicIcon size={12} /> : <KeyboardIcon size={12} />}
      {source === "voice" ? "Spoken" : "Typed"}
    </span>
  );
}

function UserBubble({ turn }: { turn: Turn }) {
  const recording = turn.status === "recording";
  const text =
    turn.userText ||
    (recording ? "Listening…" : turn.source === "voice" && turn.status === "pending" ? "Sending audio…" : turn.source === "voice" && turn.status === "running" ? "Transcribing…" : "(no words heard)");
  return (
    <div className="flex flex-col items-end gap-1">
      <div className="flex items-center gap-2">
        <SourceTag source={turn.source} />
        <span className="text-micro tabular-nums text-text-dim">{formatTime(turn.createdAt)}</span>
      </div>
      <div
        className={cn(
          "max-w-[85%] whitespace-pre-wrap break-words rounded-2xl rounded-br-md px-3.5 py-2 text-sm leading-relaxed",
          turn.userText ? "bg-accent text-white" : "border border-dashed border-accent/50 bg-accent/10 text-info-fg",
        )}
      >
        {recording ? (
          <span className="inline-flex items-center gap-2">
            <span className="h-2 w-2 animate-pulse rounded-full bg-danger" aria-hidden />
            {text}
          </span>
        ) : (
          text
        )}
      </div>
    </div>
  );
}

function EndNote({ turn }: { turn: Turn }) {
  let tone: "dim" | "warn" | "err" = "dim";
  let text: ReactNode = null;
  switch (turn.status) {
    case "cancelled":
      tone = "warn";
      text = "Stopped by the operator.";
      break;
    case "error":
      tone = "err";
      text = turn.error || "Something went wrong.";
      break;
    case "rejected":
      tone = "warn";
      text = "Not started: the robot was still busy with the previous request.";
      break;
    case "interrupted":
      tone = "warn";
      text = turn.error || "Interrupted.";
      break;
    default:
      return null;
  }
  return (
    <div
      className={cn(
        "rounded-input border px-3 py-2 text-label leading-relaxed",
        tone === "err"
          ? "border-danger/40 bg-danger/10 text-danger-fg"
          : tone === "warn"
            ? "border-warning/30 bg-warning/10 text-warning-fg"
            : "border-border text-text-dim",
      )}
    >
      {text}
    </div>
  );
}

function AssistantBlock({ turn }: { turn: Turn }) {
  const live = turn.status === "running" || turn.status === "pending";
  const last = turn.items[turn.items.length - 1];
  const waiting =
    live &&
    (turn.items.length === 0 || (last?.kind === "tool" && last.ok !== undefined));
  const ended = !live && turn.status !== "recording";
  if (turn.status === "recording") return null;
  if (!live && turn.items.length === 0 && turn.status === "done") {
    return <div className="pl-9 text-label italic text-text-dim">(no reply)</div>;
  }
  return (
    <div className="flex gap-2.5">
      <div
        className={cn(
          "mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full border text-caption font-bold",
          live ? "border-accent/60 bg-accent/15 text-accent" : "border-border bg-bg-elev-2 text-text-dim",
        )}
        aria-hidden
      >
        B
      </div>
      <div className="flex min-w-0 flex-1 flex-col gap-2">
        {turn.items.map((it, i) =>
          it.kind === "text" ? (
            <p key={i} className="whitespace-pre-wrap break-words text-sm leading-relaxed text-text">
              {it.text}
              {live && i === turn.items.length - 1 ? (
                <span className="ml-0.5 inline-block h-4 w-[2px] translate-y-0.5 animate-pulse bg-accent" aria-hidden />
              ) : null}
            </p>
          ) : (
            <ToolCard key={it.callId} tool={it} />
          ),
        )}
        {waiting ? (
          <ThinkingDots
            label={
              turn.cancelRequested
                ? "Stopping…"
                : turn.status === "pending"
                  ? turn.source === "voice"
                    ? "Sending…"
                    : "Waiting for the robot…"
                  : "Thinking…"
            }
          />
        ) : null}
        {live && !waiting && turn.cancelRequested ? <ThinkingDots label="Stopping…" /> : null}
        {turn.speaking ? (
          <span className="inline-flex items-center gap-1.5 text-caption text-text-dim">
            <SpeakerIcon size={14} /> Speaking
          </span>
        ) : null}
        {turn.status === "done" && turn.items.length > 0 && !assistantText(turn).trim() ? (
          <p className="text-label italic text-text-dim">
            The robot didn't answer after the tool.
          </p>
        ) : null}
        {ended ? <EndNote turn={turn} /> : null}
      </div>
    </div>
  );
}

/// Scrolls itself: sticks to the bottom while the operator is reading the
/// latest turn, stays put when they scrolled up.
export function Transcript({
  turns,
  empty,
  className,
}: {
  turns: Turn[];
  empty: ReactNode;
  className?: string;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const stick = useRef(true);
  useLayoutEffect(() => {
    const el = ref.current;
    // The empty state reads from the top; a conversation sticks to the end.
    if (el && stick.current && turns.length > 0) el.scrollTop = el.scrollHeight;
  });
  return (
    <div
      ref={ref}
      onScroll={(e) => {
        const el = e.currentTarget;
        stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
      }}
      className={cn("min-h-0 overflow-y-auto overscroll-contain", className)}
      aria-live="polite"
    >
      {turns.length === 0 ? (
        empty
      ) : (
        <ol className="flex flex-col gap-5 px-3 py-4 sm:px-4">
          {turns.map((t) => (
            <li key={t.id} className="flex flex-col gap-2.5 animate-fade-in">
              <UserBubble turn={t} />
              <AssistantBlock turn={t} />
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}
