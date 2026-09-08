import { useCallback, useEffect, useState } from "react";

export function cn(...xs: (string | false | null | undefined)[]): string {
  return xs.filter(Boolean).join(" ");
}

/* ---------------------------------------------------------------- buttons */

type Variant = "default" | "primary" | "ghost" | "danger";

const BTN: Record<Variant, string> = {
  default:
    "bg-zinc-800 hover:bg-zinc-700 text-zinc-200 ring-1 ring-inset ring-white/10",
  primary: "bg-indigo-600 hover:bg-indigo-500 text-white shadow-sm",
  ghost: "text-zinc-400 hover:text-zinc-100 hover:bg-zinc-800/80",
  danger: "bg-red-600/90 hover:bg-red-600 text-white",
};

export function Button({
  variant = "default",
  active,
  className = "",
  ...rest
}: React.ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: Variant;
  active?: boolean;
}) {
  return (
    <button
      {...rest}
      className={cn(
        "inline-flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-xs font-medium transition-colors",
        "disabled:pointer-events-none disabled:opacity-40",
        active
          ? "bg-zinc-600 text-white ring-1 ring-inset ring-white/20"
          : BTN[variant],
        className,
      )}
    />
  );
}

export function Kbd({ children }: { children: React.ReactNode }) {
  return (
    <kbd className="rounded border border-zinc-700 bg-zinc-800 px-1 py-0.5 font-mono text-[10px] leading-none text-zinc-400">
      {children}
    </kbd>
  );
}

/* ------------------------------------------------------------- segmented */

export function Segmented<T extends string | number>({
  value,
  onChange,
  options,
  size = "md",
}: {
  value: T;
  onChange: (v: T) => void;
  options: { value: T; label: React.ReactNode; title?: string }[];
  size?: "sm" | "md";
}) {
  return (
    <div
      className={cn(
        "inline-flex rounded-lg bg-zinc-800/80 p-0.5 ring-1 ring-inset ring-white/10",
        size === "sm" ? "text-[11px]" : "text-xs",
      )}
    >
      {options.map((o) => (
        <button
          key={String(o.value)}
          title={o.title}
          onClick={() => onChange(o.value)}
          className={cn(
            "rounded-md px-2.5 py-1 font-medium transition-colors",
            value === o.value
              ? "bg-zinc-600 text-white shadow-sm"
              : "text-zinc-400 hover:text-zinc-200",
          )}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

/* ----------------------------------------------------------------- cards */

export function Card({
  title,
  actions,
  children,
  className = "",
  bodyClass = "",
}: {
  title?: React.ReactNode;
  actions?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
  bodyClass?: string;
}) {
  return (
    <section
      className={cn(
        "rounded-xl bg-zinc-900 ring-1 ring-inset ring-white/[0.06]",
        className,
      )}
    >
      {(title || actions) && (
        <header className="flex items-center justify-between gap-3 border-b border-white/[0.06] px-3.5 py-2.5">
          <h2 className="text-[11px] font-semibold uppercase tracking-wider text-zinc-500">
            {title}
          </h2>
          {actions && <div className="flex items-center gap-1.5">{actions}</div>}
        </header>
      )}
      <div className={cn("p-3.5", bodyClass)}>{children}</div>
    </section>
  );
}

const TONES = {
  zinc: "bg-zinc-800 text-zinc-300 ring-zinc-700",
  green: "bg-emerald-950 text-emerald-300 ring-emerald-800",
  amber: "bg-amber-950 text-amber-300 ring-amber-800",
  red: "bg-red-950 text-red-300 ring-red-800",
  indigo: "bg-indigo-950 text-indigo-300 ring-indigo-800",
} as const;

export function Badge({
  tone = "zinc",
  children,
}: {
  tone?: keyof typeof TONES;
  children: React.ReactNode;
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-medium ring-1 ring-inset",
        TONES[tone],
      )}
    >
      {children}
    </span>
  );
}

/** Horizontal stacked bar of the three label classes. */
export function ClassBar({ f0, f1, f2 }: { f0: number; f1: number; f2: number }) {
  const tot = f0 + f1 + f2 || 1;
  return (
    <div className="flex h-1.5 w-full overflow-hidden rounded-full bg-zinc-800">
      <div style={{ width: `${(100 * f0) / tot}%`, background: "#d32f2f" }} />
      <div style={{ width: `${(100 * f1) / tot}%`, background: "#2ea043" }} />
      <div style={{ width: `${(100 * f2) / tot}%`, background: "#ffa500" }} />
    </div>
  );
}

export function ClassLegend() {
  return (
    <span className="inline-flex items-center gap-3 text-[11px] text-zinc-400">
      {(["blocked", "navigable", "caution"] as const).map((n, i) => (
        <span key={n} className="inline-flex items-center gap-1">
          <span
            className="inline-block h-2.5 w-2.5 rounded-[3px]"
            style={{ background: ["#d32f2f", "#2ea043", "#ffa500"][i] }}
          />
          {n}
          <Kbd>{i + 1}</Kbd>
        </span>
      ))}
    </span>
  );
}

export function Spinner({ className = "" }: { className?: string }) {
  return (
    <svg
      className={cn("h-4 w-4 animate-spin text-zinc-500", className)}
      viewBox="0 0 24 24"
      fill="none"
    >
      <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" className="opacity-20" />
      <path d="M22 12a10 10 0 0 0-10-10" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
    </svg>
  );
}

/* ---------------------------------------------------------------- images */

export function Img({
  b64,
  mime,
  label,
  className = "",
}: {
  b64: string | null | undefined;
  mime: string;
  label?: React.ReactNode;
  className?: string;
}) {
  return (
    <figure
      className={cn(
        "overflow-hidden rounded-lg bg-black ring-1 ring-inset ring-white/[0.06]",
        className,
      )}
    >
      {b64 ? (
        <img src={`data:${mime};base64,${b64}`} alt="" className="block w-full" />
      ) : (
        <div
          className="aspect-[16/10] w-full"
          style={{
            background:
              "repeating-linear-gradient(45deg,#18181b,#18181b 8px,#101013 8px,#101013 16px)",
          }}
        />
      )}
      {label && (
        <figcaption className="flex items-center justify-between px-2 py-1 text-[10px] text-zinc-500">
          {label}
        </figcaption>
      )}
    </figure>
  );
}

/* ---------------------------------------------------------------- toasts */

export type ToastKind = "info" | "success" | "error";
interface ToastItem {
  id: number;
  msg: string;
  kind: ToastKind;
}

let _toasts: ToastItem[] = [];
const _listeners = new Set<() => void>();
let _seq = 0;

function _emit() {
  _listeners.forEach((l) => l());
}

export function toast(msg: string, kind: ToastKind = "info") {
  const id = ++_seq;
  _toasts = [..._toasts, { id, msg, kind }];
  _emit();
  window.setTimeout(() => {
    _toasts = _toasts.filter((t) => t.id !== id);
    _emit();
  }, 3600);
}

export function Toasts() {
  const [, force] = useState(0);
  useEffect(() => {
    const l = () => force((n) => n + 1);
    _listeners.add(l);
    return () => {
      _listeners.delete(l);
    };
  }, []);
  return (
    <div className="pointer-events-none fixed bottom-4 right-4 z-50 flex w-80 flex-col gap-2">
      {_toasts.map((t) => (
        <div
          key={t.id}
          className={cn(
            "pointer-events-auto rounded-lg px-3.5 py-2.5 text-xs shadow-xl ring-1 backdrop-blur",
            t.kind === "success" && "bg-emerald-950/90 text-emerald-200 ring-emerald-800",
            t.kind === "error" && "bg-red-950/90 text-red-200 ring-red-800",
            t.kind === "info" && "bg-zinc-900/95 text-zinc-200 ring-white/10",
          )}
        >
          {t.msg}
        </div>
      ))}
    </div>
  );
}

/* ----------------------------------------------------------------- modal */

export function Modal({
  open,
  onClose,
  title,
  children,
}: {
  open: boolean;
  onClose: () => void;
  title: string;
  children: React.ReactNode;
}) {
  const onKey = useCallback(
    (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    },
    [onClose],
  );
  useEffect(() => {
    if (!open) return;
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onKey]);
  if (!open) return null;
  return (
    <div
      className="fixed inset-0 z-40 flex items-center justify-center bg-black/60 p-6 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="max-h-[80vh] w-full max-w-lg overflow-y-auto rounded-xl bg-zinc-900 p-5 shadow-2xl ring-1 ring-white/10"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="mb-3 flex items-center justify-between">
          <h2 className="text-sm font-semibold text-zinc-100">{title}</h2>
          <Button variant="ghost" onClick={onClose} aria-label="close">
            ✕
          </Button>
        </div>
        {children}
      </div>
    </div>
  );
}
