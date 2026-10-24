// Small transient confirmations for the Drive page (marker saved, 409, ...).
// One visible at a time; a new toast replaces the old one.

import { useCallback, useEffect, useRef, useState } from "react";

import { cn } from "../../components/ui";

export type ToastTone = "ok" | "warn" | "err";

interface Toast {
  id: number;
  tone: ToastTone;
  text: string;
}

export function useToast(): {
  toast: Toast | null;
  show: (tone: ToastTone, text: string, ms?: number) => void;
  dismiss: () => void;
} {
  const [toast, setToast] = useState<Toast | null>(null);
  const timer = useRef<number | null>(null);
  const seq = useRef(0);
  const dismiss = useCallback(() => {
    if (timer.current !== null) window.clearTimeout(timer.current);
    timer.current = null;
    setToast(null);
  }, []);
  const show = useCallback((tone: ToastTone, text: string, ms = 2800) => {
    if (timer.current !== null) window.clearTimeout(timer.current);
    seq.current += 1;
    setToast({ id: seq.current, tone, text });
    timer.current = window.setTimeout(() => {
      timer.current = null;
      setToast(null);
    }, ms);
  }, []);
  useEffect(
    () => () => {
      if (timer.current !== null) window.clearTimeout(timer.current);
    },
    [],
  );
  return { toast, show, dismiss };
}

const TONE: Record<ToastTone, string> = {
  ok: "border-success/50 text-success-fg",
  warn: "border-warning/50 text-warning-fg",
  err: "border-danger/60 text-danger-fg",
};
const DOT: Record<ToastTone, string> = {
  ok: "bg-success",
  warn: "bg-warning",
  err: "bg-danger",
};

export function ToastView({
  toast,
  onDismiss,
  className,
}: {
  toast: Toast | null;
  onDismiss: () => void;
  className?: string;
}) {
  return (
    <div
      aria-live="polite"
      className={cn("pointer-events-none flex justify-center", className)}
    >
      {toast ? (
        <button
          key={toast.id}
          type="button"
          onClick={onDismiss}
          className={cn(
            "animate-fade-in pointer-events-auto flex max-w-[min(30rem,calc(100vw-2rem))] items-center gap-2 rounded-full border bg-bg-elev/95 px-4 py-2 text-body-sm font-semibold shadow-lg backdrop-blur-md",
            TONE[toast.tone],
          )}
        >
          <span className={cn("h-2 w-2 shrink-0 rounded-full", DOT[toast.tone])} aria-hidden />
          <span className="text-left leading-snug">{toast.text}</span>
        </button>
      ) : null}
    </div>
  );
}
