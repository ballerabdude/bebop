// A small card over the Drive page (intent picker, link details). It lives
// inside the page, below the shell's status bar and under the floating
// E-STOP (z-40), so neither is ever covered. Tap outside or Esc closes it.

import { useEffect } from "react";
import type { ReactNode } from "react";

import { cn } from "../../components/ui";
import { CloseIcon } from "./icons";

export function PagePanel({
  title,
  onClose,
  children,
  className,
}: {
  title: string;
  onClose: () => void;
  children: ReactNode;
  className?: string;
}) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);
  return (
    <>
      <div className="absolute inset-0 z-[36] bg-black/45" onClick={onClose} aria-hidden />
      <div
        role="dialog"
        aria-label={title}
        className={cn(
          "animate-fade-in absolute inset-x-3 top-3 z-[37] mx-auto flex max-h-[calc(100%-1.5rem)] max-w-sm flex-col rounded-card border border-border bg-bg-elev/97 shadow-2xl backdrop-blur-md",
          className,
        )}
      >
        <div className="flex shrink-0 items-center justify-between gap-2 pl-4 pr-1">
          <h2 className="py-2 text-caption font-semibold uppercase tracking-wider text-text-dim">{title}</h2>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close"
            className="flex h-touch w-touch items-center justify-center rounded-card text-text-dim hover:text-text"
          >
            <CloseIcon />
          </button>
        </div>
        <div className="min-h-0 overflow-y-auto overscroll-contain px-4 pb-4">{children}</div>
      </div>
    </>
  );
}
