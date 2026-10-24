import { useCallback, useEffect, useId, useRef, useState } from "react";
import type { ReactNode } from "react";

import { Button } from "./Button";
import { cn } from "./cn";

/// Modal dialog. Page-owned on purpose: `window.confirm()` can be muted by
/// the browser ("prevent this page from creating additional dialogs"),
/// after which it silently returns false and destructive actions look
/// broken.
///
/// - Backdrop click and Esc close it.
/// - Focus moves into the dialog (the element marked `data-autofocus`, else
///   the first button) and Tab is trapped inside.
/// - On phones it docks to the bottom edge, above the safe area.
export function Dialog({
  open,
  onClose,
  title,
  children,
  footer,
  className,
}: {
  open: boolean;
  onClose: () => void;
  title: ReactNode;
  children?: ReactNode;
  footer?: ReactNode;
  className?: string;
}) {
  const panelRef = useRef<HTMLDivElement | null>(null);
  const titleId = useId();

  useEffect(() => {
    if (!open) return;
    const panel = panelRef.current;
    const prev = document.activeElement as HTMLElement | null;
    const first =
      panel?.querySelector<HTMLElement>("[data-autofocus]") ??
      panel?.querySelector<HTMLElement>("button, [href], input, select, textarea");
    first?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        onClose();
        return;
      }
      if (e.key !== "Tab" || !panel) return;
      const items = Array.from(
        panel.querySelectorAll<HTMLElement>(
          "button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled])",
        ),
      );
      if (items.length === 0) return;
      const firstEl = items[0];
      const lastEl = items[items.length - 1];
      if (e.shiftKey && document.activeElement === firstEl) {
        e.preventDefault();
        lastEl.focus();
      } else if (!e.shiftKey && document.activeElement === lastEl) {
        e.preventDefault();
        firstEl.focus();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("keydown", onKey);
      prev?.focus?.();
    };
  }, [open, onClose]);

  if (!open) return null;
  return (
    <div
      className="fixed inset-0 z-[60] flex items-end justify-center bg-black/55 p-0 backdrop-blur-sm sm:items-center sm:p-4"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div
        ref={panelRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        className={cn(
          "w-full max-w-md rounded-t-card border border-border bg-bg-elev shadow-2xl sm:rounded-card",
          className,
        )}
      >
        <h2 id={titleId} className="px-5 pt-4 pb-2 text-base font-semibold text-text">
          {title}
        </h2>
        {children ? (
          <div className="max-h-[60dvh] space-y-2 overflow-y-auto px-5 pb-4 text-sm leading-relaxed text-text-dim">
            {children}
          </div>
        ) : null}
        {footer ? (
          <div className="flex flex-col-reverse gap-2 rounded-b-card border-t border-border bg-bg-elev-2/40 px-5 py-3 pb-[max(0.75rem,env(safe-area-inset-bottom))] sm:flex-row sm:justify-end sm:pb-3">
            {footer}
          </div>
        ) : null}
      </div>
    </div>
  );
}

export interface ConfirmOptions {
  title: string;
  body?: ReactNode;
  confirmLabel: string;
  cancelLabel?: string;
  tone?: "danger" | "default";
}

/// Promise-based confirm: `const ok = await confirm({...})`. Render the
/// returned `dialog` element once in the screen.
export function useConfirm(): {
  confirm: (o: ConfirmOptions) => Promise<boolean>;
  dialog: ReactNode;
} {
  const [req, setReq] = useState<(ConfirmOptions & { resolve: (ok: boolean) => void }) | null>(
    null,
  );
  const confirm = useCallback(
    (o: ConfirmOptions) =>
      new Promise<boolean>((resolve) => {
        setReq((prev) => {
          prev?.resolve(false);
          return { ...o, resolve };
        });
      }),
    [],
  );
  const close = useCallback((ok: boolean) => {
    setReq((prev) => {
      prev?.resolve(ok);
      return null;
    });
  }, []);
  const onClose = useCallback(() => close(false), [close]);
  const dialog = (
    <Dialog
      open={req !== null}
      onClose={onClose}
      title={req?.title ?? ""}
      footer={
        req ? (
          <>
            <Button variant="secondary" data-autofocus onClick={() => close(false)}>
              {req.cancelLabel ?? "Cancel"}
            </Button>
            <Button
              variant={req.tone === "danger" ? "danger" : "primary"}
              onClick={() => close(true)}
            >
              {req.confirmLabel}
            </Button>
          </>
        ) : null
      }
    >
      {req?.body}
    </Dialog>
  );
  return { confirm, dialog };
}
