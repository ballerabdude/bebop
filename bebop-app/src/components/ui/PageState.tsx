import type { ReactNode } from "react";

import { cn } from "./cn";
import { Spinner } from "./Spinner";

/// Full-area loading / empty / error placeholder.
export function PageState({
  kind = "loading",
  title,
  children,
  action,
  className,
}: {
  kind?: "loading" | "empty" | "error";
  title?: ReactNode;
  children?: ReactNode;
  action?: ReactNode;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "flex flex-1 flex-col items-center justify-center gap-3 px-4 py-10 text-center text-text-dim",
        className,
      )}
    >
      {kind === "loading" ? <Spinner large /> : null}
      {title ? (
        <div
          className={cn(
            "text-base font-semibold",
            kind === "error" ? "text-danger-fg" : "text-text",
          )}
        >
          {title}
        </div>
      ) : null}
      {children ? <div className="max-w-md text-sm leading-relaxed">{children}</div> : null}
      {action}
    </div>
  );
}
