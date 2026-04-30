import type { ButtonHTMLAttributes, ReactNode } from "react";

type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "primary" | "secondary" | "ghost";
  loading?: boolean;
};

const variantClasses: Record<string, string> = {
  primary: "bg-accent text-white hover:bg-accent-press active:scale-[0.98]",
  secondary:
    "bg-bg-elev border-border text-text hover:bg-bg-elev-2 active:scale-[0.98]",
  ghost: "bg-transparent text-text-dim hover:text-text active:scale-[0.98]",
};

export function Button({
  variant = "primary",
  loading = false,
  disabled,
  children,
  className = "",
  ...rest
}: ButtonProps) {
  return (
    <button
      className={`inline-flex items-center justify-center gap-2 rounded-[var(--radius-card)] border border-transparent px-4 py-3.5 font-semibold transition-all duration-120 disabled:opacity-55 disabled:cursor-not-allowed cursor-pointer ${variantClasses[variant]} ${className}`}
      disabled={disabled || loading}
      {...rest}
    >
      {loading ? (
        <span
          className="inline-block w-3.5 h-3.5 rounded-full border-2 border-white/35 border-t-white animate-spin"
          aria-hidden
        />
      ) : null}
      <span>{children}</span>
    </button>
  );
}

export function Card({ children }: { children: ReactNode }) {
  return (
    <div className="bg-bg-elev border border-border rounded-[var(--radius-card)] px-4 py-2">
      {children}
    </div>
  );
}

export function Field({
  label,
  children,
  hint,
}: {
  label: string;
  children: ReactNode;
  hint?: string;
}) {
  return (
    <label className="flex flex-col gap-1.5">
      <span className="text-[13px] text-text-dim">{label}</span>
      {children}
      {hint ? <span className="text-xs text-text-dim">{hint}</span> : null}
    </label>
  );
}

export function Banner({
  tone = "info",
  children,
}: {
  tone?: "info" | "error" | "success";
  children: ReactNode;
}) {
  const toneClasses: Record<string, string> = {
    info: "bg-accent/10 text-[#cfe0ff]",
    error: "bg-danger/12 text-[#ffb5b8]",
    success: "bg-success/12 text-[#b1ebd2]",
  };
  return (
    <div
      className={`rounded-[var(--radius-card)] px-3.5 py-3 text-sm ${toneClasses[tone]}`}
    >
      {children}
    </div>
  );
}

export function Spinner({ large = false }: { large?: boolean }) {
  return (
    <span
      className={`inline-block rounded-full border-border border-t-accent animate-spin ${
        large ? "w-9 h-9 border-[3px]" : "w-4 h-4 border-2"
      }`}
      aria-hidden
    />
  );
}
