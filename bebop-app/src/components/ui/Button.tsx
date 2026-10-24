import { forwardRef } from "react";
import type { ButtonHTMLAttributes } from "react";

import { cn } from "./cn";
import { Spinner } from "./Spinner";

export type ButtonVariant = "primary" | "secondary" | "ghost" | "danger" | "success";
export type ButtonSize = "sm" | "md" | "lg";

export type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: ButtonVariant;
  size?: ButtonSize;
  loading?: boolean;
};

const VARIANT: Record<ButtonVariant, string> = {
  primary: "bg-accent text-white hover:bg-accent-press",
  secondary: "bg-bg-elev border-border text-text hover:bg-bg-elev-2",
  ghost: "bg-transparent text-text-dim hover:text-text hover:bg-bg-elev-2/60",
  danger: "bg-danger text-white hover:bg-danger-press",
  success: "bg-success text-white hover:bg-success-press",
};

// Heights follow the control scale (36 / 44 / 56). On coarse pointers
// (touch) every size is at least 44 px tall.
const SIZE: Record<ButtonSize, string> = {
  sm: "h-control-sm px-3 text-label pointer-coarse:h-touch",
  md: "h-control-md px-4 text-sm",
  lg: "h-control-lg px-5 text-base",
};

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  {
    variant = "primary",
    size = "md",
    loading = false,
    disabled,
    children,
    className,
    type = "button",
    ...rest
  },
  ref,
) {
  return (
    <button
      ref={ref}
      type={type}
      className={cn(
        "inline-flex shrink-0 items-center justify-center gap-2 whitespace-nowrap rounded-card border border-transparent font-semibold transition-colors duration-150 active:scale-[0.98] disabled:cursor-not-allowed disabled:opacity-55 cursor-pointer",
        VARIANT[variant],
        SIZE[size],
        className,
      )}
      disabled={disabled || loading}
      {...rest}
    >
      {loading ? <Spinner /> : null}
      {children}
    </button>
  );
});
