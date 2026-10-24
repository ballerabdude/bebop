import { clsx } from "clsx";
import type { ClassValue } from "clsx";
import { extendTailwindMerge } from "tailwind-merge";

// tailwind-merge needs to know the custom theme scales so `text-label`
// (font size) doesn't get treated as a text color and dropped against
// `text-text-dim`, and `h-control-md` merges with `h-10`.
const twMerge = extendTailwindMerge({
  extend: {
    theme: {
      text: ["micro", "caption", "label", "body-sm"],
      spacing: ["control-sm", "control-md", "control-lg", "touch"],
      radius: ["card", "input"],
    },
  },
});

/// Compose class names; later Tailwind utilities win over earlier ones.
export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}
