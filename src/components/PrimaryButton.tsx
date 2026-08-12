import type { ButtonHTMLAttributes, ReactNode } from "react";
import { cn } from "../utils/cn";

type PrimaryButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  children: ReactNode;
};

/**
 * A large, touch-friendly button used for the main action on a screen.
 * It is a normal button, so keyboard and screen readers work by default.
 */
export function PrimaryButton({
  children,
  className,
  type = "button",
  ...props
}: PrimaryButtonProps) {
  return (
    <button
      type={type}
      className={cn(
        "inline-flex min-h-14 w-full items-center justify-center gap-2 rounded-xl px-8 text-base font-semibold sm:w-auto sm:text-lg",
        "bg-brand-600 text-white shadow-lg shadow-brand-700/30 transition-colors",
        "hover:bg-brand-500 active:bg-brand-700",
        "disabled:cursor-not-allowed disabled:opacity-60",
        className,
      )}
      {...props}
    >
      {children}
    </button>
  );
}