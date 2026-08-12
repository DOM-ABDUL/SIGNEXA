import { appContent } from "../data/appContent";

/** Small app bar shown at the top of every screen. */
export function Header() {
  return (
    <header className="w-full border-b border-white/10 bg-slate-950/60">
      <div className="mx-auto flex max-w-3xl items-center gap-3 px-4 py-4 sm:px-6">
        <span
          aria-hidden="true"
          className="flex h-9 w-9 items-center justify-center rounded-lg bg-brand-600 text-lg font-bold text-white"
        >
          S
        </span>
        <span className="text-base font-semibold tracking-tight text-white sm:text-lg">{appContent.brand}</span>
        <span className="ml-auto rounded-full border border-white/15 px-3 py-1 text-xs font-medium text-slate-300">
          {appContent.statusBadge}
        </span>
      </div>
    </header>
  );
}
