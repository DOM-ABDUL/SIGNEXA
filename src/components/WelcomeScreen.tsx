import { PrimaryButton } from "./PrimaryButton";
import { appContent } from "../data/appContent";

type WelcomeScreenProps = {
  onStartCamera: () => void;
};

/**
 * The SignSync landing screen: title, purpose, and the main action button.
 */
export function WelcomeScreen({ onStartCamera }: WelcomeScreenProps) {
  return (
    <main className="flex flex-1 items-center justify-center px-4 py-12 sm:px-6 sm:py-16">
      <section className="mx-auto w-full max-w-3xl text-center">
        <h1 className="text-5xl font-bold tracking-tight text-white sm:text-6xl">
          {appContent.brand}
        </h1>

        <p className="mt-4 text-lg font-medium text-brand-400 sm:text-xl">
          {appContent.tagline}
        </p>

        <p className="mx-auto mt-5 max-w-xl text-base leading-relaxed text-slate-300 sm:text-lg">
          {appContent.description}
        </p>

        <div className="mt-10 flex justify-center">
          <PrimaryButton onClick={onStartCamera}>
            {appContent.primaryActionLabel}
          </PrimaryButton>
        </div>

        <p className="mt-4 text-sm text-slate-400">
          {appContent.helperText}
        </p>
      </section>
    </main>
  );
}