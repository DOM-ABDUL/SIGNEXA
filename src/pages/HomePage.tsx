import { useState } from "react";
import { CameraView } from "../components/CameraView";
import { Header } from "../components/Header";
import { WelcomeScreen } from "../components/WelcomeScreen";

type Screen = "welcome" | "camera";

/** Page = layout + the sections it shows. */
export function HomePage() {
  const [screen, setScreen] = useState<Screen>("welcome");

  return (
    <div className="flex min-h-screen flex-col bg-slate-950 text-slate-100">
      <Header />
      {screen === "welcome" ? (
        <WelcomeScreen onStartCamera={() => setScreen("camera")} />
      ) : (
        <main className="flex flex-1 items-center justify-center px-4 py-8 sm:px-6 sm:py-10">
          <div className="w-full max-w-3xl">
            <div className="mb-5 flex items-center justify-between gap-4">
              <h2 className="text-xl font-semibold text-white sm:text-2xl">Camera Setup</h2>
              <button
                type="button"
                onClick={() => setScreen("welcome")}
                className="rounded-lg border border-white/20 px-3 py-2 text-sm font-medium text-slate-200 hover:bg-white/10 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand-400"
              >
                Back
              </button>
            </div>
            <CameraView autoStart />
          </div>
        </main>
      )}
      <footer className="border-t border-white/10 px-4 py-5 text-center text-xs text-slate-500 sm:px-6">
        Built to work offline · Indian Sign Language
      </footer>
    </div>
  );
}
