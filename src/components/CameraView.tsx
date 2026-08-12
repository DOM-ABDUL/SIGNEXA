import { useEffect, useRef, useState } from "react";
import { PrimaryButton } from "./PrimaryButton";

type CameraStatus = "idle" | "loading" | "active" | "error";

type CameraViewProps = {
  autoStart?: boolean;
};

function getCameraErrorMessage(error: unknown): string {
  if (!(error instanceof DOMException)) {
    return "Unable to start the camera right now. Please try again.";
  }

  switch (error.name) {
    case "NotAllowedError":
    case "SecurityError":
      return "Camera permission is required for SignSync. Please allow access and try again.";
    case "NotFoundError":
      return "No camera was found on this device.";
    case "NotReadableError":
      return "The camera is busy in another application. Close that app and try again.";
    case "OverconstrainedError":
      return "The selected camera mode is not supported on this device.";
    case "AbortError":
      return "Camera startup was interrupted. Please try again.";
    default:
      return "Unable to start the camera right now. Please try again.";
  }
}

export function CameraView({ autoStart = false }: CameraViewProps) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const isMountedRef = useRef(true);
  const requestIdRef = useRef(0);

  const [status, setStatus] = useState<CameraStatus>("idle");
  const [errorMessage, setErrorMessage] = useState<string>("");

  const releaseStream = () => {
    const stream = streamRef.current;

    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  };

  const stopCamera = () => {
    releaseStream();

    setStatus("idle");
    setErrorMessage("");
  };

  const startCamera = async () => {
    if (status === "loading") {
      return;
    }

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setStatus("error");
      setErrorMessage("This browser does not support camera access.");
      return;
    }

    if (!window.isSecureContext) {
      setStatus("error");
      setErrorMessage("Camera access requires HTTPS or localhost.");
      return;
    }

    const requestId = requestIdRef.current + 1;
    requestIdRef.current = requestId;

    releaseStream();
    setStatus("loading");
    setErrorMessage("");

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          facingMode: { ideal: "user" },
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
      });

      if (!isMountedRef.current || requestId !== requestIdRef.current) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }

      streamRef.current = stream;

      stream.getVideoTracks().forEach((track) => {
        track.onended = () => {
          if (!isMountedRef.current) {
            return;
          }

          releaseStream();
          setStatus("error");
          setErrorMessage("Camera stream ended. Start the camera again.");
        };
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;

        try {
          await videoRef.current.play();
        } catch {
          releaseStream();
          setStatus("error");
          setErrorMessage("Camera started, but video playback was blocked. Tap Start Camera again.");
          return;
        }
      }

      setStatus("active");
    } catch (error) {
      if (requestId !== requestIdRef.current) {
        return;
      }

      setStatus("error");
      setErrorMessage(getCameraErrorMessage(error));
    }
  };

  useEffect(() => {
    isMountedRef.current = true;

    if (autoStart) {
      void startCamera();
    }

    return () => {
      isMountedRef.current = false;
      releaseStream();
    };
  }, [autoStart]);

  const statusMessageByState: Record<CameraStatus, string> = {
    idle: "Camera not started",
    loading: "Starting camera...",
    active: "Camera active",
    error: errorMessage || "Camera is unavailable",
  };

  return (
    <section className="w-full max-w-3xl" aria-label="Camera preview area">
      <div className="overflow-hidden rounded-2xl border border-white/10 bg-slate-900">
        <div className="relative aspect-video w-full bg-slate-950">
          <video
            ref={videoRef}
            className={`h-full w-full object-cover ${status === "active" ? "opacity-100" : "opacity-0"}`}
            autoPlay
            playsInline
            muted
          />

          {status !== "active" && (
            <div className="absolute inset-0 flex items-center justify-center px-4 text-center text-sm text-slate-400 sm:text-base">
              {status === "loading"
                ? "Waiting for camera permission..."
                : "Camera preview will appear here after you start."}
            </div>
          )}
        </div>
      </div>

      <p
        role="status"
        aria-live="polite"
        className={`mt-4 text-sm font-medium ${status === "error" ? "text-red-300" : "text-slate-200"}`}
      >
        {statusMessageByState[status]}
      </p>

      <div className="mt-5 flex flex-col gap-3 sm:flex-row">
        {status !== "active" ? (
          <PrimaryButton onClick={() => void startCamera()} disabled={status === "loading"}>
            {status === "error" ? "Try Again" : "Start Camera"}
          </PrimaryButton>
        ) : (
          <PrimaryButton onClick={stopCamera} className="bg-slate-700 shadow-slate-900/30 hover:bg-slate-600 active:bg-slate-800">
            Stop Camera
          </PrimaryButton>
        )}
      </div>
    </section>
  );
}