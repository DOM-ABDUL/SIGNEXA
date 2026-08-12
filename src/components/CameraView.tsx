import { useEffect, useRef, useState } from "react";
import type { HandLandmarker } from "@mediapipe/tasks-vision";
import { detectHandsForVideo, getHandLandmarker, closeHandLandmarker } from "../ml/mediapipe/detector";
import type { HandLandmarks, MediaPipeStatus } from "../ml/mediapipe/types";
import { PrimaryButton } from "./PrimaryButton";

type CameraStatus = "idle" | "loading" | "active" | "error";

type CameraViewProps = {
  autoStart?: boolean;
};

type HandDebugInfo = {
  handsDetected: number;
  landmarkCounts: number[];
};

const EMPTY_DEBUG_INFO: HandDebugInfo = {
  handsDetected: 0,
  landmarkCounts: [],
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

function getMediaPipeErrorMessage(error: unknown): string {
  if (error instanceof DOMException) {
    return "MediaPipe could not be initialized.";
  }

  return "MediaPipe could not be initialized.";
}

function drawLandmarksOnCanvas(canvas: HTMLCanvasElement, hands: HandLandmarks[]) {
  const context = canvas.getContext("2d");

  if (!context) {
    return;
  }

  context.clearRect(0, 0, canvas.width, canvas.height);

  context.fillStyle = "#38bdf8";

  for (const hand of hands) {
    for (const landmark of hand) {
      const x = landmark.x * canvas.width;
      const y = landmark.y * canvas.height;

      context.beginPath();
      context.arc(x, y, 4, 0, Math.PI * 2);
      context.fill();
    }
  }
}

function areDebugInfoEqual(a: HandDebugInfo, b: HandDebugInfo): boolean {
  if (a.handsDetected !== b.handsDetected) {
    return false;
  }

  if (a.landmarkCounts.length !== b.landmarkCounts.length) {
    return false;
  }

  for (let i = 0; i < a.landmarkCounts.length; i += 1) {
    if (a.landmarkCounts[i] !== b.landmarkCounts[i]) {
      return false;
    }
  }

  return true;
}

export function CameraView({ autoStart = false }: CameraViewProps) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const detectorRef = useRef<HandLandmarker | null>(null);
  const rafIdRef = useRef<number | null>(null);
  const isMountedRef = useRef(true);
  const requestIdRef = useRef(0);
  const lastVideoTimeRef = useRef(-1);
  const latestDebugInfoRef = useRef<HandDebugInfo>(EMPTY_DEBUG_INFO);

  const [status, setStatus] = useState<CameraStatus>("idle");
  const [errorMessage, setErrorMessage] = useState<string>("");
  const [mediaPipeStatus, setMediaPipeStatus] = useState<MediaPipeStatus>("idle");
  const [mediaPipeErrorMessage, setMediaPipeErrorMessage] = useState<string>("");
  const [debugInfo, setDebugInfo] = useState<HandDebugInfo>(EMPTY_DEBUG_INFO);

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

  const stopInferenceLoop = () => {
    if (rafIdRef.current !== null) {
      window.cancelAnimationFrame(rafIdRef.current);
      rafIdRef.current = null;
    }

    lastVideoTimeRef.current = -1;

    const canvas = canvasRef.current;
    const context = canvas?.getContext("2d");

    if (canvas && context) {
      context.clearRect(0, 0, canvas.width, canvas.height);
    }
  };

  const resetDetectionState = () => {
    latestDebugInfoRef.current = EMPTY_DEBUG_INFO;
    setDebugInfo(EMPTY_DEBUG_INFO);
  };

  const stopCamera = () => {
    stopInferenceLoop();
    releaseStream();

    setStatus("idle");
    setErrorMessage("");
    setMediaPipeStatus("idle");
    setMediaPipeErrorMessage("");
    resetDetectionState();
  };

  const runInferenceLoop = () => {
    const step = () => {
      if (!isMountedRef.current || !streamRef.current || !videoRef.current || !detectorRef.current) {
        return;
      }

      const video = videoRef.current;

      if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
        rafIdRef.current = window.requestAnimationFrame(step);
        return;
      }

      if (video.currentTime === lastVideoTimeRef.current) {
        rafIdRef.current = window.requestAnimationFrame(step);
        return;
      }

      lastVideoTimeRef.current = video.currentTime;

      const frame = detectHandsForVideo(detectorRef.current, video, performance.now());
      const nextDebugInfo: HandDebugInfo = {
        handsDetected: frame.hands.length,
        landmarkCounts: frame.hands.map((hand) => hand.length),
      };

      const canvas = canvasRef.current;
      if (canvas) {
        if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
        }

        drawLandmarksOnCanvas(canvas, frame.hands);
      }

      if (!areDebugInfoEqual(latestDebugInfoRef.current, nextDebugInfo)) {
        latestDebugInfoRef.current = nextDebugInfo;
        setDebugInfo(nextDebugInfo);
      }

      rafIdRef.current = window.requestAnimationFrame(step);
    };

    stopInferenceLoop();
    rafIdRef.current = window.requestAnimationFrame(step);
  };

  const ensureMediaPipeReady = async () => {
    if (detectorRef.current) {
      setMediaPipeStatus("ready");
      setMediaPipeErrorMessage("");
      return true;
    }

    setMediaPipeStatus("loading");
    setMediaPipeErrorMessage("");

    try {
      detectorRef.current = await getHandLandmarker();

      if (!isMountedRef.current) {
        return false;
      }

      setMediaPipeStatus("ready");
      return true;
    } catch (error) {
      if (!isMountedRef.current) {
        return false;
      }

      setMediaPipeStatus("error");
      setMediaPipeErrorMessage(getMediaPipeErrorMessage(error));
      return false;
    }
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

    stopInferenceLoop();
    releaseStream();
    resetDetectionState();
    setMediaPipeStatus("idle");
    setMediaPipeErrorMessage("");
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

          stopInferenceLoop();
          releaseStream();
          setStatus("error");
          setErrorMessage("Camera stream ended. Start the camera again.");
          setMediaPipeStatus("idle");
          setMediaPipeErrorMessage("");
          resetDetectionState();
        };
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;

        try {
          await videoRef.current.play();
        } catch {
          stopInferenceLoop();
          releaseStream();
          setStatus("error");
          setErrorMessage("Camera started, but video playback was blocked. Tap Start Camera again.");
          return;
        }
      }

      setStatus("active");

      const mediaPipeReady = await ensureMediaPipeReady();
      if (!mediaPipeReady || requestId !== requestIdRef.current || !streamRef.current) {
        return;
      }

      runInferenceLoop();
    } catch (error) {
      if (requestId !== requestIdRef.current) {
        return;
      }

      setStatus("error");
      setErrorMessage(getCameraErrorMessage(error));
      setMediaPipeStatus("idle");
      setMediaPipeErrorMessage("");
    }
  };

  useEffect(() => {
    isMountedRef.current = true;

    if (autoStart) {
      void startCamera();
    }

    return () => {
      isMountedRef.current = false;
      stopInferenceLoop();
      releaseStream();
      closeHandLandmarker();
      detectorRef.current = null;
    };
  }, [autoStart]);

  const statusMessageByState: Record<CameraStatus, string> = {
    idle: "Camera not started",
    loading: "Starting camera...",
    active: "Camera active",
    error: errorMessage || "Camera is unavailable",
  };

  const mediaPipeStatusMessageByState: Record<MediaPipeStatus, string> = {
    idle: "Idle",
    loading: "Loading...",
    ready: "Ready",
    error: mediaPipeErrorMessage || "MediaPipe could not be initialized.",
  };

  return (
    <section className="w-full max-w-3xl" aria-label="Camera preview area">
      <div className="overflow-hidden rounded-2xl border border-white/10 bg-slate-900">
        <div className="relative aspect-video w-full bg-slate-950">
          <video
            ref={videoRef}
            className={`h-full w-full object-contain ${status === "active" ? "opacity-100" : "opacity-0"}`}
            autoPlay
            playsInline
            muted
          />
          <canvas
            ref={canvasRef}
            width={1280}
            height={720}
            aria-hidden="true"
            className={`pointer-events-none absolute inset-0 h-full w-full ${status === "active" && mediaPipeStatus === "ready" ? "opacity-100" : "opacity-0"}`}
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

      <div className="mt-4 rounded-xl border border-white/10 bg-slate-900/70 p-4 text-sm text-slate-200">
        <p className="font-medium">MediaPipe Status: {mediaPipeStatusMessageByState[mediaPipeStatus]}</p>
        <p className="mt-2">Hands detected: {debugInfo.handsDetected}</p>
        {debugInfo.handsDetected > 0 && (
          <p className="mt-1">Landmarks detected: {debugInfo.landmarkCounts.join(" + ")}</p>
        )}
      </div>

      <div className="mt-5 flex flex-col gap-3 sm:flex-row">
        {status !== "active" ? (
          <PrimaryButton
            onClick={() => void startCamera()}
            disabled={status === "loading" || mediaPipeStatus === "loading"}
            aria-label={status === "error" ? "Try starting camera again" : "Start camera"}
          >
            {status === "error" ? "Try Again" : "Start Camera"}
          </PrimaryButton>
        ) : (
          <PrimaryButton
            onClick={stopCamera}
            className="bg-slate-700 shadow-slate-900/30 hover:bg-slate-600 active:bg-slate-800"
            aria-label="Stop camera"
          >
            Stop Camera
          </PrimaryButton>
        )}
      </div>
    </section>
  );
}