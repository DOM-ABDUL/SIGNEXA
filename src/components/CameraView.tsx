import { useEffect, useRef, useState } from "react";
import type { HandLandmarker } from "@mediapipe/tasks-vision";

import {
  FEATURE_VECTOR_LENGTH,
  HAND_LANDMARK_COUNT,
  normalizeFrameHands,
} from "../ml/landmarks/normalize";

import {
  INCLUDE10_HAND_FEATURE_DIM,
  INCLUDE10_HAND_MLP_INPUT_DIM,
  temporalPoolInclude10,
  validateInclude10HandFeatureContract,
  vectorizeInclude10Hands,
} from "../ml/landmarks/include10HandFeatures";

import type { NormalizedFrameResult } from "../ml/landmarks/types";

import {
  closeHandLandmarker,
  detectHandsForVideo,
  getHandLandmarker,
} from "../ml/mediapipe/detector";

import type {
  HandLandmarks,
  MediaPipeStatus,
} from "../ml/mediapipe/types";

import { PrimaryButton } from "./PrimaryButton";

type CameraStatus = "idle" | "loading" | "active" | "error";

type CameraViewProps = {
  autoStart?: boolean;
};

type InspectorHand = {
  handIndex: number;
  rawLandmarkCount: number;
  normalizedLandmarkCount: number;
  normalizedFeatureCount: number;
  include10FeatureCount: number;
  featurePreview: number[];
  scale: number | null;
  handedness: string;
  handednessScore: number;
  errorMessage: string | null;
};

type InspectorState = {
  handsDetected: number;
  rawLandmarkCounts: number[];
  hands: InspectorHand[];
};

type FeaturePipelineState = {
  frameCount: number;
  frameDimension: number;
  pooledDimension: number;
  lastFramePreview: number[];
  pooledPreview: number[];
  contractValid: boolean;
  errorMessage: string | null;
};

const INSPECTOR_REFRESH_MS = 250;
const FEATURE_PREVIEW_COUNT = 12;

const BUFFER_MIN_FRAMES = 24;
const BUFFER_MAX_FRAMES = 72;

const EMPTY_NORMALIZED_FRAME: NormalizedFrameResult = {
  rawHands: [],
  hands: [],
};

const EMPTY_INSPECTOR_STATE: InspectorState = {
  handsDetected: 0,
  rawLandmarkCounts: [],
  hands: [],
};

const EMPTY_FEATURE_PIPELINE: FeaturePipelineState = {
  frameCount: 0,
  frameDimension: INCLUDE10_HAND_FEATURE_DIM,
  pooledDimension: INCLUDE10_HAND_MLP_INPUT_DIM,
  lastFramePreview: [],
  pooledPreview: [],
  contractValid: false,
  errorMessage: null,
};

function getCameraErrorMessage(error: unknown): string {
  if (!(error instanceof DOMException)) {
    return "Unable to start the camera right now. Please try again.";
  }

  switch (error.name) {
    case "NotAllowedError":
    case "SecurityError":
      return "Camera permission is required for SIGNEXA. Please allow access and try again.";

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

function getMediaPipeErrorMessage(): string {
  return "MediaPipe could not be initialized.";
}

function formatValue(value: number): string {
  return value.toFixed(4);
}

function formatConfidence(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function drawLandmarksOnCanvas(
  canvas: HTMLCanvasElement,
  hands: HandLandmarks[],
): void {
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

function buildInspectorState(
  frame: NormalizedFrameResult,
  include10Frame: number[],
  handedness: Array<{
    label: "Left" | "Right" | "Unknown";
    score: number;
  }>,
): InspectorState {
  return {
    handsDetected: frame.rawHands.length,

    rawLandmarkCounts: frame.rawHands.map(
      (hand) => hand.length,
    ),

    hands: frame.hands.map((handResult, index) => {
      const rawLandmarkCount =
        frame.rawHands[index]?.length ?? 0;

      const handFeatureOffset =
        handedness[index]?.label === "Right" ? 63 : 0;

      const include10FeaturePreview =
        include10Frame.length === INCLUDE10_HAND_FEATURE_DIM
          ? include10Frame.slice(
              handFeatureOffset,
              handFeatureOffset + FEATURE_PREVIEW_COUNT,
            )
          : [];

      if (!handResult.ok) {
        return {
          handIndex: index + 1,
          rawLandmarkCount,
          normalizedLandmarkCount: 0,
          normalizedFeatureCount: 0,
          include10FeatureCount: 0,
          featurePreview: [],
          scale: null,
          handedness:
            handedness[index]?.label ?? "Unknown",
          handednessScore:
            handedness[index]?.score ?? 0,
          errorMessage: handResult.error.message,
        };
      }

      return {
        handIndex: index + 1,
        rawLandmarkCount,
        normalizedLandmarkCount:
          handResult.data.normalizedLandmarks.length,
        normalizedFeatureCount:
          handResult.data.featureVector.length,
        include10FeatureCount:
          INCLUDE10_HAND_FEATURE_DIM,
        featurePreview: include10FeaturePreview,
        scale: handResult.data.scale,
        handedness:
          handedness[index]?.label ?? "Unknown",
        handednessScore:
          handedness[index]?.score ?? 0,
        errorMessage: null,
      };
    }),
  };
}

function areInspectorStatesEqual(
  a: InspectorState,
  b: InspectorState,
): boolean {
  if (a.handsDetected !== b.handsDetected) {
    return false;
  }

  if (
    a.rawLandmarkCounts.length !==
      b.rawLandmarkCounts.length ||
    a.hands.length !== b.hands.length
  ) {
    return false;
  }

  for (
    let i = 0;
    i < a.rawLandmarkCounts.length;
    i += 1
  ) {
    if (
      a.rawLandmarkCounts[i] !==
      b.rawLandmarkCounts[i]
    ) {
      return false;
    }
  }

  for (let i = 0; i < a.hands.length; i += 1) {
    const first = a.hands[i];
    const second = b.hands[i];

    if (
      first.rawLandmarkCount !==
        second.rawLandmarkCount ||
      first.normalizedLandmarkCount !==
        second.normalizedLandmarkCount ||
      first.normalizedFeatureCount !==
        second.normalizedFeatureCount ||
      first.include10FeatureCount !==
        second.include10FeatureCount ||
      first.scale !== second.scale ||
      first.handedness !== second.handedness ||
      first.handednessScore !==
        second.handednessScore ||
      first.errorMessage !== second.errorMessage
    ) {
      return false;
    }

    if (
      first.featurePreview.length !==
      second.featurePreview.length
    ) {
      return false;
    }

    for (
      let j = 0;
      j < first.featurePreview.length;
      j += 1
    ) {
      if (
        first.featurePreview[j] !==
        second.featurePreview[j]
      ) {
        return false;
      }
    }
  }

  return true;
}

function createFeaturePipelineState(
  frameBuffer: number[][],
  lastFrame: number[],
): FeaturePipelineState {
  if (frameBuffer.length === 0) {
    return {
      ...EMPTY_FEATURE_PIPELINE,
      contractValid:
        lastFrame.length ===
        INCLUDE10_HAND_FEATURE_DIM,
    };
  }

  try {
    const pooled = temporalPoolInclude10(
      frameBuffer,
    );

    return {
      frameCount: frameBuffer.length,
      frameDimension:
        lastFrame.length,
      pooledDimension: pooled.length,
      lastFramePreview: lastFrame.slice(
        0,
        FEATURE_PREVIEW_COUNT,
      ),
      pooledPreview: pooled.slice(
        0,
        FEATURE_PREVIEW_COUNT,
      ),
      contractValid:
        lastFrame.length ===
          INCLUDE10_HAND_FEATURE_DIM &&
        pooled.length ===
          INCLUDE10_HAND_MLP_INPUT_DIM,
      errorMessage: null,
    };
  } catch (error) {
    return {
      frameCount: frameBuffer.length,
      frameDimension: lastFrame.length,
      pooledDimension: 0,
      lastFramePreview: lastFrame.slice(
        0,
        FEATURE_PREVIEW_COUNT,
      ),
      pooledPreview: [],
      contractValid: false,
      errorMessage:
        error instanceof Error
          ? error.message
          : "Temporal pooling failed.",
    };
  }
}

export function CameraView({
  autoStart = false,
}: CameraViewProps) {
  const videoRef =
    useRef<HTMLVideoElement | null>(null);

  const canvasRef =
    useRef<HTMLCanvasElement | null>(null);

  const streamRef =
    useRef<MediaStream | null>(null);

  const detectorRef =
    useRef<HandLandmarker | null>(null);

  const rafIdRef =
    useRef<number | null>(null);

  const isMountedRef =
    useRef(true);

  const requestIdRef =
    useRef(0);

  const lastVideoTimeRef =
    useRef(-1);

  const lastInspectorRefreshRef =
    useRef(0);

  const frameBufferRef =
    useRef<number[][]>([]);

  const lastFrameFeatureRef =
    useRef<number[]>(
      new Array<number>(
        INCLUDE10_HAND_FEATURE_DIM,
      ).fill(0),
    );

  const latestNormalizedFrameRef =
    useRef<NormalizedFrameResult>(
      EMPTY_NORMALIZED_FRAME,
    );

  const latestInspectorStateRef =
    useRef<InspectorState>(
      EMPTY_INSPECTOR_STATE,
    );

  const [status, setStatus] =
    useState<CameraStatus>("idle");

  const [errorMessage, setErrorMessage] =
    useState("");

  const [mediaPipeStatus, setMediaPipeStatus] =
    useState<MediaPipeStatus>("idle");

  const [
    mediaPipeErrorMessage,
    setMediaPipeErrorMessage,
  ] = useState("");

  const [inspectorState, setInspectorState] =
    useState<InspectorState>(
      EMPTY_INSPECTOR_STATE,
    );

  const [featurePipeline, setFeaturePipeline] =
    useState<FeaturePipelineState>(
      EMPTY_FEATURE_PIPELINE,
    );

  const releaseStream = () => {
    const stream = streamRef.current;

    if (stream) {
      stream
        .getTracks()
        .forEach((track) => track.stop());

      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  };

  const stopInferenceLoop = () => {
    if (rafIdRef.current !== null) {
      window.cancelAnimationFrame(
        rafIdRef.current,
      );

      rafIdRef.current = null;
    }

    lastVideoTimeRef.current = -1;

    const canvas = canvasRef.current;
    const context = canvas?.getContext("2d");

    if (canvas && context) {
      context.clearRect(
        0,
        0,
        canvas.width,
        canvas.height,
      );
    }
  };

  const resetDetectionState = () => {
    latestNormalizedFrameRef.current =
      EMPTY_NORMALIZED_FRAME;

    latestInspectorStateRef.current =
      EMPTY_INSPECTOR_STATE;

    frameBufferRef.current = [];

    lastFrameFeatureRef.current =
      new Array<number>(
        INCLUDE10_HAND_FEATURE_DIM,
      ).fill(0);

    setInspectorState(
      EMPTY_INSPECTOR_STATE,
    );

    setFeaturePipeline(
      EMPTY_FEATURE_PIPELINE,
    );
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
      if (
        !isMountedRef.current ||
        !streamRef.current ||
        !videoRef.current ||
        !detectorRef.current
      ) {
        return;
      }

      const video = videoRef.current;

      if (
        video.readyState <
        HTMLMediaElement.HAVE_CURRENT_DATA
      ) {
        rafIdRef.current =
          window.requestAnimationFrame(step);

        return;
      }

      if (
        video.currentTime ===
        lastVideoTimeRef.current
      ) {
        rafIdRef.current =
          window.requestAnimationFrame(step);

        return;
      }

      lastVideoTimeRef.current =
        video.currentTime;

      const frame = detectHandsForVideo(
        detectorRef.current,
        video,
        performance.now(),
      );

      const normalizedFrame =
        normalizeFrameHands(frame.hands);

      latestNormalizedFrameRef.current =
        normalizedFrame;

      /*
       * IMPORTANT:
       *
       * normalizedFrame is retained for the existing
       * landmark inspector and quality diagnostics.
       *
       * The INCLUDE-10 classifier receives the
       * raw MediaPipe x/y/z representation through
       * vectorizeInclude10Hands().
       */
      const include10FrameFeature =
        vectorizeInclude10Hands(
          frame.hands,
          frame.handedness,
        );

      lastFrameFeatureRef.current =
        include10FrameFeature;

      const nextInspectorState =
        buildInspectorState(
          normalizedFrame,
          include10FrameFeature,
          frame.handedness,
        );

      const inspectorChanged =
        !areInspectorStatesEqual(
          latestInspectorStateRef.current,
          nextInspectorState,
        );

      const handsCountChanged =
        latestInspectorStateRef.current
          .handsDetected !==
        nextInspectorState.handsDetected;

      const refreshIntervalReached =
        performance.now() -
          lastInspectorRefreshRef.current >=
        INSPECTOR_REFRESH_MS;

      const shouldUpdateInspector =
        inspectorChanged &&
        (handsCountChanged ||
          refreshIntervalReached);

      if (shouldUpdateInspector) {
        lastInspectorRefreshRef.current =
          performance.now();

        latestInspectorStateRef.current =
          nextInspectorState;

        setInspectorState(
          nextInspectorState,
        );
      }

      /*
       * Add every valid frame to the rolling
       * INCLUDE-10 sequence buffer.
       *
       * We keep the buffer only when at least one
       * hand is detected. A no-hand frame does not
       * contribute to recognition input.
       */
      if (frame.hands.length > 0) {
        frameBufferRef.current.push(
          include10FrameFeature,
        );

        if (
          frameBufferRef.current.length >
          BUFFER_MAX_FRAMES
        ) {
          frameBufferRef.current.shift();
        }

        if (
          frameBufferRef.current.length >=
          BUFFER_MIN_FRAMES
        ) {
          const nextPipelineState =
            createFeaturePipelineState(
              frameBufferRef.current,
              include10FrameFeature,
            );

          setFeaturePipeline(
            nextPipelineState,
          );
        } else {
          setFeaturePipeline({
            frameCount:
              frameBufferRef.current.length,
            frameDimension:
              include10FrameFeature.length,
            pooledDimension: 0,
            lastFramePreview:
              include10FrameFeature.slice(
                0,
                FEATURE_PREVIEW_COUNT,
              ),
            pooledPreview: [],
            contractValid:
              include10FrameFeature.length ===
              INCLUDE10_HAND_FEATURE_DIM,
            errorMessage: null,
          });
        }
      } else {
        /*
         * No hand detected:
         * clear the sequence because we do not want
         * unrelated gaps to become part of a sign.
         */
        frameBufferRef.current = [];

        setFeaturePipeline({
          ...EMPTY_FEATURE_PIPELINE,
          contractValid:
            include10FrameFeature.length ===
            INCLUDE10_HAND_FEATURE_DIM,
          lastFramePreview:
            include10FrameFeature.slice(
              0,
              FEATURE_PREVIEW_COUNT,
            ),
        });
      }

      const canvas = canvasRef.current;

      if (canvas) {
        if (
          canvas.width !== video.videoWidth ||
          canvas.height !== video.videoHeight
        ) {
          canvas.width =
            video.videoWidth;

          canvas.height =
            video.videoHeight;
        }

        drawLandmarksOnCanvas(
          canvas,
          frame.hands,
        );
      }

      rafIdRef.current =
        window.requestAnimationFrame(step);
    };

    stopInferenceLoop();

    rafIdRef.current =
      window.requestAnimationFrame(step);
  };

  const ensureMediaPipeReady =
    async () => {
      if (detectorRef.current) {
        setMediaPipeStatus("ready");
        setMediaPipeErrorMessage("");

        return true;
      }

      setMediaPipeStatus("loading");
      setMediaPipeErrorMessage("");

      try {
        detectorRef.current =
          await getHandLandmarker();

        if (!isMountedRef.current) {
          return false;
        }

        setMediaPipeStatus("ready");

        return true;
      } catch {
        if (!isMountedRef.current) {
          return false;
        }

        setMediaPipeStatus("error");

        setMediaPipeErrorMessage(
          getMediaPipeErrorMessage(),
        );

        return false;
      }
    };

  const startCamera = async () => {
    if (status === "loading") {
      return;
    }

    if (
      !navigator.mediaDevices ||
      !navigator.mediaDevices.getUserMedia
    ) {
      setStatus("error");

      setErrorMessage(
        "This browser does not support camera access.",
      );

      return;
    }

    if (!window.isSecureContext) {
      setStatus("error");

      setErrorMessage(
        "Camera access requires HTTPS or localhost.",
      );

      return;
    }

    const requestId =
      requestIdRef.current + 1;

    requestIdRef.current =
      requestId;

    stopInferenceLoop();
    releaseStream();
    resetDetectionState();

    setMediaPipeStatus("idle");
    setMediaPipeErrorMessage("");

    setStatus("loading");
    setErrorMessage("");

    try {
      const stream =
        await navigator.mediaDevices.getUserMedia(
          {
            audio: false,
            video: {
              facingMode: {
                ideal: "user",
              },
              width: {
                ideal: 1280,
              },
              height: {
                ideal: 720,
              },
            },
          },
        );

      if (
        !isMountedRef.current ||
        requestId !==
          requestIdRef.current
      ) {
        stream
          .getTracks()
          .forEach((track) =>
            track.stop(),
          );

        return;
      }

      streamRef.current = stream;

      stream
        .getVideoTracks()
        .forEach((track) => {
          track.onended = () => {
            if (
              !isMountedRef.current
            ) {
              return;
            }

            stopInferenceLoop();
            releaseStream();

            setStatus("error");

            setErrorMessage(
              "Camera stream ended. Start the camera again.",
            );

            setMediaPipeStatus("idle");
            setMediaPipeErrorMessage("");

            resetDetectionState();
          };
        });

      if (videoRef.current) {
        videoRef.current.srcObject =
          stream;

        try {
          await videoRef.current.play();
        } catch {
          stopInferenceLoop();
          releaseStream();

          setStatus("error");

          setErrorMessage(
            "Camera started, but video playback was blocked. Tap Start Camera again.",
          );

          return;
        }
      }

      const mediaPipeReady =
        await ensureMediaPipeReady();

      if (
        !mediaPipeReady ||
        requestId !==
          requestIdRef.current ||
        !streamRef.current
      ) {
        return;
      }

      setStatus("active");

      runInferenceLoop();
    } catch (error) {
      if (
        requestId !==
        requestIdRef.current
      ) {
        return;
      }

      setStatus("error");

      setErrorMessage(
        getCameraErrorMessage(error),
      );

      setMediaPipeStatus("idle");
      setMediaPipeErrorMessage("");
    }
  };

  useEffect(() => {
    isMountedRef.current = true;

    try {
      validateInclude10HandFeatureContract();
    } catch (error) {
      setFeaturePipeline({
        ...EMPTY_FEATURE_PIPELINE,
        contractValid: false,
        errorMessage:
          error instanceof Error
            ? error.message
            : "INCLUDE-10 feature contract validation failed.",
      });
    }

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

  const statusMessageByState: Record<
    CameraStatus,
    string
  > = {
    idle: "Camera not started",
    loading: "Starting camera...",
    active: "Camera active",
    error:
      errorMessage ||
      "Camera is unavailable",
  };

  const mediaPipeStatusMessageByState: Record<
    MediaPipeStatus,
    string
  > = {
    idle: "Idle",
    loading: "Loading...",
    ready: "Ready",
    error:
      mediaPipeErrorMessage ||
      "MediaPipe could not be initialized.",
  };

  const featureContractMessage =
    featurePipeline.contractValid
      ? "VALID"
      : featurePipeline.errorMessage ||
        "Waiting for a valid feature frame";

  return (
    <section
      className="w-full max-w-3xl"
      aria-label="Camera preview area"
    >
      <div className="overflow-hidden rounded-2xl border border-white/10 bg-slate-900">
        <div className="relative aspect-video w-full bg-slate-950">
          <video
            ref={videoRef}
            className={`h-full w-full object-contain ${
              status === "active"
                ? "opacity-100"
                : "opacity-0"
            }`}
            autoPlay
            playsInline
            muted
          />

          <canvas
            ref={canvasRef}
            width={1280}
            height={720}
            aria-hidden="true"
            className={`pointer-events-none absolute inset-0 h-full w-full ${
              status === "active" &&
              mediaPipeStatus === "ready"
                ? "opacity-100"
                : "opacity-0"
            }`}
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
        className={`mt-4 text-sm font-medium ${
          status === "error"
            ? "text-red-300"
            : "text-slate-200"
        }`}
      >
        {statusMessageByState[status]}
      </p>

      <div className="mt-4 rounded-xl border border-white/10 bg-slate-900/70 p-4 text-sm text-slate-200">
        <p className="font-medium">
          MediaPipe Status:{" "}
          {
            mediaPipeStatusMessageByState[
              mediaPipeStatus
            ]
          }
        </p>

        <p className="mt-1">
          Hands detected:{" "}
          {inspectorState.handsDetected}
        </p>

        {inspectorState.handsDetected >
          0 && (
          <p className="mt-1">
            Raw landmarks:{" "}
            {inspectorState.rawLandmarkCounts.join(
              " + ",
            )}
          </p>
        )}

        <p className="mt-1">
          INCLUDE-10 frame contract:{" "}
          {INCLUDE10_HAND_FEATURE_DIM} values
          {" "}
          (left 63 + right 63)
        </p>

        <p className="mt-1">
          Current frame dimension:{" "}
          {featurePipeline.frameDimension}
        </p>

        <p className="mt-1">
          Temporal buffer:{" "}
          {featurePipeline.frameCount} /{" "}
          {BUFFER_MAX_FRAMES} frames
        </p>

        <p className="mt-1">
          MLP input after pooling:{" "}
          {INCLUDE10_HAND_MLP_INPUT_DIM} values
        </p>

        <p className="mt-1">
          Feature contract:{" "}
          <span
            className={
              featurePipeline.contractValid
                ? "font-semibold text-emerald-300"
                : "font-semibold text-amber-300"
            }
          >
            {featureContractMessage}
          </span>
        </p>
      </div>

      <div className="mt-4 rounded-xl border border-cyan-400/30 bg-cyan-950/20 p-4 text-sm text-cyan-100">
        <p className="font-medium">
          INCLUDE-10 Feature Pipeline
        </p>

        <p className="mt-1">
          Frame:{" "}
          {featurePipeline.frameDimension}D
        </p>

        <p className="mt-1">
          Pooling: mean + std + min + max
        </p>

        <p className="mt-1">
          Pooled representation:{" "}
          {featurePipeline.pooledDimension}D
        </p>

        <p className="mt-2 break-words text-cyan-200">
          Frame preview: [
          {featurePipeline.lastFramePreview
            .map(formatValue)
            .join(", ")}
          ]
          {featurePipeline.lastFramePreview
            .length > 0
            ? " ..."
            : ""}
        </p>

        {featurePipeline.pooledPreview
          .length > 0 && (
          <p className="mt-2 break-words text-cyan-200">
            Pooled preview: [
            {featurePipeline.pooledPreview
              .map(formatValue)
              .join(", ")}
            ] ...
          </p>
        )}

        {featurePipeline.errorMessage && (
          <p className="mt-2 text-red-300">
            Pipeline error:{" "}
            {featurePipeline.errorMessage}
          </p>
        )}
      </div>

      <details className="mt-4 rounded-xl border border-white/10 bg-slate-900/70 p-4 text-sm text-slate-200">
        <summary className="cursor-pointer font-medium">
          Landmark Data Inspector
        </summary>

        <div className="mt-3 space-y-3">
          <p>
            Expected shape per hand:{" "}
            {HAND_LANDMARK_COUNT} landmarks,{" "}
            {FEATURE_VECTOR_LENGTH} normalized
            feature values.
          </p>

          <p>
            INCLUDE-10 frame vector:{" "}
            {INCLUDE10_HAND_FEATURE_DIM} values,
            ordered as left[63] then right[63].
          </p>

          <p>
            Raw hand buffers in memory:{" "}
            {
              latestNormalizedFrameRef.current
                .rawHands.length
            }
          </p>

          <p className="break-words">
            Current INCLUDE-10 frame preview: [
            {lastFrameFeatureRef.current
              .slice(
                0,
                FEATURE_PREVIEW_COUNT,
              )
              .map(formatValue)
              .join(", ")}
            ] ...
          </p>

          {inspectorState.hands.length ===
            0 && (
            <p>
              No hand feature vector is
              available.
            </p>
          )}

          {inspectorState.hands.map(
            (hand) => (
              <div
                key={hand.handIndex}
                className="rounded-lg border border-white/10 bg-slate-950/60 p-3"
              >
                <p className="font-medium">
                  Hand {hand.handIndex}
                </p>

                <p className="mt-1">
                  Raw landmarks:{" "}
                  {hand.rawLandmarkCount}
                </p>

                <p className="mt-1">
                  Handedness:{" "}
                  {hand.handedness}{" "}
                  ({formatConfidence(
                    hand.handednessScore,
                  )})
                </p>

                {hand.errorMessage ? (
                  <p className="mt-1 text-red-300">
                    Normalization error:{" "}
                    {hand.errorMessage}
                  </p>
                ) : (
                  <>
                    <p className="mt-1">
                      Normalized landmarks:{" "}
                      {
                        hand.normalizedLandmarkCount
                      }
                    </p>

                    <p className="mt-1">
                      Normalized feature vector:{" "}
                      {
                        hand.normalizedFeatureCount
                      }{" "}
                      values
                    </p>

                    <p className="mt-1">
                      INCLUDE-10 contribution:{" "}
                      {
                        hand.include10FeatureCount
                      }{" "}
                      total frame contract
                    </p>

                    {hand.scale !== null && (
                      <p className="mt-1">
                        Scale reference:{" "}
                        {formatValue(
                          hand.scale,
                        )}
                      </p>
                    )}

                    <p className="mt-1 break-words text-slate-300">
                      INCLUDE-10 feature
                      preview: [
                      {hand.featurePreview
                        .map(formatValue)
                        .join(", ")}
                      ]
                    </p>
                  </>
                )}
              </div>
            ),
          )}
        </div>
      </details>

      <div className="mt-5 flex flex-col gap-3 sm:flex-row">
        {status !== "active" ? (
          <PrimaryButton
            onClick={() =>
              void startCamera()
            }
            disabled={
              status === "loading" ||
              mediaPipeStatus ===
                "loading"
            }
            aria-label={
              status === "error"
                ? "Try starting camera again"
                : "Start camera"
            }
          >
            {status === "error"
              ? "Try Again"
              : "Start Camera"}
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