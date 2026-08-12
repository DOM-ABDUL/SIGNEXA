import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";
import type { HandDetectionFrame } from "./types";

const WASM_BASE_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm";

const HAND_MODEL_ASSET_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

let detectorPromise: Promise<HandLandmarker> | null = null;
let detectorInstance: HandLandmarker | null = null;
let detectorGeneration = 0;

async function createHandLandmarker(): Promise<HandLandmarker> {
  const vision = await FilesetResolver.forVisionTasks(WASM_BASE_URL);

  return HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: HAND_MODEL_ASSET_URL,
    },
    runningMode: "VIDEO",
    numHands: 2,
  });
}

export async function getHandLandmarker(): Promise<HandLandmarker> {
  if (detectorInstance) {
    return detectorInstance;
  }

  if (!detectorPromise) {
    const currentGeneration = detectorGeneration;

    detectorPromise = createHandLandmarker()
      .then((detector) => {
        if (currentGeneration !== detectorGeneration) {
          detector.close();
          throw new Error("HandLandmarker initialization cancelled.");
        }

        detectorInstance = detector;
        return detector;
      })
      .catch((error) => {
        detectorPromise = null;
        console.error("HandLandmarker initialization failed:", error);
        throw error;
      });
  }

  return detectorPromise;
}

export function detectHandsForVideo(
  detector: HandLandmarker,
  video: HTMLVideoElement,
  timestampMs: number,
): HandDetectionFrame {
  const result = detector.detectForVideo(video, timestampMs);

  return {
    hands: result.landmarks.map((handLandmarks) =>
      handLandmarks.map((point) => ({
        x: point.x,
        y: point.y,
        z: point.z,
      })),
    ),
    timestampMs,
  };
}

export function closeHandLandmarker(): void {
  detectorGeneration += 1;

  if (detectorInstance) {
    detectorInstance.close();
    detectorInstance = null;
  }

  detectorPromise = null;
}