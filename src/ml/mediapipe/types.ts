export type MediaPipeStatus = "idle" | "loading" | "ready" | "error";

export type LandmarkPoint = {
  x: number;
  y: number;
  z: number;
};

export type HandLandmarks = LandmarkPoint[];

export type HandDetectionFrame = {
  hands: HandLandmarks[];
  timestampMs: number;
};