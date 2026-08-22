export type HandLandmark = {
  x: number;
  y: number;
  z: number;
};

export type HandLandmarks = HandLandmark[];

export type HandednessInfo = {
  label: "Left" | "Right" | "Unknown";
  score: number;
};

export type HandDetectionFrame = {
  hands: HandLandmarks[];
  handedness: HandednessInfo[];
  timestampMs: number;
};

export type MediaPipeStatus =
  | "idle"
  | "loading"
  | "ready"
  | "error";