import type {
  Landmark3D,
  NormalizedFrameResult,
  NormalizedHandData,
  NormalizedHandResult,
  RawHandLandmarks,
} from "./types";

export const HAND_LANDMARK_COUNT = 21;
export const FEATURE_VECTOR_LENGTH = HAND_LANDMARK_COUNT * 3;

const WRIST_INDEX = 0;
const MIDDLE_FINGER_MCP_INDEX = 9;
const MIN_SCALE = 1e-6;

function isFiniteNumber(value: number): boolean {
  return Number.isFinite(value);
}

function isValidLandmark(landmark: Landmark3D): boolean {
  return isFiniteNumber(landmark.x) && isFiniteNumber(landmark.y) && isFiniteNumber(landmark.z);
}

function flattenLandmarks(landmarks: Landmark3D[]): number[] {
  return landmarks.flatMap((landmark) => [landmark.x, landmark.y, landmark.z]);
}

function isValidFeatureVector(vector: number[]): boolean {
  if (vector.length !== FEATURE_VECTOR_LENGTH) {
    return false;
  }

  return vector.every((value) => isFiniteNumber(value));
}

function normalizeSingleHand(rawHand: RawHandLandmarks): NormalizedHandResult {
  if (rawHand.length !== HAND_LANDMARK_COUNT) {
    return {
      ok: false,
      error: {
        code: "INVALID_LANDMARK_COUNT",
        message: `Expected ${HAND_LANDMARK_COUNT} landmarks, received ${rawHand.length}.`,
      },
    };
  }

  if (!rawHand.every(isValidLandmark)) {
    return {
      ok: false,
      error: {
        code: "INVALID_COORDINATE",
        message: "Landmark coordinates must be finite numbers.",
      },
    };
  }

  const wrist = rawHand[WRIST_INDEX];
  const middleFingerMcp = rawHand[MIDDLE_FINGER_MCP_INDEX];

  const scaleDistance = Math.hypot(
    middleFingerMcp.x - wrist.x,
    middleFingerMcp.y - wrist.y,
    middleFingerMcp.z - wrist.z,
  );

  if (!isFiniteNumber(scaleDistance) || scaleDistance <= MIN_SCALE) {
    return {
      ok: false,
      error: {
        code: "ZERO_SCALE",
        message: "Could not normalize hand because scale reference is zero.",
      },
    };
  }

  const normalizedLandmarks: Landmark3D[] = rawHand.map((landmark) => ({
    x: (landmark.x - wrist.x) / scaleDistance,
    y: (landmark.y - wrist.y) / scaleDistance,
    z: (landmark.z - wrist.z) / scaleDistance,
  }));

  if (!normalizedLandmarks.every(isValidLandmark)) {
    return {
      ok: false,
      error: {
        code: "INVALID_NORMALIZED_VALUE",
        message: "Normalized coordinates contain invalid numeric values.",
      },
    };
  }

  const featureVector = flattenLandmarks(normalizedLandmarks);

  if (!isValidFeatureVector(featureVector)) {
    return {
      ok: false,
      error: {
        code: "INVALID_FEATURE_VECTOR",
        message: "Feature vector does not contain exactly 63 finite values.",
      },
    };
  }

  const normalizedData: NormalizedHandData = {
    normalizedLandmarks,
    featureVector,
    scale: scaleDistance,
  };

  return {
    ok: true,
    data: normalizedData,
  };
}

export function normalizeFrameHands(rawHands: RawHandLandmarks[]): NormalizedFrameResult {
  return {
    rawHands,
    hands: rawHands.map((hand) => normalizeSingleHand(hand)),
  };
}