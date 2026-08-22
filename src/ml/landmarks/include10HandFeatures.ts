import type { HandLandmarks, HandednessInfo } from "../mediapipe/types";

export const LANDMARKS_PER_HAND = 21;
export const COORDS_PER_LANDMARK = 3;

/**
 * One MediaPipe hand:
 * 21 landmarks × (x, y, z) = 63 values.
 */
export const SINGLE_HAND_FEATURE_DIM =
  LANDMARKS_PER_HAND * COORDS_PER_LANDMARK;

/**
 * INCLUDE-10 hand feature contract:
 *
 * left hand  = 63
 * right hand = 63
 * total      = 126
 *
 * IMPORTANT:
 * The current INCLUDE-10 training extractor uses the raw
 * MediaPipe x/y/z coordinates. Do NOT apply the existing
 * normalize.ts transformation here, otherwise browser
 * inference would not match the trained model.
 */
export const INCLUDE10_HAND_FEATURE_DIM =
  SINGLE_HAND_FEATURE_DIM * 2;

/**
 * MLP input:
 *
 * 126-D frame
 * × 4 temporal statistics
 *
 * mean + std + min + max
 *
 * = 504-D
 */
export const INCLUDE10_HAND_MLP_INPUT_DIM =
  INCLUDE10_HAND_FEATURE_DIM * 4;

function zeroHand(): number[] {
  return new Array<number>(SINGLE_HAND_FEATURE_DIM).fill(0);
}

function flattenHand(landmarks: HandLandmarks): number[] {
  const output = zeroHand();

  if (landmarks.length !== LANDMARKS_PER_HAND) {
    return output;
  }

  for (let index = 0; index < LANDMARKS_PER_HAND; index += 1) {
    const landmark = landmarks[index];
    const offset = index * COORDS_PER_LANDMARK;

    output[offset] = Number.isFinite(landmark.x) ? landmark.x : 0;
    output[offset + 1] = Number.isFinite(landmark.y) ? landmark.y : 0;
    output[offset + 2] = Number.isFinite(landmark.z) ? landmark.z : 0;
  }

  return output;
}

/**
 * Convert MediaPipe's detected hands into the exact 126-D
 * INCLUDE-10 hand feature representation.
 *
 * Layout:
 *
 * [ left hand 63 values ][ right hand 63 values ]
 *
 * Missing hands are represented by zeros.
 *
 * This intentionally does NOT use normalize.ts.
 * The current INCLUDE-10 training extractor uses raw
 * MediaPipe x/y/z coordinates.
 */
export function vectorizeInclude10Hands(
  hands: HandLandmarks[],
  handedness: HandednessInfo[],
): number[] {
  let left = zeroHand();
  let right = zeroHand();

  let unknownHandAssigned = false;

  for (let index = 0; index < hands.length; index += 1) {
    const hand = hands[index];
    const side = handedness[index]?.label ?? "Unknown";

    const flattened = flattenHand(hand);

    if (side === "Left") {
      left = flattened;
      continue;
    }

    if (side === "Right") {
      right = flattened;
      continue;
    }

    /*
     * Defensive fallback for an unknown handedness result.
     *
     * If MediaPipe does not provide a usable Left/Right label,
     * assign the first unknown hand to the left slot and the
     * second unknown hand to the right slot.
     */
    if (!unknownHandAssigned) {
      left = flattened;
      unknownHandAssigned = true;
    } else {
      right = flattened;
    }
  }

  return [...left, ...right];
}

/**
 * Calculate temporal mean for each feature dimension.
 */
function calculateMean(features: number[][], dimension: number): number[] {
  const mean = new Array<number>(dimension).fill(0);

  for (const frame of features) {
    for (let index = 0; index < dimension; index += 1) {
      mean[index] += frame[index];
    }
  }

  for (let index = 0; index < dimension; index += 1) {
    mean[index] /= features.length;
  }

  return mean;
}

/**
 * Calculate population standard deviation for each feature dimension.
 *
 * This matches the training-side temporal pooling:
 *
 * sqrt(sum((x - mean)^2) / number_of_frames)
 */
function calculateStd(
  features: number[][],
  mean: number[],
  dimension: number,
): number[] {
  const variance = new Array<number>(dimension).fill(0);

  for (const frame of features) {
    for (let index = 0; index < dimension; index += 1) {
      const delta = frame[index] - mean[index];
      variance[index] += delta * delta;
    }
  }

  return variance.map((value) =>
    Math.sqrt(value / features.length),
  );
}

/**
 * Calculate temporal minimum for each feature dimension.
 */
function calculateMin(
  features: number[][],
  dimension: number,
): number[] {
  const min = new Array<number>(dimension).fill(
    Number.POSITIVE_INFINITY,
  );

  for (const frame of features) {
    for (let index = 0; index < dimension; index += 1) {
      if (frame[index] < min[index]) {
        min[index] = frame[index];
      }
    }
  }

  return min;
}

/**
 * Calculate temporal maximum for each feature dimension.
 */
function calculateMax(
  features: number[][],
  dimension: number,
): number[] {
  const max = new Array<number>(dimension).fill(
    Number.NEGATIVE_INFINITY,
  );

  for (const frame of features) {
    for (let index = 0; index < dimension; index += 1) {
      if (frame[index] > max[index]) {
        max[index] = frame[index];
      }
    }
  }

  return max;
}

/**
 * Convert a sequence of 126-D hand feature frames into
 * the 504-D input expected by the INCLUDE-10 MLP.
 *
 * Ordering:
 *
 * [mean 126]
 * [std  126]
 * [min  126]
 * [max  126]
 *
 * Total = 504.
 */
export function temporalPoolInclude10(
  features: number[][],
): number[] {
  if (features.length === 0) {
    throw new Error(
      "Cannot perform temporal pooling on an empty feature sequence.",
    );
  }

  const dimension = features[0].length;

  if (dimension !== INCLUDE10_HAND_FEATURE_DIM) {
    throw new Error(
      `Frame feature dimension mismatch. ` +
        `Expected ${INCLUDE10_HAND_FEATURE_DIM}, got ${dimension}.`,
    );
  }

  for (let frameIndex = 0; frameIndex < features.length; frameIndex += 1) {
    const frame = features[frameIndex];

    if (frame.length !== dimension) {
      throw new Error(
        `Inconsistent feature dimension at frame ${frameIndex}. ` +
          `Expected ${dimension}, got ${frame.length}.`,
      );
    }

    if (!frame.every(Number.isFinite)) {
      throw new Error(
        `Non-finite value detected in feature frame ${frameIndex}.`,
      );
    }
  }

  const mean = calculateMean(features, dimension);
  const std = calculateStd(features, mean, dimension);
  const min = calculateMin(features, dimension);
  const max = calculateMax(features, dimension);

  const pooled = [
    ...mean,
    ...std,
    ...min,
    ...max,
  ];

  if (pooled.length !== INCLUDE10_HAND_MLP_INPUT_DIM) {
    throw new Error(
      `Temporal pooling produced ${pooled.length} values. ` +
        `Expected ${INCLUDE10_HAND_MLP_INPUT_DIM}.`,
    );
  }

  return pooled;
}

/**
 * Development-time validation of the INCLUDE-10 hand feature contract.
 *
 * This does NOT validate model accuracy.
 * It only verifies that the browser feature construction matches
 * the expected dimensional and ordering contract.
 */
export function validateInclude10HandFeatureContract(): {
  checks: string[];
} {
  const makeHand = (base: number): HandLandmarks =>
    Array.from(
      { length: LANDMARKS_PER_HAND },
      (_, index) => ({
        x: base + index,
        y: base + index + 0.1,
        z: base + index + 0.2,
      }),
    );

  const leftHand = makeHand(1);
  const rightHand = makeHand(101);

  /*
   * Test 1:
   * Both hands must produce exactly 126 values.
   */
  const bothHands = vectorizeInclude10Hands(
    [leftHand, rightHand],
    [
      {
        label: "Left",
        score: 0.9,
      },
      {
        label: "Right",
        score: 0.9,
      },
    ],
  );

  if (bothHands.length !== INCLUDE10_HAND_FEATURE_DIM) {
    throw new Error(
      `Both-hands vector must contain ${INCLUDE10_HAND_FEATURE_DIM} values.`,
    );
  }

  /*
   * Test 2:
   * Left hand occupies the first 63 values.
   */
  if (
    bothHands[0] !== 1 ||
    bothHands[1] !== 1.1 ||
    bothHands[2] !== 1.2
  ) {
    throw new Error(
      "Left-hand feature ordering is incorrect.",
    );
  }

  /*
   * Last landmark of left hand:
   * index 20 × 3 = 60.
   */
  if (
    bothHands[60] !== 21 ||
    bothHands[61] !== 21.1 ||
    bothHands[62] !== 21.2
  ) {
    throw new Error(
      "Left-hand final landmark ordering is incorrect.",
    );
  }

  /*
   * Test 3:
   * Right hand occupies the final 63 values.
   */
  if (
    bothHands[63] !== 101 ||
    bothHands[64] !== 101.1 ||
    bothHands[65] !== 101.2
  ) {
    throw new Error(
      "Right-hand feature ordering is incorrect.",
    );
  }

  /*
   * Last landmark of right hand.
   */
  if (
    bothHands[123] !== 121 ||
    bothHands[124] !== 121.1 ||
    bothHands[125] !== 121.2
  ) {
    throw new Error(
      "Right-hand final landmark ordering is incorrect.",
    );
  }

  /*
   * Test 4:
   * Left-only input must zero-fill the right half.
   */
  const leftOnly = vectorizeInclude10Hands(
    [leftHand],
    [
      {
        label: "Left",
        score: 0.9,
      },
    ],
  );

  if (leftOnly.length !== INCLUDE10_HAND_FEATURE_DIM) {
    throw new Error(
      "Left-only vector must contain 126 values.",
    );
  }

  if (!leftOnly.slice(63).every((value) => value === 0)) {
    throw new Error(
      "Missing right hand must be zero-filled.",
    );
  }

  /*
   * Test 5:
   * Right-only input must zero-fill the left half.
   */
  const rightOnly = vectorizeInclude10Hands(
    [rightHand],
    [
      {
        label: "Right",
        score: 0.9,
      },
    ],
  );

  if (rightOnly.length !== INCLUDE10_HAND_FEATURE_DIM) {
    throw new Error(
      "Right-only vector must contain 126 values.",
    );
  }

  if (!rightOnly.slice(0, 63).every((value) => value === 0)) {
    throw new Error(
      "Missing left hand must be zero-filled.",
    );
  }

  /*
   * Test 6:
   * No hands must produce 126 zeros.
   */
  const noHands = vectorizeInclude10Hands([], []);

  if (
    noHands.length !== INCLUDE10_HAND_FEATURE_DIM ||
    !noHands.every((value) => value === 0)
  ) {
    throw new Error(
      "No-hand vector must contain exactly 126 zeros.",
    );
  }

  /*
   * Test 7:
   * Temporal pooling must produce exactly 504 values.
   */
  const pooled = temporalPoolInclude10([
    bothHands,
    leftOnly,
    rightOnly,
  ]);

  if (pooled.length !== INCLUDE10_HAND_MLP_INPUT_DIM) {
    throw new Error(
      `Temporal pooling must produce ${INCLUDE10_HAND_MLP_INPUT_DIM} values.`,
    );
  }

  /*
   * Test 8:
   * Ensure the pooled result contains only finite values.
   */
  if (!pooled.every(Number.isFinite)) {
    throw new Error(
      "Temporal pooled features contain non-finite values.",
    );
  }

  return {
    checks: [
      "both hands => 126 values",
      "left hand => first 63 values",
      "right hand => last 63 values",
      "left-only => right half zero-filled",
      "right-only => left half zero-filled",
      "no hands => 126 zeros",
      "temporal pooling => 504 values",
      "pooled values are finite",
    ],
  };
}