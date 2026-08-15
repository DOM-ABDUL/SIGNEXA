export type Landmark3D = {
  x: number;
  y: number;
  z: number;
};

export type RawHandLandmarks = Landmark3D[];

export type NormalizedLandmark = Landmark3D;

export type FeatureVector = number[];

export type NormalizedHandData = {
  normalizedLandmarks: NormalizedLandmark[];
  featureVector: FeatureVector;
  scale: number;
};

export type NormalizationErrorCode =
  | "INVALID_LANDMARK_COUNT"
  | "INVALID_COORDINATE"
  | "ZERO_SCALE"
  | "INVALID_NORMALIZED_VALUE"
  | "INVALID_FEATURE_VECTOR";

export type NormalizationError = {
  code: NormalizationErrorCode;
  message: string;
};

export type NormalizedHandResult =
  | {
      ok: true;
      data: NormalizedHandData;
    }
  | {
      ok: false;
      error: NormalizationError;
    };

export type NormalizedFrameResult = {
  rawHands: RawHandLandmarks[];
  hands: NormalizedHandResult[];
};