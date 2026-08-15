# SIGNEXA Training Data Pipeline

This directory is for dataset preprocessing and feature extraction experiments. It is separate from the React application in `src/` so the browser app stays focused on camera, MediaPipe, and runtime inference.

## 1. Where Raw Datasets Go

Place externally obtained datasets under:

`training/dataset/`

Do not commit large raw datasets unless the team has confirmed storage, licensing, and privacy rules. Datasets must be obtained from their official sources and used according to their own permissions.

## 2. Supported Input Formats

The pipeline is prepared for these common formats:

- CSV metadata files
- JSON metadata files
- JSONL metadata files
- image folders
- video folders
- landmark files

Only the software foundation exists right now. No external dataset has been downloaded or verified in this repository.

## 3. Validation

Validation utilities live in `src/ml/dataset/validation.ts`. They check for:

- missing sample IDs
- duplicate sample IDs
- missing labels
- missing landmarks
- wrong landmark count
- invalid coordinates such as NaN or Infinity
- incorrect feature vector length
- missing signer IDs when signer-independent evaluation is required

Invalid samples are reported. They are not silently repaired.

## 4. Normalization

The dataset pipeline reuses the same SIGNEXA normalization as the browser pipeline in `src/ml/landmarks/normalize.ts`.

Normalization works like this:

- landmark 0, the wrist, becomes the origin
- every landmark is translated relative to the wrist
- the distance from wrist landmark 0 to middle-finger MCP landmark 9 is used as the hand scale
- translated coordinates are divided by that scale

This consistency is critical. Training features and runtime camera features must use the same preprocessing logic.

## 5. Feature Vector

For one hand:

21 landmarks x 3 coordinates = 63 numerical features

The feature vector order is:

`x0,y0,z0,x1,y1,z1,...,x20,y20,z20`

Metadata is stored separately from numerical features. Metadata includes label, sample ID, dataset source, signer ID if available, hand side if available, and split.

## 6. Two-Hand Data

The current static hand pipeline supports 0, 1, or 2 detected hands.

For now, each hand keeps its own 63-value feature vector. We do not force all samples into a 126-value vector yet.

If handedness is available, the feature extraction code orders hands deterministically as left hand then right hand. If handedness is unavailable, the original order is preserved because the ambiguity should not be hidden.

## 7. Labels And Meaning Layer

Original dataset labels are preserved as `originalLabel`.

An optional future SIGNEXA concept ID can be stored separately as `signexaConceptId`. This prepares for a future meaning layer such as:

ISL sign -> NEED_HELP -> English / Hindi / Bengali / Tamil / other languages

Translation and multilingual output are not implemented in this milestone.

## 8. Signer-Independent Splitting

Signer-aware splitting lives in `src/ml/dataset/split.ts`.

If signer IDs are available, samples can be split so that a signer appears in only one split:

- train
- validation
- test

If signer IDs are missing, the code reports that signer-independent evaluation cannot be guaranteed.

## 9. Processed Outputs

Future processed files should go under:

- `training/processed/` for cleaned intermediate files
- `training/features/` for ML-ready feature files

This milestone does not write dataset files automatically.

## 10. Synthetic Fixture

`training/dataset/fixtures/synthetic_landmarks.json` is a TEST FIXTURE ONLY.

It is not real ISL data and must not be used to train a model. It only exists to verify that 21 deterministic landmarks can become one 63-value normalized feature vector.

## 11. Not Implemented Yet

This milestone does not implement:

- Random Forest
- MLP
- GRU
- MediaPipe Holistic
- dynamic recognition
- sign classification
- dataset downloading
- model training