# SIGNEXA Dataset Strategy

This file defines the locked dataset stack for the current SIGNEXA phase.

Only these three resources are in scope:

1. INCLUDE
Role: PRIMARY STATIC TRAINING DATASET
Purpose: first real dataset to process for isolated ISL recognition, feature extraction, and later static model baselines.

2. SIGN DICTIONARY DATASET
Role: VOCABULARY EXPANSION DATASET
Purpose: later vocabulary expansion beyond INCLUDE.

3. ISL-CSLTR
Role: FUTURE DYNAMIC / CONTINUOUS SIGNING DATASET
Purpose: later temporal sequence work, future Holistic pipeline, and later GRU-based continuous recognition.

No additional datasets are part of this stack.

## Dataset Progression

INCLUDE
-> Inspect actual dataset structure
-> Build INCLUDE adapter
-> Extract landmarks
-> SIGNEXA normalization
-> Feature dataset
-> Data quality validation
-> Train/validation/test split
-> Random Forest baseline
-> MLP comparison

Later expansion:

SIGN DICTIONARY DATASET
-> Vocabulary expansion

Future dynamic stage:

ISL-CSLTR
-> Holistic
-> Temporal features
-> GRU
-> Continuous signing

## Vocabulary Clarification

Initial sanity-check signs:

- HELLO
- HELP
- WATER
- STOP
- CALL

These are engineering checks only. They are not the final SIGNEXA training vocabulary.

INCLUDE is the first real training vocabulary source.

SIGNEXA V1 is expected to eventually target roughly 500 to 1,000 high-value concepts.

Do not hard-code class counts such as 5, 25, or 263.