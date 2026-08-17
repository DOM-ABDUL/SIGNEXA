#!/usr/bin/env python3
"""SIGNEXA GRU baseline training and evaluation.

This script builds temporal video sequences from frame-level normalized
landmark features and trains a lightweight GRU classifier.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

try:
    import tensorflow as tf
except ImportError as exc:  # pragma: no cover
    print("[ERROR] Missing dependency: tensorflow")
    print("Install it with: python -m pip install tensorflow")
    raise SystemExit(1) from exc


FEATURE_COUNT = 63
EXPECTED_CLASSES = 5
EXPECTED_SPLITS = ("train", "val", "test")
FEATURE_COLUMNS = [f"feature_{index}" for index in range(FEATURE_COUNT)]

DEFAULT_BATCH_SIZE = 16
DEFAULT_EPOCHS = 100
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_DROPOUT_RATE = 0.2
DEFAULT_EARLY_STOPPING_PATIENCE = 12
DEFAULT_RANDOM_SEED = 42
PAD_VALUE = 9999.0
GRU_UNITS = 64
HIDDEN_UNITS = 32


@dataclass
class FrameSample:
    split: str
    label: str
    video_path: str
    frame_index: int
    features: list[float]


@dataclass
class SequenceSample:
    split: str
    label: str
    video_path: str
    sequence: list[list[float]]


class SplitEvaluator:
    """Computes video-level metrics from predicted class labels."""

    def __init__(self, labels: list[str]) -> None:
        self.labels = labels
        self.total = 0
        self.correct = 0
        self.confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def update(self, true_label: str, predicted_label: str | None) -> None:
        predicted = predicted_label if predicted_label is not None else "__NONE__"
        self.total += 1
        if predicted == true_label:
            self.correct += 1
        self.confusion[true_label][predicted] += 1

    def accuracy(self) -> float:
        return (self.correct / self.total) if self.total > 0 else 0.0

    def per_class_metrics(self) -> dict[str, dict[str, float | int]]:
        metrics: dict[str, dict[str, float | int]] = {}

        for label in self.labels:
            tp = self.confusion[label].get(label, 0)
            fp = sum(self.confusion[other].get(label, 0) for other in self.labels if other != label)
            fn = sum(count for pred, count in self.confusion[label].items() if pred != label)
            support = sum(self.confusion[label].values())

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

            metrics[label] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }

        return metrics

    def macro_scores(self) -> dict[str, float]:
        per_class = self.per_class_metrics()
        if not per_class:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        precision = sum(float(item["precision"]) for item in per_class.values()) / len(per_class)
        recall = sum(float(item["recall"]) for item in per_class.values()) / len(per_class)
        f1 = sum(float(item["f1"]) for item in per_class.values()) / len(per_class)
        return {"precision": precision, "recall": recall, "f1": f1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SIGNEXA GRU baseline from normalized features")
    parser.add_argument("feature_csv_path", type=Path, help="Path to normalized_features.csv")
    return parser.parse_args()


def fail(message: str) -> None:
    raise ValueError(message)


def is_finite(value: float) -> bool:
    return math.isfinite(value)


def set_reproducibility(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def validate_columns(fieldnames: list[str] | None) -> None:
    if not fieldnames:
        fail("CSV has no header row")

    duplicate_columns = [name for name, count in Counter(fieldnames).items() if count > 1]
    if duplicate_columns:
        fail(f"CSV contains duplicate column names: {duplicate_columns}")

    feature_like_columns = [column for column in fieldnames if column.startswith("feature_")]
    malformed_feature_columns = [column for column in feature_like_columns if not column[8:].isdigit()]
    if malformed_feature_columns:
        fail(f"Malformed feature column names: {malformed_feature_columns}")

    discovered_feature_set = set(feature_like_columns)
    expected_feature_set = set(FEATURE_COLUMNS)

    missing = sorted(expected_feature_set - discovered_feature_set)
    if missing:
        fail(f"Missing feature columns: {missing}")

    unexpected = sorted(discovered_feature_set - expected_feature_set)
    if unexpected:
        fail(f"Unexpected feature columns: {unexpected}")

    if len(discovered_feature_set) != FEATURE_COUNT:
        fail(
            "Feature column set size mismatch. "
            f"Expected {FEATURE_COUNT}, found {len(discovered_feature_set)}"
        )


def load_frame_samples(csv_path: Path) -> tuple[list[FrameSample], dict[str, Any]]:
    if not csv_path.exists() or not csv_path.is_file():
        fail(f"Feature CSV not found: {csv_path}")

    frame_samples: list[FrameSample] = []
    labels: set[str] = set()
    split_counts: dict[str, int] = defaultdict(int)
    video_to_splits: dict[str, set[str]] = defaultdict(set)

    with csv_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        validate_columns(reader.fieldnames)

        for row_index, row in enumerate(reader):
            split = (row.get("split") or "").strip()
            label = (row.get("class_name") or "").strip()
            video_path = (row.get("video_path") or "").strip()
            raw_frame_index = (row.get("frame_index") or "").strip()

            if split not in EXPECTED_SPLITS:
                fail(f"Row {row_index} has invalid split: {split}")
            if not label:
                fail(f"Row {row_index} missing class_name")
            if not video_path:
                fail(f"Row {row_index} missing video_path")
            if raw_frame_index == "":
                fail(f"Row {row_index} missing frame_index")

            try:
                frame_index = int(float(raw_frame_index))
            except ValueError as exc:
                fail(f"Row {row_index} has invalid frame_index: {raw_frame_index}")
                raise exc

            features: list[float] = []
            for column in FEATURE_COLUMNS:
                raw_value = row.get(column)
                if raw_value is None or raw_value == "":
                    fail(f"Row {row_index} missing value for {column}")

                try:
                    value = float(raw_value)
                except ValueError as exc:
                    fail(f"Row {row_index} has non-numeric value for {column}: {raw_value}")
                    raise exc

                if not is_finite(value):
                    fail(f"Row {row_index} has non-finite value for {column}: {value}")

                features.append(value)

            if len(features) != FEATURE_COUNT:
                fail(f"Row {row_index} feature length mismatch: {len(features)}")

            frame_samples.append(
                FrameSample(
                    split=split,
                    label=label,
                    video_path=video_path,
                    frame_index=frame_index,
                    features=features,
                )
            )

            labels.add(label)
            split_counts[split] += 1
            video_to_splits[video_path].add(split)

    if not frame_samples:
        fail("Feature CSV contains no rows")

    if len(labels) != EXPECTED_CLASSES:
        fail(f"Expected exactly {EXPECTED_CLASSES} classes, found {len(labels)}: {sorted(labels)}")

    for split in EXPECTED_SPLITS:
        if split_counts.get(split, 0) == 0:
            fail(f"Split '{split}' has no rows")

    mixed_split_videos = {
        video_path: sorted(list(splits))
        for video_path, splits in video_to_splits.items()
        if len(splits) > 1
    }
    if mixed_split_videos:
        fail(f"Found videos in multiple splits: {mixed_split_videos}")

    metadata = {
        "frame_counts_by_split": {split: split_counts[split] for split in EXPECTED_SPLITS},
        "class_labels": sorted(labels),
    }

    return frame_samples, metadata


def build_video_sequences(frame_samples: list[FrameSample]) -> tuple[list[SequenceSample], list[dict[str, str]]]:
    grouped: dict[str, list[FrameSample]] = defaultdict(list)
    for sample in frame_samples:
        grouped[sample.video_path].append(sample)

    sequences: list[SequenceSample] = []
    rejected: list[dict[str, str]] = []

    for video_path, video_rows in sorted(grouped.items()):
        splits = {row.split for row in video_rows}
        labels = {row.label for row in video_rows}

        if len(splits) != 1:
            fail(f"Video {video_path} appears in multiple splits: {sorted(splits)}")
        if len(labels) != 1:
            fail(f"Video {video_path} has multiple class labels: {sorted(labels)}")

        split = next(iter(splits))
        label = next(iter(labels))

        sorted_rows = sorted(video_rows, key=lambda row: row.frame_index)
        if not sorted_rows:
            rejected.append({"video_path": video_path, "reason": "No frames after grouping"})
            continue

        frame_indices = [row.frame_index for row in sorted_rows]
        if frame_indices != sorted(frame_indices):
            fail(f"Frame ordering failure for video {video_path}")

        sequence = [row.features for row in sorted_rows]
        if len(sequence) == 0:
            rejected.append({"video_path": video_path, "reason": "Sequence has zero valid feature rows"})
            continue

        sequences.append(
            SequenceSample(
                split=split,
                label=label,
                video_path=video_path,
                sequence=sequence,
            )
        )

    return sequences, rejected


def get_sequence_length_stats(sequences: list[SequenceSample]) -> dict[str, float | int]:
    lengths = [len(item.sequence) for item in sequences]
    if not lengths:
        return {"min": 0, "max": 0, "median": 0.0, "mean": 0.0}

    return {
        "min": min(lengths),
        "max": max(lengths),
        "median": float(statistics.median(lengths)),
        "mean": float(sum(lengths) / len(lengths)),
    }


def split_sequences(sequences: list[SequenceSample]) -> dict[str, list[SequenceSample]]:
    grouped = {split: [] for split in EXPECTED_SPLITS}
    for sample in sequences:
        grouped[sample.split].append(sample)

    for split in EXPECTED_SPLITS:
        if len(grouped[split]) == 0:
            fail(f"No video sequences in split '{split}'")

    return grouped


def truncate_sequence_evenly(sequence: list[list[float]], target_length: int) -> list[list[float]]:
    """Deterministically keep evenly spaced timesteps across the full sequence."""
    if target_length <= 0:
        fail("target_length must be positive")

    if len(sequence) <= target_length:
        return sequence

    selected_indices = np.linspace(0, len(sequence) - 1, num=target_length, dtype=int)
    return [sequence[int(index)] for index in selected_indices.tolist()]


def pad_sequences_to_numpy(
    sequences: list[SequenceSample],
    max_sequence_length: int,
) -> tuple[np.ndarray, list[str], list[str], int]:
    if max_sequence_length <= 0:
        fail("max_sequence_length must be positive")

    padded = np.full((len(sequences), max_sequence_length, FEATURE_COUNT), PAD_VALUE, dtype=np.float32)
    labels: list[str] = []
    video_paths: list[str] = []
    truncated_count = 0

    for sequence_index, sample in enumerate(sequences):
        sequence_values = sample.sequence
        if len(sequence_values) > max_sequence_length:
            sequence_values = truncate_sequence_evenly(sequence_values, max_sequence_length)
            truncated_count += 1

        sequence_length = len(sequence_values)
        if sequence_length <= 0:
            fail(f"Video {sample.video_path} has empty sequence")

        sequence_array = np.array(sequence_values, dtype=np.float32)
        padded[sequence_index, :sequence_length, :] = sequence_array
        labels.append(sample.label)
        video_paths.append(sample.video_path)

    return padded, labels, video_paths, truncated_count


def to_label_indices(labels: list[str], class_labels: list[str]) -> np.ndarray:
    label_to_index = {label: idx for idx, label in enumerate(class_labels)}
    return np.array([label_to_index[label] for label in labels], dtype=np.int64)


def compute_class_weights(train_labels: list[str], class_labels: list[str]) -> dict[int, float]:
    counts = Counter(train_labels)
    total = float(len(train_labels))
    class_count = len(class_labels)

    weights: dict[int, float] = {}
    for index, label in enumerate(class_labels):
        count = float(counts.get(label, 0))
        if count <= 0:
            weights[index] = 0.0
        else:
            weights[index] = total / (class_count * count)
    return weights


def build_model(class_count: int, max_sequence_length: int) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(max_sequence_length, FEATURE_COUNT)),
            # Masking lets GRU ignore right-padded timesteps.
            tf.keras.layers.Masking(mask_value=PAD_VALUE),
            tf.keras.layers.GRU(GRU_UNITS),
            tf.keras.layers.Dense(HIDDEN_UNITS, activation="relu"),
            tf.keras.layers.Dropout(DEFAULT_DROPOUT_RATE),
            tf.keras.layers.Dense(class_count, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=DEFAULT_LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def evaluate_predictions(true_labels: list[str], predicted_indices: np.ndarray, class_labels: list[str]) -> SplitEvaluator:
    evaluator = SplitEvaluator(class_labels)
    index_to_label = {index: label for index, label in enumerate(class_labels)}

    for true_label, predicted_index in zip(true_labels, predicted_indices.tolist()):
        evaluator.update(true_label, index_to_label.get(int(predicted_index)))

    return evaluator


def write_training_history(path: Path, history: tf.keras.callbacks.History) -> None:
    keys = list(history.history.keys())
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["epoch", *keys])
        writer.writeheader()

        epoch_count = len(history.history[keys[0]]) if keys else 0
        for epoch in range(epoch_count):
            row = {"epoch": epoch + 1}
            for key in keys:
                row[key] = float(history.history[key][epoch])
            writer.writerow(row)


def write_confusion_matrix_csv(path: Path, evaluators: dict[str, SplitEvaluator], class_labels: list[str]) -> None:
    predicted_labels = class_labels + ["__NONE__"]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["split", "true_label", "predicted_label", "count"])
        writer.writeheader()

        for split, evaluator in evaluators.items():
            for true_label in class_labels:
                for predicted_label in predicted_labels:
                    count = evaluator.confusion[true_label].get(predicted_label, 0)
                    writer.writerow(
                        {
                            "split": split,
                            "true_label": true_label,
                            "predicted_label": predicted_label,
                            "count": count,
                        }
                    )


def write_classification_report_csv(path: Path, evaluators: dict[str, SplitEvaluator], class_labels: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["split", "class_name", "precision", "recall", "f1", "support", "tp", "fp", "fn"],
        )
        writer.writeheader()

        for split, evaluator in evaluators.items():
            per_class = evaluator.per_class_metrics()
            for label in class_labels:
                metrics = per_class[label]
                writer.writerow(
                    {
                        "split": split,
                        "class_name": label,
                        "precision": round(float(metrics["precision"]), 6),
                        "recall": round(float(metrics["recall"]), 6),
                        "f1": round(float(metrics["f1"]), 6),
                        "support": int(metrics["support"]),
                        "tp": int(metrics["tp"]),
                        "fp": int(metrics["fp"]),
                        "fn": int(metrics["fn"]),
                    }
                )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def to_metrics_payload(evaluators: dict[str, SplitEvaluator], class_labels: list[str]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for split, evaluator in evaluators.items():
        payload[split] = {
            "accuracy": round(evaluator.accuracy(), 6),
            "macro": {name: round(value, 6) for name, value in evaluator.macro_scores().items()},
            "per_class": {
                label: {
                    "precision": round(float(metrics["precision"]), 6),
                    "recall": round(float(metrics["recall"]), 6),
                    "f1": round(float(metrics["f1"]), 6),
                    "support": int(metrics["support"]),
                    "tp": int(metrics["tp"]),
                    "fp": int(metrics["fp"]),
                    "fn": int(metrics["fn"]),
                }
                for label, metrics in evaluator.per_class_metrics().items()
            },
        }
    return payload


def main() -> int:
    args = parse_args()
    feature_csv_path = args.feature_csv_path.resolve()

    try:
        frame_samples, frame_metadata = load_frame_samples(feature_csv_path)
        sequences, rejected_sequences = build_video_sequences(frame_samples)
        split_to_sequences = split_sequences(sequences)
    except ValueError as exc:
        print(f"[ERROR] Dataset validation failed: {exc}")
        return 1

    class_labels = frame_metadata["class_labels"]
    set_reproducibility(DEFAULT_RANDOM_SEED)

    train_sequences = split_to_sequences["train"]
    val_sequences = split_to_sequences["val"]
    test_sequences = split_to_sequences["test"]

    train_sequence_lengths = [len(sample.sequence) for sample in train_sequences]
    if not train_sequence_lengths:
        fail("No training sequences available to derive max sequence length")

    # Protocol correction: fixed input length must come only from TRAIN sequences.
    max_sequence_length = max(train_sequence_lengths)

    sequence_lengths = [len(sample.sequence) for sample in sequences]
    sequence_stats = get_sequence_length_stats(sequences)

    x_train, train_labels, train_videos, truncated_train_sequences = pad_sequences_to_numpy(
        train_sequences,
        max_sequence_length,
    )
    x_val, val_labels, val_videos, truncated_val_sequences = pad_sequences_to_numpy(
        val_sequences,
        max_sequence_length,
    )
    x_test, test_labels, test_videos, truncated_test_sequences = pad_sequences_to_numpy(
        test_sequences,
        max_sequence_length,
    )

    y_train = to_label_indices(train_labels, class_labels)
    y_val = to_label_indices(val_labels, class_labels)
    y_test = to_label_indices(test_labels, class_labels)

    class_weights = compute_class_weights(train_labels, class_labels)

    model = build_model(class_count=len(class_labels), max_sequence_length=max_sequence_length)

    output_dir = feature_csv_path.parent / "gru"
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "gru_model.keras"
    metrics_path = output_dir / "gru_metrics.json"
    confusion_path = output_dir / "gru_confusion_matrix.csv"
    report_path = output_dir / "gru_classification_report.csv"
    summary_path = output_dir / "gru_training_summary.json"
    history_path = output_dir / "gru_training_history.csv"
    sequence_summary_path = output_dir / "gru_sequence_summary.json"

    callbacks: list[tf.keras.callbacks.Callback] = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=DEFAULT_EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
    ]

    history = model.fit(
        x=x_train,
        y=y_train,
        validation_data=(x_val, y_val),
        epochs=DEFAULT_EPOCHS,
        batch_size=DEFAULT_BATCH_SIZE,
        class_weight=class_weights,
        verbose=2,
        callbacks=callbacks,
        shuffle=False,
    )

    best_model = tf.keras.models.load_model(model_path)

    train_pred = np.argmax(best_model.predict(x_train, verbose=0), axis=1)
    val_pred = np.argmax(best_model.predict(x_val, verbose=0), axis=1)
    test_pred = np.argmax(best_model.predict(x_test, verbose=0), axis=1)

    evaluators = {
        "train": evaluate_predictions(train_labels, train_pred, class_labels),
        "val": evaluate_predictions(val_labels, val_pred, class_labels),
        "test": evaluate_predictions(test_labels, test_pred, class_labels),
    }

    metrics_payload = to_metrics_payload(evaluators, class_labels)
    write_json(metrics_path, metrics_payload)
    write_confusion_matrix_csv(confusion_path, evaluators, class_labels)
    write_classification_report_csv(report_path, evaluators, class_labels)
    write_training_history(history_path, history)

    house_label = "19._House"
    house_test_support = metrics_payload["test"]["per_class"].get(house_label, {}).get("support", 0)

    sequence_summary = {
        "total_videos": len(sequences),
        "train_videos": len(train_sequences),
        "val_videos": len(val_sequences),
        "test_videos": len(test_sequences),
        "feature_dimension": FEATURE_COUNT,
        "class_labels": class_labels,
        "sequence_length_stats": sequence_stats,
        "padding": {
            "strategy": "right_padding",
            "pad_value": PAD_VALUE,
            "max_sequence_length": max_sequence_length,
            "derived_from": "train_only",
            "masking_enabled": True,
        },
        "truncation": {
            "strategy": "evenly_spaced_temporal_sampling",
            "max_sequence_length": max_sequence_length,
            "truncated_train_sequences": truncated_train_sequences,
            "truncated_val_sequences": truncated_val_sequences,
            "truncated_test_sequences": truncated_test_sequences,
        },
        "rejected_sequences": rejected_sequences,
        "train_video_paths": train_videos,
        "val_video_paths": val_videos,
        "test_video_paths": test_videos,
    }
    write_json(sequence_summary_path, sequence_summary)

    model_size_bytes = model_path.stat().st_size if model_path.exists() else -1
    training_summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "feature_csv_path": str(feature_csv_path),
        "output_dir": str(output_dir),
        "dataset": {
            "frame_counts_by_split": frame_metadata["frame_counts_by_split"],
            "video_counts_by_split": {
                "train": len(train_sequences),
                "val": len(val_sequences),
                "test": len(test_sequences),
            },
            "class_labels": class_labels,
            "feature_count": FEATURE_COUNT,
        },
        "model": {
            "type": "tensorflow.keras.Sequential",
            "input_feature_dim": FEATURE_COUNT,
            "gru_units": GRU_UNITS,
            "dense_units": HIDDEN_UNITS,
            "dropout_rate": DEFAULT_DROPOUT_RATE,
            "output_classes": len(class_labels),
            "parameter_count": int(best_model.count_params()),
            "model_file": str(model_path),
            "model_size_bytes": model_size_bytes,
        },
        "training": {
            "random_seed": DEFAULT_RANDOM_SEED,
            "optimizer": "Adam",
            "learning_rate": DEFAULT_LEARNING_RATE,
            "loss": "sparse_categorical_crossentropy",
            "batch_size": DEFAULT_BATCH_SIZE,
            "requested_epochs": DEFAULT_EPOCHS,
            "executed_epochs": len(history.history.get("loss", [])),
            "early_stopping": {
                "monitor": "val_loss",
                "patience": DEFAULT_EARLY_STOPPING_PATIENCE,
                "restore_best_weights": True,
            },
            "class_weights": {str(index): float(value) for index, value in class_weights.items()},
            "protocol": {
                "unit_of_training": "video_sequence",
                "one_video_equals_one_sequence": True,
                "frame_ordering": "sorted_by_frame_index",
                "fixed_input_length_derived_from": "train_only",
                "truncation_strategy": "evenly_spaced_temporal_sampling",
                "training_max_sequence_length": max_sequence_length,
                "truncated_train_sequences": truncated_train_sequences,
                "truncated_val_sequences": truncated_val_sequences,
                "truncated_test_sequences": truncated_test_sequences,
                "test_used_for_training": False,
                "shuffle": False,
            },
        },
        "notes": {
            "house_test_support": house_test_support,
            "house_test_message": (
                "19._House has no held-out test videos; test metrics for this class are unavailable."
                if house_test_support == 0
                else "19._House has held-out test support."
            ),
        },
        "outputs": {
            "gru_model": str(model_path),
            "gru_metrics": str(metrics_path),
            "gru_confusion_matrix": str(confusion_path),
            "gru_classification_report": str(report_path),
            "gru_training_summary": str(summary_path),
            "gru_training_history": str(history_path),
            "gru_sequence_summary": str(sequence_summary_path),
        },
    }
    write_json(summary_path, training_summary)

    print(f"[INFO] Model saved: {model_path}")
    print(f"[INFO] Metrics saved: {metrics_path}")
    print(f"[INFO] Confusion matrix saved: {confusion_path}")
    print(f"[INFO] Classification report saved: {report_path}")
    print(f"[INFO] Training summary saved: {summary_path}")
    print(f"[INFO] Sequence summary saved: {sequence_summary_path}")
    print(f"[INFO] Training history saved: {history_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())