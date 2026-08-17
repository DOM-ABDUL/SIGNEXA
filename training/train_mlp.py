#!/usr/bin/env python3
"""SIGNEXA MLP baseline training and evaluation.

This script trains a lightweight frame-level MLP on normalized landmark
features. It is meant for local execution and does not depend on the React app.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
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

DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 100
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_DROPOUT_RATE = 0.2
DEFAULT_EARLY_STOPPING_PATIENCE = 12
DEFAULT_RANDOM_SEED = 42


@dataclass
class Sample:
    split: str
    label: str
    video_path: str
    features: list[float]


class SplitEvaluator:
    """Computes split metrics from predicted class indices."""

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

        precision = sum(float(m["precision"]) for m in per_class.values()) / len(per_class)
        recall = sum(float(m["recall"]) for m in per_class.values()) / len(per_class)
        f1 = sum(float(m["f1"]) for m in per_class.values()) / len(per_class)
        return {"precision": precision, "recall": recall, "f1": f1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SIGNEXA MLP baseline from normalized features")
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

    # Best-effort deterministic ops; may vary by hardware/backend.
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


def load_and_validate_samples(csv_path: Path) -> tuple[list[Sample], dict[str, Any]]:
    if not csv_path.exists() or not csv_path.is_file():
        fail(f"Feature CSV not found: {csv_path}")

    samples: list[Sample] = []
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

            if split not in EXPECTED_SPLITS:
                fail(f"Row {row_index} has invalid split: {split}")
            if not label:
                fail(f"Row {row_index} missing class_name")
            if not video_path:
                fail(f"Row {row_index} missing video_path")

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

            samples.append(Sample(split=split, label=label, video_path=video_path, features=features))
            labels.add(label)
            split_counts[split] += 1
            video_to_splits[video_path].add(split)

    if not samples:
        fail("Feature CSV contains no samples")

    if len(labels) != EXPECTED_CLASSES:
        fail(f"Expected exactly {EXPECTED_CLASSES} classes, found {len(labels)}: {sorted(labels)}")

    for split in EXPECTED_SPLITS:
        if split_counts.get(split, 0) == 0:
            fail(f"Split '{split}' has no samples")

    mixed_split_videos = {
        video_path: sorted(list(splits))
        for video_path, splits in video_to_splits.items()
        if len(splits) > 1
    }
    if mixed_split_videos:
        fail(f"Found videos in multiple splits: {mixed_split_videos}")

    metadata = {
        "class_labels": sorted(labels),
        "feature_count": FEATURE_COUNT,
        "split_counts": {split: split_counts[split] for split in EXPECTED_SPLITS},
        "video_counts_by_split": {
            split: len([video for video, splits in video_to_splits.items() if split in splits])
            for split in EXPECTED_SPLITS
        },
        "unique_video_count": len(video_to_splits),
    }

    return samples, metadata


def prepare_split_arrays(samples: list[Sample], labels: list[str]) -> dict[str, tuple[np.ndarray, np.ndarray, list[str]]]:
    label_to_index = {label: index for index, label in enumerate(labels)}
    grouped: dict[str, list[Sample]] = {split: [] for split in EXPECTED_SPLITS}

    for sample in samples:
        grouped[sample.split].append(sample)

    arrays: dict[str, tuple[np.ndarray, np.ndarray, list[str]]] = {}
    for split, split_samples in grouped.items():
        x = np.array([sample.features for sample in split_samples], dtype=np.float32)
        y = np.array([label_to_index[sample.label] for sample in split_samples], dtype=np.int64)
        y_labels = [sample.label for sample in split_samples]
        arrays[split] = (x, y, y_labels)

    return arrays


def compute_class_weights(y_train: np.ndarray, class_count: int) -> dict[int, float]:
    counts = Counter(int(value) for value in y_train.tolist())
    total = float(len(y_train))
    if total <= 0:
        return {}

    # Inverse-frequency weighting, computed from train split only.
    weights: dict[int, float] = {}
    for class_index in range(class_count):
        count = float(counts.get(class_index, 0))
        if count <= 0:
            weights[class_index] = 0.0
        else:
            weights[class_index] = total / (class_count * count)
    return weights


def build_model(class_count: int) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(FEATURE_COUNT,)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(DEFAULT_DROPOUT_RATE),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(DEFAULT_DROPOUT_RATE),
            tf.keras.layers.Dense(class_count, activation="softmax"),
        ]
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=DEFAULT_LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def evaluate_predictions(true_labels: list[str], predicted_indices: np.ndarray, labels: list[str]) -> SplitEvaluator:
    evaluator = SplitEvaluator(labels)
    index_to_label = {index: label for index, label in enumerate(labels)}

    for true_label, predicted_index in zip(true_labels, predicted_indices.tolist()):
        predicted_label = index_to_label.get(int(predicted_index))
        evaluator.update(true_label, predicted_label)

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
                value = history.history[key][epoch]
                row[key] = float(value)
            writer.writerow(row)


def write_confusion_matrix_csv(path: Path, evaluators: dict[str, SplitEvaluator], labels: list[str]) -> None:
    predicted_labels = labels + ["__NONE__"]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["split", "true_label", "predicted_label", "count"])
        writer.writeheader()

        for split, evaluator in evaluators.items():
            for true_label in labels:
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


def write_classification_report_csv(path: Path, evaluators: dict[str, SplitEvaluator], labels: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["split", "class_name", "precision", "recall", "f1", "support", "tp", "fp", "fn"],
        )
        writer.writeheader()

        for split, evaluator in evaluators.items():
            per_class = evaluator.per_class_metrics()
            for label in labels:
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


def to_metrics_payload(evaluators: dict[str, SplitEvaluator], labels: list[str]) -> dict[str, Any]:
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
        samples, dataset_info = load_and_validate_samples(feature_csv_path)
    except ValueError as exc:
        print(f"[ERROR] Dataset validation failed: {exc}")
        return 1

    labels = dataset_info["class_labels"]
    set_reproducibility(DEFAULT_RANDOM_SEED)

    split_arrays = prepare_split_arrays(samples, labels)
    x_train, y_train, train_labels = split_arrays["train"]
    x_val, y_val, val_labels = split_arrays["val"]
    x_test, y_test, test_labels = split_arrays["test"]

    model = build_model(class_count=len(labels))
    class_weights = compute_class_weights(y_train, class_count=len(labels))

    output_dir = feature_csv_path.parent / "mlp"
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "mlp_model.keras"
    metrics_path = output_dir / "mlp_metrics.json"
    confusion_path = output_dir / "mlp_confusion_matrix.csv"
    report_path = output_dir / "mlp_classification_report.csv"
    summary_path = output_dir / "mlp_training_summary.json"
    history_path = output_dir / "training_history.csv"

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

    # Load best checkpoint before final split evaluations.
    best_model = tf.keras.models.load_model(model_path)

    train_probs = best_model.predict(x_train, verbose=0)
    val_probs = best_model.predict(x_val, verbose=0)
    test_probs = best_model.predict(x_test, verbose=0)

    train_pred = np.argmax(train_probs, axis=1)
    val_pred = np.argmax(val_probs, axis=1)
    test_pred = np.argmax(test_probs, axis=1)

    evaluators = {
        "train": evaluate_predictions(train_labels, train_pred, labels),
        "val": evaluate_predictions(val_labels, val_pred, labels),
        "test": evaluate_predictions(test_labels, test_pred, labels),
    }

    metrics_payload = to_metrics_payload(evaluators, labels)
    write_json(metrics_path, metrics_payload)
    write_confusion_matrix_csv(confusion_path, evaluators, labels)
    write_classification_report_csv(report_path, evaluators, labels)
    write_training_history(history_path, history)

    house_test_support = metrics_payload["test"]["per_class"].get("19._House", {}).get("support", 0)
    model_size_bytes = model_path.stat().st_size if model_path.exists() else -1

    summary_payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "feature_csv_path": str(feature_csv_path),
        "output_dir": str(output_dir),
        "dataset": dataset_info,
        "model": {
            "type": "tensorflow.keras.Sequential",
            "input_features": FEATURE_COUNT,
            "output_classes": len(labels),
            "class_labels": labels,
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
            "class_weights": {str(key): float(value) for key, value in class_weights.items()},
            "split_protocol": {
                "train_used_for_fit": True,
                "val_used_for_fit_validation": True,
                "test_used_for_fit": False,
                "shuffle": False,
            },
        },
        "notes": {
            "house_test_support": house_test_support,
            "house_test_message": (
                "House has no held-out test videos; test metrics for House are unavailable."
                if house_test_support == 0
                else "House has held-out test support."
            ),
        },
        "outputs": {
            "mlp_model": str(model_path),
            "mlp_metrics": str(metrics_path),
            "mlp_confusion_matrix": str(confusion_path),
            "mlp_classification_report": str(report_path),
            "mlp_training_summary": str(summary_path),
            "training_history": str(history_path),
        },
    }

    write_json(summary_path, summary_payload)

    print(f"[INFO] Model saved: {model_path}")
    print(f"[INFO] Metrics saved: {metrics_path}")
    print(f"[INFO] Confusion matrix saved: {confusion_path}")
    print(f"[INFO] Classification report saved: {report_path}")
    print(f"[INFO] Training summary saved: {summary_path}")
    print(f"[INFO] Training history saved: {history_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())