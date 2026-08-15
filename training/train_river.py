#!/usr/bin/env python3
"""SIGNEXA River baseline training and evaluation script.

This script trains an incremental multiclass classifier from a feature CSV.
It is designed for local execution and does not depend on the React app.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import river
    from river import naive_bayes
except ImportError as exc:  # pragma: no cover
    print("[ERROR] Missing dependency: river")
    print("Install it with: python -m pip install river")
    raise SystemExit(1) from exc


FEATURE_COUNT = 63
EXPECTED_SPLITS = ("train", "val", "test")
EXPECTED_CLASSES = 5
FEATURE_COLUMNS = [f"feature_{index}" for index in range(FEATURE_COUNT)]


@dataclass
class Sample:
    split: str
    label: str
    video_path: str
    features: dict[str, float]


class SplitEvaluator:
    """Tracks split metrics from predictions without updating the model."""

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
            fn = sum(count for predicted, count in self.confusion[label].items() if predicted != label)
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
    parser = argparse.ArgumentParser(description="Train and evaluate SIGNEXA River baseline from normalized features")
    parser.add_argument("feature_csv_path", type=Path, help="Path to normalized_features.csv")
    return parser.parse_args()


def is_finite(value: float) -> bool:
    return math.isfinite(value)


def fail(message: str) -> None:
    raise ValueError(message)


def validate_columns(fieldnames: list[str] | None) -> None:
    if not fieldnames:
        fail("CSV has no header row.")

    duplicate_columns = [name for name, count in Counter(fieldnames).items() if count > 1]
    if duplicate_columns:
        fail(f"CSV contains duplicate column names: {duplicate_columns}")

    missing = [column for column in FEATURE_COLUMNS if column not in fieldnames]
    if missing:
        fail(f"Missing feature columns: {missing}")

    feature_like_columns = [column for column in fieldnames if column.startswith("feature_")]

    malformed_feature_columns = [
        column
        for column in feature_like_columns
        if not column[8:].isdigit()
    ]
    if malformed_feature_columns:
        fail(f"Malformed feature column names: {malformed_feature_columns}")

    discovered_feature_set = set(feature_like_columns)
    expected_feature_set = set(FEATURE_COLUMNS)

    unexpected_feature_columns = sorted(discovered_feature_set - expected_feature_set)
    if unexpected_feature_columns:
        fail(f"Unexpected feature columns found: {unexpected_feature_columns}")

    missing_feature_columns = sorted(expected_feature_set - discovered_feature_set)
    if missing_feature_columns:
        fail(f"Missing feature columns: {missing_feature_columns}")

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
                fail(f"Row {row_index} is missing class_name label")
            if not video_path:
                fail(f"Row {row_index} is missing video_path")

            features: dict[str, float] = {}
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

                features[column] = value

            # Enforce that only feature_0..feature_62 are used as model input.
            if set(features.keys()) != set(FEATURE_COLUMNS):
                fail(f"Row {row_index} feature key mismatch")

            sample = Sample(split=split, label=label, video_path=video_path, features=features)
            samples.append(sample)

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
        fail(f"Found videos appearing in multiple splits: {mixed_split_videos}")

    metadata = {
        "class_labels": sorted(labels),
        "split_counts": {split: split_counts[split] for split in EXPECTED_SPLITS},
        "video_counts_by_split": {
            split: len([video for video, splits in video_to_splits.items() if split in splits])
            for split in EXPECTED_SPLITS
        },
        "unique_video_count": len(video_to_splits),
        "feature_count": FEATURE_COUNT,
    }

    return samples, metadata


def run_incremental_training(samples: list[Sample], labels: list[str]) -> tuple[Any, dict[str, SplitEvaluator]]:
    model = naive_bayes.GaussianNB()

    evaluators = {
        "train": SplitEvaluator(labels),
        "val": SplitEvaluator(labels),
        "test": SplitEvaluator(labels),
    }

    for sample in samples:
        prediction = model.predict_one(sample.features)
        evaluators[sample.split].update(sample.label, prediction)

        # Core River concept: update incrementally one sample at a time on TRAIN only.
        if sample.split == "train":
            model.learn_one(sample.features, sample.label)

    return model, evaluators


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
            "macro": {key: round(value, 6) for key, value in evaluator.macro_scores().items()},
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
    model, evaluators = run_incremental_training(samples, labels)

    output_dir = feature_csv_path.parent / "river"
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "river_model.pkl"
    metrics_json_path = output_dir / "river_metrics.json"
    confusion_csv_path = output_dir / "river_confusion_matrix.csv"
    report_csv_path = output_dir / "river_classification_report.csv"
    summary_json_path = output_dir / "river_training_summary.json"

    model_artifact = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "class_labels": labels,
        "model_type": "river.naive_bayes.GaussianNB",
        "library": "river",
        "river_version": river.__version__,
    }

    with model_path.open("wb") as file:
        pickle.dump(model_artifact, file)

    metrics_payload = to_metrics_payload(evaluators, labels)
    write_json(metrics_json_path, metrics_payload)
    write_confusion_matrix_csv(confusion_csv_path, evaluators, labels)
    write_classification_report_csv(report_csv_path, evaluators, labels)

    house_test_support = metrics_payload["test"]["per_class"].get("House", {}).get("support", 0)

    training_summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "feature_csv_path": str(feature_csv_path),
        "output_dir": str(output_dir),
        "model": {
            "type": "river.naive_bayes.GaussianNB",
            "library": "river",
            "river_version": river.__version__,
            "feature_count": FEATURE_COUNT,
            "class_labels": labels,
        },
        "dataset": dataset_info,
        "protocol": {
            "ordering": "CSV row order preserved (deterministic)",
            "training_mode": "incremental_online",
            "train_step": "predict_then_learn_one",
            "validation_step": "predict_only_no_learn",
            "test_step": "predict_only_no_learn",
            "train_updates_use_only_split": "train",
            "validation_test_updates_disabled": True,
        },
        "notes": {
            "house_test_support": house_test_support,
            "house_test_message": (
                "House has no held-out test videos in this dataset split configuration."
                if house_test_support == 0
                else "House has held-out test support."
            ),
        },
        "outputs": {
            "river_model": str(model_path),
            "river_metrics": str(metrics_json_path),
            "river_confusion_matrix": str(confusion_csv_path),
            "river_classification_report": str(report_csv_path),
            "river_training_summary": str(summary_json_path),
        },
    }

    write_json(summary_json_path, training_summary)

    print(f"[INFO] Model saved: {model_path}")
    print(f"[INFO] Metrics saved: {metrics_json_path}")
    print(f"[INFO] Confusion matrix saved: {confusion_csv_path}")
    print(f"[INFO] Classification report saved: {report_csv_path}")
    print(f"[INFO] Training summary saved: {summary_json_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())