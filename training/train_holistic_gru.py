#!/usr/bin/env python3
"""
Train a holistic GRU classifier for INCLUDE-50 controlled 5-class benchmark.

This script is intentionally separate from hand-only training code and is
built for fair comparison against the existing hand-only GRU baseline.

Key guarantees:
- Video-level sequence modeling only (one video = one sample)
- Strict split integrity checks (train/val/test)
- Strict feature schema checks (feature_0..feature_1665)
- Deterministic seeding where practical
- Test split used only for final evaluation
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight


FEATURE_DIM = 1666
EXPECTED_FRAMES_PER_VIDEO = 8
EXPECTED_FEATURE_COLUMNS = [f"feature_{i}" for i in range(FEATURE_DIM)]
EXPECTED_SPLITS = {"train", "val", "test"}


@dataclass
class DatasetColumns:
    video_col: str
    split_col: str
    label_col: str
    frame_col: str


@dataclass
class SequenceDataset:
    x: np.ndarray
    y_idx: np.ndarray
    y_label: np.ndarray
    video_ids: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train holistic GRU on precomputed features")
    parser.add_argument(
        "dataset_root",
        nargs="?",
        default=r"C:\Users\SONIBARE\include-metadata\first5_dataset",
        help="Path to first5_dataset root",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, default=120, help="Max training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate")
    parser.add_argument("--gru-units", type=int, default=96, help="GRU hidden units")
    parser.add_argument("--dropout", type=float, default=0.25, help="Dropout rate")
    parser.add_argument("--recurrent-dropout", type=float, default=0.0, help="GRU recurrent dropout")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    return parser.parse_args()


def set_deterministic(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def resolve_paths(dataset_root: Path) -> Tuple[Path, Path]:
    csv_path = dataset_root / "holistic_features" / "holistic_features.csv"
    out_dir = dataset_root / "holistic_features" / "gru"
    return csv_path, out_dir


def _find_column(df: pd.DataFrame, candidates: Sequence[str], purpose: str) -> str:
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    raise ValueError(f"Could not infer {purpose} column. Tried: {candidates}")


def detect_columns(df: pd.DataFrame) -> DatasetColumns:
    video_col = _find_column(
        df,
        [
            "video_id",
            "video",
            "video_name",
            "video_filename",
            "video_file",
            "filename",
            "file_name",
            "video_path",
        ],
        "video id",
    )
    split_col = _find_column(df, ["split", "dataset_split", "subset"], "split")
    label_col = _find_column(
        df,
        ["label", "class", "class_name", "class_label", "sign_label", "target"],
        "class label",
    )
    frame_col = _find_column(
        df,
        ["frame_index", "frame_idx", "frame_no", "frame_number", "frame", "timestep", "time_index"],
        "frame order",
    )
    return DatasetColumns(video_col=video_col, split_col=split_col, label_col=label_col, frame_col=frame_col)


def validate_feature_schema(df: pd.DataFrame) -> None:
    missing = [c for c in EXPECTED_FEATURE_COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c.startswith("feature_") and c not in EXPECTED_FEATURE_COLUMNS]

    if missing:
        raise ValueError(f"Missing expected feature columns: first missing={missing[:5]} (total {len(missing)})")
    if extra:
        raise ValueError(f"Unexpected extra feature columns: first extra={extra[:5]} (total {len(extra)})")

    feature_cols = [c for c in df.columns if c.startswith("feature_")]
    if len(feature_cols) != FEATURE_DIM:
        raise ValueError(f"Feature dimension mismatch: got {len(feature_cols)}, expected {FEATURE_DIM}")

    if feature_cols != EXPECTED_FEATURE_COLUMNS:
        raise ValueError("Feature column order is not exactly feature_0..feature_1665")


def normalize_split_value(value: str) -> str:
    raw = str(value).strip().lower()
    if raw in {"train", "training"}:
        return "train"
    if raw in {"val", "valid", "validation", "dev"}:
        return "val"
    if raw in {"test", "testing"}:
        return "test"
    return raw


def validate_integrity(df: pd.DataFrame, cols: DatasetColumns) -> None:
    if df.empty:
        raise ValueError("Input CSV is empty")

    if df[EXPECTED_FEATURE_COLUMNS].isna().any().any():
        raise ValueError("NaN values detected in feature columns")

    features_np = df[EXPECTED_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    if not np.isfinite(features_np).all():
        raise ValueError("Infinity or non-finite values detected in feature columns")

    df = df.copy()
    df[cols.split_col] = df[cols.split_col].map(normalize_split_value)

    split_values = set(df[cols.split_col].unique())
    invalid_splits = split_values - EXPECTED_SPLITS
    if invalid_splits:
        raise ValueError(f"Unexpected split values detected: {sorted(invalid_splits)}")

    group = df.groupby(cols.video_col, sort=False)

    split_per_video = group[cols.split_col].nunique()
    if (split_per_video != 1).any():
        bad = split_per_video[split_per_video != 1]
        raise ValueError(f"Videos with multiple splits found: {bad.index.tolist()[:10]}")

    class_per_video = group[cols.label_col].nunique()
    if (class_per_video != 1).any():
        bad = class_per_video[class_per_video != 1]
        raise ValueError(f"Videos with multiple class labels found: {bad.index.tolist()[:10]}")

    frames_per_video = group.size()
    if (frames_per_video != EXPECTED_FRAMES_PER_VIDEO).any():
        bad = frames_per_video[frames_per_video != EXPECTED_FRAMES_PER_VIDEO]
        preview = bad.head(10).to_dict()
        raise ValueError(
            f"Every video must have exactly {EXPECTED_FRAMES_PER_VIDEO} frames; violations={len(bad)} preview={preview}"
        )

    for vid, g in group:
        frame_series = g[cols.frame_col]
        if frame_series.isna().any():
            raise ValueError(f"Missing frame order for video: {vid}")

        try:
            frame_vals = frame_series.astype(int).to_numpy()
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Non-integer frame order values for video: {vid}") from exc

        unique_sorted = np.unique(frame_vals)
        if len(unique_sorted) != EXPECTED_FRAMES_PER_VIDEO:
            raise ValueError(f"Duplicate/non-unique frame indexes for video: {vid}")

        diffs = np.diff(np.sort(unique_sorted))
        if not np.all(diffs == 1):
            raise ValueError(
                f"Frame order for video {vid} is not contiguous with step=1. Values={unique_sorted.tolist()}"
            )


def build_video_sequences(df: pd.DataFrame, cols: DatasetColumns) -> Tuple[Dict[str, SequenceDataset], List[str]]:
    df = df.copy()
    df[cols.split_col] = df[cols.split_col].map(normalize_split_value)

    videos = []
    for video_id, g in df.groupby(cols.video_col, sort=False):
        g_sorted = g.sort_values(cols.frame_col)
        x_seq = g_sorted[EXPECTED_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        if x_seq.shape != (EXPECTED_FRAMES_PER_VIDEO, FEATURE_DIM):
            raise ValueError(
                f"Sequence shape mismatch for video {video_id}: got {x_seq.shape}, expected ({EXPECTED_FRAMES_PER_VIDEO}, {FEATURE_DIM})"
            )

        label_values = g_sorted[cols.label_col].astype(str).unique()
        split_values = g_sorted[cols.split_col].astype(str).unique()
        if len(label_values) != 1 or len(split_values) != 1:
            raise ValueError(f"Video {video_id} has inconsistent label/split assignments")

        videos.append(
            {
                "video_id": str(video_id),
                "split": split_values[0],
                "label": label_values[0],
                "x": x_seq,
            }
        )

    classes = sorted({v["label"] for v in videos})
    class_to_idx = {c: i for i, c in enumerate(classes)}

    split_to_items = {"train": [], "val": [], "test": []}
    for v in videos:
        split_to_items[v["split"]].append(v)

    datasets: Dict[str, SequenceDataset] = {}
    for split_name, items in split_to_items.items():
        if not items:
            raise ValueError(f"No videos in split: {split_name}")

        x = np.stack([it["x"] for it in items], axis=0)
        y_label = np.array([it["label"] for it in items], dtype=object)
        y_idx = np.array([class_to_idx[label] for label in y_label], dtype=np.int64)
        video_ids = np.array([it["video_id"] for it in items], dtype=object)

        datasets[split_name] = SequenceDataset(x=x, y_idx=y_idx, y_label=y_label, video_ids=video_ids)

    return datasets, classes


def build_model(
    timesteps: int,
    feature_dim: int,
    num_classes: int,
    gru_units: int,
    dropout: float,
    recurrent_dropout: float,
    learning_rate: float,
) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(timesteps, feature_dim), name="input_sequence")
    x = tf.keras.layers.Masking(mask_value=0.0, name="masking")(inputs)
    x = tf.keras.layers.GRU(
        gru_units,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        name="gru",
    )(x)
    x = tf.keras.layers.Dense(64, activation="relu", name="dense_64")(x)
    x = tf.keras.layers.Dropout(dropout, name="dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="classifier")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="holistic_gru")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def compute_class_weights(y_train_idx: np.ndarray, classes_count: int) -> Dict[int, float]:
    class_ids = np.arange(classes_count)
    present_classes = np.unique(y_train_idx)
    weights_present = compute_class_weight(
        class_weight="balanced",
        classes=present_classes,
        y=y_train_idx,
    )

    weights = {int(c): 1.0 for c in class_ids}
    for c, w in zip(present_classes, weights_present):
        weights[int(c)] = float(w)
    return weights


def evaluate_split(
    split_name: str,
    y_true_idx: np.ndarray,
    y_pred_idx: np.ndarray,
    class_names: Sequence[str],
) -> Dict:
    acc = float(accuracy_score(y_true_idx, y_pred_idx))

    per_prec, per_rec, per_f1, per_support = precision_recall_fscore_support(
        y_true_idx,
        y_pred_idx,
        labels=np.arange(len(class_names)),
        zero_division=0,
    )

    macro_prec = float(np.mean(per_prec))
    macro_rec = float(np.mean(per_rec))
    macro_f1 = float(np.mean(per_f1))

    cm = confusion_matrix(y_true_idx, y_pred_idx, labels=np.arange(len(class_names)))

    class_stats = {}
    for i, cls in enumerate(class_names):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        class_stats[cls] = {
            "precision": float(per_prec[i]),
            "recall": float(per_rec[i]),
            "f1": float(per_f1[i]),
            "support": int(per_support[i]),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    return {
        "split": split_name,
        "accuracy": acc,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "macro_f1": macro_f1,
        "per_class": class_stats,
        "confusion_matrix": cm.tolist(),
        "class_order": list(class_names),
    }


def save_reports(
    out_dir: Path,
    metrics: Dict,
    class_names: Sequence[str],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = out_dir / "holistic_gru_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    cm = np.array(metrics["test"]["confusion_matrix"], dtype=int)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.index.name = "true_label"
    cm_df.to_csv(out_dir / "holistic_gru_confusion_matrix.csv")

    rows = []
    for split_name in ("train", "val", "test"):
        split_m = metrics[split_name]
        for cls in class_names:
            cls_m = split_m["per_class"][cls]
            rows.append(
                {
                    "split": split_name,
                    "class": cls,
                    "precision": cls_m["precision"],
                    "recall": cls_m["recall"],
                    "f1": cls_m["f1"],
                    "support": cls_m["support"],
                    "tp": cls_m["tp"],
                    "fp": cls_m["fp"],
                    "fn": cls_m["fn"],
                }
            )

        rows.append(
            {
                "split": split_name,
                "class": "__macro__",
                "precision": split_m["macro_precision"],
                "recall": split_m["macro_recall"],
                "f1": split_m["macro_f1"],
                "support": int(sum(split_m["per_class"][c]["support"] for c in class_names)),
                "tp": "",
                "fp": "",
                "fn": "",
            }
        )

    pd.DataFrame(rows).to_csv(out_dir / "holistic_gru_classification_report.csv", index=False)


def bytes_to_mb(num_bytes: int) -> float:
    return float(num_bytes) / (1024.0 * 1024.0)


def main() -> None:
    args = parse_args()
    set_deterministic(args.seed)

    dataset_root = Path(args.dataset_root)
    csv_path, out_dir = resolve_paths(dataset_root)

    if not csv_path.exists():
        raise FileNotFoundError(f"Holistic feature CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    validate_feature_schema(df)
    cols = detect_columns(df)
    validate_integrity(df, cols)

    datasets, class_names = build_video_sequences(df, cols)
    num_classes = len(class_names)

    x_train, y_train = datasets["train"].x, datasets["train"].y_idx
    x_val, y_val = datasets["val"].x, datasets["val"].y_idx
    x_test, y_test = datasets["test"].x, datasets["test"].y_idx

    model = build_model(
        timesteps=EXPECTED_FRAMES_PER_VIDEO,
        feature_dim=FEATURE_DIM,
        num_classes=num_classes,
        gru_units=args.gru_units,
        dropout=args.dropout,
        recurrent_dropout=args.recurrent_dropout,
        learning_rate=args.learning_rate,
    )

    class_weights = compute_class_weights(y_train, classes_count=num_classes)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=args.patience,
            restore_best_weights=True,
            verbose=1,
        )
    ]

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=2,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "holistic_gru_model.keras"
    model.save(model_path)

    # Evaluate after best-weights restoration from early stopping.
    y_pred_train = np.argmax(model.predict(x_train, verbose=0), axis=1)
    y_pred_val = np.argmax(model.predict(x_val, verbose=0), axis=1)
    y_pred_test = np.argmax(model.predict(x_test, verbose=0), axis=1)

    train_metrics = evaluate_split("train", y_train, y_pred_train, class_names)
    val_metrics = evaluate_split("val", y_val, y_pred_val, class_names)
    test_metrics = evaluate_split("test", y_test, y_pred_test, class_names)

    history_df = pd.DataFrame(history.history)
    history_df.insert(0, "epoch", np.arange(1, len(history_df) + 1))
    history_path = out_dir / "holistic_gru_history.csv"
    history_df.to_csv(history_path, index=False)

    model_size_mb = bytes_to_mb(model_path.stat().st_size) if model_path.exists() else None

    metrics = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "data": {
            "dataset_root": str(dataset_root),
            "holistic_csv": str(csv_path),
            "expected_feature_dim": FEATURE_DIM,
            "expected_frames_per_video": EXPECTED_FRAMES_PER_VIDEO,
            "detected_columns": asdict(cols),
            "class_order": class_names,
            "split_video_counts": {
                split: int(len(ds.video_ids))
                for split, ds in datasets.items()
            },
            "split_frame_counts": {
                split: int(len(ds.video_ids) * EXPECTED_FRAMES_PER_VIDEO)
                for split, ds in datasets.items()
            },
        },
        "model": {
            "name": model.name,
            "input_shape": [EXPECTED_FRAMES_PER_VIDEO, FEATURE_DIM],
            "num_classes": num_classes,
            "parameters": int(model.count_params()),
            "gru_units": args.gru_units,
            "dropout": args.dropout,
            "recurrent_dropout": args.recurrent_dropout,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "max_epochs": args.epochs,
            "trained_epochs": int(len(history.history.get("loss", []))),
            "early_stopping_patience": args.patience,
            "model_path": str(model_path),
            "model_size_mb": model_size_mb,
        },
        "class_weights_train_only": class_weights,
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
    }

    save_reports(out_dir, metrics, class_names)

    summary = {
        "timestamp_utc": metrics["timestamp_utc"],
        "input_csv": str(csv_path),
        "output_dir": str(out_dir),
        "input_shape_per_video": [EXPECTED_FRAMES_PER_VIDEO, FEATURE_DIM],
        "num_classes": num_classes,
        "class_order": class_names,
        "model_name": model.name,
        "parameters": int(model.count_params()),
        "trained_epochs": int(len(history.history.get("loss", []))),
        "best_epoch_val_loss": int(np.argmin(history.history["val_loss"]) + 1),
        "train_accuracy": train_metrics["accuracy"],
        "train_macro_f1": train_metrics["macro_f1"],
        "val_accuracy": val_metrics["accuracy"],
        "val_macro_f1": val_metrics["macro_f1"],
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "notes": [
            "Controlled 5-class experiment only; do not over-generalize.",
            "Test interpretation should include per-class support and macro-F1.",
            "Test split is held out from training and early stopping.",
        ],
    }

    summary_path = out_dir / "holistic_gru_training_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[OK] Holistic GRU training complete")
    print(f"[OK] Model saved: {model_path}")
    print(f"[OK] Metrics saved: {out_dir / 'holistic_gru_metrics.json'}")
    print(f"[OK] Confusion matrix saved: {out_dir / 'holistic_gru_confusion_matrix.csv'}")
    print(f"[OK] Classification report saved: {out_dir / 'holistic_gru_classification_report.csv'}")
    print(f"[OK] Summary saved: {summary_path}")
    print(
        f"[RESULT] train acc={train_metrics['accuracy']:.6f} f1={train_metrics['macro_f1']:.6f} | "
        f"val acc={val_metrics['accuracy']:.6f} f1={val_metrics['macro_f1']:.6f} | "
        f"test acc={test_metrics['accuracy']:.6f} f1={test_metrics['macro_f1']:.6f}"
    )


if __name__ == "__main__":
    main()
