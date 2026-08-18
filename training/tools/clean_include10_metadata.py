#!/usr/bin/env python3
"""
SIGNEXA - INCLUDE-10 metadata cleaner (read-only source handling).

Input (read-only):
    <metadata_root>/include50_video_manifest.csv

Outputs:
    <metadata_root>/include10_clean_dataset_manifest.csv
    <metadata_root>/include10_metadata_cleaning_report.csv
    <metadata_root>/include10_metadata_cleaning_summary.json

This utility does NOT modify the raw manifest and does NOT download/extract/preprocess/train.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Dict, List

import pandas as pd


SELECTED_CLASSES = [
    "1. Dog",
    "19. House",
    "40. I",
    "48. Hello",
    "55. Thank you",
    "51. Good Morning",
    "61. Father",
    "11. Car",
    "28. Store or Shop",
    "35. Bank",
]
SELECTED_SET = set(SELECTED_CLASSES)

REASON_PATH_MISMATCH = "path_label_mismatch"
REASON_DUPLICATE_EXACT = "duplicate_exact_removed"
REASON_CONFLICTING = "conflicting_metadata_rejected"
REASON_AMBIGUOUS = "ambiguous_metadata_rejected"


def info(msg: str) -> None:
    print(f"[INFO] {msg}")


def warning(msg: str) -> None:
    print(f"[WARNING] {msg}")


def error(msg: str) -> None:
    print(f"[ERROR] {msg}")


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def split_norm(value: Any) -> str:
    s = clean_text(value).lower()
    if s in {"train", "training"}:
        return "train"
    if s in {"val", "valid", "validation", "dev"}:
        return "val"
    if s in {"test", "testing"}:
        return "test"
    return s


def extract_class_dir(video_path: str) -> str:
    p = clean_text(video_path)
    if not p:
        return ""

    norm = p.replace("\\", "/")
    parts = [seg.strip() for seg in norm.split("/") if seg.strip()]
    if len(parts) < 2:
        return ""
    return parts[-2]


def extract_filename(video_path: str) -> str:
    p = clean_text(video_path)
    if not p:
        return ""

    try:
        return PureWindowsPath(p).name
    except Exception:
        try:
            return PurePosixPath(p.replace("\\", "/")).name
        except Exception:
            parts = p.replace("\\", "/").split("/")
            return parts[-1] if parts else ""


def build_report_row(row: pd.Series, reason: str) -> Dict[str, Any]:
    return {
        "video_path": clean_text(row.get("video_path", "")),
        "original_label": clean_text(row.get("label", "")),
        "parent_label": clean_text(row.get("parent_label", "")),
        "split": clean_text(row.get("split", "")),
        "reason": reason,
    }


def ensure_required_columns(df: pd.DataFrame) -> None:
    required = ["split", "parent_label", "label", "video_path"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Raw manifest missing required columns: {missing}")


def validate_final_dataset(
    final_df: pd.DataFrame,
    raw_support_df: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[str]]:
    validation_errors: List[str] = []

    final_labels = set(final_df["label"].map(clean_text).tolist()) if not final_df.empty else set()
    missing_classes = [cls for cls in SELECTED_CLASSES if cls not in final_labels]
    unexpected_classes = sorted(final_labels - SELECTED_SET)

    if missing_classes:
        validation_errors.append(f"Missing selected classes in cleaned manifest: {missing_classes}")
    if unexpected_classes:
        validation_errors.append(f"Unselected classes present in cleaned manifest: {unexpected_classes}")
    if len(final_labels) != 10:
        validation_errors.append(f"Expected exactly 10 class names; found {len(final_labels)}")

    if not final_df.empty and final_df["video_path"].duplicated().any():
        validation_errors.append("Duplicate video_path detected in cleaned manifest")

    if not final_df.empty:
        path_class_series = final_df["video_path"].map(extract_class_dir)
        mismatch = path_class_series != final_df["label"].map(clean_text)
        if mismatch.any():
            validation_errors.append("Path/label consistency violation exists in cleaned manifest")

    class_summary_rows: List[Dict[str, Any]] = []

    for cls in SELECTED_CLASSES:
        cls_df = final_df[final_df["label"].map(clean_text) == cls].copy()
        cls_df["_split_norm"] = cls_df["split"].map(split_norm)

        train_count = int((cls_df["_split_norm"] == "train").sum())
        val_count = int((cls_df["_split_norm"] == "val").sum())
        test_count = int((cls_df["_split_norm"] == "test").sum())

        raw_cls_df = raw_support_df[raw_support_df["label"].map(clean_text) == cls].copy()
        raw_test_available = int((raw_cls_df["_split_norm"] == "test").sum())

        if train_count == 0:
            validation_errors.append(f"Class '{cls}' has zero train examples in cleaned manifest")
        if val_count == 0:
            validation_errors.append(f"Class '{cls}' has zero validation examples in cleaned manifest")
        if test_count == 0:
            if raw_test_available > 0:
                validation_errors.append(
                    f"Class '{cls}' has zero test examples after cleaning despite raw test availability ({raw_test_available})"
                )
            else:
                validation_errors.append(
                    f"Class '{cls}' has zero test examples and raw metadata has insufficient path-consistent test support"
                )

        categories = sorted({clean_text(v) for v in cls_df["parent_label"].tolist() if clean_text(v)})
        category_display = " ; ".join(categories) if categories else "N/A"

        class_summary_rows.append(
            {
                "class": cls,
                "category": category_display,
                "train": train_count,
                "val": val_count,
                "test": test_count,
                "total": int(len(cls_df)),
            }
        )

    return class_summary_rows, validation_errors


def run(metadata_root: Path) -> int:
    if not metadata_root.exists() or not metadata_root.is_dir():
        error(f"Directory does not exist: {metadata_root}")
        return 1

    raw_manifest_path = metadata_root / "include50_video_manifest.csv"
    if not raw_manifest_path.exists():
        error(f"Raw manifest not found: {raw_manifest_path}")
        return 1

    clean_manifest_path = metadata_root / "include10_clean_dataset_manifest.csv"
    cleaning_report_path = metadata_root / "include10_metadata_cleaning_report.csv"
    summary_json_path = metadata_root / "include10_metadata_cleaning_summary.json"

    try:
        df_raw = pd.read_csv(raw_manifest_path, dtype=str).fillna("")
    except Exception as exc:
        error(f"Failed to read raw manifest: {exc}")
        return 1

    try:
        ensure_required_columns(df_raw)
    except Exception as exc:
        error(str(exc))
        return 1

    raw_record_count = int(len(df_raw))

    df = df_raw.copy()
    df["split"] = df["split"].map(clean_text)
    df["parent_label"] = df["parent_label"].map(clean_text)
    df["label"] = df["label"].map(clean_text)
    df["video_path"] = df["video_path"].map(clean_text)

    df["_path_class"] = df["video_path"].map(extract_class_dir)
    df["_path_label_match"] = df["_path_class"] == df["label"]

    selected_raw_df = df[df["label"].isin(SELECTED_SET)].copy()
    selected_raw_record_count = int(len(selected_raw_df))

    if selected_raw_record_count == 0:
        error("No records found for selected 10 classes.")
        return 1

    selected_video_paths = set(selected_raw_df["video_path"].tolist())
    relevant_df = df[df["video_path"].isin(selected_video_paths)].copy()

    raw_support_df = selected_raw_df[selected_raw_df["_path_label_match"]].copy()
    raw_support_df["_split_norm"] = raw_support_df["split"].map(split_norm)

    kept_rows: List[Dict[str, Any]] = []
    report_rows: List[Dict[str, Any]] = []

    for video_path, group in relevant_df.groupby("video_path", sort=False):
        _ = video_path
        group = group.copy()

        selected_group = group[group["label"].isin(SELECTED_SET)].copy()
        non_selected_group = group[~group["label"].isin(SELECTED_SET)].copy()

        if selected_group.empty:
            continue

        selected_valid = selected_group[selected_group["_path_label_match"]].copy()
        selected_invalid = selected_group[~selected_group["_path_label_match"]].copy()

        for _, row in selected_invalid.iterrows():
            report_rows.append(build_report_row(row, REASON_PATH_MISMATCH))

        for _, row in non_selected_group.iterrows():
            report_rows.append(build_report_row(row, REASON_CONFLICTING))

        if selected_valid.empty:
            continue

        # Remove exact duplicate metadata rows among selected-valid.
        selected_valid_unique = selected_valid.drop_duplicates(keep="first")

        if len(selected_valid_unique) < len(selected_valid):
            duplicate_rows = selected_valid.loc[selected_valid.index.difference(selected_valid_unique.index)]
            for _, row in duplicate_rows.iterrows():
                report_rows.append(build_report_row(row, REASON_DUPLICATE_EXACT))

        # If multiple conflicting selected-valid rows remain for same path -> ambiguous reject.
        if len(selected_valid_unique) > 1:
            for _, row in selected_valid_unique.iterrows():
                report_rows.append(build_report_row(row, REASON_AMBIGUOUS))
            continue

        kept_rows.append(selected_valid_unique.iloc[0].to_dict())

    if kept_rows:
        final_df = pd.DataFrame(kept_rows)
    else:
        final_df = pd.DataFrame(columns=list(df.columns))

    if not final_df.empty and final_df["video_path"].duplicated().any():
        dup_paths = final_df.loc[final_df["video_path"].duplicated(), "video_path"].tolist()
        error(f"Internal safety check failed: duplicate video_path in final dataset: {dup_paths[:10]}")
        return 1

    final_df = final_df.copy()
    final_df["filename"] = final_df["video_path"].map(extract_filename)

    for col in ["_path_class", "_path_label_match", "_split_norm"]:
        if col in final_df.columns:
            final_df = final_df.drop(columns=[col])

    required_front = ["split", "parent_label", "label", "video_path", "filename"]
    front_cols = [col for col in required_front if col in final_df.columns]
    other_cols = [col for col in final_df.columns if col not in front_cols]
    final_df = final_df[front_cols + other_cols]

    report_df = pd.DataFrame(report_rows)
    if report_df.empty:
        report_df = pd.DataFrame(columns=["video_path", "original_label", "parent_label", "split", "reason"])

    # Always produce auditable outputs.
    final_df.to_csv(clean_manifest_path, index=False)
    report_df.to_csv(cleaning_report_path, index=False)

    class_summary_rows, validation_errors = validate_final_dataset(final_df, raw_support_df)

    final_unique_video_count = int(final_df["video_path"].nunique()) if not final_df.empty else 0
    exact_duplicates_removed = int((report_df["reason"] == REASON_DUPLICATE_EXACT).sum())
    conflicting_metadata_rejected = int((report_df["reason"] == REASON_CONFLICTING).sum())
    ambiguous_videos_rejected = int(
        report_df.loc[report_df["reason"] == REASON_AMBIGUOUS, "video_path"].nunique()
    )
    rejected_record_count = int(len(report_df))

    summary_obj: Dict[str, Any] = {
        "raw_manifest_path": str(raw_manifest_path),
        "clean_manifest_path": str(clean_manifest_path),
        "cleaning_report_path": str(cleaning_report_path),
        "selected_classes": SELECTED_CLASSES,
        "raw_record_count": raw_record_count,
        "selected_raw_record_count": selected_raw_record_count,
        "rejected_record_count": rejected_record_count,
        "exact_duplicates_removed": exact_duplicates_removed,
        "conflicting_metadata_rejected": conflicting_metadata_rejected,
        "ambiguous_videos_rejected": ambiguous_videos_rejected,
        "final_unique_video_count": final_unique_video_count,
        "class_summary": class_summary_rows,
        "validation_passed": len(validation_errors) == 0,
        "validation_errors": validation_errors,
    }

    with summary_json_path.open("w", encoding="utf-8") as f:
        json.dump(summary_obj, f, indent=2)

    print("\nRAW RECORD COUNT")
    print(raw_record_count)

    print("\nSELECTED RAW RECORD COUNT")
    print(selected_raw_record_count)

    print("\nREJECTED RECORD COUNT")
    print(rejected_record_count)

    print("\nEXACT DUPLICATES REMOVED")
    print(exact_duplicates_removed)

    print("\nCONFLICTING METADATA REJECTED")
    print(conflicting_metadata_rejected)

    print("\nAMBIGUOUS VIDEOS REJECTED")
    print(ambiguous_videos_rejected)

    print("\nFINAL UNIQUE VIDEO COUNT")
    print(final_unique_video_count)

    print("\nFINAL 10-CLASS SUMMARY")
    print(pd.DataFrame(class_summary_rows).to_string(index=False))

    print("\nCLEANING REPORT PATH")
    print(cleaning_report_path)

    print("\nCLEAN MANIFEST PATH")
    print(clean_manifest_path)

    print("\nSUMMARY JSON PATH")
    print(summary_json_path)

    if validation_errors:
        for msg in validation_errors:
            error(msg)
        error("Validation failed. Outputs were written for audit; fix issues before proceeding.")
        return 1

    info("Validation passed for cleaned INCLUDE-10 manifest.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create INCLUDE-10 cleaned metadata manifest from INCLUDE-50 manifest."
    )
    parser.add_argument(
        "metadata_root",
        help="Path to metadata root (contains include50_video_manifest.csv)",
    )
    args = parser.parse_args()

    exit_code = run(Path(args.metadata_root))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
