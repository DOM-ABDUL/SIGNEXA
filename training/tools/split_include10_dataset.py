import argparse
import json
import random
from pathlib import Path

import pandas as pd


RANDOM_SEED = 42
EXPECTED_CLASS_COUNT = 10
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def compute_split_counts(total: int) -> tuple[int, int, int]:
    if total < 3:
        raise ValueError(
            f"Each class needs at least 3 videos for train/val/test allocation. Got {total}."
        )

    raw = {
        "train": TRAIN_RATIO * total,
        "val": VAL_RATIO * total,
        "test": TEST_RATIO * total,
    }
    counts = {k: int(v) for k, v in raw.items()}

    remainder = total - sum(counts.values())
    fractional_order = sorted(
        raw.keys(),
        key=lambda k: (raw[k] - counts[k], {"train": 0, "val": 1, "test": 2}[k]),
        reverse=True,
    )

    for i in range(remainder):
        counts[fractional_order[i % len(fractional_order)]] += 1

    # Enforce minimum 1 sample per split by borrowing from larger splits.
    for split_name in ("train", "val", "test"):
        if counts[split_name] == 0:
            donor = max((s for s in ("train", "val", "test") if counts[s] > 1), key=counts.get, default=None)
            if donor is None:
                raise ValueError(f"Cannot enforce minimum split count for class size {total}.")
            counts[donor] -= 1
            counts[split_name] += 1

    if sum(counts.values()) != total:
        raise ValueError("Split counts do not sum to class total.")

    return counts["train"], counts["val"], counts["test"]


def create_split(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = {"label", "video_path"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

    duplicate_count = int(df.duplicated(subset=["video_path"]).sum())
    if duplicate_count != 0:
        raise ValueError(f"Found duplicate video_path entries: {duplicate_count}")

    classes = sorted(df["label"].dropna().unique().tolist())
    if len(classes) != EXPECTED_CLASS_COUNT:
        raise ValueError(
            f"Expected exactly {EXPECTED_CLASS_COUNT} classes, found {len(classes)}."
        )

    rng = random.Random(RANDOM_SEED)
    split_assignment: dict[str, str] = {}

    for cls in classes:
        class_rows = df[df["label"] == cls]
        video_paths = class_rows["video_path"].tolist()
        rng.shuffle(video_paths)

        train_count, val_count, test_count = compute_split_counts(len(video_paths))
        train_videos = video_paths[:train_count]
        val_videos = video_paths[train_count : train_count + val_count]
        test_videos = video_paths[train_count + val_count : train_count + val_count + test_count]

        for vp in train_videos:
            split_assignment[vp] = "train"
        for vp in val_videos:
            split_assignment[vp] = "val"
        for vp in test_videos:
            split_assignment[vp] = "test"

    out_df = df.copy()
    out_df["split"] = out_df["video_path"].map(split_assignment)

    if "filename" not in out_df.columns:
        out_df["filename"] = out_df["video_path"].astype(str).apply(lambda p: Path(p).name)

    preferred_order = ["split", "parent_label", "label", "video_path", "filename"]
    ordered_existing = [c for c in preferred_order if c in out_df.columns]
    remaining = [c for c in out_df.columns if c not in ordered_existing]
    out_df = out_df[ordered_existing + remaining]

    return out_df


def validate_output(in_df: pd.DataFrame, out_df: pd.DataFrame) -> dict:
    checks = {}

    checks["exactly_10_classes"] = int(out_df["label"].nunique()) == EXPECTED_CLASS_COUNT
    checks["no_duplicate_video_path"] = int(out_df.duplicated(subset=["video_path"]).sum()) == 0

    split_per_video = out_df.groupby("video_path")["split"].nunique()
    checks["video_in_exactly_one_split"] = bool((split_per_video == 1).all())

    split_counts = out_df.groupby(["label", "split"]).size().unstack(fill_value=0)
    for col in ("train", "val", "test"):
        if col not in split_counts.columns:
            split_counts[col] = 0
    checks["no_zero_train_per_class"] = bool((split_counts["train"] > 0).all())
    checks["no_zero_val_per_class"] = bool((split_counts["val"] > 0).all())
    checks["no_zero_test_per_class"] = bool((split_counts["test"] > 0).all())

    checks["total_video_count_unchanged"] = len(in_df) == len(out_df)
    checks["no_records_lost"] = set(in_df["video_path"]) == set(out_df["video_path"])
    checks["no_records_duplicated"] = len(out_df) == out_df["video_path"].nunique()
    checks["all_original_columns_preserved"] = set(in_df.columns).issubset(set(out_df.columns))

    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise ValueError(f"Validation failed for checks: {failed}")

    videos_in_multiple_splits = int((split_per_video > 1).sum())

    per_class = (
        out_df.groupby(["label", "split"]).size().unstack(fill_value=0).reindex(columns=["train", "val", "test"], fill_value=0)
    )
    per_class["total"] = per_class.sum(axis=1)

    totals = {
        "train": int((out_df["split"] == "train").sum()),
        "val": int((out_df["split"] == "val").sum()),
        "test": int((out_df["split"] == "test").sum()),
        "total": int(len(out_df)),
    }

    return {
        "checks": checks,
        "videos_in_multiple_splits": videos_in_multiple_splits,
        "per_class": per_class,
        "totals": totals,
    }


def print_report(input_count: int, output_count: int, validation: dict) -> None:
    print(f"INPUT VIDEO COUNT: {input_count}")
    print(f"OUTPUT VIDEO COUNT: {output_count}")
    print(f"RANDOM SEED: {RANDOM_SEED}")
    print("")
    print("INCLUDE-10 SPLIT SUMMARY")
    print("class | train | val | test | total")

    per_class = validation["per_class"]
    for cls in per_class.index:
        row = per_class.loc[cls]
        print(f"{cls} | {int(row['train'])} | {int(row['val'])} | {int(row['test'])} | {int(row['total'])}")

    totals = validation["totals"]
    print(
        f"TOTAL | {totals['train']} | {totals['val']} | {totals['test']} | {totals['total']}"
    )
    print("")
    print(
        f"Videos appearing in multiple splits: {validation['videos_in_multiple_splits']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create reproducible custom INCLUDE-10 video-level stratified split."
    )
    parser.add_argument("input_manifest", type=str, help="Path to include10_clean_dataset_manifest.csv")
    args = parser.parse_args()

    input_path = Path(args.input_manifest)
    if not input_path.exists():
        raise FileNotFoundError(f"Input manifest not found: {input_path}")

    output_manifest = input_path.parent / "include10_dataset_manifest.csv"
    output_summary = input_path.parent / "include10_split_summary.json"

    in_df = pd.read_csv(input_path)
    out_df = create_split(in_df)
    validation = validate_output(in_df, out_df)

    out_df.to_csv(output_manifest, index=False)

    summary = {
        "split_type": "custom_video_level_stratified",
        "random_seed": RANDOM_SEED,
        "source_manifest": input_path.name,
        "reason": "Custom split created because the original metadata split had insufficient valid test support for at least one selected INCLUDE-10 class.",
        "input_video_count": int(len(in_df)),
        "output_video_count": int(len(out_df)),
        "videos_in_multiple_splits": int(validation["videos_in_multiple_splits"]),
        "totals": validation["totals"],
        "per_class": {
            cls: {
                "train": int(row["train"]),
                "val": int(row["val"]),
                "test": int(row["test"]),
                "total": int(row["total"]),
            }
            for cls, row in validation["per_class"].iterrows()
        },
        "validation_checks": validation["checks"],
    }

    with output_summary.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print_report(len(in_df), len(out_df), validation)


if __name__ == "__main__":
    main()