#!/usr/bin/env python3
"""
SIGNEXA: Inspect INCLUDE-50 metadata for the selected 10-class vocabulary.

Read-only utility:
- Discovers metadata files (CSV/TSV/JSON/JSONL)
- Selects the most relevant metadata file
- Filters exactly the configured 10 classes by FULL class name match
- Validates unique video_path identity
- Prints per-video table and summaries
- Writes include10_selected_videos.csv

No downloading/extraction/training/preprocessing/modification of dataset assets.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path, PureWindowsPath
from typing import Any, Dict, List, Optional, Sequence


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

SUPPORTED_EXTS = {".csv", ".tsv", ".json", ".jsonl"}

CLASS_KEYS = ["label", "class", "class_name", "label_name", "sign_label", "gloss", "target"]
VIDEO_PATH_KEYS = [
    "video_path",
    "relative_video_path",
    "relative_path",
    "path",
    "file_path",
    "filepath",
    "video",
]
CATEGORY_KEYS = ["parent_label", "parent", "category", "parent_category", "group"]
SPLIT_KEYS = ["split", "dataset_split", "subset", "partition"]
ARCHIVE_KEYS = ["source_archive", "archive", "archive_name", "zip", "zip_file", "source_zip"]


def info(msg: str) -> None:
    print(f"[INFO] {msg}")


def warning(msg: str) -> None:
    print(f"[WARNING] {msg}")


def error(msg: str) -> None:
    print(f"[ERROR] {msg}")


def normalize_col(name: str) -> str:
    return str(name).strip().lower()


def pick_col(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    cmap = {normalize_col(c): c for c in columns}
    for cand in candidates:
        key = normalize_col(cand)
        if key in cmap:
            return cmap[key]
    return None


def discover_files(root: Path) -> List[Path]:
    return sorted(
        [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]
    )


def parse_csv_like(path: Path, delim: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=delim)
        if reader.fieldnames is None:
            raise ValueError("Missing header")
        for row in reader:
            rows.append(dict(row))
    return rows


def parse_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)

    if isinstance(data, list) and all(isinstance(x, dict) for x in data):
        return [dict(x) for x in data]

    if isinstance(data, dict):
        for key in ("records", "rows", "data", "items", "metadata"):
            value = data.get(key)
            if isinstance(value, list) and all(isinstance(x, dict) for x in value):
                return [dict(x) for x in value]

    raise ValueError("Unsupported JSON structure")


def parse_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for i, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            obj = json.loads(text)
            if not isinstance(obj, dict):
                raise ValueError(f"Line {i} is not a JSON object")
            rows.append(dict(obj))
    return rows


def parse_file(path: Path) -> List[Dict[str, Any]]:
    ext = path.suffix.lower()
    if ext == ".csv":
        return parse_csv_like(path, ",")
    if ext == ".tsv":
        return parse_csv_like(path, "\t")
    if ext == ".json":
        return parse_json(path)
    if ext == ".jsonl":
        return parse_jsonl(path)
    raise ValueError(f"Unsupported extension: {ext}")


def format_table(rows: List[Dict[str, Any]], headers: List[str]) -> str:
    if not rows:
        return "(no rows)"

    widths = {h: len(h) for h in headers}
    for row in rows:
        for h in headers:
            widths[h] = max(widths[h], len(str(row.get(h, ""))))

    header = " | ".join(h.ljust(widths[h]) for h in headers)
    separator = "-+-".join("-" * widths[h] for h in headers)
    body = [
        " | ".join(str(row.get(h, "")).ljust(widths[h]) for h in headers)
        for row in rows
    ]

    return "\n".join([header, separator, *body])


def clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def get_filename(video_path: str) -> str:
    if not video_path:
        return ""
    try:
        return PureWindowsPath(video_path).name
    except Exception:
        parts = video_path.replace("\\", "/").split("/")
        return parts[-1] if parts else ""


def choose_best_metadata_file(files: List[Path]) -> Dict[str, Any]:
    candidates: List[Dict[str, Any]] = []
    failures: List[str] = []

    for file_path in files:
        try:
            records = parse_file(file_path)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{file_path}: {exc}")
            continue

        if not records:
            continue

        columns = sorted({k for row in records[:1000] for k in row.keys()})
        class_col = pick_col(columns, CLASS_KEYS)
        video_col = pick_col(columns, VIDEO_PATH_KEYS)
        category_col = pick_col(columns, CATEGORY_KEYS)
        split_col = pick_col(columns, SPLIT_KEYS)
        archive_col = pick_col(columns, ARCHIVE_KEYS)

        score = 0
        if class_col:
            score += 120
        if video_col:
            score += 140
        if category_col:
            score += 30
        if split_col:
            score += 30
        if archive_col:
            score += 20
        score += min(len(records) // 50, 50)

        if class_col and video_col:
            candidates.append(
                {
                    "path": file_path,
                    "records": records,
                    "columns": columns,
                    "class_col": class_col,
                    "video_col": video_col,
                    "category_col": category_col,
                    "split_col": split_col,
                    "archive_col": archive_col,
                    "score": score,
                }
            )

    if not candidates:
        details = "\n".join(failures[:20])
        raise RuntimeError(
            "No parseable metadata file with both class and video-path information was found."
            + (f"\nParse failures (first 20):\n{details}" if details else "")
        )

    candidates.sort(key=lambda x: x["score"], reverse=True)

    info("Candidate metadata files (ranked):")
    for i, candidate in enumerate(candidates, start=1):
        info(
            f"  {i:02d}. score={candidate['score']:3d} rows={len(candidate['records']):6d} "
            f"file={candidate['path']} class={candidate['class_col']} "
            f"video_path={candidate['video_col']}"
        )

    return candidates[0]


def run(metadata_root: Path, out_csv: Path) -> int:
    if not metadata_root.exists() or not metadata_root.is_dir():
        error(f"Directory does not exist: {metadata_root}")
        return 1

    if len(set(SELECTED_CLASSES)) != 10:
        error("Internal config error: selected class list must contain exactly 10 unique class names.")
        return 1

    info(f"Scanning metadata root: {metadata_root}")
    files = discover_files(metadata_root)
    if not files:
        error("No metadata files found (.csv/.tsv/.json/.jsonl).")
        return 1

    info(f"Discovered supported files: {len(files)}")

    try:
        selected_meta = choose_best_metadata_file(files)
    except Exception as exc:  # noqa: BLE001
        error(str(exc))
        return 1

    path = selected_meta["path"]
    records = selected_meta["records"]
    class_col = selected_meta["class_col"]
    video_col = selected_meta["video_col"]
    category_col = selected_meta["category_col"]
    split_col = selected_meta["split_col"]
    archive_col = selected_meta["archive_col"]

    info(f"Metadata file used: {path}")
    info(f"Total metadata records: {len(records)}")
    info(f"Class column: {class_col}")
    info(f"Video identity column (required): {video_col}")
    info(f"Category column: {category_col or 'N/A'}")
    info(f"Split column: {split_col or 'N/A'}")
    info(f"Archive/source column: {archive_col or 'N/A'}")

    selected_set = set(SELECTED_CLASSES)

    filtered_rows: List[Dict[str, Any]] = []
    for row in records:
        class_name = clean(row.get(class_col))
        if not class_name:
            continue
        if class_name not in selected_set:
            continue

        video_path = clean(row.get(video_col))
        if not video_path:
            continue

        output_row: Dict[str, Any] = {
            "class_name": class_name,
            "label": class_name,
            "category": clean(row.get(category_col)) if category_col else "N/A",
            "video_path": video_path,
            "filename": get_filename(video_path),
            "split": clean(row.get(split_col)) if split_col else "",
        }

        if archive_col:
            output_row["source_archive"] = clean(row.get(archive_col)) or "N/A"

        filtered_rows.append(output_row)

    found_classes = sorted({row["class_name"] for row in filtered_rows})
    missing_classes = [cls for cls in SELECTED_CLASSES if cls not in found_classes]

    if missing_classes:
        error("One or more selected classes were not found in metadata:")
        for cls in missing_classes:
            error(f"  - {cls}")
        return 1

    unexpected_classes = sorted(set(found_classes) - selected_set)
    if unexpected_classes:
        error(f"Unexpected classes included after filtering: {unexpected_classes}")
        return 1

    unique_map: Dict[tuple, Dict[str, Any]] = {}
    for row in filtered_rows:
        key = (row["class_name"], row["video_path"])
        if key not in unique_map:
            unique_map[key] = row

    unique_rows = list(unique_map.values())

    video_path_counts: Dict[str, int] = {}
    filename_counts: Dict[str, int] = {}

    for row in unique_rows:
        vp = row["video_path"]
        fn = row["filename"]
        video_path_counts[vp] = video_path_counts.get(vp, 0) + 1
        filename_counts[fn] = filename_counts.get(fn, 0) + 1

    duplicate_video_paths = sum(1 for _, cnt in video_path_counts.items() if cnt > 1)
    duplicate_filenames = sum(1 for _, cnt in filename_counts.items() if cnt > 1)

    if duplicate_video_paths > 0:
        error(f"Duplicate video_path values found: {duplicate_video_paths}. This is not allowed.")
        return 1

    unique_rows.sort(key=lambda x: (x["class_name"], x["video_path"]))

    print("\nSELECTED VIDEOS (ONE ROW PER UNIQUE VIDEO)")
    headers = ["class_name", "label", "category", "video_path", "filename", "split"]
    if archive_col:
        headers.append("source_archive")
    print(format_table(unique_rows, headers))

    summary_map: Dict[str, Dict[str, Any]] = {}
    for row in unique_rows:
        class_name = row["class_name"]
        if class_name not in summary_map:
            summary_map[class_name] = {
                "class_name": class_name,
                "category": row["category"] or "N/A",
                "unique_videos": 0,
            }
        summary_map[class_name]["unique_videos"] += 1

    summary_rows = sorted(summary_map.values(), key=lambda x: x["class_name"])

    print("\nSELECTED CLASS SUMMARY")
    print(format_table(summary_rows, ["class_name", "category", "unique_videos"]))

    print("\nTOTAL SELECTED VIDEOS")
    print(len(unique_rows))

    print("\nDUPLICATE CHECK")
    print(f"duplicate video_path count: {duplicate_video_paths}")
    print(f"duplicate filename count: {duplicate_filenames}")

    if split_col:
        split_summary: Dict[str, Dict[str, int]] = {}
        for row in unique_rows:
            class_name = row["class_name"]
            split_value = clean(row.get("split")) or "N/A"
            split_summary.setdefault(class_name, {})
            split_summary[class_name][split_value] = split_summary[class_name].get(split_value, 0) + 1

        split_rows: List[Dict[str, Any]] = []
        for class_name in sorted(split_summary.keys()):
            counts = split_summary[class_name]
            split_rows.append(
                {
                    "class_name": class_name,
                    "train": counts.get("train", 0),
                    "val": counts.get("val", 0),
                    "test": counts.get("test", 0),
                    "other": sum(v for k, v in counts.items() if k not in {"train", "val", "test"}),
                }
            )

        print("\nSPLIT COUNTS PER SELECTED CLASS")
        print(format_table(split_rows, ["class_name", "train", "val", "test", "other"]))
    else:
        print("\nSplit information not available in source metadata.")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_columns = ["class_name", "label", "category", "video_path", "filename", "split"]
    if archive_col:
        out_columns.append("source_archive")

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_columns)
        writer.writeheader()
        for row in unique_rows:
            writer.writerow({col: row.get(col, "") for col in out_columns})

    info(f"Wrote selected videos CSV: {out_csv}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect INCLUDE metadata for SIGNEXA selected 10 classes (read-only)."
    )
    parser.add_argument("metadata_root", help="Path to metadata root directory")
    parser.add_argument(
        "--out",
        default="include10_selected_videos.csv",
        help="Output CSV path (default: include10_selected_videos.csv)",
    )
    args = parser.parse_args()

    exit_code = run(Path(args.metadata_root), Path(args.out))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
