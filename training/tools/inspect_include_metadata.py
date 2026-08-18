#!/usr/bin/env python3
"""
SIGNEXA utility: inspect INCLUDE metadata (read-only).

- Discovers candidate metadata files under a supplied root directory.
- Supports CSV, TSV, JSON, and JSONL.
- Detects likely class/video/category/archive fields.
- Summarizes available classes and unique video counts per class.

This script does NOT download, extract, modify, move, or delete anything.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


EXISTING_CLASSES = {
    "1._Dog",
    "19._House",
    "40._I",
    "48._Hello",
    "55._Thank_you",
}
EXISTING_LABELS = {1, 19, 40, 48, 55}

SUPPORTED_EXTS = {".csv", ".tsv", ".json", ".jsonl"}

VIDEO_PATH_KEYS = [
    "video_path",
    "relative_video_path",
    "relative_path",
    "path",
    "file_path",
    "filepath",
    "video",
    "video_file",
    "video_filename",
    "clip_path",
]

FILENAME_KEYS = ["filename", "file_name", "video_name", "name"]

CLASS_NAME_KEYS = [
    "label",
    "class",
    "class_name",
    "label_name",
    "sign",
    "sign_label",
    "gloss",
    "target",
    "word",
]

LABEL_NUM_KEYS = [
    "label_id",
    "class_id",
    "numeric_label",
    "label_number",
    "id",
    "class_index",
]

CATEGORY_KEYS = [
    "parent_label",
    "parent",
    "category",
    "parent_category",
    "group",
    "super_category",
    "topic",
]

ARCHIVE_KEYS = [
    "archive",
    "archive_name",
    "source_archive",
    "zip",
    "zip_file",
    "source_zip",
    "source",
    "source_file",
]


@dataclass
class MetadataCandidate:
    path: Path
    records: List[Dict[str, Any]]
    columns: List[str]
    class_field: Optional[str]
    label_num_field: Optional[str]
    video_path_field: Optional[str]
    video_fallback_field: Optional[str]
    category_field: Optional[str]
    archive_field: Optional[str]
    score: int


def info(msg: str) -> None:
    print(f"[INFO] {msg}")


def warning(msg: str) -> None:
    print(f"[WARNING] {msg}")


def error(msg: str) -> None:
    print(f"[ERROR] {msg}")


def normalize_key(name: str) -> str:
    return str(name).strip().lower()


def pick_field(columns: Sequence[str], keys: Sequence[str]) -> Optional[str]:
    cmap = {normalize_key(c): c for c in columns}
    for k in keys:
        if normalize_key(k) in cmap:
            return cmap[normalize_key(k)]
    return None


def discover_files(root: Path) -> List[Path]:
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]
    return sorted(files)


def parse_csv_or_tsv(path: Path, delimiter: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError("No header found")
        for row in reader:
            rows.append(dict(row))
    return rows


def parse_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)

    if isinstance(data, list):
        if all(isinstance(x, dict) for x in data):
            return [dict(x) for x in data]
        raise ValueError("JSON list does not contain objects")

    if isinstance(data, dict):
        for key in ("records", "data", "items", "rows", "metadata"):
            v = data.get(key)
            if isinstance(v, list) and all(isinstance(x, dict) for x in v):
                return [dict(x) for x in v]
        raise ValueError("JSON object does not contain a recognizable records list")

    raise ValueError("Unsupported JSON root type")


def parse_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            obj = json.loads(s)
            if not isinstance(obj, dict):
                raise ValueError(f"Line {line_no} is not a JSON object")
            rows.append(dict(obj))
    return rows


def parse_metadata_file(path: Path) -> List[Dict[str, Any]]:
    ext = path.suffix.lower()
    if ext == ".csv":
        return parse_csv_or_tsv(path, delimiter=",")
    if ext == ".tsv":
        return parse_csv_or_tsv(path, delimiter="\t")
    if ext == ".json":
        return parse_json(path)
    if ext == ".jsonl":
        return parse_jsonl(path)
    raise ValueError(f"Unsupported extension: {ext}")


def infer_columns(records: Sequence[Dict[str, Any]]) -> List[str]:
    cols = set()
    for r in records[:1000]:
        cols.update(r.keys())
    return sorted(cols)


def score_candidate(columns: Sequence[str], rows_count: int) -> Tuple[int, Optional[str], Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]]:
    class_field = pick_field(columns, CLASS_NAME_KEYS)
    label_num_field = pick_field(columns, LABEL_NUM_KEYS)
    video_path_field = pick_field(columns, VIDEO_PATH_KEYS)
    video_fallback_field = pick_field(columns, FILENAME_KEYS)
    category_field = pick_field(columns, CATEGORY_KEYS)
    archive_field = pick_field(columns, ARCHIVE_KEYS)

    score = 0
    if class_field:
        score += 120
    if label_num_field:
        score += 60
    if video_path_field:
        score += 140
    elif video_fallback_field:
        score += 50
    if category_field:
        score += 30
    if archive_field:
        score += 25
    if rows_count > 0:
        score += min(rows_count // 50, 40)

    return (
        score,
        class_field,
        label_num_field,
        video_path_field,
        video_fallback_field,
        category_field,
        archive_field,
    )


def to_int_maybe(v: Any) -> Optional[int]:
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    try:
        return int(float(s))
    except Exception:
        return None


def parse_label_from_text(text: Any) -> Tuple[Optional[int], str]:
    s = str(text).strip()
    if not s:
        return None, "UNKNOWN"

    # examples: "1._Dog", "1. Dog", "1_Dog", "1 - Dog"
    m = re.match(r"^\s*(\d+)\s*[\._\- ]+\s*(.+?)\s*$", s)
    if m:
        num = int(m.group(1))
        name = m.group(2).strip()
        return num, name

    return None, s


def clean_text(v: Any, default: str = "N/A") -> str:
    if v is None:
        return default
    s = str(v).strip()
    return s if s else default


def canonical_class(label_num: Optional[int], class_name: str) -> str:
    base = class_name.strip().replace(" ", "_")
    if label_num is None:
        return base
    return f"{label_num}._{base}"


def summarize(records: Sequence[Dict[str, Any]], candidate: MetadataCandidate) -> Tuple[List[Dict[str, Any]], Dict[str, int], str]:
    if not records:
        raise ValueError("Metadata file contains zero records")

    class_field = candidate.class_field
    if not class_field:
        raise ValueError("No class/label field identified in selected metadata file")

    video_field_used: Optional[str] = None
    if candidate.video_path_field:
        video_field_used = candidate.video_path_field
    elif candidate.video_fallback_field:
        video_field_used = candidate.video_fallback_field
        warning(
            "video_path field was not found; using fallback video identity field "
            f"'{video_field_used}'. Filename-only identity may not be globally unique."
        )
    else:
        warning("No video path/name field found; using metadata row index as identity fallback")

    grouped: Dict[Tuple[Optional[int], str, str, str], set] = {}

    for idx, row in enumerate(records):
        raw_class = row.get(class_field)
        parsed_num, parsed_name = parse_label_from_text(raw_class)

        label_num = to_int_maybe(row.get(candidate.label_num_field)) if candidate.label_num_field else parsed_num
        class_name = clean_text(parsed_name, default="UNKNOWN")

        category = clean_text(row.get(candidate.category_field), default="N/A") if candidate.category_field else "N/A"
        archive = clean_text(row.get(candidate.archive_field), default="N/A") if candidate.archive_field else "N/A"

        if video_field_used:
            vid = clean_text(row.get(video_field_used), default=f"__row_{idx}")
        else:
            vid = f"__row_{idx}"

        key = (label_num, class_name, category, archive)
        grouped.setdefault(key, set()).add(vid)

    # consolidate by (label_num, class_name), while keeping representative category/archive stats
    merged: Dict[Tuple[Optional[int], str], Dict[str, Any]] = {}
    for (label_num, class_name, category, archive), vids in grouped.items():
        k = (label_num, class_name)
        if k not in merged:
            merged[k] = {
                "label": label_num,
                "class_name": class_name,
                "videos": set(),
                "categories": set(),
                "archives": set(),
            }
        merged[k]["videos"].update(vids)
        if category != "N/A":
            merged[k]["categories"].add(category)
        if archive != "N/A":
            merged[k]["archives"].add(archive)

    rows: List[Dict[str, Any]] = []
    for (_, _), data in merged.items():
        label_num = data["label"]
        class_name = data["class_name"]
        canonical = canonical_class(label_num, class_name)

        is_existing = (canonical in EXISTING_CLASSES) or (label_num in EXISTING_LABELS)
        marker = "[EXISTING]" if is_existing else "[NEW CANDIDATE]"

        categories = sorted(data["categories"])
        archives = sorted(data["archives"])

        cat_display = " ; ".join(categories[:3]) if categories else "N/A"
        if len(categories) > 3:
            cat_display += f" ; ... (+{len(categories) - 3} more)"

        arc_display = " ; ".join(archives[:3]) if archives else "N/A"
        if len(archives) > 3:
            arc_display += f" ; ... (+{len(archives) - 3} more)"

        rows.append(
            {
                "marker": marker,
                "label": label_num if label_num is not None else "N/A",
                "class_name": canonical,
                "category": cat_display,
                "videos": len(data["videos"]),
                "archive": arc_display,
            }
        )

    rows.sort(key=lambda r: (-int(r["videos"]), str(r["label"]), r["class_name"]))

    totals = {
        "total_records": len(records),
        "unique_labels": len({r["label"] for r in rows}),
        "unique_class_names": len({r["class_name"] for r in rows}),
    }

    return rows, totals, (video_field_used or "row_index_fallback")


def format_table(rows: Sequence[Dict[str, Any]], headers: Sequence[str]) -> str:
    if not rows:
        return "(no rows)"

    widths: Dict[str, int] = {}
    for h in headers:
        widths[h] = len(h)
    for row in rows:
        for h in headers:
            widths[h] = max(widths[h], len(str(row.get(h, ""))))

    line_header = " | ".join(h.ljust(widths[h]) for h in headers)
    line_sep = "-+-".join("-" * widths[h] for h in headers)
    line_rows = [" | ".join(str(row.get(h, "")).ljust(widths[h]) for h in headers) for row in rows]

    return "\n".join([line_header, line_sep, *line_rows])


def select_best_candidate(candidates: Sequence[MetadataCandidate]) -> MetadataCandidate:
    if not candidates:
        raise ValueError("No metadata candidates found")

    ranked = sorted(candidates, key=lambda c: c.score, reverse=True)

    info("Candidate metadata files (ranked):")
    for i, c in enumerate(ranked, start=1):
        info(
            f"  {i:02d}. score={c.score:3d} rows={len(c.records):6d} file={c.path} "
            f"class={c.class_field or 'N/A'} video_path={c.video_path_field or 'N/A'}"
        )

    return ranked[0]


def load_candidates(root: Path) -> List[MetadataCandidate]:
    paths = discover_files(root)
    if not paths:
        raise FileNotFoundError(f"No metadata files found under: {root}")

    candidates: List[MetadataCandidate] = []
    parse_failures: List[str] = []

    for path in paths:
        try:
            records = parse_metadata_file(path)
        except Exception as exc:  # noqa: BLE001
            parse_failures.append(f"{path}: {exc}")
            continue

        if not records:
            continue

        columns = infer_columns(records)
        (
            score,
            class_field,
            label_num_field,
            video_path_field,
            video_fallback_field,
            category_field,
            archive_field,
        ) = score_candidate(columns, len(records))

        # only keep likely class-bearing files
        if not class_field:
            continue

        candidates.append(
            MetadataCandidate(
                path=path,
                records=records,
                columns=columns,
                class_field=class_field,
                label_num_field=label_num_field,
                video_path_field=video_path_field,
                video_fallback_field=video_fallback_field,
                category_field=category_field,
                archive_field=archive_field,
                score=score,
            )
        )

    if not candidates:
        details = "\n".join(parse_failures[:20])
        raise RuntimeError(
            "Could not identify any parseable metadata file with class/label information.\n"
            + (f"Parse failures (first 20):\n{details}" if details else "")
        )

    if parse_failures:
        warning(f"Some files could not be parsed (count={len(parse_failures)}).")

    return candidates


def run(root_dir: str) -> int:
    root = Path(root_dir)
    if not root.exists() or not root.is_dir():
        error(f"Supplied directory does not exist or is not a directory: {root}")
        return 1

    info(f"Scanning metadata root: {root}")

    try:
        candidates = load_candidates(root)
        selected = select_best_candidate(candidates)
        rows, totals, video_field_used = summarize(selected.records, selected)
    except Exception as exc:  # noqa: BLE001
        error(str(exc))
        return 1

    info(f"Metadata file used: {selected.path}")
    info(f"Total metadata records: {totals['total_records']}")
    info(f"Number of unique labels: {totals['unique_labels']}")
    info(f"Number of unique class names: {totals['unique_class_names']}")
    info(f"Video identity field used for unique counts: {video_field_used}")
    info(f"Class field used: {selected.class_field}")
    info(f"Category field used: {selected.category_field or 'N/A'}")
    info(f"Archive/source field used: {selected.archive_field or 'N/A'}")

    print("\nCLASS SUMMARY")
    print(
        format_table(
            rows,
            headers=["marker", "label", "class_name", "category", "videos", "archive"],
        )
    )

    new_candidates = [r for r in rows if r["marker"] == "[NEW CANDIDATE]"]
    new_candidates.sort(key=lambda r: (-int(r["videos"]), str(r["label"]), r["class_name"]))

    print("\nNEW CLASS CANDIDATES")
    if not new_candidates:
        print("(none found)")
    else:
        print(
            format_table(
                new_candidates,
                headers=["label", "class_name", "category", "videos", "archive"],
            )
        )

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect INCLUDE metadata and summarize classes (read-only)."
    )
    parser.add_argument(
        "metadata_root",
        help="Path to local metadata/dataset root (e.g., C:\\Users\\SONIBARE\\include-metadata)",
    )
    args = parser.parse_args()

    rc = run(args.metadata_root)
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
