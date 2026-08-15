#!/usr/bin/env python3
"""Local preprocessing utility for SIGNEXA INCLUDE-50 datasets.

This script is designed for local execution on your machine. It is not run
against local Windows paths from this environment.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

HAND_LANDMARK_COUNT = 21
FEATURE_VECTOR_LENGTH = 63
WRIST_INDEX = 0
MIDDLE_MCP_INDEX = 9
MIN_SCALE = 1e-6

DEFAULT_FRAMES_PER_VIDEO = 8
DEFAULT_MAX_VIDEOS_PER_CLASS = 2
DEFAULT_LOW_DETECTION_THRESHOLD = 0.5

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
VALID_SPLITS = {"train", "val", "test"}

# This corrupted record was intentionally excluded from source metadata.
EXCLUDED_RELATIVE_PATH_SUFFIXES = {
    "places/19. house/adjectives/78. long/mvi_5106.mov",
}


@dataclass
class FeatureRecord:
    split: str
    class_name: str
    label: str
    video_path: str
    filename: str
    frame_index: int
    timestamp_ms: float
    hand_index: int
    feature_vector: list[float]


@dataclass
class VideoProcessingResult:
    split: str
    class_name: str
    label: str
    filename: str
    video_path: str
    sampled_frame_count: int
    detected_frame_count: int
    detection_rate: float
    valid_normalized_frame_count: int
    invalid_frame_count: int
    feature_record_count: int
    status: str
    error: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SIGNEXA local INCLUDE preprocessing tool")
    parser.add_argument(
        "dataset_path",
        type=Path,
        help="Path to dataset root containing train/, val/, and test/ directories.",
    )
    parser.add_argument(
        "--frames-per-video",
        type=int,
        default=DEFAULT_FRAMES_PER_VIDEO,
        help="Number of evenly spaced frames to sample per video (default: 8).",
    )
    parser.add_argument(
        "--max-videos-per-class",
        type=int,
        default=DEFAULT_MAX_VIDEOS_PER_CLASS,
        help="Maximum videos per class in smoke mode (default: 2).",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Process full dataset (train/val/test). Without this, smoke mode uses train split only.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Optional local path to hand_landmarker.task.",
    )
    parser.add_argument(
        "--low-detection-threshold",
        type=float,
        default=DEFAULT_LOW_DETECTION_THRESHOLD,
        help="Flag videos with detection rate below this value in summary (default: 0.5).",
    )
    return parser.parse_args()


def is_video_file(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTENSIONS


def is_excluded_video(relative_path: str) -> bool:
    normalized = relative_path.replace("\\", "/").strip().lower()
    return any(normalized.endswith(suffix) for suffix in EXCLUDED_RELATIVE_PATH_SUFFIXES)


def discover_class_videos(split_dir: Path, dataset_root: Path) -> dict[str, list[Path]]:
    class_to_videos: dict[str, list[Path]] = {}

    if not split_dir.exists() or not split_dir.is_dir():
        return class_to_videos

    for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
        videos: list[Path] = []
        for video_path in sorted(class_dir.rglob("*")):
            if not video_path.is_file() or not is_video_file(video_path):
                continue

            relative_path = str(video_path.relative_to(dataset_root))
            if is_excluded_video(relative_path):
                continue

            videos.append(video_path)

        class_to_videos[class_dir.name] = videos

    return class_to_videos


def choose_videos(class_to_videos: dict[str, list[Path]], max_videos_per_class: int | None) -> list[tuple[str, Path]]:
    selected: list[tuple[str, Path]] = []
    for class_name in sorted(class_to_videos):
        videos = class_to_videos[class_name]
        chosen = videos if max_videos_per_class is None else videos[:max_videos_per_class]
        selected.extend((class_name, video_path) for video_path in chosen)
    return selected


def ensure_model(model_path: Path) -> Path:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if model_path.exists():
        return model_path

    print(f"[INFO] Downloading MediaPipe hand model to: {model_path}")
    urllib.request.urlretrieve(MODEL_URL, model_path)
    return model_path


def create_detector(model_path: Path) -> vision.HandLandmarker:
    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
    )
    return vision.HandLandmarker.create_from_options(options)


def is_finite(value: float) -> bool:
    return math.isfinite(value)


def normalize_landmarks(raw_hand: list[tuple[float, float, float]]) -> tuple[bool, list[float], str]:
    """Same normalization and validation logic used in current SIGNEXA pipeline."""
    if len(raw_hand) != HAND_LANDMARK_COUNT:
        return False, [], f"Expected {HAND_LANDMARK_COUNT} landmarks, received {len(raw_hand)}"

    for x, y, z in raw_hand:
        if not (is_finite(x) and is_finite(y) and is_finite(z)):
            return False, [], "Invalid coordinate value"

    wrist = raw_hand[WRIST_INDEX]
    middle_mcp = raw_hand[MIDDLE_MCP_INDEX]

    scale = math.sqrt(
        (middle_mcp[0] - wrist[0]) ** 2
        + (middle_mcp[1] - wrist[1]) ** 2
        + (middle_mcp[2] - wrist[2]) ** 2
    )

    if not is_finite(scale) or scale <= MIN_SCALE:
        return False, [], "Scale reference is zero or invalid"

    normalized_points: list[tuple[float, float, float]] = []
    for x, y, z in raw_hand:
        nx = (x - wrist[0]) / scale
        ny = (y - wrist[1]) / scale
        nz = (z - wrist[2]) / scale

        if not (is_finite(nx) and is_finite(ny) and is_finite(nz)):
            return False, [], "Normalized coordinate is invalid"

        normalized_points.append((nx, ny, nz))

    feature_vector = [value for point in normalized_points for value in point]

    if len(feature_vector) != FEATURE_VECTOR_LENGTH:
        return False, [], f"Expected {FEATURE_VECTOR_LENGTH} features, received {len(feature_vector)}"

    if not all(is_finite(value) for value in feature_vector):
        return False, [], "Feature vector contains invalid values"

    return True, feature_vector, ""


def evenly_spaced_indices(total_frames: int, sample_count: int) -> list[int]:
    if total_frames <= 0 or sample_count <= 0:
        return []

    sample_count = min(sample_count, total_frames)
    indices = np.linspace(0, total_frames - 1, num=sample_count, dtype=int)
    return sorted({int(index) for index in indices.tolist()})


def count_frames_fallback(capture: cv2.VideoCapture) -> int:
    count = 0
    while True:
        ok, _ = capture.read()
        if not ok:
            break
        count += 1
    return count


def get_frame_timestamp_ms(capture: cv2.VideoCapture, frame_index: int, fps: float) -> float:
    timestamp_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
    if is_finite(timestamp_ms) and timestamp_ms > 0:
        return round(timestamp_ms, 3)

    if fps > 0:
        return round((frame_index / fps) * 1000.0, 3)

    return -1.0


def process_video(
    video_path: Path,
    dataset_root: Path,
    split: str,
    class_name: str,
    detector: vision.HandLandmarker,
    frames_per_video: int,
) -> tuple[VideoProcessingResult, list[FeatureRecord]]:
    relative_path = str(video_path.relative_to(dataset_root))
    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        return (
            VideoProcessingResult(
                split=split,
                class_name=class_name,
                label=class_name,
                filename=video_path.name,
                video_path=relative_path,
                sampled_frame_count=0,
                detected_frame_count=0,
                detection_rate=0.0,
                valid_normalized_frame_count=0,
                invalid_frame_count=0,
                feature_record_count=0,
                status="FAILED",
                error="Unreadable video or unsupported format",
            ),
            [],
        )

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)

    if total_frames <= 0:
        total_frames = count_frames_fallback(capture)
        capture.release()
        capture = cv2.VideoCapture(str(video_path))

    if total_frames <= 0:
        capture.release()
        return (
            VideoProcessingResult(
                split=split,
                class_name=class_name,
                label=class_name,
                filename=video_path.name,
                video_path=relative_path,
                sampled_frame_count=0,
                detected_frame_count=0,
                detection_rate=0.0,
                valid_normalized_frame_count=0,
                invalid_frame_count=0,
                feature_record_count=0,
                status="FAILED",
                error="No frames available in video",
            ),
            [],
        )

    indices = evenly_spaced_indices(total_frames, frames_per_video)
    sampled_frame_count = len(indices)
    detected_frame_count = 0
    valid_normalized_frame_count = 0
    invalid_frame_count = 0
    feature_records: list[FeatureRecord] = []
    errors: list[str] = []

    for frame_index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()

        if not ok or frame is None:
            invalid_frame_count += 1
            errors.append(f"Frame read failed at index {frame_index}")
            continue

        timestamp_ms = get_frame_timestamp_ms(capture, frame_index, fps)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        try:
            result = detector.detect(mp_image)
        except Exception as exc:  # noqa: BLE001
            invalid_frame_count += 1
            errors.append(f"MediaPipe detection failed at frame {frame_index}: {exc}")
            continue

        if not result.hand_landmarks:
            invalid_frame_count += 1
            continue

        detected_frame_count += 1
        frame_valid = False

        for hand_index, hand in enumerate(result.hand_landmarks):
            raw_hand = [(landmark.x, landmark.y, landmark.z) for landmark in hand]
            ok_norm, feature_vector, err = normalize_landmarks(raw_hand)

            if not ok_norm:
                errors.append(f"Normalization failed at frame {frame_index}, hand {hand_index}: {err}")
                continue

            frame_valid = True
            feature_records.append(
                FeatureRecord(
                    split=split,
                    class_name=class_name,
                    label=class_name,
                    video_path=relative_path,
                    filename=video_path.name,
                    frame_index=frame_index,
                    timestamp_ms=timestamp_ms,
                    hand_index=hand_index,
                    feature_vector=feature_vector,
                )
            )

        if frame_valid:
            valid_normalized_frame_count += 1
        else:
            invalid_frame_count += 1

    capture.release()

    detection_rate = (detected_frame_count / sampled_frame_count) if sampled_frame_count > 0 else 0.0

    if sampled_frame_count == 0:
        status = "FAILED"
        error = "No sampled frames"
    elif valid_normalized_frame_count == 0:
        status = "FAILED"
        error = "No valid normalized frames"
    elif errors:
        status = "PARTIAL"
        error = errors[0]
    else:
        status = "OK"
        error = ""

    report = VideoProcessingResult(
        split=split,
        class_name=class_name,
        label=class_name,
        filename=video_path.name,
        video_path=relative_path,
        sampled_frame_count=sampled_frame_count,
        detected_frame_count=detected_frame_count,
        detection_rate=round(detection_rate, 6),
        valid_normalized_frame_count=valid_normalized_frame_count,
        invalid_frame_count=invalid_frame_count,
        feature_record_count=len(feature_records),
        status=status,
        error=error,
    )

    return report, feature_records


def write_video_report_csv(path: Path, rows: Iterable[VideoProcessingResult]) -> None:
    fieldnames = [
        "split",
        "class_name",
        "label",
        "filename",
        "video_path",
        "sampled_frame_count",
        "detected_frame_count",
        "detection_rate",
        "valid_normalized_frame_count",
        "invalid_frame_count",
        "feature_record_count",
        "status",
        "error",
    ]

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_feature_csv(path: Path, rows: Iterable[FeatureRecord]) -> None:
    feature_columns = [f"feature_{index}" for index in range(FEATURE_VECTOR_LENGTH)]
    fieldnames = [
        "split",
        "class_name",
        "label",
        "video_path",
        "filename",
        "frame_index",
        "timestamp_ms",
        "hand_index",
        *feature_columns,
    ]

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            payload = {
                "split": row.split,
                "class_name": row.class_name,
                "label": row.label,
                "video_path": row.video_path,
                "filename": row.filename,
                "frame_index": row.frame_index,
                "timestamp_ms": row.timestamp_ms,
                "hand_index": row.hand_index,
            }
            payload.update({f"feature_{index}": row.feature_vector[index] for index in range(FEATURE_VECTOR_LENGTH)})
            writer.writerow(payload)


def read_feature_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    issues: list[str] = []

    if not path.exists():
        return [], ["Feature CSV does not exist"]

    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows.append(row)

    return rows, issues


def validate_feature_records(feature_rows: list[FeatureRecord], dataset_root: Path) -> list[str]:
    issues: list[str] = []

    for index, row in enumerate(feature_rows):
        if row.split not in VALID_SPLITS:
            issues.append(f"Record {index} has invalid split: {row.split}")

        if len(row.feature_vector) != FEATURE_VECTOR_LENGTH:
            issues.append(f"Record {index} has invalid feature length: {len(row.feature_vector)}")

        if not all(is_finite(value) for value in row.feature_vector):
            issues.append(f"Record {index} contains non-finite feature values")

        source_video = dataset_root / row.video_path
        if not source_video.exists():
            issues.append(f"Record {index} references missing source video: {row.video_path}")

        if is_excluded_video(row.video_path):
            issues.append(f"Record {index} references excluded corrupted path: {row.video_path}")

    return issues


def validate_feature_csv(path: Path) -> list[str]:
    rows, issues = read_feature_csv(path)
    if issues:
        return issues

    feature_columns = [f"feature_{index}" for index in range(FEATURE_VECTOR_LENGTH)]

    for row_index, row in enumerate(rows):
        for column in feature_columns:
            if column not in row:
                issues.append(f"CSV row {row_index} missing column {column}")
                continue

            value = row[column]
            try:
                float_value = float(value)
            except ValueError:
                issues.append(f"CSV row {row_index} has non-numeric value in {column}: {value}")
                continue

            if not is_finite(float_value):
                issues.append(f"CSV row {row_index} has non-finite value in {column}")

    return issues


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def to_count_map(items: Iterable[str]) -> dict[str, int]:
    count_map: dict[str, int] = {}
    for item in items:
        count_map[item] = count_map.get(item, 0) + 1
    return dict(sorted(count_map.items()))


def main() -> int:
    args = parse_args()
    dataset_path = args.dataset_path.resolve()

    if not dataset_path.exists() or not dataset_path.is_dir():
        print(f"[ERROR] Dataset path does not exist: {dataset_path}")
        return 1

    smoke_mode = not args.full

    if smoke_mode:
        output_dir = dataset_path / "smoke_test"
        report_csv_name = "smoke_test_report.csv"
        summary_json_name = "smoke_test_summary.json"
        feature_csv_name = "smoke_test_features.csv"
    else:
        output_dir = dataset_path / "features"
        report_csv_name = "extraction_report.csv"
        summary_json_name = "extraction_summary.json"
        feature_csv_name = "normalized_features.csv"

    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = args.model_path or (output_dir / "models" / "hand_landmarker.task")

    try:
        model_path = ensure_model(model_path)
        detector = create_detector(model_path)
    except Exception as exc:  # noqa: BLE001
        summary_payload = {
            "runMode": "smoke" if smoke_mode else "full",
            "datasetPath": str(dataset_path),
            "status": "FAILED",
            "error": f"MediaPipe initialization failure: {exc}",
            "results": [],
        }
        write_summary(output_dir / summary_json_name, summary_payload)
        print(f"[ERROR] {summary_payload['error']}")
        return 1

    splits = ["train"] if smoke_mode else ["train", "val", "test"]
    max_videos_per_class = max(args.max_videos_per_class, 0) if smoke_mode else None
    frames_per_video = max(args.frames_per_video, 1)

    all_video_reports: list[VideoProcessingResult] = []
    all_feature_records: list[FeatureRecord] = []
    selected_counts: dict[str, dict[str, int]] = {}

    for split in splits:
        split_dir = dataset_path / split
        class_to_videos = discover_class_videos(split_dir, dataset_path)
        selected = choose_videos(class_to_videos, max_videos_per_class)

        selected_counts[split] = {
            class_name: len(videos if max_videos_per_class is None else videos[:max_videos_per_class])
            for class_name, videos in sorted(class_to_videos.items())
        }

        for class_name, video_path in selected:
            report, feature_rows = process_video(
                video_path=video_path,
                dataset_root=dataset_path,
                split=split,
                class_name=class_name,
                detector=detector,
                frames_per_video=frames_per_video,
            )
            all_video_reports.append(report)
            all_feature_records.extend(feature_rows)

    report_csv_path = output_dir / report_csv_name
    feature_csv_path = output_dir / feature_csv_name
    summary_json_path = output_dir / summary_json_name

    write_video_report_csv(report_csv_path, all_video_reports)
    write_feature_csv(feature_csv_path, all_feature_records)

    in_memory_validation_issues = validate_feature_records(all_feature_records, dataset_path)
    csv_validation_issues = validate_feature_csv(feature_csv_path)
    validation_issues = in_memory_validation_issues + csv_validation_issues

    total_videos = len(all_video_reports)
    successful_videos = sum(1 for row in all_video_reports if row.status in {"OK", "PARTIAL"})
    failed_videos = total_videos - successful_videos
    total_sampled_frames = sum(row.sampled_frame_count for row in all_video_reports)
    total_detected_frames = sum(row.detected_frame_count for row in all_video_reports)
    total_valid_frames = sum(row.valid_normalized_frame_count for row in all_video_reports)
    total_invalid_frames = sum(row.invalid_frame_count for row in all_video_reports)
    detection_rate = (total_detected_frames / total_sampled_frames) if total_sampled_frames > 0 else 0.0

    low_detection_videos = [
        {
            "split": row.split,
            "class_name": row.class_name,
            "filename": row.filename,
            "video_path": row.video_path,
            "detection_rate": row.detection_rate,
        }
        for row in all_video_reports
        if row.sampled_frame_count > 0 and row.detection_rate < args.low_detection_threshold
    ]

    summary_payload = {
        "runMode": "smoke" if smoke_mode else "full",
        "datasetPath": str(dataset_path),
        "splits": splits,
        "maxVideosPerClass": max_videos_per_class,
        "framesPerVideo": frames_per_video,
        "modelPath": str(model_path),
        "selectedCounts": selected_counts,
        "aggregate": {
            "totalVideos": total_videos,
            "successfullyProcessedVideos": successful_videos,
            "completelyFailedVideos": failed_videos,
            "totalSampledFrames": total_sampled_frames,
            "detectedFrames": total_detected_frames,
            "detectionRate": round(detection_rate, 6),
            "validNormalizedFrames": total_valid_frames,
            "invalidFrames": total_invalid_frames,
            "totalFeatureRecords": len(all_feature_records),
        },
        "countsByClass": {
            "videoCount": to_count_map(row.class_name for row in all_video_reports),
            "featureRecordCount": to_count_map(row.class_name for row in all_feature_records),
        },
        "countsBySplit": {
            "videoCount": to_count_map(row.split for row in all_video_reports),
            "featureRecordCount": to_count_map(row.split for row in all_feature_records),
        },
        "lowDetectionVideos": low_detection_videos,
        "validation": {
            "issuesCount": len(validation_issues),
            "issues": validation_issues,
        },
        "outputFiles": {
            "featureCsv": str(feature_csv_path),
            "reportCsv": str(report_csv_path),
            "summaryJson": str(summary_json_path),
        },
    }

    write_summary(summary_json_path, summary_payload)

    print(f"[INFO] Feature CSV: {feature_csv_path}")
    print(f"[INFO] Video report CSV: {report_csv_path}")
    print(f"[INFO] Summary JSON: {summary_json_path}")
    print(
        "[INFO] Completed. "
        f"Videos: {total_videos}, "
        f"feature records: {len(all_feature_records)}, "
        f"validation issues: {len(validation_issues)}"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())