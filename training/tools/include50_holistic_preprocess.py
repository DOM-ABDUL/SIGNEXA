#!/usr/bin/env python3
"""SIGNEXA INCLUDE-50 holistic feature extraction using MediaPipe Tasks API.

This script intentionally avoids legacy mp.solutions and uses MediaPipe Tasks
landmarkers for hand, pose, and face per frame.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

FEATURE_RANGE_MIN = -20.0
FEATURE_RANGE_MAX = 20.0
DEFAULT_FRAMES_PER_VIDEO = 8
DEFAULT_SMOKE_VIDEOS_PER_CLASS = 2

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
VALID_SPLITS = ("train", "val", "test")

# Keep compatibility with the previous holistic schema:
# face features include the first 468 mesh points only.
TARGET_FACE_LANDMARK_COUNT = 468

MODEL_URLS = {
    "hand_landmarker": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "pose_landmarker": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "face_landmarker": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
}

# This corrupted record was intentionally excluded from source metadata.
EXCLUDED_RELATIVE_PATH_SUFFIXES = {
    "places/19. house/adjectives/78. long/mvi_5106.mov",
}


@dataclass
class VideoSelection:
    split: str
    class_name: str
    video_path: Path


@dataclass
class FrameFeatureRecord:
    split: str
    class_name: str
    label: str
    video_path: str
    filename: str
    frame_index: int
    timestamp_ms: float
    features: list[float]


@dataclass
class VideoExtractionReport:
    split: str
    class_name: str
    video_path: str
    filename: str
    sampled_frame_count: int
    valid_frame_count: int
    left_hand_detection_rate: float
    right_hand_detection_rate: float
    pose_detection_rate: float
    face_detection_rate: float
    status: str
    error: str


@dataclass
class FeatureSchema:
    feature_dimension: int
    feature_names: list[str]
    landmark_groups: dict[str, int]
    normalization_method: str
    missing_landmark_strategy: str
    sampling_strategy: str
    mediapipe_configuration: dict[str, Any]
    coordinate_reference_system: str
    normalized_range: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SIGNEXA holistic feature preprocessing (MediaPipe Tasks)")
    parser.add_argument("dataset_path", type=Path, help="Path to dataset root containing train/ val/ test/")

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--smoke-test", action="store_true", help="Run deterministic smoke subset only")
    mode_group.add_argument("--full", action="store_true", help="Process full dataset across train/val/test")

    parser.add_argument(
        "--frames-per-video",
        type=int,
        default=DEFAULT_FRAMES_PER_VIDEO,
        help="Number of evenly spaced frames sampled per video (default: 8)",
    )
    parser.add_argument(
        "--smoke-videos-per-class",
        type=int,
        default=DEFAULT_SMOKE_VIDEOS_PER_CLASS,
        help="Videos per class for smoke mode (default: 2)",
    )
    return parser.parse_args()


def fail(message: str) -> None:
    raise ValueError(message)


def finite(value: float) -> bool:
    return math.isfinite(value)


def is_video_file(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTENSIONS


def is_excluded_video(relative_path: str) -> bool:
    normalized = relative_path.replace("\\", "/").strip().lower()
    return any(normalized.endswith(suffix) for suffix in EXCLUDED_RELATIVE_PATH_SUFFIXES)


def ensure_model_assets(models_dir: Path) -> dict[str, Path]:
    models_dir.mkdir(parents=True, exist_ok=True)
    local_paths: dict[str, Path] = {}

    for model_name, url in MODEL_URLS.items():
        filename = url.rsplit("/", maxsplit=1)[-1]
        local_path = models_dir / filename
        if not local_path.exists():
            print(f"[INFO] Downloading {model_name} model: {filename}")
            urllib.request.urlretrieve(url, local_path)
        local_paths[model_name] = local_path

    return local_paths


def create_tasks_landmarkers(model_paths: dict[str, Path]) -> tuple[vision.HandLandmarker, vision.PoseLandmarker, vision.FaceLandmarker]:
    hand_options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model_paths["hand_landmarker"])),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
    )
    pose_options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model_paths["pose_landmarker"])),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
    )
    face_options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model_paths["face_landmarker"])),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )

    hand = vision.HandLandmarker.create_from_options(hand_options)
    pose = vision.PoseLandmarker.create_from_options(pose_options)
    face = vision.FaceLandmarker.create_from_options(face_options)
    return hand, pose, face


def discover_dataset(dataset_path: Path) -> dict[str, dict[str, list[Path]]]:
    split_map: dict[str, dict[str, list[Path]]] = {split: {} for split in VALID_SPLITS}
    video_to_split: dict[str, str] = {}

    for split in VALID_SPLITS:
        split_dir = dataset_path / split
        if not split_dir.exists() or not split_dir.is_dir():
            continue

        for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            videos: list[Path] = []
            for candidate in sorted(class_dir.rglob("*")):
                if not candidate.is_file() or not is_video_file(candidate):
                    continue

                relative = str(candidate.relative_to(dataset_path))
                if is_excluded_video(relative):
                    continue

                if relative in video_to_split and video_to_split[relative] != split:
                    fail(
                        "Video split leakage detected. "
                        f"Video {relative} appears in both {video_to_split[relative]} and {split}."
                    )

                video_to_split[relative] = split
                videos.append(candidate)

            split_map[split][class_dir.name] = videos

    return split_map


def select_videos(split_map: dict[str, dict[str, list[Path]]], smoke_test: bool, smoke_videos_per_class: int) -> list[VideoSelection]:
    selections: list[VideoSelection] = []

    if smoke_test:
        train_classes = split_map.get("train", {})
        for class_name in sorted(train_classes.keys()):
            videos = train_classes[class_name][: max(smoke_videos_per_class, 0)]
            for video_path in videos:
                selections.append(VideoSelection(split="train", class_name=class_name, video_path=video_path))
        return selections

    for split in VALID_SPLITS:
        for class_name in sorted(split_map.get(split, {}).keys()):
            for video_path in split_map[split][class_name]:
                selections.append(VideoSelection(split=split, class_name=class_name, video_path=video_path))

    return selections


def build_feature_schema(face_landmark_count_used: int, face_landmark_count_detected: int) -> FeatureSchema:
    feature_names: list[str] = []

    for index in range(21):
        feature_names.extend([f"left_hand_{index}_x", f"left_hand_{index}_y", f"left_hand_{index}_z"])

    for index in range(21):
        feature_names.extend([f"right_hand_{index}_x", f"right_hand_{index}_y", f"right_hand_{index}_z"])

    for index in range(33):
        feature_names.extend([f"pose_{index}_x", f"pose_{index}_y", f"pose_{index}_z", f"pose_{index}_visibility"])

    for index in range(face_landmark_count_used):
        feature_names.extend([f"face_{index}_x", f"face_{index}_y", f"face_{index}_z"])

    feature_names.extend(["presence_left_hand", "presence_right_hand", "presence_pose", "presence_face"])

    return FeatureSchema(
        feature_dimension=len(feature_names),
        feature_names=feature_names,
        landmark_groups={
            "left_hand_landmarks": 21,
            "right_hand_landmarks": 21,
            "pose_landmarks": 33,
            "face_landmarks_detected": face_landmark_count_detected,
            "face_landmarks_used": face_landmark_count_used,
        },
        normalization_method=(
            "Reference-relative normalization using pose shoulder midpoint as origin when shoulders are available, "
            "with shoulder distance as scale; fallback to pose nose and unit scale."
        ),
        missing_landmark_strategy=(
            "Missing groups use zero-valued landmark blocks and explicit presence indicators. "
            "Face uses the first 468 landmarks for compatibility with previous schema."
        ),
        sampling_strategy="Evenly spaced deterministic frame sampling within each video.",
        mediapipe_configuration={
            "api": "mediapipe.tasks.python.vision",
            "landmarkers": ["HandLandmarker", "PoseLandmarker", "FaceLandmarker"],
            "running_mode": "IMAGE",
            "hand_num_hands": 2,
            "pose_num_poses": 1,
            "face_num_faces": 1,
            "model_files": {
                "hand_landmarker": MODEL_URLS["hand_landmarker"].rsplit("/", maxsplit=1)[-1],
                "pose_landmarker": MODEL_URLS["pose_landmarker"].rsplit("/", maxsplit=1)[-1],
                "face_landmarker": MODEL_URLS["face_landmarker"].rsplit("/", maxsplit=1)[-1],
            },
            "model_urls": MODEL_URLS,
        },
        coordinate_reference_system=(
            "MediaPipe normalized image coordinates transformed into shoulder-relative normalized coordinates."
        ),
        normalized_range={"min": FEATURE_RANGE_MIN, "max": FEATURE_RANGE_MAX},
    )


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
    if finite(timestamp_ms) and timestamp_ms > 0:
        return round(timestamp_ms, 3)
    if fps > 0:
        return round((frame_index / fps) * 1000.0, 3)
    return -1.0


def get_reference_from_pose(pose_landmarks: list[Any] | None) -> tuple[tuple[float, float, float], float]:
    if pose_landmarks and len(pose_landmarks) >= 13:
        left = pose_landmarks[11]
        right = pose_landmarks[12]

        if all(finite(value) for value in [left.x, left.y, left.z, right.x, right.y, right.z]):
            ref_x = (left.x + right.x) / 2.0
            ref_y = (left.y + right.y) / 2.0
            ref_z = (left.z + right.z) / 2.0
            scale = math.sqrt((left.x - right.x) ** 2 + (left.y - right.y) ** 2 + (left.z - right.z) ** 2)
            if finite(scale) and scale > 1e-6:
                return (ref_x, ref_y, ref_z), scale

        nose = pose_landmarks[0]
        if all(finite(value) for value in [nose.x, nose.y, nose.z]):
            return (nose.x, nose.y, nose.z), 1.0

    return (0.0, 0.0, 0.0), 1.0


def normalize_point(
    x: float,
    y: float,
    z: float,
    reference: tuple[float, float, float],
    scale: float,
) -> tuple[float, float, float]:
    if scale <= 1e-6:
        scale = 1.0
    return (
        (x - reference[0]) / scale,
        (y - reference[1]) / scale,
        (z - reference[2]) / scale,
    )


def extract_points_block(
    points: list[Any] | None,
    expected_count: int,
    include_visibility: bool,
    reference: tuple[float, float, float],
    scale: float,
) -> tuple[list[float], int]:
    if not points or len(points) < expected_count:
        size = expected_count * (4 if include_visibility else 3)
        return [0.0] * size, 0

    values: list[float] = []
    for index in range(expected_count):
        point = points[index]

        if not all(finite(value) for value in [point.x, point.y, point.z]):
            if include_visibility:
                values.extend([0.0, 0.0, 0.0, 0.0])
            else:
                values.extend([0.0, 0.0, 0.0])
            continue

        nx, ny, nz = normalize_point(point.x, point.y, point.z, reference, scale)

        if include_visibility:
            visibility = 0.0
            if hasattr(point, "visibility") and finite(float(point.visibility)):
                visibility = float(point.visibility)
            values.extend([nx, ny, nz, visibility])
        else:
            values.extend([nx, ny, nz])

    return values, 1


def assign_hands_by_handedness(hand_result: Any) -> tuple[list[Any] | None, list[Any] | None]:
    left_points: list[Any] | None = None
    right_points: list[Any] | None = None

    landmarks_list = getattr(hand_result, "hand_landmarks", []) or []
    handedness_list = getattr(hand_result, "handedness", []) or []

    for index, landmarks in enumerate(landmarks_list):
        side = "unknown"
        if index < len(handedness_list) and handedness_list[index]:
            category = handedness_list[index][0]
            side = str(getattr(category, "category_name", "unknown")).lower()

        if side == "left" and left_points is None:
            left_points = landmarks
        elif side == "right" and right_points is None:
            right_points = landmarks
        elif left_points is None:
            left_points = landmarks
        elif right_points is None:
            right_points = landmarks

    return left_points, right_points


def build_frame_feature_vector(
    hand_result: Any,
    pose_result: Any,
    face_result: Any,
    face_landmark_count_used: int,
) -> tuple[list[float], dict[str, int], int]:
    pose_landmarks = pose_result.pose_landmarks[0] if pose_result.pose_landmarks else None
    reference, scale = get_reference_from_pose(pose_landmarks)

    left_points, right_points = assign_hands_by_handedness(hand_result)

    left_values, left_present = extract_points_block(
        left_points,
        expected_count=21,
        include_visibility=False,
        reference=reference,
        scale=scale,
    )
    right_values, right_present = extract_points_block(
        right_points,
        expected_count=21,
        include_visibility=False,
        reference=reference,
        scale=scale,
    )
    pose_values, pose_present = extract_points_block(
        pose_landmarks,
        expected_count=33,
        include_visibility=True,
        reference=reference,
        scale=scale,
    )

    face_points_detected = face_result.face_landmarks[0] if face_result.face_landmarks else None
    detected_face_count = len(face_points_detected) if face_points_detected else 0
    face_values, face_present = extract_points_block(
        face_points_detected,
        expected_count=face_landmark_count_used,
        include_visibility=False,
        reference=reference,
        scale=scale,
    )

    feature_vector = [
        *left_values,
        *right_values,
        *pose_values,
        *face_values,
        float(left_present),
        float(right_present),
        float(pose_present),
        float(face_present),
    ]

    presence = {
        "left_hand": left_present,
        "right_hand": right_present,
        "pose": pose_present,
        "face": face_present,
    }

    return feature_vector, presence, detected_face_count


def validate_feature_vector(feature_vector: list[float], expected_dimension: int) -> str:
    if len(feature_vector) != expected_dimension:
        return f"Feature dimension mismatch: expected {expected_dimension}, got {len(feature_vector)}"

    for value in feature_vector:
        if not finite(value):
            return "Feature vector contains NaN or Infinity"
        if value < FEATURE_RANGE_MIN or value > FEATURE_RANGE_MAX:
            return (
                "Feature value out of expected normalized range "
                f"[{FEATURE_RANGE_MIN}, {FEATURE_RANGE_MAX}]"
            )

    return ""


def extract_video(
    selection: VideoSelection,
    dataset_path: Path,
    hand_landmarker: vision.HandLandmarker,
    pose_landmarker: vision.PoseLandmarker,
    face_landmarker: vision.FaceLandmarker,
    frames_per_video: int,
    expected_dimension: int,
    face_landmark_count_used: int,
) -> tuple[VideoExtractionReport, list[FrameFeatureRecord], int]:
    relative_video_path = str(selection.video_path.relative_to(dataset_path))
    capture = cv2.VideoCapture(str(selection.video_path))
    if not capture.isOpened():
        return (
            VideoExtractionReport(
                split=selection.split,
                class_name=selection.class_name,
                video_path=relative_video_path,
                filename=selection.video_path.name,
                sampled_frame_count=0,
                valid_frame_count=0,
                left_hand_detection_rate=0.0,
                right_hand_detection_rate=0.0,
                pose_detection_rate=0.0,
                face_detection_rate=0.0,
                status="FAILED",
                error="Unreadable video or unsupported format",
            ),
            [],
            0,
        )

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)

    if total_frames <= 0:
        total_frames = count_frames_fallback(capture)
        capture.release()
        capture = cv2.VideoCapture(str(selection.video_path))

    if total_frames <= 0:
        capture.release()
        return (
            VideoExtractionReport(
                split=selection.split,
                class_name=selection.class_name,
                video_path=relative_video_path,
                filename=selection.video_path.name,
                sampled_frame_count=0,
                valid_frame_count=0,
                left_hand_detection_rate=0.0,
                right_hand_detection_rate=0.0,
                pose_detection_rate=0.0,
                face_detection_rate=0.0,
                status="FAILED",
                error="No frames in video",
            ),
            [],
            0,
        )

    frame_indices = evenly_spaced_indices(total_frames, max(frames_per_video, 1))
    sampled_frame_count = len(frame_indices)
    feature_rows: list[FrameFeatureRecord] = []

    left_detected = 0
    right_detected = 0
    pose_detected = 0
    face_detected = 0
    max_face_count_seen = 0
    errors: list[str] = []

    for frame_index in frame_indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok or frame is None:
            errors.append(f"Frame read failed at index {frame_index}")
            continue

        timestamp_ms = get_frame_timestamp_ms(capture, frame_index, fps)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        try:
            hand_result = hand_landmarker.detect(mp_image)
            pose_result = pose_landmarker.detect(mp_image)
            face_result = face_landmarker.detect(mp_image)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Tasks processing failed at frame {frame_index}: {exc}")
            continue

        feature_vector, presence, face_count = build_frame_feature_vector(
            hand_result=hand_result,
            pose_result=pose_result,
            face_result=face_result,
            face_landmark_count_used=face_landmark_count_used,
        )

        max_face_count_seen = max(max_face_count_seen, face_count)

        validation_error = validate_feature_vector(feature_vector, expected_dimension)
        if validation_error:
            errors.append(f"Frame {frame_index} invalid: {validation_error}")
            continue

        left_detected += presence["left_hand"]
        right_detected += presence["right_hand"]
        pose_detected += presence["pose"]
        face_detected += presence["face"]

        feature_rows.append(
            FrameFeatureRecord(
                split=selection.split,
                class_name=selection.class_name,
                label=selection.class_name,
                video_path=relative_video_path,
                filename=selection.video_path.name,
                frame_index=frame_index,
                timestamp_ms=timestamp_ms,
                features=feature_vector,
            )
        )

    capture.release()

    valid_frame_count = len(feature_rows)
    left_rate = left_detected / sampled_frame_count if sampled_frame_count > 0 else 0.0
    right_rate = right_detected / sampled_frame_count if sampled_frame_count > 0 else 0.0
    pose_rate = pose_detected / sampled_frame_count if sampled_frame_count > 0 else 0.0
    face_rate = face_detected / sampled_frame_count if sampled_frame_count > 0 else 0.0

    status = "OK"
    error = ""
    if valid_frame_count == 0:
        status = "FAILED"
        error = errors[0] if errors else "No valid frames"
    elif errors:
        status = "PARTIAL"
        error = errors[0]

    return (
        VideoExtractionReport(
            split=selection.split,
            class_name=selection.class_name,
            video_path=relative_video_path,
            filename=selection.video_path.name,
            sampled_frame_count=sampled_frame_count,
            valid_frame_count=valid_frame_count,
            left_hand_detection_rate=round(left_rate, 6),
            right_hand_detection_rate=round(right_rate, 6),
            pose_detection_rate=round(pose_rate, 6),
            face_detection_rate=round(face_rate, 6),
            status=status,
            error=error,
        ),
        feature_rows,
        max_face_count_seen,
    )


def write_feature_csv(path: Path, rows: list[FrameFeatureRecord], schema: FeatureSchema) -> None:
    feature_columns = [f"feature_{index}" for index in range(schema.feature_dimension)]
    fieldnames = [
        "split",
        "class_name",
        "label",
        "video_path",
        "filename",
        "frame_index",
        "timestamp_ms",
        *feature_columns,
    ]

    with path.open("w", encoding="utf-8", newline="") as file:
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
            }
            for index, value in enumerate(row.features):
                payload[f"feature_{index}"] = value
            writer.writerow(payload)


def write_video_report_csv(path: Path, rows: list[VideoExtractionReport]) -> None:
    fieldnames = [
        "split",
        "class_name",
        "video_path",
        "filename",
        "sampled_frame_count",
        "valid_frame_count",
        "left_hand_detection_rate",
        "right_hand_detection_rate",
        "pose_detection_rate",
        "face_detection_rate",
        "status",
        "error",
    ]

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def validate_outputs(rows: list[FrameFeatureRecord], schema: FeatureSchema) -> list[str]:
    issues: list[str] = []

    split_by_video: dict[str, str] = {}
    class_by_video: dict[str, str] = {}
    frame_indices_by_video: dict[str, list[int]] = {}

    for row_index, row in enumerate(rows):
        if row.split not in VALID_SPLITS:
            issues.append(f"Row {row_index} has invalid split {row.split}")

        if len(row.features) != schema.feature_dimension:
            issues.append(f"Row {row_index} feature dimension mismatch")

        for value in row.features:
            if not finite(value):
                issues.append(f"Row {row_index} has NaN/Infinity feature value")
                break
            if value < FEATURE_RANGE_MIN or value > FEATURE_RANGE_MAX:
                issues.append(f"Row {row_index} has feature value outside normalized range")
                break

        if row.video_path in split_by_video and split_by_video[row.video_path] != row.split:
            issues.append(f"Video {row.video_path} appears in multiple splits")
        split_by_video[row.video_path] = row.split

        if row.video_path in class_by_video and class_by_video[row.video_path] != row.class_name:
            issues.append(f"Video {row.video_path} has multiple class labels")
        class_by_video[row.video_path] = row.class_name

        frame_indices_by_video.setdefault(row.video_path, []).append(row.frame_index)

        if is_excluded_video(row.video_path):
            issues.append(f"Excluded corrupted metadata row found in output: {row.video_path}")

    for video_path, indices in frame_indices_by_video.items():
        if indices != sorted(indices):
            issues.append(f"Frame indices are not monotonically ordered for video {video_path}")

    return issues


def count_by_key(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def main() -> int:
    args = parse_args()
    dataset_path = args.dataset_path.resolve()

    if not dataset_path.exists() or not dataset_path.is_dir():
        print(f"[ERROR] Dataset path does not exist: {dataset_path}")
        return 1

    try:
        split_map = discover_dataset(dataset_path)
        selected_videos = select_videos(
            split_map,
            smoke_test=args.smoke_test,
            smoke_videos_per_class=max(args.smoke_videos_per_class, 0),
        )
    except ValueError as exc:
        print(f"[ERROR] Dataset integrity validation failed: {exc}")
        return 1

    if not selected_videos:
        print("[ERROR] No videos selected for processing")
        return 1

    output_dir = dataset_path / "holistic_features"
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_csv_path = output_dir / "holistic_features.csv"
    report_csv_path = output_dir / "holistic_extraction_report.csv"
    summary_json_path = output_dir / "holistic_extraction_summary.json"
    schema_json_path = output_dir / "holistic_feature_schema.json"

    models_dir = output_dir / "models"
    try:
        model_paths = ensure_model_assets(models_dir)
        hand_landmarker, pose_landmarker, face_landmarker = create_tasks_landmarkers(model_paths)
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] MediaPipe Tasks initialization failed: {exc}")
        return 1

    # Keep compatibility with previous experiment schema by using first 468 face points.
    face_landmark_count_used = TARGET_FACE_LANDMARK_COUNT
    max_face_landmark_count_detected = 0

    schema = build_feature_schema(
        face_landmark_count_used=face_landmark_count_used,
        face_landmark_count_detected=face_landmark_count_used,
    )
    print(f"[INFO] Holistic feature dimension: {schema.feature_dimension}")

    all_feature_rows: list[FrameFeatureRecord] = []
    all_video_reports: list[VideoExtractionReport] = []

    try:
        for selection in selected_videos:
            report, feature_rows, face_count_seen = extract_video(
                selection=selection,
                dataset_path=dataset_path,
                hand_landmarker=hand_landmarker,
                pose_landmarker=pose_landmarker,
                face_landmarker=face_landmarker,
                frames_per_video=max(args.frames_per_video, 1),
                expected_dimension=schema.feature_dimension,
                face_landmark_count_used=face_landmark_count_used,
            )
            all_video_reports.append(report)
            all_feature_rows.extend(feature_rows)
            max_face_landmark_count_detected = max(max_face_landmark_count_detected, face_count_seen)
    finally:
        hand_landmarker.close()
        pose_landmarker.close()
        face_landmarker.close()

    # Update schema with observed face landmark count from runtime detections.
    schema.landmark_groups["face_landmarks_detected"] = max_face_landmark_count_detected

    write_feature_csv(feature_csv_path, all_feature_rows, schema)
    write_video_report_csv(report_csv_path, all_video_reports)

    validation_issues = validate_outputs(all_feature_rows, schema)

    sequence_lengths = [report.valid_frame_count for report in all_video_reports]
    sequence_length_stats = {
        "min": min(sequence_lengths) if sequence_lengths else 0,
        "max": max(sequence_lengths) if sequence_lengths else 0,
        "median": float(statistics.median(sequence_lengths)) if sequence_lengths else 0.0,
        "mean": float(sum(sequence_lengths) / len(sequence_lengths)) if sequence_lengths else 0.0,
    }

    left_rates = [report.left_hand_detection_rate for report in all_video_reports]
    right_rates = [report.right_hand_detection_rate for report in all_video_reports]
    pose_rates = [report.pose_detection_rate for report in all_video_reports]
    face_rates = [report.face_detection_rate for report in all_video_reports]

    summary_payload = {
        "run_mode": "smoke_test" if args.smoke_test else "full",
        "dataset_path": str(dataset_path),
        "mediapipe_version": getattr(mp, "__version__", "unknown"),
        "tasks_apis_used": ["HandLandmarker", "PoseLandmarker", "FaceLandmarker"],
        "model_assets": {name: str(path) for name, path in model_paths.items()},
        "selected_video_count": len(selected_videos),
        "videos_processed": len(all_video_reports),
        "successful_videos": sum(1 for row in all_video_reports if row.status in {"OK", "PARTIAL"}),
        "failed_videos": sum(1 for row in all_video_reports if row.status == "FAILED"),
        "total_feature_rows": len(all_feature_rows),
        "feature_dimension": schema.feature_dimension,
        "counts_by_split": count_by_key([row.split for row in all_feature_rows]),
        "counts_by_class": count_by_key([row.class_name for row in all_feature_rows]),
        "video_counts_by_split": count_by_key([row.split for row in all_video_reports]),
        "sequence_length_stats": sequence_length_stats,
        "detection_rate_summary": {
            "left_hand_mean": float(sum(left_rates) / len(left_rates)) if left_rates else 0.0,
            "right_hand_mean": float(sum(right_rates) / len(right_rates)) if right_rates else 0.0,
            "pose_mean": float(sum(pose_rates) / len(pose_rates)) if pose_rates else 0.0,
            "face_mean": float(sum(face_rates) / len(face_rates)) if face_rates else 0.0,
        },
        "validation": {
            "issues_count": len(validation_issues),
            "issues": validation_issues,
            "status": "PASS" if len(validation_issues) == 0 else "FAIL",
        },
        "output_files": {
            "holistic_features_csv": str(feature_csv_path),
            "holistic_extraction_report_csv": str(report_csv_path),
            "holistic_extraction_summary_json": str(summary_json_path),
            "holistic_feature_schema_json": str(schema_json_path),
        },
    }

    write_json(schema_json_path, asdict(schema))
    write_json(summary_json_path, summary_payload)

    print(f"[INFO] Feature CSV: {feature_csv_path}")
    print(f"[INFO] Report CSV: {report_csv_path}")
    print(f"[INFO] Summary JSON: {summary_json_path}")
    print(f"[INFO] Schema JSON: {schema_json_path}")
    print(
        "[INFO] Extraction summary: "
        f"videos={len(all_video_reports)}, feature_rows={len(all_feature_rows)}, validation_issues={len(validation_issues)}"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())