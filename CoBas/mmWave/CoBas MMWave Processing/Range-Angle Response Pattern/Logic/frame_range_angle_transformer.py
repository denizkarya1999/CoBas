"""One-time conversion of legacy clean frames to the selected display window."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from .config import (
    FRAME_WINDOW_MARKER_FILENAME,
    MAXIMUM_ANGLE_DEGREES,
    MAXIMUM_RANGE_METERS,
    MINIMUM_ANGLE_DEGREES,
    MINIMUM_RANGE_METERS,
    RANGE_BIN_SPACING_M,
    RANGE_FFT_SIZE,
)
from .reference_frame_generator import (
    BATTERY_FOLDER_PATTERN,
    ReferenceFrameResult,
    generate_random_reference,
)
from .session_logger import normalize_battery_level
from .video_frame_recorder import FRAMES_DIRECTORY, frame_directory_for_battery


# Physical extent used by frames collected before the window was narrowed.
LEGACY_MINIMUM_ANGLE_DEGREES = -90.0
LEGACY_MAXIMUM_ANGLE_DEGREES = 90.0
LEGACY_MINIMUM_RANGE_METERS = 0.0
LEGACY_MAXIMUM_RANGE_METERS = (RANGE_FFT_SIZE - 1) * RANGE_BIN_SPACING_M


@dataclass(frozen=True, slots=True)
class BatteryTransformResult:
    """Outcome of transforming one battery's clean images and reference."""

    battery_level_percent: int
    transformed_frame_count: int
    skipped_already_transformed: bool
    reference: ReferenceFrameResult


def transform_saved_battery_frames(
    battery_level_percent: str | int,
) -> BatteryTransformResult:
    """Crop legacy images in place and regenerate their one reference image."""
    percentage = normalize_battery_level(battery_level_percent)
    frames_directory = frame_directory_for_battery(percentage)
    marker = frames_directory / FRAME_WINDOW_MARKER_FILENAME

    if marker.exists():
        reference = generate_random_reference(percentage)
        return BatteryTransformResult(percentage, 0, True, reference)

    saved_frames = sorted(frames_directory.glob("frame_*.jpg"))
    if not saved_frames:
        raise FileNotFoundError(
            f"No legacy frames were found in {frames_directory}"
        )

    cv2 = _load_opencv()
    for frame_path in saved_frames:
        _transform_frame_in_place(cv2, frame_path)

    marker.write_text(
        "Legacy frames cropped to range 0.05-0.50 m and angle "
        "-60 to +60 degrees.\n",
        encoding="utf-8",
    )
    reference = generate_random_reference(percentage, replace_existing=True)
    return BatteryTransformResult(
        percentage,
        len(saved_frames),
        False,
        reference,
    )


def transform_all_saved_batteries() -> list[BatteryTransformResult]:
    """Transform each recognized battery folder exactly once."""
    if not FRAMES_DIRECTORY.is_dir():
        return []

    levels: list[int] = []
    for path in FRAMES_DIRECTORY.iterdir():
        match = BATTERY_FOLDER_PATTERN.fullmatch(path.name)
        if path.is_dir() and match:
            levels.append(normalize_battery_level(match.group(1)))

    return [
        transform_saved_battery_frames(level)
        for level in sorted(set(levels))
    ]


def _transform_frame_in_place(cv2, frame_path: Path) -> None:
    image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read saved frame {frame_path}")

    height, width = image.shape[:2]
    left = math.floor(
        (MINIMUM_ANGLE_DEGREES - LEGACY_MINIMUM_ANGLE_DEGREES)
        / (LEGACY_MAXIMUM_ANGLE_DEGREES - LEGACY_MINIMUM_ANGLE_DEGREES)
        * width
    )
    right = math.ceil(
        (MAXIMUM_ANGLE_DEGREES - LEGACY_MINIMUM_ANGLE_DEGREES)
        / (LEGACY_MAXIMUM_ANGLE_DEGREES - LEGACY_MINIMUM_ANGLE_DEGREES)
        * width
    )
    top = math.floor(
        (LEGACY_MAXIMUM_RANGE_METERS - MAXIMUM_RANGE_METERS)
        / (LEGACY_MAXIMUM_RANGE_METERS - LEGACY_MINIMUM_RANGE_METERS)
        * height
    )
    bottom = math.ceil(
        (LEGACY_MAXIMUM_RANGE_METERS - MINIMUM_RANGE_METERS)
        / (LEGACY_MAXIMUM_RANGE_METERS - LEGACY_MINIMUM_RANGE_METERS)
        * height
    )

    left = max(0, min(left, width - 1))
    right = max(left + 1, min(right, width))
    top = max(0, min(top, height - 1))
    bottom = max(top + 1, min(bottom, height))
    cropped = image[top:bottom, left:right]
    transformed = cv2.resize(
        cropped,
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )

    temporary_path = frame_path.with_name(
        f".{frame_path.stem}.window_transform.jpg"
    )
    try:
        if not cv2.imwrite(str(temporary_path), transformed):
            raise RuntimeError(f"Could not write transformed frame {frame_path}")
        temporary_path.replace(frame_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _load_opencv():
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            "OpenCV is required to transform existing saved frames."
        ) from error
    return cv2
