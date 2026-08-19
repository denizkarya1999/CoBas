"""Generate one calibrated, randomly selected reference image per battery."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure

from .config import (
    DISPLAY_DYNAMIC_RANGE_DB,
    MAXIMUM_ANGLE_DEGREES,
    MAXIMUM_RANGE_METERS,
    MINIMUM_ANGLE_DEGREES,
    MINIMUM_RANGE_METERS,
)
from .session_logger import normalize_battery_level
from .video_frame_recorder import FRAMES_DIRECTORY, frame_directory_for_battery

APPLICATION_DIRECTORY = Path(__file__).resolve().parents[1]
REFERENCES_DIRECTORY = APPLICATION_DIRECTORY / "References"
FRAME_FILENAME_PATTERN = "frame_*.jpg"
BATTERY_FOLDER_PATTERN = re.compile(r"^(\d+)_Percent$")


@dataclass(frozen=True, slots=True)
class ReferenceFrameResult:
    """Paths and status for one battery's reference image."""

    battery_level_percent: int
    selected_frame: Path | None
    reference_image: Path
    created: bool


def reference_path_for_battery(battery_level_percent: str | int) -> Path:
    """Return a fixed path that guarantees at most one reference per battery."""
    percentage = normalize_battery_level(battery_level_percent)
    return REFERENCES_DIRECTORY / f"{percentage}_Percent_Battery_Reference.jpg"


def generate_random_reference(
    battery_level_percent: str | int,
    replace_existing: bool = False,
    *,
    frames_directory: Path | None = None,
    references_directory: Path | None = None,
) -> ReferenceFrameResult:
    """Select one saved frame randomly and add calibrated plot references."""
    percentage = normalize_battery_level(battery_level_percent)
    output_path = (
        Path(references_directory) / f"{percentage}_Percent_Battery_Reference.jpg"
        if references_directory is not None
        else reference_path_for_battery(percentage)
    )

    # A fixed output name preserves the one-reference-per-battery rule. An
    # existing reference is kept instead of silently choosing a replacement.
    if output_path.exists() and not replace_existing:
        return ReferenceFrameResult(percentage, None, output_path, False)

    source_directory = (
        Path(frames_directory)
        if frames_directory is not None
        else frame_directory_for_battery(percentage)
    )
    saved_frames = sorted(source_directory.glob(FRAME_FILENAME_PATTERN))
    if not saved_frames:
        raise FileNotFoundError(
            f"No saved spectrogram frames were found in {source_directory}"
        )
    selected_frame = random.SystemRandom().choice(saved_frames)

    image = _read_reference_image(selected_frame)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _save_calibrated_reference(image, percentage, output_path)
    return ReferenceFrameResult(percentage, selected_frame, output_path, True)


def generate_references_for_saved_batteries() -> list[ReferenceFrameResult]:
    """Generate missing references for every recognized battery frame folder."""
    results: list[ReferenceFrameResult] = []
    if not FRAMES_DIRECTORY.is_dir():
        return results

    battery_levels: list[int] = []
    for path in FRAMES_DIRECTORY.iterdir():
        match = BATTERY_FOLDER_PATTERN.fullmatch(path.name)
        if path.is_dir() and match:
            battery_levels.append(normalize_battery_level(match.group(1)))

    for battery_level in sorted(set(battery_levels)):
        results.append(generate_random_reference(battery_level))
    return results


def _read_reference_image(path: Path) -> np.ndarray:
    try:
        from matplotlib import image as matplotlib_image

        image = matplotlib_image.imread(path)
    except Exception as error:
        raise RuntimeError(f"Could not read saved frame {path}") from error
    if image.ndim not in (2, 3):
        raise ValueError(f"Saved frame has an unsupported image shape: {image.shape}")
    return image


def _save_calibrated_reference(
    image: np.ndarray,
    battery_level_percent: int,
    output_path: Path,
) -> None:
    figure = Figure(figsize=(10.0, 4.8), dpi=150, constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.add_subplot(111)
    axes.imshow(
        image,
        origin="upper",
        aspect="auto",
        interpolation="bilinear",
        extent=(
            MINIMUM_ANGLE_DEGREES,
            MAXIMUM_ANGLE_DEGREES,
            MINIMUM_RANGE_METERS,
            MAXIMUM_RANGE_METERS,
        ),
    )
    axes.set_title(f"{battery_level_percent}% Battery Range-Angle Reference")
    axes.set_xlabel("Angle (degrees)")
    axes.set_ylabel("Range (meters)")
    axes.set_xlim(MINIMUM_ANGLE_DEGREES, MAXIMUM_ANGLE_DEGREES)
    axes.set_ylim(MINIMUM_RANGE_METERS, MAXIMUM_RANGE_METERS)
    axes.set_xticks(np.linspace(MINIMUM_ANGLE_DEGREES, MAXIMUM_ANGLE_DEGREES, 7))
    axes.set_yticks(np.linspace(MINIMUM_RANGE_METERS, MAXIMUM_RANGE_METERS, 7))

    # The clean source frame already contains Viridis RGB pixels. This separate
    # scalar mappable adds the calibrated legend without recoloring the image.
    energy_scale = ScalarMappable(
        cmap="viridis",
        norm=Normalize(vmin=-DISPLAY_DYNAMIC_RANGE_DB, vmax=0.0),
    )
    energy_scale.set_array([])
    colorbar = figure.colorbar(energy_scale, ax=axes, pad=0.025)
    colorbar.set_label("Energy (dB)")

    figure.savefig(output_path, format="jpg", dpi=150)
    figure.clear()
