"""Record clean spectrogram pixels temporarily and extract them as frames."""

from __future__ import annotations

import math
import queue
import tempfile
from dataclasses import dataclass
from pathlib import Path
from threading import Event

import numpy as np

from .config import (
    DISPLAY_DYNAMIC_RANGE_DB,
    FRAME_WINDOW_MARKER_FILENAME,
    SPECTROGRAM_FRAME_HEIGHT,
    SPECTROGRAM_FRAME_RATE,
    SPECTROGRAM_FRAME_WIDTH,
    SPECTROGRAM_VIDEO_CODEC,
)
from .range_angle_processor import RangeAngleFrame
from .recording_clock import PausableRecordingClock
from .session_logger import normalize_battery_level


APPLICATION_DIRECTORY = Path(__file__).resolve().parents[1]
FRAMES_DIRECTORY = APPLICATION_DIRECTORY / "Frames"


def normalize_recording_duration(requested_seconds: str | int | float) -> float:
    """Validate a positive number of spectrogram-recording seconds."""
    try:
        duration = float(str(requested_seconds).strip())
    except ValueError as error:
        raise ValueError("Video duration must be a number of seconds") from error
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("Video duration must be greater than zero seconds")
    return duration


def frame_directory_for_battery(battery_level_percent: str | int) -> Path:
    """Return a filesystem-safe folder such as Frames/20_Percent."""
    percentage = normalize_battery_level(battery_level_percent)
    return FRAMES_DIRECTORY / f"{percentage}_Percent"


@dataclass(frozen=True, slots=True)
class VideoFrameResult:
    """Summary of one temporary recording and extraction operation."""

    requested_duration_seconds: float
    recorded_duration_seconds: float
    frame_count: int
    frames_directory: Path


class TemporarySpectrogramVideoRecorder:
    """Write heatmaps to a temporary AVI, extract JPEGs, and delete the AVI."""

    def __init__(
        self,
        battery_level_percent: str | int,
        duration_seconds: str | int | float,
    ) -> None:
        self.battery_level_percent = normalize_battery_level(
            battery_level_percent
        )
        self.duration_seconds = normalize_recording_duration(duration_seconds)
        self.frames_directory = frame_directory_for_battery(
            self.battery_level_percent
        )

    def validate_preconditions(self) -> None:
        """Fail before radar log creation if video output cannot be produced."""
        self._load_opencv()
        if self.frames_directory.exists():
            raise FileExistsError(
                f"Frame folder already exists: {self.frames_directory}"
            )

    def record_and_extract(
        self,
        frame_queue: queue.Queue[RangeAngleFrame],
        stop_event: Event,
        recording_started_at: float | None = None,
        recording_clock: PausableRecordingClock | None = None,
    ) -> VideoFrameResult:
        """Record queued response maps for the duration and extract each frame."""
        cv2 = self._load_opencv()
        self.validate_preconditions()
        FRAMES_DIRECTORY.mkdir(parents=True, exist_ok=True)

        clock = recording_clock or PausableRecordingClock(
            self.duration_seconds,
            recording_started_at,
        )

        with tempfile.TemporaryDirectory(prefix="mmwave_spectrogram_") as temporary:
            video_path = Path(temporary) / "temporary_spectrogram.avi"
            written_frames, elapsed = self._write_temporary_video(
                cv2,
                video_path,
                frame_queue,
                stop_event,
                clock,
            )
            if written_frames == 0:
                return VideoFrameResult(
                    requested_duration_seconds=self.duration_seconds,
                    recorded_duration_seconds=elapsed,
                    frame_count=0,
                    frames_directory=self.frames_directory,
                )

            extracted_frames = self._extract_frames(cv2, video_path)
            self._write_window_marker()

        # TemporaryDirectory removes the intermediate AVI after extraction.
        return VideoFrameResult(
            requested_duration_seconds=self.duration_seconds,
            recorded_duration_seconds=elapsed,
            frame_count=extracted_frames,
            frames_directory=self.frames_directory,
        )

    def _write_temporary_video(
        self,
        cv2,
        video_path: Path,
        frame_queue: queue.Queue[RangeAngleFrame],
        stop_event: Event,
        recording_clock: PausableRecordingClock,
    ) -> tuple[int, float]:
        fourcc = cv2.VideoWriter_fourcc(*SPECTROGRAM_VIDEO_CODEC)
        writer = cv2.VideoWriter(
            str(video_path),
            fourcc,
            SPECTROGRAM_FRAME_RATE,
            (SPECTROGRAM_FRAME_WIDTH, SPECTROGRAM_FRAME_HEIGHT),
        )
        if not writer.isOpened():
            writer.release()
            raise RuntimeError(
                f"Could not create temporary video with "
                f"{SPECTROGRAM_VIDEO_CODEC} codec"
            )

        frame_interval_seconds = 1.0 / SPECTROGRAM_FRAME_RATE
        next_frame_at = 0.0
        frame_count = 0
        try:
            while True:
                if recording_clock.is_paused:
                    if stop_event.wait(0.05):
                        break
                    continue

                elapsed = recording_clock.elapsed_seconds
                if elapsed >= self.duration_seconds:
                    break
                queue_timeout = min(0.1, self.duration_seconds - elapsed)

                if stop_event.is_set() and frame_queue.empty():
                    break

                try:
                    response_frame = frame_queue.get(timeout=queue_timeout)
                except queue.Empty:
                    continue

                if recording_clock.is_paused:
                    # A boundary may have been reached while queue.get was
                    # waiting. Do not carry that stale frame into the next
                    # battery position.
                    continue

                elapsed = recording_clock.elapsed_seconds
                if elapsed >= self.duration_seconds:
                    break
                if elapsed < next_frame_at:
                    # Radar responses can arrive faster than the requested
                    # video rate. Keep only the first response at each 0.5 s
                    # boundary so a 2 FPS recording also extracts at 2 FPS.
                    continue

                writer.write(self._render_clean_spectrogram(cv2, response_frame))
                frame_count += 1
                while next_frame_at <= elapsed:
                    next_frame_at += frame_interval_seconds
        finally:
            writer.release()

        return frame_count, recording_clock.elapsed_seconds

    @staticmethod
    def _render_clean_spectrogram(cv2, frame: RangeAngleFrame) -> np.ndarray:
        """Render only heatmap pixels—no title, axes, ticks, or colorbar."""
        normalized = (
            np.clip(
                (frame.power_db + DISPLAY_DYNAMIC_RANGE_DB)
                / DISPLAY_DYNAMIC_RANGE_DB,
                0.0,
                1.0,
            )
            * 255.0
        ).astype(np.uint8)

        # The GUI uses origin="lower". Flip the matrix so low ranges appear at
        # the bottom of an ordinary top-origin video/image coordinate system.
        colored = cv2.applyColorMap(np.flipud(normalized), cv2.COLORMAP_VIRIDIS)
        return cv2.resize(
            colored,
            (SPECTROGRAM_FRAME_WIDTH, SPECTROGRAM_FRAME_HEIGHT),
            interpolation=cv2.INTER_LINEAR,
        )

    def _extract_frames(self, cv2, video_path: Path) -> int:
        """Decode every temporary-video image into a numbered JPEG file."""
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError("Could not reopen the temporary spectrogram video")

        self.frames_directory.mkdir(parents=False, exist_ok=False)
        frame_number = 0
        try:
            while True:
                success, image = capture.read()
                if not success:
                    break
                frame_number += 1
                frame_path = self.frames_directory / f"frame_{frame_number:06d}.jpg"
                if not cv2.imwrite(str(frame_path), image):
                    raise RuntimeError(f"Could not write extracted frame {frame_path}")
        finally:
            capture.release()

        if frame_number == 0:
            raise RuntimeError("The temporary video did not contain readable frames")
        return frame_number

    def _write_window_marker(self) -> None:
        """Mark new frames as already using the configured physical window."""
        marker = self.frames_directory / FRAME_WINDOW_MARKER_FILENAME
        marker.write_text(
            "Frames use range 0.20-0.50 m and angle -60 to +60 degrees.\n",
            encoding="utf-8",
        )

    @staticmethod
    def _load_opencv():
        try:
            import cv2
        except ImportError as error:
            raise RuntimeError(
                "OpenCV is required for spectrogram video recording. "
                "Install requirements.txt."
            ) from error
        return cv2
