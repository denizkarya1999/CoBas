"""Externally controlled mmWave capture used by the CoBas battery workflow.

CoBas owns chirp duration and battery-position transitions.  This bridge only
opens the radar, captures while CoBas marks a position active, and writes the
Range-Angle application's existing logs, clean frames, and reference image.
"""

from __future__ import annotations

import queue
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from PIL import Image

APP_DIRECTORY = Path(__file__).resolve().parents[1]
COBAS_SOURCE_ROOT = APP_DIRECTORY.parent
RANGE_ANGLE_DIRECTORY = (
    COBAS_SOURCE_ROOT
    / "mmWave"
    / "CoBas MMWave Processing"
    / "Range-Angle Response Pattern"
)

if not RANGE_ANGLE_DIRECTORY.is_dir():
    raise RuntimeError(
        f"Range-Angle Response Pattern source was not found at {RANGE_ANGLE_DIRECTORY}"
    )

range_angle_path = str(RANGE_ANGLE_DIRECTORY)
if range_angle_path not in sys.path:
    sys.path.insert(0, range_angle_path)

from Logic.config import (
    FRAME_WINDOW_MARKER_FILENAME,
    SPECTROGRAM_FRAME_RATE,
)
from Logic.range_angle_processor import (
    RangeAngleFrame,
    RangeAngleProcessor,
)
from Logic.raw_iq_source import RawIQFrameSource
from Logic.reference_frame_generator import (
    generate_random_reference,
)
from Logic.session_logger import RangeAngleSessionLogger
from Logic.video_frame_recorder import (
    TemporarySpectrogramVideoRecorder,
)

EventKind = Literal["status", "ready", "frame", "error", "stopped"]


@dataclass(frozen=True, slots=True)
class MMWaveCaptureEvent:
    """One thread-safe radar event consumed by the Tkinter application."""

    kind: EventKind
    payload: str | RangeAngleFrame


class _CleanFrameWriter:
    """Save the existing clean spectrogram rendering at the dataset rate."""

    def __init__(self, frames_directory: Path) -> None:
        self.frames_directory = frames_directory
        self.frame_count = 0
        self._next_frame_at: float | None = None
        self._cv2 = None

    def validate_preconditions(self) -> None:
        try:
            import cv2
        except ImportError as error:
            raise RuntimeError(
                "OpenCV is required to save mmWave spectrogram frames"
            ) from error

        self._cv2 = cv2
        if self.frames_directory.exists() and any(
            self.frames_directory.glob("frame_*.jpg")
        ):
            raise FileExistsError(
                f"mmWave frames already exist: {self.frames_directory}"
            )
        self.frames_directory.mkdir(parents=True, exist_ok=True)

    def resume(self) -> None:
        """Make the first response after a position change immediately due."""
        self._next_frame_at = None

    def write_if_due(self, frame: RangeAngleFrame) -> bool:
        if self._cv2 is None:
            raise RuntimeError("mmWave frame writer was not initialized")

        now = time.monotonic()
        if self._next_frame_at is not None and now < self._next_frame_at:
            return False

        image = TemporarySpectrogramVideoRecorder._render_clean_spectrogram(
            self._cv2,
            frame,
        )
        self.frame_count += 1
        output_path = self.frames_directory / f"frame_{self.frame_count:06d}.jpg"
        if not self._cv2.imwrite(str(output_path), image):
            raise RuntimeError(f"Could not save mmWave frame: {output_path}")

        self._next_frame_at = now + (1.0 / SPECTROGRAM_FRAME_RATE)
        return True

    def finish(self) -> None:
        if not self.frame_count:
            return
        marker = self.frames_directory / FRAME_WINDOW_MARKER_FILENAME
        marker.touch(exist_ok=True)

    @staticmethod
    def preview_image(frame: RangeAngleFrame) -> Image.Image:
        try:
            import cv2
        except ImportError as error:
            raise RuntimeError(
                "OpenCV is required to display mmWave spectrogram frames"
            ) from error

        bgr = TemporarySpectrogramVideoRecorder._render_clean_spectrogram(
            cv2,
            frame,
        )
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)


class MMWaveCaptureService:
    """Capture one battery session under external chirp/position control."""

    def __init__(
        self,
        battery_level_percent: int,
        output_directory: str | Path,
    ) -> None:
        self.battery_level_percent = int(battery_level_percent)
        self.output_directory = Path(output_directory)
        self.logs_directory = self.output_directory / "Logs"
        self.frames_directory = self.output_directory / "Frames"
        self.references_directory = self.output_directory / "References"
        self.events: queue.Queue[MMWaveCaptureEvent] = queue.Queue(maxsize=32)

        self._processor = RangeAngleProcessor()
        self._frame_writer = _CleanFrameWriter(self.frames_directory)
        self._stop_event = threading.Event()
        self._capture_enabled = threading.Event()
        self._discard_requested = threading.Event()
        self._ready_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._position_number: int | None = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def is_ready(self) -> bool:
        return self._ready_event.is_set() and self.is_running

    @property
    def position_number(self) -> int | None:
        return self._position_number

    def start(self) -> None:
        if self.is_running:
            return
        self._stop_event.clear()
        self._capture_enabled.clear()
        self._discard_requested.clear()
        self._ready_event.clear()
        self._position_number = None
        self._processor.reset()
        self._thread = threading.Thread(
            target=self._run,
            name="cobas-mmwave-capture",
            daemon=True,
        )
        self._thread.start()

    def resume(self, position_number: int) -> None:
        if not self.is_ready:
            raise RuntimeError("mmWave radar is not ready")
        self._position_number = int(position_number)
        self._discard_requested.set()
        self._processor.reset()
        self._frame_writer.resume()
        self._capture_enabled.set()
        self._publish(
            MMWaveCaptureEvent(
                "status",
                f"Capturing mmWave Position {self._position_number}",
            )
        )

    def pause(self) -> None:
        self._capture_enabled.clear()
        self._discard_requested.set()
        self._position_number = None
        if self.is_running:
            self._publish(MMWaveCaptureEvent("status", "mmWave capture paused"))

    def stop(self, wait: bool = False, timeout: float = 10.0) -> None:
        self._capture_enabled.clear()
        self._stop_event.set()
        if wait and self._thread is not None:
            self._thread.join(timeout=timeout)

    def preview_image(self, frame: RangeAngleFrame) -> Image.Image:
        return self._frame_writer.preview_image(frame)

    def _publish(self, event: MMWaveCaptureEvent) -> None:
        try:
            self.events.put_nowait(event)
        except queue.Full:
            try:
                self.events.get_nowait()
            except queue.Empty:
                pass
            self.events.put_nowait(event)

    def _run(self) -> None:
        session_name = f"{self.battery_level_percent}_Percent_Battery"
        self._publish(
            MMWaveCaptureEvent(
                "status",
                "Connecting and configuring IWR6843AOP radar...",
            )
        )
        try:
            self._frame_writer.validate_preconditions()
            with (
                RawIQFrameSource() as source,
                RangeAngleSessionLogger(
                    self.battery_level_percent,
                    session_name=session_name,
                    log_directory=self.logs_directory,
                ) as logger,
            ):
                self._ready_event.set()
                self._publish(
                    MMWaveCaptureEvent(
                        "ready",
                        "IWR6843AOP radar ready",
                    )
                )

                while not self._stop_event.is_set():
                    if not self._capture_enabled.wait(timeout=0.1):
                        continue

                    if self._discard_requested.is_set():
                        source.discard_pending_data()
                        self._discard_requested.clear()

                    for raw_frame in source.read_frames():
                        if (
                            self._stop_event.is_set()
                            or not self._capture_enabled.is_set()
                        ):
                            break
                        processed = self._processor.process(raw_frame)
                        logger.write_frame(raw_frame, processed)
                        self._frame_writer.write_if_due(processed)
                        self._publish(MMWaveCaptureEvent("frame", processed))

            self._frame_writer.finish()
            if self._frame_writer.frame_count:
                generate_random_reference(
                    self.battery_level_percent,
                    frames_directory=self.frames_directory,
                    references_directory=self.references_directory,
                )
            self._publish(
                MMWaveCaptureEvent(
                    "stopped",
                    f"Saved {self._frame_writer.frame_count} mmWave frame(s)",
                )
            )
        except Exception as error:  # noqa: BLE001 - worker boundary
            self._publish(MMWaveCaptureEvent("error", str(error)))
        finally:
            self._capture_enabled.clear()
            self._ready_event.clear()
