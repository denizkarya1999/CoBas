"""Externally controlled mmWave capture used by the CoBas battery workflow.

CoBas owns chirp duration and battery-position transitions. This bridge opens
and validates the radar stream, captures while CoBas marks a position active,
and writes the Range-Angle application's logs, clean frames, and reference.
"""

from __future__ import annotations

import queue
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Literal

import numpy as np
from PIL import Image, ImageDraw, ImageFont

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

from Logic.config import (  # noqa: E402
    DISPLAY_DYNAMIC_RANGE_DB,
    FRAME_WINDOW_MARKER_FILENAME,
    MAXIMUM_ANGLE_DEGREES,
    MAXIMUM_RANGE_METERS,
    MINIMUM_ANGLE_DEGREES,
    MINIMUM_RANGE_METERS,
    SPECTROGRAM_FRAME_RATE,
)
from Logic.range_angle_processor import (  # noqa: E402
    RangeAngleFrame,
    RangeAngleProcessor,
)
from Logic.raw_iq_source import (  # noqa: E402
    FIRST_IQ_FRAME_TIMEOUT_SECONDS,
    RawIQFrameSource,
)
from Logic.reference_frame_generator import (  # noqa: E402
    generate_random_reference,
)
from Logic.session_logger import RangeAngleSessionLogger  # noqa: E402
from Logic.video_frame_recorder import (  # noqa: E402
    TemporarySpectrogramVideoRecorder,
)

EventKind = Literal["status", "ready", "frame", "error", "stopped"]

PREVIEW_BACKGROUND = (2, 6, 23)
PREVIEW_FOREGROUND = (226, 232, 240)
PREVIEW_MUTED = (148, 163, 184)
PREVIEW_GRID = (65, 78, 98)
PREVIEW_WIDTH = 900
PREVIEW_HEIGHT = 440
HEATMAP_BOUNDS = (86, 52, 746, 344)
COLORBAR_BOUNDS = (790, 52, 814, 344)


@dataclass(frozen=True, slots=True)
class MMWaveCaptureEvent:
    """One thread-safe radar event consumed by the Tkinter application."""

    kind: EventKind
    payload: str | RangeAngleFrame


class _CleanFrameWriter:
    """Save clean ML frames and render a calibrated live spectrogram."""

    _font_cache: ClassVar[dict] = {}

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
        marker.write_text(
            "Frames use range 0.20-0.50 m and angle -60 to +60 degrees.\n",
            encoding="utf-8",
        )

    @classmethod
    def _font(cls, size: int, bold: bool = False):
        key = (size, bold)
        if key in cls._font_cache:
            return cls._font_cache[key]
        font_name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
        try:
            font = ImageFont.truetype(font_name, size)
        except OSError:
            font = ImageFont.load_default()
        cls._font_cache[key] = font
        return font

    @classmethod
    def _draw_axes(cls, image: Image.Image, frame: RangeAngleFrame) -> None:
        draw = ImageDraw.Draw(image)
        left, top, right, bottom = HEATMAP_BOUNDS
        tick_font = cls._font(13)
        label_font = cls._font(15, bold=True)
        detail_font = cls._font(13)

        draw.text(
            (left, 15),
            "Live mmWave Range-Angle Spectrogram",
            fill=PREVIEW_FOREGROUND,
            font=cls._font(18, bold=True),
        )

        for angle in np.linspace(
            MINIMUM_ANGLE_DEGREES,
            MAXIMUM_ANGLE_DEGREES,
            7,
        ):
            fraction = (angle - MINIMUM_ANGLE_DEGREES) / (
                MAXIMUM_ANGLE_DEGREES - MINIMUM_ANGLE_DEGREES
            )
            x = round(left + fraction * (right - left))
            draw.line((x, top, x, bottom), fill=PREVIEW_GRID, width=1)
            label = f"{angle:.0f}°"
            box = draw.textbbox((0, 0), label, font=tick_font)
            draw.text(
                (x - (box[2] - box[0]) / 2, bottom + 7),
                label,
                fill=PREVIEW_MUTED,
                font=tick_font,
            )

        for distance in np.linspace(
            MINIMUM_RANGE_METERS,
            MAXIMUM_RANGE_METERS,
            7,
        ):
            fraction = (distance - MINIMUM_RANGE_METERS) / (
                MAXIMUM_RANGE_METERS - MINIMUM_RANGE_METERS
            )
            y = round(bottom - fraction * (bottom - top))
            draw.line((left, y, right, y), fill=PREVIEW_GRID, width=1)
            label = f"{distance:.2f}"
            box = draw.textbbox((0, 0), label, font=tick_font)
            draw.text(
                (left - (box[2] - box[0]) - 9, y - (box[3] - box[1]) / 2),
                label,
                fill=PREVIEW_MUTED,
                font=tick_font,
            )

        draw.rectangle(HEATMAP_BOUNDS, outline=PREVIEW_FOREGROUND, width=1)
        x_label = "Angle (degrees)"
        x_box = draw.textbbox((0, 0), x_label, font=label_font)
        draw.text(
            ((left + right - (x_box[2] - x_box[0])) / 2, bottom + 31),
            x_label,
            fill=PREVIEW_FOREGROUND,
            font=label_font,
        )

        y_label = Image.new("RGBA", (170, 25), (0, 0, 0, 0))
        y_draw = ImageDraw.Draw(y_label)
        y_draw.text(
            (0, 1),
            "Range (meters)",
            fill=(*PREVIEW_FOREGROUND, 255),
            font=label_font,
        )
        y_label = y_label.rotate(90, expand=True)
        image.paste(
            y_label,
            (11, round((top + bottom - y_label.height) / 2)),
            y_label,
        )

        draw.text(
            (left, PREVIEW_HEIGHT - 24),
            (
                f"Frame {frame.frame_number}  •  Peak "
                f"{frame.peak_range_meters:.2f} m, "
                f"{frame.peak_angle_degrees:+.1f}°  •  "
                f"Array {frame.beamforming_channel_count}/"
                f"{frame.input_antenna_count} ch"
            ),
            fill=PREVIEW_MUTED,
            font=detail_font,
        )

    @classmethod
    def _draw_colorbar(cls, image: Image.Image, cv2) -> None:
        left, top, right, bottom = COLORBAR_BOUNDS
        gradient = np.linspace(255, 0, bottom - top + 1, dtype=np.uint8)
        gradient = np.repeat(gradient[:, np.newaxis], right - left + 1, axis=1)
        colored = cv2.applyColorMap(gradient, cv2.COLORMAP_VIRIDIS)
        colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        image.paste(Image.fromarray(colored), (left, top))

        draw = ImageDraw.Draw(image)
        tick_font = cls._font(12)
        draw.rectangle(COLORBAR_BOUNDS, outline=PREVIEW_FOREGROUND, width=1)
        for power_db in np.linspace(0.0, -DISPLAY_DYNAMIC_RANGE_DB, 6):
            fraction = -power_db / DISPLAY_DYNAMIC_RANGE_DB
            y = round(top + fraction * (bottom - top))
            draw.line((right, y, right + 6, y), fill=PREVIEW_FOREGROUND, width=1)
            draw.text(
                (right + 10, y - 7),
                f"{power_db:.0f}",
                fill=PREVIEW_MUTED,
                font=tick_font,
            )
        draw.text(
            (left - 14, bottom + 12),
            "Relative Power\n(dB)",
            fill=PREVIEW_FOREGROUND,
            font=cls._font(11, bold=True),
            spacing=1,
            align="center",
        )

    @classmethod
    def preview_image(cls, frame: RangeAngleFrame) -> Image.Image:
        """Return a fast live plot with physical axes and a dB reference."""
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
        left, top, right, bottom = HEATMAP_BOUNDS
        heatmap = Image.fromarray(rgb).resize(
            (right - left + 1, bottom - top + 1),
            Image.Resampling.BILINEAR,
        )
        preview = Image.new(
            "RGB",
            (PREVIEW_WIDTH, PREVIEW_HEIGHT),
            PREVIEW_BACKGROUND,
        )
        preview.paste(heatmap, (left, top))
        cls._draw_axes(preview, frame)
        cls._draw_colorbar(preview, cv2)
        return preview


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

    @staticmethod
    def _stream_failure_message(source: RawIQFrameSource, phase: str) -> str:
        if source.total_packets:
            return (
                f"mmWave USB1 {phase}, but {source.total_packets} packet(s) "
                "contained no compatible complex range-I/Q TLV. Confirm the "
                "IWR6843AOP is running the SDK 3.x out-of-box firmware."
            )
        return (
            f"No mmWave USB1 binary frames {phase}. Check radar power, firmware, "
            "USB cable, /dev/ttyUSB0 and /dev/ttyUSB1 assignments, and serial "
            "permissions."
        )

    def _wait_until_streaming(
        self,
        source: RawIQFrameSource,
        timeout_seconds: float = FIRST_IQ_FRAME_TIMEOUT_SECONDS,
    ) -> bool:
        """Require a valid I/Q frame before reporting the radar as ready."""
        deadline = time.monotonic() + timeout_seconds
        while not self._stop_event.is_set():
            if source.read_frames():
                return True
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    self._stream_failure_message(source, "did not become ready")
                )
        return False

    def _run(self) -> None:
        session_name = f"{self.battery_level_percent}_Percent_Battery"
        self._publish(
            MMWaveCaptureEvent(
                "status",
                "Connecting, configuring, and validating IWR6843AOP radar...",
            )
        )
        try:
            self._frame_writer.validate_preconditions()
            with RawIQFrameSource() as source:
                if not self._wait_until_streaming(source):
                    return

                with RangeAngleSessionLogger(
                    self.battery_level_percent,
                    session_name=session_name,
                    log_directory=self.logs_directory,
                ) as logger:
                    self._ready_event.set()
                    self._publish(
                        MMWaveCaptureEvent(
                            "ready",
                            "IWR6843AOP radar ready; valid I/Q stream confirmed",
                        )
                    )

                    frame_deadline: float | None = None
                    while not self._stop_event.is_set():
                        if not self._capture_enabled.wait(timeout=0.1):
                            frame_deadline = None
                            continue

                        if self._discard_requested.is_set():
                            source.discard_pending_data()
                            self._discard_requested.clear()
                            frame_deadline = (
                                time.monotonic() + FIRST_IQ_FRAME_TIMEOUT_SECONDS
                            )

                        raw_frames = source.read_frames()
                        if raw_frames:
                            frame_deadline = (
                                time.monotonic() + FIRST_IQ_FRAME_TIMEOUT_SECONDS
                            )

                        for raw_frame in raw_frames:
                            if (
                                self._stop_event.is_set()
                                or not self._capture_enabled.is_set()
                            ):
                                break
                            processed = self._processor.process(raw_frame)
                            logger.write_frame(raw_frame, processed)
                            self._frame_writer.write_if_due(processed)
                            self._publish(MMWaveCaptureEvent("frame", processed))

                        if (
                            frame_deadline is not None
                            and time.monotonic() >= frame_deadline
                        ):
                            raise RuntimeError(
                                self._stream_failure_message(
                                    source,
                                    "stopped during capture",
                                )
                            )

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
