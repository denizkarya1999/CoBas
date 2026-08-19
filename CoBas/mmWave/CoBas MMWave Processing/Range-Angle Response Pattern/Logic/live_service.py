"""Background acquisition service that keeps hardware work out of the GUI."""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Literal

from .config import EVENT_QUEUE_SIZE, SPECTROGRAM_FRAME_QUEUE_SIZE
from .range_angle_processor import RangeAngleFrame, RangeAngleProcessor
from .raw_iq_source import RawIQFrameSource
from .recording_clock import PausableRecordingClock
from .reference_frame_generator import generate_random_reference
from .session_logger import RangeAngleSessionLogger
from .video_frame_recorder import TemporarySpectrogramVideoRecorder


EventKind = Literal["status", "frame", "timer", "position_change", "error"]


def normalize_position_count(requested_count: str | int) -> int:
    """Validate the number of equal battery-position recording intervals."""
    text = str(requested_count).strip()
    try:
        count = int(text)
    except ValueError as error:
        raise ValueError(
            "Number of battery positions must be a whole number"
        ) from error
    if str(count) != text and text != f"+{count}":
        raise ValueError("Number of battery positions must be a whole number")
    if count <= 0:
        raise ValueError("Number of battery positions must be at least 1")
    return count


@dataclass(frozen=True, slots=True)
class SessionTimer:
    """Elapsed and remaining time for the duration-controlled capture."""

    elapsed_seconds: float
    remaining_seconds: float
    total_seconds: float


@dataclass(frozen=True, slots=True)
class PositionChangeRequest:
    """Instruction to move from one completed position to the next."""

    completed_position: int
    next_position: int
    total_positions: int
    elapsed_seconds: float
    remaining_seconds: float


@dataclass(frozen=True, slots=True)
class LiveEvent:
    kind: EventKind
    payload: str | RangeAngleFrame | SessionTimer | PositionChangeRequest


class LiveRangeAngleService:
    """Own the worker thread, radar source, and latest-frame event queue."""

    def __init__(self) -> None:
        self.events: queue.Queue[LiveEvent] = queue.Queue(maxsize=EVENT_QUEUE_SIZE)
        self._processor = RangeAngleProcessor()
        self._stop_event = threading.Event()
        self._position_change_confirmed = threading.Event()
        self._thread: threading.Thread | None = None
        self._video_thread: threading.Thread | None = None
        self._video_frames: queue.Queue[RangeAngleFrame] = queue.Queue(
            maxsize=SPECTROGRAM_FRAME_QUEUE_SIZE
        )
        self._video_error: str | None = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(
        self,
        battery_level_percent: int,
        video_duration_seconds: float,
        position_count: int = 1,
    ) -> None:
        if self.is_running:
            return
        positions = normalize_position_count(position_count)
        self._processor.reset()
        self._stop_event.clear()
        self._position_change_confirmed.clear()
        self._video_error = None
        self._video_thread = None
        self._video_frames = queue.Queue(maxsize=SPECTROGRAM_FRAME_QUEUE_SIZE)
        self._thread = threading.Thread(
            target=self._run,
            args=(battery_level_percent, video_duration_seconds, positions),
            name="range-angle-capture",
            daemon=True,
        )
        self._thread.start()

    def stop(self, wait: bool = False, timeout: float = 3.0) -> None:
        self._stop_event.set()
        self._position_change_confirmed.set()
        if wait and self._thread is not None:
            self._thread.join(timeout=timeout)

    def confirm_position_change(self) -> None:
        """Continue recording after the user has repositioned the battery."""
        self._position_change_confirmed.set()

    def _publish(self, event: LiveEvent) -> None:
        # Keep the UI current instead of accumulating stale heatmap frames.
        try:
            self.events.put_nowait(event)
        except queue.Full:
            try:
                self.events.get_nowait()
            except queue.Empty:
                pass
            self.events.put_nowait(event)

    def _run(
        self,
        battery_level_percent: int,
        video_duration_seconds: float,
        position_count: int,
    ) -> None:
        self._publish(LiveEvent("status", "Connecting and configuring radar..."))
        failed = False
        try:
            video_recorder = TemporarySpectrogramVideoRecorder(
                battery_level_percent,
                video_duration_seconds,
            )
            video_recorder.validate_preconditions()
            with (
                RawIQFrameSource() as source,
                RangeAngleSessionLogger(battery_level_percent) as logger,
            ):
                recording_clock = PausableRecordingClock(
                    video_recorder.duration_seconds
                )
                self._video_thread = threading.Thread(
                    target=self._record_spectrogram_video,
                    args=(video_recorder, recording_clock),
                    name="spectrogram-video-capture",
                    daemon=False,
                )
                self._video_thread.start()
                self._publish_timer(recording_clock)
                self._publish(
                    LiveEvent(
                        "status",
                        f"Live USB1 stream connected — logging "
                        f"'{logger.paths.session_name}' and recording "
                        f"position 1/{position_count} for a total of "
                        f"{video_recorder.duration_seconds:g} s",
                    )
                )
                next_position = 2
                while not self._stop_event.is_set():
                    elapsed = recording_clock.elapsed_seconds
                    if elapsed >= video_recorder.duration_seconds:
                        break

                    change_at = (
                        video_recorder.duration_seconds
                        * (next_position - 1)
                        / position_count
                        if next_position <= position_count
                        else None
                    )
                    if change_at is not None and elapsed >= change_at:
                        if not self._pause_for_position_change(
                            source,
                            recording_clock,
                            next_position,
                            position_count,
                        ):
                            break
                        next_position += 1
                        continue

                    for raw_frame in source.read_frames():
                        elapsed = recording_clock.elapsed_seconds
                        if (
                            self._stop_event.is_set()
                            or elapsed >= video_recorder.duration_seconds
                        ):
                            break
                        if change_at is not None and elapsed >= change_at:
                            # Restart the read loop after the pause so no frames
                            # already parsed for the old position are retained.
                            break
                        processed = self._processor.process(raw_frame)
                        logger.write_frame(raw_frame, processed)
                        self._offer_video_frame(processed)
                        self._publish(LiveEvent("frame", processed))
                    self._publish_timer(recording_clock)
                self._publish_timer(recording_clock)
        except Exception as error:
            failed = True
            self._publish(LiveEvent("error", str(error)))
        finally:
            self._stop_event.set()
            if self._video_thread is not None:
                self._video_thread.join()
            if not failed and self._video_error is None:
                self._publish(LiveEvent("status", "Stopped"))

    def _offer_video_frame(self, frame: RangeAngleFrame) -> None:
        """Keep the recorder current without blocking USB1 acquisition."""
        if self._video_thread is None or not self._video_thread.is_alive():
            return
        try:
            self._video_frames.put_nowait(frame)
        except queue.Full:
            try:
                self._video_frames.get_nowait()
            except queue.Empty:
                pass
            self._video_frames.put_nowait(frame)

    def _record_spectrogram_video(
        self,
        recorder: TemporarySpectrogramVideoRecorder,
        recording_clock: PausableRecordingClock,
    ) -> None:
        try:
            result = recorder.record_and_extract(
                self._video_frames,
                self._stop_event,
                recording_clock=recording_clock,
            )
            reference_result = None
            if result.frame_count:
                reference_result = generate_random_reference(
                    recorder.battery_level_percent
                )
            reference_message = (
                f"; reference: {reference_result.reference_image}"
                if reference_result is not None
                else ""
            )
            self._publish(
                LiveEvent(
                    "status",
                    f"Saved {result.frame_count} clean spectrogram frame(s) to "
                    f"{result.frames_directory}{reference_message}",
                )
            )
        except Exception as error:
            self._video_error = f"Spectrogram video failed: {error}"
            self._publish(LiveEvent("error", self._video_error))
            self._stop_event.set()

    def _pause_for_position_change(
        self,
        source: RawIQFrameSource,
        recording_clock: PausableRecordingClock,
        next_position: int,
        total_positions: int,
    ) -> bool:
        """Pause all recording until the UI confirms the battery was moved."""
        elapsed = recording_clock.pause()
        self._clear_video_frames()
        self._position_change_confirmed.clear()
        self._publish_timer(recording_clock)
        self._publish(
            LiveEvent(
                "position_change",
                PositionChangeRequest(
                    completed_position=next_position - 1,
                    next_position=next_position,
                    total_positions=total_positions,
                    elapsed_seconds=elapsed,
                    remaining_seconds=recording_clock.remaining_seconds,
                ),
            )
        )

        while not self._stop_event.is_set():
            if self._position_change_confirmed.wait(timeout=0.1):
                break
        if self._stop_event.is_set():
            return False

        # Data produced while the battery was being moved must not become the
        # first samples of the next position.
        source.discard_pending_data()
        self._clear_video_frames()
        self._processor.reset()
        recording_clock.resume()
        self._publish(
            LiveEvent(
                "status",
                f"Recording battery position {next_position}/{total_positions}",
            )
        )
        return True

    def _clear_video_frames(self) -> None:
        while True:
            try:
                self._video_frames.get_nowait()
            except queue.Empty:
                return

    def _publish_timer(self, recording_clock: PausableRecordingClock) -> None:
        elapsed = recording_clock.elapsed_seconds
        total_seconds = recording_clock.duration_seconds
        self._publish(
            LiveEvent(
                "timer",
                SessionTimer(
                    elapsed_seconds=elapsed,
                    remaining_seconds=max(total_seconds - elapsed, 0.0),
                    total_seconds=total_seconds,
                ),
            )
        )
