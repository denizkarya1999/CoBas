"""Thread-safe active-recording clock that excludes user pauses."""

from __future__ import annotations

import math
import threading
import time


class PausableRecordingClock:
    """Measure only time spent actively recording across pause intervals."""

    def __init__(
        self,
        duration_seconds: float,
        started_at: float | None = None,
    ) -> None:
        duration = float(duration_seconds)
        if not math.isfinite(duration) or duration <= 0.0:
            raise ValueError("Recording duration must be greater than zero seconds")

        self.duration_seconds = duration
        self._lock = threading.Lock()
        self._elapsed_before_resume = 0.0
        self._resumed_at = time.monotonic() if started_at is None else started_at
        self._paused = False

    @property
    def is_paused(self) -> bool:
        with self._lock:
            return self._paused

    @property
    def elapsed_seconds(self) -> float:
        with self._lock:
            elapsed = self._elapsed_before_resume
            if not self._paused:
                elapsed += max(time.monotonic() - self._resumed_at, 0.0)
            return min(elapsed, self.duration_seconds)

    @property
    def remaining_seconds(self) -> float:
        return max(self.duration_seconds - self.elapsed_seconds, 0.0)

    def pause(self) -> float:
        """Pause the clock and return the captured active elapsed time."""
        with self._lock:
            if not self._paused:
                self._elapsed_before_resume = min(
                    self._elapsed_before_resume
                    + max(time.monotonic() - self._resumed_at, 0.0),
                    self.duration_seconds,
                )
                self._paused = True
            return self._elapsed_before_resume

    def resume(self) -> None:
        """Resume without counting time spent paused."""
        with self._lock:
            if self._paused:
                self._resumed_at = time.monotonic()
                self._paused = False
