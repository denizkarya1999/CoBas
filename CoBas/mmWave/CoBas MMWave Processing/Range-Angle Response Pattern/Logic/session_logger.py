"""Paired CSV logging for raw complex samples and displayed response maps."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Self, TextIO

from .config import DISPLAY_RANGE_BIN_INDICES
from .range_angle_processor import RangeAngleFrame
from .raw_iq_source import IQFrame

APPLICATION_DIRECTORY = Path(__file__).resolve().parents[1]
LOG_DIRECTORY = APPLICATION_DIRECTORY / "Logs"
RAW_IQ_LOG_DIRECTORY = LOG_DIRECTORY / "Raw IQ Signals"
RANGE_ANGLE_LOG_DIRECTORY = LOG_DIRECTORY / "Range-Angle Responses"
LOGGED_RANGE_BINS = frozenset(DISPLAY_RANGE_BIN_INDICES)


@dataclass(frozen=True, slots=True)
class SessionLogPaths:
    """The two files produced for one named live session."""

    session_name: str
    raw_iq_csv: Path
    range_angle_csv: Path


def normalize_session_name(requested_name: str) -> str:
    """Return a safe filename stem while preserving readable spaces."""
    name = requested_name.strip()
    if name.lower().endswith(".csv"):
        name = name[:-4].rstrip()
    if not name:
        raise ValueError("Log name cannot be empty")
    if name in {".", ".."} or "/" in name or "\\" in name:
        raise ValueError("Log name must be a filename, not a path")
    if any(ord(character) < 32 for character in name):
        raise ValueError("Log name cannot contain control characters")
    return name


def session_log_paths(
    requested_name: str,
    log_directory: Path | None = None,
) -> SessionLogPaths:
    """Resolve matching CSV names in the two dedicated log directories."""
    session_name = normalize_session_name(requested_name)
    filename = f"{session_name}.csv"
    root = Path(log_directory) if log_directory is not None else LOG_DIRECTORY
    return SessionLogPaths(
        session_name=session_name,
        raw_iq_csv=root / "Raw IQ Signals" / filename,
        range_angle_csv=root / "Range-Angle Responses" / filename,
    )


def normalize_battery_level(requested_level: str | int) -> int:
    """Convert values such as ``20`` or ``20%`` to an integer percentage."""
    text = str(requested_level).strip()
    if text.endswith("%"):
        text = text[:-1].strip()
    try:
        percentage = int(text)
    except ValueError as error:
        raise ValueError(
            "Battery level must be a whole percentage, such as 20%"
        ) from error
    if not 0 <= percentage <= 100:
        raise ValueError("Battery level must be between 0% and 100%")
    return percentage


def battery_session_log_paths(
    battery_level_percent: str | int,
) -> SessionLogPaths:
    """Resolve paired names such as ``20_Percent_Battery.csv``."""
    percentage = normalize_battery_level(battery_level_percent)
    return session_log_paths(f"{percentage}_Percent_Battery")


class RangeAngleSessionLogger:
    """Write the minimum I/Q rows and exact display matrix for every frame."""

    RAW_IQ_HEADER = (
        "frame_number",
        "range_bin",
        "virtual_antenna",
        "i",
        "q",
    )

    def __init__(
        self,
        battery_level_percent: str | int | None = None,
        *,
        session_name: str | None = None,
        log_directory: Path | None = None,
    ) -> None:
        if session_name is None:
            if battery_level_percent is None:
                raise ValueError("A battery level or explicit session name is required")
            self.battery_level_percent = normalize_battery_level(battery_level_percent)
            session_name = f"{self.battery_level_percent}_Percent_Battery"
        else:
            self.battery_level_percent = (
                normalize_battery_level(battery_level_percent)
                if battery_level_percent is not None
                else None
            )
        self.paths = session_log_paths(session_name, log_directory)
        self._raw_stream: TextIO | None = None
        self._response_stream: TextIO | None = None
        self._raw_writer = None
        self._response_writer = None
        self._response_header_written = False

    def __enter__(self) -> Self:
        self.paths.raw_iq_csv.parent.mkdir(parents=True, exist_ok=True)
        self.paths.range_angle_csv.parent.mkdir(parents=True, exist_ok=True)

        # Exclusive creation protects earlier sessions from silent overwrite.
        self._raw_stream = self.paths.raw_iq_csv.open(
            "x",
            encoding="utf-8",
            newline="",
        )
        try:
            self._response_stream = self.paths.range_angle_csv.open(
                "x",
                encoding="utf-8",
                newline="",
            )
        except BaseException:
            self._raw_stream.close()
            # This raw file was created by the immediately preceding open, so
            # roll it back if its paired response file cannot be created.
            self.paths.raw_iq_csv.unlink(missing_ok=True)
            raise

        self._raw_writer = csv.writer(self._raw_stream)
        self._response_writer = csv.writer(self._response_stream)
        self._raw_writer.writerow(self.RAW_IQ_HEADER)
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._raw_stream is not None:
            self._raw_stream.close()
        if self._response_stream is not None:
            self._response_stream.close()

    def write_frame(
        self,
        raw_frame: IQFrame,
        response_frame: RangeAngleFrame,
    ) -> None:
        """Append matching raw and display-ready data, then flush both files."""
        if self._raw_writer is None or self._response_writer is None:
            raise RuntimeError("Session logger is not open")
        if raw_frame.frame_number != response_frame.frame_number:
            raise ValueError("Raw and range-angle frame numbers do not match")

        self._raw_writer.writerows(
            (
                sample.frame_number,
                sample.range_bin,
                sample.virtual_antenna,
                sample.i,
                sample.q,
            )
            for sample in raw_frame.samples
            if sample.range_bin in LOGGED_RANGE_BINS
        )

        # Write the response header just before the first matrix because the
        # processor owns the authoritative angle grid. The angle grid is fixed
        # for a session, so it only needs to be encoded once as CSV columns.
        if not self._response_header_written:
            self._write_response_header(response_frame)
            self._response_header_written = True

        for range_meters, power_row in zip(
            response_frame.ranges_meters,
            response_frame.power_db,
        ):
            self._response_writer.writerow(
                (
                    response_frame.frame_number,
                    f"{float(range_meters):.6f}",
                    *(f"{float(value):.3f}" for value in power_row),
                )
            )

        # A completed radar frame is the durability boundary for both files.
        if self._raw_stream is not None:
            self._raw_stream.flush()
        if self._response_stream is not None:
            self._response_stream.flush()

    def _write_response_header(self, response_frame: RangeAngleFrame) -> None:
        if self._response_writer is None:
            raise RuntimeError("Range-angle CSV writer is not open")
        angle_columns = (
            f"power_db_at_{float(angle):+.1f}_deg"
            for angle in response_frame.angles_degrees
        )
        self._response_writer.writerow(("frame_number", "range_meters", *angle_columns))
