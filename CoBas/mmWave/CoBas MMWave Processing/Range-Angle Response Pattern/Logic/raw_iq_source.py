"""Adapter that reuses the sibling Raw IQ Signals USB reader and parser."""

from __future__ import annotations

import sys
from pathlib import Path
from types import TracebackType


COBAS_DIRECTORY = Path(__file__).resolve().parents[2]
RAW_IQ_DIRECTORY = COBAS_DIRECTORY / "Raw IQ Signals"
RAW_IQ_LOGIC_FILE = RAW_IQ_DIRECTORY / "iq_logic.py"

if not RAW_IQ_LOGIC_FILE.is_file():
    raise RuntimeError(f"Raw IQ logic was not found at {RAW_IQ_LOGIC_FILE}")

# The source directory contains spaces, so add the directory containing
# iq_logic.py to Python's module search path before importing it normally.
raw_iq_path = str(RAW_IQ_DIRECTORY)
if raw_iq_path not in sys.path:
    sys.path.insert(0, raw_iq_path)

from iq_logic import (  # noqa: E402
    IQFrame,
    MMWavePacketParser,
    READ_SIZE,
    RadarUARTSource,
)


class RawIQFrameSource:
    """Yield complete complex range-I/Q frames from USB1."""

    def __init__(self) -> None:
        self._uart = RadarUARTSource()
        self._parser = MMWavePacketParser()
        self._is_open = False

    @property
    def malformed_packets(self) -> int:
        return self._parser.malformed_packets

    @property
    def discarded_bytes(self) -> int:
        return self._parser.discarded_bytes

    def __enter__(self) -> "RawIQFrameSource":
        # RadarUARTSource opens USB1 first, then configures/starts through USB0.
        self._uart.__enter__()
        self._is_open = True
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._is_open:
            self._uart.__exit__(exception_type, exception, traceback)
            self._is_open = False

    def read_frames(self) -> list[IQFrame]:
        if not self._is_open:
            raise RuntimeError("Raw I/Q frame source is not open")
        return self._parser.feed(self._uart.read(READ_SIZE))

    def discard_pending_data(self) -> None:
        """Drop USB1 bytes and partial packets collected during a user pause."""
        if not self._is_open:
            raise RuntimeError("Raw I/Q frame source is not open")
        data_port = self._uart._data
        if data_port is None:
            raise RuntimeError("Radar USB1 data port is not open")
        data_port.reset_input_buffer()
        self._parser = MMWavePacketParser()
