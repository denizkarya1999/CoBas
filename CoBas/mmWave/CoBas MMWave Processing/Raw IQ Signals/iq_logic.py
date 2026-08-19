"""IWR6843AOP USB1 packet parsing and complex range-bin I/Q logging logic."""

from __future__ import annotations

import csv
import struct
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType


# Fixed hardware configuration. USB0 is the text command interface and USB1 is
# the high-speed binary data interface exposed by the board's CP2105 bridge.
# The application intentionally has no CLI arguments or external config file.
CLI_PORT = "/dev/ttyUSB0"
DATA_PORT = "/dev/ttyUSB1"
CLI_BAUD_RATE = 115_200
DATA_BAUD_RATE = 921_600
SERIAL_TIMEOUT_SECONDS = 0.25
CLI_REPLY_TIMEOUT_SECONDS = 2.0
FIRST_IQ_FRAME_TIMEOUT_SECONDS = 8.0
READ_SIZE = 65_536
CSV_FLUSH_EVERY = 256
NUM_RANGE_BINS = 64

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
LOG_DIRECTORY = SCRIPT_DIRECTORY / "Logs"

# Every SDK output packet begins with this eight-byte synchronization marker.
# The remaining header fields are: version, total packet length, platform,
# frame number, CPU cycles, detected-object count, TLV count, and subframe.
MAGIC_WORD = b"\x02\x01\x04\x03\x06\x05\x08\x07"
PACKET_HEADER = struct.Struct("<8s8I")

# A TLV header contains an unsigned type and payload length. The length does not
# include this eight-byte TLV header or the padding at the end of a packet.
TLV_HEADER = struct.Struct("<2I")
COMPLEX_INT16 = struct.Struct("<hh")  # TI cmplx16ImRe_t: imaginary, then real.
MAX_PACKET_LENGTH = 1_000_000

# Non-AOP demos normally use type 4. IWR6843AOP uses the 2D-AoA version, type 8,
# which contains all active azimuth/elevation virtual antenna symbols.
AZIMUTH_STATIC_HEAT_MAP_TLV = 4
AZIMUTH_ELEVATION_STATIC_HEAT_MAP_TLV = 8
COMPLEX_IQ_TLV_TYPES = {
    AZIMUTH_STATIC_HEAT_MAP_TLV,
    AZIMUTH_ELEVATION_STATIC_HEAT_MAP_TLV,
}


# Complete SDK 3.x out-of-box profile. guiMonitor enables the zero-Doppler
# complex range-FFT matrix. On IWR6843AOP the firmware emits TLV type 8.
RADAR_CONFIGURATION = (
    # Reset any previous sensor state before applying a complete profile.
    "sensorStop",
    "flushCfg",
    "dfeDataOutputMode 1",

    # Enable four receivers, three transmitters, and complex 16-bit ADC data.
    "channelCfg 15 7 0",
    "adcCfg 2 1",
    "adcbufCfg -1 0 1 1 1",

    # Use 64 ADC/range samples and one chirp for each enabled transmitter.
    "profileCfg 0 60 7 7 57.14 0 0 70 1 64 2000 0 0 158",
    "chirpCfg 0 0 0 0 0 0 0 1",
    "chirpCfg 1 1 0 0 0 0 0 2",
    "chirpCfg 2 2 0 0 0 0 0 4",

    # Repeat the three-chirp sequence 64 times every 120 milliseconds.
    "frameCfg 0 2 64 0 120 1 0",
    "lowPower 0 0",

    # Enable only the complex zero-Doppler antenna matrix on the data UART.
    "guiMonitor -1 0 0 0 1 0 0",

    # Required processing-chain settings for the SDK 3.x out-of-box firmware.
    "cfarCfg -1 0 2 8 4 3 0 15 1",
    "cfarCfg -1 1 0 4 2 3 1 15 1",
    "multiObjBeamForming -1 1 0.5",
    "clutterRemoval -1 0",
    "calibDcRangeSig -1 0 -5 8 256",
    "extendedMaxVelocity -1 0",
    "lvdsStreamCfg -1 0 0 0",
    (
        "compRangeBiasAndRxChanPhase 0.0 "
        "1 0 -1 0 1 0 -1 0 1 0 -1 0 1 0 -1 0 "
        "1 0 -1 0 1 0 -1 0"
    ),
    "measureRangeBiasAndRxChanPhase 0 1.5 0.2",
    "CQRxSatMonitor 0 3 5 63 0",
    "CQSigImgMonitor 0 63 4",
    "analogMonitor 0 0",
    "bpmCfg -1 0 0 1",
    "aoaFovCfg -1 -90 90 -90 90",
    "cfarFovCfg -1 0 0.30 0.60",
    "cfarFovCfg -1 1 -10 10",
    "calibData 0 0 0",

    # Starting last ensures USB1 is already open before binary output begins.
    "sensorStart",
)


def log_path_for_name(log_name: str) -> Path:
    """Validate a user-facing log name and resolve it inside Logs."""
    name = log_name.strip()
    if not name:
        raise ValueError("Log name cannot be empty")
    if name in {".", ".."} or "/" in name or "\\" in name:
        raise ValueError("Log name must be a filename, not a path")
    if any(ord(character) < 32 for character in name):
        raise ValueError("Log name cannot contain control characters")
    if Path(name).suffix.lower() != ".csv":
        name += ".csv"
    return LOG_DIRECTORY / name


@dataclass(frozen=True, slots=True)
class IQSample:
    """Minimum values needed to place one complex symbol in a heatmap matrix."""

    frame_number: int
    range_bin: int
    virtual_antenna: int
    i: int
    q: int


@dataclass(frozen=True, slots=True)
class IQFrame:
    """All complex antenna symbols decoded from one radar output frame."""

    frame_number: int
    virtual_antenna_count: int
    samples: tuple[IQSample, ...]


@dataclass(frozen=True, slots=True)
class CaptureInfo:
    """Startup details supplied to the terminal interface."""

    cli_port: str
    data_port: str
    log_path: Path


@dataclass(frozen=True, slots=True)
class CaptureResult:
    """Capture totals and parser health counters reported at shutdown."""

    frame_count: int
    sample_count: int
    log_path: Path
    interrupted: bool
    malformed_packets: int
    discarded_bytes: int


class MMWavePacketParser:
    """Incrementally synchronize and decode SDK 3.x USB1 TLV packets."""

    def __init__(self) -> None:
        self._buffer = bytearray()
        self.total_packets = 0
        self.targetless_packets = 0
        self.malformed_packets = 0
        self.discarded_bytes = 0

    def feed(self, data: bytes) -> list[IQFrame]:
        """Accept any-sized serial chunk and return every complete I/Q frame."""
        self._buffer.extend(data)
        frames: list[IQFrame] = []

        while True:
            # Serial reads may begin midway through a packet. Search for the SDK
            # magic word and retain seven trailing bytes in case it is split
            # between this read and the next one.
            magic_offset = self._buffer.find(MAGIC_WORD)
            if magic_offset < 0:
                keep = min(len(self._buffer), len(MAGIC_WORD) - 1)
                discard = len(self._buffer) - keep
                if discard:
                    del self._buffer[:discard]
                    self.discarded_bytes += discard
                return frames

            if magic_offset:
                del self._buffer[:magic_offset]
                self.discarded_bytes += magic_offset

            if len(self._buffer) < PACKET_HEADER.size:
                return frames

            header = PACKET_HEADER.unpack_from(self._buffer)
            total_length = header[2]

            # A corrupt length could otherwise make the parser wait forever or
            # allocate unreasonable memory. Drop one byte and resynchronize.
            if not PACKET_HEADER.size <= total_length <= MAX_PACKET_LENGTH:
                del self._buffer[0]
                self.malformed_packets += 1
                self.discarded_bytes += 1
                continue
            if len(self._buffer) < total_length:
                # Keep a partial packet buffered until more USB1 bytes arrive.
                return frames

            # total_length includes the header, TLVs, and final 32-byte padding.
            packet = bytes(self._buffer[:total_length])
            del self._buffer[:total_length]
            self.total_packets += 1
            frame = self._parse_packet(packet, header)
            if frame is None:
                self.targetless_packets += 1
            else:
                frames.append(frame)

    def _parse_packet(self, packet: bytes, header: tuple) -> IQFrame | None:
        """Walk a complete packet's TLVs and decode its complex matrix."""
        frame_number = header[4]
        tlv_count = header[7]
        offset = PACKET_HEADER.size

        for _ in range(tlv_count):
            if offset + TLV_HEADER.size > len(packet):
                self.malformed_packets += 1
                return None
            tlv_type, payload_length = TLV_HEADER.unpack_from(packet, offset)
            payload_start = offset + TLV_HEADER.size
            payload_end = payload_start + payload_length

            # Validate the advertised payload boundary before slicing it.
            if payload_end > len(packet):
                self.malformed_packets += 1
                return None

            if tlv_type in COMPLEX_IQ_TLV_TYPES:
                return self._decode_complex_iq(
                    packet[payload_start:payload_end], frame_number
                )
            offset = payload_end
        return None

    def _decode_complex_iq(self, payload: bytes, frame_number: int) -> IQFrame | None:
        """Decode TI's range-major [range][virtual antenna] complex matrix."""

        # One virtual antenna contributes one complex int16 value at every
        # range bin. Divisibility lets us derive the active antenna count from
        # the payload instead of assuming a particular firmware geometry.
        bytes_per_antenna = NUM_RANGE_BINS * COMPLEX_INT16.size
        if not payload or len(payload) % bytes_per_antenna:
            self.malformed_packets += 1
            return None

        antenna_count = len(payload) // bytes_per_antenna
        if not 1 <= antenna_count <= 32:
            self.malformed_packets += 1
            return None

        samples: list[IQSample] = []
        offset = 0
        for range_bin in range(NUM_RANGE_BINS):
            for virtual_antenna in range(antenna_count):
                # cmplx16ImRe_t puts imaginary first. Expose conventional I/Q
                # naming by assigning real to I and imaginary to Q.
                q_value, i_value = COMPLEX_INT16.unpack_from(payload, offset)
                offset += COMPLEX_INT16.size
                samples.append(
                    IQSample(
                        frame_number=frame_number,
                        range_bin=range_bin,
                        virtual_antenna=virtual_antenna,
                        i=i_value,
                        q=q_value,
                    )
                )

        return IQFrame(frame_number, antenna_count, tuple(samples))


class RadarUARTSource:
    """Configure on USB0 and read SDK binary packets from USB1."""

    def __init__(self) -> None:
        self._cli = None
        self._data = None

    def __enter__(self) -> "RadarUARTSource":
        """Open both UARTs and apply every embedded command in order."""
        try:
            import serial
        except ImportError as error:
            raise RuntimeError(
                "PySerial is required. Install the folder's requirements.txt."
            ) from error

        try:
            # USB1 is opened before sensorStart so the first binary frame is kept.
            self._data = serial.Serial(
                DATA_PORT,
                DATA_BAUD_RATE,
                timeout=SERIAL_TIMEOUT_SECONDS,
            )
            self._data.reset_input_buffer()
            self._cli = serial.Serial(
                CLI_PORT,
                CLI_BAUD_RATE,
                timeout=0.1,
                write_timeout=1.0,
            )
            self._cli.reset_input_buffer()
            time.sleep(0.1)
            for command in RADAR_CONFIGURATION:
                # sensorStop may legitimately report that an idle sensor is
                # already stopped; all other command errors are fatal.
                self._send_command(command, allow_error=(command == "sensorStop"))
        except BaseException:
            self.close(stop_sensor=False)
            raise
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close(stop_sensor=True)

    def _send_command(self, command: str, allow_error: bool = False) -> str:
        """Send one USB0 command and collect its reply through the next prompt."""
        if self._cli is None:
            raise RuntimeError("Radar CLI port is not open")
        self._cli.reset_input_buffer()
        self._cli.write((command + "\n").encode("ascii"))
        self._cli.flush()

        deadline = time.monotonic() + CLI_REPLY_TIMEOUT_SECONDS
        response = bytearray()
        while time.monotonic() < deadline:
            chunk = self._cli.read(self._cli.in_waiting or 1)
            if chunk:
                response.extend(chunk)
                # The prompt marks the end of an SDK CLI command response.
                if b"mmwDemo:/>" in response:
                    break

        text = response.decode("utf-8", errors="replace").strip()
        if "error" in text.lower() and not allow_error:
            raise RuntimeError(f"Radar rejected '{command}': {text}")
        return text

    def read(self, size: int) -> bytes:
        """Read currently available USB1 bytes without exceeding size."""
        if self._data is None:
            raise RuntimeError("Radar USB1 data port is not open")
        return self._data.read(min(size, self._data.in_waiting or 1))

    def close(self, stop_sensor: bool) -> None:
        """Stop RF operation when possible, then release both serial ports."""
        if stop_sensor and self._cli is not None and self._cli.is_open:
            try:
                self._send_command("sensorStop", allow_error=True)
            except BaseException:
                pass
        if self._cli is not None and self._cli.is_open:
            self._cli.close()
        if self._data is not None and self._data.is_open:
            self._data.close()


class CSVSampleLogger:
    """Write only the values required to construct a range-azimuth heatmap."""

    HEADER = (
        "frame_number",
        "range_bin",
        "virtual_antenna",
        "i",
        "q",
    )

    def __init__(self, path: Path) -> None:
        self.path = path
        self._stream = None
        self._writer = None
        self._unflushed_rows = 0

    def __enter__(self) -> "CSVSampleLogger":
        # Create Logs on first use while also supporting other explicit paths.
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation prevents an explicitly chosen log from being
        # silently overwritten. Timestamped default names are normally unique.
        self._stream = self.path.open("x", encoding="utf-8", newline="")
        self._writer = csv.writer(self._stream)
        self._writer.writerow(self.HEADER)
        self._stream.flush()
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._stream is not None:
            self._stream.close()

    def write(self, sample: IQSample) -> None:
        if self._writer is None or self._stream is None:
            raise RuntimeError("CSV logger is not open")
        self._writer.writerow(
            (
                sample.frame_number,
                sample.range_bin,
                sample.virtual_antenna,
                sample.i,
                sample.q,
            )
        )
        self._unflushed_rows += 1
        if self._unflushed_rows >= CSV_FLUSH_EVERY:
            # Periodic flushing limits data loss if power is removed mid-run.
            self._stream.flush()
            self._unflushed_rows = 0


def capture(
    log_name: str,
    on_started: Callable[[CaptureInfo], None] | None = None,
    on_sample: Callable[[IQSample], None] | None = None,
) -> CaptureResult:
    """Start the radar, decode USB1 complex I/Q frames, and log every sample."""
    log_path = log_path_for_name(log_name)
    info = CaptureInfo(CLI_PORT, DATA_PORT, log_path)

    # UI output is injected as callbacks so this module remains independent of
    # terminal formatting or any future graphical interface.
    if on_started is not None:
        on_started(info)

    parser = MMWavePacketParser()
    frame_count = 0
    sample_count = 0
    interrupted = False

    try:
        with RadarUARTSource() as source, CSVSampleLogger(log_path) as logger:
            first_frame_deadline = time.monotonic() + FIRST_IQ_FRAME_TIMEOUT_SECONDS
            while True:
                chunk = source.read(READ_SIZE)
                for frame in parser.feed(chunk):
                    frame_count += 1
                    for sample in frame.samples:
                        logger.write(sample)
                        if on_sample is not None:
                            on_sample(sample)
                        sample_count += 1

                # Fail with a useful distinction between an inactive USB1 port
                # and active packets produced by incompatible firmware/config.
                if frame_count == 0 and time.monotonic() >= first_frame_deadline:
                    if parser.total_packets:
                        raise RuntimeError(
                            "USB1 packets arrived, but no complex range-I/Q TLV was "
                            "found. Confirm the board is running the SDK 3.x AOP "
                            "out-of-box firmware."
                        )
                    raise RuntimeError(
                        "No USB1 binary frames arrived after sensorStart. Check the "
                        "firmware, USB cable, port assignments, and board power."
                    )
    except KeyboardInterrupt:
        interrupted = True

    return CaptureResult(
        frame_count=frame_count,
        sample_count=sample_count,
        log_path=log_path,
        interrupted=interrupted,
        malformed_packets=parser.malformed_packets,
        discarded_bytes=parser.discarded_bytes,
    )
