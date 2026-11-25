"""GUI-independent thermal camera, rendering, and recording logic."""

import ctypes
import math
import os
import queue
import shutil
import struct
import subprocess
import threading
import tempfile
import time
from pathlib import Path

try:
    # Package import used by the integrated CoBas application.
    from .celsius_heat_map import CelsiusHeatMap
except ImportError:
    # Standalone Hardware scripts place this directory directly on sys.path.
    from celsius_heat_map import CelsiusHeatMap


ROOT = Path(__file__).resolve().parent
LIB = ROOT / "mlx90642-library"
if not LIB.is_dir():
    LIB = ROOT.parents[2] / "Hardware" / "mlx90642-library"
# The path is configurable so an installer or developer can choose where the
# ctypes-compatible shared-library build is written.
SHARED_LIB = Path(os.environ.get("MLX90642_SHARED_LIB", "/tmp/libmlx90642.so"))

# The sensor is read either on the Pi's own I2C header or through an MCP2221A
# USB-to-I2C bridge. Each path has its own small C transport file; the rest of
# the vendor driver is shared. MLX90642_TRANSPORT picks one explicitly
# ("i2c" or "mcp2221"); the default, "auto", uses a bridge when one is plugged in.
TRANSPORT_I2C = "i2c"
TRANSPORT_MCP2221 = "mcp2221"
# An Arduino Nano ESP32 running CoBas/Hardware/arduino-mlx90642 reads the sensor
# itself and streams frames over USB serial; no C library is built for it.
TRANSPORT_SERIAL = "serial"
TRANSPORT_SOURCES = {
    TRANSPORT_I2C: "MLX90642_linux_i2c.c",
    TRANSPORT_MCP2221: "MLX90642_mcp2221_hidraw.c",
}
ARDUINO_VENDOR_ID = 0x2341
SERIAL_BAUD = 115200
# Packet layout shared with the sketch: magic, sequence, pixel count, flags,
# board millis, 768 little-endian words, CRC-16/CCITT-FALSE of all before it.
SERIAL_MAGIC = b"CBT\x01"
SERIAL_HEADER = struct.Struct("<4sHHHI")
SERIAL_PIXELS = struct.Struct("<768h")
SERIAL_PACKET_SIZE = SERIAL_HEADER.size + SERIAL_PIXELS.size + 2
SERIAL_FIRST_FRAME_TIMEOUT_SECONDS = 20.0
SERIAL_FRAME_TIMEOUT_SECONDS = 5.0
MCP2221_VENDOR_ID = 0x04D8
MCP2221_PRODUCT_ID = 0x00DD
# Repository file that grants users access to the bridge's HID node.
MCP2221_UDEV_RULES = "CoBas/Hardware/99-mcp2221a.rules"
HIDRAW_CLASS = Path("/sys/class/hidraw")
I2C_ADAPTER_CLASS = Path("/sys/class/i2c-adapter")
DEFAULT_I2C_BUS = "/dev/i2c-1"


class Transport:
    """How the driver reaches the sensor: the Pi's I2C header or an MCP2221A bridge."""

    def __init__(self, kind, device, via_bridge):
        # kind selects the C transport source; device is the node it opens.
        self.kind = kind
        self.device = device
        self.via_bridge = via_bridge

    @property
    def label(self):
        if self.kind == TRANSPORT_SERIAL:
            return "Arduino Nano ESP32"
        return "MCP2221A USB bridge" if self.via_bridge else "I2C header"

    @property
    def status_suffix(self):
        """Short tag for status lines on the small touch display."""
        if self.kind == TRANSPORT_SERIAL:
            return " (Arduino)"
        return " (MCP2221A)" if self.via_bridge else ""

    def describe(self):
        if self.device is None:
            return self.label
        return f"{self.label} at {self.device}"

    def explain_failure(self, error):
        """The error text shown to the operator, with the path the sensor was reached on."""
        message = f"{error} [{self.describe()}]"
        if self.via_bridge and "acknowledge" in str(error):
            message += (
                "; check the camera's SDA, SCL, 3.3 V and GND wires to the bridge"
            )
        return message


def find_mcp2221_hidraw():
    """Return the /dev/hidraw node of an attached MCP2221A, or None."""
    for uevent in sorted(HIDRAW_CLASS.glob("hidraw*/device/uevent")):
        try:
            text = uevent.read_text()
        except OSError:
            continue
        for line in text.splitlines():
            if not line.startswith("HID_ID="):
                continue
            # HID_ID=0003:000004D8:000000DD is bus:vendor:product in hex.
            parts = line[len("HID_ID="):].split(":")
            try:
                vendor, product = int(parts[1], 16), int(parts[2], 16)
            except (IndexError, ValueError):
                continue
            if (vendor, product) == (MCP2221_VENDOR_ID, MCP2221_PRODUCT_ID):
                return str(Path("/dev") / uevent.parents[1].name)
    return None


def find_mcp2221_i2c_bus():
    """Return /dev/i2c-N of a bridge the kernel's own hid_mcp2221 driver claimed, or None."""
    for name_file in sorted(I2C_ADAPTER_CLASS.glob("i2c-*/name")):
        try:
            name = name_file.read_text()
        except OSError:
            continue
        if "mcp2221" in name.lower():
            return str(Path("/dev") / name_file.parent.name)
    return None


def find_arduino_serial_port():
    """Return the serial port of an attached Arduino (the Nano ESP32 camera), or None."""
    try:
        from serial.tools import list_ports
    except ImportError:
        return None
    for port in sorted(list_ports.comports(), key=lambda entry: entry.device):
        if port.vid == ARDUINO_VENDOR_ID:
            return port.device
    return None


def select_transport():
    """Decide how to reach the sensor and hand the chosen device to the transport."""
    requested = os.environ.get("MLX90642_TRANSPORT", "auto").strip().lower() or "auto"
    choices = ("auto", TRANSPORT_I2C, TRANSPORT_MCP2221, TRANSPORT_SERIAL)
    if requested not in choices:
        raise DriverError(
            f"MLX90642_TRANSPORT must be one of {', '.join(choices)}, not {requested!r}"
        )

    if requested in ("auto", TRANSPORT_SERIAL):
        port = os.environ.get("MLX90642_SERIAL_PORT") or find_arduino_serial_port()
        if port:
            if not os.path.exists(port):
                raise DriverError(f"no Arduino at {port}")
            if not os.access(port, os.R_OK | os.W_OK):
                raise DriverError(
                    f"the Arduino at {port} is not accessible to this user; "
                    "add the user to the dialout group and log in again"
                )
            return Transport(TRANSPORT_SERIAL, port, via_bridge=False)
        if requested == TRANSPORT_SERIAL:
            raise DriverError("no Arduino is attached (nothing with USB vendor ID 2341)")

    if requested != TRANSPORT_I2C:
        hidraw = os.environ.get("MLX90642_MCP2221_HIDRAW") or find_mcp2221_hidraw()
        if hidraw:
            if not os.path.exists(hidraw):
                raise DriverError(f"no MCP2221A USB bridge at {hidraw}")
            if not os.access(hidraw, os.R_OK | os.W_OK):
                raise DriverError(
                    f"the MCP2221A bridge at {hidraw} is not accessible to this user; "
                    f"install {MCP2221_UDEV_RULES} (see the README), then unplug and "
                    "replug the bridge"
                )
            # The C transport reads this variable, so it opens the node checked here.
            os.environ["MLX90642_MCP2221_HIDRAW"] = hidraw
            return Transport(TRANSPORT_MCP2221, hidraw, via_bridge=True)

        bus = find_mcp2221_i2c_bus()
        if bus:
            # A kernel with hid_mcp2221 presents the bridge as an ordinary I2C
            # adapter; the plain I2C transport then works unchanged on it.
            os.environ.setdefault("MLX90642_I2C_BUS", bus)
            return Transport(TRANSPORT_I2C, os.environ["MLX90642_I2C_BUS"], via_bridge=True)

        if requested == TRANSPORT_MCP2221:
            raise DriverError(
                "no MCP2221A USB bridge is attached (nothing with USB ID 04d8:00dd)"
            )

    bus = os.environ.get("MLX90642_I2C_BUS") or DEFAULT_I2C_BUS
    return Transport(TRANSPORT_I2C, bus, via_bridge=False)


def build_transport(kind):
    """A transport for compiling only: no device has to be attached."""
    if kind not in TRANSPORT_SOURCES:
        raise DriverError(f"unknown transport {kind!r}")
    return Transport(kind, None, via_bridge=kind == TRANSPORT_MCP2221)


def transport_library_path(output, transport):
    """Each transport gets its own build, so switching never overwrites a loaded library."""
    output = Path(output)
    if transport.kind == TRANSPORT_I2C:
        return output
    return output.with_name(f"{output.stem}-{transport.kind}{output.suffix}")

# These dimensions are part of the sensor/driver ABI: every frame passed across
# the C boundary contains exactly this many 16-bit pixel values.
SENSOR_WIDTH = 32
SENSOR_HEIGHT = 24
SENSOR_PIXELS = SENSOR_WIDTH * SENSOR_HEIGHT
# Recording dimensions and cadence are configurable at process startup. Reading
# them here gives the worker, renderer, recorder, and GUI one consistent view.
RECORD_WIDTH = int(os.environ.get("MLX90642_RECORD_WIDTH", "640"))
RECORD_HEIGHT = int(os.environ.get("MLX90642_RECORD_HEIGHT", "480"))
RECORD_FPS = int(os.environ.get("MLX90642_RECORD_FPS", "8"))
# A single frame can fail during a sensor read-window transition even after the
# transport-level retries. Do not terminate acquisition on one transient miss.
FRAME_FAILURE_RETRY_LIMIT = 3
FRAME_FAILURE_RETRY_DELAY_SECONDS = 0.05
# The driver gives the sensor only 200 ms beyond its nominal frame period to
# deliver the first synchronized frame, and it sometimes needs longer. Starting
# again succeeds, so do that here instead of failing the whole capture.
INITIALIZE_ATTEMPTS = 3
INITIALIZE_RETRY_DELAY_SECONDS = 0.1

# Keep the public palette name for existing callers. It contains one unique RGB
# value for every 0.05 °C interval on the fixed 0–60 °C color scale.
PALETTE = list(CelsiusHeatMap.COLORS_BY_CELSIUS.values())


class DriverError(RuntimeError):
    # Distinguish driver build/I2C failures from GUI and recording failures.
    pass


def build_shared_library(output=SHARED_LIB, transport=None):
    # The vendor driver is plain C, so build its platform layer and the narrow
    # Python bridge into one shared object that ctypes can load directly.
    if transport is None:
        transport = select_transport()
    output = transport_library_path(output, transport)
    sources = [
        LIB / "MLX90642_python.c",
        LIB / "src" / "MLX90642.c",
        LIB / "src" / TRANSPORT_SOURCES[transport.kind],
    ]
    command = [
        "gcc",
        "-shared",
        "-fPIC",
        "-Wall",
        "-Wextra",
        "-O2",
        "-I",
        LIB / "inc",
        *sources,
        "-o",
        output,
    ]

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Replacing an inode is safe even while a camera has the previous driver loaded.
    # Compiling straight over a loaded .so can corrupt its mapped code pages.
    with tempfile.NamedTemporaryFile(prefix=output.stem + "_", suffix=".so", dir=output.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
    command[-1] = temporary_path
    try:
        result = subprocess.run([str(part) for part in command], cwd=ROOT,
                                text=True, capture_output=True, check=False)
        if result.returncode != 0:
            details = (result.stderr or result.stdout).strip()
            raise DriverError(f"driver build failed:\n{details}")
        os.replace(temporary_path, output)
    finally:
        temporary_path.unlink(missing_ok=True)
    return output


def signed_word(value):
    # ctypes receives the driver's bit pattern as uint16_t. Pixel temperatures
    # are signed two's-complement values, so restore the negative half here.
    return value - 65536 if value >= 32768 else value


def raw_to_celsius(value):
    # Temperature-format frames use a fixed scale of 50 counts per degree C.
    return value / 50.0


def frame_to_celsius(frame):
    """Convert one complete frame of raw camera values to degrees Celsius."""
    return [raw_to_celsius(value) for value in frame]


def frame_statistics(frame):
    """Return the minimum, center, and maximum raw values for one frame."""
    min_value = min(frame)
    max_value = max(frame)
    # The flat driver buffer is row-major, so convert the center row/column to
    # the corresponding one-dimensional index.
    center_index = (SENSOR_HEIGHT // 2) * SENSOR_WIDTH + (SENSOR_WIDTH // 2)
    return min_value, frame[center_index], max_value


class MLX90642Camera:
    def __init__(self, library_path):
        # CDLL keeps the C driver behind a small Python interface; frame memory
        # remains Python-owned and is passed into C only for the duration read.
        self._library = ctypes.CDLL(str(library_path))
        self._frame_type = ctypes.c_uint16 * SENSOR_PIXELS

        # Declare the C ABI explicitly. Without these signatures ctypes would
        # assume C int arguments/results, which is unsafe for pointers and the
        # uint16_t poll limit.
        self._library.MLX90642_PythonInit.argtypes = []
        self._library.MLX90642_PythonInit.restype = ctypes.c_int
        self._library.MLX90642_PythonReadFrame.argtypes = [
            ctypes.POINTER(ctypes.c_uint16)
        ]
        self._library.MLX90642_PythonReadFrame.restype = ctypes.c_int
        self._library.MLX90642_PythonWaitForNextFrame.argtypes = [
            ctypes.c_uint16
        ]
        self._library.MLX90642_PythonWaitForNextFrame.restype = ctypes.c_int
        # The MCP2221A transport can say why a call failed (no acknowledge from
        # the sensor, bridge unplugged, bus line levels); the header transport
        # has no such hook, so the detail is optional.
        try:
            self._library.MLX90642_TransportLastError.argtypes = []
            self._library.MLX90642_TransportLastError.restype = ctypes.c_char_p
            self._last_error = self._library.MLX90642_TransportLastError
        except AttributeError:
            self._last_error = None

    def _failure(self, call, status):
        message = f"{call} failed: {status}"
        detail = self._last_error() if self._last_error is not None else None
        if detail:
            message += ": " + detail.decode("utf-8", "replace")
        return DriverError(message)

    def initialize(self):
        # The C initializer synchronizes a measurement and waits until the
        # first frame is ready, so a successful return makes read_frame usable.
        status = self._library.MLX90642_PythonInit()
        if status < 0:
            raise self._failure("MLX90642_Init", status)

    def read_frame(self):
        # Keep the ctypes array alive for the entire C call; the driver fills it
        # in place with one complete, flat 32 x 24 frame in sensor order.
        frame = self._frame_type()
        status = self._library.MLX90642_PythonReadFrame(frame)
        if status < 0:
            raise self._failure("MLX90642_GetImage", status)

        return [signed_word(frame[index]) for index in range(SENSOR_PIXELS)]

    def wait_for_next_frame(self):
        # Waiting for the read window to close and reopen prevents the worker
        # from publishing the same sensor frame repeatedly.
        status = self._library.MLX90642_PythonWaitForNextFrame(1000)
        if status < 0:
            raise self._failure("MLX90642_IsReadWindowOpen", status)


class SerialCamera:
    """Frames from the Arduino Nano ESP32 over USB serial; same interface as MLX90642Camera."""

    def __init__(self, device):
        try:
            import serial
        except ImportError as error:
            raise DriverError("pyserial is not installed in the app's environment") from error
        # The baud rate is nominal: the board's USB serial ignores it.
        self._serial = serial.Serial(device, SERIAL_BAUD, timeout=0.2)
        self._buffer = bytearray()
        self._pending = None
        self.last_status_line = None
        self.sequence = None
        self.dropped = 0

    def close(self):
        try:
            self._serial.close()
        except Exception:
            pass

    def initialize(self):
        # The board initialises the sensor by itself and reports in '#' lines;
        # asking it to stream and waiting for the first frame covers both.
        self._buffer.clear()
        self._serial.reset_input_buffer()
        self._serial.write(b"S")
        self._pending = self._read_packet(SERIAL_FIRST_FRAME_TIMEOUT_SECONDS)

    def read_frame(self):
        if self._pending is not None:
            frame, self._pending = self._pending, None
            return frame
        return self._read_packet(SERIAL_FRAME_TIMEOUT_SECONDS)

    def wait_for_next_frame(self):
        # The board paces the stream at the sensor's rate; nothing to wait for.
        return None

    def _read_packet(self, timeout_seconds):
        deadline = time.monotonic() + timeout_seconds
        while True:
            packet = self._extract_packet()
            if packet is not None:
                return packet
            if time.monotonic() >= deadline:
                detail = f": board says {self.last_status_line!r}" if self.last_status_line else ""
                raise DriverError(
                    f"no frame from the Arduino within {timeout_seconds:g} s{detail}"
                )
            chunk = self._serial.read(4096)
            if chunk:
                self._buffer.extend(chunk)

    def _extract_packet(self):
        """Pop one valid frame from the buffer; keep '#' status lines; skip noise."""
        while True:
            start = self._buffer.find(SERIAL_MAGIC)
            if start < 0:
                self._note_status_lines(self._buffer)
                # Keep a partial magic at the tail.
                del self._buffer[:max(0, len(self._buffer) - len(SERIAL_MAGIC) + 1)]
                return None
            if start > 0:
                self._note_status_lines(self._buffer[:start])
                del self._buffer[:start]
            if len(self._buffer) < SERIAL_PACKET_SIZE:
                return None
            packet = bytes(self._buffer[:SERIAL_PACKET_SIZE])
            body, crc = packet[:-2], int.from_bytes(packet[-2:], "little")
            if crc16_ccitt(body) != crc:
                # Corrupt or misaligned: drop the magic and search again.
                self.dropped += 1
                del self._buffer[:len(SERIAL_MAGIC)]
                continue
            del self._buffer[:SERIAL_PACKET_SIZE]
            _, sequence, pixels, _flags, _millis = SERIAL_HEADER.unpack_from(body)
            if pixels != SENSOR_PIXELS:
                self.dropped += 1
                continue
            self.sequence = sequence
            return list(SERIAL_PIXELS.unpack_from(body, SERIAL_HEADER.size))

    def _note_status_lines(self, data):
        for line in bytes(data).split(b"\n"):
            if line.startswith(b"#"):
                self.last_status_line = line[1:].strip().decode("utf-8", "replace")


def crc16_ccitt(data):
    """CRC-16/CCITT-FALSE, as computed by the sketch."""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


class MockCamera:
    def __init__(self):
        self._tick = 0

    def initialize(self):
        return None

    def read_frame(self):
        self._tick += 1
        frame = []
        # Move a Gaussian hot spot along two different periods so mock frames
        # evolve continuously instead of repeating a short obvious loop.
        hot_x = 16 + math.sin(self._tick / 9.0) * 8
        hot_y = 12 + math.cos(self._tick / 13.0) * 6

        for y in range(SENSOR_HEIGHT):
            for x in range(SENSOR_WIDTH):
                base = 1180 + y * 7 + x * 2
                dx = x - hot_x
                dy = y - hot_y
                # Combine a localized heat source with a horizontal wave over a
                # gentle baseline gradient to exercise the full color mapping.
                hot = 620 * math.exp(-(dx * dx + dy * dy) / 36.0)
                wave = 55 * math.sin((x + self._tick) / 4.0)
                frame.append(int(base + hot + wave))

        return frame

    def wait_for_next_frame(self):
        # Match the recorder cadence closely enough for realistic UI testing.
        time.sleep(1.0 / RECORD_FPS)


class CameraWorker(threading.Thread):
    def __init__(self, events, stop_event, mock=False):
        super().__init__(daemon=True)
        self.events = events
        self.stop_event = stop_event
        self.mock = mock
        # Set once the sensor path is known; consumers show it in status lines.
        self.transport = None

    def run(self):
        transport = None
        try:
            # Driver compilation and blocking I2C calls stay off the caller's
            # event thread; results cross back through the thread-safe queue.
            if self.mock:
                camera = MockCamera()
            else:
                transport = select_transport()
                self.transport = transport
                if transport.kind == TRANSPORT_SERIAL:
                    self.events.put(("status", f"Opening {transport.device} ({transport.label})"))
                    camera = SerialCamera(transport.device)
                else:
                    self.events.put(("status", f"Building driver ({transport.label})"))
                    library_path = build_shared_library(transport=transport)
                    camera = MLX90642Camera(library_path)

            self.events.put(("status", "Initializing camera" + (f" ({transport.label})" if transport else "")))
            if not self._initialize(camera):
                return
            self.events.put(("status", "Live"))

            # Event tuples are intentionally data-only: consumers can render,
            # record, or test acquisition without this module knowing a GUI.
            consecutive_failures = 0
            while not self.stop_event.is_set():
                try:
                    # Read the current frame, then require a closed-to-open
                    # window transition before fetching the following frame.
                    frame = camera.read_frame()
                    self.events.put(("frame", frame, time.monotonic()))
                    camera.wait_for_next_frame()
                    consecutive_failures = 0
                except DriverError as exc:
                    consecutive_failures += 1
                    if consecutive_failures >= FRAME_FAILURE_RETRY_LIMIT:
                        raise DriverError(
                            "thermal frame acquisition failed "
                            f"{consecutive_failures} consecutive times: {exc}"
                        ) from exc

                    self.events.put(
                        (
                            "status",
                            "Recovering thermal camera "
                            f"({consecutive_failures}/"
                            f"{FRAME_FAILURE_RETRY_LIMIT})",
                        )
                    )
                    if self.stop_event.wait(
                        FRAME_FAILURE_RETRY_DELAY_SECONDS
                    ):
                        break
        except Exception as exc:
            # Marshal failures to the consumer instead of letting a daemon
            # thread terminate silently where the user cannot see the cause.
            message = transport.explain_failure(exc) if transport else str(exc)
            self.events.put(("error", message))

    def _initialize(self, camera):
        """Start the sensor, retrying a transient failure. False means a stop was requested."""
        for attempt in range(1, INITIALIZE_ATTEMPTS + 1):
            try:
                camera.initialize()
                return True
            except DriverError:
                if attempt == INITIALIZE_ATTEMPTS:
                    raise
                self.events.put(
                    ("status", f"Retrying thermal camera start ({attempt}/{INITIALIZE_ATTEMPTS - 1})")
                )
                if self.stop_event.wait(INITIALIZE_RETRY_DELAY_SECONDS):
                    return False
        return True


class ThermalRenderer:
    COLOR_SCALE_MIN_CELSIUS = CelsiusHeatMap.MIN_CELSIUS
    COLOR_SCALE_MAX_CELSIUS = CelsiusHeatMap.MAX_CELSIUS
    COLOR_RESOLUTION_CELSIUS = CelsiusHeatMap.COLOR_RESOLUTION_CELSIUS
    SENSOR_NETD_CELSIUS = CelsiusHeatMap.SENSOR_NETD_CELSIUS
    DEFAULT_MIN_CELSIUS = CelsiusHeatMap.DEFAULT_DISPLAY_MIN_CELSIUS
    DEFAULT_MAX_CELSIUS = CelsiusHeatMap.DEFAULT_DISPLAY_MAX_CELSIUS

    def __init__(self, min_celsius=None, max_celsius=None):
        # The heat map owns the complete ordered spectrum. A renderer assigns
        # its selected minimum and maximum to the spectrum endpoints and clamps
        # pixels outside that temperature interval to those endpoint colors.
        self.heat_map = CelsiusHeatMap()
        if min_celsius is None:
            min_celsius = self.DEFAULT_MIN_CELSIUS
        if max_celsius is None:
            max_celsius = self.DEFAULT_MAX_CELSIUS

        try:
            min_celsius = float(min_celsius)
            max_celsius = float(max_celsius)
        except (TypeError, ValueError) as exc:
            raise TypeError("temperature range values must be real numbers") from exc

        if not math.isfinite(min_celsius) or not math.isfinite(max_celsius):
            raise ValueError("temperature range values must be finite")
        if max_celsius <= min_celsius:
            raise ValueError("maximum temperature must be greater than minimum")
        if (
            min_celsius < self.COLOR_SCALE_MIN_CELSIUS
            or max_celsius > self.COLOR_SCALE_MAX_CELSIUS
        ):
            raise ValueError(
                "temperature range must stay within the fixed "
                f"{self.COLOR_SCALE_MIN_CELSIUS:g}–"
                f"{self.COLOR_SCALE_MAX_CELSIUS:g} °C color scale"
            )

        self.min_celsius = min_celsius
        self.max_celsius = max_celsius
        # Keep the coldest absolute measurement from the most recently
        # generated color map for diagnostics and legend consumers.
        self.last_frame_min_celsius = None
        # Coordinate maps depend only on output dimensions, so cache them across
        # frames and avoid repeating bilinear setup at recording frame rate.
        self._maps = {}

    def scale_color(self, normalized):
        """Return the RGB color at one normalized point on this renderer's scale."""
        normalized = max(0.0, min(1.0, float(normalized)))
        temperature = self.min_celsius + normalized * (
            self.max_celsius - self.min_celsius
        )
        # Passing the fixed Celsius endpoints also lets renderer variants reuse
        # this method while supplying their own _color implementation.
        return self._color(
            temperature,
            self.min_celsius,
            self.max_celsius,
            {},
        )

    def legend_celsius_range(self, frame=None):
        """Return the temperatures assigned to the full spectrum endpoints."""
        return self.min_celsius, self.max_celsius

    def legend_extrema_celsius(self, frame=None):
        """Return the current frame's minimum and maximum Celsius values."""
        if frame is None:
            return None

        temperatures = [
            temperature
            for temperature in frame_to_celsius(frame)
            if math.isfinite(temperature)
        ]
        if not temperatures:
            return None

        return min(temperatures), max(temperatures)

    def _get_map(self, width, height):
        key = (width, height)
        if key in self._maps:
            return self._maps[key]

        if width <= 1:
            # A single output column maps to the first sensor column and avoids
            # division by zero in the general interpolation formula.
            x_map = [(0, 0, 0.0)]
        else:
            x_map = []
            for x in range(width):
                # Map each destination coordinate into the 32-column source and
                # retain both neighbors plus the fractional interpolation weight.
                source_x = x * (SENSOR_WIDTH - 1) / (width - 1)
                x0 = int(source_x)
                x1 = min(x0 + 1, SENSOR_WIDTH - 1)
                x_map.append((x0, x1, source_x - x0))

        if height <= 1:
            # The equivalent one-row special case avoids the height denominator.
            y_map = [(0, 0, 0.0)]
        else:
            y_map = []
            for y in range(height):
                source_y = y * (SENSOR_HEIGHT - 1) / (height - 1)
                y0 = int(source_y)
                y1 = min(y0 + 1, SENSOR_HEIGHT - 1)
                y_map.append((y0, y1, source_y - y0))

        self._maps[key] = (x_map, y_map)
        return x_map, y_map

    def _color(self, value, min_value, max_value, color_cache):
        # Clamp to the selected window, then normalize that complete window
        # across the unchanged 0–60 palette. For example, a 15–30 °C selection
        # maps 15 °C to palette position 0 and 30 °C to palette position 60.
        temperature = max(self.min_celsius, min(self.max_celsius, value))
        cached = color_cache.get(temperature)
        if cached is not None:
            return cached

        normalized = (
            (temperature - self.min_celsius)
            / (self.max_celsius - self.min_celsius)
        )
        palette_temperature = self.COLOR_SCALE_MIN_CELSIUS + normalized * (
            self.COLOR_SCALE_MAX_CELSIUS - self.COLOR_SCALE_MIN_CELSIUS
        )
        color = self.heat_map.rgb_for_celsius(palette_temperature)
        color_cache[temperature] = color
        return color

    def _validated_frame_celsius(self, frame):
        """Convert and validate one complete frame in absolute Celsius."""
        temperatures = frame_to_celsius(frame)
        if not temperatures:
            raise ValueError("thermal frame must contain at least one pixel")
        if not all(math.isfinite(temperature) for temperature in temperatures):
            raise ValueError("thermal frame contains a non-finite temperature")

        self.last_frame_min_celsius = min(temperatures)
        return temperatures

    def frame_colors(self, frame):
        # Assign colors from absolute Celsius values so image pixels and the
        # fixed Celsius legend describe the same physical temperature scale.
        temperatures = self._validated_frame_celsius(frame)
        min_value = min(temperatures)
        max_value = max(temperatures)
        color_cache = {}
        colors = []

        for temperature in temperatures:
            red, green, blue = self._color(
                temperature,
                min_value,
                max_value,
                color_cache,
            )
            colors.append(f"#{red:02x}{green:02x}{blue:02x}")

        return colors

    def render_rgb(self, frame, width, height):
        # Build the 32x24 thermal image from absolute Celsius values, then
        # enlarge the completed RGB pixels for video output.
        width = max(1, int(width))
        height = max(1, int(height))
        temperatures = self._validated_frame_celsius(frame)
        if len(temperatures) != SENSOR_PIXELS:
            raise ValueError(
                f"thermal frame must contain exactly {SENSOR_PIXELS} pixels; "
                f"received {len(temperatures)}"
            )

        min_value = min(temperatures)
        max_value = max(temperatures)
        color_cache = {}
        source_colors = [
            self._color(temperature, min_value, max_value, color_cache)
            for temperature in temperatures
        ]
        x_map, y_map = self._get_map(width, height)
        output = bytearray(width * height * 3)
        offset = 0

        for y0, y1, y_weight in y_map:
            row0 = y0 * SENSOR_WIDTH
            row1 = y1 * SENSOR_WIDTH

            for x0, x1, x_weight in x_map:
                # Bilinearly expand the already colorized source pixels. This
                # preserves the selected range's spectrum association before
                # the low-resolution sensor image becomes a dense RGB image.
                top_left = source_colors[row0 + x0]
                top_right = source_colors[row0 + x1]
                bottom_left = source_colors[row1 + x0]
                bottom_right = source_colors[row1 + x1]

                for channel in range(3):
                    top = top_left[channel] + (
                        top_right[channel] - top_left[channel]
                    ) * x_weight
                    bottom = bottom_left[channel] + (
                        bottom_right[channel] - bottom_left[channel]
                    ) * x_weight
                    output[offset + channel] = round(
                        top + (bottom - top) * y_weight
                    )
                offset += 3

        return bytes(output)


class FfmpegRecorder:
    def __init__(self, path, width, height, fps):
        # Resolve once so the child process receives an explicit executable and
        # failure is reported before any writer thread is started.
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg is not installed")

        self.path = Path(path)
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_bytes = width * height * 3
        self.frames_written = 0
        self.frames_dropped = 0
        self.error = None
        # Buffer roughly four seconds of frames. Producers never block when the
        # encoder falls behind; write_frame records a drop instead.
        self._queue = queue.Queue(maxsize=fps * 4)
        self._closed = False
        self._process = subprocess.Popen(
            # Feed headerless RGB24 frames on stdin and let ffmpeg encode a
            # broadly compatible H.264/yuv420p MP4 at the configured cadence.
            [
                ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgb24",
                "-video_size",
                f"{width}x{height}",
                "-framerate",
                str(fps),
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-pix_fmt",
                "yuv420p",
                str(self.path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        self._thread = threading.Thread(target=self._write_loop, daemon=True)
        self._thread.start()

    def _write_loop(self):
        assert self._process.stdin is not None

        while True:
            item = self._queue.get()
            # None is an in-band sentinel placed after the last accepted frame.
            if item is None:
                break

            try:
                self._process.stdin.write(item)
                self.frames_written += 1
            except Exception as exc:
                # Save the exception for close(), which runs in caller context
                # and can present the failure instead of losing it in the thread.
                self.error = exc
                break

        try:
            self._process.stdin.close()
        except Exception:
            pass

    def write_frame(self, rgb):
        if self._closed:
            return

        if len(rgb) != self.frame_bytes:
            self.frames_dropped += 1
            self.error = ValueError(
                "RGB frame size mismatch: "
                f"expected {self.frame_bytes} bytes for "
                f"{self.width}x{self.height}, received {len(rgb)}"
            )
            return

        try:
            # UI responsiveness takes priority over preserving every frame.
            self._queue.put_nowait(rgb)
        except queue.Full:
            self.frames_dropped += 1

    def close(self):
        if self._closed:
            return

        self._closed = True
        # Guarantee room for the sentinel. If the queue is saturated, discard
        # the oldest pending frames rather than blocking shutdown indefinitely.
        while True:
            try:
                self._queue.put(None, timeout=0.1)
                break
            except queue.Full:
                self.frames_dropped += 1
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass

        self._thread.join(timeout=10)
        # Closing stdin in the writer tells ffmpeg to flush and finalize the MP4.
        return_code = self._process.wait(timeout=10)
        if return_code != 0:
            stderr = b""
            if self._process.stderr is not None:
                stderr = self._process.stderr.read()
            message = stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(message or f"ffmpeg failed with exit code {return_code}")

        if self.error is not None:
            raise RuntimeError(str(self.error))
