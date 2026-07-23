"""GUI-independent thermal camera, rendering, and recording logic."""

import ctypes
import math
import os
import queue
import shutil
import subprocess
import threading
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
# The path is configurable so an installer or developer can choose where the
# ctypes-compatible shared-library build is written.
SHARED_LIB = Path(os.environ.get("MLX90642_SHARED_LIB", "/tmp/libmlx90642.so"))

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

# Keep the public palette name for existing callers. It contains one unique RGB
# value for every 0.05 °C interval on the fixed 0–60 °C color scale.
PALETTE = list(CelsiusHeatMap.COLORS_BY_CELSIUS.values())


class DriverError(RuntimeError):
    # Distinguish driver build/I2C failures from GUI and recording failures.
    pass


def build_shared_library(output=SHARED_LIB):
    # The vendor driver is plain C, so build its platform layer and the narrow
    # Python bridge into one shared object that ctypes can load directly.
    sources = [
        LIB / "MLX90642_python.c",
        LIB / "src" / "MLX90642.c",
        LIB / "src" / "MLX90642_linux_i2c.c",
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

    result = subprocess.run(
        [str(part) for part in command],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        # gcc normally reports useful diagnostics on stderr, but keep stdout as
        # a fallback for toolchains that route diagnostics differently.
        details = (result.stderr or result.stdout).strip()
        raise DriverError(f"driver build failed:\n{details}")

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

    def initialize(self):
        # The C initializer synchronizes a measurement and waits until the
        # first frame is ready, so a successful return makes read_frame usable.
        status = self._library.MLX90642_PythonInit()
        if status < 0:
            raise DriverError(f"MLX90642_Init failed: {status}")

    def read_frame(self):
        # Keep the ctypes array alive for the entire C call; the driver fills it
        # in place with one complete, flat 32 x 24 frame in sensor order.
        frame = self._frame_type()
        status = self._library.MLX90642_PythonReadFrame(frame)
        if status < 0:
            raise DriverError(f"MLX90642_GetImage failed: {status}")

        return [signed_word(frame[index]) for index in range(SENSOR_PIXELS)]

    def wait_for_next_frame(self):
        # Waiting for the read window to close and reopen prevents the worker
        # from publishing the same sensor frame repeatedly.
        status = self._library.MLX90642_PythonWaitForNextFrame(1000)
        if status < 0:
            raise DriverError(f"MLX90642_IsReadWindowOpen failed: {status}")


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

    def run(self):
        try:
            # Driver compilation and blocking I2C calls stay off the caller's
            # event thread; results cross back through the thread-safe queue.
            if self.mock:
                camera = MockCamera()
            else:
                library_path = build_shared_library()
                camera = MLX90642Camera(library_path)

            self.events.put(("status", "Initializing camera"))
            camera.initialize()
            self.events.put(("status", "Live"))

            # Event tuples are intentionally data-only: consumers can render,
            # record, or test acquisition without this module knowing a GUI.
            while not self.stop_event.is_set():
                # Read the current frame, then require a closed-to-open window
                # transition before fetching the following frame.
                frame = camera.read_frame()
                self.events.put(("frame", frame, time.monotonic()))
                camera.wait_for_next_frame()
        except Exception as exc:
            # Marshal failures to the consumer instead of letting a daemon
            # thread terminate silently where the user cannot see the cause.
            self.events.put(("error", str(exc)))


class ThermalRenderer:
    COLOR_SCALE_MIN_CELSIUS = CelsiusHeatMap.MIN_CELSIUS
    COLOR_SCALE_MAX_CELSIUS = CelsiusHeatMap.MAX_CELSIUS
    COLOR_RESOLUTION_CELSIUS = CelsiusHeatMap.COLOR_RESOLUTION_CELSIUS
    SENSOR_NETD_CELSIUS = CelsiusHeatMap.SENSOR_NETD_CELSIUS
    DEFAULT_MIN_CELSIUS = CelsiusHeatMap.DEFAULT_DISPLAY_MIN_CELSIUS
    DEFAULT_MAX_CELSIUS = CelsiusHeatMap.DEFAULT_DISPLAY_MAX_CELSIUS

    def __init__(self, min_celsius=None, max_celsius=None):
        # The heat map always owns the physical 0–60 °C color associations. A
        # renderer's selected range only crops that scale and clamps pixels
        # outside the visible interval to its endpoint colors.
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
        """Return the selected endpoints cropped from the fixed color scale."""
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
        # Clamp to the selected window, then look up that physical temperature
        # on the fixed 0–60 °C map. For example, 15–30 °C uses palette colors
        # 15 through 30 rather than remapping navy through white.
        temperature = max(self.min_celsius, min(self.max_celsius, value))
        cached = color_cache.get(temperature)
        if cached is not None:
            return cached

        color = self.heat_map.rgb_for_celsius(temperature)
        color_cache[temperature] = color
        return color

    def frame_colors(self, frame):
        # Convert raw sensor values first, then assign the RGB color associated
        # with each pixel's physical temperature.
        temperatures = frame_to_celsius(frame)
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
        # Build the 32x24 thermal vision first: raw value -> Celsius -> RGB.
        # The completed RGB pixels are then enlarged for video output.
        width = max(1, int(width))
        height = max(1, int(height))
        temperatures = frame_to_celsius(frame)
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
                # preserves the fixed Celsius-to-color association before the
                # low-resolution sensor image becomes a dense RGB image.
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
