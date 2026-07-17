import ctypes
import math
import os
import queue
import shutil
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


APP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = APP_ROOT.parent
LIB = REPO_ROOT / "Hardware" / "mlx90642-library"
SHARED_LIB = Path(os.environ.get("MLX90642_SHARED_LIB", "/tmp/libmlx90642.so"))

SENSOR_WIDTH = 32
SENSOR_HEIGHT = 24
SENSOR_PIXELS = SENSOR_WIDTH * SENSOR_HEIGHT
RECORD_WIDTH = int(os.environ.get("MLX90642_RECORD_WIDTH", "640"))
RECORD_HEIGHT = int(os.environ.get("MLX90642_RECORD_HEIGHT", "480"))
RECORD_FPS = int(os.environ.get("MLX90642_RECORD_FPS", "8"))
SCALE_RECORD_WIDTH = int(os.environ.get("MLX90642_SCALE_RECORD_WIDTH", "240"))
SCALE_RECORD_HEIGHT = int(os.environ.get("MLX90642_SCALE_RECORD_HEIGHT", "480"))
READ_FAILURE_LIMIT = int(os.environ.get("MLX90642_READ_FAILURE_LIMIT", "5"))
WAIT_FAILURE_LIMIT = int(os.environ.get("MLX90642_WAIT_FAILURE_LIMIT", "30"))
WAIT_FAILURE_BACKOFF_SECONDS = float(
    os.environ.get("MLX90642_WAIT_FAILURE_BACKOFF_SECONDS", "0.05")
)

PALETTE = [
    (0, 0, 48),
    (0, 0, 130),
    (0, 74, 210),
    (0, 190, 255),
    (0, 245, 150),
    (252, 245, 60),
    (255, 156, 36),
    (235, 48, 28),
    (255, 255, 255),
]

PREVIEW_BACKGROUND = (2, 6, 23)
LEGEND_BACKGROUND = (17, 21, 29)
LEGEND_BORDER = (48, 55, 68)
LEGEND_FOREGROUND = (243, 245, 248)
LEGEND_TICK_COUNT = 5
LEGEND_PANEL_WIDTH = 116
LEGEND_GAP = 8
PREVIEW_MARGIN = 6
MIN_LEGEND_HEIGHT = 120


class DriverError(RuntimeError):
    pass


class FrameWaitError(DriverError):
    def __init__(self, status):
        self.status = status
        super().__init__(f"MLX90642_IsReadWindowOpen failed: {status}")


def build_shared_library(output=SHARED_LIB):
    if not LIB.exists():
        raise DriverError(f"MLX90642 library folder not found: {LIB}")

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
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        details = (result.stderr or result.stdout).strip()
        raise DriverError(f"driver build failed:\n{details}")

    return output


def signed_word(value):
    return value - 65536 if value >= 32768 else value


def raw_to_celsius(value):
    return value / 50.0


class MLX90642Camera:
    def __init__(self, library_path):
        self._library = ctypes.CDLL(str(library_path))
        self._frame_type = ctypes.c_uint16 * SENSOR_PIXELS

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
        status = self._library.MLX90642_PythonInit()
        if status < 0:
            raise DriverError(f"MLX90642_Init failed: {status}")

    def read_frame(self):
        frame = self._frame_type()
        status = self._library.MLX90642_PythonReadFrame(frame)
        if status < 0:
            raise DriverError(f"MLX90642_GetImage failed: {status}")

        return [signed_word(frame[index]) for index in range(SENSOR_PIXELS)]

    def wait_for_next_frame(self):
        status = self._library.MLX90642_PythonWaitForNextFrame(1000)
        if status < 0:
            raise FrameWaitError(status)


class MockThermalCamera:
    def __init__(self):
        self._tick = 0

    def initialize(self):
        return None

    def read_frame(self):
        self._tick += 1
        frame = []
        hot_x = 16 + math.sin(self._tick / 9.0) * 8
        hot_y = 12 + math.cos(self._tick / 13.0) * 6

        for y in range(SENSOR_HEIGHT):
            for x in range(SENSOR_WIDTH):
                base = 1180 + y * 7 + x * 2
                dx = x - hot_x
                dy = y - hot_y
                hot = 620 * math.exp(-(dx * dx + dy * dy) / 36.0)
                wave = 55 * math.sin((x + self._tick) / 4.0)
                frame.append(int(base + hot + wave))

        return frame

    def wait_for_next_frame(self):
        time.sleep(1.0 / RECORD_FPS)


class CameraWorker(threading.Thread):
    def __init__(self, events, stop_event, mock=False):
        super().__init__(daemon=True)
        self.events = events
        self.stop_event = stop_event
        self.mock = mock

    def _put_event(self, event):
        try:
            self.events.put_nowait(event)
            return
        except queue.Full:
            pass

        if event[0] == "frame":
            try:
                self.events.get_nowait()
            except queue.Empty:
                pass

        try:
            self.events.put_nowait(event)
        except queue.Full:
            pass

    def run(self):
        try:
            if self.mock:
                camera = MockThermalCamera()
            else:
                library_path = build_shared_library()
                camera = MLX90642Camera(library_path)

            self._put_event(("status", "Initializing thermal camera"))
            camera.initialize()
            self._put_event(("status", "Thermal live"))
            read_failures = 0
            wait_failures = 0

            while not self.stop_event.is_set():
                try:
                    frame = camera.read_frame()
                    read_failures = 0
                except DriverError as exc:
                    read_failures += 1
                    if read_failures >= READ_FAILURE_LIMIT:
                        raise

                    self._put_event((
                        "status",
                        f"Thermal read retry {read_failures}/{READ_FAILURE_LIMIT}"
                    ))
                    time.sleep(WAIT_FAILURE_BACKOFF_SECONDS)
                    continue

                self._put_event(("frame", frame, time.monotonic()))

                try:
                    camera.wait_for_next_frame()
                    wait_failures = 0
                except FrameWaitError as exc:
                    wait_failures += 1

                    if wait_failures >= WAIT_FAILURE_LIMIT:
                        raise

                    self._put_event((
                        "status",
                        f"Thermal wait retry {wait_failures}/{WAIT_FAILURE_LIMIT}"
                    ))
                    time.sleep(WAIT_FAILURE_BACKOFF_SECONDS)
        except Exception as exc:
            self._put_event(("error", str(exc)))


class ThermalRenderer:
    def __init__(self):
        self._fonts = {}

    def scale_color(self, normalized):
        """Return the RGB color at one normalized point on the thermal scale."""
        return self._color(normalized, 0.0, 1.0, {})

    def temperature_legend_title(self):
        """Describe the endpoints of this renderer's palette."""
        return "ESTIMATED °C RANGE\nWHITE HOT · BLUE COLD"

    def _color(self, value, min_value, max_value, color_cache):
        rounded = int(value)
        cached = color_cache.get(rounded)
        if cached is not None:
            return cached

        if max_value <= min_value:
            color = PALETTE[len(PALETTE) // 2]
            color_cache[rounded] = color
            return color

        t = (value - min_value) / (max_value - min_value)
        t = max(0.0, min(1.0, t))
        position = t * (len(PALETTE) - 1)
        index = int(position)
        next_index = min(index + 1, len(PALETTE) - 1)
        weight = position - index
        left = PALETTE[index]
        right = PALETTE[next_index]
        color = (
            int(left[0] + (right[0] - left[0]) * weight),
            int(left[1] + (right[1] - left[1]) * weight),
            int(left[2] + (right[2] - left[2]) * weight),
        )
        color_cache[rounded] = color
        return color

    def render_sensor_image(self, frame):
        min_value = min(frame)
        max_value = max(frame)
        color_cache = {}
        output = bytearray(SENSOR_PIXELS * 3)
        offset = 0

        for value in frame:
            red, green, blue = self._color(value, min_value, max_value, color_cache)
            output[offset] = red
            output[offset + 1] = green
            output[offset + 2] = blue
            offset += 3

        return Image.frombytes("RGB", (SENSOR_WIDTH, SENSOR_HEIGHT), bytes(output))

    def render_image(self, frame, width, height):
        width = max(1, int(width))
        height = max(1, int(height))
        image = self.render_sensor_image(frame)
        return image.resize((width, height), Image.Resampling.BICUBIC)

    def _font(self, size, bold=False):
        """Load and cache a portable legend font."""
        key = (size, bold)
        if key in self._fonts:
            return self._fonts[key]

        font_name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
        try:
            font = ImageFont.truetype(font_name, size)
        except OSError:
            font = ImageFont.load_default()

        self._fonts[key] = font
        return font

    @staticmethod
    def _fit_thermal_image(available_width, available_height):
        """Fit the 4:3 sensor view without stretching its proportions."""
        target_ratio = SENSOR_WIDTH / SENSOR_HEIGHT
        if available_width / available_height > target_ratio:
            image_height = available_height
            image_width = image_height * target_ratio
        else:
            image_width = available_width
            image_height = image_width / target_ratio

        return max(1, int(image_width)), max(1, int(image_height))

    def _draw_temperature_legend(self, image, frame, bounds):
        """Draw this renderer's live Celsius scale inside the given bounds."""
        panel_left, panel_top, panel_right, panel_bottom = bounds
        panel_width = panel_right - panel_left + 1
        panel_height = panel_bottom - panel_top + 1
        expanded = panel_width >= 160 and panel_height >= 240
        draw = ImageDraw.Draw(image)
        draw.rectangle(
            bounds,
            fill=LEGEND_BACKGROUND,
            outline=LEGEND_BORDER,
        )

        title_padding = 14 if expanded else 9
        title_top = 12 if expanded else 7
        title_font = self._font(10 if expanded else 7, bold=True)
        tick_font = self._font(12 if expanded else 9)
        draw.multiline_text(
            (panel_left + title_padding, panel_top + title_top),
            self.temperature_legend_title(),
            fill=LEGEND_FOREGROUND,
            font=title_font,
            spacing=2 if expanded else 1,
        )

        bar_left = panel_left + (16 if expanded else 10)
        bar_right = bar_left + (34 if expanded else 18)
        bar_top = panel_top + (66 if expanded else 38)
        bar_bottom = panel_bottom - (22 if expanded else 10)
        bar_height = max(1, bar_bottom - bar_top)

        for y_offset in range(bar_height + 1):
            fraction = y_offset / bar_height
            color = self.scale_color(1.0 - fraction)
            draw.line(
                (bar_left, bar_top + y_offset, bar_right, bar_top + y_offset),
                fill=color,
            )

        draw.rectangle(
            (bar_left, bar_top, bar_right, bar_bottom),
            outline=LEGEND_FOREGROUND,
        )

        min_value = min(frame)
        max_value = max(frame)
        tick_length = 8 if expanded else 5
        text_gap = 7 if expanded else 3
        for index in range(LEGEND_TICK_COUNT):
            fraction = index / (LEGEND_TICK_COUNT - 1)
            y = int(round(bar_top + fraction * bar_height))
            raw_value = max_value + (min_value - max_value) * fraction
            label = f"{raw_to_celsius(raw_value):.1f} °C"
            draw.line(
                (bar_right, y, bar_right + tick_length, y),
                fill=LEGEND_FOREGROUND,
            )

            text_box = draw.textbbox((0, 0), label, font=tick_font)
            text_height = text_box[3] - text_box[1]
            draw.text(
                (
                    bar_right + tick_length + text_gap,
                    y - text_height // 2 - text_box[1],
                ),
                label,
                fill=LEGEND_FOREGROUND,
                font=tick_font,
            )

    def render_scale_image(self, frame, width, height):
        """Render the dynamic temperature scale as its own video frame."""
        width = max(1, int(width))
        height = max(1, int(height))
        image = Image.new("RGB", (width, height), PREVIEW_BACKGROUND)
        margin = 10 if width >= 40 and height >= 40 else 0
        self._draw_temperature_legend(
            image,
            frame,
            (margin, margin, width - margin - 1, height - margin - 1),
        )
        return image

    def render_preview_image(self, frame, width, height):
        """Render a letterboxed live view with a dynamic Celsius scale."""
        width = max(1, int(width))
        height = max(1, int(height))
        preview = Image.new("RGB", (width, height), PREVIEW_BACKGROUND)

        available_height = max(1, height - 2 * PREVIEW_MARGIN)
        image_width_available = (
            width
            - 2 * PREVIEW_MARGIN
            - LEGEND_GAP
            - LEGEND_PANEL_WIDTH
        )
        show_legend = image_width_available >= SENSOR_WIDTH

        if show_legend:
            image_width, image_height = self._fit_thermal_image(
                image_width_available,
                available_height,
            )

            # A narrow view can technically fit both columns but leave too
            # little height for readable ticks; use image-only mode there.
            show_legend = image_height >= MIN_LEGEND_HEIGHT

        if show_legend:
            group_width = image_width + LEGEND_GAP + LEGEND_PANEL_WIDTH
            image_left = (width - group_width) // 2
        else:
            image_width, image_height = self._fit_thermal_image(
                width,
                height,
            )
            image_left = (width - image_width) // 2

        image_top = (height - image_height) // 2
        thermal_image = self.render_image(frame, image_width, image_height)
        preview.paste(thermal_image, (image_left, image_top))

        if not show_legend:
            return preview

        legend_left = image_left + image_width + LEGEND_GAP
        legend_top = image_top
        legend_right = legend_left + LEGEND_PANEL_WIDTH - 1
        legend_bottom = legend_top + image_height - 1
        self._draw_temperature_legend(
            preview,
            frame,
            (legend_left, legend_top, legend_right, legend_bottom),
        )

        return preview


class GrayscaleThermalRenderer(ThermalRenderer):
    """Map the current frame's coldest value to black and hottest to white."""

    def temperature_legend_title(self):
        return "ESTIMATED °C RANGE\nWHITE HOT · BLACK COLD"

    def _color(self, value, min_value, max_value, color_cache):
        rounded = int(value)
        cached = color_cache.get(rounded)
        if cached is not None:
            return cached

        if max_value <= min_value:
            intensity = 127
        else:
            normalized = (value - min_value) / (max_value - min_value)
            normalized = max(0.0, min(1.0, normalized))
            intensity = int(normalized * 255)

        color = (intensity, intensity, intensity)
        color_cache[rounded] = color
        return color


class FfmpegRecorder:
    def __init__(self, path, width, height, fps):
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
        self._queue = queue.Queue(maxsize=max(2, fps * 2))
        self._closed = False
        self._process = subprocess.Popen(
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
            if item is None:
                break

            try:
                self._process.stdin.write(item)
                self.frames_written += 1
            except Exception as exc:
                self.error = exc
                break

        try:
            self._process.stdin.close()
        except Exception:
            pass

    def write_frame(self, rgb):
        if self._closed:
            return False

        if len(rgb) != self.frame_bytes:
            self.frames_dropped += 1
            self.error = ValueError(
                "RGB frame size mismatch: "
                f"expected {self.frame_bytes} bytes for "
                f"{self.width}x{self.height}, received {len(rgb)}"
            )
            return False

        try:
            self._queue.put_nowait(rgb)
            return True
        except queue.Full:
            self.frames_dropped += 1
            return False

    def close(self):
        if self._closed:
            return

        self._closed = True
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
        return_code = self._process.wait(timeout=10)
        if return_code != 0:
            stderr = b""
            if self._process.stderr is not None:
                stderr = self._process.stderr.read()
            message = stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(message or f"ffmpeg failed with exit code {return_code}")

        if self.error is not None:
            raise RuntimeError(str(self.error))


class ThermalCamera:
    def __init__(self, output_dir="Captures", mock=False):
        self.output_dir = output_dir
        self.mock = mock or os.environ.get("COBAS_THERMAL_MOCK") == "1"
        self.rgb_renderer = ThermalRenderer()
        self.grayscale_renderer = GrayscaleThermalRenderer()
        self.display_mode = "rgb"
        self.renderer = self.rgb_renderer

        self.events = queue.Queue(maxsize=8)
        self.stop_event = threading.Event()
        self.worker = None
        self.is_tracking = False
        self.status = "Thermal idle"
        self.error = None

        self.latest_frame = None
        self.latest_frame_time = None
        self.frame_lock = threading.Lock()

        self.recorder = None
        self.scale_recorder = None
        self.record_thread = None
        self.is_recording = False
        self.record_start_time = None
        self.record_renderer = None
        self.temp_video_path = None
        self.final_video_path = None
        self.scale_video_path = None
        self.record_width = RECORD_WIDTH
        self.record_height = RECORD_HEIGHT
        self.record_fps = RECORD_FPS
        self.scale_record_width = SCALE_RECORD_WIDTH
        self.scale_record_height = SCALE_RECORD_HEIGHT

        os.makedirs(self.output_dir, exist_ok=True)

    def set_display_mode(self, mode):
        """Select the RGB or greyscale renderer when not recording."""
        normalized_mode = str(mode).strip().lower()
        if normalized_mode == "regular":
            normalized_mode = "rgb"
        if normalized_mode == "greyscale":
            normalized_mode = "grayscale"

        if normalized_mode not in ("rgb", "grayscale"):
            raise ValueError(f"Unknown thermal display mode: {mode}")

        if self.is_recording:
            return False

        self.display_mode = normalized_mode
        if normalized_mode == "grayscale":
            self.renderer = self.grayscale_renderer
        else:
            self.renderer = self.rgb_renderer

        return True

    def start_camera(self):
        if self.worker is not None and self.worker.is_alive():
            self.is_tracking = True
            return True

        self.stop_event.clear()
        self.events = queue.Queue(maxsize=8)
        self.error = None
        self.status = "Starting thermal camera"
        self.worker = CameraWorker(self.events, self.stop_event, mock=self.mock)
        self.worker.start()
        self.is_tracking = True
        return True

    def stop_camera(self):
        if (
            self.is_recording
            or self.recorder is not None
            or self.scale_recorder is not None
        ):
            self.stop_recording()
        self.stop_event.set()

        if self.worker is not None:
            self.worker.join(timeout=2)
            self.worker = None

        self.is_tracking = False
        self.status = "Thermal idle"

    def poll_events(self):
        events = []

        while True:
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break

            events.append(event)
            kind = event[0]

            if kind == "status":
                self.status = event[1]
            elif kind == "error":
                self.status = "Thermal error"
                self.error = event[1]
                self.is_tracking = False
            elif kind == "frame":
                with self.frame_lock:
                    self.latest_frame = event[1]
                    self.latest_frame_time = event[2]
                self.status = "Thermal live"

        return events

    def has_frame(self):
        with self.frame_lock:
            return self.latest_frame is not None

    def _copy_latest_frame(self):
        with self.frame_lock:
            if self.latest_frame is None:
                return None

            return list(self.latest_frame)

    def get_preview_image(self, width, height):
        frame = self._copy_latest_frame()
        if frame is None:
            return None

        return self.renderer.render_preview_image(frame, width, height)

    def _wait_for_frame(self, timeout_seconds=2.0):
        deadline = time.monotonic() + max(0.0, timeout_seconds)

        while time.monotonic() < deadline:
            self.poll_events()
            frame = self._copy_latest_frame()
            if frame is not None:
                return frame
            time.sleep(0.05)

        return None

    def start_recording(self, timestamp=None):
        if self.is_recording:
            return self.final_video_path

        frame = self._wait_for_frame(timeout_seconds=2.0)
        if frame is None:
            print("[WARNING] Thermal recording could not start: no frame yet.")
            return None

        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.temp_video_path = os.path.join(
            self.output_dir,
            f"CoBas_V1_Thermal_TempVideo_{timestamp}.mp4"
        )
        self.final_video_path = os.path.join(
            self.output_dir,
            f"CoBas_V1_Thermal_Video_{timestamp}.mp4"
        )
        self.scale_video_path = os.path.join(
            self.output_dir,
            f"CoBas_V1_Thermal_Scale_Video_{timestamp}.mp4"
        )

        try:
            self.recorder = FfmpegRecorder(
                self.temp_video_path,
                self.record_width,
                self.record_height,
                self.record_fps
            )
        except Exception as exc:
            self.recorder = None
            print(f"[WARNING] Thermal recording could not start: {exc}")
            return None

        try:
            self.scale_recorder = FfmpegRecorder(
                self.scale_video_path,
                self.scale_record_width,
                self.scale_record_height,
                self.record_fps
            )
        except Exception as exc:
            try:
                self.recorder.close()
            except Exception:
                pass

            self.recorder = None
            self.scale_recorder = None
            print(
                "[WARNING] Thermal scale video could not start: "
                f"{exc}"
            )
            return None

        self.is_recording = True
        self.record_start_time = time.time()
        self.record_renderer = self.renderer
        self.record_thread = threading.Thread(
            target=self._record_loop,
            daemon=True
        )
        self.record_thread.start()
        return self.final_video_path

    def _record_loop(self):
        next_frame_time = time.monotonic()
        frame_interval = 1.0 / max(1, self.record_fps)

        while (
            self.is_recording
            and self.recorder is not None
            and self.scale_recorder is not None
        ):
            frame = self._copy_latest_frame()

            if frame is not None:
                renderer = self.record_renderer or self.renderer
                image = renderer.render_image(
                    frame,
                    self.record_width,
                    self.record_height
                )
                frame_accepted = self.recorder.write_frame(image.tobytes())
                if self.recorder.error is not None:
                    print(
                        "[WARNING] Thermal recorder rejected a frame: "
                        f"{self.recorder.error}"
                    )
                    break

                if frame_accepted:
                    scale_image = renderer.render_scale_image(
                        frame,
                        self.scale_record_width,
                        self.scale_record_height
                    )
                    self.scale_recorder.write_frame(scale_image.tobytes())
                    if self.scale_recorder.error is not None:
                        print(
                            "[WARNING] Thermal scale recorder rejected a frame: "
                            f"{self.scale_recorder.error}"
                        )
                        break

            next_frame_time += frame_interval
            sleep_seconds = max(0.0, next_frame_time - time.monotonic())
            time.sleep(sleep_seconds)

    def stop_recording(self, audio_path=None):
        if (
            not self.is_recording
            and self.recorder is None
            and self.scale_recorder is None
        ):
            return None

        self.is_recording = False

        if self.record_thread is not None:
            self.record_thread.join(timeout=3)
            self.record_thread = None

        recorder = self.recorder
        scale_recorder = self.scale_recorder
        self.recorder = None
        self.scale_recorder = None
        self.record_start_time = None
        self.record_renderer = None

        recording_error = None
        if recorder is not None:
            try:
                recorder.close()
            except Exception as exc:
                recording_error = exc

        scale_error = None
        if scale_recorder is not None:
            try:
                scale_recorder.close()
            except Exception as exc:
                scale_error = exc

        if recorder is None:
            return None

        if recording_error is not None:
            print(f"[WARNING] Thermal recording failed: {recording_error}")
            return None

        if scale_error is not None:
            print(f"[WARNING] Thermal scale video failed: {scale_error}")

        if recorder.frames_dropped:
            print(f"[WARNING] Thermal frames dropped: {recorder.frames_dropped}")

        if scale_recorder is not None and scale_recorder.frames_dropped:
            print(
                "[WARNING] Thermal scale frames dropped: "
                f"{scale_recorder.frames_dropped}"
            )

        if (
            scale_error is None
            and self.scale_video_path
            and os.path.exists(self.scale_video_path)
        ):
            print(
                "[INFO] Thermal scale video saved: "
                f"{self.scale_video_path}"
            )

        if audio_path and os.path.exists(audio_path):
            return self._merge_video_audio(
                self.temp_video_path,
                audio_path,
                self.final_video_path
            )

        print("[WARNING] Thermal video saved without audio because audio was unavailable.")
        return self.temp_video_path

    def _merge_video_audio(self, video_path, audio_path, output_path):
        if not video_path or not audio_path:
            return video_path

        if not os.path.exists(video_path):
            print("Thermal video file does not exist.")
            return None

        if not os.path.exists(audio_path):
            print("Audio file does not exist for thermal merge.")
            return video_path

        command = [
            "ffmpeg",
            "-y",
            "-i", video_path,
            "-i", audio_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            output_path
        ]

        try:
            subprocess.run(
                command,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

            if os.path.exists(video_path):
                os.remove(video_path)

            return output_path

        except Exception as e:
            print(f"Thermal FFmpeg merge failed: {e}")
            return video_path

    def take_photo(self):
        frame = self._copy_latest_frame()
        if frame is None:
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"CoBas_V1_Thermal_Photo_{timestamp}.jpg"
        filepath = os.path.join(self.output_dir, filename)
        image = self.renderer.render_image(frame, RECORD_WIDTH, RECORD_HEIGHT)
        image.save(filepath, quality=95)
        return filepath

    def get_recording_seconds(self):
        if not self.is_recording or self.record_start_time is None:
            return 0

        return int(time.time() - self.record_start_time)


def extract_thermal_images(thermal_video_path, output_folder):
    thermal_video_path = Path(thermal_video_path)
    output_folder = Path(output_folder)

    if not thermal_video_path.exists():
        print(f"Thermal video does not exist: {thermal_video_path}")
        return None

    thermal_images_folder = output_folder / "Thermal_Images"

    if thermal_images_folder.exists():
        shutil.rmtree(thermal_images_folder)

    thermal_images_folder.mkdir(parents=True, exist_ok=True)
    output_pattern = thermal_images_folder / f"{thermal_video_path.stem}_thermal_frame%03d.jpg"

    command = [
        "ffmpeg",
        "-y",
        "-i", str(thermal_video_path),
        "-vf", "fps=0.5",
        "-start_number", "0",
        "-qscale:v", "2",
        str(output_pattern)
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    if result.returncode != 0:
        print(f"Thermal image extraction failed: {thermal_video_path}")
        return None

    print(f"Thermal images saved in: {thermal_images_folder}")
    return thermal_images_folder
