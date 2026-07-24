"""CoBas application bridge for thermal-camera logic and rendered images."""

import os
import queue
import shutil
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from . import thermal_camera_logic as backend_logic

from .grayscale_camera_logic import (
    GrayscaleThermalRenderer as BackendGrayscaleThermalRenderer,
)


APP_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = APP_ROOT.parent
HARDWARE_ROOT = REPO_ROOT / "Hardware"

# The copied Hardware module expects the vendor C source beside itself. Keep
# its code byte-identical while directing those runtime paths to the existing
# repository copy of the MLX90642 library.
backend_logic.ROOT = HARDWARE_ROOT
backend_logic.LIB = HARDWARE_ROOT / "mlx90642-library"

CameraWorker = backend_logic.CameraWorker
FfmpegRecorder = backend_logic.FfmpegRecorder
RECORD_FPS = backend_logic.RECORD_FPS
RECORD_HEIGHT = backend_logic.RECORD_HEIGHT
RECORD_WIDTH = backend_logic.RECORD_WIDTH
SENSOR_HEIGHT = backend_logic.SENSOR_HEIGHT
SENSOR_WIDTH = backend_logic.SENSOR_WIDTH
BackendColoredThermalRenderer = backend_logic.ThermalRenderer
raw_to_celsius = backend_logic.raw_to_celsius
RAW_COUNTS_PER_CELSIUS = 50.0


SCALE_RECORD_WIDTH = int(os.environ.get("MLX90642_SCALE_RECORD_WIDTH", "240"))
SCALE_RECORD_HEIGHT = int(os.environ.get("MLX90642_SCALE_RECORD_HEIGHT", "480"))

PREVIEW_BACKGROUND = (2, 6, 23)
LEGEND_BACKGROUND = (17, 21, 29)
LEGEND_BORDER = (48, 55, 68)
LEGEND_FOREGROUND = (243, 245, 248)
LEGEND_TICK_COUNT = 5
LEGEND_PANEL_WIDTH = 116
LEGEND_GAP = 8
PREVIEW_MARGIN = 6
MIN_LEGEND_HEIGHT = 120
LEGEND_MIN_MARKER_COLOR = (66, 212, 255)
LEGEND_MAX_MARKER_COLOR = (255, 91, 69)


class _PreviewRendererMixin:
    """Render the CoBas PIL preview and its live temperature legend."""

    def temperature_legend_title(self):
        return "ESTIMATED °C RANGE\nWHITE HOT · BLUE COLD"

    def _font(self, size, bold=False):
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
        target_ratio = SENSOR_WIDTH / SENSOR_HEIGHT
        if available_width / available_height > target_ratio:
            image_height = available_height
            image_width = image_height * target_ratio
        else:
            image_width = available_width
            image_height = image_width / target_ratio

        return max(1, int(image_width)), max(1, int(image_height))

    def _draw_legend_extrema(
        self,
        draw,
        extrema,
        celsius_range,
        panel_left,
        bar_left,
        bar_right,
        bar_top,
        bar_bottom,
        marker_font,
        expanded,
    ):
        """Draw changing MIN/MAX pointers on a fixed Celsius reference."""
        if extrema is None or celsius_range is None:
            return

        range_min, range_max = celsius_range
        if range_max <= range_min:
            return

        frame_min, frame_max = extrema
        bar_height = max(1, bar_bottom - bar_top)

        def marker_y(temperature):
            clamped = max(range_min, min(range_max, temperature))
            fraction = (range_max - clamped) / (range_max - range_min)
            return bar_top + fraction * bar_height

        marker_data = (
            ("MAX", frame_max, marker_y(frame_max), LEGEND_MAX_MARKER_COLOR),
            ("MIN", frame_min, marker_y(frame_min), LEGEND_MIN_MARKER_COLOR),
        )
        label_gap = 18 if expanded else 12
        label_min_y = bar_top + label_gap / 2
        label_max_y = bar_bottom - label_gap / 2
        label_positions = [
            max(label_min_y, min(label_max_y, point_y))
            for _, _, point_y, _ in marker_data
        ]

        # Keep nearby MIN and MAX labels readable without moving their pointer
        # endpoints away from the actual temperature positions.
        for index in range(1, len(label_positions)):
            label_positions[index] = max(
                label_positions[index],
                label_positions[index - 1] + label_gap,
            )

        overflow = label_positions[-1] - label_max_y
        if overflow > 0:
            label_positions = [position - overflow for position in label_positions]

        for index in range(len(label_positions) - 2, -1, -1):
            label_positions[index] = min(
                label_positions[index],
                label_positions[index + 1] - label_gap,
            )

        underflow = label_min_y - label_positions[0]
        if underflow > 0:
            label_positions = [position + underflow for position in label_positions]

        label_padding = 8 if expanded else 5
        elbow_x = bar_left - (5 if expanded else 3)
        for (name, temperature, point_y, color), label_y in zip(
            marker_data,
            label_positions,
        ):
            label = f"{name} {temperature:.2f}°"
            text_box = draw.textbbox((0, 0), label, font=marker_font)
            text_width = text_box[2] - text_box[0]
            text_height = text_box[3] - text_box[1]
            text_right = bar_left - label_padding
            text_x = max(panel_left + 2, text_right - text_width)
            text_y = label_y - text_height / 2 - text_box[1]

            draw.text(
                (text_x, text_y),
                label,
                fill=color,
                font=marker_font,
            )
            draw.line(
                (
                    text_right + 2,
                    label_y,
                    elbow_x,
                    label_y,
                    elbow_x,
                    point_y,
                    bar_right,
                    point_y,
                ),
                fill=color,
                width=2 if expanded else 1,
            )

    def _draw_temperature_legend(self, image, frame, bounds):
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
        marker_font = self._font(9 if expanded else 7, bold=True)
        draw.multiline_text(
            (panel_left + title_padding, panel_top + title_top),
            self.temperature_legend_title(),
            fill=LEGEND_FOREGROUND,
            font=title_font,
            spacing=2 if expanded else 1,
        )

        celsius_range = self.legend_celsius_range(frame)
        extrema = self.legend_extrema_celsius(frame)
        if extrema is None:
            bar_left = panel_left + (16 if expanded else 10)
        else:
            # Reserve the left side for live MIN/MAX labels while retaining
            # enough room on the right for the fixed Celsius tick labels.
            bar_left = panel_left + (82 if expanded else 48)
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

        tick_length = 8 if expanded else 5
        text_gap = 7 if expanded else 3
        for index in range(LEGEND_TICK_COUNT):
            fraction = index / (LEGEND_TICK_COUNT - 1)
            y = int(round(bar_top + fraction * bar_height))
            if celsius_range is None:
                label = "--.-- °C"
            else:
                min_celsius, max_celsius = celsius_range
                temperature = max_celsius + (
                    min_celsius - max_celsius
                ) * fraction
                label = f"{temperature:.2f} °C"
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

        self._draw_legend_extrema(
            draw,
            extrema,
            celsius_range,
            panel_left,
            bar_left,
            bar_right,
            bar_top,
            bar_bottom,
            marker_font,
            expanded,
        )

    def render_sensor_image(self, frame):
        rgb = self.render_rgb(frame, SENSOR_WIDTH, SENSOR_HEIGHT)
        return Image.frombytes("RGB", (SENSOR_WIDTH, SENSOR_HEIGHT), rgb)

    def render_image(self, frame, width, height):
        width = max(1, int(width))
        height = max(1, int(height))
        rgb = self.render_rgb(frame, width, height)
        return Image.frombytes("RGB", (width, height), rgb)

    def render_preview_frame(self, frame, width, height):
        image = self.render_sensor_image(frame)
        return image.resize((width, height), Image.Resampling.NEAREST)

    def render_scale_image(self, frame, width, height):
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
            show_legend = image_height >= MIN_LEGEND_HEIGHT

        if show_legend:
            group_width = image_width + LEGEND_GAP + LEGEND_PANEL_WIDTH
            image_left = (width - group_width) // 2
        else:
            image_width, image_height = self._fit_thermal_image(width, height)
            image_left = (width - image_width) // 2

        image_top = (height - image_height) // 2
        thermal_image = self.render_preview_frame(
            frame,
            image_width,
            image_height,
        )
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


class ColoredThermalRenderer(
    BackendColoredThermalRenderer,
    _PreviewRendererMixin,
):
    """Use the fixed Hardware Celsius colors in the CoBas presentation."""

    def __init__(self, min_celsius=None, max_celsius=None):
        BackendColoredThermalRenderer.__init__(
            self,
            min_celsius,
            max_celsius,
        )
        self._fonts = {}

    def temperature_legend_title(self):
        min_celsius, max_celsius = self.legend_celsius_range()
        color_resolution = self.COLOR_RESOLUTION_CELSIUS
        return (
            f"0–60 °C · {color_resolution:.2f} °C COLORS\n"
            f"DISPLAYING {min_celsius:g}–{max_celsius:g} °C"
        )


class GrayscaleThermalRenderer(
    BackendGrayscaleThermalRenderer,
    _PreviewRendererMixin,
):
    """Use the dynamic Hardware grayscale backend in CoBas."""

    def __init__(self, *args, **kwargs):
        BackendGrayscaleThermalRenderer.__init__(self, *args, **kwargs)
        self._fonts = {}

    def temperature_legend_title(self):
        celsius_range = self.legend_celsius_range()
        if celsius_range is None:
            return "ESTIMATED °C RANGE\nWHITE HOT · BLACK COLD"

        min_celsius, max_celsius = celsius_range
        return (
            f"FIXED {min_celsius:g}–{max_celsius:g} °C RANGE\n"
            "WHITE HOT · BLACK COLD"
        )


class ThermalCamera:
    """Bridge thermal acquisition and rendering into the CoBas application."""

    def __init__(
        self,
        output_dir="Captures",
        mock=False,
        min_celsius=None,
        max_celsius=None,
    ):
        self.output_dir = output_dir
        self.mock = mock or os.environ.get("COBAS_THERMAL_MOCK") == "1"
        self.rgb_renderer = ColoredThermalRenderer(min_celsius, max_celsius)
        selected_min, selected_max = self.rgb_renderer.legend_celsius_range()
        self.grayscale_renderer = GrayscaleThermalRenderer(
            min_celsius=selected_min,
            max_celsius=selected_max,
            display_min=selected_min * RAW_COUNTS_PER_CELSIUS,
            display_max=selected_max * RAW_COUNTS_PER_CELSIUS,
        )
        self.display_mode = "rgb"
        self.renderer = self.rgb_renderer

        # Hardware's worker uses blocking queue.put(), matching its standalone
        # GUI, so this bridge intentionally supplies an unbounded event queue.
        self.events = queue.Queue()
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
        self.record_start_monotonic = None
        self.record_renderer = None
        self.temp_video_path = None
        self.final_video_path = None
        self.scale_video_path = None
        self.temperature_log_path = None
        self.temperature_average_path = None
        self.temperature_log_file = None
        self.temperature_log_start_time = None
        self.temperature_recording_started_at = None
        self.temperature_sample_count = 0
        self.temperature_minimum_total = 0.0
        self.temperature_maximum_total = 0.0
        self.temperature_logging_error = None
        self.record_width = RECORD_WIDTH
        self.record_height = RECORD_HEIGHT
        self.record_fps = RECORD_FPS
        self.scale_record_width = SCALE_RECORD_WIDTH
        self.scale_record_height = SCALE_RECORD_HEIGHT

        os.makedirs(self.output_dir, exist_ok=True)

    def get_temperature_range(self):
        """Return the stable Celsius endpoints used by both renderers."""
        return self.rgb_renderer.legend_celsius_range()

    def set_temperature_range(self, min_celsius, max_celsius):
        """Apply a new fixed spectrum range unless recording has started."""
        if self.is_recording:
            return False

        # Construct both replacements before publishing either one, so invalid
        # input cannot leave the colored and grayscale modes out of sync.
        rgb_renderer = ColoredThermalRenderer(min_celsius, max_celsius)
        selected_min, selected_max = rgb_renderer.legend_celsius_range()
        grayscale_renderer = GrayscaleThermalRenderer(
            min_celsius=selected_min,
            max_celsius=selected_max,
            display_min=selected_min * RAW_COUNTS_PER_CELSIUS,
            display_max=selected_max * RAW_COUNTS_PER_CELSIUS,
        )

        self.rgb_renderer = rgb_renderer
        self.grayscale_renderer = grayscale_renderer
        if self.display_mode == "grayscale":
            self.renderer = self.grayscale_renderer
        else:
            self.renderer = self.rgb_renderer
        return True

    def set_display_mode(self, mode):
        """Select the colored or grayscale renderer when not recording."""
        normalized_mode = str(mode).strip().lower()
        if normalized_mode in ("regular", "color", "colored", "coloured"):
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
        self.events = queue.Queue()
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
        """Return a PIL image ready for the app's Tkinter preview label."""
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

    def _start_temperature_logging(self):
        """Create the live per-frame thermal min/max log for a recording."""
        self.temperature_sample_count = 0
        self.temperature_minimum_total = 0.0
        self.temperature_maximum_total = 0.0
        self.temperature_logging_error = None
        self.temperature_log_start_time = time.monotonic()
        self.temperature_recording_started_at = datetime.now()

        try:
            self.temperature_log_file = open(
                self.temperature_log_path,
                "w",
                encoding="utf-8",
                buffering=1,
            )
            self.temperature_log_file.write(
                "Thermal Temperature Min/Max Log\n"
                f"Recording started: "
                f"{self.temperature_recording_started_at.isoformat(timespec='milliseconds')}\n"
                "Units: degrees Celsius\n\n"
                "Sample\tElapsed Seconds\tMinimum Temperature\tMaximum Temperature\n"
            )
        except Exception:
            if self.temperature_log_file is not None:
                self.temperature_log_file.close()
                self.temperature_log_file = None
            raise

    def _log_frame_temperatures(self, frame):
        """Append one recorded frame's extrema and update running averages."""
        if self.temperature_log_file is None:
            return

        minimum_raw, _, maximum_raw = backend_logic.frame_statistics(frame)
        minimum_celsius = raw_to_celsius(minimum_raw)
        maximum_celsius = raw_to_celsius(maximum_raw)
        sample_number = self.temperature_sample_count + 1
        elapsed_seconds = max(
            0.0,
            time.monotonic() - self.temperature_log_start_time,
        )

        self.temperature_log_file.write(
            f"{sample_number}\t{elapsed_seconds:.3f}\t"
            f"{minimum_celsius:.2f}\t{maximum_celsius:.2f}\n"
        )
        self.temperature_sample_count = sample_number
        self.temperature_minimum_total += minimum_celsius
        self.temperature_maximum_total += maximum_celsius

    def _finish_temperature_logging(self):
        """Close the live log and write average per-frame min/max values."""
        if self.temperature_log_file is not None:
            try:
                self.temperature_log_file.close()
            except Exception as exc:
                self.temperature_logging_error = (
                    self.temperature_logging_error or exc
                )
            finally:
                self.temperature_log_file = None

        if not self.temperature_average_path:
            return

        stopped_at = datetime.now()
        with open(self.temperature_average_path, "w", encoding="utf-8") as file:
            file.write("Average Thermal Minimum/Maximum Temperatures\n")
            if self.temperature_recording_started_at is not None:
                file.write(
                    "Recording started: "
                    f"{self.temperature_recording_started_at.isoformat(timespec='milliseconds')}\n"
                )
            file.write(
                f"Recording stopped: {stopped_at.isoformat(timespec='milliseconds')}\n"
                "Units: degrees Celsius\n"
                f"Samples: {self.temperature_sample_count}\n\n"
            )

            if self.temperature_sample_count:
                average_minimum = (
                    self.temperature_minimum_total
                    / self.temperature_sample_count
                )
                average_maximum = (
                    self.temperature_maximum_total
                    / self.temperature_sample_count
                )
                file.write(
                    f"Average minimum temperature: {average_minimum:.2f}\n"
                    f"Average maximum temperature: {average_maximum:.2f}\n"
                )
            else:
                file.write(
                    "Average minimum temperature: unavailable\n"
                    "Average maximum temperature: unavailable\n"
                )

            if self.temperature_logging_error is not None:
                file.write(
                    "\nLogging warning: "
                    f"{self.temperature_logging_error}\n"
                )

        self.temperature_log_start_time = None
        self.temperature_recording_started_at = None

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
        self.temperature_log_path = os.path.join(
            self.output_dir,
            f"CoBas_V1_Thermal_Temperature_Log_{timestamp}.txt"
        )
        self.temperature_average_path = os.path.join(
            self.output_dir,
            f"CoBas_V1_Thermal_Temperature_Averages_{timestamp}.txt"
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
            print(f"[WARNING] Thermal scale video could not start: {exc}")
            return None

        try:
            self._start_temperature_logging()
        except Exception as exc:
            try:
                self.scale_recorder.close()
            except Exception:
                pass
            try:
                self.recorder.close()
            except Exception:
                pass

            self.recorder = None
            self.scale_recorder = None
            print(f"[WARNING] Thermal temperature log could not start: {exc}")
            return None

        self.is_recording = True
        self.record_start_time = time.time()
        self.record_start_monotonic = time.monotonic()
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
                self.recorder.write_frame(image.tobytes())
                if self.recorder.error is not None:
                    print(
                        "[WARNING] Thermal recorder rejected a frame: "
                        f"{self.recorder.error}"
                    )
                    break

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

                try:
                    self._log_frame_temperatures(frame)
                except Exception as exc:
                    self.temperature_logging_error = exc
                    print(
                        "[WARNING] Thermal temperature logging failed: "
                        f"{exc}"
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
        recording_end_monotonic = time.monotonic()

        if self.record_thread is not None:
            self.record_thread.join(timeout=3)
            self.record_thread = None

        recorder = self.recorder
        scale_recorder = self.scale_recorder
        self.recorder = None
        self.scale_recorder = None
        self.record_start_time = None
        self.record_renderer = None

        try:
            self._finish_temperature_logging()
        except Exception as exc:
            print(f"[WARNING] Thermal temperature summary failed: {exc}")

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
            self.record_start_monotonic = None
            return None

        if self.record_start_monotonic is not None:
            recording_duration = (
                recording_end_monotonic - self.record_start_monotonic
            )
            average_fps = (
                recorder.frames_written / recording_duration
                if recording_duration > 0
                else 0.0
            )
            print(
                f"[INFO] Thermal camera average FPS: {average_fps:.2f} "
                f"({recorder.frames_written} frames over "
                f"{recording_duration:.2f} seconds)"
            )

        self.record_start_monotonic = None

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
            print(f"[INFO] Thermal scale video saved: {self.scale_video_path}")

        if (
            self.temperature_log_path
            and os.path.exists(self.temperature_log_path)
        ):
            print(
                "[INFO] Thermal temperature log saved: "
                f"{self.temperature_log_path}"
            )

        if (
            self.temperature_average_path
            and os.path.exists(self.temperature_average_path)
        ):
            print(
                "[INFO] Thermal temperature averages saved: "
                f"{self.temperature_average_path}"
            )

        if audio_path and os.path.exists(audio_path):
            return self._merge_video_audio(
                self.temp_video_path,
                audio_path,
                self.final_video_path
            )

        print(
            "[WARNING] Thermal video saved without audio because audio "
            "was unavailable."
        )
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
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", "aac",
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
        except Exception as exc:
            print(f"Thermal FFmpeg merge failed: {exc}")
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
    """Extract one thermal image every two seconds from a recorded video."""
    thermal_video_path = Path(thermal_video_path)
    output_folder = Path(output_folder)

    if not thermal_video_path.exists():
        print(f"Thermal video does not exist: {thermal_video_path}")
        return None

    thermal_images_folder = output_folder / "Thermal_Images"

    if thermal_images_folder.exists():
        shutil.rmtree(thermal_images_folder)

    thermal_images_folder.mkdir(parents=True, exist_ok=True)
    output_pattern = (
        thermal_images_folder
        / f"{thermal_video_path.stem}_thermal_frame%03d.jpg"
    )

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
