#!/usr/bin/env python3
import argparse
import ctypes
import math
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, ttk
import tkinter as tk


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
RECORD_WIDTH = int(os.environ.get("MLX90642_RECORD_WIDTH", "640"))
RECORD_HEIGHT = int(os.environ.get("MLX90642_RECORD_HEIGHT", "480"))
RECORD_FPS = int(os.environ.get("MLX90642_RECORD_FPS", "8"))

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


class DriverError(RuntimeError):
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


class MLX90642Camera:
    def __init__(self, library_path):
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

    def run(self):
        try:
            # Driver compilation and blocking I2C calls stay off Tk's event
            # thread; results cross back through the thread-safe queue.
            if self.mock:
                camera = MockCamera()
            else:
                library_path = build_shared_library()
                camera = MLX90642Camera(library_path)

            self.events.put(("status", "Initializing camera"))
            camera.initialize()
            self.events.put(("status", "Live"))

            while not self.stop_event.is_set():
                # Read the current frame, then require a closed-to-open window
                # transition before fetching the following frame.
                frame = camera.read_frame()
                self.events.put(("frame", frame, time.monotonic()))
                camera.wait_for_next_frame()
        except Exception as exc:
            self.events.put(("error", str(exc)))


class ThermalRenderer:
    def __init__(self):
        self._maps = {}

    def _get_map(self, width, height):
        key = (width, height)
        if key in self._maps:
            return self._maps[key]

        if width <= 1:
            x_map = [(0, 0, 0.0)]
        else:
            x_map = []
            for x in range(width):
                source_x = x * (SENSOR_WIDTH - 1) / (width - 1)
                x0 = int(source_x)
                x1 = min(x0 + 1, SENSOR_WIDTH - 1)
                x_map.append((x0, x1, source_x - x0))

        if height <= 1:
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

    def frame_colors(self, frame):
        min_value = min(frame)
        max_value = max(frame)
        color_cache = {}
        colors = []

        for value in frame:
            red, green, blue = self._color(value, min_value, max_value, color_cache)
            colors.append(f"#{red:02x}{green:02x}{blue:02x}")

        return colors

    def render_rgb(self, frame, width, height):
        width = max(1, int(width))
        height = max(1, int(height))
        min_value = min(frame)
        max_value = max(frame)
        x_map, y_map = self._get_map(width, height)
        color_cache = {}
        output = bytearray(width * height * 3)
        offset = 0

        for y0, y1, y_weight in y_map:
            row0 = y0 * SENSOR_WIDTH
            row1 = y1 * SENSOR_WIDTH

            for x0, x1, x_weight in x_map:
                top_left = frame[row0 + x0]
                top_right = frame[row0 + x1]
                bottom_left = frame[row1 + x0]
                bottom_right = frame[row1 + x1]
                top = top_left + (top_right - top_left) * x_weight
                bottom = bottom_left + (bottom_right - bottom_left) * x_weight
                value = top + (bottom - top) * y_weight
                red, green, blue = self._color(value, min_value, max_value, color_cache)
                output[offset] = red
                output[offset + 1] = green
                output[offset + 2] = blue
                offset += 3

        return bytes(output)


class FfmpegRecorder:
    def __init__(self, path, width, height, fps):
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg is not installed")

        self.path = Path(path)
        self.width = width
        self.height = height
        self.fps = fps
        self.frames_written = 0
        self.frames_dropped = 0
        self.error = None
        self._queue = queue.Queue(maxsize=fps * 4)
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
            return

        try:
            self._queue.put_nowait(rgb)
        except queue.Full:
            self.frames_dropped += 1

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


class ThermalCameraApp(tk.Tk):
    def __init__(self, mock=False):
        super().__init__()
        self.title("MLX90642 Thermal Camera")
        self.minsize(760, 620)

        self.renderer = ThermalRenderer()
        self.events = queue.Queue()
        self.stop_event = threading.Event()
        self.worker = CameraWorker(self.events, self.stop_event, mock=mock)
        self.latest_frame = None
        self.pixel_items = []
        self.message_item = None
        self.status_text = tk.StringVar(value="Starting")
        self.min_text = tk.StringVar(value="Min --")
        self.center_text = tk.StringVar(value="Center --")
        self.max_text = tk.StringVar(value="Max --")
        self.record_text = tk.StringVar(value="Live preview")
        self.recorder = None
        self.recording_started = None
        self.record_after_id = None

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._close)
        self.worker.start()
        self.after(20, self._process_events)

    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(self, bg="#070910", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", self._handle_canvas_resize)
        self.message_item = self.canvas.create_text(
            20,
            20,
            anchor="nw",
            fill="#d7dde8",
            text="Starting camera...",
        )

        toolbar = ttk.Frame(self, padding=(10, 8))
        toolbar.grid(row=1, column=0, sticky="ew")
        toolbar.columnconfigure(8, weight=1)

        self.record_button = ttk.Button(
            toolbar,
            text="Record",
            command=self._toggle_recording,
            state="disabled",
        )
        self.record_button.grid(row=0, column=0, padx=(0, 6))

        ttk.Label(toolbar, textvariable=self.min_text).grid(row=0, column=1, padx=(10, 12))
        ttk.Label(toolbar, textvariable=self.center_text).grid(row=0, column=2, padx=(0, 12))
        ttk.Label(toolbar, textvariable=self.max_text).grid(row=0, column=3, padx=(0, 16))
        ttk.Label(toolbar, textvariable=self.record_text).grid(row=0, column=4, padx=(0, 16))
        ttk.Label(toolbar, textvariable=self.status_text).grid(row=0, column=8, sticky="e")

    def _handle_canvas_resize(self, event=None):
        self._place_canvas_message()
        self._redraw_latest()

    def _place_canvas_message(self):
        if self.message_item is None:
            return

        self.canvas.coords(self.message_item, 20, 20)

    def _set_canvas_message(self, message):
        if self.message_item is None:
            return

        self.canvas.itemconfigure(self.message_item, text=message, state="normal")
        self._place_canvas_message()

    def _display_bounds(self):
        canvas_width = max(1, self.canvas.winfo_width())
        canvas_height = max(1, self.canvas.winfo_height())

        target_ratio = SENSOR_WIDTH / SENSOR_HEIGHT
        if canvas_width / canvas_height > target_ratio:
            height = canvas_height
            width = int(height * target_ratio)
            left = (canvas_width - width) / 2
            top = 0
        else:
            width = canvas_width
            height = int(width / target_ratio)
            left = 0
            top = (canvas_height - height) / 2

        return left, top, max(1, width), max(1, height)

    def _process_events(self):
        try:
            while True:
                event = self.events.get_nowait()
                kind = event[0]

                if kind == "status":
                    self.status_text.set(event[1])
                    if event[1] != "Live":
                        self._set_canvas_message(event[1])
                elif kind == "error":
                    self.status_text.set("Camera error")
                    self.record_button.configure(state="disabled")
                    self._set_canvas_message(event[1])
                    messagebox.showerror("Camera Error", event[1])
                elif kind == "frame":
                    self.latest_frame = event[1]
                    self._update_stats(self.latest_frame)
                    self._redraw_latest()
                    if self.recorder is None:
                        self.record_button.configure(state="normal")
        except queue.Empty:
            pass

        self.after(20, self._process_events)

    def _update_stats(self, frame):
        min_value = min(frame)
        max_value = max(frame)
        center_value = frame[(SENSOR_HEIGHT // 2) * SENSOR_WIDTH + (SENSOR_WIDTH // 2)]

        self.min_text.set(f"Min {raw_to_celsius(min_value):.2f} C")
        self.center_text.set(f"Center {raw_to_celsius(center_value):.2f} C")
        self.max_text.set(f"Max {raw_to_celsius(max_value):.2f} C")

    def _redraw_latest(self, event=None):
        if self.latest_frame is None:
            return

        if self.message_item is not None:
            self.canvas.itemconfigure(self.message_item, state="hidden")

        if not self.pixel_items:
            for _ in range(SENSOR_PIXELS):
                item = self.canvas.create_rectangle(
                    0,
                    0,
                    1,
                    1,
                    width=0,
                    outline="",
                    fill="#000000",
                )
                self.pixel_items.append(item)

        left, top, width, height = self._display_bounds()
        cell_width = width / SENSOR_WIDTH
        cell_height = height / SENSOR_HEIGHT
        colors = self.renderer.frame_colors(self.latest_frame)

        for row in range(SENSOR_HEIGHT):
            y0 = top + row * cell_height
            y1 = top + (row + 1) * cell_height

            for col in range(SENSOR_WIDTH):
                index = row * SENSOR_WIDTH + col
                x0 = left + col * cell_width
                x1 = left + (col + 1) * cell_width
                self.canvas.coords(self.pixel_items[index], x0, y0, x1, y1)
                self.canvas.itemconfigure(self.pixel_items[index], fill=colors[index])

    def _toggle_recording(self):
        if self.recorder is None:
            self._start_recording()
        else:
            self._stop_recording()

    def _start_recording(self):
        if self.recorder is not None:
            return

        if self.latest_frame is None:
            messagebox.showinfo("Camera", "Waiting for the first frame.")
            return

        recordings_dir = ROOT / "recordings"
        recordings_dir.mkdir(exist_ok=True)
        path = recordings_dir / datetime.now().strftime("thermal_%Y%m%d_%H%M%S.mp4")

        try:
            self.recorder = FfmpegRecorder(path, RECORD_WIDTH, RECORD_HEIGHT, RECORD_FPS)
        except Exception as exc:
            self.recorder = None
            messagebox.showerror("Recording Error", str(exc))
            return

        self.recording_started = time.monotonic()
        self.record_button.configure(text="Stop Recording", state="normal")
        self.status_text.set("Recording")
        self._record_tick()

    def _record_tick(self):
        if self.recorder is None:
            return

        if self.latest_frame is not None:
            rgb = self.renderer.render_rgb(self.latest_frame, RECORD_WIDTH, RECORD_HEIGHT)
            self.recorder.write_frame(rgb)

        elapsed = 0.0
        if self.recording_started is not None:
            elapsed = time.monotonic() - self.recording_started

        dropped = self.recorder.frames_dropped
        suffix = f", dropped {dropped}" if dropped else ""
        self.record_text.set(
            f"REC {elapsed:0.1f}s, {self.recorder.frames_written} frames{suffix}"
        )
        self.record_after_id = self.after(int(1000 / RECORD_FPS), self._record_tick)

    def _stop_recording(self):
        if self.recorder is None:
            return

        recorder = self.recorder
        self.recorder = None
        self.record_button.configure(text="Record", state="normal")

        if self.record_after_id is not None:
            self.after_cancel(self.record_after_id)
            self.record_after_id = None

        try:
            recorder.close()
            self.record_text.set(f"Saved {recorder.path.name}")
            self.status_text.set("Live")
        except Exception as exc:
            self.record_text.set("Recording failed")
            self.status_text.set("Live")
            messagebox.showerror("Recording Error", str(exc))

    def _close(self):
        if self.recorder is not None:
            self._stop_recording()

        self.stop_event.set()
        self.destroy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    if args.build_only:
        output = build_shared_library()
        print(output)
        return 0

    app = ThermalCameraApp(mock=args.mock)
    app.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
