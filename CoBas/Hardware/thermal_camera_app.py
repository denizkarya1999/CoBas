#!/usr/bin/env python3
"""Tkinter front end for the thermal camera logic."""

import argparse
import queue
import sys
import threading
import time
from datetime import datetime
from tkinter import messagebox, ttk
import tkinter as tk

# Import moved symbols explicitly so existing callers can continue importing
# them from thermal_camera_app while their implementations live in logic.
from thermal_camera_logic import (
    CameraWorker,
    DriverError,
    FfmpegRecorder,
    LIB,
    MLX90642Camera,
    MockCamera,
    PALETTE,
    RECORD_FPS,
    RECORD_HEIGHT,
    RECORD_WIDTH,
    ROOT,
    SENSOR_HEIGHT,
    SENSOR_PIXELS,
    SENSOR_WIDTH,
    SHARED_LIB,
    ThermalRenderer,
    build_shared_library,
    frame_statistics,
    raw_to_celsius,
    signed_word,
)


class ThermalCameraApp(tk.Tk):
    def __init__(self, mock=False):
        super().__init__()
        self.title("MLX90642 Thermal Camera")
        self.minsize(760, 620)

        # The worker owns blocking camera operations and communicates through
        # this queue; only the Tk thread reads the queue and updates widgets.
        self.renderer = ThermalRenderer()
        self.events = queue.Queue()
        self.stop_event = threading.Event()
        self.worker = CameraWorker(self.events, self.stop_event, mock=mock)

        # Keep the latest sensor frame independently of the canvas so resize
        # events and recording ticks can reuse it without another camera read.
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
        # Polling with after() keeps all widget access on Tk's event thread.
        self.after(20, self._process_events)

    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(self, bg="#070910", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        # Resizing changes pixel-cell geometry, not the underlying frame data.
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
        # The expanding final column pushes status text to the right edge.
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
        # Reposition both possible canvas states: startup/error text and image.
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

        # Letterbox the 4:3 sensor grid inside any canvas size so pixels keep
        # their spatial proportions instead of stretching with the window.
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
        # Drain every event currently available so acquisition cannot build up
        # a backlog while Tk is otherwise idle. get_nowait() never blocks Tk.
        try:
            while True:
                event = self.events.get_nowait()
                kind = event[0]

                # CameraWorker's small tuple protocol keeps the background
                # thread independent of Tkinter and its thread restrictions.
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

        # Reschedule even when no events arrived; future worker results will be
        # picked up without the worker ever calling into Tk directly.
        self.after(20, self._process_events)

    def _update_stats(self, frame):
        min_value, center_value, max_value = frame_statistics(frame)

        self.min_text.set(f"Min {raw_to_celsius(min_value):.2f} C")
        self.center_text.set(f"Center {raw_to_celsius(center_value):.2f} C")
        self.max_text.set(f"Max {raw_to_celsius(max_value):.2f} C")

    def _redraw_latest(self, event=None):
        if self.latest_frame is None:
            return

        if self.message_item is not None:
            self.canvas.itemconfigure(self.message_item, state="hidden")

        if not self.pixel_items:
            # Allocate one rectangle per sensor pixel once. Later frames and
            # resizes mutate existing canvas items instead of recreating 768.
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

        # Frame data and canvas items share the same row-major pixel index.
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
            # A recorder needs a real frame immediately; starting earlier would
            # create a video process with nothing meaningful to feed it.
            messagebox.showinfo("Camera", "Waiting for the first frame.")
            return

        # Store recordings beside the application with human-readable,
        # timestamped names rather than using the current directory.
        recordings_dir = ROOT / "recordings"
        recordings_dir.mkdir(exist_ok=True)
        path = recordings_dir / datetime.now().strftime("thermal_%Y%m%d_%H%M%S.mp4")

        try:
            # FfmpegRecorder writes on its own thread so encoder or pipe delays
            # do not freeze Tk's event loop.
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
            # Recording samples the latest available frame at RECORD_FPS. This
            # deliberately decouples video timing from sensor acquisition.
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
        # Keep the timer identifier so stopping can cancel the pending callback.
        self.record_after_id = self.after(int(1000 / RECORD_FPS), self._record_tick)

    def _stop_recording(self):
        if self.recorder is None:
            return

        recorder = self.recorder
        # Detach first so a queued UI callback cannot submit more frames while
        # close() drains and terminates the encoder.
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
        # Finalize the output before destroying widgets so recording errors can
        # still be reported through the existing UI path.
        if self.recorder is not None:
            self._stop_recording()

        # CameraWorker is a daemon; the event requests cooperative shutdown
        # without making window closure wait on a blocking sensor operation.
        self.stop_event.set()
        self.destroy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    if args.build_only:
        # Installation uses this path to compile the driver without creating a
        # Tk root window or requiring camera hardware to be connected.
        output = build_shared_library()
        print(output)
        return 0

    app = ThermalCameraApp(mock=args.mock)
    app.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
