#!/usr/bin/env python3
"""Large-pixel greyscale MLX90642 camera app."""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from tkinter import messagebox
import tkinter as tk


# Find the app and project folders.
APP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = APP_ROOT.parent

# Add the shared code folder once.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import the shared camera window.
from thermal_camera_app import ThermalCameraApp  # noqa: E402

# Import camera and recorder tools.
from thermal_camera_logic import (  # noqa: E402
    FfmpegRecorder,
    RECORD_FPS,
    SENSOR_HEIGHT,
    SENSOR_WIDTH,
    build_shared_library,
)

# Import the greyscale renderer.
from grayscale_camera_logic import GrayscaleThermalRenderer  # noqa: E402

# Use 1280x960 recording by default.
RECORD_WIDTH = int(os.environ.get("MLX90642_RECORD_WIDTH", "1280"))
RECORD_HEIGHT = int(os.environ.get("MLX90642_RECORD_HEIGHT", "960"))


class GrayscaleThermalCameraApp(ThermalCameraApp):
    """Greyscale thermal camera window."""

    def __init__(self, mock=False):
        # Build the shared camera window.
        super().__init__(mock=mock)

        # Set the greyscale title.
        self.title("MLX90642 Grayscale Thermal Camera")

        # Use the greyscale renderer.
        self.renderer = GrayscaleThermalRenderer()

        # Maximize after Tk starts.
        self.after_idle(self._maximize_window)

    def _maximize_window(self):
        """Use the largest desktop window while retaining recording controls."""
        # Try the normal maximize state.
        try:
            self.state("zoomed")
            return
        except tk.TclError:
            # Try another method below.
            pass

        # Try the Linux zoom attribute.
        try:
            self.attributes("-zoomed", True)
        except tk.TclError:
            # Fill the screen as a fallback.
            width = self.winfo_screenwidth()
            height = self.winfo_screenheight()
            self.geometry(f"{width}x{height}+0+0")

    def _display_bounds(self):
        """Center the largest integer-scaled 32x24 image in the live canvas."""
        # Read the canvas size.
        canvas_width = max(1, self.canvas.winfo_width())
        canvas_height = max(1, self.canvas.winfo_height())

        # Find the largest square pixel size.
        pixel_size = min(
            canvas_width // SENSOR_WIDTH,
            canvas_height // SENSOR_HEIGHT,
        )

        if pixel_size < 1:
            # Use the base layout for tiny windows.
            return super()._display_bounds()

        # Scale the thermal image.
        width = pixel_size * SENSOR_WIDTH
        height = pixel_size * SENSOR_HEIGHT

        # Center the image.
        left = (canvas_width - width) // 2
        top = (canvas_height - height) // 2
        return left, top, width, height

    def _start_recording(self):
        """Start a timestamped MP4 inside this variant's recordings folder."""
        # Ignore a second start request.
        if self.recorder is not None:
            return

        # Wait for the first camera frame.
        if self.latest_frame is None:
            messagebox.showinfo("Camera", "Waiting for the first frame.")
            return

        # Create the local recordings folder.
        recordings_dir = APP_ROOT / "recordings"
        recordings_dir.mkdir(exist_ok=True)

        # Create a unique file name.
        path = recordings_dir / datetime.now().strftime(
            "grayscale_thermal_%Y%m%d_%H%M%S.mp4"
        )

        try:
            # Start the MP4 recorder.
            self.recorder = FfmpegRecorder(
                path,
                RECORD_WIDTH,
                RECORD_HEIGHT,
                RECORD_FPS,
            )
        except Exception as exc:
            # Show recorder errors.
            self.recorder = None
            messagebox.showerror("Recording Error", str(exc))
            return

        # Start the recording timer.
        self.recording_started = time.monotonic()

        # Update the controls.
        self.record_button.configure(text="Stop Recording", state="normal")
        self.status_text.set("Recording")
        self._record_tick()


def main():
    """Run the camera app."""
    parser = argparse.ArgumentParser()

    # Add command-line options.
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    if args.build_only:
        # Build the camera driver only.
        print(build_shared_library())
        return 0

    # Open the camera window.
    app = GrayscaleThermalCameraApp(mock=args.mock)
    app.mainloop()
    return 0


# Run only when opened directly.
if __name__ == "__main__":
    sys.exit(main())
