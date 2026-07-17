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

# Import the shared camera window and legend dimensions.
from thermal_camera_app import (  # noqa: E402
    CANVAS_MARGIN,
    LEGEND_GAP,
    LEGEND_PANEL_WIDTH,
    MAX_LEGEND_HEIGHT,
    MIN_LEGEND_HEIGHT,
    ThermalCameraApp,
)

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

        # Keep the inherited recording callback on this variant's dimensions.
        # Without these overrides FFmpeg expects 1280x960 while the callback
        # supplies 640x480 chunks, which tiles several images into each frame.
        self.record_width = RECORD_WIDTH
        self.record_height = RECORD_HEIGHT
        self.record_fps = RECORD_FPS

        # Refresh the shared legend with the greyscale renderer.
        self._update_temperature_legend()

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
        """Return the integer-scaled image area beside the Celsius legend."""
        display_bounds, _ = self._canvas_layout()
        return display_bounds

    def _canvas_layout(self):
        """Lay out the camera image and its optional temperature legend."""
        # Read the canvas size.
        canvas_width = max(1, self.canvas.winfo_width())
        canvas_height = max(1, self.canvas.winfo_height())

        # Reserve a panel to the right while keeping a margin around the group.
        available_width = (
            canvas_width
            - 2 * CANVAS_MARGIN
            - LEGEND_GAP
            - LEGEND_PANEL_WIDTH
        )
        available_height = canvas_height - 2 * CANVAS_MARGIN

        # Find the largest square pixel size.
        pixel_size = min(
            available_width // SENSOR_WIDTH,
            available_height // SENSOR_HEIGHT,
        )

        if pixel_size < 1 or available_height < MIN_LEGEND_HEIGHT:
            # Tiny windows use the full base layout and hide the legend rather
            # than squeezing either the image or the labels into unreadability.
            return super()._image_only_bounds(), None

        # Scale the thermal image.
        width = pixel_size * SENSOR_WIDTH
        height = pixel_size * SENSOR_HEIGHT

        # Center the image and legend together as one visual group.
        group_width = width + LEGEND_GAP + LEGEND_PANEL_WIDTH
        left = (canvas_width - group_width) // 2
        top = (canvas_height - height) // 2
        display_bounds = (left, top, width, height)

        # Avoid an excessively tall legend on large or portrait displays.
        legend_height = min(height, MAX_LEGEND_HEIGHT)
        legend_left = left + width + LEGEND_GAP
        legend_top = (canvas_height - legend_height) // 2
        legend_bounds = (
            legend_left,
            legend_top,
            LEGEND_PANEL_WIDTH,
            legend_height,
        )
        return display_bounds, legend_bounds

    def _temperature_legend_title(self):
        """Describe the endpoints of the greyscale palette."""
        return "ESTIMATED °C RANGE\nWHITE HOT · BLACK COLD"

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
                self.record_width,
                self.record_height,
                self.record_fps,
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
