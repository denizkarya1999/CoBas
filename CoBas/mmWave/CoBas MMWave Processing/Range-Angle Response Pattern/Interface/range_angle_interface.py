#!/usr/bin/env python3
"""Tkinter interface for the live IWR6843AOP range-angle response."""

from __future__ import annotations

import queue
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from Logic.config import (
    ANGLE_POINT_COUNT,
    DISPLAY_DYNAMIC_RANGE_DB,
    DISPLAY_RANGE_BIN_COUNT,
    GUI_REFRESH_INTERVAL_MS,
    MAXIMUM_ANGLE_DEGREES,
    MAXIMUM_RANGE_METERS,
    MINIMUM_ANGLE_DEGREES,
    MINIMUM_RANGE_METERS,
)
from Logic.live_service import (
    LiveEvent,
    LiveRangeAngleService,
    PositionChangeRequest,
    SessionTimer,
    normalize_position_count,
)
from Logic.range_angle_processor import RangeAngleFrame
from Logic.session_logger import battery_session_log_paths, normalize_battery_level
from Logic.video_frame_recorder import (
    frame_directory_for_battery,
    normalize_recording_duration,
)


FIGURE_HEADER = "IWR6843AOP Range-Angle Response Pattern"


class RangeAngleInterface:
    """Render worker-thread events without performing signal work in Tk."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(FIGURE_HEADER)
        self.root.geometry("1080x720")
        self.root.minsize(760, 520)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.service = LiveRangeAngleService()
        self._closed = False

        self.status_text = tk.StringVar(value="Ready")
        self.frame_text = tk.StringVar(value="Frame: --")
        self.peak_text = tk.StringVar(value="Peak: --")
        self.timer_text = tk.StringVar(value="Remaining: -- s")
        self.display_limits_text = tk.StringVar(
            value=(
                f"Range Limits: {MINIMUM_RANGE_METERS:.2f} m to "
                f"{MAXIMUM_RANGE_METERS:.2f} m  |  "
                f"Angle Limits: {MINIMUM_ANGLE_DEGREES:.0f}° to "
                f"{MAXIMUM_ANGLE_DEGREES:+.0f}°"
            )
        )

        self._build_controls()
        self._build_plot()
        self._build_status_bar()

        self.root.after(GUI_REFRESH_INTERVAL_MS, self._poll_events)
        self.root.after(200, self.start)

    def _build_controls(self) -> None:
        controls = ttk.Frame(self.root, padding=(12, 10, 12, 4))
        controls.pack(fill=tk.X)

        self.stop_button = ttk.Button(
            controls,
            text="Stop",
            command=self.stop,
            state=tk.DISABLED,
        )
        self.stop_button.pack(side=tk.RIGHT, padx=(8, 0))

        self.start_button = ttk.Button(
            controls,
            text="Start",
            command=self.start,
        )
        self.start_button.pack(side=tk.RIGHT)

    def _build_plot(self) -> None:
        figure = Figure(figsize=(9.6, 5.8), dpi=100, constrained_layout=True)
        self.axes = figure.add_subplot(111)

        initial = np.full(
            (DISPLAY_RANGE_BIN_COUNT, ANGLE_POINT_COUNT),
            -DISPLAY_DYNAMIC_RANGE_DB,
            dtype=np.float64,
        )
        self.image = self.axes.imshow(
            initial,
            origin="lower",
            aspect="auto",
            interpolation="bilinear",
            extent=(
                MINIMUM_ANGLE_DEGREES,
                MAXIMUM_ANGLE_DEGREES,
                MINIMUM_RANGE_METERS,
                MAXIMUM_RANGE_METERS,
            ),
            cmap="viridis",
            vmin=-DISPLAY_DYNAMIC_RANGE_DB,
            vmax=0.0,
        )
        self.axes.set_title(FIGURE_HEADER)
        self.axes.set_xlabel("Angle (degrees)")
        self.axes.set_ylabel("Range (meters)")
        self.axes.set_xlim(MINIMUM_ANGLE_DEGREES, MAXIMUM_ANGLE_DEGREES)
        self.axes.set_ylim(MINIMUM_RANGE_METERS, MAXIMUM_RANGE_METERS)
        angle_ticks = np.linspace(
            MINIMUM_ANGLE_DEGREES,
            MAXIMUM_ANGLE_DEGREES,
            7,
        )
        self.axes.set_xticks(angle_ticks)
        range_ticks = np.linspace(MINIMUM_RANGE_METERS, MAXIMUM_RANGE_METERS, 7)
        self.axes.set_yticks(range_ticks)
        self.axes.set_yticklabels(
            [f"{range_value:.2f}" for range_value in range_ticks]
        )
        self.axes.grid(color="black", alpha=0.14, linewidth=0.6)

        colorbar = figure.colorbar(self.image, ax=self.axes, pad=0.025)
        colorbar.set_label("Relative Power (dB)")

        self.canvas = FigureCanvasTkAgg(figure, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=6)
        self.canvas.draw()

    def _build_status_bar(self) -> None:
        status = ttk.Frame(self.root, padding=(12, 4, 12, 10))
        status.pack(fill=tk.X)
        activity = ttk.Frame(status)
        activity.pack(fill=tk.X)
        ttk.Label(activity, textvariable=self.status_text).pack(side=tk.LEFT)
        ttk.Label(activity, textvariable=self.peak_text).pack(side=tk.RIGHT)
        ttk.Label(activity, textvariable=self.frame_text).pack(side=tk.RIGHT, padx=20)
        ttk.Label(activity, textvariable=self.timer_text).pack(side=tk.RIGHT, padx=20)
        ttk.Label(
            status,
            textvariable=self.display_limits_text,
        ).pack(anchor=tk.E, pady=(4, 0))

    def start(self) -> None:
        if self.service.is_running:
            return
        battery_level_percent = self._prompt_battery_level()
        if battery_level_percent is None:
            self.status_text.set("Start cancelled")
            return
        video_duration_seconds = self._prompt_video_duration()
        if video_duration_seconds is None:
            self.status_text.set("Start cancelled")
            return
        position_count = self._prompt_position_count()
        if position_count is None:
            self.status_text.set("Start cancelled")
            return

        self.status_text.set(
            f"Starting {battery_level_percent}% logs and "
            f"{video_duration_seconds:g} s recording across "
            f"{position_count} battery position(s)..."
        )
        self.timer_text.set(f"Remaining: {video_duration_seconds:.1f} s")
        self.start_button.configure(state=tk.DISABLED)
        self.stop_button.configure(state=tk.NORMAL)
        self.service.start(
            battery_level_percent,
            video_duration_seconds,
            position_count,
        )

    def _prompt_battery_level(self) -> int | None:
        """Ask for the battery percentage stored with both session logs."""
        while True:
            requested_level = simpledialog.askstring(
                "Battery level",
                "Enter the battery level (for example 10% or 20%):",
                parent=self.root,
            )
            if requested_level is None:
                return None
            try:
                battery_level_percent = normalize_battery_level(requested_level)
            except ValueError as error:
                messagebox.showerror(
                    "Invalid battery level",
                    str(error),
                    parent=self.root,
                )
                continue

            paths = battery_session_log_paths(battery_level_percent)
            conflicts = [
                path
                for path in (paths.raw_iq_csv, paths.range_angle_csv)
                if path.exists()
            ]
            frames_directory = frame_directory_for_battery(
                battery_level_percent
            )
            if frames_directory.exists():
                conflicts.append(frames_directory)
            if conflicts:
                filenames = "\n".join(str(path) for path in conflicts)
                messagebox.showerror(
                    "Log already exists",
                    f"Logs for {battery_level_percent}% already exist:\n{filenames}",
                    parent=self.root,
                )
                continue
            return battery_level_percent

    def _prompt_video_duration(self) -> float | None:
        """Ask how many seconds of clean spectrogram video to record."""
        while True:
            requested_seconds = simpledialog.askstring(
                "Spectrogram video duration",
                "Enter the spectrogram recording duration in seconds:",
                parent=self.root,
            )
            if requested_seconds is None:
                return None
            try:
                return normalize_recording_duration(requested_seconds)
            except ValueError as error:
                messagebox.showerror(
                    "Invalid recording duration",
                    str(error),
                    parent=self.root,
                )

    def _prompt_position_count(self) -> int | None:
        """Ask how many equal recording intervals/positions are required."""
        while True:
            requested_count = simpledialog.askstring(
                "Battery positions",
                "Enter the number of battery positions to record.\n"
                "The recording duration will be divided equally between them:",
                initialvalue="1",
                parent=self.root,
            )
            if requested_count is None:
                return None
            try:
                return normalize_position_count(requested_count)
            except ValueError as error:
                messagebox.showerror(
                    "Invalid number of positions",
                    str(error),
                    parent=self.root,
                )

    def stop(self) -> None:
        if not self.service.is_running:
            return
        self.status_text.set("Stopping radar...")
        self.stop_button.configure(state=tk.DISABLED)
        self.service.stop()

    def _poll_events(self) -> None:
        if self._closed:
            return
        try:
            while True:
                self._handle_event(self.service.events.get_nowait())
        except queue.Empty:
            pass
        self.root.after(GUI_REFRESH_INTERVAL_MS, self._poll_events)

    def _handle_event(self, event: LiveEvent) -> None:
        if event.kind == "frame" and isinstance(event.payload, RangeAngleFrame):
            self._display_frame(event.payload)
            return
        if event.kind == "timer" and isinstance(event.payload, SessionTimer):
            self.timer_text.set(
                f"Remaining: {event.payload.remaining_seconds:.1f} s"
            )
            return
        if event.kind == "position_change" and isinstance(
            event.payload,
            PositionChangeRequest,
        ):
            self._handle_position_change(event.payload)
            return

        message = str(event.payload)
        self.status_text.set(message)
        if event.kind == "error":
            self.start_button.configure(state=tk.NORMAL)
            self.stop_button.configure(state=tk.DISABLED)
        elif message == "Stopped":
            self.start_button.configure(state=tk.NORMAL)
            self.stop_button.configure(state=tk.DISABLED)

    def _handle_position_change(self, request: PositionChangeRequest) -> None:
        self.status_text.set(
            f"Paused after position {request.completed_position}/"
            f"{request.total_positions}"
        )
        self.timer_text.set(f"Remaining: {request.remaining_seconds:.1f} s")
        messagebox.showinfo(
            "Change battery position",
            f"Position {request.completed_position} of "
            f"{request.total_positions} is complete.\n\n"
            f"Move the battery to position {request.next_position}, then click OK "
            "to continue recording.\n\n"
            "The recording timer is paused.",
            parent=self.root,
        )
        if not self._closed and self.service.is_running:
            self.status_text.set(
                f"Resuming position {request.next_position}/"
                f"{request.total_positions}..."
            )
            self.service.confirm_position_change()

    def _display_frame(self, frame: RangeAngleFrame) -> None:
        self.image.set_data(frame.power_db)
        self.frame_text.set(f"Frame: {frame.frame_number}")
        self.peak_text.set(
            f"Peak: {frame.peak_range_meters:.2f} m, "
            f"{frame.peak_angle_degrees:.1f}°  |  "
            f"Array: {frame.beamforming_channel_count}/{frame.input_antenna_count} ch"
        )
        self.canvas.draw_idle()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.service.stop(wait=True)
        self.root.destroy()


def main() -> int:
    root = tk.Tk()
    RangeAngleInterface(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
