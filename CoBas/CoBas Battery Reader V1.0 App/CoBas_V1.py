"""CoBas V1 battery capture with mmWave, thermal, and acoustic sensing."""

from __future__ import annotations

import os
import queue
import shutil
import subprocess
import sys
import threading
import tkinter as tk
import wave
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, simpledialog, ttk

import sounddevice as sd
from About import show_about_window
from Camera.thermal_camera import ThermalCamera
from MMWave import MMWaveCaptureService
from PIL import Image, ImageTk
from Settings import SettingsWindow
from Style import COLORS, FONTS, PREVIEW, SPACING, WINDOW, apply_styles

APP_TITLE = "CoBas Battery Reader V1.0"
APP_WM_CLASS = "cobas_battery_reader_v1"
ICON_SIZES = (16, 24, 32, 48, 64, 128, 256)
PULSE_DURATION_SECONDS = 2.0
DATASET_FRAME_INTERVAL_SECONDS = 0.5
DATASET_FRAMES_PER_SECOND = 1.0 / DATASET_FRAME_INTERVAL_SECONDS
DATASET_FRAMES_PER_PULSE = int(PULSE_DURATION_SECONDS / DATASET_FRAME_INTERVAL_SECONDS)


def parse_battery_percentage(value):
    """Return a validated whole-number battery percentage."""
    text = str(value).strip()
    if text.endswith("%"):
        text = text[:-1].strip()
    if not text.isdigit():
        raise ValueError("Enter a whole-number battery percentage from 0 to 100.")
    percentage = int(text)
    if not 0 <= percentage <= 100:
        raise ValueError("Enter a whole-number battery percentage from 0 to 100.")
    return percentage


def battery_output_folder_name(percentage):
    return f"{parse_battery_percentage(percentage)}_Percent_Battery"


def battery_output_directory(base_dir, percentage):
    return os.path.join(
        base_dir,
        "Captures",
        battery_output_folder_name(percentage),
    )


class AudioInputConfiguration:
    """Store the microphone selection used by the chirp recorder."""

    def __init__(self):
        self.microphone_device_id = None
        self.microphone_device_name = "System Default Microphone"

    @staticmethod
    def get_input_microphones():
        microphones = [{"id": None, "name": "System Default Microphone"}]
        try:
            for index, device in enumerate(sd.query_devices()):
                if device.get("max_input_channels", 0) > 0:
                    microphones.append(
                        {"id": index, "name": device.get("name", f"Input {index}")}
                    )
        except (OSError, sd.PortAudioError) as error:
            print(f"[WARNING] Could not list microphones: {error}")
        return microphones

    def set_microphone_device(self, device_id, device_name):
        self.microphone_device_id = device_id
        self.microphone_device_name = device_name


class CoBasV1App:
    """Coordinate one battery capture without duplicating mmWave timing logic."""

    def __init__(self, root):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.iconname(APP_TITLE)
        self.root.withdraw()

        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.battery_percentage = self.request_battery_percentage()
        self.battery_output_name = battery_output_folder_name(self.battery_percentage)
        self.captures_dir = battery_output_directory(
            self.base_dir,
            self.battery_percentage,
        )
        os.makedirs(self.captures_dir, exist_ok=True)

        self.audio = AudioInputConfiguration()
        self.thermal_camera = ThermalCamera(output_dir=self.captures_dir)
        self.mmwave_frame_rate = DATASET_FRAMES_PER_SECOND
        self.mmwave_capture = self.new_mmwave_capture()

        self.is_closing = False
        self.is_preparing_tracking = False
        self.awaiting_radar_ready = False
        self.pulse_sequence_active = False
        self.tracking_start_token = 0
        self.pulse_process = None
        self.pulse_process_lock = threading.Lock()

        self.pulse_count_text = tk.StringVar(master=self.root, value="20")
        self.position_count_text = tk.StringVar(master=self.root, value="4")
        self.requested_pulse_count = 1
        self.requested_position_count = 1
        self.pulses_per_position = 1
        self.current_position_number = 1
        self.current_pulse_number = 0
        self.pulse_recordings = []
        self.position_segments = []
        self.pulse_sequence_started_at = None
        self.thermal_segment_started_at = None
        self.current_recording_timestamp = None
        self.export_started = False

        self.mmwave_preview_photo = None
        self.thermal_preview_photo = None
        self.thermal_preview_after_id = None
        self.mmwave_poll_after_id = None

        self.thermal_scale_mode = tk.StringVar(
            master=self.root,
            value=self.thermal_camera.display_mode,
        )
        thermal_min, thermal_max = self.thermal_camera.get_temperature_range()
        self.thermal_min_text = tk.StringVar(value=f"{thermal_min:g}")
        self.thermal_max_text = tk.StringVar(value=f"{thermal_max:g}")
        self.thermal_scale_buttons = []

        self.set_app_icon()
        self.root.geometry(f"{WINDOW['width']}x{WINDOW['height']}")
        self.root.resizable(False, False)
        self.root.minsize(WINDOW["width"], WINDOW["height"])
        self.root.maxsize(WINDOW["width"], WINDOW["height"])
        apply_styles(self.root)
        self.build_gui()
        self.refresh_info_panel()
        self.mmwave_poll_after_id = self.root.after(50, self.poll_mmwave_events)
        self.root.deiconify()

    def new_mmwave_capture(self):
        return MMWaveCaptureService(
            self.battery_percentage,
            Path(self.captures_dir) / "mmWave Data",
        )

    def request_battery_percentage(self):
        while True:
            value = simpledialog.askstring(
                "Battery Level",
                "Enter the battery percentage (for example 20 or 20%):",
                parent=self.root,
            )
            if value is None:
                self.root.destroy()
                raise SystemExit(0)
            try:
                return parse_battery_percentage(value)
            except ValueError as error:
                messagebox.showerror("Invalid Battery Level", str(error))

    def set_app_icon(self):
        icon_path = os.path.join(self.base_dir, "Assets", "icon.png")
        if not os.path.exists(icon_path):
            return
        try:
            source = Image.open(icon_path).convert("RGBA")
            self.icon_images = [
                ImageTk.PhotoImage(source.resize((size, size))) for size in ICON_SIZES
            ]
            self.root.iconphoto(True, *self.icon_images)
        except (OSError, tk.TclError) as error:
            print(f"[WARNING] App icon could not be loaded: {error}")

    def build_gui(self):
        outer = ttk.Frame(self.root, style="Main.TFrame")
        outer.pack(fill="both", expand=True)
        self.build_toolbar(outer)

        main = ttk.Frame(outer, style="Main.TFrame")
        main.pack(
            fill="both",
            expand=True,
            padx=SPACING["main_padx"],
            pady=SPACING["main_pady"],
        )
        main.columnconfigure(0, weight=4)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        preview_panel = ttk.Frame(main, style="Panel.TFrame")
        preview_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        control_panel = ttk.Frame(main, style="Panel.TFrame")
        control_panel.grid(row=0, column=1, sticky="nsew")
        self.build_preview_panel(preview_panel)
        self.build_control_panel(control_panel)

    def build_toolbar(self, parent):
        toolbar = ttk.Frame(parent, style="Toolbar.TFrame")
        toolbar.pack(fill="x")
        left = ttk.Frame(toolbar, style="Toolbar.TFrame")
        left.pack(
            side="left",
            padx=SPACING["toolbar_padx"],
            pady=SPACING["toolbar_pady"],
        )
        ttk.Button(
            left,
            text="Settings",
            style="Settings.TButton",
            command=lambda: SettingsWindow(self.root, self),
        ).pack(side="left", padx=(0, 4))
        ttk.Button(
            left,
            text="About",
            style="Settings.TButton",
            command=lambda: show_about_window(self.root),
        ).pack(side="left", padx=4)

    def build_preview_panel(self, parent):
        header = ttk.Frame(parent, style="Panel.TFrame")
        header.pack(fill="x", padx=SPACING["panel_padx"], pady=(8, 4))
        ttk.Label(
            header,
            text="Live Sensors",
            style="PanelTitle.TLabel",
        ).pack(side="left")
        self.live_indicator_label = tk.Label(
            header,
            text="● READY",
            bg=COLORS["panel_bg"],
            fg=COLORS["accent"],
            font=FONTS["status"],
        )
        self.live_indicator_label.pack(side="right")

        preview_area = ttk.Frame(parent, style="Panel.TFrame")
        preview_area.pack(
            fill="both",
            expand=True,
            padx=SPACING["panel_padx"],
            pady=(0, 6),
        )
        preview_area.columnconfigure(0, weight=1, uniform="sensor_preview")
        preview_area.columnconfigure(1, weight=1, uniform="sensor_preview")
        preview_area.rowconfigure(0, weight=1)

        mmwave_frame = ttk.Frame(preview_area, style="Panel.TFrame")
        thermal_frame = ttk.Frame(preview_area, style="Panel.TFrame")
        mmwave_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        thermal_frame.grid(row=0, column=1, sticky="nsew", padx=(4, 0))
        for frame in (mmwave_frame, thermal_frame):
            frame.rowconfigure(1, weight=1)
            frame.columnconfigure(0, weight=1)

        ttk.Label(
            mmwave_frame,
            text="Range-Angle Response Pattern",
            style="PanelText.TLabel",
        ).grid(row=0, column=0, sticky="w", pady=(0, 3))

        thermal_header = ttk.Frame(thermal_frame, style="Panel.TFrame")
        thermal_header.grid(row=0, column=0, sticky="ew", pady=(0, 3))
        ttk.Label(
            thermal_header,
            text="Thermal Sensor",
            style="PanelText.TLabel",
        ).pack(side="left")
        self.build_thermal_controls(thermal_header)

        self.mmwave_label = tk.Label(
            mmwave_frame,
            text="mmWave radar is ready.\n\nClick 'Start Tracking' to connect.",
            bg=COLORS["preview_bg"],
            fg=COLORS["muted_text"],
            font=FONTS["preview_text"],
            bd=0,
        )
        self.mmwave_label.grid(row=1, column=0, sticky="nsew")
        self.thermal_video_label = tk.Label(
            thermal_frame,
            text="Thermal sensor is ready.\n\nClick 'Start Tracking' to begin.",
            bg=COLORS["preview_bg"],
            fg=COLORS["muted_text"],
            font=FONTS["preview_text"],
            bd=0,
        )
        self.thermal_video_label.grid(row=1, column=0, sticky="nsew")

        bottom = ttk.Frame(parent, style="Panel.TFrame")
        bottom.pack(fill="x", padx=SPACING["panel_padx"], pady=(0, 8))
        self.status_label = ttk.Label(
            bottom,
            text="Status: Ready. Click 'Start Tracking' to begin.",
            style="Info.TLabel",
        )
        self.status_label.pack(side="left")
        self.record_timer_label = ttk.Label(
            bottom,
            text="Pulses: 0/20",
            style="Info.TLabel",
        )
        self.record_timer_label.pack(side="right")

    def build_thermal_controls(self, parent):
        controls = ttk.Frame(parent, style="Panel.TFrame")
        controls.pack(side="right")
        ttk.Label(controls, text="Min", style="PanelText.TLabel").pack(side="left")
        self.thermal_min_entry = ttk.Entry(
            controls,
            textvariable=self.thermal_min_text,
            width=4,
            justify="center",
        )
        self.thermal_min_entry.pack(side="left", padx=(2, 3))
        ttk.Label(controls, text="Max", style="PanelText.TLabel").pack(side="left")
        self.thermal_max_entry = ttk.Entry(
            controls,
            textvariable=self.thermal_max_text,
            width=4,
            justify="center",
        )
        self.thermal_max_entry.pack(side="left", padx=(2, 3))
        self.thermal_range_button = ttk.Button(
            controls,
            text="Set",
            width=3,
            command=self.configure_thermal_temperature_range,
        )
        self.thermal_range_button.pack(side="left", padx=(0, 3))
        for label, value in (("Color", "rgb"), ("Grey", "grayscale")):
            button = ttk.Radiobutton(
                controls,
                text=label,
                value=value,
                variable=self.thermal_scale_mode,
                command=self.change_thermal_scale_mode,
                style="ThermalScale.TRadiobutton",
            )
            button.pack(side="left", padx=(2, 0))
            self.thermal_scale_buttons.append(button)

    def build_control_panel(self, parent):
        controls = ttk.Frame(parent, style="Panel.TFrame")
        controls.pack(
            fill="x",
            padx=SPACING["panel_padx"],
            pady=(8, 4),
        )
        ttk.Label(controls, text="Tracking", style="PanelTitle.TLabel").pack(
            anchor="w", pady=(0, 3)
        )
        self.pulse_count_spinbox = self.build_spinbox_row(
            controls,
            "Chirp count",
            self.pulse_count_text,
            999,
        )
        self.position_count_spinbox = self.build_spinbox_row(
            controls,
            "Battery positions",
            self.position_count_text,
            99,
        )
        self.track_button = ttk.Button(
            controls,
            text="Start Tracking",
            style="Start.TButton",
            command=self.toggle_tracking,
        )
        self.track_button.pack(fill="x", pady=SPACING["button_pady"])

        info = ttk.Frame(parent, style="Panel.TFrame")
        info.pack(
            fill="both",
            expand=True,
            padx=SPACING["panel_padx"],
            pady=(8, 8),
        )
        ttk.Label(info, text="System", style="PanelTitle.TLabel").pack(
            anchor="w", pady=(0, 5)
        )
        self.radar_info_label = ttk.Label(
            info,
            text="mmWave: idle",
            style="PanelText.TLabel",
            wraplength=180,
        )
        self.radar_info_label.pack(anchor="w")
        self.thermal_info_label = ttk.Label(
            info,
            text="Thermal: idle",
            style="PanelText.TLabel",
            wraplength=180,
        )
        self.thermal_info_label.pack(anchor="w")
        self.microphone_info_label = ttk.Label(
            info,
            text="Mic: System Default",
            style="PanelText.TLabel",
            wraplength=180,
        )
        self.microphone_info_label.pack(anchor="w")
        self.output_info_label = ttk.Label(
            info,
            text="Output: Captures/",
            style="PanelText.TLabel",
            wraplength=180,
        )
        self.output_info_label.pack(anchor="w")
        self.rate_info_label = ttk.Label(
            info,
            text="",
            style="PanelText.TLabel",
            wraplength=180,
        )
        self.rate_info_label.pack(anchor="w")

    @staticmethod
    def build_spinbox_row(parent, label, variable, maximum):
        row = ttk.Frame(parent, style="Panel.TFrame")
        row.pack(fill="x", pady=(0, 4))
        ttk.Label(row, text=label, style="PanelText.TLabel").pack(side="left")
        spinbox = ttk.Spinbox(
            row,
            from_=1,
            to=maximum,
            increment=1,
            textvariable=variable,
            width=5,
            justify="center",
        )
        spinbox.pack(side="right")
        return spinbox

    def get_preview_dimensions(self, label):
        width = label.winfo_width()
        height = label.winfo_height()
        if width <= 1:
            width = max(1, PREVIEW["width"] // 2 - 8)
        if height <= 1:
            height = PREVIEW["height"]
        return width, height

    def update_status(self, message, indicator="● READY"):
        self.status_label.config(text=message)
        self.live_indicator_label.config(
            text=indicator,
            fg=self.get_indicator_color(indicator),
        )

    @staticmethod
    def get_indicator_color(indicator):
        if "ERROR" in indicator:
            return COLORS["error"]
        if "REC" in indicator:
            return COLORS.get("recording", "#ef4444")
        if "WARNING" in indicator or "STARTING" in indicator:
            return COLORS.get("warning", "#f59e0b")
        if "LIVE" in indicator:
            return COLORS.get("success", "#22c55e")
        return COLORS["accent"]

    def refresh_info_panel(self):
        if self.mmwave_capture.is_ready:
            radar_status = "ready"
        elif self.mmwave_capture.is_running:
            radar_status = "connecting"
        else:
            radar_status = "idle"
        self.radar_info_label.config(text=f"mmWave: {radar_status}")
        self.thermal_info_label.config(text=f"Thermal: {self.thermal_camera.status}")
        self.microphone_info_label.config(
            text=f"Mic: {self.audio.microphone_device_name}"
        )
        self.output_info_label.config(text=f"Output: {self.captures_dir}/")
        self.rate_info_label.config(
            text=(
                f"mmWave export: {self.mmwave_frame_rate:g} FPS\n"
                f"Thermal: {self.thermal_camera.record_fps:g} FPS"
            )
        )

    def apply_microphone_source_from_settings(self, device_id, device_name):
        if self.pulse_sequence_active:
            messagebox.showwarning(
                "Capture Active",
                "Stop tracking before changing the microphone.",
            )
            return
        self.audio.set_microphone_device(device_id, device_name)
        self.refresh_info_panel()
        self.update_status(f"Status: Microphone set to {device_name}")

    def change_thermal_scale_mode(self):
        requested = self.thermal_scale_mode.get()
        if not self.thermal_camera.set_display_mode(requested):
            self.thermal_scale_mode.set(self.thermal_camera.display_mode)

    def configure_thermal_temperature_range(self):
        try:
            changed = self.thermal_camera.set_temperature_range(
                self.thermal_min_text.get(),
                self.thermal_max_text.get(),
            )
        except (TypeError, ValueError) as error:
            messagebox.showerror("Invalid Thermal Range", str(error))
            return
        if not changed:
            messagebox.showinfo(
                "Thermal Range",
                "The thermal range is locked while recording.",
            )

    def set_thermal_controls_state(self, state):
        for widget in (
            self.thermal_min_entry,
            self.thermal_max_entry,
            self.thermal_range_button,
            *self.thermal_scale_buttons,
        ):
            widget.configure(state=state)

    def start_thermal_feed(self):
        self.thermal_camera.start_camera()
        if self.thermal_preview_after_id is None:
            self.update_thermal_feed()

    def update_thermal_feed(self):
        self.thermal_preview_after_id = None
        for event in self.thermal_camera.poll_events():
            if event[0] == "error":
                self.thermal_video_label.config(
                    image="",
                    text=f"Thermal sensor unavailable.\n\n{event[1]}",
                    fg=COLORS["warning"],
                )
                self.thermal_video_label.image = None

        width, height = self.get_preview_dimensions(self.thermal_video_label)
        image = self.thermal_camera.get_preview_image(width, height)
        if image is not None:
            self.thermal_preview_photo = ImageTk.PhotoImage(image=image)
            self.thermal_video_label.config(
                image=self.thermal_preview_photo,
                text="",
            )
            self.thermal_video_label.image = self.thermal_preview_photo
        self.refresh_info_panel()
        if self.thermal_camera.is_tracking:
            self.thermal_preview_after_id = self.root.after(
                150,
                self.update_thermal_feed,
            )

    def cancel_thermal_preview_loop(self):
        if self.thermal_preview_after_id is not None:
            try:
                self.root.after_cancel(self.thermal_preview_after_id)
            except tk.TclError:
                pass
            self.thermal_preview_after_id = None

    def poll_mmwave_events(self):
        self.mmwave_poll_after_id = None
        try:
            while True:
                event = self.mmwave_capture.events.get_nowait()
                if event.kind == "frame":
                    self.display_mmwave_frame(event.payload)
                elif event.kind == "ready":
                    self.refresh_info_panel()
                    self.update_status("Status: mmWave radar ready", "● LIVE")
                    if self.awaiting_radar_ready and self.pulse_sequence_active:
                        self.awaiting_radar_ready = False
                        token = self.tracking_start_token
                        self.root.after(
                            0,
                            lambda token=token: self.start_pulse_sequence(token),
                        )
                elif event.kind == "error":
                    self.update_status(
                        f"Status: mmWave error: {event.payload}",
                        "● ERROR",
                    )
                    if self.pulse_sequence_active:
                        self.abort_capture(f"mmWave capture failed: {event.payload}")
                elif event.kind == "status":
                    self.radar_info_label.config(text=f"mmWave: {event.payload}")
                elif event.kind == "stopped":
                    self.radar_info_label.config(text="mmWave: saved")
        except queue.Empty:
            pass

        if not self.is_closing:
            self.mmwave_poll_after_id = self.root.after(
                50,
                self.poll_mmwave_events,
            )

    def display_mmwave_frame(self, frame):
        try:
            image = self.mmwave_capture.preview_image(frame)
            width, height = self.get_preview_dimensions(self.mmwave_label)
            image.thumbnail((width, height), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (width, height), (2, 6, 23))
            left = (width - image.width) // 2
            top = (height - image.height) // 2
            canvas.paste(image, (left, top))
            self.mmwave_preview_photo = ImageTk.PhotoImage(canvas)
            self.mmwave_label.config(image=self.mmwave_preview_photo, text="")
            self.mmwave_label.image = self.mmwave_preview_photo
        except (RuntimeError, ValueError, tk.TclError) as error:
            self.update_status(
                f"Status: mmWave preview failed: {error}",
                "● WARNING",
            )

    def toggle_tracking(self):
        if self.is_preparing_tracking:
            return
        if self.pulse_sequence_active:
            self.stop_tracking()
        else:
            self.start_tracking()

    def start_tracking(self):
        try:
            pulse_count = int(self.pulse_count_text.get())
            position_count = int(self.position_count_text.get())
            if pulse_count < 1 or position_count < 1:
                raise ValueError
        except (TypeError, ValueError):
            messagebox.showwarning(
                "Invalid Capture Settings",
                "Chirp count and battery positions must be whole numbers above zero.",
            )
            return
        if position_count > pulse_count:
            messagebox.showwarning(
                "Too Many Positions",
                "The position count cannot be greater than the chirp count.",
            )
            return
        if pulse_count % position_count:
            messagebox.showwarning(
                "Chirps Must Divide Evenly",
                "The chirp count must divide evenly across battery positions.",
            )
            return

        self.tracking_start_token += 1
        self.requested_pulse_count = pulse_count
        self.requested_position_count = position_count
        self.pulses_per_position = pulse_count // position_count
        self.current_position_number = 1
        self.current_pulse_number = 0
        self.pulse_recordings = []
        self.position_segments = []
        self.pulse_sequence_started_at = None
        self.thermal_segment_started_at = None
        self.export_started = False
        self.pulse_sequence_active = True
        self.is_preparing_tracking = True
        self.awaiting_radar_ready = True

        self.mmwave_capture = self.new_mmwave_capture()
        self.start_thermal_feed()
        self.mmwave_capture.start()
        self.pulse_count_spinbox.configure(state="disabled")
        self.position_count_spinbox.configure(state="disabled")
        self.set_thermal_controls_state("disabled")
        self.track_button.configure(
            text="Stop Tracking",
            style="Stop.TButton",
            state="normal",
        )
        self.mmwave_label.config(
            image="",
            text="Connecting to IWR6843AOP radar...",
        )
        self.mmwave_label.image = None
        self.update_status(
            "Status: Connecting and configuring mmWave radar...",
            "● STARTING",
        )

    def get_pulse_command(self):
        pulse_folder = os.path.join(self.base_dir, "Pulse Generation")
        pulse_script = os.path.join(pulse_folder, "pulse_protocol_generator.py")
        if not os.path.exists(pulse_script):
            return None, None
        command = [
            sys.executable,
            pulse_script,
            "--mode",
            "generate-play-record",
            "--count",
            str(self.pulses_per_position),
            "--record-output-template",
            os.path.abspath(self.get_pulse_recording_template()),
            "--wait-for-start",
        ]
        if self.audio.microphone_device_id is not None:
            command.extend(["--input-device", str(self.audio.microphone_device_id)])
        return command, pulse_folder

    def get_pulse_recording_template(self):
        timestamp = (
            self.current_recording_timestamp
            or datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
        )
        return os.path.join(
            self.captures_dir,
            f"CoBas_V1_PulseVoice_{timestamp}_"
            f"Position_{self.current_position_number}_{{pulse:03d}}.wav",
        )

    def start_pulse_sequence(self, start_token):
        if start_token != self.tracking_start_token:
            return
        position_number = self.current_position_number
        pulse_count = self.pulses_per_position
        global_offset = (position_number - 1) * pulse_count
        self.current_recording_timestamp = (
            datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
        )

        def worker():
            command, pulse_folder = self.get_pulse_command()
            if command is None:
                self.root.after(
                    0,
                    lambda: self.abort_capture("Pulse generator was not found"),
                )
                return
            try:
                process = subprocess.Popen(
                    command,
                    cwd=pulse_folder,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                with self.pulse_process_lock:
                    self.pulse_process = process
            except (OSError, subprocess.SubprocessError) as error:
                self.root.after(
                    0,
                    lambda error=error: self.abort_capture(
                        f"Pulse sequence could not start: {error}"
                    ),
                )
                return

            completed = set()
            ready_received = False
            start_result = {"started": False}
            start_result_event = threading.Event()

            def start_recording():
                try:
                    start_result["started"] = self.handle_sequence_ready(
                        start_token,
                        position_number,
                    )
                finally:
                    start_result_event.set()

            for output_line in process.stdout:
                line = output_line.strip()
                if line:
                    print(f"[CHIRP SEQUENCE] {line}")
                if line.startswith("SEQUENCE_READY"):
                    ready_received = True
                    self.root.after(0, start_recording)
                    while not start_result_event.wait(timeout=0.1):
                        if start_token != self.tracking_start_token:
                            return
                    if not start_result["started"]:
                        process.terminate()
                        break
                    try:
                        process.stdin.write("START\n")
                        process.stdin.flush()
                    except (BrokenPipeError, OSError):
                        process.terminate()
                        break
                elif line.startswith("PLAYBACK_STARTED"):
                    parts = line.split(maxsplit=2)
                    if len(parts) != 3:
                        continue
                    try:
                        position_pulse = int(parts[1])
                        playback_started_at = float(parts[2])
                    except ValueError:
                        continue
                    if position_pulse == 1:
                        self.pulse_sequence_started_at = playback_started_at
                    global_pulse = global_offset + position_pulse
                    self.root.after(
                        0,
                        lambda global_pulse=global_pulse, position_pulse=position_pulse: (
                            self.handle_pulse_started(
                                start_token,
                                position_number,
                                position_pulse,
                                global_pulse,
                            )
                        ),
                    )
                elif line.startswith("PULSE_FINISHED"):
                    parts = line.split(maxsplit=2)
                    if len(parts) != 3:
                        continue
                    try:
                        position_pulse = int(parts[1])
                    except ValueError:
                        continue
                    recording_path = parts[2]
                    if position_pulse in completed or not os.path.exists(
                        recording_path
                    ):
                        continue
                    completed.add(position_pulse)
                    global_pulse = global_offset + position_pulse
                    self.pulse_recordings.append(
                        {
                            "path": recording_path,
                            "pulse_number": global_pulse,
                            "position": position_number,
                            "position_pulse_number": position_pulse,
                        }
                    )
                    self.root.after(
                        0,
                        lambda global_pulse=global_pulse: self.handle_pulse_completed(
                            start_token,
                            position_number,
                            global_pulse,
                        ),
                    )

            return_code = process.wait()
            with self.pulse_process_lock:
                if self.pulse_process is process:
                    self.pulse_process = None
            if start_token != self.tracking_start_token:
                return
            if return_code or not ready_received or len(completed) != pulse_count:
                self.root.after(
                    0,
                    lambda: self.abort_capture(
                        "Chirp playback or microphone recording failed"
                    ),
                )
                return
            self.root.after(
                0,
                lambda: self.handle_position_finished(
                    start_token,
                    position_number,
                ),
            )

        threading.Thread(target=worker, daemon=True).start()

    def handle_sequence_ready(self, start_token, position_number):
        if (
            start_token != self.tracking_start_token
            or position_number != self.current_position_number
            or not self.mmwave_capture.is_ready
        ):
            return False
        timestamp = self.current_recording_timestamp
        thermal_path = self.thermal_camera.start_recording(timestamp=timestamp)
        if not thermal_path:
            self.abort_capture("Thermal recording could not start")
            return False
        self.thermal_segment_started_at = self.thermal_camera.record_start_time
        try:
            self.mmwave_capture.resume(position_number)
        except RuntimeError as error:
            self.thermal_camera.stop_recording()
            self.abort_capture(f"mmWave capture could not resume: {error}")
            return False
        self.is_preparing_tracking = False
        target_duration = self.pulses_per_position * PULSE_DURATION_SECONDS
        self.update_status(
            f"Status: Position {position_number}/{self.requested_position_count}; "
            f"recording {target_duration:g} seconds",
            "● REC",
        )
        return True

    def handle_pulse_started(
        self,
        start_token,
        position_number,
        position_pulse_number,
        global_pulse_number,
    ):
        if start_token != self.tracking_start_token:
            return
        self.current_pulse_number = global_pulse_number
        self.update_status(
            f"Status: Position {position_number}/{self.requested_position_count}, "
            f"chirp {position_pulse_number}/{self.pulses_per_position}",
            "● REC",
        )

    def handle_pulse_completed(
        self,
        start_token,
        position_number,
        global_pulse_number,
    ):
        if start_token != self.tracking_start_token:
            return
        self.record_timer_label.config(
            text=(
                f"Position {position_number}/{self.requested_position_count} • "
                f"Chirps {global_pulse_number}/{self.requested_pulse_count}"
            )
        )

    def finalize_position_segment(self, position_number):
        existing = next(
            (
                item
                for item in self.position_segments
                if item["position"] == position_number
            ),
            None,
        )
        if existing is not None:
            return existing

        self.mmwave_capture.pause()
        position_recordings = [
            item
            for item in self.pulse_recordings
            if item.get("position") == position_number
        ]
        target_duration = len(position_recordings) * PULSE_DURATION_SECONDS
        thermal_path = None
        if self.thermal_camera.is_recording:
            thermal_path = self.thermal_camera.stop_recording()

        if not position_recordings or not thermal_path:
            self.remove_thermal_temporary_outputs()
            return None

        start_offset = 0.0
        if self.pulse_sequence_started_at and self.thermal_segment_started_at:
            start_offset = max(
                0.0,
                self.pulse_sequence_started_at - self.thermal_segment_started_at,
            )
        thermal_path = self.sync_video_duration(
            thermal_path,
            target_duration,
            start_offset,
        )
        segment = {
            "position": position_number,
            "pulse_recordings": list(position_recordings),
            "expected_image_count": (
                len(position_recordings) * DATASET_FRAMES_PER_PULSE
            ),
            "thermal_video_path": thermal_path,
            "scale_video_path": self.thermal_camera.scale_video_path,
            "temperature_log_path": self.thermal_camera.temperature_log_path,
            "temperature_average_path": (self.thermal_camera.temperature_average_path),
        }
        self.position_segments.append(segment)
        return segment

    def handle_position_finished(self, start_token, position_number):
        if start_token != self.tracking_start_token:
            return
        self.is_preparing_tracking = True
        self.update_status(
            f"Status: Finalizing Position {position_number}...",
            "● WARNING",
        )
        segment = self.finalize_position_segment(position_number)
        if segment is None:
            self.abort_capture(
                f"Position {position_number} did not produce complete data"
            )
            return
        if position_number >= self.requested_position_count:
            self.finish_capture(
                f"Completed {len(self.pulse_recordings)} chirps across "
                f"{len(self.position_segments)} positions"
            )
            return

        next_position = position_number + 1
        should_continue = messagebox.askokcancel(
            "Change Battery Position",
            f"Position {position_number} is complete.\n\n"
            f"Move the battery to Position {next_position}, then click OK.\n\n"
            "Capture is paused and repositioning time is excluded.",
            parent=self.root,
        )
        if not should_continue:
            self.finish_capture(
                f"Stopped after Position {position_number}; captured "
                f"{len(self.pulse_recordings)} chirps"
            )
            return
        self.current_position_number = next_position
        self.pulse_sequence_started_at = None
        self.thermal_segment_started_at = None
        self.update_status(
            f"Status: Preparing Position {next_position}...",
            "● STARTING",
        )
        self.start_pulse_sequence(start_token)

    def abort_capture(self, message):
        if not self.pulse_sequence_active:
            return
        print(f"[WARNING] {message}")
        self.stop_pulse_process()
        self.finalize_position_segment(self.current_position_number)
        self.finish_capture(message, warning=True)

    def finish_capture(self, message, warning=False):
        self.is_preparing_tracking = False
        self.awaiting_radar_ready = False
        self.pulse_sequence_active = False
        self.mmwave_capture.pause()
        self.mmwave_capture.stop()
        self.cancel_thermal_preview_loop()
        self.thermal_camera.stop_camera()
        self.track_button.configure(
            text="Start Tracking",
            style="Start.TButton",
            state="normal",
        )
        self.pulse_count_spinbox.configure(state="normal")
        self.position_count_spinbox.configure(state="normal")
        self.set_thermal_controls_state("normal")
        self.update_status(
            f"Status: {message}; generating battery outputs",
            "● WARNING" if warning else "● IDLE",
        )
        if self.position_segments and not self.export_started:
            self.export_started = True
            threading.Thread(
                target=self.export_battery_capture,
                daemon=True,
            ).start()

    def stop_tracking(self):
        self.tracking_start_token += 1
        self.stop_pulse_process()
        self.finalize_position_segment(self.current_position_number)
        self.finish_capture(
            f"Stopped after {len(self.pulse_recordings)} completed chirps",
            warning=True,
        )

    def stop_pulse_process(self):
        with self.pulse_process_lock:
            process = self.pulse_process
        if process is None or process.poll() is not None:
            return
        try:
            process.terminate()
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)
        except (OSError, subprocess.SubprocessError) as error:
            print(f"[WARNING] Could not stop chirp process: {error}")
        finally:
            with self.pulse_process_lock:
                if self.pulse_process is process:
                    self.pulse_process = None

    def sync_video_duration(self, video_path, duration_seconds, start_offset=0.0):
        if not video_path or not os.path.exists(video_path) or duration_seconds <= 0:
            return video_path
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            print("[WARNING] ffmpeg is unavailable; thermal duration was not aligned")
            return video_path
        base, extension = os.path.splitext(video_path)
        temporary_path = f"{base}_aligned{extension}"
        duration = f"{duration_seconds:.3f}"
        start = f"{max(0.0, start_offset):.6f}"
        command = [
            ffmpeg,
            "-y",
            "-i",
            video_path,
            "-vf",
            (
                f"trim=start={start},setpts=PTS-STARTPTS,"
                f"tpad=stop_mode=clone:stop_duration={duration},"
                f"trim=duration={duration},setpts=PTS-STARTPTS"
            ),
            "-t",
            duration,
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            temporary_path,
        ]
        try:
            subprocess.run(
                command,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            os.replace(temporary_path, video_path)
        except (OSError, subprocess.CalledProcessError) as error:
            print(f"[WARNING] Could not align thermal video: {error}")
            if os.path.exists(temporary_path):
                os.remove(temporary_path)
        return video_path

    def export_battery_capture(self):
        try:
            self.mmwave_capture.stop(wait=True, timeout=20.0)
            thermal_frames = os.path.join(self.captures_dir, "Thermal_Frames")
            if os.path.isdir(thermal_frames):
                shutil.rmtree(thermal_frames)
            os.makedirs(thermal_frames, exist_ok=True)

            for segment in sorted(
                self.position_segments,
                key=lambda item: item["position"],
            ):
                self.extract_thermal_frames(segment, thermal_frames)

            thermal_video = os.path.join(self.captures_dir, "Thermal_Video.mp4")
            self.concatenate_thermal_videos(self.position_segments, thermal_video)
            voice_path = os.path.join(self.captures_dir, "Voice_Recording.wav")
            self.build_battery_voice_recording(voice_path)
            self.combine_thermal_text_outputs(
                "temperature_log_path",
                os.path.join(self.captures_dir, "Thermal_Temperature_Log.txt"),
            )
            self.combine_thermal_text_outputs(
                "temperature_average_path",
                os.path.join(
                    self.captures_dir,
                    "Thermal_Temperature_Averages.txt",
                ),
            )
            self.clean_position_temporary_outputs()
            self.root.after(0, self.handle_export_finished)
        except (
            OSError,
            RuntimeError,
            subprocess.CalledProcessError,
            wave.Error,
        ) as error:
            print(f"[ERROR] Battery output generation failed: {error}")
            if not self.is_closing:
                self.root.after(
                    0,
                    lambda error=error: self.update_status(
                        f"Status: Output generation failed: {error}",
                        "● ERROR",
                    ),
                )

    def extract_thermal_frames(self, segment, output_directory):
        input_path = segment["thermal_video_path"]
        position = segment["position"]
        expected = segment["expected_image_count"]
        prefix = f"Thermal_Image_{self.battery_percentage}_Percent_Position_{position}_"
        command = [
            "ffmpeg",
            "-y",
            "-i",
            input_path,
            "-vf",
            f"fps={DATASET_FRAMES_PER_SECOND:g}",
            "-frames:v",
            str(expected),
            "-start_number",
            "1",
            os.path.join(output_directory, f"{prefix}%03d.png"),
        ]
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        extracted = sum(
            name.startswith(prefix) and name.endswith(".png")
            for name in os.listdir(output_directory)
        )
        if extracted != expected:
            raise RuntimeError(
                f"Expected {expected} thermal frames for Position {position}, "
                f"but extracted {extracted}"
            )

    @staticmethod
    def concatenate_thermal_videos(segments, output_path):
        inputs = [
            item["thermal_video_path"]
            for item in sorted(segments, key=lambda value: value["position"])
            if os.path.exists(item["thermal_video_path"])
        ]
        if not inputs:
            raise RuntimeError("No thermal video segments were available")
        if len(inputs) == 1:
            os.replace(inputs[0], output_path)
            return
        command = ["ffmpeg", "-y"]
        for input_path in inputs:
            command.extend(["-i", input_path])
        streams = "".join(f"[{index}:v:0]" for index in range(len(inputs)))
        command.extend(
            [
                "-filter_complex",
                f"{streams}concat=n={len(inputs)}:v=1:a=0[thermal]",
                "-map",
                "[thermal]",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-pix_fmt",
                "yuv420p",
                output_path,
            ]
        )
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for input_path in inputs:
            os.remove(input_path)

    def build_battery_voice_recording(self, output_path):
        recordings = [
            item
            for item in sorted(
                self.pulse_recordings,
                key=lambda value: value["pulse_number"],
            )
            if os.path.exists(item["path"])
        ]
        if not recordings:
            raise RuntimeError("No completed voice recordings were available")

        temporary_path = f"{output_path}.tmp.wav"
        expected_format = None
        with wave.open(temporary_path, "wb") as output_wav:
            for recording in recordings:
                with wave.open(recording["path"], "rb") as input_wav:
                    current_format = (
                        input_wav.getnchannels(),
                        input_wav.getsampwidth(),
                        input_wav.getframerate(),
                    )
                    if expected_format is None:
                        expected_format = current_format
                        output_wav.setnchannels(current_format[0])
                        output_wav.setsampwidth(current_format[1])
                        output_wav.setframerate(current_format[2])
                    elif current_format != expected_format:
                        raise RuntimeError(
                            f"Incompatible chirp recording: {recording['path']}"
                        )
                    output_wav.writeframes(input_wav.readframes(input_wav.getnframes()))
        os.replace(temporary_path, output_path)
        for recording in recordings:
            os.remove(recording["path"])

    def combine_thermal_text_outputs(self, key, output_path):
        sources = [
            (item["position"], item.get(key))
            for item in sorted(
                self.position_segments,
                key=lambda value: value["position"],
            )
            if item.get(key) and os.path.exists(item[key])
        ]
        if not sources:
            return
        temporary_path = f"{output_path}.tmp"
        with open(temporary_path, "w", encoding="utf-8") as output_file:
            for index, (position, source_path) in enumerate(sources):
                if index:
                    output_file.write("\n")
                output_file.write(f"===== Position {position} =====\n")
                with open(source_path, encoding="utf-8") as source_file:
                    output_file.write(source_file.read().rstrip())
                output_file.write("\n")
        os.replace(temporary_path, output_path)
        for _, source_path in sources:
            os.remove(source_path)

    def remove_thermal_temporary_outputs(self):
        for path in (
            self.thermal_camera.temp_video_path,
            self.thermal_camera.final_video_path,
            self.thermal_camera.scale_video_path,
            self.thermal_camera.temperature_log_path,
            self.thermal_camera.temperature_average_path,
        ):
            if path and os.path.exists(path):
                os.remove(path)

    def clean_position_temporary_outputs(self):
        final_video = os.path.join(self.captures_dir, "Thermal_Video.mp4")
        final_voice = os.path.join(self.captures_dir, "Voice_Recording.wav")
        protected = {os.path.abspath(final_video), os.path.abspath(final_voice)}
        for segment in self.position_segments:
            for key in (
                "thermal_video_path",
                "scale_video_path",
                "temperature_log_path",
                "temperature_average_path",
            ):
                path = segment.get(key)
                if (
                    path
                    and os.path.abspath(path) not in protected
                    and os.path.exists(path)
                ):
                    os.remove(path)

    def handle_export_finished(self):
        self.mmwave_label.config(
            image="",
            text="Battery capture saved.\n\nClick 'Start Tracking' to record again.",
        )
        self.mmwave_label.image = None
        self.thermal_video_label.config(
            image="",
            text="Thermal capture saved.\n\nClick 'Start Tracking' to record again.",
        )
        self.thermal_video_label.image = None
        self.update_status(
            f"Status: Battery outputs saved in {self.captures_dir}",
            "● IDLE",
        )
        messagebox.showinfo(
            "Battery Capture Saved",
            "Generated one battery-level Thermal_Video.mp4 and "
            "Voice_Recording.wav, position-aware thermal frames, and mmWave "
            f"logs, frames, and reference data.\n\nSaved in: {self.captures_dir}",
            parent=self.root,
        )

    def on_close(self):
        self.is_closing = True
        if self.mmwave_poll_after_id is not None:
            try:
                self.root.after_cancel(self.mmwave_poll_after_id)
            except tk.TclError:
                pass
        self.cancel_thermal_preview_loop()
        self.tracking_start_token += 1
        self.stop_pulse_process()
        if self.mmwave_capture.is_running:
            self.mmwave_capture.pause()
            self.mmwave_capture.stop(wait=True)
        if self.thermal_camera.is_recording:
            self.thermal_camera.stop_recording()
        self.thermal_camera.stop_camera()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk(className=APP_WM_CLASS)
    app = CoBasV1App(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
