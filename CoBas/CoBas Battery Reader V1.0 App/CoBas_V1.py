"""CoBas V1 battery capture with mmWave, thermal, and acoustic sensing."""

from __future__ import annotations

import os
import queue
import shutil
import subprocess
import sys
import threading
import tempfile
import tkinter as tk
import wave
from datetime import datetime
from pathlib import Path
from tkinter import ttk
from TouchUI import NumberControl, ask_number, dialogs as messagebox

from AudioDevices import list_audio_choices
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
DATASET_FRAMES_PER_PULSE = 1
SENSOR_FRAME_CAPTURE_TIMEOUT_SECONDS = 3.0


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
    """Keep stable microphone and speaker selections for each chirp process."""

    def __init__(self):
        self.microphone_device_id = None
        self.microphone_device_name = "Automatic USB microphone"
        self.speaker_device_id = None
        self.speaker_device_name = "Automatic USB speaker"

    @staticmethod
    def _choices(kind):
        label = "Automatic USB microphone" if kind == "input" else "Automatic USB speaker"
        return [{"id": None, "name": label}] + [
            {"id": item["selector"], "name": item["label"]}
            for item in list_audio_choices(kind)
        ]

    @classmethod
    def get_input_microphones(cls):
        return cls._choices("input")

    @classmethod
    def get_output_speakers(cls):
        return cls._choices("output")

    def set_microphone_device(self, device_id, device_name):
        self.microphone_device_id = device_id
        self.microphone_device_name = device_name

    def set_speaker_device(self, device_id, device_name):
        self.speaker_device_id = device_id
        self.speaker_device_name = device_name


class CoBasV1App:
    """Coordinate one battery capture without duplicating mmWave timing logic."""

    def __init__(self, root):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.iconname(APP_TITLE)
        self.root.withdraw()
        self.root.geometry("800x480+0+0")
        self.root.overrideredirect(True)
        self.touch_mode = True
        apply_styles(self.root)

        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.battery_percentage = self.request_battery_percentage()
        self.battery_output_name = battery_output_folder_name(self.battery_percentage)
        self.prepare_capture_session()
        self.capture_attempted = False
        self.ui_callbacks = queue.Queue()
        self.export_in_progress = False

        self.audio = AudioInputConfiguration()
        self.thermal_camera = ThermalCamera(
            output_dir=self.thermal_references_dir,
        )
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
        self.voice_recording_count = 0
        self.position_segments = []
        self.thermal_frame_records = []
        self.thermal_frames_staging_directory = None
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
        self.root.resizable(True, True)
        self.root.minsize(760, 440)
        self.build_gui()
        self.refresh_info_panel()
        self.mmwave_poll_after_id = self.root.after(50, self.poll_mmwave_events)
        self.root.deiconify()

    def prepare_capture_session(self):
        """Reserve a fresh output folder without modifying earlier captures."""
        battery_directory = Path(battery_output_directory(self.base_dir, self.battery_percentage))
        battery_directory.mkdir(parents=True, exist_ok=True)
        prefix = datetime.now().astimezone().strftime("Session_%Y%m%d_%H%M%S_")
        self.captures_dir = tempfile.mkdtemp(prefix=prefix, dir=battery_directory)
        self.mmwave_frames_dir = os.path.join(self.captures_dir, "mmWave Frames")
        self.thermal_frames_dir = os.path.join(self.captures_dir, "Thermal Frames")
        self.voices_dir = os.path.join(self.captures_dir, "Voices")
        self.references_dir = os.path.join(self.captures_dir, "References")
        self.mmwave_references_dir = os.path.join(self.references_dir, "mmWave References")
        self.thermal_references_dir = os.path.join(self.references_dir, "Thermal Data References")
        for path in (self.mmwave_frames_dir, self.thermal_frames_dir, self.voices_dir,
                     self.mmwave_references_dir, self.thermal_references_dir):
            os.makedirs(path, exist_ok=True)
        if hasattr(self, "thermal_camera"):
            self.thermal_camera.output_dir = self.thermal_references_dir

    def post_ui(self, callback, *args, token=None):
        """Workers enqueue updates; only the Tk thread may touch widgets or capture state."""
        if not self.is_closing:
            self.ui_callbacks.put((token, callback, args))

    def drain_ui_callbacks(self):
        for _ in range(100):
            try:
                token, callback, args = self.ui_callbacks.get_nowait()
            except queue.Empty:
                break
            if self.is_closing:
                break
            if token is not None and token != self.tracking_start_token:
                continue
            callback(*args)

    def new_mmwave_capture(self):
        return MMWaveCaptureService(
            self.battery_percentage,
            self.mmwave_references_dir,
            frames_directory=self.mmwave_frames_dir,
            references_directory=self.mmwave_references_dir,
        )

    def request_battery_percentage(self):
        value = ask_number(self.root, "Battery percentage (%)", minimum=0, maximum=100)
        if value is None:
            self.root.destroy()
            raise SystemExit(0)
        return value

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

        # Reserve the capture action and status before allocating preview space.
        footer = ttk.Frame(outer, style="Panel.TFrame", padding=(10, 6), height=76)
        footer.pack(side="bottom", fill="x")
        footer.pack_propagate(False)
        self.track_button = ttk.Button(footer, text="Start Tracking", style="Start.TButton",
                                       command=self.toggle_tracking, width=17)
        self.track_button.pack(side="right", padx=(10, 0))
        self.record_timer_label = ttk.Label(footer, text="Pulses: 0/20", style="PanelTitle.TLabel")
        self.record_timer_label.pack(anchor="w")
        self.status_label = ttk.Label(footer, text="Ready. Tap Start Tracking to begin.",
                                      style="Info.TLabel", wraplength=500)
        self.status_label.pack(fill="x", expand=True, anchor="w")
        self.status_label.bind("<Button-1>", lambda e: messagebox.showinfo("Capture status", self.status_label.cget("text"), parent=self.root))
        footer.bind("<Configure>", lambda e: self.status_label.configure(wraplength=max(220, e.width - 270)))

        self.notebook = ttk.Notebook(outer, style="Touch.TNotebook")
        self.notebook.pack(fill="both", expand=True, padx=8, pady=6)
        preview = ttk.Frame(self.notebook, style="Panel.TFrame")
        setup = ttk.Frame(self.notebook, style="Panel.TFrame")
        system = ttk.Frame(self.notebook, style="Panel.TFrame")
        self.notebook.add(preview, text="Live sensors")
        self.notebook.add(setup, text="Capture setup")
        self.notebook.add(system, text="System")
        self.build_preview_panel(preview)
        self.build_control_panel(setup)
        self.build_system_panel(system)

    def build_toolbar(self, parent):
        toolbar = ttk.Frame(parent, style="Toolbar.TFrame", padding=(8, 4))
        toolbar.pack(fill="x")
        ttk.Label(toolbar, text=f"CoBas  ·  {self.battery_percentage}%", style="ToolbarTitle.TLabel").pack(side="left", padx=(0, 10))
        ttk.Button(toolbar, text="Close", width=5, style="Danger.TButton", command=self.request_close).pack(side="right", padx=(6,0))
        self.window_button = ttk.Button(toolbar, text="Window", width=7, style="Tool.TButton", command=self.toggle_touch_mode)
        self.window_button.pack(side="right", padx=3)
        ttk.Button(toolbar, text="About", width=5, style="Tool.TButton", command=lambda: show_about_window(self.root)).pack(side="right", padx=3)
        ttk.Button(toolbar, text="Settings", width=7, style="Settings.TButton", command=lambda: SettingsWindow(self.root, self)).pack(side="right", padx=3)

    def toggle_touch_mode(self):
        self.touch_mode = not self.touch_mode
        self.root.overrideredirect(self.touch_mode)
        if self.touch_mode:
            self.root.geometry("800x480+0+0")
        else:
            width = min(980, self.root.winfo_screenwidth())
            height = min(620, self.root.winfo_screenheight() - 40)
            x = 810 if self.root.winfo_screenwidth() >= width + 810 else 0
            self.root.geometry(f"{width}x{height}+{x}+30")
        self.window_button.configure(text="Window" if self.touch_mode else "Touch")

    def request_close(self):
        if self.export_in_progress:
            messagebox.showinfo("Saving capture", "Please wait for saving to finish before closing CoBas.", parent=self.root)
            return
        if self.pulse_sequence_active:
            if not messagebox.askokcancel("Close CoBas?", "A capture or export may still be running. Close the app?", parent=self.root):
                return
        self.on_close()

    def build_preview_panel(self, parent):
        header = ttk.Frame(parent, style="Panel.TFrame")
        header.pack(fill="x", padx=10, pady=(5, 4))
        ttk.Label(header, text="Live sensors", style="PanelTitle.TLabel").pack(side="left")
        self.live_indicator_label = tk.Label(header, text="● READY", bg=COLORS["panel_bg"],
                                            fg=COLORS["accent"], font=FONTS["status"])
        self.live_indicator_label.pack(side="right")
        area = ttk.Frame(parent, style="Panel.TFrame")
        area.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        area.rowconfigure(1, weight=1)
        for col, title in enumerate(("mmWave radar", "Thermal camera")):
            area.columnconfigure(col, weight=1, uniform="sensor")
            ttk.Label(area, text=title, style="PanelText.TLabel").grid(row=0, column=col, sticky="w", padx=4, pady=(0,4))
        self.mmwave_label = tk.Label(area, text="Radar ready.\n\nTap Start Tracking to connect.",
                                     bg=COLORS["preview_bg"], fg=COLORS["muted_text"],
                                     font=FONTS["preview_text"], width=1, height=1, wraplength=330, bd=0)
        self.thermal_video_label = tk.Label(area, text="Thermal camera ready.\n\nTap Start Tracking to begin.",
                                            bg=COLORS["preview_bg"], fg=COLORS["muted_text"],
                                            font=FONTS["preview_text"], width=1, height=1, wraplength=330, bd=0)
        for col, label in enumerate((self.mmwave_label, self.thermal_video_label)):
            label.grid(row=1, column=col, sticky="nsew", padx=4)
            label.bind("<Configure>", lambda e, widget=label: widget.configure(wraplength=max(100, e.width - 12)))

    def build_control_panel(self, parent):
        parent.columnconfigure(0, weight=1, uniform="setup")
        parent.columnconfigure(1, weight=1, uniform="setup")
        controls = ttk.Frame(parent, style="Panel.TFrame", padding=10)
        controls.grid(row=0, column=0, sticky="nsew")
        thermal = ttk.Frame(parent, style="Panel.TFrame", padding=10)
        thermal.grid(row=0, column=1, sticky="nsew")
        ttk.Label(controls, text="Capture", style="PanelTitle.TLabel").pack(anchor="w", pady=(0,5))
        self.pulse_count_spinbox = self.build_spinbox_row(controls, "Chirp count", self.pulse_count_text, 999)
        self.position_count_spinbox = self.build_spinbox_row(controls, "Battery positions", self.position_count_text, 99)
        ttk.Label(controls, text="Tap a number to open the keypad.", style="PanelText.TLabel").pack(anchor="w", pady=(5,0))
        self.build_thermal_controls(thermal)

    def build_thermal_controls(self, parent):
        ttk.Label(parent, text="Thermal display (°C)", style="PanelTitle.TLabel").pack(anchor="w", pady=(0,5))
        row = ttk.Frame(parent, style="Panel.TFrame")
        row.pack(fill="x")
        for title, variable, attr in (("Minimum", self.thermal_min_text, "thermal_min_entry"),
                                      ("Maximum", self.thermal_max_text, "thermal_max_entry")):
            ttk.Label(row, text=title, style="PanelText.TLabel").pack(anchor="w")
            control = NumberControl(row, variable, title + " temperature", minimum=0, maximum=60, decimal=True)
            control.pack(fill="x", pady=(0,4))
            setattr(self, attr, control)
        modes = ttk.Frame(parent, style="Panel.TFrame")
        modes.pack(fill="x", pady=(2,0))
        self.thermal_range_button = ttk.Button(modes, text="Apply", style="Primary.TButton", width=5,
                                               command=self.configure_thermal_temperature_range)
        self.thermal_range_button.pack(side="left", padx=(0,6))
        for label, value in (("Color", "rgb"), ("Grey", "grayscale")):
            button = ttk.Radiobutton(modes, text=label, value=value, variable=self.thermal_scale_mode,
                                      command=self.change_thermal_scale_mode, style="ThermalScale.TRadiobutton")
            button.pack(side="left")
            self.thermal_scale_buttons.append(button)

    def build_system_panel(self, parent):
        info = ttk.Frame(parent, style="Panel.TFrame", padding=12)
        info.pack(fill="both", expand=True)
        for attr in ("radar_info_label", "thermal_info_label", "microphone_info_label", "speaker_info_label", "output_info_label", "rate_info_label"):
            label = ttk.Label(info, style="PanelText.TLabel", wraplength=720)
            label.pack(anchor="w", pady=(0,6))
            setattr(self, attr, label)

    @staticmethod
    def build_spinbox_row(parent, label, variable, maximum):
        ttk.Label(parent, text=label, style="PanelText.TLabel").pack(anchor="w")
        control = NumberControl(parent, variable, label, maximum=maximum)
        control.pack(fill="x", pady=(0,4))
        return control

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
        self.speaker_info_label.config(text=f"Speaker: {self.audio.speaker_device_name}")
        self.output_info_label.config(text=f"Output: {self.captures_dir}/")
        self.rate_info_label.config(
            text=(
                "Dataset export: 1 mmWave + 1 thermal frame/chirp\n"
                f"Thermal video: {self.thermal_camera.record_fps:g} FPS"
            )
        )

    def apply_audio_sources_from_settings(self, microphone, speaker):
        if self.pulse_sequence_active:
            messagebox.showwarning(
                "Capture Active",
                "Stop tracking before changing audio devices.",
            )
            return False
        self.audio.set_microphone_device(microphone["id"], microphone["name"])
        self.audio.set_speaker_device(speaker["id"], speaker["name"])
        self.refresh_info_panel()
        self.update_status("Status: Microphone and speaker settings updated")
        return True

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
        capture_error = None
        for event in self.thermal_camera.poll_events():
            if event[0] == "error":
                capture_error = str(event[1])
                self.thermal_video_label.config(
                    image="",
                    text=f"Thermal sensor unavailable.\n\n{capture_error}",
                    fg=COLORS["warning"],
                )
                self.thermal_video_label.image = None

        if capture_error and self.pulse_sequence_active and not self.is_closing:
            self.abort_capture(capture_error)
            return

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
        self.drain_ui_callbacks()
        if self.is_closing:
            return
        try:
            while True:
                event = self.mmwave_capture.events.get_nowait()
                if event.kind == "frame":
                    self.display_mmwave_frame(event.payload)
                elif event.kind == "ready":
                    self.refresh_info_panel()
                    self.update_status(
                        "Status: mmWave radar ready; valid I/Q stream confirmed",
                        "● LIVE",
                    )
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
        if self.export_in_progress or self.is_closing:
            return
        if self.pulse_sequence_active:
            self.stop_tracking()
        elif not self.is_preparing_tracking:
            self.start_tracking()

    def start_tracking(self):
        if self.is_closing or self.export_in_progress or self.pulse_sequence_active:
            return
        if self.mmwave_capture.is_running:
            self.update_status("Status: Previous radar session is still stopping. Try again shortly.", "● WARNING")
            return
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

        if self.capture_attempted:
            try:
                self.prepare_capture_session()
            except OSError as error:
                messagebox.showerror("Capture folder", f"Cannot create a new capture folder: {error}", parent=self.root)
                return
        self.capture_attempted = True
        self.refresh_info_panel()
        self.tracking_start_token += 1
        self.requested_pulse_count = pulse_count
        self.requested_position_count = position_count
        self.pulses_per_position = pulse_count // position_count
        self.current_position_number = 1
        self.current_pulse_number = 0
        self.pulse_recordings = []
        self.voice_recording_count = 0
        self.position_segments = []
        self.thermal_frame_records = []
        capture_identifier = datetime.now().astimezone().strftime(
            "%Y%m%d_%H%M%S_%f"
        )
        self.thermal_frames_staging_directory = (
            Path(self.captures_dir)
            / f".Thermal_Frames_{capture_identifier}"
        )
        self.pulse_sequence_started_at = None
        self.thermal_segment_started_at = None
        self.export_started = False
        self.pulse_sequence_active = True
        self.is_preparing_tracking = True
        self.awaiting_radar_ready = True

        self.mmwave_capture = self.new_mmwave_capture()
        try:
            self.start_thermal_feed()
            self.mmwave_capture.start()
        except (OSError, RuntimeError) as error:
            self.abort_capture(f"Sensors could not start: {error}")
            return
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
        if self.audio.speaker_device_id is not None:
            command.extend(["--output-device", str(self.audio.speaker_device_id)])
        return command, pulse_folder

    def get_pulse_recording_template(self):
        timestamp = (
            self.current_recording_timestamp
            or datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
        )
        return os.path.join(
            self.voices_dir,
            f"CoBas_V1_PulseVoice_{timestamp}_"
            f"Position_{self.current_position_number}_{{pulse:03d}}.wav",
        )

    def start_pulse_sequence(self, start_token):
        if start_token != self.tracking_start_token or not self.pulse_sequence_active:
            return
        position_number = self.current_position_number
        pulse_count = self.pulses_per_position
        global_offset = (position_number - 1) * pulse_count
        self.current_recording_timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
        # Snapshot the command before a worker starts, so retries cannot change its paths.
        command, pulse_folder = self.get_pulse_command()
        if command is None:
            self.abort_capture("Pulse generator was not found")
            return

        def worker():
            process = None
            completed = set()
            ready_received = False
            audio_error = None
            start_result = {"started": False}
            start_result_event = threading.Event()

            def publish(callback, *args):
                self.post_ui(callback, *args, token=start_token)

            def start_recording():
                try:
                    start_result["started"] = self.handle_sequence_ready(start_token, position_number)
                finally:
                    start_result_event.set()

            try:
                process = subprocess.Popen(command, cwd=pulse_folder, stdin=subprocess.PIPE,
                                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           text=True, bufsize=1)
                with self.pulse_process_lock:
                    if start_token != self.tracking_start_token or self.is_closing:
                        return
                    self.pulse_process = process
                for output_line in process.stdout:
                    if start_token != self.tracking_start_token or self.is_closing:
                        return
                    line = output_line.strip()
                    if line:
                        print(f"[CHIRP SEQUENCE] {line}")
                    if line.startswith("AUDIO_ERROR "):
                        audio_error = line[len("AUDIO_ERROR "):]
                    if line.startswith("SEQUENCE_READY"):
                        if ready_received:
                            raise RuntimeError("Pulse generator reported readiness twice")
                        ready_received = True
                        publish(start_recording)
                        while not start_result_event.wait(timeout=0.1):
                            if start_token != self.tracking_start_token or self.is_closing:
                                return
                        if not start_result["started"]:
                            return
                        process.stdin.write("START\n")
                        process.stdin.flush()
                    elif line.startswith("PLAYBACK_STARTED"):
                        parts = line.split(maxsplit=2)
                        if len(parts) != 3:
                            continue
                        position_pulse, started_at = int(parts[1]), float(parts[2])
                        publish(self.handle_pulse_started, start_token, position_number,
                                position_pulse, global_offset + position_pulse, started_at)
                    elif line.startswith("PULSE_FINISHED"):
                        parts = line.split(maxsplit=2)
                        if len(parts) != 3:
                            continue
                        position_pulse, path = int(parts[1]), parts[2]
                        if position_pulse in completed:
                            continue
                        if not 1 <= position_pulse <= pulse_count or not os.path.isfile(path):
                            raise RuntimeError("Pulse generator reported an invalid recording")
                        completed.add(position_pulse)
                        publish(self.register_pulse_recording, start_token, position_number,
                                position_pulse, global_offset + position_pulse, path)
                return_code = process.wait()
                if return_code or not ready_received or len(completed) != pulse_count:
                    publish(self.abort_capture, audio_error or "Chirp playback or microphone recording failed")
                else:
                    publish(self.handle_position_finished, start_token, position_number)
            except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
                publish(self.abort_capture, f"Pulse sequence failed: {error}")
            finally:
                if process is not None:
                    if process.poll() is None:
                        try:
                            process.terminate()
                            process.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                        except ProcessLookupError:
                            pass
                    for pipe in (process.stdin, process.stdout):
                        if pipe is not None:
                            try:
                                pipe.close()
                            except OSError:
                                pass
                    with self.pulse_process_lock:
                        if self.pulse_process is process:
                            self.pulse_process = None

        threading.Thread(target=worker, daemon=True).start()

    def register_pulse_recording(self, start_token, position, position_pulse, global_pulse, path):
        if start_token != self.tracking_start_token or not self.pulse_sequence_active:
            return
        self.pulse_recordings.append({"path": path, "pulse_number": global_pulse,
                                      "position": position, "position_pulse_number": position_pulse})
        self.handle_pulse_completed(start_token, position, position_pulse, global_pulse)

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
        playback_started_at=None,
    ):
        if start_token != self.tracking_start_token:
            return
        if position_pulse_number == 1 and playback_started_at is not None:
            self.pulse_sequence_started_at = playback_started_at
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
        position_pulse_number,
        global_pulse_number,
    ):
        if start_token != self.tracking_start_token:
            return
        try:
            self.capture_chirp_sensor_frames(
                position_number,
                position_pulse_number,
                global_pulse_number,
            )
        except (OSError, RuntimeError, ValueError) as error:
            self.abort_capture(
                f"Sensor frames could not be saved for chirp "
                f"{global_pulse_number}: {error}"
            )
            return
        self.record_timer_label.config(
            text=(
                f"Position {position_number}/{self.requested_position_count} • "
                f"Chirps {global_pulse_number}/{self.requested_pulse_count}"
            )
        )

    def capture_chirp_sensor_frames(
        self,
        position_number,
        position_pulse_number,
        global_pulse_number,
    ):
        """Save one thermal image and request one mmWave image per chirp."""
        if self.thermal_frames_staging_directory is None:
            raise RuntimeError("Thermal chirp-frame output was not initialized")

        thermal_path = self.thermal_frames_staging_directory / (
            f"Thermal_Image_{self.battery_percentage}_Percent_"
            f"Position_{position_number}_{position_pulse_number:03d}.png"
        )
        saved_path = self.thermal_camera.save_latest_frame(thermal_path)
        try:
            self.mmwave_capture.request_chirp_frame(global_pulse_number)
        except (RuntimeError, ValueError):
            Path(saved_path).unlink(missing_ok=True)
            raise

        self.thermal_frame_records.append(
            {
                "path": saved_path,
                "pulse_number": global_pulse_number,
                "position": position_number,
                "position_pulse_number": position_pulse_number,
            }
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

        position_recordings = [
            item
            for item in self.pulse_recordings
            if item.get("position") == position_number
        ]
        thermal_frame_records = sorted(
            (
                item
                for item in self.thermal_frame_records
                if item.get("position") == position_number
            ),
            key=lambda item: item["pulse_number"],
        )
        expected_pulse_numbers = [
            item["pulse_number"]
            for item in sorted(
                position_recordings,
                key=lambda item: item["pulse_number"],
            )
        ]
        thermal_pulse_numbers = [
            item["pulse_number"] for item in thermal_frame_records
        ]
        mmwave_frames_complete = not expected_pulse_numbers
        if expected_pulse_numbers:
            mmwave_frames_complete = self.mmwave_capture.wait_for_frame_count(
                expected_pulse_numbers[-1],
                timeout=SENSOR_FRAME_CAPTURE_TIMEOUT_SECONDS,
            )

        self.mmwave_capture.pause()
        target_duration = len(position_recordings) * PULSE_DURATION_SECONDS
        thermal_path = None
        if self.thermal_camera.is_recording:
            thermal_path = self.thermal_camera.stop_recording()

        if (
            not position_recordings
            or not thermal_path
            or not mmwave_frames_complete
            or thermal_pulse_numbers != expected_pulse_numbers
        ):
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
            "thermal_frame_records": thermal_frame_records,
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
            f"Move the battery to Position {next_position}, then tap OK.\n\n"
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
        self.tracking_start_token += 1
        self.stop_pulse_process()
        self.finalize_position_segment(self.current_position_number)
        self.finish_capture(message, warning=True)

    def finish_capture(self, message, warning=False):
        self.tracking_start_token += 1
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
            self.export_in_progress = True
            self.track_button.configure(text="Saving…", state="disabled")
            threading.Thread(
                target=self.export_battery_capture,
                daemon=True,
            ).start()
        elif not self.position_segments:
            self.remove_thermal_frame_staging_directory()

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
            if self.mmwave_capture.is_running:
                raise RuntimeError("Radar is still saving; capture files have been preserved")
            self.voice_recording_count = len(
                self.validate_chirp_voice_recordings()
            )
            if self.mmwave_capture.frame_count != self.voice_recording_count:
                raise RuntimeError(
                    f"Expected {self.voice_recording_count} mmWave frames, "
                    f"but saved {self.mmwave_capture.frame_count}"
                )
            self.finalize_thermal_frames(self.voice_recording_count)

            thermal_video = os.path.join(
                self.thermal_references_dir,
                "Thermal_Video.mp4",
            )
            self.concatenate_thermal_videos(self.position_segments, thermal_video)
            self.combine_thermal_text_outputs(
                "temperature_log_path",
                os.path.join(
                    self.thermal_references_dir,
                    "Thermal_Temperature_Log.txt",
                ),
            )
            self.combine_thermal_text_outputs(
                "temperature_average_path",
                os.path.join(
                    self.thermal_references_dir,
                    "Thermal_Temperature_Averages.txt",
                ),
            )
            self.clean_position_temporary_outputs()
            self.post_ui(self.handle_export_finished)
        except Exception as error:  # Worker boundary: always unlock and report failed exports.
            print(f"[ERROR] Battery output generation failed: {error}")
            self.post_ui(self.handle_export_failed, str(error))

    def handle_export_failed(self, error):
        self.export_in_progress = False
        self.track_button.configure(text="Start Tracking", style="Start.TButton", state="normal")
        self.update_status(f"Status: Saving failed: {error}. Files retained in {self.captures_dir}", "● ERROR")

    def finalize_thermal_frames(self, expected_count):
        """Publish the chirp-aligned thermal images after validating the set."""
        records = sorted(
            self.thermal_frame_records,
            key=lambda item: item["pulse_number"],
        )
        pulse_numbers = [item["pulse_number"] for item in records]
        if pulse_numbers != list(range(1, expected_count + 1)):
            raise RuntimeError(
                "Thermal frames are not a complete chirp-by-chirp sequence: "
                f"expected chirps 1-{expected_count}, received {pulse_numbers}"
            )

        staging_directory = self.thermal_frames_staging_directory
        if staging_directory is None or not staging_directory.is_dir():
            raise RuntimeError("Thermal chirp-frame staging directory is missing")

        recorded_paths = [Path(item["path"]) for item in records]
        if len(set(recorded_paths)) != expected_count:
            raise RuntimeError("Each chirp must have one unique thermal frame")
        if any(not path.is_file() for path in recorded_paths):
            raise RuntimeError("One or more thermal chirp frames are missing")

        staged_paths = sorted(staging_directory.glob("*.png"))
        if len(staged_paths) != expected_count or set(staged_paths) != set(
            recorded_paths
        ):
            raise RuntimeError(
                f"Expected {expected_count} thermal frames, "
                f"but found {len(staged_paths)}"
            )

        output_directory = Path(self.thermal_frames_dir)
        if output_directory.is_dir():
            if any(output_directory.iterdir()):
                raise RuntimeError(f"Thermal frame output already contains data: {output_directory}")
            output_directory.rmdir()
        elif output_directory.exists():
            raise RuntimeError(
                f"Thermal frame output is not a directory: {output_directory}"
            )
        staging_directory.replace(output_directory)

    def remove_thermal_frame_staging_directory(self):
        staging_directory = self.thermal_frames_staging_directory
        if staging_directory is not None and staging_directory.is_dir():
            shutil.rmtree(staging_directory)

    @staticmethod
    def concatenate_thermal_videos(segments, output_path):
        inputs = [
            item["thermal_video_path"]
            for item in sorted(segments, key=lambda value: value["position"])
            if os.path.exists(item["thermal_video_path"])
        ]
        if not inputs or len(inputs) != len(segments):
            raise RuntimeError("One or more thermal video segments are missing")
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

    def validate_chirp_voice_recordings(self):
        """Require one complete, unique WAV file for every finished chirp."""
        recordings = sorted(
            self.pulse_recordings,
            key=lambda value: value["pulse_number"],
        )
        if not recordings:
            raise RuntimeError("No completed voice recordings were available")

        pulse_numbers = [item["pulse_number"] for item in recordings]
        expected_numbers = list(range(1, len(recordings) + 1))
        if pulse_numbers != expected_numbers:
            raise RuntimeError(
                "Voice recordings are not a complete chirp-by-chirp sequence: "
                f"expected chirps 1-{len(recordings)}, received {pulse_numbers}"
            )

        recording_paths = [
            os.path.abspath(item["path"])
            for item in recordings
        ]
        if len(set(recording_paths)) != len(recording_paths):
            raise RuntimeError(
                "Each chirp must have its own unique voice recording file"
            )

        expected_format = None
        for recording in recordings:
            recording_path = os.path.abspath(recording["path"])
            if not os.path.isfile(recording_path):
                raise RuntimeError(
                    f"Voice recording is missing for chirp "
                    f"{recording['pulse_number']}: {recording_path}"
                )
            with wave.open(recording_path, "rb") as input_wav:
                current_format = (
                    input_wav.getnchannels(),
                    input_wav.getsampwidth(),
                    input_wav.getframerate(),
                )
                expected_frames = round(48_000 * PULSE_DURATION_SECONDS)
                if current_format != (1, 2, 48_000) or input_wav.getnframes() != expected_frames:
                    raise RuntimeError(f"Chirp {recording['pulse_number']} must contain exactly two seconds of 48 kHz mono 16-bit audio: {recording_path}")
                if len(input_wav.readframes(expected_frames)) != expected_frames * 2:
                    raise RuntimeError(f"Chirp {recording['pulse_number']} audio data is truncated: {recording_path}")
                if expected_format is None:
                    expected_format = current_format
                elif current_format != expected_format:
                    raise RuntimeError(
                        f"Incompatible chirp recording: {recording_path}"
                    )

        return recording_paths

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
        final_video = os.path.join(
            self.thermal_references_dir,
            "Thermal_Video.mp4",
        )
        protected = {os.path.abspath(final_video)}
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
        self.export_in_progress = False
        self.track_button.configure(text="Start Tracking", style="Start.TButton", state="normal")
        self.mmwave_label.config(
            image="",
            text="Battery capture saved.\n\nTap 'Start Tracking' to record again.",
        )
        self.mmwave_label.image = None
        self.thermal_video_label.config(
            image="",
            text="Thermal capture saved.\n\nTap 'Start Tracking' to record again.",
        )
        self.thermal_video_label.image = None
        self.update_status(
            f"Status: Battery outputs saved in {self.captures_dir}",
            "● IDLE",
        )
        self.prompt_for_another_dataset()

    def prompt_for_another_dataset(self):
        """Relaunch for another battery or close after a completed export."""
        collect_another = messagebox.askyesno(
            "Battery Dataset Complete",
            "Would you like to collect another battery dataset?\n\n"
            f"The completed dataset was saved in:\n{self.captures_dir}",
            parent=self.root,
        )
        if collect_another:
            self.restart_application()
        else:
            self.on_close()

    def restart_application(self):
        """Close all current resources and replace this process with a fresh app."""
        script_path = os.path.abspath(__file__)
        command_arguments = [
            sys.executable,
            script_path,
            *sys.argv[1:],
        ]
        self.on_close()
        os.execv(sys.executable, command_arguments)

    def on_close(self):
        if self.is_closing or self.export_in_progress:
            return
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
    root.protocol("WM_DELETE_WINDOW", app.request_close)
    root.mainloop()
