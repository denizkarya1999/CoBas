import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from PIL import Image, ImageTk
import os
import cv2
import sys
import threading
import subprocess
import shutil
import time
import wave
from datetime import datetime

from Camera.Camera import Camera
from Camera.thermal_camera import ThermalCamera
from Style import COLORS, FONTS, WINDOW, PREVIEW, SPACING, apply_styles
from Settings import SettingsWindow
from About import show_about_window


APP_TITLE = "CoBas Battery Reader V1.0"
APP_WM_CLASS = "cobas_battery_reader_v1"
ICON_SIZES = (16, 24, 32, 48, 64, 128, 256)
PULSE_DURATION_SECONDS = 2.0
DATASET_FRAME_INTERVAL_SECONDS = 0.5
DATASET_FRAMES_PER_SECOND = 1.0 / DATASET_FRAME_INTERVAL_SECONDS
DATASET_FRAMES_PER_PULSE = int(
    PULSE_DURATION_SECONDS / DATASET_FRAME_INTERVAL_SECONDS
)


def parse_battery_percentage(value):
    """Return a validated whole-number battery percentage from user input."""
    text = str(value).strip()
    if text.endswith("%"):
        text = text[:-1].strip()

    if not text.isdigit():
        raise ValueError(
            "Enter a whole-number battery percentage from 0 to 100."
        )

    percentage = int(text)
    if not 0 <= percentage <= 100:
        raise ValueError(
            "Enter a whole-number battery percentage from 0 to 100."
        )
    return percentage


def battery_output_folder_name(percentage):
    """Return the canonical folder name for one battery-level session."""
    return f"{parse_battery_percentage(percentage)}_Percent_Battery"


def battery_output_directory(base_dir, percentage):
    """Return the capture root that contains every session output."""
    return os.path.join(
        base_dir,
        "Captures",
        battery_output_folder_name(percentage),
    )


class CoBasV1App:
    """
    Main GUI application for CoBas Battery Reader V1.0.

    This file handles:
    - Main application window
    - Toolbar
    - Camera preview
    - Start/Stop tracking
    - Restart camera
    - Front/back camera switching
    - Photo capture
    - Video/audio recording
    - Zoom controls
    - Status display
    - Dynamic status icon color
    """

    def __init__(self, root):
        """
        Initialize the main GUI application.
        """

        self.root = root
        self.root.title(APP_TITLE)
        self.root.iconname(APP_TITLE)
        self.root.withdraw()

        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.battery_percentage = self.request_battery_percentage()
        self.battery_output_name = battery_output_folder_name(
            self.battery_percentage
        )
        self.captures_dir = battery_output_directory(
            self.base_dir,
            self.battery_percentage,
        )

        # Load application icon before building the UI.
        self.set_app_icon()

        # Use fixed window size from Style.py.
        self.window_width = WINDOW["width"]
        self.window_height = WINDOW["height"]

        self.root.geometry(f"{self.window_width}x{self.window_height}")
        self.root.resizable(False, False)
        self.root.minsize(self.window_width, self.window_height)
        self.root.maxsize(self.window_width, self.window_height)

        # Used to prevent app from staying minimized.
        self.is_closing = False
        self.root.bind("<Unmap>", self.prevent_minimize)

        # Camera backends.
        self.camera = Camera(
            camera_index=Camera.default_camera_source(),
            output_dir=self.captures_dir
        )
        self.thermal_camera = ThermalCamera(output_dir=self.captures_dir)

        # Stores path of the current video file.
        self.current_video_path = None
        self.current_thermal_video_path = None
        self.average_fps_log_path = None

        # Stores last processed video path to avoid processing the same video twice.
        self.last_processed_video_path = None

        # Repeated two-second pulse sequence state.
        self.pulse_process = None
        self.pulse_process_lock = threading.Lock()
        self.is_preparing_tracking = False
        self.tracking_start_token = 0
        self.pulse_playback_end_time = None
        self.pulse_count_text = tk.StringVar(master=self.root, value="20")
        self.pulse_count_spinbox = None
        self.requested_pulse_count = 1
        self.current_pulse_number = 0
        self.pulse_sequence_active = False
        self.pulse_recordings = []
        self.pulse_sequence_started_at = None
        self.current_recording_timestamp = None

        # Preview loop state.
        self.preview_loop_running = False
        self.preview_after_id = None
        self.thermal_preview_after_id = None

        # Stores Tkinter image reference for preview.
        self.preview_photo = None
        self.thermal_preview_photo = None

        # Controls the palette used by the thermal preview and recorder.
        self.thermal_scale_mode = tk.StringVar(
            master=self.root,
            value=self.thermal_camera.display_mode
        )
        self.thermal_scale_buttons = []
        thermal_min, thermal_max = self.thermal_camera.get_temperature_range()
        self.thermal_min_text = tk.StringVar(
            master=self.root,
            value=f"{thermal_min:g}"
        )
        self.thermal_max_text = tk.StringVar(
            master=self.root,
            value=f"{thermal_max:g}"
        )
        self.thermal_min_entry = None
        self.thermal_max_entry = None
        self.thermal_range_button = None

        # Apply styles from Style.py.
        self.style = apply_styles(self.root)

        # Build all GUI components.
        self.build_gui()

        self.update_status(
            "Status: Ready. Click 'Start Tracking' to begin.",
            "● READY"
        )
        self.root.deiconify()

    def request_battery_percentage(self):
        """Ask for the required battery level before creating output paths."""
        prompt = (
            "What battery percentage level are you collecting data for? "
            "(Ex. 50%, 100% etc.)"
        )

        while True:
            response = simpledialog.askstring(
                "Battery Percentage Level",
                prompt,
                parent=self.root,
            )
            if response is None:
                self.root.destroy()
                raise SystemExit(0)

            try:
                return parse_battery_percentage(response)
            except ValueError as exc:
                messagebox.showerror(
                    "Invalid Battery Percentage",
                    str(exc),
                    parent=self.root,
                )

    # --------------------------------------------------
    # Window Behavior
    # --------------------------------------------------

    def prevent_minimize(self, event):
        """
        Restore the window if minimized.

        Tkinter cannot always remove the minimize button cleanly
        on Linux, so this restores the window if it becomes minimized.
        """

        if self.is_closing:
            return

        try:
            if self.root.state() == "iconic":
                self.root.after(100, self.root.deiconify)
        except tk.TclError:
            pass

    def set_app_icon(self):
        """
        Set application window and taskbar icon.

        Expected path:
            Assets/icon.png
        """

        base_dir = os.path.dirname(os.path.abspath(__file__))

        icon_path = os.path.join(
            base_dir,
            "Assets",
            "icon.png"
        )

        if os.path.exists(icon_path):
            icon_images = []

            try:
                source_image = Image.open(icon_path).convert("RGBA")
                resample_filter = Image.Resampling.LANCZOS

                for size in ICON_SIZES:
                    resized_icon = source_image.resize(
                        (size, size),
                        resample_filter
                    )

                    icon_images.append(ImageTk.PhotoImage(resized_icon))

            except Exception as e:
                print(f"[WARNING] Could not build resized app icons: {e}")
                icon_images = [tk.PhotoImage(file=icon_path)]

            # Sets title-bar/taskbar icon where supported by the window manager.
            self.root.iconphoto(True, *icon_images)

            # Reapply after the window is mapped; some Linux window managers
            # ignore the first icon request if it happens too early.
            self.root.after(
                200,
                lambda: self.root.iconphoto(True, *icon_images)
            )

            # Keep references so Python does not garbage collect the images.
            self.root.icon_images = icon_images

            print(f"[INFO] App icon loaded: {icon_path}")
        else:
            print(f"[WARNING] Icon file not found: {icon_path}")

    # --------------------------------------------------
    # GUI Layout
    # --------------------------------------------------

    def build_gui(self):
        """
        Build the main dashboard layout.

        The UI contains:
        - Toolbar at the top
        - Camera preview on the left
        - Control panel on the right
        """

        outer_frame = ttk.Frame(self.root, style="Main.TFrame")
        outer_frame.pack(fill="both", expand=True)

        self.build_toolbar(outer_frame)

        main_frame = ttk.Frame(outer_frame, style="Main.TFrame")
        main_frame.pack(
            fill="both",
            expand=True,
            padx=SPACING["main_padx"],
            pady=SPACING["main_pady"]
        )

        content_frame = ttk.Frame(main_frame, style="Main.TFrame")
        content_frame.pack(fill="both", expand=True)

        # Left side gets more space because it contains the camera preview.
        content_frame.columnconfigure(0, weight=4)

        # Right side contains compact controls.
        content_frame.columnconfigure(1, weight=1)

        content_frame.rowconfigure(0, weight=1)

        preview_panel = ttk.Frame(content_frame, style="Panel.TFrame")
        preview_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        control_panel = ttk.Frame(content_frame, style="Panel.TFrame")
        control_panel.grid(row=0, column=1, sticky="nsew")

        self.build_preview_panel(preview_panel)
        self.build_control_panel(control_panel)

    def build_toolbar(self, parent):
        """
        Build the top toolbar.

        The toolbar only includes Settings and About.
        """

        toolbar = ttk.Frame(parent, style="Toolbar.TFrame")
        toolbar.pack(fill="x")

        left_toolbar = ttk.Frame(toolbar, style="Toolbar.TFrame")
        left_toolbar.pack(
            side="left",
            padx=SPACING["toolbar_padx"],
            pady=SPACING["toolbar_pady"]
        )

        ttk.Button(
            left_toolbar,
            text="Settings",
            style="Settings.TButton",
            command=self.open_settings
        ).pack(side="left", padx=(0, 4))

        ttk.Button(
            left_toolbar,
            text="About",
            style="Settings.TButton",
            command=self.open_about
        ).pack(side="left", padx=4)

    def build_preview_panel(self, parent):
        """
        Build the left camera preview panel.
        """

        preview_header = ttk.Frame(parent, style="Panel.TFrame")
        preview_header.pack(
            fill="x",
            padx=SPACING["panel_padx"],
            pady=(8, 4)
        )

        ttk.Label(
            preview_header,
            text="Live Cameras",
            style="PanelTitle.TLabel"
        ).pack(side="left")

        # Dynamic status icon.
        # This is tk.Label instead of ttk.Label because we change fg color.
        self.live_indicator_label = tk.Label(
            preview_header,
            text="● READY",
            bg=COLORS["panel_bg"],
            fg=COLORS["accent"],
            font=FONTS["status"]
        )
        self.live_indicator_label.pack(side="right")

        self.preview_area = ttk.Frame(parent, style="Panel.TFrame")
        self.preview_area.pack(
            fill="both",
            expand=True,
            padx=SPACING["panel_padx"],
            pady=(0, 6)
        )
        self.preview_area.columnconfigure(0, weight=1, uniform="camera_preview")
        self.preview_area.columnconfigure(1, weight=1, uniform="camera_preview")
        self.preview_area.rowconfigure(0, weight=1)

        regular_preview_frame = ttk.Frame(
            self.preview_area,
            style="Panel.TFrame"
        )
        thermal_preview_frame = ttk.Frame(
            self.preview_area,
            style="Panel.TFrame"
        )

        regular_preview_frame.grid(
            row=0,
            column=0,
            sticky="nsew",
            padx=(0, 4)
        )
        thermal_preview_frame.grid(
            row=0,
            column=1,
            sticky="nsew",
            padx=(4, 0)
        )
        regular_preview_frame.rowconfigure(1, weight=1)
        regular_preview_frame.columnconfigure(0, weight=1)
        thermal_preview_frame.rowconfigure(1, weight=1)
        thermal_preview_frame.columnconfigure(0, weight=1)

        ttk.Label(
            regular_preview_frame,
            text="Regular Camera",
            style="PanelText.TLabel"
        ).grid(row=0, column=0, sticky="w", pady=(0, 3))

        thermal_header = ttk.Frame(
            thermal_preview_frame,
            style="Panel.TFrame"
        )
        thermal_header.grid(row=0, column=0, sticky="ew", pady=(0, 3))

        ttk.Label(
            thermal_header,
            text="Thermal Camera",
            style="PanelText.TLabel"
        ).pack(side="left")

        thermal_scale_controls = ttk.Frame(
            thermal_header,
            style="Panel.TFrame"
        )
        thermal_scale_controls.pack(side="right")

        ttk.Label(
            thermal_scale_controls,
            text="Min",
            style="PanelText.TLabel"
        ).pack(side="left", padx=(0, 2))

        self.thermal_min_entry = ttk.Entry(
            thermal_scale_controls,
            textvariable=self.thermal_min_text,
            width=4,
            justify="center"
        )
        self.thermal_min_entry.pack(side="left", padx=(0, 3))
        self.thermal_min_entry.bind(
            "<Return>",
            self.configure_thermal_temperature_range
        )

        ttk.Label(
            thermal_scale_controls,
            text="Max",
            style="PanelText.TLabel"
        ).pack(side="left", padx=(0, 2))

        self.thermal_max_entry = ttk.Entry(
            thermal_scale_controls,
            textvariable=self.thermal_max_text,
            width=4,
            justify="center"
        )
        self.thermal_max_entry.pack(side="left", padx=(0, 3))
        self.thermal_max_entry.bind(
            "<Return>",
            self.configure_thermal_temperature_range
        )

        ttk.Label(
            thermal_scale_controls,
            text="°C",
            style="PanelText.TLabel"
        ).pack(side="left", padx=(0, 1))

        self.thermal_range_button = ttk.Button(
            thermal_scale_controls,
            text="Set",
            width=3,
            command=self.configure_thermal_temperature_range
        )
        self.thermal_range_button.pack(side="left", padx=(0, 2))

        for label, value in (("Regular", "rgb"), ("Greyscale", "grayscale")):
            button = ttk.Radiobutton(
                thermal_scale_controls,
                text=label,
                value=value,
                variable=self.thermal_scale_mode,
                command=self.change_thermal_scale_mode,
                style="ThermalScale.TRadiobutton"
            )
            button.pack(side="left", padx=(2, 0))
            self.thermal_scale_buttons.append(button)

        self.video_label = tk.Label(
            regular_preview_frame,
            text="Camera is ready.\n\nClick 'Start Tracking' to begin.",
            bg=COLORS["preview_bg"],
            fg=COLORS["muted_text"],
            font=FONTS["preview_text"],
            bd=0,
            relief="flat"
        )

        self.video_label.grid(
            row=1,
            column=0,
            sticky="nsew"
        )

        self.thermal_video_label = tk.Label(
            thermal_preview_frame,
            text="Thermal camera is ready.\n\nClick 'Start Tracking' to begin.",
            bg=COLORS["preview_bg"],
            fg=COLORS["muted_text"],
            font=FONTS["preview_text"],
            bd=0,
            relief="flat"
        )

        self.thermal_video_label.grid(
            row=1,
            column=0,
            sticky="nsew"
        )

        bottom_bar = ttk.Frame(parent, style="Panel.TFrame")
        bottom_bar.pack(
            fill="x",
            padx=SPACING["panel_padx"],
            pady=(0, 8)
        )

        self.status_label = ttk.Label(
            bottom_bar,
            text="Status: Ready. Click 'Start Tracking' to begin.",
            style="Info.TLabel"
        )
        self.status_label.pack(side="left")

        self.processing_indicator_frame = ttk.Frame(
            bottom_bar,
            style="Panel.TFrame"
        )

        self.processing_indicator_label = ttk.Label(
            self.processing_indicator_frame,
            text="Processing...",
            style="Info.TLabel"
        )
        self.processing_indicator_label.pack(side="left", padx=(12, 6))

        self.processing_progress_bar = ttk.Progressbar(
            self.processing_indicator_frame,
            mode="indeterminate",
            length=130
        )
        self.processing_progress_bar.pack(side="left")

        self.record_timer_label = ttk.Label(
            bottom_bar,
            text="Pulses: 0/20",
            style="Info.TLabel"
        )
        self.record_timer_label.pack(side="right")

    def build_control_panel(self, parent):
        """
        Build the right control panel.
        """

        # --------------------------------------------------
        # Tracking controls
        # --------------------------------------------------

        controls_section = ttk.Frame(parent, style="Panel.TFrame")
        controls_section.pack(
            fill="x",
            padx=SPACING["panel_padx"],
            pady=(8, 4)
        )

        ttk.Label(
            controls_section,
            text="Tracking",
            style="PanelTitle.TLabel"
        ).pack(anchor="w", pady=(0, 3))

        pulse_count_row = ttk.Frame(
            controls_section,
            style="Panel.TFrame"
        )
        pulse_count_row.pack(fill="x", pady=(0, 4))

        ttk.Label(
            pulse_count_row,
            text="Pulse count",
            style="PanelText.TLabel"
        ).pack(side="left")

        self.pulse_count_spinbox = ttk.Spinbox(
            pulse_count_row,
            from_=1,
            to=999,
            increment=1,
            textvariable=self.pulse_count_text,
            width=5,
            justify="center"
        )
        self.pulse_count_spinbox.pack(side="right")

        # Start Tracking is green.
        # When camera is active, this button changes to red Stop Tracking.
        self.track_button = ttk.Button(
            controls_section,
            text="Start Tracking",
            style="Start.TButton",
            command=self.toggle_tracking
        )
        self.track_button.pack(
            fill="x",
            pady=SPACING["button_pady"]
        )

        # Restart Camera uses the normal dark/blue tool color.
        self.restart_button = ttk.Button(
            controls_section,
            text="Restart Camera",
            style="Restart.TButton",
            command=self.restart_camera
        )
        self.restart_button.pack(
            fill="x",
            pady=SPACING["button_pady"]
        )

        # --------------------------------------------------
        # Camera switch controls
        # --------------------------------------------------

        switch_section = ttk.Frame(parent, style="Panel.TFrame")
        switch_section.pack(
            fill="x",
            padx=SPACING["panel_padx"],
            pady=SPACING["panel_pady"]
        )

        ttk.Label(
            switch_section,
            text="Camera",
            style="PanelTitle.TLabel"
        ).pack(anchor="w", pady=(0, 3))

        # Switch Camera is cyan.
        self.switch_camera_button = ttk.Button(
            switch_section,
            text="Switch Camera",
            style="Camera.TButton",
            command=self.switch_front_back_camera
        )
        self.switch_camera_button.pack(
            fill="x",
            pady=SPACING["button_pady"]
        )

        # --------------------------------------------------
        # Capture controls
        # --------------------------------------------------

        capture_section = ttk.Frame(parent, style="Panel.TFrame")
        capture_section.pack(
            fill="x",
            padx=SPACING["panel_padx"],
            pady=SPACING["panel_pady"]
        )

        ttk.Label(
            capture_section,
            text="Capture",
            style="PanelTitle.TLabel"
        ).pack(anchor="w", pady=(0, 3))

        # Take Photo is green.
        self.photo_button = ttk.Button(
            capture_section,
            text="Take Photo",
            style="Capture.TButton",
            command=self.take_photo
        )
        self.photo_button.pack(
            fill="x",
            pady=SPACING["button_pady"]
        )

        # Capture Video is green.
        # When recording starts, this changes to red Stop Recording.
        self.record_button = ttk.Button(
            capture_section,
            text="Capture Video",
            style="Capture.TButton",
            command=self.toggle_recording
        )
        self.record_button.pack(
            fill="x",
            pady=SPACING["button_pady"]
        )

        # --------------------------------------------------
        # Zoom controls
        # --------------------------------------------------

        zoom_section = ttk.Frame(parent, style="Panel.TFrame")
        zoom_section.pack(
            fill="x",
            padx=SPACING["panel_padx"],
            pady=SPACING["panel_pady"]
        )

        ttk.Label(
            zoom_section,
            text="Zoom",
            style="PanelTitle.TLabel"
        ).pack(anchor="w", pady=(0, 3))

        zoom_buttons_frame = ttk.Frame(zoom_section, style="Panel.TFrame")
        zoom_buttons_frame.pack(fill="x")

        ttk.Button(
            zoom_buttons_frame,
            text="-",
            style="Zoom.TButton",
            command=self.zoom_out
        ).pack(side="left", fill="x", expand=True, padx=(0, 3))

        ttk.Button(
            zoom_buttons_frame,
            text="+",
            style="Zoom.TButton",
            command=self.zoom_in
        ).pack(side="left", fill="x", expand=True, padx=(3, 0))

        ttk.Button(
            zoom_section,
            text="Reset",
            style="Zoom.TButton",
            command=self.reset_zoom
        ).pack(fill="x", pady=(4, 0))

        self.zoom_label = ttk.Label(
            zoom_section,
            text="Zoom: 1.0x",
            style="PanelText.TLabel"
        )
        self.zoom_label.pack(anchor="center", pady=(4, 0))

        # --------------------------------------------------
        # System info
        # --------------------------------------------------

        info_section = ttk.Frame(parent, style="Panel.TFrame")
        info_section.pack(
            fill="both",
            expand=True,
            padx=SPACING["panel_padx"],
            pady=(4, 8)
        )

        ttk.Label(
            info_section,
            text="System",
            style="PanelTitle.TLabel"
        ).pack(anchor="w", pady=(0, 3))

        self.camera_info_label = ttk.Label(
            info_section,
            text="Camera: /dev/video0",
            style="PanelText.TLabel",
            wraplength=180
        )
        self.camera_info_label.pack(anchor="w", pady=0)

        self.thermal_info_label = ttk.Label(
            info_section,
            text="Thermal: idle",
            style="PanelText.TLabel",
            wraplength=180
        )
        self.thermal_info_label.pack(anchor="w", pady=0)

        self.microphone_info_label = ttk.Label(
            info_section,
            text="Mic: Default",
            style="PanelText.TLabel",
            wraplength=180
        )
        self.microphone_info_label.pack(anchor="w", pady=0)

        self.output_info_label = ttk.Label(
            info_section,
            text="Output: captures/",
            style="PanelText.TLabel",
            wraplength=180
        )
        self.output_info_label.pack(anchor="w", pady=0)

        self.fps_info_label = ttk.Label(
            info_section,
            text=(
                f"Regular Camera FPS: {self.camera.record_fps:g}\n"
                f"Thermal Camera FPS: {self.thermal_camera.record_fps:g}"
            ),
            style="PanelText.TLabel",
            wraplength=180
        )
        self.fps_info_label.pack(anchor="w", pady=0)

        self.camera_direction_label = ttk.Label(
            info_section,
            text="Switch: /dev/video0 ↔ /dev/video1",
            style="PanelText.TLabel",
            wraplength=180
        )
        self.camera_direction_label.pack(anchor="w", pady=0)

    # --------------------------------------------------
    # Helper Methods
    # --------------------------------------------------

    def get_preview_dimensions(self, label):
        """
        Return the current display size for one side of the split preview.
        """

        width = label.winfo_width()
        height = label.winfo_height()

        if width <= 1:
            width = max(1, (PREVIEW["width"] // 2) - 8)

        if height <= 1:
            height = PREVIEW["height"]

        return width, height

    def cancel_preview_loop(self):
        """
        Cancel the scheduled camera preview loop.

        This prevents multiple root.after() loops from stacking when:
        - Start Tracking is clicked repeatedly
        - Stop Tracking is clicked
        - Restart Camera is clicked
        - Switch Camera is clicked
        """

        if self.preview_after_id is not None:
            try:
                self.root.after_cancel(self.preview_after_id)
            except tk.TclError:
                pass

            self.preview_after_id = None

        self.preview_loop_running = False

    def cancel_thermal_preview_loop(self):
        """
        Cancel the scheduled thermal preview loop.
        """

        if self.thermal_preview_after_id is not None:
            try:
                self.root.after_cancel(self.thermal_preview_after_id)
            except tk.TclError:
                pass

            self.thermal_preview_after_id = None

    def start_thermal_camera_feed(self):
        """
        Start the thermal camera worker and its lightweight preview loop.
        """

        self.thermal_camera.start_camera()

        if self.thermal_preview_after_id is None:
            self.update_thermal_feed()

    def change_thermal_scale_mode(self):
        """Switch the CoBas thermal preview between regular and greyscale."""
        requested_mode = self.thermal_scale_mode.get()
        if not self.thermal_camera.set_display_mode(requested_mode):
            self.thermal_scale_mode.set(self.thermal_camera.display_mode)

    def configure_thermal_temperature_range(self, _event=None):
        """Apply the minimum and maximum temperatures from the textboxes."""
        if self.thermal_camera.is_recording:
            messagebox.showinfo(
                "Thermal Display Range",
                "The temperature range is locked while recording.",
                parent=self.root
            )
            return

        try:
            changed = self.thermal_camera.set_temperature_range(
                self.thermal_min_text.get(),
                self.thermal_max_text.get()
            )
        except (TypeError, ValueError) as exc:
            messagebox.showerror(
                "Invalid Thermal Display Range",
                str(exc),
                parent=self.root
            )
            return

        if not changed:
            messagebox.showinfo(
                "Thermal Display Range",
                "The temperature range is locked while recording.",
                parent=self.root
            )
            return

        selected_min, selected_max = self.thermal_camera.get_temperature_range()
        self.thermal_min_text.set(f"{selected_min:g}")
        self.thermal_max_text.set(f"{selected_max:g}")
        self.update_status(
            "Status: Thermal display range updated. Ready to record.",
            "● READY"
        )

    def refresh_thermal_scale_controls(self):
        """Lock the palette and fixed range during thermal recording."""
        state = "disabled" if self.thermal_camera.is_recording else "normal"
        for button in self.thermal_scale_buttons:
            button.configure(state=state)
        if self.thermal_min_entry is not None:
            self.thermal_min_entry.configure(state=state)
        if self.thermal_max_entry is not None:
            self.thermal_max_entry.configure(state=state)
        if self.thermal_range_button is not None:
            self.thermal_range_button.configure(state=state)

        if self.thermal_scale_mode.get() != self.thermal_camera.display_mode:
            self.thermal_scale_mode.set(self.thermal_camera.display_mode)

        thermal_min, thermal_max = self.thermal_camera.get_temperature_range()
        expected_min = f"{thermal_min:g}"
        expected_max = f"{thermal_max:g}"
        if self.thermal_camera.is_recording:
            if self.thermal_min_text.get() != expected_min:
                self.thermal_min_text.set(expected_min)
            if self.thermal_max_text.get() != expected_max:
                self.thermal_max_text.set(expected_max)

    def start_active_recordings(
        self,
        record_audio=True,
        recording_timestamp=None,
    ):
        """Start the synchronized regular and thermal video recorders."""
        if self.camera.is_recording or self.thermal_camera.is_recording:
            print("[INFO] A synchronized recording is already active.")
            return self.current_video_path, self.current_thermal_video_path

        self.set_recording_volume_to_maximum()
        if recording_timestamp is None:
            recording_timestamp = datetime.now().strftime(
                "%Y%m%d_%H%M%S_%f"
            )
        self.current_recording_timestamp = recording_timestamp
        self.current_video_path = None
        self.current_thermal_video_path = None
        self.average_fps_log_path = os.path.join(
            self.captures_dir,
            f"CoBas_V1_Camera_Average_FPS_{recording_timestamp}.txt"
        )

        self.current_video_path = self.camera.start_recording(
            timestamp=recording_timestamp,
            record_audio=record_audio
        )

        if self.current_video_path:
            self.current_thermal_video_path = self.thermal_camera.start_recording(
                timestamp=recording_timestamp
            )
            self.refresh_thermal_scale_controls()

        return self.current_video_path, self.current_thermal_video_path

    def write_average_fps_log(
        self,
        regular_was_recording,
        thermal_was_recording,
    ):
        """Write measured regular and thermal camera FPS for this session."""
        if not regular_was_recording and not thermal_was_recording:
            return None

        if not self.average_fps_log_path:
            timestamp = self.current_recording_timestamp
            if not timestamp:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            self.average_fps_log_path = os.path.join(
                self.captures_dir,
                f"CoBas_V1_Camera_Average_FPS_{timestamp}.txt"
            )

        camera_entries = (
            ("Regular camera", self.camera, regular_was_recording),
            ("Thermal camera", self.thermal_camera, thermal_was_recording),
        )

        try:
            with open(
                self.average_fps_log_path,
                "w",
                encoding="utf-8",
            ) as file:
                file.write(
                    "Camera Average FPS Log\n"
                    f"Recording stopped: "
                    f"{datetime.now().isoformat(timespec='milliseconds')}\n"
                    "Average FPS is measured as recorded frames divided by "
                    "wall-clock recording duration.\n\n"
                )

                for label, camera, was_recording in camera_entries:
                    average_fps = camera.last_recording_average_fps
                    duration = camera.last_recording_duration
                    frame_count = camera.last_recording_frame_count

                    if (
                        was_recording
                        and average_fps is not None
                        and duration is not None
                    ):
                        file.write(
                            f"{label} average FPS: {average_fps:.2f}\n"
                            f"{label} frames: {frame_count}\n"
                            f"{label} duration: {duration:.2f} seconds\n\n"
                        )
                    else:
                        file.write(
                            f"{label} average FPS: unavailable\n"
                            f"{label} frames: unavailable\n"
                            f"{label} duration: unavailable\n\n"
                        )

            print(
                "[INFO] Camera average FPS log saved: "
                f"{self.average_fps_log_path}"
            )
            return self.average_fps_log_path
        except Exception as exc:
            print(f"[WARNING] Camera average FPS log could not be saved: {exc}")
            return None

    def set_recording_volume_to_maximum(self):
        """Unmute the default output device and set it to exactly 100%."""
        volume_command_sets = []

        wpctl = shutil.which("wpctl")
        pactl = shutil.which("pactl")
        amixer = shutil.which("amixer")

        if wpctl:
            volume_command_sets.append([
                [wpctl, "set-mute", "@DEFAULT_AUDIO_SINK@", "0"],
                [wpctl, "set-volume", "@DEFAULT_AUDIO_SINK@", "1.0"],
            ])
        if pactl:
            volume_command_sets.append([
                [pactl, "set-sink-mute", "@DEFAULT_SINK@", "0"],
                [pactl, "set-sink-volume", "@DEFAULT_SINK@", "100%"],
            ])
        if amixer:
            volume_command_sets.append([
                [amixer, "sset", "Master", "unmute"],
                [amixer, "sset", "Master", "100%"],
            ])

        if not volume_command_sets:
            print("[WARNING] No supported system volume control was found.")
            return False

        for volume_commands in volume_command_sets:
            try:
                for command in volume_commands:
                    subprocess.run(
                        command,
                        check=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                print("[INFO] Recording output volume set to 100%.")
                return True
            except Exception:
                continue

        print("[WARNING] Could not set recording volume to 100%.")
        return False

    def update_thermal_feed(self):
        """
        Refresh the thermal preview at a lower rate than the regular camera.
        """

        self.thermal_preview_after_id = None
        events = self.thermal_camera.poll_events()

        for event in events:
            if event[0] == "error":
                print(f"[WARNING] Thermal camera error: {event[1]}")
                self.thermal_video_label.config(
                    image="",
                    text=f"Thermal camera unavailable.\n\n{event[1]}",
                    bg=COLORS["preview_bg"],
                    fg=COLORS["warning"]
                )
                self.thermal_video_label.image = None

        preview_width, preview_height = self.get_preview_dimensions(
            self.thermal_video_label
        )
        preview_image = self.thermal_camera.get_preview_image(
            preview_width,
            preview_height
        )

        if preview_image is not None:
            self.thermal_preview_photo = ImageTk.PhotoImage(image=preview_image)
            self.thermal_video_label.config(
                image=self.thermal_preview_photo,
                text=""
            )
            self.thermal_video_label.image = self.thermal_preview_photo
        elif self.thermal_camera.error is None:
            self.thermal_video_label.config(
                image="",
                text=f"{self.thermal_camera.status}...",
                bg=COLORS["preview_bg"],
                fg=COLORS["muted_text"]
            )
            self.thermal_video_label.image = None

        self.refresh_info_panel()

        if self.thermal_camera.is_tracking:
            self.thermal_preview_after_id = self.root.after(
                150,
                self.update_thermal_feed
            )

    def stop_active_recordings(
        self,
        target_duration_seconds=None,
        audio_path_override=None,
        capture_window_start_time=None,
    ):
        """
        Stop regular and thermal recorders, reusing one synchronized WAV.

        Capture is stopped first on both recorders so video durations stay
        synchronized before FFmpeg merge/finalization begins. The WAV can be
        continuous microphone audio or the repeated-pulse timeline override.
        When a target duration is supplied, every camera video is conformed to
        that exact duration before capture outputs are exported. A capture
        window start removes recorder setup lead-in while keeping the regular
        and thermal datasets on the same pulse timeline.
        """

        saved_video_path = None
        saved_thermal_video_path = None
        saved_voice_path = None
        audio_path = None

        regular_was_recording = self.camera.is_recording
        thermal_was_recording = self.thermal_camera.is_recording
        capture_stop_time = time.time()
        regular_record_start_time = self.camera.record_start_time
        thermal_record_start_time = self.thermal_camera.record_start_time

        def wall_timing(record_start_time):
            if record_start_time is None:
                return 0.0, None

            recording_duration = max(
                0.0,
                capture_stop_time - record_start_time,
            )
            if capture_window_start_time is None:
                return 0.0, recording_duration

            start_offset = max(
                0.0,
                capture_window_start_time - record_start_time,
            )
            return start_offset, recording_duration

        regular_start_offset, regular_wall_duration = wall_timing(
            regular_record_start_time
        )
        thermal_start_offset, thermal_wall_duration = wall_timing(
            thermal_record_start_time
        )

        if regular_was_recording:
            audio_path = self.camera.stop_recording_capture_phase()

        if audio_path_override and os.path.exists(audio_path_override):
            audio_path = audio_path_override

        if thermal_was_recording:
            saved_thermal_video_path = self.thermal_camera.stop_recording(
                audio_path=audio_path
            )
            self.refresh_thermal_scale_controls()

            if saved_thermal_video_path:
                self.current_thermal_video_path = saved_thermal_video_path

        self.write_average_fps_log(
            regular_was_recording,
            thermal_was_recording,
        )

        if regular_was_recording:
            saved_video_path = self.camera.finalize_recording(
                audio_path=audio_path,
                keep_audio=True
            )

            if saved_video_path:
                self.current_video_path = saved_video_path

        if target_duration_seconds is not None:
            synchronized_video_path = self.sync_video_duration(
                saved_video_path,
                target_duration_seconds,
                tolerance_seconds=0.001,
                wall_start_offset_seconds=regular_start_offset,
                wall_recording_duration_seconds=regular_wall_duration,
                audio_start_offset_seconds=regular_start_offset,
            )
            if synchronized_video_path:
                saved_video_path = synchronized_video_path
                self.current_video_path = synchronized_video_path

            synchronized_thermal_path = self.sync_video_duration(
                saved_thermal_video_path,
                target_duration_seconds,
                tolerance_seconds=0.001,
                wall_start_offset_seconds=thermal_start_offset,
                wall_recording_duration_seconds=thermal_wall_duration,
                audio_start_offset_seconds=regular_start_offset,
            )
            if synchronized_thermal_path:
                saved_thermal_video_path = synchronized_thermal_path
                self.current_thermal_video_path = synchronized_thermal_path

            scale_video_path = self.thermal_camera.scale_video_path
            if scale_video_path:
                self.thermal_camera.scale_video_path = self.sync_video_duration(
                    scale_video_path,
                    target_duration_seconds,
                    tolerance_seconds=0.001,
                    wall_start_offset_seconds=thermal_start_offset,
                    wall_recording_duration_seconds=thermal_wall_duration,
                )

        elif saved_video_path and saved_thermal_video_path:
            thermal_duration = self.get_video_duration_seconds(
                saved_thermal_video_path
            )

            if thermal_duration is not None:
                synced_video_path = self.sync_video_duration(
                    saved_video_path,
                    thermal_duration
                )

                if synced_video_path:
                    saved_video_path = synced_video_path
                    self.current_video_path = synced_video_path

        self.camera.cleanup_temp_audio()

        return saved_video_path, saved_thermal_video_path, saved_voice_path

    def finalize_capture_session(
        self,
        target_duration_seconds=None,
        audio_path_override=None,
        capture_window_start_time=None,
    ):
        """Stop and export one capture session through the single output pipeline."""
        saved_paths = self.stop_active_recordings(
            target_duration_seconds=target_duration_seconds,
            audio_path_override=audio_path_override,
            capture_window_start_time=capture_window_start_time,
        )
        saved_video_path, saved_thermal_video_path, _ = saved_paths

        video_path = saved_video_path or self.current_video_path
        thermal_video_path = (
            saved_thermal_video_path or self.current_thermal_video_path
        )

        if video_path and os.path.exists(video_path):
            self.export_capture_session(video_path, thermal_video_path)
        else:
            print("[INFO] No unprocessed capture session was found.")

        return saved_paths

    def get_video_duration_seconds(self, video_path):
        """
        Read the video stream duration in seconds using ffprobe.

        The container duration can be longer than the encoded video stream
        when a full-length microphone track is attached to a short camera
        stream, so synchronization must inspect the video stream itself.
        """

        if not video_path or not os.path.exists(video_path):
            return None

        ffprobe = shutil.which("ffprobe")

        if ffprobe is None:
            print("[WARNING] ffprobe is not installed; cannot read video duration.")
            return None

        command = [
            ffprobe,
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path,
        ]

        try:
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True
            )
            value = result.stdout.strip()

            if not value:
                return None

            return float(value)

        except Exception as e:
            print(f"[WARNING] Could not read video duration: {e}")
            return None

    def get_audio_duration_seconds(self, video_path):
        """Read the first audio stream duration in seconds using ffprobe."""

        if not video_path or not os.path.exists(video_path):
            return None

        ffprobe = shutil.which("ffprobe")

        if ffprobe is None:
            return None

        command = [
            ffprobe,
            "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path,
        ]

        try:
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True
            )
            value = result.stdout.strip()

            if not value:
                return None

            return float(value)

        except Exception:
            return None

    def video_has_audio_stream(self, video_path):
        """
        Return True when the video file contains an audio stream.
        """

        ffprobe = shutil.which("ffprobe")

        if ffprobe is None:
            return False

        command = [
            ffprobe,
            "-v", "error",
            "-select_streams", "a",
            "-show_entries", "stream=index",
            "-of", "csv=p=0",
            video_path,
        ]

        try:
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True
            )
            return bool(result.stdout.strip())

        except Exception:
            return False

    def sync_video_duration(
        self,
        video_path,
        target_duration_seconds,
        tolerance_seconds=0.05,
        wall_start_offset_seconds=0.0,
        wall_recording_duration_seconds=None,
        audio_start_offset_seconds=None,
    ):
        """
        Align a captured video window and force it to the target duration.

        Camera encoders can produce different media durations from the same
        wall-clock capture interval. The wall-clock start offset is translated
        into the video's timeline before trimming, then the selected frames are
        retimed across the requested duration instead of freezing one frame.
        """

        if not video_path or not os.path.exists(video_path):
            return video_path

        if target_duration_seconds is None or target_duration_seconds <= 0:
            return video_path

        current_duration = self.get_video_duration_seconds(video_path)

        if current_duration is None:
            return video_path

        media_start_offset = max(0.0, wall_start_offset_seconds)
        media_seconds_per_wall_second = 1.0
        if (
            wall_recording_duration_seconds is not None
            and wall_recording_duration_seconds > 0
        ):
            media_seconds_per_wall_second = (
                current_duration / wall_recording_duration_seconds
            )
            media_start_offset *= media_seconds_per_wall_second
        media_start_offset = min(
            media_start_offset,
            max(0.0, current_duration - 0.001),
        )
        available_duration = max(
            0.001, current_duration - media_start_offset
        )
        selected_duration = min(
            available_duration,
            target_duration_seconds * media_seconds_per_wall_second,
        )

        has_audio = self.video_has_audio_stream(video_path)
        video_matches_target = (
            media_start_offset <= tolerance_seconds
            and abs(current_duration - target_duration_seconds)
            <= tolerance_seconds
        )

        if video_matches_target:
            if not has_audio:
                return video_path

            audio_duration = self.get_audio_duration_seconds(video_path)
            if (
                audio_duration is None
                or abs(audio_duration - target_duration_seconds)
                <= tolerance_seconds
            ):
                return video_path

        ffmpeg = shutil.which("ffmpeg")

        if ffmpeg is None:
            print("[WARNING] ffmpeg is not installed; cannot synchronize video duration.")
            return video_path

        target_text = f"{target_duration_seconds:.3f}"
        start_text = f"{media_start_offset:.6f}"
        selected_text = f"{selected_duration:.6f}"
        duration_scale = target_duration_seconds / selected_duration
        scale_text = f"{duration_scale:.9f}"

        base, ext = os.path.splitext(video_path)
        temp_output_path = f"{base}_sync_tmp{ext}"

        command = [
            ffmpeg,
            "-y",
            "-i", video_path,
            "-vf", (
                f"trim=start={start_text}:duration={selected_text},"
                "setpts=PTS-STARTPTS,"
                f"setpts={scale_text}*PTS,"
                f"tpad=stop_mode=clone:stop_duration={target_text},"
                "fps=fps=source_fps,"
                f"trim=duration={target_text},setpts=PTS-STARTPTS"
            ),
            "-t", target_text,
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-pix_fmt", "yuv420p",
        ]

        if has_audio:
            if audio_start_offset_seconds is None:
                audio_start_offset_seconds = wall_start_offset_seconds
            audio_start_text = (
                f"{max(0.0, audio_start_offset_seconds):.6f}"
            )
            command.extend(
                [
                    "-map", "0:v:0",
                    "-map", "0:a:0",
                    "-af", (
                        f"atrim=start={audio_start_text},"
                        "asetpts=PTS-STARTPTS,apad,"
                        f"atrim=duration={target_text}"
                    ),
                    "-c:a", "aac",
                ]
            )
        else:
            command.extend(["-an"])

        command.append(temp_output_path)

        try:
            subprocess.run(
                command,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

            os.replace(temp_output_path, video_path)
            print(
                "[INFO] Video duration synchronized "
                f"({current_duration:.2f}s -> {target_duration_seconds:.2f}s)."
            )

            return video_path

        except Exception as e:
            print(f"[WARNING] Could not synchronize video duration: {e}")

            if os.path.exists(temp_output_path):
                try:
                    os.remove(temp_output_path)
                except Exception:
                    pass

            return video_path

    def get_indicator_color(self, indicator_text):
        """
        Return a color for the live status icon based on indicator text.
        """

        if indicator_text is None:
            return COLORS["accent"]

        if "LIVE" in indicator_text:
            return COLORS["success"]

        if "REC" in indicator_text:
            return COLORS["record"]

        if "ERROR" in indicator_text:
            return COLORS["error"]

        if "WARNING" in indicator_text:
            return COLORS["warning"]

        if "STARTING" in indicator_text:
            return COLORS["warning"]

        if "RESTARTING" in indicator_text:
            return COLORS["warning"]

        if "SWITCHING" in indicator_text:
            return COLORS["warning"]

        if "IDLE" in indicator_text:
            return COLORS["idle"]

        if "READY" in indicator_text:
            return COLORS["accent"]

        return COLORS["accent"]

    def update_tracking_button(self):
        """
        Update tracking button text and color depending on camera state.

        Camera inactive:
            Start Tracking = green

        Camera active:
            Stop Tracking = red
        """

        if self.camera.is_tracking:
            self.track_button.config(
                text="Stop Tracking",
                style="Stop.TButton"
            )
        else:
            self.track_button.config(
                text="Start Tracking",
                style="Start.TButton"
            )

    def update_status(self, message, indicator_text=None):
        """
        Update bottom status label and live indicator.

        Status color examples:
        - LIVE: green
        - REC: red
        - ERROR: red
        - WARNING: yellow
        - STARTING: yellow
        - RESTARTING: yellow
        - SWITCHING: yellow
        - IDLE: gray
        - READY: blue
        """

        self.status_label.config(text=message)

        if indicator_text is not None:
            self.live_indicator_label.config(
                text=indicator_text,
                fg=self.get_indicator_color(indicator_text)
            )

    def set_processing_indicator(self, is_processing, message="Processing..."):
        """
        Show or hide the post-capture processing progress indicator.
        """

        if is_processing:
            self.processing_indicator_label.config(text=message)
            if not self.processing_indicator_frame.winfo_ismapped():
                self.processing_indicator_frame.pack(side="left")
            self.processing_progress_bar.start(10)
            return

        self.processing_progress_bar.stop()
        if self.processing_indicator_frame.winfo_ismapped():
            self.processing_indicator_frame.pack_forget()

    def refresh_info_panel(self):
        """
        Refresh all system information labels.
        """

        self.camera_info_label.config(
            text=f"Camera: {self.camera.camera_index}"
        )

        thermal_status = self.thermal_camera.status
        if self.thermal_camera.is_recording:
            thermal_status = f"Recording @ {self.thermal_camera.record_fps} FPS"
        elif self.thermal_camera.error:
            thermal_status = "unavailable"

        self.thermal_info_label.config(
            text=f"Thermal: {thermal_status}"
        )
        self.refresh_thermal_scale_controls()

        self.microphone_info_label.config(
            text=f"Mic: {self.camera.microphone_device_name}"
        )

        self.output_info_label.config(
            text=f"Output: {self.camera.output_dir}/"
        )

        self.fps_info_label.config(
            text=(
                f"Regular Camera FPS: {self.camera.record_fps:g}\n"
                f"Thermal Camera FPS: {self.thermal_camera.record_fps:g}"
            )
        )

        self.zoom_label.config(
            text=f"Zoom: {self.camera.zoom_factor}x"
        )

    def run_capture_ffmpeg(self, label, command):
        """Run one FFmpeg capture-export command."""
        print(f"[INFO] Starting {label}...")
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print(f"[INFO] Finished {label}.")

    def move_capture_output(self, source_path, output_path):
        """Move a finalized recording to its canonical session location."""
        source_path = os.path.abspath(source_path)
        output_path = os.path.abspath(output_path)

        if source_path != output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            # Canonical battery-folder outputs represent the latest completed
            # capture at that charge level. Replace the previous known file
            # atomically instead of creating another timestamped folder.
            os.replace(source_path, output_path)

    def generate_capture_outputs(
        self,
        video_path,
        thermal_video_path,
        scale_video_path,
        pulse_voice_paths,
        temperature_log_path,
        temperature_average_path,
        average_fps_log_path,
        output_folder,
        expected_image_count=None,
    ):
        """Generate all requested capture artifacts without an external script."""
        os.makedirs(output_folder, exist_ok=True)
        camera_frames = os.path.join(output_folder, "Camera_Frames")
        thermal_frames = os.path.join(output_folder, "Thermal_Frames")
        pulse_voice_folder = os.path.join(
            output_folder,
            "Voice_Recordings",
        )
        # Refresh only folders owned by this export. Never remove the selected
        # battery folder or unrelated files the user placed inside it.
        for generated_folder in (
            camera_frames,
            thermal_frames,
            pulse_voice_folder,
        ):
            if os.path.isdir(generated_folder):
                shutil.rmtree(generated_folder)

        os.makedirs(camera_frames, exist_ok=True)
        os.makedirs(thermal_frames, exist_ok=True)

        frame_limit_arguments = []
        if expected_image_count is not None:
            frame_limit_arguments = [
                "-frames:v",
                str(expected_image_count),
            ]

        self.run_capture_ffmpeg(
            "camera frame extraction",
            [
                "ffmpeg",
                "-y",
                "-i",
                video_path,
                "-vf",
                f"fps={DATASET_FRAMES_PER_SECOND:g}",
                "-start_number",
                "0",
                "-qscale:v",
                "2",
                *frame_limit_arguments,
                os.path.join(camera_frames, "Camera_Frame_%03d.jpg"),
            ]
        )
        self.run_capture_ffmpeg(
            "thermal frame extraction",
            [
                "ffmpeg",
                "-y",
                "-i",
                thermal_video_path,
                "-vf",
                f"fps={DATASET_FRAMES_PER_SECOND:g}",
                "-start_number",
                "0",
                "-qscale:v",
                "2",
                *frame_limit_arguments,
                os.path.join(thermal_frames, "Thermal_Frame_%03d.jpg"),
            ]
        )

        if expected_image_count is not None:
            extracted_counts = {
                "regular": len(os.listdir(camera_frames)),
                "thermal": len(os.listdir(thermal_frames)),
            }
            for label, extracted_count in extracted_counts.items():
                if extracted_count != expected_image_count:
                    raise RuntimeError(
                        f"Expected {expected_image_count} {label} images, "
                        f"but extracted {extracted_count}."
                    )

        if pulse_voice_paths:
            os.makedirs(pulse_voice_folder, exist_ok=True)

            for index, pulse_voice_path in enumerate(
                pulse_voice_paths,
                start=1,
            ):
                if not pulse_voice_path or not os.path.exists(pulse_voice_path):
                    print(
                        "[WARNING] Pulse voice recording is unavailable: "
                        f"{pulse_voice_path}"
                    )
                    continue

                self.move_capture_output(
                    pulse_voice_path,
                    os.path.join(
                        pulse_voice_folder,
                        f"Voice_{index:03d}.wav"
                    )
                )

        capture_moves = (
            (video_path, os.path.join(output_folder, "Camera_Video.mp4")),
            (
                thermal_video_path,
                os.path.join(output_folder, "Thermal_Video.mp4")
            ),
            (
                scale_video_path,
                os.path.join(output_folder, "Thermal_Range_Video.mp4")
            ),
            (
                temperature_log_path,
                os.path.join(output_folder, "Thermal_Temperature_Log.txt")
            ),
            (
                temperature_average_path,
                os.path.join(output_folder, "Thermal_Temperature_Averages.txt")
            ),
            (
                average_fps_log_path,
                os.path.join(output_folder, "Camera_Average_FPS.txt")
            ),
        )
        for source_path, output_path in capture_moves:
            self.move_capture_output(source_path, output_path)

    def export_capture_session(self, video_path, thermal_video_path=None):
        """
        Export the capture artifacts from the latest recording.

        Outputs: camera frames, thermal frames, per-pulse voice recordings,
        camera video, thermal video, thermal-range video, two temperature
        text files, and the measured camera average-FPS log.
        """

        if video_path is None:
            print("[INFO] No video path available for processing.")
            return

        if not os.path.exists(video_path):
            print(f"[WARNING] Video file does not exist: {video_path}")
            return

        if self.last_processed_video_path == video_path:
            print(f"[INFO] Video already processed: {video_path}")
            return

        if thermal_video_path is None:
            thermal_video_path = self.current_thermal_video_path

        scale_video_path = self.thermal_camera.scale_video_path
        temperature_log_path = self.thermal_camera.temperature_log_path
        temperature_average_path = self.thermal_camera.temperature_average_path
        average_fps_log_path = self.average_fps_log_path
        pulse_voice_paths = [
            recording["path"]
            for recording in self.pulse_recordings
            if recording.get("path")
        ]
        expected_image_count = None
        if pulse_voice_paths:
            expected_image_count = (
                len(pulse_voice_paths) * DATASET_FRAMES_PER_PULSE
            )

        required_outputs = (
            (thermal_video_path, "thermal video"),
            (scale_video_path, "thermal-range video"),
            (temperature_log_path, "thermal temperature log"),
            (temperature_average_path, "thermal temperature averages"),
            (average_fps_log_path, "camera average FPS log"),
        )
        for required_path, label in required_outputs:
            if not required_path or not os.path.exists(required_path):
                print(f"[WARNING] {label.title()} is unavailable: {required_path}")
                self.update_status(
                    f"Status: {label.title()} is unavailable",
                    "● WARNING"
                )
                return

        # The battery-level directory is the session container. Do not create a
        # second timestamped ``*_Image_and_Video`` directory beneath it.
        output_folder = self.captures_dir
        self.last_processed_video_path = video_path

        def worker():
            print(f"[INFO] Generating capture outputs for: {video_path}")

            self.root.after(
                0,
                lambda: (
                    self.update_status(
                        "Status: Generating camera, thermal, and voice outputs...",
                        "● WARNING"
                    ),
                    self.set_processing_indicator(True, "Generating outputs...")
                )
            )

            try:
                self.generate_capture_outputs(
                    video_path,
                    thermal_video_path,
                    scale_video_path,
                    pulse_voice_paths,
                    temperature_log_path,
                    temperature_average_path,
                    average_fps_log_path,
                    output_folder,
                    expected_image_count=expected_image_count,
                )
                print(f"[INFO] Capture outputs saved in: {output_folder}")

                self.root.after(
                    0,
                    lambda: self.handle_capture_outputs_finished(
                        output_folder,
                        len(pulse_voice_paths),
                        expected_image_count,
                    )
                )

            except Exception as e:
                print(f"[ERROR] Capture output generation failed: {e}")
                if self.last_processed_video_path == video_path:
                    self.last_processed_video_path = None

                self.root.after(
                    0,
                    lambda: (
                        self.set_processing_indicator(False),
                        self.update_status(
                            "Status: Capture output generation failed",
                            "● WARNING"
                        )
                    )
                )

        threading.Thread(
            target=worker,
            daemon=True
        ).start()

    def handle_capture_outputs_finished(
        self,
        output_folder,
        pulse_recording_count=0,
        image_count_per_camera=None,
    ):
        """Report successful capture-only output generation."""

        self.set_processing_indicator(False)
        self.refresh_info_panel()

        self.update_status(
            f"Status: Capture outputs saved in {output_folder}",
            "● IDLE"
        )
        pulse_output_message = ""
        if pulse_recording_count:
            pulse_output_message = (
                f"• Voice_Recordings "
                f"({pulse_recording_count} pulse WAV files)\n"
            )
        image_output_message = ""
        if image_count_per_camera is not None:
            image_output_message = (
                f"• Camera_Frames ({image_count_per_camera} images)\n"
                f"• Thermal_Frames ({image_count_per_camera} images)\n"
            )
        else:
            image_output_message = (
                "• Camera_Frames\n"
                "• Thermal_Frames\n"
            )

        messagebox.showinfo(
            "Capture Outputs Saved",
            "Generated outputs:\n"
            f"{image_output_message}"
            f"{pulse_output_message}"
            "• Camera_Video.mp4\n"
            "• Thermal_Video.mp4\n"
            "• Thermal_Range_Video.mp4\n"
            "• Thermal_Temperature_Log.txt\n"
            "• Thermal_Temperature_Averages.txt\n"
            "• Camera_Average_FPS.txt\n\n"
            f"Saved in: {output_folder}"
        )

    def get_pulse_command(self):
        """Return the command for one continuous repeated-pulse sequence."""
        pulse_folder = os.path.join(self.base_dir, "Pulse Generation")
        pulse_script = os.path.join(
            pulse_folder,
            "pulse_protocol_generator.py"
        )

        if not os.path.exists(pulse_script):
            print(f"[WARNING] Pulse generator not found: {pulse_script}")
            return None, None

        command = [
            sys.executable,
            pulse_script,
            "--mode",
            "generate-play-record",
            "--count",
            str(self.requested_pulse_count),
            "--record-output-template",
            os.path.abspath(self.get_pulse_recording_template()),
            "--wait-for-start",
        ]

        if self.camera.microphone_device_id is not None:
            command.extend(
                [
                    "--input-device",
                    str(self.camera.microphone_device_id),
                ]
            )

        return command, pulse_folder

    def get_pulse_recording_template(self):
        """Return the temporary WAV template for a repeated-pulse session."""
        timestamp = (
            self.current_recording_timestamp
            or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        )
        return os.path.join(
            self.captures_dir,
            f"CoBas_V1_PulseVoice_{timestamp}_{{pulse:03d}}.wav"
        )

    def get_pulse_recording_path(self, pulse_number):
        """Return the temporary WAV path for one pulse recording."""
        return self.get_pulse_recording_template().format(
            pulse=pulse_number
        )

    @staticmethod
    def write_silence_frames(wav_file, frame_count, frame_size):
        """Write silence without allocating the entire interval in memory."""
        remaining = max(0, int(frame_count))
        silence_chunk = b"\0" * (4096 * frame_size)

        while remaining:
            chunk_frames = min(remaining, 4096)
            wav_file.writeframesraw(
                silence_chunk[:chunk_frames * frame_size]
            )
            remaining -= chunk_frames

    def build_pulse_session_audio(self):
        """
        Build one timeline-aligned WAV for the continuous camera videos.

        The individual two-second WAV files remain separate for export. Silence
        fills the periods when the microphone was closed between pulses.
        """
        if (
            not self.camera.is_recording
            or self.camera.record_start_time is None
        ):
            return None

        available_recordings = [
            recording
            for recording in self.pulse_recordings
            if os.path.exists(recording.get("path", ""))
        ]

        if not available_recordings:
            return None

        output_path = self.camera.temp_audio_path
        if not output_path:
            return None

        first_path = available_recordings[0]["path"]
        try:
            with wave.open(first_path, "rb") as first_wav:
                channels = first_wav.getnchannels()
                sample_width = first_wav.getsampwidth()
                sample_rate = first_wav.getframerate()
        except Exception as error:
            print(f"[WARNING] Could not inspect pulse recording: {error}")
            return None

        frame_size = channels * sample_width
        current_frame = 0
        session_duration = 0.0
        if self.camera.record_start_time is not None:
            session_duration = max(
                0.0,
                time.time() - self.camera.record_start_time
            )

        try:
            with wave.open(output_path, "wb") as output_wav:
                output_wav.setnchannels(channels)
                output_wav.setsampwidth(sample_width)
                output_wav.setframerate(sample_rate)

                for recording in sorted(
                    available_recordings,
                    key=lambda item: item["offset_seconds"],
                ):
                    target_frame = max(
                        0,
                        int(round(
                            recording["offset_seconds"] * sample_rate
                        )),
                    )

                    if target_frame > current_frame:
                        self.write_silence_frames(
                            output_wav,
                            target_frame - current_frame,
                            frame_size,
                        )
                        current_frame = target_frame

                    with wave.open(recording["path"], "rb") as pulse_wav:
                        pulse_format = (
                            pulse_wav.getnchannels(),
                            pulse_wav.getsampwidth(),
                            pulse_wav.getframerate(),
                        )
                        expected_format = (
                            channels,
                            sample_width,
                            sample_rate,
                        )
                        if pulse_format != expected_format:
                            print(
                                "[WARNING] Skipping pulse recording with "
                                f"incompatible WAV format: {recording['path']}"
                            )
                            continue

                        overlap_frames = max(
                            0,
                            current_frame - target_frame,
                        )
                        if overlap_frames:
                            pulse_wav.setpos(
                                min(overlap_frames, pulse_wav.getnframes())
                            )

                        pulse_frames = pulse_wav.readframes(
                            pulse_wav.getnframes() - pulse_wav.tell()
                        )
                        output_wav.writeframesraw(pulse_frames)
                        current_frame += len(pulse_frames) // frame_size

                session_frames = int(round(
                    session_duration * sample_rate
                ))
                if session_frames > current_frame:
                    self.write_silence_frames(
                        output_wav,
                        session_frames - current_frame,
                        frame_size,
                    )

            return output_path

        except Exception as error:
            print(f"[WARNING] Could not build pulse audio timeline: {error}")
            return None

    def start_pulse_sequence(self, start_token):
        """Run one gap-free process for all requested two-second pulses."""

        def worker():
            if start_token != self.tracking_start_token:
                return

            command, pulse_folder = self.get_pulse_command()
            if command is None:
                self.root.after(
                    0,
                    lambda: self.handle_pulse_sequence_failed(
                        start_token,
                        "Pulse generator was not found",
                    )
                )
                return

            try:
                with self.pulse_process_lock:
                    if start_token != self.tracking_start_token:
                        return

                    pulse_process = subprocess.Popen(
                        command,
                        cwd=pulse_folder,
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                    )
                    self.pulse_process = pulse_process
            except Exception as error:
                self.root.after(
                    0,
                    lambda error=error: self.handle_pulse_sequence_failed(
                        start_token,
                        f"Pulse sequence could not start: {error}",
                    ),
                )
                return

            pulse_offsets = {}
            completed_pulses = set()
            ready_received = False
            start_result = {"started": False}
            start_result_event = threading.Event()

            def start_recordings():
                try:
                    start_result["started"] = (
                        self.handle_pulse_sequence_ready(start_token)
                    )
                finally:
                    start_result_event.set()

            for line in pulse_process.stdout:
                line = line.strip()
                if line:
                    print(f"[PULSE SEQUENCE] {line}")

                if line.startswith("SEQUENCE_READY"):
                    ready_received = True
                    self.root.after(0, start_recordings)

                    while not start_result_event.wait(timeout=0.1):
                        if start_token != self.tracking_start_token:
                            return

                    if not start_result["started"]:
                        try:
                            pulse_process.terminate()
                        except Exception:
                            pass
                        break

                    try:
                        pulse_process.stdin.write("START\n")
                        pulse_process.stdin.flush()
                    except Exception as error:
                        print(
                            "[WARNING] Could not release pulse sequence: "
                            f"{error}"
                        )
                        try:
                            pulse_process.terminate()
                        except Exception:
                            pass
                        break

                elif line.startswith("PLAYBACK_STARTED"):
                    parts = line.split(maxsplit=2)
                    if len(parts) != 3:
                        continue

                    try:
                        pulse_number = int(parts[1])
                        playback_started_at = float(parts[2])
                    except (TypeError, ValueError):
                        continue

                    if pulse_number == 1:
                        self.pulse_sequence_started_at = (
                            playback_started_at
                        )
                    if self.camera.record_start_time is not None:
                        pulse_offset = max(
                            0.0,
                            playback_started_at
                            - self.camera.record_start_time,
                        )
                    else:
                        pulse_offset = (
                            (pulse_number - 1) * PULSE_DURATION_SECONDS
                        )
                    pulse_offsets[pulse_number] = pulse_offset
                    self.root.after(
                        0,
                        lambda pulse_number=pulse_number: (
                            self.handle_pulse_started(
                                pulse_number,
                                start_token,
                            )
                        ),
                    )

                elif line.startswith("PULSE_FINISHED"):
                    parts = line.split(maxsplit=2)
                    if len(parts) != 3:
                        continue

                    try:
                        pulse_number = int(parts[1])
                    except (TypeError, ValueError):
                        continue

                    recording_path = parts[2]
                    pulse_offset = pulse_offsets.get(pulse_number)
                    if (
                        pulse_number in completed_pulses
                        or pulse_offset is None
                        or not os.path.exists(recording_path)
                    ):
                        continue

                    completed_pulses.add(pulse_number)
                    self.pulse_recordings.append(
                        {
                            "path": recording_path,
                            "offset_seconds": pulse_offset,
                        }
                    )
                    self.root.after(
                        0,
                        lambda pulse_number=pulse_number: (
                            self.handle_pulse_completed(
                                pulse_number,
                                start_token,
                            )
                        ),
                    )

            return_code = pulse_process.wait()
            with self.pulse_process_lock:
                if self.pulse_process is pulse_process:
                    self.pulse_process = None

            if start_token != self.tracking_start_token:
                return

            if (
                return_code != 0
                or not ready_received
                or len(completed_pulses) != self.requested_pulse_count
            ):
                self.root.after(
                    0,
                    lambda: self.handle_pulse_sequence_failed(
                        start_token,
                        "Pulse sequence playback or recording failed",
                    ),
                )
                return

            self.root.after(
                0,
                lambda: self.handle_pulse_sequence_finished(start_token)
            )

        threading.Thread(target=worker, daemon=True).start()

    def handle_pulse_sequence_ready(self, start_token):
        """Start both camera recorders, then allow pulse playback to begin."""
        if start_token != self.tracking_start_token:
            return False

        self.start_active_recordings(
            record_audio=False,
            recording_timestamp=self.current_recording_timestamp,
        )
        if not self.current_video_path or not self.current_thermal_video_path:
            print("[WARNING] Both synchronized camera recorders are required.")
            return False

        self.is_preparing_tracking = False
        self.track_button.config(state="normal")
        self.record_button.config(
            text="Stop Recording",
            style="VideoStop.TButton",
            state="disabled",
        )
        target_duration = (
            self.requested_pulse_count * PULSE_DURATION_SECONDS
        )
        self.update_status(
            (
                "Status: Cameras recording continuously for "
                f"{target_duration:g} seconds; starting pulse 1"
            ),
            "● REC",
        )
        return True

    def handle_pulse_started(self, pulse_number, start_token):
        """Update the interface when one pulse begins."""
        if start_token != self.tracking_start_token:
            return

        self.current_pulse_number = pulse_number
        self.pulse_playback_end_time = (
            time.time() + PULSE_DURATION_SECONDS
        )
        self.update_status(
            (
                f"Status: Playing and recording pulse {pulse_number} "
                f"of {self.requested_pulse_count}"
            ),
            "● REC",
        )

    def handle_pulse_completed(self, pulse_number, start_token):
        """Report that one microphone stream was closed successfully."""
        if start_token != self.tracking_start_token:
            return

        self.pulse_playback_end_time = None
        self.record_timer_label.config(
            text=(
                f"Pulses: {pulse_number}/{self.requested_pulse_count}"
            )
        )

        if pulse_number < self.requested_pulse_count:
            self.update_status(
                (
                    f"Status: Pulse {pulse_number} recorded; "
                    f"preparing pulse {pulse_number + 1}"
                ),
                "● REC",
            )

    def handle_pulse_sequence_failed(self, start_token, message):
        """Stop the cameras safely if a pulse cannot be played or recorded."""
        if start_token != self.tracking_start_token:
            return

        print(f"[WARNING] {message}")
        self.finish_pulse_sequence(
            completion_message=message,
            indicator_text="● WARNING",
        )

    def handle_pulse_sequence_finished(self, start_token):
        """Finalize the continuous camera session after the last pulse."""
        if start_token != self.tracking_start_token:
            return

        self.finish_pulse_sequence(
            completion_message=(
                f"Completed {len(self.pulse_recordings)} "
                "pulse recordings"
            ),
            indicator_text="● IDLE",
        )

    def finish_pulse_sequence(self, completion_message, indicator_text):
        """Stop both cameras once and export the repeated-pulse session."""
        self.is_preparing_tracking = False
        self.pulse_sequence_active = False
        self.pulse_playback_end_time = None
        with self.pulse_process_lock:
            self.pulse_process = None
        self.cancel_preview_loop()
        self.cancel_thermal_preview_loop()

        session_audio_path = self.build_pulse_session_audio()
        target_duration = (
            len(self.pulse_recordings) * PULSE_DURATION_SECONDS
            if self.pulse_recordings
            else None
        )
        saved_video_path, saved_thermal_video_path, _ = (
            self.finalize_capture_session(
                target_duration_seconds=target_duration,
                audio_path_override=session_audio_path,
                capture_window_start_time=self.pulse_sequence_started_at,
            )
        )

        self.camera.stop_camera()
        self.thermal_camera.stop_camera()

        self.video_label.config(
            image="",
            text=(
                "Pulse recording complete.\n\n"
                "Click 'Start Tracking' to start again."
            ),
            bg=COLORS["preview_bg"],
            fg=COLORS["muted_text"],
        )
        self.video_label.image = None
        self.thermal_video_label.config(
            image="",
            text=(
                "Thermal recording complete.\n\n"
                "Click 'Start Tracking' to start again."
            ),
            bg=COLORS["preview_bg"],
            fg=COLORS["muted_text"],
        )
        self.thermal_video_label.image = None

        self.record_button.config(
            text="Capture Video",
            style="Capture.TButton",
            state="normal",
        )
        self.record_timer_label.config(
            text=(
                f"Pulses recorded: {len(self.pulse_recordings)}"
            )
        )
        self.pulse_count_spinbox.config(state="normal")
        self.track_button.config(state="normal")
        self.update_tracking_button()
        self.refresh_info_panel()

        status_message = f"Status: {completion_message}"
        if saved_video_path and saved_thermal_video_path:
            status_message += "; camera and thermal videos saved"
        self.update_status(status_message, indicator_text)

    def stop_pulse_sequence_process(self):
        """Stop the active pulse playback/recording subprocess."""
        with self.pulse_process_lock:
            pulse_process = self.pulse_process

        if pulse_process is None:
            return

        if pulse_process.poll() is not None:
            with self.pulse_process_lock:
                if self.pulse_process is pulse_process:
                    self.pulse_process = None
            return

        try:
            pulse_process.terminate()
            pulse_process.wait(timeout=2)
            print("[INFO] Pulse playback and recording stopped.")

        except subprocess.TimeoutExpired:
            pulse_process.kill()
            pulse_process.wait(timeout=2)
            print("[INFO] Pulse playback and recording killed.")

        except Exception as error:
            print(f"[WARNING] Could not stop pulse process: {error}")

        finally:
            with self.pulse_process_lock:
                if self.pulse_process is pulse_process:
                    self.pulse_process = None

    # --------------------------------------------------
    # External Windows
    # --------------------------------------------------

    def open_settings(self):
        """
        Open the Settings window from Settings.py.
        """

        SettingsWindow(self.root, self)

    def open_about(self):
        """
        Open the About window from About.py.
        """

        show_about_window()

    # --------------------------------------------------
    # Settings.py Callbacks
    # --------------------------------------------------

    def apply_camera_source_from_settings(self, selected_source):
        """
        Called by Settings.py when the user applies a new camera source.
        """

        # Convert numeric string camera indexes to integers.
        if selected_source in ["0", "1"]:
            selected_source = int(selected_source)

        # If tracking is active, stop it before changing camera source.
        was_tracking = self.camera.is_tracking

        if was_tracking:
            self.stop_tracking()

        # Set selected camera source.
        self.camera.set_camera_source(selected_source)

        # Update camera info label immediately.
        self.camera_info_label.config(
            text=f"Camera: {self.camera.camera_index}"
        )

        self.update_status(
            f"Status: Camera source set to {self.camera.camera_index}",
            "● READY"
        )

        # Restart only if tracking was active before the source changed.
        if was_tracking:
            self.root.after(300, self.start_tracking)

    def apply_microphone_source_from_settings(
        self,
        microphone_device_id,
        microphone_device_name
    ):
        """
        Called by Settings.py when the user applies a new microphone source.
        """

        # Do not allow microphone changes during active recording.
        if self.camera.is_recording:
            messagebox.showwarning(
                "Recording Active",
                "Stop recording before changing the microphone."
            )
            return

        # Store selected microphone in the camera backend.
        self.camera.set_microphone_device(
            microphone_device_id,
            microphone_device_name
        )

        self.refresh_info_panel()

        self.update_status(
            f"Status: Microphone set to {microphone_device_name}",
            "● LIVE" if self.camera.is_tracking else "● READY"
        )

    # --------------------------------------------------
    # Tracking Actions
    # --------------------------------------------------

    def toggle_tracking(self):
        """
        Toggle camera tracking.

        If tracking is off:
            start tracking.

        If tracking is on:
            stop tracking.
        """

        if self.is_preparing_tracking:
            return

        if self.camera.is_tracking:
            self.stop_tracking()
        else:
            self.start_tracking()

    def start_tracking(self):
        """
        Start a user-configured repeated two-second pulse capture.
        """

        print("Start Tracking clicked")

        try:
            requested_pulse_count = int(self.pulse_count_text.get())
            if requested_pulse_count < 1:
                raise ValueError
        except (TypeError, ValueError):
            messagebox.showwarning(
                "Invalid Pulse Count",
                "Enter a whole number greater than zero for the pulse count."
            )
            self.pulse_count_spinbox.focus_set()
            return

        # Prevent duplicate preview loops before starting.
        self.cancel_preview_loop()
        self.cancel_thermal_preview_loop()

        self.is_preparing_tracking = True
        self.pulse_sequence_active = True
        with self.pulse_process_lock:
            self.tracking_start_token += 1
            start_token = self.tracking_start_token
        self.requested_pulse_count = requested_pulse_count
        self.current_pulse_number = 0
        self.pulse_recordings = []
        self.pulse_sequence_started_at = None
        self.current_recording_timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )
        self.current_video_path = None
        self.current_thermal_video_path = None

        self.start_thermal_camera_feed()
        self.pulse_count_spinbox.config(state="disabled")
        self.record_button.config(state="disabled")

        self.track_button.config(
            text="Preparing...",
            style="Start.TButton",
            state="disabled"
        )

        self.video_label.config(
            image="",
            text=(
                "Preparing continuous camera recording...\n\n"
                f"{requested_pulse_count} pulses requested."
            ),
            bg=COLORS["preview_bg"],
            fg=COLORS["muted_text"]
        )
        self.video_label.image = None

        self.thermal_video_label.config(
            image="",
            text=(
                "Preparing thermal camera...\n\n"
                "It will record from the first pulse through the last."
            ),
            bg=COLORS["preview_bg"],
            fg=COLORS["muted_text"]
        )
        self.thermal_video_label.image = None

        self.update_status(
            (
                "Status: Preparing cameras for "
                f"{requested_pulse_count} pulse recordings..."
            ),
            "● STARTING"
        )

        self.root.after(
            0,
            lambda: self.begin_tracking_capture(start_token)
        )

    def begin_tracking_capture(self, start_token):
        """
        Start both continuous camera recorders before the first pulse.
        """

        if start_token != self.tracking_start_token:
            return

        self.update_status(
            "Status: Starting continuous camera recording...",
            "● STARTING"
        )

        if not self.thermal_camera.is_tracking:
            self.start_thermal_camera_feed()

        # Open camera through the camera backend.
        started = self.camera.start_camera()

        if started:
            self.refresh_info_panel()

            self.update_status(
                f"Status: Tracking battery using {self.camera.camera_index}",
                "● LIVE"
            )

            # Mark preview loop active.
            self.preview_loop_running = True

            # Change Start Tracking button into Stop Tracking.
            self.update_tracking_button()

            # Prepare one gap-free audio process first. It waits for both
            # camera recorders before beginning the first pulse.
            self.update_camera_feed()
            self.start_pulse_sequence(start_token)

        else:
            self.is_preparing_tracking = False
            self.pulse_sequence_active = False
            self.stop_pulse_sequence_process()
            self.thermal_camera.stop_camera()
            self.cancel_thermal_preview_loop()
            self.pulse_count_spinbox.config(state="normal")
            self.record_button.config(state="normal")
            self.track_button.config(state="normal")

            # Keep button as Start Tracking if camera failed.
            self.update_tracking_button()

            self.update_status(
                "Status: Camera failed to open",
                "● ERROR"
            )

            self.video_label.config(
                image="",
                text="Camera could not be opened.\n\nClose Cheese and try /dev/video1.",
                bg=COLORS["preview_bg"],
                fg=COLORS["error"]
            )
            self.video_label.image = None

            messagebox.showerror(
                "Camera Error",
                "Could not open the camera.\n\n"
                "Try these:\n"
                "1. Close Cheese completely.\n"
                "2. Confirm Pixel 8a is still in Android Webcam mode.\n"
                "3. Try /dev/video0 or /dev/video1.\n"
                "4. Run: v4l2-ctl --list-devices"
            )

    def stop_tracking(self):
        """
        Stop camera preview safely.

        If video recording is active, stop it first.
        Then generate the capture-only camera, thermal, and voice outputs.
        """

        self.is_preparing_tracking = False
        with self.pulse_process_lock:
            self.tracking_start_token += 1
        self.pulse_playback_end_time = None
        self.pulse_sequence_active = False
        self.track_button.config(state="normal")

        # Cancel preview update loop first.
        self.cancel_preview_loop()
        self.cancel_thermal_preview_loop()

        # Stop the current pulse before closing the continuous camera session.
        self.stop_pulse_sequence_process()

        session_audio_path = self.build_pulse_session_audio()
        target_duration = (
            len(self.pulse_recordings) * PULSE_DURATION_SECONDS
            if self.pulse_recordings
            else None
        )
        saved_video_path, saved_thermal_video_path, _ = (
            self.finalize_capture_session(
                target_duration_seconds=target_duration,
                audio_path_override=session_audio_path,
                capture_window_start_time=self.pulse_sequence_started_at,
            )
        )

        # Release camera through backend.
        self.camera.stop_camera()
        self.thermal_camera.stop_camera()

        # Reset preview display.
        self.video_label.config(
            image="",
            text="Tracking stopped.\n\nClick 'Start Tracking' to start again.",
            bg=COLORS["preview_bg"],
            fg=COLORS["muted_text"]
        )
        self.video_label.image = None

        self.thermal_video_label.config(
            image="",
            text="Thermal tracking stopped.\n\nClick 'Start Tracking' to start again.",
            bg=COLORS["preview_bg"],
            fg=COLORS["muted_text"]
        )
        self.thermal_video_label.image = None

        # Reset record button and timer.
        self.record_button.config(
            text="Capture Video",
            style="Capture.TButton",
            state="normal",
        )
        self.record_timer_label.config(
            text=f"Pulses recorded: {len(self.pulse_recordings)}"
        )
        self.pulse_count_spinbox.config(state="normal")

        self.update_status(
            "Status: Tracking stopped",
            "● IDLE"
        )

        self.refresh_info_panel()

        # Change Stop Tracking button back into Start Tracking.
        self.update_tracking_button()

    def restart_camera(self):
        """
        Restart the camera safely.
        """

        print("Restart Camera clicked")

        # Restarting while recording would corrupt or interrupt recording.
        if self.camera.is_recording or self.thermal_camera.is_recording:
            messagebox.showwarning(
                "Recording Active",
                "Stop recording before restarting the camera."
            )
            return

        # Cancel preview loop.
        self.cancel_preview_loop()
        self.cancel_thermal_preview_loop()

        # Release active camera if needed.
        if self.camera.is_tracking:
            self.camera.stop_camera()

        if self.thermal_camera.is_tracking:
            self.thermal_camera.stop_camera()

        # Update preview text.
        self.video_label.config(
            image="",
            text="Restarting camera...\n\nPlease wait.",
            bg=COLORS["preview_bg"],
            fg=COLORS["muted_text"]
        )
        self.video_label.image = None

        self.thermal_video_label.config(
            image="",
            text="Restarting thermal camera...\n\nPlease wait.",
            bg=COLORS["preview_bg"],
            fg=COLORS["muted_text"]
        )
        self.thermal_video_label.image = None

        self.update_status(
            "Status: Restarting camera...",
            "● RESTARTING"
        )

        # Restart after short delay to let camera fully release.
        self.root.after(300, self.start_tracking)

    def update_camera_feed(self):
        """
        Continuously update live camera preview.
        """

        # Stop preview loop if the camera is no longer tracking.
        if not self.camera.is_tracking:
            self.cancel_preview_loop()
            self.update_tracking_button()
            return

        # Read one frame from the camera backend.
        frame = self.camera.read_frame()

        if frame is not None:
            # Save frame to video if recording is active.
            self.camera.write_video_frame(frame)

            # Convert OpenCV BGR image to RGB for Tkinter/PIL.
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Convert NumPy frame to PIL image.
            image = Image.fromarray(frame_rgb)

            preview_width, preview_height = self.get_preview_dimensions(
                self.video_label
            )
            image = image.resize(
                (preview_width, preview_height),
                Image.Resampling.LANCZOS
            )

            # Convert PIL image to Tkinter image.
            self.preview_photo = ImageTk.PhotoImage(image=image)

            # Display the frame in the preview label.
            self.video_label.config(
                image=self.preview_photo,
                text=""
            )

            # Keep reference so Tkinter does not remove the image.
            self.video_label.image = self.preview_photo

        else:
            self.update_status(
                "Status: Camera connection lost after automatic recovery",
                "● ERROR"
            )

        # Update recording timer while recording.
        if self.camera.is_recording:
            if (
                self.pulse_sequence_active
                and self.pulse_playback_end_time is not None
            ):
                self.record_timer_label.config(
                    text=(
                        f"Pulse {self.current_pulse_number}/"
                        f"{self.requested_pulse_count}"
                    )
                )
            elif self.pulse_sequence_active:
                self.record_timer_label.config(
                    text=(
                        f"Pulses: {len(self.pulse_recordings)}/"
                        f"{self.requested_pulse_count}"
                    )
                )
            else:
                self.record_timer_label.config(
                    text="Recording active"
                )

        # Schedule next frame update.
        frame_interval_ms = max(
            1,
            int(round(1000.0 / self.camera.record_fps)),
        )
        if self.camera.is_tracking:
            self.preview_after_id = self.root.after(
                frame_interval_ms,
                self.update_camera_feed,
            )

    def switch_front_back_camera(self):
        """
        Switch between /dev/video0 and /dev/video1.

        In Android Webcam mode, these device nodes may represent
        different camera streams, such as front and back cameras.
        """

        # Do not switch camera while recording.
        if self.camera.is_recording:
            messagebox.showwarning(
                "Recording Active",
                "Stop recording before switching the camera."
            )
            return

        self.update_status(
            "Status: Switching camera...",
            "● SWITCHING"
        )

        # Stop current camera first.
        if self.camera.is_tracking:
            self.cancel_preview_loop()
            self.camera.stop_camera()

        # Toggle camera source in backend.
        new_source = self.camera.switch_camera_source()

        # Show new source immediately.
        self.camera_info_label.config(
            text=f"Camera: {new_source}"
        )

        # Start the newly selected camera.
        started = self.camera.start_camera()

        if started:
            self.refresh_info_panel()

            self.update_status(
                f"Status: Switched camera to {self.camera.camera_index}",
                "● LIVE"
            )

            self.preview_loop_running = True
            self.update_tracking_button()
            self.update_camera_feed()

        else:
            self.update_tracking_button()

            self.update_status(
                f"Status: Failed to switch to {new_source}",
                "● ERROR"
            )

            messagebox.showerror(
                "Camera Switch Error",
                f"Could not switch to {new_source}.\n\n"
                "Try opening Settings and selecting the other source manually."
            )

    # --------------------------------------------------
    # Photo and Video
    # --------------------------------------------------

    def take_photo(self):
        """
        Take one photo from the current camera frame.
        """

        # Camera must be active to take a photo.
        if not self.camera.is_tracking:
            messagebox.showwarning(
                "Camera Not Active",
                "Start tracking before taking a photo."
            )
            return

        # Capture photo through camera backend.
        filepath = self.camera.take_photo()
        thermal_filepath = self.thermal_camera.take_photo()

        if filepath:
            if thermal_filepath:
                status_message = (
                    f"Status: Photo saved to {filepath}; "
                    f"thermal photo saved to {thermal_filepath}"
                )
            else:
                status_message = f"Status: Photo saved to {filepath}"

            self.update_status(
                status_message,
                "● LIVE"
            )
        else:
            messagebox.showerror(
                "Photo Error",
                "Could not capture photo."
            )

    def toggle_recording(self):
        """
        Start or stop video recording with audio.
        """

        if self.pulse_sequence_active:
            messagebox.showinfo(
                "Pulse Sequence Active",
                "Use 'Stop Tracking' to stop the repeated-pulse session."
            )
            return

        # Camera must be active before recording.
        if not self.camera.is_tracking:
            messagebox.showwarning(
                "Camera Not Active",
                "Start tracking before recording video."
            )
            return

        # Start recording.
        if not self.camera.is_recording:
            self.start_active_recordings()

            if self.current_video_path:
                self.record_button.config(
                    text="Stop Recording",
                    style="VideoStop.TButton"
                )

                self.update_status(
                    "Status: Recording regular and thermal video/audio",
                    "● REC"
                )

                if not self.current_thermal_video_path:
                    self.update_status(
                        "Status: Regular recording active; thermal recording unavailable",
                        "● WARNING"
                    )
            else:
                messagebox.showerror(
                    "Recording Error",
                    "Could not start video recording."
                )

        # Stop recording.
        else:
            saved_video_path, saved_thermal_video_path, _ = (
                self.finalize_capture_session()
            )

            self.record_button.config(
                text="Capture Video",
                style="Capture.TButton"
            )

            self.record_timer_label.config(text="Recording stopped")

            if saved_video_path:
                self.current_video_path = saved_video_path

                if saved_thermal_video_path:
                    self.current_thermal_video_path = saved_thermal_video_path

                self.update_status(
                    "Status: Regular and thermal videos saved with audio",
                    "● LIVE"
                )
            else:
                self.update_status(
                    "Status: Recording stopped, but file could not be saved",
                    "● WARNING"
                )

    # --------------------------------------------------
    # Zoom
    # --------------------------------------------------

    def zoom_in(self):
        """
        Increase digital zoom.
        """

        self.camera.zoom_in()
        self.refresh_info_panel()

        self.update_status(
            f"Status: Zoom set to {self.camera.zoom_factor}x",
            "● LIVE" if self.camera.is_tracking else "● READY"
        )

    def zoom_out(self):
        """
        Decrease digital zoom.
        """

        self.camera.zoom_out()
        self.refresh_info_panel()

        self.update_status(
            f"Status: Zoom set to {self.camera.zoom_factor}x",
            "● LIVE" if self.camera.is_tracking else "● READY"
        )

    def reset_zoom(self):
        """
        Reset digital zoom to normal.
        """

        self.camera.reset_zoom()
        self.refresh_info_panel()

        self.update_status(
            "Status: Zoom reset to 1.0x",
            "● LIVE" if self.camera.is_tracking else "● READY"
        )

    # --------------------------------------------------
    # Close App
    # --------------------------------------------------

    def on_close(self):
        """
        Safely close app.
        """

        self.is_closing = True

        # Stop scheduled preview loop.
        self.cancel_preview_loop()
        self.cancel_thermal_preview_loop()

        # Stop pulse playback/recording if app closes during a sequence.
        with self.pulse_process_lock:
            self.tracking_start_token += 1
        self.pulse_sequence_active = False
        self.stop_pulse_sequence_process()

        # Finalize any active recordings before releasing camera resources.
        session_audio_path = self.build_pulse_session_audio()
        self.stop_active_recordings(audio_path_override=session_audio_path)

        # Release camera and recording resources.
        self.camera.stop_camera()
        self.thermal_camera.stop_camera()

        # Close Tkinter window.
        self.root.destroy()


if __name__ == "__main__":
    # Create main Tkinter window.
    root = tk.Tk(className=APP_WM_CLASS)

    # Create CoBas app instance.
    app = CoBasV1App(root)

    # Make sure camera resources are released when the user closes the app.
    root.protocol("WM_DELETE_WINDOW", app.on_close)

    # Start Tkinter event loop.
    root.mainloop()
