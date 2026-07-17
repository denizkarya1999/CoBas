import tkinter as tk
from tkinter import ttk, messagebox
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
from Camera.ThermalCamera import ThermalCamera
from Style import COLORS, FONTS, WINDOW, PREVIEW, SPACING, apply_styles
from Settings import SettingsWindow
from About import show_about_window


APP_TITLE = "CoBas Battery Reader V1.0"
APP_WM_CLASS = "cobas_battery_reader_v1"
ICON_SIZES = (16, 24, 32, 48, 64, 128, 256)


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

        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.captures_dir = os.path.join(self.base_dir, "Captures")

        # Camera backends.
        self.camera = Camera(
            camera_index="/dev/video0",
            output_dir=self.captures_dir
        )
        self.thermal_camera = ThermalCamera(output_dir=self.captures_dir)

        # Stores path of the current video file.
        self.current_video_path = None
        self.current_thermal_video_path = None
        self.current_voice_path = None

        # Stores last processed video path to avoid processing the same video twice.
        self.last_processed_video_path = None

        # Background pulse protocol generation process.
        self.pulse_process = None
        self.is_preparing_tracking = False
        self.tracking_start_token = 0
        self.pulse_playback_end_time = None

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

        # Apply styles from Style.py.
        self.style = apply_styles(self.root)

        # Build all GUI components.
        self.build_gui()

        self.update_status(
            "Status: Ready. Click 'Start Tracking' to begin.",
            "● READY"
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

        for label, value in (("Regular", "rgb"), ("Greyscale", "grayscale")):
            button = ttk.Radiobutton(
                thermal_scale_controls,
                text=label,
                value=value,
                variable=self.thermal_scale_mode,
                command=self.change_thermal_scale_mode,
                style="ThermalScale.TRadiobutton"
            )
            button.pack(side="left", padx=(4, 0))
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
            text="Recording: 0 second",
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
            text="FPS: 20",
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

    def refresh_thermal_scale_controls(self):
        """Lock the palette while a thermal recording is in progress."""
        state = "disabled" if self.thermal_camera.is_recording else "normal"
        for button in self.thermal_scale_buttons:
            button.configure(state=state)

        if self.thermal_scale_mode.get() != self.thermal_camera.display_mode:
            self.thermal_scale_mode.set(self.thermal_camera.display_mode)

    def start_active_recordings(self):
        """Start the single synchronized camera and thermal recording pipeline."""
        if self.camera.is_recording or self.thermal_camera.is_recording:
            print("[INFO] A synchronized recording is already active.")
            return self.current_video_path, self.current_thermal_video_path

        self.set_recording_volume_to_maximum()
        recording_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.current_video_path = None
        self.current_thermal_video_path = None
        self.current_voice_path = None

        self.current_video_path = self.camera.start_recording(
            timestamp=recording_timestamp
        )

        if self.current_video_path:
            self.current_thermal_video_path = self.thermal_camera.start_recording(
                timestamp=recording_timestamp
            )
            self.refresh_thermal_scale_controls()

        return self.current_video_path, self.current_thermal_video_path

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

    def stop_active_recordings(self):
        """
        Stop regular and thermal recorders, reusing the single captured WAV.

        Capture is stopped first on both recorders so video durations stay
        synchronized before FFmpeg merge/finalization begins.
        """

        saved_video_path = None
        saved_thermal_video_path = None
        saved_voice_path = None
        audio_path = None

        regular_was_recording = self.camera.is_recording
        thermal_was_recording = self.thermal_camera.is_recording

        if regular_was_recording:
            audio_path = self.camera.stop_recording_capture_phase()

        if thermal_was_recording:
            saved_thermal_video_path = self.thermal_camera.stop_recording(
                audio_path=audio_path
            )
            self.refresh_thermal_scale_controls()

            if saved_thermal_video_path:
                self.current_thermal_video_path = saved_thermal_video_path

        if regular_was_recording:
            saved_video_path = self.camera.finalize_recording(
                audio_path=audio_path,
                keep_audio=True
            )
            saved_voice_path = self.camera.last_saved_voice_path

            if saved_video_path:
                self.current_video_path = saved_video_path

            if saved_voice_path:
                self.current_voice_path = saved_voice_path

        if saved_video_path and saved_thermal_video_path:
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

    def finalize_capture_session(self):
        """Stop and export one capture session through the single output pipeline."""
        saved_paths = self.stop_active_recordings()
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
        Read video duration in seconds using ffprobe.
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
            "-show_entries", "format=duration",
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

    def sync_video_duration(self, video_path, target_duration_seconds):
        """
        Force video duration to match target duration exactly.

        The regular camera writes frames from the Tkinter preview loop, so its
        encoded duration can be shorter than real capture time if the loop runs
        below the configured FPS. Retiming spreads the captured frames across
        the thermal duration instead of freezing the last frame.
        """

        if not video_path or not os.path.exists(video_path):
            return video_path

        if target_duration_seconds is None or target_duration_seconds <= 0:
            return video_path

        current_duration = self.get_video_duration_seconds(video_path)

        if current_duration is None:
            return video_path

        if abs(current_duration - target_duration_seconds) <= 0.05:
            return video_path

        ffmpeg = shutil.which("ffmpeg")

        if ffmpeg is None:
            print("[WARNING] ffmpeg is not installed; cannot synchronize video duration.")
            return video_path

        target_text = f"{target_duration_seconds:.3f}"
        duration_scale = target_duration_seconds / current_duration
        scale_text = f"{duration_scale:.9f}"

        base, ext = os.path.splitext(video_path)
        temp_output_path = f"{base}_sync_tmp{ext}"
        has_audio = self.video_has_audio_stream(video_path)

        command = [
            ffmpeg,
            "-y",
            "-i", video_path,
            "-vf", f"setpts={scale_text}*PTS,trim=duration={target_text},setpts=PTS-STARTPTS",
            "-t", target_text,
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-pix_fmt", "yuv420p",
        ]

        if has_audio:
            command.extend(
                [
                    "-map", "0:v:0",
                    "-map", "0:a:0",
                    "-af", f"apad,atrim=duration={target_text}",
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
                "[INFO] Regular video duration synchronized "
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
            text=f"FPS: {self.camera.record_fps}"
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
            shutil.move(source_path, output_path)

    def generate_capture_outputs(
        self,
        video_path,
        thermal_video_path,
        scale_video_path,
        voice_path,
        output_folder
    ):
        """Generate the six requested capture artifacts without an external script."""
        if os.path.isdir(output_folder):
            shutil.rmtree(output_folder)

        camera_frames = os.path.join(output_folder, "Camera_Frames")
        thermal_frames = os.path.join(output_folder, "Thermal_Frames")
        os.makedirs(camera_frames, exist_ok=True)
        os.makedirs(thermal_frames, exist_ok=True)

        self.run_capture_ffmpeg(
            "camera frame extraction",
            [
                "ffmpeg",
                "-y",
                "-i",
                video_path,
                "-vf",
                "fps=0.5",
                "-start_number",
                "0",
                "-qscale:v",
                "2",
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
                "fps=0.5",
                "-start_number",
                "0",
                "-qscale:v",
                "2",
                os.path.join(thermal_frames, "Thermal_Frame_%03d.jpg"),
            ]
        )

        voice_output = os.path.join(output_folder, "Voice.wav")
        if voice_path and os.path.exists(voice_path):
            self.move_capture_output(voice_path, voice_output)
        else:
            self.run_capture_ffmpeg(
                "single voice extraction",
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    video_path,
                    "-vn",
                    "-ar",
                    "48000",
                    "-ac",
                    "1",
                    "-c:a",
                    "pcm_s16le",
                    voice_output,
                ]
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
        )
        for source_path, output_path in capture_moves:
            self.move_capture_output(source_path, output_path)

    def export_capture_session(self, video_path, thermal_video_path=None):
        """
        Export the six capture artifacts from the latest recording.

        Outputs: camera frames, thermal frames, one voice, camera video,
        thermal video, and the separately recorded thermal-range video.
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
        voice_path = self.current_voice_path

        required_outputs = (
            (thermal_video_path, "thermal video"),
            (scale_video_path, "thermal-range video"),
        )
        for required_path, label in required_outputs:
            if not required_path or not os.path.exists(required_path):
                print(f"[WARNING] {label.title()} is unavailable: {required_path}")
                self.update_status(
                    f"Status: {label.title()} is unavailable",
                    "● WARNING"
                )
                return

        video_stem = os.path.splitext(os.path.basename(video_path))[0]
        output_folder = os.path.join(
            self.captures_dir,
            f"{video_stem}_Image_and_Video"
        )
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
                    voice_path,
                    output_folder
                )
                print(f"[INFO] Capture outputs saved in: {output_folder}")

                self.root.after(
                    0,
                    lambda: self.handle_capture_outputs_finished(output_folder)
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

    def handle_capture_outputs_finished(self, output_folder):
        """Report successful capture-only output generation."""

        self.set_processing_indicator(False)
        self.refresh_info_panel()

        self.update_status(
            f"Status: Capture outputs saved in {output_folder}",
            "● IDLE"
        )
        messagebox.showinfo(
            "Capture Outputs Saved",
            "Generated outputs:\n"
            "• Camera_Frames\n"
            "• Thermal_Frames\n"
            "• Voice.wav\n"
            "• Camera_Video.mp4\n"
            "• Thermal_Video.mp4\n"
            "• Thermal_Range_Video.mp4\n\n"
            f"Saved in: {output_folder}"
        )

    def get_pulse_protocol_command(self, mode):
        """
        Return pulse protocol command and working folder for the given mode.
        """

        pulse_folder = os.path.join(
            self.base_dir,
            "Pulse Generation"
        )
        pulse_script = os.path.join(
            pulse_folder,
            "pulse_protocol_generator.py"
        )

        if not os.path.exists(pulse_script):
            print(f"[WARNING] Pulse protocol generator not found: {pulse_script}")
            return None, None

        return [sys.executable, pulse_script, "--mode", mode], pulse_folder

    def get_pulse_protocol_audio_path(self):
        """
        Return the generated pulse protocol WAV path.
        """

        return os.path.join(
            self.base_dir,
            "Pulse Generation",
            "Inputs",
            "5_15sPause_BeaconProtocol.wav"
        )

    def get_pulse_protocol_duration_seconds(self):
        """
        Return the generated pulse protocol audio duration in seconds.
        """

        audio_path = self.get_pulse_protocol_audio_path()

        if not os.path.exists(audio_path):
            return None

        try:
            with wave.open(audio_path, "rb") as wf:
                return wf.getnframes() / wf.getframerate()

        except Exception as e:
            print(f"[WARNING] Could not read pulse protocol duration: {e}")
            return None

    def generate_pulse_protocol_audio(self):
        """
        Generate pulse protocol audio before tracking starts.
        """

        command, pulse_folder = self.get_pulse_protocol_command("generate-only")

        if command is None:
            return False

        try:
            subprocess.run(
                command,
                cwd=pulse_folder,
                check=True
            )
            print("[INFO] Pulse protocol audio generated.")
            return True

        except Exception as e:
            print(f"[WARNING] Could not generate pulse protocol audio: {e}")
            return False

    def start_pulse_protocol_playback(self):
        """
        Start playback for the already generated pulse protocol audio.
        """

        if self.pulse_process is not None and self.pulse_process.poll() is None:
            print("[INFO] Pulse protocol playback is already running.")
            return self.pulse_process

        command, pulse_folder = self.get_pulse_protocol_command("play-existing")

        if command is None:
            return None

        try:
            self.pulse_process = subprocess.Popen(
                command,
                cwd=pulse_folder,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1
            )
            print("[INFO] Pulse protocol playback started.")
            return self.pulse_process

        except Exception as e:
            self.pulse_process = None
            print(f"[WARNING] Could not start pulse protocol playback: {e}")
            return None

    def start_tracking_after_pulse_generation(self, start_token):
        """
        Generate audio first, then play it and start tracking as playback begins.
        """

        def worker():
            generated = self.generate_pulse_protocol_audio()

            self.root.after(
                0,
                lambda: self.start_pulse_playback_and_tracking(
                    generated,
                    start_token
                )
            )

        threading.Thread(
            target=worker,
            daemon=True
        ).start()

    def start_pulse_playback_and_tracking(self, generated, start_token):
        """
        Start pulse playback, then wait for playback start before tracking.
        """

        self.is_preparing_tracking = False

        if start_token != self.tracking_start_token:
            return

        if not generated:
            self.track_button.config(state="normal")
            self.update_tracking_button()
            self.update_status(
                "Status: Pulse protocol could not be generated",
                "● WARNING"
            )
            return

        self.update_status(
            "Status: Pulse audio generated. Starting playback...",
            "● STARTING"
        )

        pulse_process = self.start_pulse_protocol_playback()

        if pulse_process is None:
            self.track_button.config(state="normal")
            self.update_tracking_button()
            self.update_status(
                "Status: Pulse protocol could not start; tracking not started",
                "● WARNING"
            )
            return

        def worker():
            playback_started = False

            for line in pulse_process.stdout:
                if "PLAYBACK_STARTED" in line:
                    playback_started = True
                    self.root.after(
                        0,
                        lambda: self.begin_tracking_capture(
                            pulse_process,
                            start_token
                        )
                    )
                    break

            if not playback_started:
                self.root.after(
                    0,
                    lambda: self.handle_pulse_playback_start_failed(pulse_process)
                )
                return

            pulse_process.wait()
            self.root.after(
                0,
                lambda: self.handle_pulse_playback_finished(pulse_process)
            )

        threading.Thread(
            target=worker,
            daemon=True
        ).start()

    def handle_pulse_playback_start_failed(self, pulse_process):
        """
        Reset UI if playback exits before confirming it started.
        """

        if self.pulse_process is not pulse_process:
            return

        self.pulse_process = None
        self.track_button.config(state="normal")
        self.update_tracking_button()
        self.update_status(
            "Status: Pulse playback failed before tracking started",
            "● WARNING"
        )

    def handle_pulse_playback_finished(self, pulse_process):
        """
        Stop automatic recording and tracking after pulse playback ends.
        """

        if self.pulse_process is not pulse_process:
            return

        self.pulse_process = None
        self.pulse_playback_end_time = None

        if not self.camera.is_tracking:
            return

        self.cancel_preview_loop()

        saved_video_path, saved_thermal_video_path, saved_voice_path = self.finalize_capture_session()

        self.camera.stop_camera()
        self.thermal_camera.stop_camera()
        self.cancel_thermal_preview_loop()

        self.video_label.config(
            image="",
            text="Timed recording complete.\n\nClick 'Start Tracking' to start again.",
            bg=COLORS["preview_bg"],
            fg=COLORS["muted_text"]
        )
        self.video_label.image = None

        self.thermal_video_label.config(
            image="",
            text="Timed thermal recording complete.\n\nClick 'Start Tracking' to start again.",
            bg=COLORS["preview_bg"],
            fg=COLORS["muted_text"]
        )
        self.thermal_video_label.image = None

        self.record_button.config(
            text="Capture Video",
            style="Capture.TButton"
        )
        self.record_timer_label.config(text="Recording: 0 second")
        self.track_button.config(state="normal")
        self.update_tracking_button()
        self.refresh_info_panel()

        if saved_video_path:
            status_message = f"Status: Timed recording complete. Video saved to {saved_video_path}"
            if saved_voice_path:
                status_message = f"{status_message}; voice saved to {saved_voice_path}"

            self.update_status(status_message, "● IDLE")
        else:
            self.update_status(
                "Status: Timed recording complete",
                "● IDLE"
            )

    def stop_pulse_protocol_generation(self):
        """
        Stop pulse protocol generation if it is still running.
        """

        if self.pulse_process is None:
            return

        if self.pulse_process.poll() is not None:
            self.pulse_process = None
            return

        try:
            self.pulse_process.terminate()
            self.pulse_process.wait(timeout=2)
            print("[INFO] Pulse protocol generation stopped.")

        except subprocess.TimeoutExpired:
            self.pulse_process.kill()
            self.pulse_process.wait(timeout=2)
            print("[INFO] Pulse protocol generation killed.")

        except Exception as e:
            print(f"[WARNING] Could not stop pulse protocol generation: {e}")

        finally:
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
        Start camera preview safely.
        """

        print("Start Tracking clicked")

        # Prevent duplicate preview loops before starting.
        self.cancel_preview_loop()
        self.cancel_thermal_preview_loop()

        self.is_preparing_tracking = True
        self.tracking_start_token += 1
        start_token = self.tracking_start_token
        self.current_video_path = None
        self.current_thermal_video_path = None
        self.current_voice_path = None

        self.start_thermal_camera_feed()

        self.track_button.config(
            text="Preparing...",
            style="Start.TButton",
            state="disabled"
        )

        self.video_label.config(
            image="",
            text="Generating pulse audio...\n\nTracking will start when playback starts.",
            bg=COLORS["preview_bg"],
            fg=COLORS["muted_text"]
        )
        self.video_label.image = None

        self.thermal_video_label.config(
            image="",
            text="Preparing thermal camera...\n\nRecording starts with pulse playback.",
            bg=COLORS["preview_bg"],
            fg=COLORS["muted_text"]
        )
        self.thermal_video_label.image = None

        self.update_status(
            "Status: Generating pulse audio and preparing thermal camera...",
            "● STARTING"
        )

        self.start_tracking_after_pulse_generation(start_token)

    def begin_tracking_capture(self, pulse_process, start_token):
        """
        Start camera tracking and recording after pulse playback starts.
        """

        if start_token != self.tracking_start_token:
            return

        if self.pulse_process is not pulse_process:
            return

        self.track_button.config(state="normal")
        pulse_duration_seconds = self.get_pulse_protocol_duration_seconds()

        if pulse_duration_seconds is not None:
            self.pulse_playback_end_time = time.time() + pulse_duration_seconds
        else:
            self.pulse_playback_end_time = None

        self.update_status(
            "Status: Pulse playback started. Starting camera...",
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

            # Automatically start the single synchronized recording pipeline.
            self.start_active_recordings()

            if self.current_video_path:
                self.record_button.config(
                    text="Stop Recording",
                    style="VideoStop.TButton"
                )
                self.update_status(
                    "Status: Recording regular and thermal video during pulse playback",
                    "● REC"
                )

                if not self.current_thermal_video_path:
                    print("[WARNING] Thermal recording did not start.")
                    self.update_status(
                        "Status: Regular recording active; thermal recording unavailable",
                        "● WARNING"
                    )
            else:
                print("[WARNING] Could not start automatic video recording.")
                self.record_button.config(
                    text="Capture Video",
                    style="Capture.TButton"
                )
                self.update_status(
                    "Status: Failed to start recording during pulse playback",
                    "● WARNING"
                )

            # Start refreshing frames after recording is initialized.
            self.update_camera_feed()

        else:
            self.stop_pulse_protocol_generation()
            self.thermal_camera.stop_camera()
            self.cancel_thermal_preview_loop()

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
        self.tracking_start_token += 1
        self.pulse_playback_end_time = None
        self.track_button.config(state="normal")

        # Cancel preview update loop first.
        self.cancel_preview_loop()
        self.cancel_thermal_preview_loop()

        # Stop pulse protocol generation alongside recording/tracking.
        self.stop_pulse_protocol_generation()

        saved_video_path, saved_thermal_video_path, saved_voice_path = self.finalize_capture_session()

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
            style="Capture.TButton"
        )
        self.record_timer_label.config(text="Recording: 0 second")

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
                "Status: Camera opened, but no frame received",
                "● WARNING"
            )

        # Update recording timer while recording.
        if self.camera.is_recording:
            if self.pulse_playback_end_time is not None:
                remaining = max(
                    0,
                    int(self.pulse_playback_end_time - time.time() + 0.999)
                )
                second_text = "second" if remaining == 1 else "seconds"
                self.record_timer_label.config(
                    text=f"Pulse remaining: {remaining} {second_text}"
                )
            else:
                seconds = self.camera.get_recording_seconds()
                second_text = "second" if seconds == 1 else "seconds"
                self.record_timer_label.config(
                    text=f"Recording: {seconds} {second_text}"
                )

        # Schedule next frame update.
        self.preview_after_id = self.root.after(30, self.update_camera_feed)

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
            saved_video_path, saved_thermal_video_path, saved_voice_path = self.finalize_capture_session()

            self.record_button.config(
                text="Capture Video",
                style="Capture.TButton"
            )

            self.record_timer_label.config(text="Recording: 0 second")

            if saved_video_path:
                self.current_video_path = saved_video_path

                if saved_thermal_video_path:
                    self.current_thermal_video_path = saved_thermal_video_path

                if saved_voice_path:
                    self.update_status(
                        f"Status: Regular and thermal videos saved with voice at {saved_voice_path}",
                        "● LIVE"
                    )
                else:
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

        # Stop pulse generation if app closes while tracking.
        self.stop_pulse_protocol_generation()

        # Finalize any active recordings before releasing camera resources.
        self.stop_active_recordings()

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
