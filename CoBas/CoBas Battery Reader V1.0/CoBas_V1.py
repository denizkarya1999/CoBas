import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2

from Camera import Camera
from Style import COLORS, FONTS, WINDOW, PREVIEW, SPACING, apply_styles
from Settings import SettingsWindow
from About import show_about_window


class CoBasV1App:
    """
    Main GUI application for CoBas_V1.

    This file handles:
    - Main app window
    - Toolbar
    - Camera preview
    - Tracking controls
    - Front/back camera switching
    - Capture controls
    - Zoom controls
    - Status messages
    - Microphone status display

    Separated files:
    - Camera.py handles camera and microphone operations
    - Style.py handles GUI styling
    - Settings.py handles camera and microphone source selection
    - About.py handles the About popup
    """

    def __init__(self, root):
        self.root = root
        self.root.title("CoBas_V1")

        # Fixed window size from Style.py.
        self.window_width = WINDOW["width"]
        self.window_height = WINDOW["height"]

        self.root.geometry(f"{self.window_width}x{self.window_height}")
        self.root.resizable(False, False)
        self.root.minsize(self.window_width, self.window_height)
        self.root.maxsize(self.window_width, self.window_height)

        # Restore app if minimized.
        self.is_closing = False
        self.root.bind("<Unmap>", self.prevent_minimize)

        # Camera backend.
        # The camera backend also stores the selected microphone.
        self.camera = Camera(camera_index="/dev/video0")

        # App state.
        self.current_video_path = None
        self.preview_loop_running = False
        self.preview_photo = None

        # Apply external GUI style.
        self.style = apply_styles(self.root)

        # Build GUI.
        self.build_gui()

        # Start camera automatically after GUI loads.
        self.root.after(700, self.auto_start_camera)

    # --------------------------------------------------
    # Window Behavior
    # --------------------------------------------------

    def prevent_minimize(self, event):
        """
        Restore the window if minimized.
        """

        if self.is_closing:
            return

        try:
            if self.root.state() == "iconic":
                self.root.after(100, self.root.deiconify)
        except tk.TclError:
            pass

    # --------------------------------------------------
    # GUI Layout
    # --------------------------------------------------

    def build_gui(self):
        """
        Build the compact dashboard layout.
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

        # Header.
        header_frame = ttk.Frame(main_frame, style="Main.TFrame")
        header_frame.pack(fill="x", pady=(0, 6))

        ttk.Label(
            header_frame,
            text="CoBas_V1",
            style="Header.TLabel"
        ).pack(anchor="w")

        ttk.Label(
            header_frame,
            text="Camera-based battery reader prototype",
            style="SubHeader.TLabel"
        ).pack(anchor="w")

        # Main content.
        content_frame = ttk.Frame(main_frame, style="Main.TFrame")
        content_frame.pack(fill="both", expand=True)

        content_frame.columnconfigure(0, weight=3)
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
        Build top toolbar.

        Settings and About are opened from separate files:
        - Settings.py
        - About.py
        """

        toolbar = ttk.Frame(parent, style="Toolbar.TFrame")
        toolbar.pack(fill="x")

        left_toolbar = ttk.Frame(toolbar, style="Toolbar.TFrame")
        left_toolbar.pack(
            side="left",
            padx=SPACING["toolbar_padx"],
            pady=SPACING["toolbar_pady"]
        )

        ttk.Label(
            left_toolbar,
            text="CoBas_V1",
            style="ToolbarTitle.TLabel"
        ).pack(side="left")

        ttk.Label(
            left_toolbar,
            text=" | Battery Camera Reader",
            style="ToolbarText.TLabel"
        ).pack(side="left")

        right_toolbar = ttk.Frame(toolbar, style="Toolbar.TFrame")
        right_toolbar.pack(
            side="right",
            padx=SPACING["toolbar_padx"],
            pady=SPACING["toolbar_pady"]
        )

        ttk.Button(
            right_toolbar,
            text="Settings",
            style="Toolbar.TButton",
            command=self.open_settings
        ).pack(side="left", padx=3)

        ttk.Button(
            right_toolbar,
            text="About",
            style="Toolbar.TButton",
            command=self.open_about
        ).pack(side="left", padx=3)

        ttk.Button(
            right_toolbar,
            text="Exit",
            style="Toolbar.TButton",
            command=self.on_close
        ).pack(side="left", padx=3)

    def build_preview_panel(self, parent):
        """
        Build compact camera preview panel.
        """

        preview_header = ttk.Frame(parent, style="Panel.TFrame")
        preview_header.pack(
            fill="x",
            padx=SPACING["panel_padx"],
            pady=(10, 5)
        )

        ttk.Label(
            preview_header,
            text="Live Camera Preview",
            style="PanelTitle.TLabel"
        ).pack(side="left")

        self.live_indicator_label = ttk.Label(
            preview_header,
            text="● STARTING",
            style="Status.TLabel"
        )
        self.live_indicator_label.pack(side="right")

        self.video_label = tk.Label(
            parent,
            text="Initializing camera...\n\nPlease wait.",
            bg=COLORS["preview_bg"],
            fg=COLORS["muted_text"],
            font=FONTS["preview_text"],
            bd=0,
            relief="flat"
        )

        self.video_label.pack(
            fill="both",
            expand=True,
            padx=SPACING["panel_padx"],
            pady=(0, 8)
        )

        bottom_bar = ttk.Frame(parent, style="Panel.TFrame")
        bottom_bar.pack(
            fill="x",
            padx=SPACING["panel_padx"],
            pady=(0, 10)
        )

        self.status_label = ttk.Label(
            bottom_bar,
            text="Status: Initializing camera...",
            style="Info.TLabel"
        )
        self.status_label.pack(side="left")

        self.record_timer_label = ttk.Label(
            bottom_bar,
            text="Recording: 0 second",
            style="Info.TLabel"
        )
        self.record_timer_label.pack(side="right")

    def build_control_panel(self, parent):
        """
        Build compact right control panel.

        Camera source and microphone source selection are handled inside Settings.py.
        """

        # --------------------------------------------------
        # Tracking controls
        # --------------------------------------------------

        controls_section = ttk.Frame(parent, style="Panel.TFrame")
        controls_section.pack(
            fill="x",
            padx=SPACING["panel_padx"],
            pady=(10, 6)
        )

        ttk.Label(
            controls_section,
            text="Tracking",
            style="PanelTitle.TLabel"
        ).pack(anchor="w", pady=(0, 4))

        self.track_button = ttk.Button(
            controls_section,
            text="Restart Tracking",
            style="Primary.TButton",
            command=self.track_battery
        )
        self.track_button.pack(
            fill="x",
            pady=SPACING["button_pady"]
        )

        self.stop_button = ttk.Button(
            controls_section,
            text="Stop Tracking",
            style="Danger.TButton",
            command=self.stop_tracking
        )
        self.stop_button.pack(
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
            text="Camera Direction",
            style="PanelTitle.TLabel"
        ).pack(anchor="w", pady=(0, 4))

        self.switch_camera_button = ttk.Button(
            switch_section,
            text="Switch Front/Back Camera",
            style="Tool.TButton",
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
        ).pack(anchor="w", pady=(0, 4))

        self.photo_button = ttk.Button(
            capture_section,
            text="Take Photo",
            style="Tool.TButton",
            command=self.take_photo
        )
        self.photo_button.pack(
            fill="x",
            pady=SPACING["button_pady"]
        )

        self.record_button = ttk.Button(
            capture_section,
            text="Record Video",
            style="Tool.TButton",
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
        ).pack(anchor="w", pady=(0, 4))

        zoom_buttons_frame = ttk.Frame(zoom_section, style="Panel.TFrame")
        zoom_buttons_frame.pack(fill="x")

        ttk.Button(
            zoom_buttons_frame,
            text="Out",
            style="Tool.TButton",
            command=self.zoom_out
        ).pack(side="left", fill="x", expand=True, padx=(0, 3))

        ttk.Button(
            zoom_buttons_frame,
            text="In",
            style="Tool.TButton",
            command=self.zoom_in
        ).pack(side="left", fill="x", expand=True, padx=(3, 0))

        ttk.Button(
            zoom_section,
            text="Reset Zoom",
            style="Tool.TButton",
            command=self.reset_zoom
        ).pack(fill="x", pady=(5, 0))

        self.zoom_label = ttk.Label(
            zoom_section,
            text="Zoom: 1.0x",
            style="PanelText.TLabel"
        )
        self.zoom_label.pack(anchor="center", pady=(5, 0))

        # --------------------------------------------------
        # System info
        # --------------------------------------------------

        info_section = ttk.Frame(parent, style="Panel.TFrame")
        info_section.pack(
            fill="both",
            expand=True,
            padx=SPACING["panel_padx"],
            pady=(6, 10)
        )

        ttk.Label(
            info_section,
            text="System Info",
            style="PanelTitle.TLabel"
        ).pack(anchor="w", pady=(0, 4))

        self.camera_info_label = ttk.Label(
            info_section,
            text="Camera: /dev/video0",
            style="PanelText.TLabel",
            wraplength=190
        )
        self.camera_info_label.pack(anchor="w", pady=1)

        self.microphone_info_label = ttk.Label(
            info_section,
            text="Microphone: System Default Microphone",
            style="PanelText.TLabel",
            wraplength=190
        )
        self.microphone_info_label.pack(anchor="w", pady=1)

        self.output_info_label = ttk.Label(
            info_section,
            text="Output: captures/",
            style="PanelText.TLabel",
            wraplength=190
        )
        self.output_info_label.pack(anchor="w", pady=1)

        self.fps_info_label = ttk.Label(
            info_section,
            text="Recording FPS: 20",
            style="PanelText.TLabel",
            wraplength=190
        )
        self.fps_info_label.pack(anchor="w", pady=1)

        self.camera_direction_label = ttk.Label(
            info_section,
            text="Switching: /dev/video0 ↔ /dev/video1",
            style="PanelText.TLabel",
            wraplength=190
        )
        self.camera_direction_label.pack(anchor="w", pady=1)

    # --------------------------------------------------
    # Helper Methods
    # --------------------------------------------------

    def update_status(self, message, indicator_text=None):
        """
        Update bottom status label and live indicator.
        """

        self.status_label.config(text=message)

        if indicator_text is not None:
            self.live_indicator_label.config(text=indicator_text)

    def refresh_info_panel(self):
        """
        Refresh system information labels.
        """

        self.camera_info_label.config(
            text=f"Camera: {self.camera.camera_index}"
        )

        self.microphone_info_label.config(
            text=f"Microphone: {self.camera.microphone_device_name}"
        )

        self.output_info_label.config(
            text=f"Output: {self.camera.output_dir}/"
        )

        self.fps_info_label.config(
            text=f"Recording FPS: {self.camera.record_fps}"
        )

        self.zoom_label.config(
            text=f"Zoom: {self.camera.zoom_factor}x"
        )

    def auto_start_camera(self):
        """
        Start camera automatically.
        """

        self.track_battery()

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

        if selected_source in ["0", "1"]:
            selected_source = int(selected_source)

        was_tracking = self.camera.is_tracking

        if was_tracking:
            self.stop_tracking()

        self.camera.set_camera_source(selected_source)

        self.camera_info_label.config(
            text=f"Camera: {self.camera.camera_index}"
        )

        self.update_status(
            f"Status: Camera source set to {self.camera.camera_index}",
            "● READY"
        )

        # Restart automatically with the new camera source.
        self.root.after(300, self.track_battery)

    def apply_microphone_source_from_settings(
        self,
        microphone_device_id,
        microphone_device_name
    ):
        """
        Called by Settings.py when the user applies a new microphone source.
        """

        if self.camera.is_recording:
            messagebox.showwarning(
                "Recording Active",
                "Stop recording before changing the microphone."
            )
            return

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
    # Camera Actions
    # --------------------------------------------------

    def track_battery(self):
        """
        Start or restart camera preview.
        """

        print("Track Battery / Restart Tracking clicked")

        if self.camera.is_tracking:
            self.camera.stop_camera()
            self.preview_loop_running = False

        self.update_status(
            "Status: Starting camera...",
            "● STARTING"
        )

        started = self.camera.start_camera()

        if started:
            self.refresh_info_panel()

            self.update_status(
                f"Status: Tracking battery using {self.camera.camera_index}",
                "● LIVE"
            )

            if not self.preview_loop_running:
                self.preview_loop_running = True
                self.update_camera_feed()

        else:
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
        Stop camera preview.
        """

        self.camera.stop_camera()
        self.preview_loop_running = False

        self.video_label.config(
            image="",
            text="Tracking stopped.\n\nClick 'Restart Tracking' to start again.",
            bg=COLORS["preview_bg"],
            fg=COLORS["muted_text"]
        )
        self.video_label.image = None

        self.record_button.config(text="Record Video")
        self.record_timer_label.config(text="Recording: 0 second")

        self.update_status(
            "Status: Tracking stopped",
            "● IDLE"
        )

        self.refresh_info_panel()

    def update_camera_feed(self):
        """
        Continuously update live camera preview.
        """

        if not self.camera.is_tracking:
            self.preview_loop_running = False
            return

        frame = self.camera.read_frame()

        if frame is not None:
            self.camera.write_video_frame(frame)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)

            image = image.resize((PREVIEW["width"], PREVIEW["height"]))

            self.preview_photo = ImageTk.PhotoImage(image=image)

            self.video_label.config(
                image=self.preview_photo,
                text=""
            )

            self.video_label.image = self.preview_photo

        else:
            self.update_status(
                "Status: Camera opened, but no frame received",
                "● WARNING"
            )

        if self.camera.is_recording:
            seconds = self.camera.get_recording_seconds()
            self.record_timer_label.config(
                text=f"Recording: {seconds} second(s)"
            )

        self.root.after(30, self.update_camera_feed)

    def switch_front_back_camera(self):
        """
        Switch between /dev/video0 and /dev/video1.

        This is intended for Android Webcam mode, where one device node
        may represent the front camera and the other may represent the back camera.
        """

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
            self.camera.stop_camera()
            self.preview_loop_running = False

        # Toggle camera source.
        new_source = self.camera.switch_camera_source()

        self.camera_info_label.config(
            text=f"Camera: {new_source}"
        )

        # Start camera again.
        started = self.camera.start_camera()

        if started:
            self.refresh_info_panel()

            self.update_status(
                f"Status: Switched camera to {self.camera.camera_index}",
                "● LIVE"
            )

            if not self.preview_loop_running:
                self.preview_loop_running = True
                self.update_camera_feed()

        else:
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
        Take one photo.
        """

        if not self.camera.is_tracking:
            messagebox.showwarning(
                "Camera Not Active",
                "Start tracking before taking a photo."
            )
            return

        filepath = self.camera.take_photo()

        if filepath:
            self.update_status(
                f"Status: Photo saved to {filepath}",
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

        if not self.camera.is_tracking:
            messagebox.showwarning(
                "Camera Not Active",
                "Start tracking before recording video."
            )
            return

        # Start recording.
        if not self.camera.is_recording:
            self.current_video_path = self.camera.start_recording()

            if self.current_video_path:
                self.record_button.config(text="Stop Recording")

                self.update_status(
                    f"Status: Recording video/audio to {self.current_video_path}",
                    "● REC"
                )
            else:
                messagebox.showerror(
                    "Recording Error",
                    "Could not start video recording."
                )

        # Stop recording.
        else:
            saved_video_path = self.camera.stop_recording()

            self.record_button.config(text="Record Video")
            self.record_timer_label.config(text="Recording: 0 second")

            if saved_video_path:
                self.current_video_path = saved_video_path

                self.update_status(
                    f"Status: Video with audio saved to {self.current_video_path}",
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
        Zoom in.
        """

        self.camera.zoom_in()
        self.refresh_info_panel()

        self.update_status(
            f"Status: Zoom set to {self.camera.zoom_factor}x",
            "● LIVE" if self.camera.is_tracking else "● READY"
        )

    def zoom_out(self):
        """
        Zoom out.
        """

        self.camera.zoom_out()
        self.refresh_info_panel()

        self.update_status(
            f"Status: Zoom set to {self.camera.zoom_factor}x",
            "● LIVE" if self.camera.is_tracking else "● READY"
        )

    def reset_zoom(self):
        """
        Reset zoom.
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
        self.camera.stop_camera()
        self.preview_loop_running = False
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = CoBasV1App(root)

    root.protocol("WM_DELETE_WINDOW", app.on_close)

    root.mainloop()