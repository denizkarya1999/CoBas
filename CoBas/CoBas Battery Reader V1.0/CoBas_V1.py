import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import os
import cv2

from Camera.Camera import Camera
from Style import COLORS, FONTS, WINDOW, PREVIEW, SPACING, apply_styles
from Settings import SettingsWindow
from About import show_about_window


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
        self.root.title("CoBas Battery Reader V1.0")

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

        # Camera backend.
        self.camera = Camera(camera_index="/dev/video0")

        # Stores path of the current video file.
        self.current_video_path = None

        # Preview loop state.
        self.preview_loop_running = False
        self.preview_after_id = None

        # Stores Tkinter image reference for preview.
        self.preview_photo = None

        # Apply styles from Style.py.
        self.style = apply_styles(self.root)

        # Build all GUI components.
        self.build_gui()

        # Start camera automatically after GUI loads.
        self.root.after(700, self.auto_start_camera)

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
            icon_image = tk.PhotoImage(file=icon_path)

            # Sets title-bar/taskbar icon where supported.
            self.root.iconphoto(True, icon_image)

            # Keep reference so Python does not garbage collect the image.
            self.root.icon_image = icon_image

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
            text="Live Camera",
            style="PanelTitle.TLabel"
        ).pack(side="left")

        # Dynamic status icon.
        # This is tk.Label instead of ttk.Label because we change fg color.
        self.live_indicator_label = tk.Label(
            preview_header,
            text="● STARTING",
            bg=COLORS["panel_bg"],
            fg=COLORS["warning"],
            font=FONTS["status"]
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
            pady=(0, 6)
        )

        bottom_bar = ttk.Frame(parent, style="Panel.TFrame")
        bottom_bar.pack(
            fill="x",
            padx=SPACING["panel_padx"],
            pady=(0, 8)
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

    def refresh_info_panel(self):
        """
        Refresh all system information labels.
        """

        self.camera_info_label.config(
            text=f"Camera: {self.camera.camera_index}"
        )

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

    def auto_start_camera(self):
        """
        Start camera automatically after the GUI is visible.
        """

        self.start_tracking()

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

        # Restart automatically with the new camera source.
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

        self.update_status(
            "Status: Starting camera...",
            "● STARTING"
        )

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

            # Start refreshing frames.
            self.update_camera_feed()

        else:
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
        """

        # Cancel preview update loop first.
        self.cancel_preview_loop()

        # Release camera through backend.
        self.camera.stop_camera()

        # Reset preview display.
        self.video_label.config(
            image="",
            text="Tracking stopped.\n\nClick 'Start Tracking' to start again.",
            bg=COLORS["preview_bg"],
            fg=COLORS["muted_text"]
        )
        self.video_label.image = None

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
        if self.camera.is_recording:
            messagebox.showwarning(
                "Recording Active",
                "Stop recording before restarting the camera."
            )
            return

        # Cancel preview loop.
        self.cancel_preview_loop()

        # Release active camera if needed.
        if self.camera.is_tracking:
            self.camera.stop_camera()

        # Update preview text.
        self.video_label.config(
            image="",
            text="Restarting camera...\n\nPlease wait.",
            bg=COLORS["preview_bg"],
            fg=COLORS["muted_text"]
        )
        self.video_label.image = None

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

            # Resize preview to fit GUI.
            image = image.resize((PREVIEW["width"], PREVIEW["height"]))

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
            seconds = self.camera.get_recording_seconds()
            self.record_timer_label.config(
                text=f"Recording: {seconds} second(s)"
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

        # Camera must be active before recording.
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
                self.record_button.config(
                    text="Stop Recording",
                    style="VideoStop.TButton"
                )

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

            self.record_button.config(
                text="Capture Video",
                style="Capture.TButton"
            )

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

        # Release camera and recording resources.
        self.camera.stop_camera()

        # Close Tkinter window.
        self.root.destroy()


if __name__ == "__main__":
    # Create main Tkinter window.
    root = tk.Tk()

    # Create CoBas app instance.
    app = CoBasV1App(root)

    # Make sure camera resources are released when the user closes the app.
    root.protocol("WM_DELETE_WINDOW", app.on_close)

    # Start Tkinter event loop.
    root.mainloop()