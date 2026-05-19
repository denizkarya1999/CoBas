import tkinter as tk
from tkinter import ttk

from Style import COLORS


class SettingsWindow:
    """
    Settings window for CoBas_V1.

    Handles:
    - Camera source selection
    - Microphone source selection
    - Current configuration display
    """

    def __init__(self, parent, app):
        self.parent = parent
        self.app = app

        self.window = tk.Toplevel(parent)
        self.window.title("Settings")
        self.window.geometry("460x390")
        self.window.resizable(False, False)
        self.window.configure(bg=COLORS["main_bg"])

        # Keep Settings window above the main window.
        self.window.transient(parent)
        self.window.grab_set()

        # Current camera source.
        self.selected_camera_source = tk.StringVar(
            value=str(self.app.camera.camera_index)
        )

        # Load available microphones from Camera.py.
        self.microphones = self.app.camera.get_input_microphones()

        self.microphone_display_values = []
        self.microphone_lookup = {}

        for mic in self.microphones:
            display_name = self._build_microphone_display_name(mic)
            self.microphone_display_values.append(display_name)
            self.microphone_lookup[display_name] = mic

        current_mic_display = self._get_current_microphone_display()

        self.selected_microphone_source = tk.StringVar(
            value=current_mic_display
        )

        self.build_window()

    # --------------------------------------------------
    # Helper Methods
    # --------------------------------------------------

    def _build_microphone_display_name(self, mic):
        """
        Create a readable microphone dropdown label.
        """

        if mic["id"] is None:
            return mic["name"]

        return f'{mic["id"]}: {mic["name"]}'

    def _get_current_microphone_display(self):
        """
        Find the dropdown label matching the selected microphone.
        """

        current_id = self.app.camera.microphone_device_id

        for display_name, mic in self.microphone_lookup.items():
            if mic["id"] == current_id:
                return display_name

        return "System Default Microphone"

    # --------------------------------------------------
    # GUI Layout
    # --------------------------------------------------

    def build_window(self):
        """
        Build the Settings window UI.
        """

        settings_frame = ttk.Frame(self.window, style="Panel.TFrame")
        settings_frame.pack(fill="both", expand=True, padx=16, pady=16)

        # --------------------------------------------------
        # Description
        # --------------------------------------------------

        ttk.Label(
            settings_frame,
            text="Select the camera and microphone sources used by CoBas.",
            style="PanelText.TLabel",
            wraplength=420,
            justify="left"
        ).pack(anchor="w", pady=(0, 16))

        # --------------------------------------------------
        # Camera Source Selection
        # --------------------------------------------------

        camera_section = ttk.Frame(settings_frame, style="Panel.TFrame")
        camera_section.pack(fill="x", pady=(0, 14))

        ttk.Label(
            camera_section,
            text="Camera Source",
            style="PanelTitle.TLabel"
        ).pack(anchor="w", pady=(0, 5))

        self.camera_source_combo = ttk.Combobox(
            camera_section,
            textvariable=self.selected_camera_source,
            values=[
                "/dev/video0",
                "/dev/video1",
                "0",
                "1"
            ],
            state="readonly"
        )
        self.camera_source_combo.pack(fill="x", pady=(0, 7))

        ttk.Button(
            camera_section,
            text="Apply Camera Source",
            style="Primary.TButton",
            command=self.apply_camera_source
        ).pack(fill="x")

        # --------------------------------------------------
        # Microphone Source Selection
        # --------------------------------------------------

        microphone_section = ttk.Frame(settings_frame, style="Panel.TFrame")
        microphone_section.pack(fill="x", pady=(0, 14))

        ttk.Label(
            microphone_section,
            text="Microphone Source",
            style="PanelTitle.TLabel"
        ).pack(anchor="w", pady=(0, 5))

        self.microphone_source_combo = ttk.Combobox(
            microphone_section,
            textvariable=self.selected_microphone_source,
            values=self.microphone_display_values,
            state="readonly"
        )
        self.microphone_source_combo.pack(fill="x", pady=(0, 7))

        ttk.Button(
            microphone_section,
            text="Apply Microphone Source",
            style="Primary.TButton",
            command=self.apply_microphone_source
        ).pack(fill="x")

        # --------------------------------------------------
        # Current Configuration
        # --------------------------------------------------

        info_section = ttk.Frame(settings_frame, style="Panel.TFrame")
        info_section.pack(fill="both", expand=True)

        ttk.Label(
            info_section,
            text="Current Configuration",
            style="PanelTitle.TLabel"
        ).pack(anchor="w", pady=(0, 6))

        info_text = (
            f"Camera: {self.app.camera.camera_index}\n"
            f"Microphone: {self.app.camera.microphone_device_name}\n"
            f"Zoom: {self.app.camera.zoom_factor}x\n"
            f"Recording FPS: {self.app.camera.record_fps}\n"
            f"Audio Sample Rate: {self.app.camera.audio_sample_rate} Hz\n"
            f"Output Folder: {self.app.camera.output_dir}\n\n"
            "Tip: If an external microphone does not appear, reconnect it and reopen Settings."
        )

        ttk.Label(
            info_section,
            text=info_text,
            style="PanelText.TLabel",
            wraplength=420,
            justify="left"
        ).pack(anchor="w")

    # --------------------------------------------------
    # Apply Methods
    # --------------------------------------------------

    def apply_camera_source(self):
        """
        Apply selected camera source and close Settings.
        """

        selected_source = self.selected_camera_source.get()

        self.app.apply_camera_source_from_settings(selected_source)

        self.window.destroy()

    def apply_microphone_source(self):
        """
        Apply selected microphone source and close Settings.
        """

        selected_display_name = self.selected_microphone_source.get()

        selected_mic = self.microphone_lookup.get(selected_display_name)

        if selected_mic is None:
            return

        self.app.apply_microphone_source_from_settings(
            selected_mic["id"],
            selected_mic["name"]
        )

        self.window.destroy()