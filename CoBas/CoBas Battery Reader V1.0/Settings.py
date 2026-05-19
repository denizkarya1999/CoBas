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
        self.window.title("CoBas_V1 Settings")
        self.window.geometry("460x470")
        self.window.resizable(False, False)
        self.window.configure(bg=COLORS["main_bg"])

        self.window.transient(parent)
        self.window.grab_set()

        self.selected_camera_source = tk.StringVar(
            value=str(self.app.camera.camera_index)
        )

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

    def _build_microphone_display_name(self, mic):
        """
        Create a readable dropdown label.
        """

        if mic["id"] is None:
            return mic["name"]

        return f'{mic["id"]}: {mic["name"]}'

    def _get_current_microphone_display(self):
        """
        Find the dropdown label matching the current selected microphone.
        """

        current_id = self.app.camera.microphone_device_id

        for display_name, mic in self.microphone_lookup.items():
            if mic["id"] == current_id:
                return display_name

        return "System Default Microphone"

    def build_window(self):
        settings_frame = ttk.Frame(self.window, style="Panel.TFrame")
        settings_frame.pack(fill="both", expand=True, padx=14, pady=14)

        ttk.Label(
            settings_frame,
            text="Settings",
            style="PanelTitle.TLabel"
        ).pack(anchor="w", pady=(0, 10))

        # --------------------------------------------------
        # Camera Source Selection
        # --------------------------------------------------

        ttk.Label(
            settings_frame,
            text="Camera Source",
            style="PanelText.TLabel"
        ).pack(anchor="w", pady=(0, 4))

        self.camera_source_combo = ttk.Combobox(
            settings_frame,
            textvariable=self.selected_camera_source,
            values=[
                "/dev/video0",
                "/dev/video1",
                "0",
                "1"
            ],
            state="readonly"
        )
        self.camera_source_combo.pack(fill="x", pady=(0, 10))

        ttk.Button(
            settings_frame,
            text="Apply Camera Source",
            style="Primary.TButton",
            command=self.apply_camera_source
        ).pack(fill="x", pady=(0, 14))

        # --------------------------------------------------
        # Microphone Source Selection
        # --------------------------------------------------

        ttk.Label(
            settings_frame,
            text="Microphone Source",
            style="PanelText.TLabel"
        ).pack(anchor="w", pady=(0, 4))

        self.microphone_source_combo = ttk.Combobox(
            settings_frame,
            textvariable=self.selected_microphone_source,
            values=self.microphone_display_values,
            state="readonly"
        )
        self.microphone_source_combo.pack(fill="x", pady=(0, 10))

        ttk.Button(
            settings_frame,
            text="Apply Microphone Source",
            style="Primary.TButton",
            command=self.apply_microphone_source
        ).pack(fill="x", pady=(0, 14))

        # --------------------------------------------------
        # Current App Information
        # --------------------------------------------------

        ttk.Label(
            settings_frame,
            text="Current Configuration",
            style="PanelTitle.TLabel"
        ).pack(anchor="w", pady=(4, 8))

        info_text = (
            f"Active Camera: {self.app.camera.camera_index}\n"
            f"Active Microphone: {self.app.camera.microphone_device_name}\n"
            f"Current Zoom: {self.app.camera.zoom_factor}x\n"
            f"Recording FPS: {self.app.camera.record_fps}\n"
            f"Audio Sample Rate: {self.app.camera.audio_sample_rate} Hz\n"
            f"Output Folder: {self.app.camera.output_dir}\n\n"
            "Tip: If an external mic does not appear, reconnect it and reopen Settings."
        )

        ttk.Label(
            settings_frame,
            text=info_text,
            style="PanelText.TLabel",
            wraplength=420,
            justify="left"
        ).pack(anchor="w", pady=(0, 10))

        ttk.Button(
            settings_frame,
            text="Close",
            style="Tool.TButton",
            command=self.window.destroy
        ).pack(fill="x", pady=(8, 0))

    def apply_camera_source(self):
        selected_source = self.selected_camera_source.get()

        self.app.apply_camera_source_from_settings(selected_source)

        self.window.destroy()

    def apply_microphone_source(self):
        selected_display_name = self.selected_microphone_source.get()

        selected_mic = self.microphone_lookup.get(selected_display_name)

        if selected_mic is None:
            return

        self.app.apply_microphone_source_from_settings(
            selected_mic["id"],
            selected_mic["name"]
        )

        self.window.destroy()