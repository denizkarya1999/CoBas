"""Settings dialog for the integrated CoBas sensor application."""

import tkinter as tk
from tkinter import ttk

from Style import COLORS


class SettingsWindow:
    """Select the microphone and show the fixed radar/thermal configuration."""

    def __init__(self, parent, app):
        self.parent = parent
        self.app = app

        self.window = tk.Toplevel(parent)
        self.window.title("Settings")
        self.window.geometry("460x360")
        self.window.resizable(False, False)
        self.window.configure(bg=COLORS["main_bg"])
        self.window.transient(parent)
        self.window.grab_set()

        self.microphones = self.app.audio.get_input_microphones()
        self.microphone_display_values = []
        self.microphone_lookup = {}
        for microphone in self.microphones:
            display_name = self._display_name(microphone)
            self.microphone_display_values.append(display_name)
            self.microphone_lookup[display_name] = microphone

        self.selected_microphone_source = tk.StringVar(
            value=self._current_microphone_display()
        )
        self.build_window()

    @staticmethod
    def _display_name(microphone):
        if microphone["id"] is None:
            return microphone["name"]
        return f"{microphone['id']}: {microphone['name']}"

    def _current_microphone_display(self):
        for display_name, microphone in self.microphone_lookup.items():
            if microphone["id"] == self.app.audio.microphone_device_id:
                return display_name
        return "System Default Microphone"

    def build_window(self):
        settings_frame = ttk.Frame(self.window, style="Panel.TFrame")
        settings_frame.pack(fill="both", expand=True, padx=16, pady=16)

        ttk.Label(
            settings_frame,
            text=(
                "The IWR6843AOP radar uses /dev/ttyUSB0 and /dev/ttyUSB1. "
                "Select the microphone used to save every chirp as its own "
                "voice WAV recording."
            ),
            style="PanelText.TLabel",
            wraplength=420,
            justify="left",
        ).pack(anchor="w", pady=(0, 16))

        microphone_section = ttk.Frame(settings_frame, style="Panel.TFrame")
        microphone_section.pack(fill="x", pady=(0, 16))
        ttk.Label(
            microphone_section,
            text="Microphone Source",
            style="PanelTitle.TLabel",
        ).pack(anchor="w", pady=(0, 5))
        ttk.Combobox(
            microphone_section,
            textvariable=self.selected_microphone_source,
            values=self.microphone_display_values,
            state="readonly",
        ).pack(fill="x", pady=(0, 7))
        ttk.Button(
            microphone_section,
            text="Apply Microphone Source",
            style="Primary.TButton",
            command=self.apply_microphone_source,
        ).pack(fill="x")

        info_section = ttk.Frame(settings_frame, style="Panel.TFrame")
        info_section.pack(fill="both", expand=True)
        ttk.Label(
            info_section,
            text="Current Configuration",
            style="PanelTitle.TLabel",
        ).pack(anchor="w", pady=(0, 6))
        ttk.Label(
            info_section,
            text=(
                "mmWave CLI: /dev/ttyUSB0\n"
                "mmWave data: /dev/ttyUSB1\n"
                f"Thermal sensor: {self.app.thermal_camera.status}\n"
                f"Microphone: {self.app.audio.microphone_device_name}\n"
                f"Thermal FPS: {self.app.thermal_camera.record_fps:g}\n"
                "Dataset frames: 1 mmWave + 1 thermal per chirp\n"
                f"mmWave frames: {self.app.mmwave_frames_dir}\n"
                f"Thermal frames: {self.app.thermal_frames_dir}\n"
                f"Voices: {self.app.voices_dir}\n"
                f"References: {self.app.references_dir}"
            ),
            style="PanelText.TLabel",
            wraplength=420,
            justify="left",
        ).pack(anchor="w")

    def apply_microphone_source(self):
        microphone = self.microphone_lookup.get(self.selected_microphone_source.get())
        if microphone is None:
            return
        self.app.apply_microphone_source_from_settings(
            microphone["id"],
            microphone["name"],
        )
        self.window.destroy()
