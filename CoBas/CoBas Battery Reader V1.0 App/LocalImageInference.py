"""Local Image Inference: estimate the battery level of one picture from disk.

No chirp is played and no sensor is read. The user opens an image, the model
chosen on this screen classifies it in the background, and the screen shows
the same "Battery Level: N%" read-out as a live estimate.

The picture is taken as the image that model was trained on: a thermal camera
image or a mmWave frame for those standalone models, the spectrogram for the
voice standalone, and the already fused image for the fusion models (see
``cobas_inference.preprocess.image_model_input``).
"""

from __future__ import annotations

import os
import re
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk

from PIL import Image, ImageTk

from InferenceMode import (
    DEFAULT_MODELS_DIRECTORY,
    FAMILY_LABELS,
    INFERENCE_THREADS,
    ModelPickerDialog,
    discover_models,
    format_battery_level,
)
from Style import COLORS, FONTS
from TouchUI import TouchDialog, ask_number, dialogs

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp")
RESULT_FONT = ("Arial", -26, "bold")
PREVIEW_BACKGROUND = (2, 6, 23)  # COLORS["preview_bg"] as RGB

_EXPECTED_IMAGE = {
    ("thermal",): "a thermal camera image",
    ("mmwave",): "a mmWave range-angle frame",
    ("voice",): "a voice spectrogram image",
}


# --------------------------------------------------------------------------- #
# Models for a single image
# --------------------------------------------------------------------------- #

def expected_image(choice):
    """What to open for this model, e.g. 'a thermal camera image'."""
    known = _EXPECTED_IMAGE.get(tuple(choice.modalities))
    if known:
        return known
    return f"a fused {FAMILY_LABELS.get(choice.family, choice.family)} image"


def image_models(choices):
    """Standalone models first: a raw sensor frame is what is usually on disk."""
    return sorted(choices, key=lambda c: (len(c.modalities), c.path.parent.name, c.family))


def default_image_model(choices):
    """Prefer the thermal standalone model; otherwise the first choice."""
    for choice in choices:
        if choice.family == "thermal":
            return choice
    return choices[0] if choices else None


# --------------------------------------------------------------------------- #
# Touch file browser
# --------------------------------------------------------------------------- #

def _natural_key(path):
    """Sort 'Session_2' before 'Session_10' and ignore case."""
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path.name.lower())]


def list_directory(directory):
    """Return (sub-folders, image files) of ``directory``; hidden entries are skipped."""
    folders, images = [], []
    with os.scandir(directory) as entries:
        for entry in entries:
            if entry.name.startswith("."):
                continue
            try:
                if entry.is_dir():
                    folders.append(Path(entry.path))
                elif entry.is_file() and Path(entry.name).suffix.lower() in IMAGE_SUFFIXES:
                    images.append(Path(entry.path))
            except OSError:
                continue  # A broken link or an unreadable entry is simply not listed.
    return sorted(folders, key=_natural_key), sorted(images, key=_natural_key)


def shorten(text, limit, keep_end=False):
    """Fit ``text`` on one touch-button line, keeping its start (or its end)."""
    if len(text) <= limit:
        return text
    return "…" + text[-(limit - 1):] if keep_end else text[: limit - 1] + "…"


class ImageBrowserDialog(TouchDialog):
    """Tap a folder to enter it, tap an image to open it. Returns the image path."""

    PAGE_SIZE = 4

    def __init__(self, parent, start_directory):
        super().__init__(parent, "Open an image", width=780, height=470)
        self.directory = Path(start_directory)
        self.folders, self.images = [], []
        self.problem = None
        self.page = 0
        self.path_label = ttk.Label(self.body, style="PanelText.TLabel")
        self.path_label.pack(anchor="w", pady=(0, 6))
        self.list_frame = ttk.Frame(self.body, style="Panel.TFrame")
        self.list_frame.pack(fill="both", expand=True)
        page_row = ttk.Frame(self.body, style="Panel.TFrame")
        page_row.pack(fill="x", pady=6)
        self.previous = ttk.Button(page_row, text="Previous", style="Tool.TButton", command=lambda: self.change_page(-1))
        self.previous.pack(side="left")
        self.next = ttk.Button(page_row, text="Next", style="Tool.TButton", command=lambda: self.change_page(1))
        self.next.pack(side="right")
        self.page_button = ttk.Button(page_row, style="Tool.TButton", command=self.jump_to_page)
        self.page_button.pack(side="left", fill="x", expand=True, padx=8)
        actions = ttk.Frame(self.body, style="Panel.TFrame")
        actions.pack(fill="x")
        for column in range(3):
            actions.columnconfigure(column, weight=1, uniform="action")
        ttk.Button(actions, text="Cancel", style="Tool.TButton", command=self.cancel).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(actions, text="Home folder", style="Tool.TButton",
                   command=lambda: self.open_directory(Path.home())).grid(row=0, column=1, sticky="ew", padx=4)
        self.up = ttk.Button(actions, text="Up one folder", style="Tool.TButton",
                             command=lambda: self.open_directory(self.directory.parent))
        self.up.grid(row=0, column=2, sticky="ew", padx=(4, 0))
        self.open_directory(self.directory)

    @property
    def page_count(self):
        entries = len(self.folders) + len(self.images)
        return max(1, (entries + self.PAGE_SIZE - 1) // self.PAGE_SIZE)

    def open_directory(self, directory):
        self.directory = Path(directory)
        self.page = 0
        try:
            self.folders, self.images = list_directory(self.directory)
            self.problem = None
        except OSError as error:
            self.folders, self.images = [], []
            self.problem = f"This folder cannot be read: {error.strerror or error}"
        self.render()

    def change_page(self, delta):
        self.page += delta
        self.render()

    def jump_to_page(self):
        page = ask_number(self, "Go to page", initial=self.page + 1, minimum=1, maximum=self.page_count)
        if page is not None:
            self.page = page - 1
            self.render()

    def render(self):
        for child in self.list_frame.winfo_children():
            child.destroy()
        pages = self.page_count
        self.page = max(0, min(self.page, pages - 1))
        self.path_label.configure(text=shorten(str(self.directory), 84, keep_end=True))
        entries = [(folder, True) for folder in self.folders] + [(image, False) for image in self.images]
        if not entries:
            ttk.Label(
                self.list_frame,
                text=self.problem or "No folders or images here (PNG, JPG or BMP).",
                style="PanelText.TLabel", wraplength=740,
            ).pack(anchor="w")
        for path, is_folder in entries[self.page * self.PAGE_SIZE:(self.page + 1) * self.PAGE_SIZE]:
            if is_folder:
                button = ttk.Button(self.list_frame, text=shorten(path.name, 58) + "  ›", style="Tool.TButton",
                                    command=lambda p=path: self.open_directory(p))
            else:
                button = ttk.Button(self.list_frame, text=shorten(path.name, 60), style="Primary.TButton",
                                    command=lambda p=path: self.finish(p))
            button.pack(fill="x", pady=3)
        self.page_button.configure(text=f"Page {self.page + 1} / {pages}", state="normal" if pages > 1 else "disabled")
        self.previous.configure(state="disabled" if self.page == 0 else "normal")
        self.next.configure(state="disabled" if self.page >= pages - 1 else "normal")
        self.up.configure(state="disabled" if self.directory.parent == self.directory else "normal")


# --------------------------------------------------------------------------- #
# Background classification
# --------------------------------------------------------------------------- #

def _load_predictor(choice):
    from cobas_inference import Predictor

    return Predictor.from_model(choice.path, threads=INFERENCE_THREADS)


class LocalImageRunner:
    """Keep the chosen model loaded and classify one image off the Tk thread."""

    def __init__(self, post_ui, on_result, on_error, load_predictor=_load_predictor):
        self.post_ui = post_ui
        self.on_result = on_result
        self.on_error = on_error
        self.load_predictor = load_predictor
        self.latest = 0
        self._work = threading.Lock()
        self._predictor = None
        self._predictor_path = None

    def estimate(self, request, choice, image_path):
        """Classify ``image_path``; ``request`` comes back with the answer.

        A newer request supersedes an older one that has not started yet.
        """
        self.latest = request
        worker = threading.Thread(
            target=self._run, args=(request, choice, Path(image_path)),
            name="cobas-local-image", daemon=True,
        )
        worker.start()
        return worker

    def _run(self, request, choice, image_path):
        with self._work:
            if request != self.latest:
                return
            try:
                if self._predictor is None or self._predictor_path != choice.path:
                    self._predictor = None  # Free the previous model before loading the next.
                    self._predictor = self.load_predictor(choice)
                    self._predictor_path = choice.path
                prediction = self._predictor.predict_image(image_path)
            except Exception as error:  # Worker boundary: report, never crash the UI.
                self.post_ui(self.on_error, request, str(error))
                return
        self.post_ui(self.on_result, request, prediction.label)


# --------------------------------------------------------------------------- #
# The screen
# --------------------------------------------------------------------------- #

class LocalImageInferencePanel:
    """The 'Local Image Inference' tab: open a picture, show its battery level."""

    def __init__(self, parent, app):
        self.app = app
        self.root = app.root
        self.choices = image_models(discover_models())
        self.choice = default_image_model(self.choices)
        captures = Path(app.base_dir) / "Captures"
        self.directory = captures if captures.is_dir() else Path.home()
        self.image_path = None
        self.picture = None
        self.request = 0
        self.preview_photo = None
        self.preview_size = None
        self.runner = LocalImageRunner(app.post_ui, self.show_result, self.show_error)
        self.build(parent)
        self.refresh_model()

    def build(self, parent):
        parent.columnconfigure(0, weight=1, uniform="local-image")
        parent.columnconfigure(1, weight=1, uniform="local-image")
        parent.rowconfigure(0, weight=1)
        picture = ttk.Frame(parent, style="Panel.TFrame")
        picture.grid(row=0, column=0, sticky="nsew", padx=(8, 0), pady=8)
        self.result_label = tk.Label(picture, text=format_battery_level("—"), bg=COLORS["panel_bg"],
                                     fg=COLORS["accent"], font=RESULT_FONT)
        self.result_label.pack(side="bottom", fill="x", pady=(6, 0))
        self.preview = tk.Label(picture, text="No image opened.\n\nTap Open Image to choose one.",
                                bg=COLORS["preview_bg"], fg=COLORS["muted_text"],
                                font=FONTS["preview_text"], width=1, height=1, bd=0)
        self.preview.pack(fill="both", expand=True)
        self.preview.bind("<Configure>", self.handle_preview_resized)

        controls = ttk.Frame(parent, style="Panel.TFrame", padding=10)
        controls.grid(row=0, column=1, sticky="nsew")
        ttk.Label(controls, text="Local Image Inference", style="PanelTitle.TLabel").pack(anchor="w", pady=(0, 5))
        ttk.Label(controls, text="Model", style="PanelText.TLabel").pack(anchor="w")
        self.model_button = ttk.Button(controls, style="Tool.TButton", command=self.choose_model)
        self.model_button.pack(fill="x", pady=(0, 6))
        self.open_button = ttk.Button(controls, text="Open Image", style="Primary.TButton", command=self.open_image)
        self.open_button.pack(fill="x")
        self.status_label = ttk.Label(controls, style="PanelText.TLabel", wraplength=350)
        self.status_label.pack(anchor="w", pady=(6, 0))
        controls.bind("<Configure>", lambda e: self.status_label.configure(wraplength=max(200, e.width - 24)))

    # -- model ---------------------------------------------------------------

    def refresh_model(self):
        if self.choice is None:
            self.model_button.configure(text="Choose a model…")
            self.set_status("No models found. Tap the model button once checkpoints are installed.")
            return
        self.model_button.configure(text=shorten(self.choice.short_label, 36))
        if self.image_path is None:
            self.set_status(f"Open {expected_image(self.choice)} to estimate its battery level.")

    def choose_model(self):
        if self.capture_is_running("changing the model"):
            return
        self.choices = image_models(discover_models())
        choice = ModelPickerDialog(self.root, self.choices, self.choice).show()
        if choice is None:
            return
        self.choice = choice
        self.refresh_model()
        if self.image_path is not None:
            self.estimate()

    # -- image ---------------------------------------------------------------

    def open_image(self):
        if self.capture_is_running("opening an image"):
            return
        if self.choice is None:
            dialogs.showwarning("No model", f"No models found. Copy checkpoints into:\n{DEFAULT_MODELS_DIRECTORY}",
                                parent=self.root)
            return
        path = ImageBrowserDialog(self.root, self.directory).show()
        if path is None:
            return
        self.directory = path.parent
        try:
            with Image.open(path) as image:
                self.picture = image.convert("RGB")
        except (OSError, ValueError) as error:
            dialogs.showerror("Image", f"{path.name} could not be opened as an image:\n{error}", parent=self.root)
            return
        self.image_path = path
        self.render_preview()
        self.estimate()

    def capture_is_running(self, action):
        """A live estimate needs the processor; keep local inference out of its way."""
        if self.app.pulse_sequence_active or self.app.export_in_progress:
            dialogs.showwarning("Estimate running", f"Wait for the current estimate to finish before {action}.",
                                parent=self.root)
            return True
        return False

    def handle_preview_resized(self, event):
        if self.image_path is not None and (event.width, event.height) != self.preview_size:
            self.render_preview()

    def render_preview(self):
        width, height = max(1, self.preview.winfo_width()), max(1, self.preview.winfo_height())
        self.preview_size = (width, height)
        image = self.picture.copy()
        # Small frames (a 32x24 thermal image) are enlarged so they are visible.
        scale = min(width / image.width, height / image.height)
        size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
        image = image.resize(size, Image.Resampling.LANCZOS if scale < 1 else Image.Resampling.NEAREST)
        canvas = Image.new("RGB", (width, height), PREVIEW_BACKGROUND)
        canvas.paste(image, ((width - image.width) // 2, (height - image.height) // 2))
        self.preview_photo = ImageTk.PhotoImage(canvas)
        self.preview.configure(image=self.preview_photo, text="")

    # -- estimate ------------------------------------------------------------

    def estimate(self):
        self.request += 1
        self.result_label.configure(text=format_battery_level("…"))
        self.set_status(f"{self.image_path.name}\nEstimating with {self.choice.short_label}…")
        self.runner.estimate(self.request, self.choice, self.image_path)

    def show_result(self, request, label):
        if request != self.request:
            return
        self.result_label.configure(text=format_battery_level(label))
        self.set_status(f"{self.image_path.name}\nModel expects {expected_image(self.choice)}.")

    def show_error(self, request, message):
        if request != self.request:
            return
        print(f"[WARNING] Local image inference: {message}")
        self.result_label.configure(text=format_battery_level("—"))
        self.set_status(f"{self.image_path.name}\nNo estimate: {message}", error=True)

    def set_status(self, message, error=False):
        self.status_label.configure(text=message, foreground=COLORS["error" if error else "panel_text"])
