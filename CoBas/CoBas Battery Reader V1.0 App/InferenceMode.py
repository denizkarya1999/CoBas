"""Inference Mode for the CoBas Battery Reader.

Data Collection Mode is the app as it always was. Inference Mode reuses the
exact same chirp capture path, then hands each finished chirp (thermal PNG,
mmWave JPG, voice WAV) to the selected fusion model from ``CoBas/Inference``
and shows the estimated battery level on screen.
"""

from __future__ import annotations

import os
import queue
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from tkinter import ttk

from Style import COLORS
from TouchUI import TouchDialog

APP_DIRECTORY = Path(__file__).resolve().parent
INFERENCE_PACKAGE_DIRECTORY = APP_DIRECTORY.parent / "Inference"
DEFAULT_MODELS_DIRECTORY = Path(
    os.environ.get("COBAS_MODELS_DIR", str(INFERENCE_PACKAGE_DIRECTORY / "models"))
)

MODE_COLLECT = "collect"
MODE_INFER = "infer"
MODE_ENVIRONMENT_VARIABLE = "COBAS_APP_MODE"

MMWAVE_FRAME_TIMEOUT_SECONDS = 4.0
INFERENCE_THREADS = 4

FAMILY_LABELS = {
    "mmwave_thermal_voice": "mmWave + Thermal + Voice",
    "mmwave_thermal": "mmWave + Thermal",
    "mmwave_voice": "mmWave + Voice",
    "thermal_voice": "Thermal + Voice",
    "thermal": "Thermal only",
    "mmwave": "mmWave only",
    "voice": "Voice only",
}

if str(INFERENCE_PACKAGE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(INFERENCE_PACKAGE_DIRECTORY))


# --------------------------------------------------------------------------- #
# Startup mode chooser
# --------------------------------------------------------------------------- #

class ModeDialog(TouchDialog):
    """Two large buttons: Data Collection or Inference. Cancel closes the app."""

    def __init__(self, parent):
        super().__init__(parent, "How will CoBas be used?", width=640, height=330)
        ttk.Label(
            self.body,
            text=(
                "Data Collection records labelled chirps for training, exactly as before.\n"
                "Inference plays one chirp, runs the selected model on the thermal, "
                "mmWave and voice captures, and shows the estimated battery level."
            ),
            style="PanelText.TLabel",
            wraplength=600,
        ).pack(anchor="w", pady=(0, 12))
        choices = ttk.Frame(self.body, style="Panel.TFrame")
        choices.pack(fill="both", expand=True)
        for column in range(2):
            choices.columnconfigure(column, weight=1, uniform="mode")
        choices.rowconfigure(0, weight=1)
        ttk.Button(
            choices, text="Data Collection", style="Primary.TButton",
            command=lambda: self.finish(MODE_COLLECT),
        ).grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        ttk.Button(
            choices, text="Inference", style="Start.TButton",
            command=lambda: self.finish(MODE_INFER),
        ).grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        ttk.Button(
            self.body, text="Close CoBas", style="Tool.TButton", command=self.cancel,
        ).pack(fill="x", pady=(10, 0))


def choose_app_mode(parent):
    """Return MODE_COLLECT or MODE_INFER, honouring COBAS_APP_MODE; None means close."""
    preset = os.environ.get(MODE_ENVIRONMENT_VARIABLE, "").strip().lower()
    if preset in ("collect", "collection", "data", "data-collection"):
        return MODE_COLLECT
    if preset in ("infer", "inference"):
        return MODE_INFER
    return ModeDialog(parent).show()


# --------------------------------------------------------------------------- #
# Model catalogue and picker
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ModelChoice:
    path: Path
    family: str
    modalities: tuple
    label: str

    @property
    def short_label(self) -> str:
        """Compact form for a touch button: '1-1-1 · mmWave+Thermal+Voice'."""
        ratio = self.path.parent.name.replace(" Fusions", "").replace(" - ", "-").strip()
        family = FAMILY_LABELS.get(self.family, self.family).replace(" + ", "+")
        return f"{ratio} · {family}"


def discover_models(models_directory=DEFAULT_MODELS_DIRECTORY):
    """List PyTorch checkpoints (.pth or .pt) with a known recipe."""
    from cobas_inference.specs import is_model_file, spec_for_model_file

    models_directory = Path(models_directory)
    if not models_directory.is_dir():
        return []
    choices = []
    for path in sorted(models_directory.rglob("*")):
        if not is_model_file(path) or not path.is_file():
            continue
        try:
            spec = spec_for_model_file(path)
        except KeyError:
            continue
        family_label = FAMILY_LABELS.get(spec.family, spec.family)
        label = f"{path.parent.name} · {family_label} ({path.suffix[1:]})"
        choices.append(ModelChoice(path, spec.family, spec.modalities, label))
    # Three-modal first, then two-modal, then standalone; within a group by folder name.
    choices.sort(key=lambda choice: (-len(choice.modalities), choice.path.parent.name, choice.family))
    return choices


def default_model(choices):
    """Prefer the equal-weight three-modal checkpoint; otherwise the first choice."""
    for choice in choices:
        if choice.family == "mmwave_thermal_voice" and choice.path.parent.name.startswith("1 - 1 - 1"):
            return choice
    return choices[0] if choices else None


class ModelPickerDialog(TouchDialog):
    PAGE_SIZE = 3

    def __init__(self, parent, choices, selected=None):
        super().__init__(parent, "Choose a model", width=740, height=440)
        self.choices = list(choices)
        self.selected = selected
        self.page = 0
        if self.selected in self.choices:
            self.page = self.choices.index(self.selected) // self.PAGE_SIZE
        ttk.Label(
            self.body,
            text="Models are read from the Inference models folder. The folder name is the fusion ratio.",
            style="PanelText.TLabel", wraplength=700,
        ).pack(anchor="w", pady=(0, 6))
        self.list_frame = ttk.Frame(self.body, style="Panel.TFrame")
        self.list_frame.pack(fill="both", expand=True)
        page_row = ttk.Frame(self.body, style="Panel.TFrame")
        page_row.pack(fill="x", pady=6)
        self.previous = ttk.Button(page_row, text="Previous", style="Tool.TButton", command=lambda: self.change_page(-1))
        self.previous.pack(side="left")
        self.page_label = ttk.Label(page_row, style="PanelText.TLabel", anchor="center")
        self.page_label.pack(side="left", fill="x", expand=True)
        self.next = ttk.Button(page_row, text="Next", style="Tool.TButton", command=lambda: self.change_page(1))
        self.next.pack(side="right")
        actions = ttk.Frame(self.body, style="Panel.TFrame")
        actions.pack(fill="x")
        ttk.Button(actions, text="Cancel", style="Tool.TButton", command=self.cancel).pack(side="left", fill="x", expand=True, padx=(0, 4))
        ttk.Button(actions, text="Use model", style="Primary.TButton", command=self.accept).pack(side="left", fill="x", expand=True, padx=(4, 0))
        self.render()

    def change_page(self, delta):
        self.page += delta
        self.render()

    def render(self):
        for child in self.list_frame.winfo_children():
            child.destroy()
        pages = max(1, (len(self.choices) + self.PAGE_SIZE - 1) // self.PAGE_SIZE)
        self.page = max(0, min(self.page, pages - 1))
        if not self.choices:
            ttk.Label(
                self.list_frame,
                text=f"No models found. Copy checkpoints into:\n{DEFAULT_MODELS_DIRECTORY}",
                style="PanelText.TLabel", wraplength=700,
            ).pack(anchor="w")
        for choice in self.choices[self.page * self.PAGE_SIZE:(self.page + 1) * self.PAGE_SIZE]:
            selected = choice == self.selected
            text = ("✓  " if selected else "") + choice.label
            ttk.Button(
                self.list_frame, text=text,
                style="Primary.TButton" if selected else "Tool.TButton",
                command=lambda c=choice: self.select(c),
            ).pack(fill="x", pady=3)
        self.page_label.configure(text=f"{self.page + 1} / {pages}")
        self.previous.configure(state="disabled" if self.page == 0 else "normal")
        self.next.configure(state="disabled" if self.page >= pages - 1 else "normal")

    def select(self, choice):
        self.selected = choice
        self.render()

    def accept(self):
        self.finish(self.selected)


# --------------------------------------------------------------------------- #
# File readiness helpers
# --------------------------------------------------------------------------- #

def wait_for_stable_file(path, timeout_seconds, poll_seconds=0.06):
    """Wait until ``path`` exists, stops growing and decodes as an image or WAV.

    The mmWave writer saves its JPEG from another thread after the chirp is
    reported, so the file can exist before it is complete.
    """
    path = Path(path)
    deadline = time.monotonic() + timeout_seconds
    last_size = -1
    while time.monotonic() < deadline:
        if path.is_file():
            size = path.stat().st_size
            if size > 0 and size == last_size and _decodes(path):
                return path
            last_size = size
        time.sleep(poll_seconds)
    return None


def _decodes(path):
    try:
        if path.suffix.lower() == ".wav":
            import wave

            with wave.open(str(path), "rb") as handle:
                return handle.getnframes() > 0
        from PIL import Image

        with Image.open(path) as image:
            image.load()
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Background engine
# --------------------------------------------------------------------------- #

@dataclass
class ChirpJob:
    token: int
    position: int
    index: int
    global_index: int
    thermal_path: str
    thermal_fallback_path: str
    voice_path: str
    mmwave_path: str


@dataclass
class ChirpResult:
    token: int
    job: ChirpJob
    label: str
    confidence: float
    top3: list
    vote_label: str
    vote_confidence: float
    consistency: float
    count: int
    elapsed_ms: float


class InferenceEngine:
    """Load a model in the background and classify chirps as they complete."""

    def __init__(self, post_ui, on_result, on_status, on_error):
        self.post_ui = post_ui
        self.on_result = on_result
        self.on_status = on_status
        self.on_error = on_error
        self.choice = None
        self.predictor = None
        self.aggregator = None
        self.session_directory = None
        self._jobs = queue.Queue()
        self._lock = threading.Lock()
        self._loading = False
        self._stopped = False
        self._pending = 0
        self._idle = threading.Condition()
        self._worker = threading.Thread(target=self._run, name="cobas-inference", daemon=True)
        self._worker.start()

    # -- model selection -----------------------------------------------------

    @property
    def is_ready(self):
        with self._lock:
            return self.predictor is not None and not self._loading

    @property
    def is_loading(self):
        with self._lock:
            return self._loading

    def select_model(self, choice):
        with self._lock:
            self.choice = choice
            self.predictor = None
            self._loading = True
        self.post_ui(self.on_status, f"Loading {choice.short_label}…")
        threading.Thread(target=self._load, args=(choice,), daemon=True).start()

    def _load(self, choice):
        try:
            from cobas_inference import Predictor

            predictor = Predictor.from_model(choice.path, threads=INFERENCE_THREADS)
        except Exception as error:  # Worker boundary: report, never crash the UI.
            with self._lock:
                if self.choice is choice:
                    self._loading = False
            self.post_ui(self.on_error, f"Model could not be loaded: {error}")
            return
        with self._lock:
            if self.choice is not choice:
                return  # A newer selection superseded this load.
            self.predictor = predictor
            self._loading = False
        needs = ", ".join(predictor.requires())
        self.post_ui(self.on_status, f"Model ready: {choice.short_label} (uses {needs})")

    # -- sessions and jobs ---------------------------------------------------

    def start_session(self, session_directory):
        """Reset the running vote for a new estimate; nothing is written to disk."""
        from cobas_inference import VoteAggregator

        with self._lock:
            predictor = self.predictor
        if predictor is None:
            raise RuntimeError("No model is loaded")
        self.session_directory = Path(session_directory)
        self.aggregator = VoteAggregator(predictor.class_names)

    def enqueue(self, job):
        with self._idle:
            self._pending += 1
        self._jobs.put(job)

    def wait_until_idle(self, timeout_seconds):
        """Block until every queued chirp has been classified (or the timeout passes)."""
        deadline = time.monotonic() + timeout_seconds
        with self._idle:
            while self._pending > 0:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._idle.wait(remaining)
        return True

    def stop(self):
        self._stopped = True
        self._jobs.put(None)

    # -- worker --------------------------------------------------------------

    def _run(self):
        # Drain every queued chirp; only the sentinel from stop() ends the loop.
        while True:
            job = self._jobs.get()
            if job is None:
                break
            try:
                result = self._process(job)
            except Exception as error:  # Worker boundary.
                self.post_ui(self.on_error, f"Chirp {job.global_index}: {error}")
                result = None
            finally:
                with self._idle:
                    self._pending = max(0, self._pending - 1)
                    self._idle.notify_all()
            if result is not None:
                self.post_ui(self.on_result, result)

    def _process(self, job):
        with self._lock:
            predictor = self.predictor
            aggregator = self.aggregator
        if predictor is None or aggregator is None:
            raise RuntimeError("No model is loaded for this session")

        paths = {}
        needs = predictor.requires()
        if "thermal" in needs:
            thermal = wait_for_stable_file(job.thermal_path, 1.0) or wait_for_stable_file(job.thermal_fallback_path, 1.0)
            if thermal is None:
                raise RuntimeError("thermal frame is missing")
            paths["thermal"] = thermal
        if "voice" in needs:
            voice = wait_for_stable_file(job.voice_path, 2.0)
            if voice is None:
                raise RuntimeError("voice recording is missing")
            paths["voice"] = voice
        if "mmwave" in needs:
            mmwave = wait_for_stable_file(job.mmwave_path, MMWAVE_FRAME_TIMEOUT_SECONDS)
            if mmwave is None:
                raise RuntimeError("mmWave frame was not written in time")
            paths["mmwave"] = mmwave

        started = time.perf_counter()
        prediction = predictor.predict_paths(paths)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        aggregator.add(prediction)
        vote = aggregator.result()
        consistency = aggregator.consistency()
        return ChirpResult(
            token=job.token, job=job,
            label=prediction.label, confidence=prediction.confidence, top3=prediction.top(3),
            vote_label=vote.label, vote_confidence=vote.confidence,
            consistency=consistency, count=aggregator.count, elapsed_ms=elapsed_ms,
        )

def format_battery_level(label):
    """The read-out the device shows: 'Battery Level: 60%'."""
    return f"Battery Level: {label}"


def format_estimate(result):
    """Footer text: the running vote in the Battery Level form."""
    return format_battery_level(result.vote_label)


def format_detail(result):
    """Two short lines under the read-out: confidence and top-3, then chirp agreement."""
    top = ", ".join(f"{label} {p:.0%}" for label, p in result.top3)
    return (
        f"Confidence {result.vote_confidence:.0%} · top-3: {top}\n"
        f"Chirps: {result.count} · agreement {result.consistency:.0%} · last chirp {result.label}"
    )
