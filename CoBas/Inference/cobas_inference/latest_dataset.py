"""Build and check the CoBas Latest Dataset: a few ready-to-open files per battery level for every model.

Each file is what that model takes in Local File Inference: the saved thermal
frame, mmWave frame or voice recording for the standalone models (unchanged
copies; the app turns a voice recording into its spectrogram), and the fused
image for a fusion model (built with that checkpoint's own fusion weights).
The layout mirrors the models folder::

    CoBas Latest Dataset/
        Standalone/CoBas Thermal Model/60_Percent_Battery/Thermal_Image_60_Percent_Position_1_011.png
        Standalone/CoBas Voice Reflection Recognition Model/60_Percent_Battery/Voice_Recording_60_Percent_Position_1_011.wav
        1 - 1 - 1 Fusions/CoBas mmWave Thermal Voice Fusion Model/60_Percent_Battery/Fused_Image_60_Percent_Position_1_011.png
        manifest.csv
        README.txt

Every model gets the same chirps, spread evenly over each class's recordings.
"""

from __future__ import annotations

import csv
import shutil
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Callable, Mapping

from PIL import Image

from . import preprocess
from .session import ChirpTriplet, discover_session, find_sessions, label_from_path
from .specs import MMWAVE, MODALITIES, THERMAL, VOICE, FusionSpec, is_model_file, resolve_spec

# Folder names spell out the checkpoint file names.
MODEL_NAMES = {
    "thermal": "CoBas Thermal Model",
    "mmwave": "CoBas mmWave Model",
    "voice": "CoBas Voice Reflection Recognition Model",
    "mmwave_thermal_voice": "CoBas mmWave Thermal Voice Fusion Model",
    "mmwave_thermal": "CoBas mmWave Thermal Fusion Model",
    "mmwave_voice": "CoBas mmWave and Voice Fusion Model",
    "thermal_voice": "CoBas Thermal and Voice Model",
}
# What a standalone model's file is called; thermal frames keep the app's own name.
SAVED_FILE_PREFIX = {THERMAL: "Thermal_Image", MMWAVE: "mmWave_Frame", VOICE: "Voice_Recording"}
MANIFEST = "manifest.csv"
README = "README.txt"


@dataclass(frozen=True)
class DatasetModel:
    path: Path
    spec: FusionSpec

    @property
    def group(self) -> str:
        """The models sub-folder: 'Standalone' or a ratio such as '1 - 1 - 1 Fusions'."""
        return self.path.parent.name

    @property
    def name(self) -> str:
        return MODEL_NAMES[self.spec.family]

    @property
    def folder(self) -> Path:
        return Path(self.group) / self.name


def _checkpoint_metadata(path: Path) -> Mapping:
    from .model import checkpoint_metadata, load_checkpoint

    return checkpoint_metadata(load_checkpoint(path))


def find_models(models_directory: str | Path, read_metadata: Callable[[Path], Mapping] = _checkpoint_metadata) -> list[DatasetModel]:
    """Every checkpoint (.pth or .pt) with a known recipe; standalone models first, then by ratio folder."""
    models = []
    for path in sorted(p for p in Path(models_directory).rglob("*") if p.is_file() and is_model_file(p)):
        try:
            spec = resolve_spec(path, read_metadata(path))
        except KeyError:
            continue  # Not one of the CoBas model files.
        models.append(DatasetModel(path, spec))
    return sorted(models, key=lambda m: (len(m.spec.modalities) > 1, m.group, m.name))


def pick_evenly(items: list, count: int) -> list:
    """``count`` items spread evenly over ``items`` (centred, so never just the first ones)."""
    if count >= len(items):
        return list(items)
    step = len(items) / count
    return [items[int(step * i + step / 2)] for i in range(count)]


def uses_saved_file(spec: FusionSpec) -> bool:
    """A standalone model opens the file the app saved; a fusion model needs its fused image."""
    return len(spec.modalities) == 1


def file_name(spec: FusionSpec, chirp: ChirpTriplet, percent: int, session_tag: str = "") -> str:
    """Kind, battery level, position and chirp number, e.g. 'Voice_Recording_60_Percent_Position_1_011.wav'."""
    suffix = f"{percent}_Percent_Position_{chirp.position}_{chirp.index:03d}"
    if session_tag:
        suffix = f"{suffix}_{session_tag}"
    if uses_saved_file(spec):
        modality = spec.modalities[0]
        return f"{SAVED_FILE_PREFIX[modality]}_{suffix}{chirp.paths()[modality].suffix.lower()}"
    return f"Fused_Image_{suffix}.png"


def write_file(spec: FusionSpec, chirp: ChirpTriplet, target: Path) -> None:
    """Copy the saved frame or recording unchanged, or save the fused image as a PNG."""
    paths = chirp.paths()
    if uses_saved_file(spec):
        shutil.copyfile(paths[spec.modalities[0]], target)
    else:
        Image.fromarray(preprocess.fused_image(paths, spec)).save(target)


def chirps_by_class(source: str | Path) -> dict[str, list[tuple[Path, ChirpTriplet]]]:
    """Complete chirps (thermal, mmWave and voice all present) of every labelled session, per class folder."""
    classes: dict[str, list[tuple[Path, ChirpTriplet]]] = defaultdict(list)
    for session in find_sessions(source):
        if label_from_path(session) is None:
            continue
        try:
            chirps = discover_session(session)
        except FileNotFoundError:
            continue  # An aborted capture: the folder skeleton with no chirp files.
        complete = [(session, chirp) for chirp in chirps if all(m in chirp.paths() for m in MODALITIES)]
        if complete:
            classes[session.parent.name].extend(complete)
    return dict(sorted(classes.items(), key=lambda item: int(label_from_path(Path(item[0])).rstrip("%"))))


def build(
    source: str | Path,
    models_directory: str | Path,
    out: str | Path,
    per_class: int = 10,
    read_metadata: Callable[[Path], Mapping] = _checkpoint_metadata,
    progress: Callable[[str], None] = print,
) -> list[dict]:
    """Write the dataset to ``out`` (which must not exist yet) and return the manifest rows."""
    source, out = Path(source), Path(out)
    if out.exists():
        raise FileExistsError(f"{out} already exists; remove it or choose another folder")
    models = find_models(models_directory, read_metadata)
    if not models:
        raise FileNotFoundError(f"No CoBas checkpoints found under {models_directory}")
    classes = chirps_by_class(source)
    if not classes:
        raise FileNotFoundError(f"No labelled capture sessions with complete chirps under {source}")

    rows: list[dict] = []
    out.mkdir(parents=True)
    for model in models:
        written = len(rows)
        for class_folder, candidates in classes.items():
            percent = int(label_from_path(Path(class_folder)).rstrip("%"))
            several_sessions = len({session for session, _ in candidates}) > 1
            target_directory = out / model.folder / class_folder
            target_directory.mkdir(parents=True)
            for session, chirp in pick_evenly(candidates, per_class):
                tag = session.name.removeprefix("Session_") if several_sessions else ""
                name = file_name(model.spec, chirp, percent, tag)
                write_file(model.spec, chirp, target_directory / name)
                rows.append({
                    "group": model.group,
                    "model": model.name,
                    "checkpoint": model.path.name,
                    "class": class_folder,
                    "file": str(model.folder / class_folder / name),
                    "session": session.name,
                    "position": chirp.position,
                    "chirp": chirp.index,
                    "sources": "; ".join(chirp.paths()[m].name for m in model.spec.modalities),
                })
        progress(f"{model.folder}: {len(rows) - written} files")

    with open(out / MANIFEST, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / README).write_text(_readme(out.name, source, models, classes, per_class), encoding="utf-8")
    return rows


def _readme(title: str, source: Path, models: list[DatasetModel], classes: Mapping[str, list], per_class: int) -> str:
    per_model = sum(min(per_class, len(chirps)) for chirps in classes.values())
    lines = [
        title,
        "=" * len(title),
        "",
        f"Built on {date.today().isoformat()} from: {source}",
        f"{len(models)} models x {len(classes)} battery levels x {per_class} files "
        f"= {per_model} files per model, {per_model * len(models)} in total.",
        "",
        "Layout: <models sub-folder>/<model name>/<battery level>/<file>",
        "The sub-folders and model names follow CoBas/Inference/models, so each",
        "checkpoint has its own files. A fusion model's images are fused with that",
        "checkpoint's own weights and must not be used with another ratio.",
        "",
        "What each model's file is:",
        "  CoBas Thermal Model                       the saved thermal frame (unchanged copy)",
        "  CoBas mmWave Model                        the saved mmWave frame (unchanged copy)",
        "  CoBas Voice Reflection Recognition Model  the saved voice recording (unchanged copy);",
        "                                            the app converts it to a spectrogram",
        "  fusion models                             the fused 256x256 image",
        "",
        "Every model has the same chirps: for each battery level, the chirps are",
        f"spread evenly over that level's recordings ({per_class} per level). The",
        "position and chirp number are in each file name, and manifest.csv lists the",
        "session and source files of every file.",
        "",
        "To use: in the app's Local File Inference tab choose the model, tap Open",
        "File and open a file from that model's folder. The battery level in the",
        "folder name is the expected read-out.",
        "",
        "To check every file against its model:",
        "  python -m cobas_inference check-latest-dataset <this folder>",
        "",
        "Models:",
    ]
    lines += [f"  {model.folder}  ({model.path.name})" for model in models]
    return "\n".join(lines) + "\n"


def check(
    dataset: str | Path,
    models_directory: str | Path,
    load_predictor: Callable[[Path], object] | None = None,
    progress: Callable[[str], None] = print,
    read_metadata: Callable[[Path], Mapping] = _checkpoint_metadata,
) -> list[dict]:
    """Classify every file with its own model; return one row per model with the misses."""
    if load_predictor is None:
        from .predictor import Predictor

        load_predictor = Predictor.from_model
    dataset = Path(dataset)
    results = []
    for model in find_models(models_directory, read_metadata):
        folder = dataset / model.folder
        if not folder.is_dir():
            progress(f"{model.folder}: no files in the dataset")
            continue
        predictor = load_predictor(model.path)
        total, misses = 0, []
        for class_folder in sorted(p for p in folder.iterdir() if p.is_dir()):
            expected = label_from_path(class_folder)
            for file in sorted(class_folder.iterdir()):
                predicted = predictor.predict_file(file).label
                total += 1
                if predicted != expected:
                    misses.append((file.name, expected, predicted))
        results.append({"model": str(model.folder), "files": total, "correct": total - len(misses), "misses": misses})
        progress(f"{model.folder}: {total - len(misses)}/{total} correct")
        for name, expected, predicted in misses:
            progress(f"    {name}: recorded {expected}, read as {predicted}")
    return results
