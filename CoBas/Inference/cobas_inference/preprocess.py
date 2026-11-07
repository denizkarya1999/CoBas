"""Turn raw CoBas captures into the exact tensor the fusion models were trained on.

Each function mirrors one step of the Colab notebooks, including their
rounding and resampling choices, so a regenerated fused image is byte-identical
to the one that was used in training (see tests/test_parity.py).
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from PIL import Image

from .specs import (
    DIV255,
    FUSED_SIZE,
    IMAGENET,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MODEL_INPUT_SIZE,
    PER_IMAGE_MINMAX,
    RAW255,
    UNIT,
    VOICE,
    FusionSpec,
    ImageSpec,
    VoiceSpec,
)

_PIL_FILTERS = {
    "bilinear": Image.Resampling.BILINEAR,
    "lanczos": Image.Resampling.LANCZOS,
}


# --------------------------------------------------------------------------- #
# Raster modalities (mmWave JPG, thermal PNG)
# --------------------------------------------------------------------------- #

def load_rgb(path: str | Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def resize(image: Image.Image, size: int, method: str) -> Image.Image:
    """PIL resize to ``size`` x ``size``; a no-op when already that size (as PIL does)."""
    return image.resize((size, size), _PIL_FILTERS[method])


def center_crop(image: Image.Image, size: int) -> Image.Image:
    """torchvision CenterCrop semantics for a square crop."""
    width, height = image.size
    left = int(round((width - size) / 2.0))
    top = int(round((height - size) / 2.0))
    return image.crop((left, top, left + size, top + size))


def modality_image_256(path: str | Path, spec: ImageSpec) -> np.ndarray:
    """Raster modality -> uint8 [256, 256, 3], exactly as the notebooks saved it."""
    image = load_rgb(path)
    if spec.pre_crop_224:
        # Thermal-Voice notebook: Resize(256) -> CenterCrop(224) saved to disk,
        # then the fusion step reloaded it and resized to 256 with LANCZOS.
        image = resize(image, FUSED_SIZE, "bilinear")
        image = center_crop(image, MODEL_INPUT_SIZE)
        image = resize(image, FUSED_SIZE, "lanczos")
    else:
        image = resize(image, FUSED_SIZE, spec.resize)
    array = np.asarray(image, dtype=np.uint8)
    if array.shape != (FUSED_SIZE, FUSED_SIZE, 3):
        raise ValueError(f"Unexpected image shape {array.shape} for {path}")
    return array


# --------------------------------------------------------------------------- #
# Voice reflection -> mel spectrogram image
# --------------------------------------------------------------------------- #

def load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """Load mono float32 audio at its native rate (librosa.load(sr=None, mono=True))."""
    import librosa  # heavy import; keep it local so the package imports without audio deps

    signal, sample_rate = librosa.load(str(path), sr=None, mono=True)
    if signal.size == 0:
        raise ValueError(f"Empty voice recording: {path}")
    return signal.astype(np.float32, copy=False), int(sample_rate)


def mel_db(signal: np.ndarray, sample_rate: int, spec: VoiceSpec) -> np.ndarray:
    """Power mel-spectrogram in dB relative to its maximum (float32, [n_mels, frames])."""
    import librosa

    nyquist = sample_rate / 2.0
    if nyquist <= spec.fmin:
        raise ValueError(
            f"Sample rate {sample_rate} Hz is too low for a {spec.fmin:g} Hz band"
        )
    fmax = min(spec.fmax, nyquist)
    power = librosa.feature.melspectrogram(
        y=signal, sr=sample_rate, n_mels=spec.n_mels, fmin=spec.fmin, fmax=fmax
    )
    db = librosa.power_to_db(power, ref=np.max)
    if not np.isfinite(db).all():
        raise ValueError("Non-finite spectrogram values")
    return db.astype(np.float32, copy=False)


def spectrogram_to_uint8(db: np.ndarray, spec: VoiceSpec) -> np.ndarray:
    """Per-clip min-max to 0..255 with the notebook's quantization rule."""
    db = db.astype(np.float32, copy=False)
    db_min = float(db.min())
    db_range = float(db.max() - db_min)
    if db_range <= float(np.finfo(np.float32).eps):
        return np.zeros(db.shape, dtype=np.uint8)
    if spec.quantize == "rint":
        unit = np.clip((db - db_min) / db_range, 0.0, 1.0)
        return np.rint(unit * 255.0).astype(np.uint8)
    if spec.quantize == "floor":
        # Thermal-Voice notebook: 255 * (db - min) / (max - min) then .astype(np.uint8)
        scaled = 255 * (db - np.float32(db_min)) / np.float32(db_range)
        return scaled.astype(np.uint8)
    raise ValueError(f"Unknown quantize rule {spec.quantize!r}")


def voice_spectrogram_256(path: str | Path, spec: VoiceSpec) -> np.ndarray:
    """WAV -> uint8 [256, 256] grayscale spectrogram with low frequencies at the bottom."""
    signal, sample_rate = load_audio(path)
    matrix = spectrogram_to_uint8(mel_db(signal, sample_rate, spec), spec)
    if spec.flip_stage == "before_resize":
        matrix = np.flipud(matrix)
    image = Image.fromarray(np.ascontiguousarray(matrix), mode="L")
    image = image.resize((FUSED_SIZE, FUSED_SIZE), Image.Resampling.LANCZOS)
    if spec.flip_stage == "after_resize":
        image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    elif spec.flip_stage != "before_resize":
        raise ValueError(f"Unknown flip stage {spec.flip_stage!r}")
    return np.asarray(image, dtype=np.uint8)


def gray_to_rgb(matrix: np.ndarray) -> np.ndarray:
    """PIL ``convert('RGB')`` of an 'L' image: three identical channels."""
    return np.repeat(matrix[:, :, None], 3, axis=2)


# --------------------------------------------------------------------------- #
# Fusion and tensor
# --------------------------------------------------------------------------- #

def minmax_unit(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError("Image contains NaN or infinite values")
    low = float(array.min())
    span = float(array.max() - low)
    if span <= float(np.finfo(np.float32).eps):
        return np.zeros_like(array, dtype=np.float32)
    return np.clip((array - low) / span, 0.0, 1.0)


def _weighted_sum(arrays: Sequence[np.ndarray], weights: Sequence[float], equal: bool) -> np.ndarray:
    if equal:
        # Notebooks wrote (a + b + c) / 3.0 and (a + b) / 2.0 for equal weights.
        total = arrays[0]
        for array in arrays[1:]:
            total = total + array
        return total / np.float32(len(arrays))
    accumulator = arrays[0] * np.float32(weights[0])
    for array, weight in zip(arrays[1:], weights[1:]):
        accumulator = accumulator + array * np.float32(weight)
    return accumulator


def fuse(arrays: Sequence[np.ndarray], weights: Sequence[float], mode: str) -> np.ndarray:
    """Combine 256x256x3 uint8 modality images into the fused uint8 image."""
    if len(arrays) != len(weights):
        raise ValueError("One weight per modality image is required")
    shapes = {a.shape for a in arrays}
    if len(shapes) != 1:
        raise ValueError(f"Modality images differ in shape: {shapes}")
    floats = [np.asarray(a, dtype=np.float32) for a in arrays]
    equal = max(weights) - min(weights) < 1e-9

    if mode == PER_IMAGE_MINMAX:
        fused = _weighted_sum([minmax_unit(a) for a in floats], weights, equal)
        return np.rint(np.clip(fused, 0.0, 1.0) * 255.0).astype(np.uint8)
    if mode == DIV255:
        fused = _weighted_sum([a / np.float32(255.0) for a in floats], weights, equal)
        return np.rint(np.clip(fused, 0.0, 1.0) * 255.0).astype(np.uint8)
    if mode == RAW255:
        fused = _weighted_sum(floats, weights, equal)
        return np.rint(np.clip(fused, 0.0, 255.0)).astype(np.uint8)
    raise ValueError(f"Unknown fusion mode {mode!r}")


def to_model_input(fused: np.ndarray, tensor_norm: str) -> np.ndarray:
    """Fused uint8 [256,256,3] -> float32 [1, 3, 224, 224] ready for the network."""
    if fused.shape != (FUSED_SIZE, FUSED_SIZE, 3):
        raise ValueError(f"Fused image must be 256x256x3, got {fused.shape}")
    offset = (FUSED_SIZE - MODEL_INPUT_SIZE) // 2
    crop = fused[offset : offset + MODEL_INPUT_SIZE, offset : offset + MODEL_INPUT_SIZE]
    tensor = crop.astype(np.float32) / np.float32(255.0)
    tensor = np.ascontiguousarray(tensor.transpose(2, 0, 1))
    if tensor_norm == IMAGENET:
        mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)[:, None, None]
        std = np.asarray(IMAGENET_STD, dtype=np.float32)[:, None, None]
        tensor = (tensor - mean) / std
    elif tensor_norm != UNIT:
        raise ValueError(f"Unknown tensor normalization {tensor_norm!r}")
    return tensor[None, ...]


# --------------------------------------------------------------------------- #
# End-to-end helpers
# --------------------------------------------------------------------------- #

def modality_arrays(paths: Mapping[str, str | Path], spec: FusionSpec) -> list[np.ndarray]:
    """Load every modality the spec needs, in spec order, as 256x256x3 uint8."""
    arrays: list[np.ndarray] = []
    for name in spec.modalities:
        path = paths.get(name)
        if path is None:
            raise ValueError(f"Model {spec.family} needs a {name} input")
        if name == VOICE:
            if spec.voice is None:
                raise ValueError("Spec has no voice recipe")
            arrays.append(gray_to_rgb(voice_spectrogram_256(path, spec.voice)))
        else:
            arrays.append(modality_image_256(path, spec.images[name]))
    return arrays


def fused_image(paths: Mapping[str, str | Path], spec: FusionSpec) -> np.ndarray:
    return fuse(modality_arrays(paths, spec), spec.weights, spec.fusion_norm)


def model_input(paths: Mapping[str, str | Path], spec: FusionSpec) -> np.ndarray:
    return to_model_input(fused_image(paths, spec), spec.tensor_norm)
