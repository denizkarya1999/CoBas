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
    NATIVE,
    PER_IMAGE_MINMAX,
    RAW255,
    RESIZE224,
    RESIZE256_CROP224,
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

# The mmWave standalone runs on the frame at its native size (724x256, about
# 0.19 megapixels). A photo opened by mistake would cost gigabytes of
# activations on the Pi, so anything far beyond a radar frame is refused.
NATIVE_MAX_PIXELS = 1_000_000

# Voice recordings Local File Inference accepts. Lossless only: a lossy codec
# (OGG, MP3) throws away the 17.5-20 kHz band the voice model reads.
AUDIO_SUFFIXES = (".wav", ".flac")


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
    """Raster modality -> uint8 [256, 256, 3], exactly as the notebooks saved it.

    With ``spec.resize == "none"`` (mmWave standalone) the native frame is
    returned unchanged, e.g. [256, 724, 3].
    """
    image = load_rgb(path)
    if spec.resize == "none":
        return np.asarray(image, dtype=np.uint8)
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


def spectrogram_to_rgb(db: np.ndarray, colormap: str) -> np.ndarray:
    """Voice standalone: per-clip min-max, matplotlib colormap, floor to uint8 RGB."""
    import matplotlib

    db = db.astype(np.float32, copy=False)
    db_min, db_max = db.min(), db.max()  # np.float32 scalars, as in the notebook
    if db_max > db_min:
        unit = (db - db_min) / (db_max - db_min)
    else:
        unit = np.zeros_like(db)
    try:
        cmap = matplotlib.colormaps[colormap]
    except AttributeError:  # older matplotlib
        from matplotlib import cm

        cmap = cm.get_cmap(colormap)
    coloured = cmap(unit)[:, :, :3]
    return (coloured * 255).astype(np.uint8)


def voice_spectrogram_256(path: str | Path, spec: VoiceSpec) -> np.ndarray:
    """WAV -> spectrogram image with low frequencies at the bottom.

    Fusion families: uint8 [256, 256] grayscale (LANCZOS). Voice standalone
    (``colormap`` set, ``save_size`` None): uint8 [n_mels, frames, 3] at native size.
    """
    if spec.flip_stage not in ("before_resize", "after_resize"):
        raise ValueError(f"Unknown flip stage {spec.flip_stage!r}")
    signal, sample_rate = load_audio(path)
    db = mel_db(signal, sample_rate, spec)
    if spec.colormap:
        matrix = spectrogram_to_rgb(db, spec.colormap)
    else:
        matrix = spectrogram_to_uint8(db, spec)
    if spec.flip_stage == "before_resize":
        matrix = np.flipud(matrix)
    if spec.save_size is None:
        if spec.flip_stage == "after_resize":
            matrix = np.flipud(matrix)
        return np.ascontiguousarray(matrix)
    image = Image.fromarray(np.ascontiguousarray(matrix), mode="L" if matrix.ndim == 2 else "RGB")
    image = image.resize((spec.save_size, spec.save_size), Image.Resampling.LANCZOS)
    if spec.flip_stage == "after_resize":
        image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
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


def to_model_input(fused: np.ndarray, tensor_norm: str, pipeline: str = RESIZE256_CROP224) -> np.ndarray:
    """Fused/single uint8 image [H, W, 3] -> float32 [1, 3, h, w] ready for the network.

    - RESIZE256_CROP224: 256x256 input, center crop to 224 (fusion families)
    - RESIZE224: torchvision Resize((224, 224)), bilinear (voice standalone)
    - NATIVE: no resize or crop (mmWave standalone, 256x724)
    """
    if fused.ndim != 3 or fused.shape[2] != 3:
        raise ValueError(f"Image must be HxWx3, got {fused.shape}")
    if pipeline == RESIZE256_CROP224:
        if fused.shape != (FUSED_SIZE, FUSED_SIZE, 3):
            raise ValueError(f"Fused image must be 256x256x3, got {fused.shape}")
        offset = (FUSED_SIZE - MODEL_INPUT_SIZE) // 2
        crop = fused[offset : offset + MODEL_INPUT_SIZE, offset : offset + MODEL_INPUT_SIZE]
    elif pipeline == RESIZE224:
        image = Image.fromarray(np.ascontiguousarray(fused)).resize(
            (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), Image.Resampling.BILINEAR
        )
        crop = np.asarray(image, dtype=np.uint8)
    elif pipeline == NATIVE:
        crop = fused
    else:
        raise ValueError(f"Unknown tensor pipeline {pipeline!r}")
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
            spectrogram = voice_spectrogram_256(path, spec.voice)
            arrays.append(gray_to_rgb(spectrogram) if spectrogram.ndim == 2 else spectrogram)
        else:
            arrays.append(modality_image_256(path, spec.images[name]))
    return arrays


def fused_image(paths: Mapping[str, str | Path], spec: FusionSpec) -> np.ndarray:
    return fuse(modality_arrays(paths, spec), spec.weights, spec.fusion_norm)


def model_input(paths: Mapping[str, str | Path], spec: FusionSpec) -> np.ndarray:
    return to_model_input(fused_image(paths, spec), spec.tensor_norm, spec.tensor_pipeline)


def is_raw_frame_model(spec: FusionSpec) -> bool:
    """True for the thermal and mmWave standalone models, whose input is one sensor frame."""
    return len(spec.modalities) == 1 and spec.modalities[0] in spec.images


def is_audio(path: str | Path) -> bool:
    """True for a voice recording (by file extension)."""
    return Path(path).suffix.lower() in AUDIO_SUFFIXES


def reads_voice_recordings(spec: FusionSpec) -> bool:
    """Only the voice standalone can work from a voice recording alone."""
    return spec.modalities == (VOICE,)


def require_voice_model(spec: FusionSpec) -> None:
    if not reads_voice_recordings(spec):
        raise ValueError("a voice recording can only be read by the voice standalone model")


def file_model_input(path: str | Path, spec: FusionSpec) -> np.ndarray:
    """One file from disk -> network input, with no chirp capture behind it.

    A voice recording is converted to the voice standalone's spectrogram. The
    thermal and mmWave standalone models take the raw sensor frame through
    their usual recipe. Every other picture is taken as the image the notebook
    fed the network: the fused image for the fusion families (256x256; any
    other size is resized to it) and the magma spectrogram for the voice
    standalone.
    """
    if is_audio(path):
        require_voice_model(spec)
        return model_input({VOICE: path}, spec)
    if is_raw_frame_model(spec):
        with Image.open(path) as image:
            width, height = image.size
        if spec.tensor_pipeline == NATIVE and width * height > NATIVE_MAX_PIXELS:
            raise ValueError(
                f"{width}x{height} is too large for this model, "
                "which runs on a radar frame at its own size (724x256)"
            )
        return model_input({spec.modalities[0]: path}, spec)
    image = load_rgb(path)
    if spec.tensor_pipeline == RESIZE256_CROP224 and image.size != (FUSED_SIZE, FUSED_SIZE):
        image = resize(image, FUSED_SIZE, "bilinear")
    return to_model_input(np.asarray(image, dtype=np.uint8), spec.tensor_norm, spec.tensor_pipeline)
