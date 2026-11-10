"""Per-family preprocessing specifications for the CoBas fusion checkpoints.

Every number here was read from the Colab training notebooks in the Drive
"Training Pipelines" folder (October 2026). The four families differ in how
each modality image is resized, how the images are combined, and how the fused
image is normalized before ResNet-18. A wrong spec silently degrades accuracy,
so the registry is explicit and checkpoint metadata is used to cross-check it.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

MMWAVE = "mmwave"
THERMAL = "thermal"
VOICE = "voice"
MODALITIES = (MMWAVE, THERMAL, VOICE)

CLASS_NAMES: tuple[str, ...] = tuple(f"{value}%" for value in range(0, 101, 10))

# Models are PyTorch checkpoints; either extension is accepted.
MODEL_SUFFIXES = (".pth", ".pt")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

FUSED_SIZE = 256
MODEL_INPUT_SIZE = 224

# Fusion arithmetic modes (see preprocess.fuse).
PER_IMAGE_MINMAX = "per_image_minmax"  # each image min-max -> [0,1], weighted sum, x255
DIV255 = "div255"  # each image /255, weighted sum, clip [0,1], x255
RAW255 = "raw255"  # weighted sum on 0..255, clip, round

# Tensor normalization after the crop/resize (see preprocess.to_model_input).
IMAGENET = "imagenet"  # ToTensor then Normalize(ImageNet mean/std)
UNIT = "unit"  # ToTensor only (0..1)

# How the fused (or single) image becomes the network input.
RESIZE256_CROP224 = "resize256_crop224"  # fusion families: 256x256 image, center crop 224
RESIZE224 = "resize224"  # voice standalone: Resize((224, 224)) bilinear, no crop
NATIVE = "native"  # mmWave standalone: ToTensor on the raw frame, no resize


@dataclass(frozen=True)
class VoiceSpec:
    """How a 2 s chirp recording becomes a spectrogram image."""

    fmin: float = 17_500.0
    fmax: float = 20_000.0
    n_mels: int = 224
    # "rint": np.rint(unit * 255); "floor": (255 * unit).astype(np.uint8)
    quantize: str = "rint"
    # Low frequencies must end up at the bottom of the image. The notebooks flip
    # either the mel matrix before the LANCZOS resize or the finished 256x256
    # image after it; the two differ by at most one LSB, so keep them distinct.
    flip_stage: str = "before_resize"
    # Fusion families save a grayscale 256x256 LANCZOS image; the voice
    # standalone applies matplotlib's magma colormap and keeps the native
    # n_mels x frames size (the network transform resizes it to 224x224).
    colormap: str | None = None
    save_size: int | None = 256


@dataclass(frozen=True)
class ImageSpec:
    """How a raster modality (mmWave JPG or thermal PNG) is prepared."""

    # PIL filter used to reach 256x256 from the source size; "none" keeps the
    # native frame (mmWave standalone).
    resize: str = "bilinear"  # "bilinear" | "lanczos" | "none"
    # Thermal-Voice quirk: Resize(256) -> CenterCrop(224) was saved to disk and
    # the fusion step later resized that 224 image back to 256 with LANCZOS.
    pre_crop_224: bool = False


@dataclass(frozen=True)
class FusionSpec:
    family: str
    model_filename: str
    modalities: tuple[str, ...]
    weights: tuple[float, ...]
    fusion_norm: str
    tensor_norm: str
    images: Mapping[str, ImageSpec] = field(default_factory=dict)
    voice: VoiceSpec | None = None
    weights_source: str = "registry default (equal)"
    tensor_pipeline: str = RESIZE256_CROP224

    def with_weights(self, weights: Sequence[float], source: str) -> "FusionSpec":
        total = float(sum(weights))
        if total <= 0:
            raise ValueError(f"Fusion weights must be positive: {weights}")
        normalized = tuple(float(w) / total for w in weights)
        if len(normalized) != len(self.modalities):
            raise ValueError(
                f"{len(normalized)} weights given for {len(self.modalities)} modalities"
            )
        return replace(self, weights=normalized, weights_source=source)

    @property
    def equal_weights(self) -> bool:
        return max(self.weights) - min(self.weights) < 1e-9

    def describe(self) -> str:
        parts = [
            f"family={self.family}",
            "modalities=" + ":".join(self.modalities),
            "weights=" + ":".join(f"{w:.4f}" for w in self.weights) + f" ({self.weights_source})",
            f"fusion={self.fusion_norm}",
            f"tensor={self.tensor_norm} via {self.tensor_pipeline}",
        ]
        for name in self.modalities:
            if name == VOICE and self.voice is not None:
                colour = f" {self.voice.colormap}" if self.voice.colormap else " gray"
                size = f" {self.voice.save_size}px" if self.voice.save_size else " native"
                parts.append(
                    f"voice=mel{self.voice.n_mels} {self.voice.fmin:g}-{self.voice.fmax:g}Hz "
                    f"{self.voice.quantize} flip:{self.voice.flip_stage}{colour}{size}"
                )
            elif name in self.images:
                image = self.images[name]
                extra = " pre-crop224" if image.pre_crop_224 else ""
                parts.append(f"{name}={image.resize}{extra}")
        return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Registry keyed by checkpoint filename
# --------------------------------------------------------------------------- #

THREE_MODAL = FusionSpec(
    family="mmwave_thermal_voice",
    model_filename="CoBas_mmwave_thermal_voice_fusion_model.pth",
    modalities=(MMWAVE, THERMAL, VOICE),
    weights=(1 / 3, 1 / 3, 1 / 3),
    fusion_norm=PER_IMAGE_MINMAX,
    tensor_norm=IMAGENET,
    images={MMWAVE: ImageSpec("bilinear"), THERMAL: ImageSpec("bilinear")},
    voice=VoiceSpec(quantize="rint", flip_stage="before_resize"),
)

MMWAVE_THERMAL = FusionSpec(
    family="mmwave_thermal",
    model_filename="CoBas_mmwave_thermal_fusion_model.pth",
    modalities=(MMWAVE, THERMAL),
    weights=(0.5, 0.5),
    fusion_norm=DIV255,
    tensor_norm=UNIT,
    images={MMWAVE: ImageSpec("bilinear"), THERMAL: ImageSpec("bilinear")},
    voice=None,
)

MMWAVE_VOICE = FusionSpec(
    family="mmwave_voice",
    model_filename="CoBas_mmwave_and_voice_fusion_model.pth",
    modalities=(MMWAVE, VOICE),
    weights=(0.5, 0.5),
    fusion_norm=RAW255,
    tensor_norm=UNIT,
    images={MMWAVE: ImageSpec("bilinear")},
    voice=VoiceSpec(quantize="rint", flip_stage="after_resize"),
)

THERMAL_VOICE = FusionSpec(
    family="thermal_voice",
    model_filename="CoBas_thermal_and_voice_model.pth",
    modalities=(THERMAL, VOICE),
    weights=(0.5, 0.5),
    fusion_norm=RAW255,
    tensor_norm=UNIT,
    images={THERMAL: ImageSpec("bilinear", pre_crop_224=True)},
    voice=VoiceSpec(quantize="floor", flip_stage="after_resize"),
)

# --------------------------------------------------------------------------- #
# Standalone (single-modality) models from the "Standalone" training notebooks
# --------------------------------------------------------------------------- #

THERMAL_ONLY = FusionSpec(
    family="thermal",
    model_filename="CoBas_thermal_model.pth",
    modalities=(THERMAL,),
    weights=(1.0,),
    fusion_norm=RAW255,  # a single image passes through unchanged
    tensor_norm=UNIT,  # Resize(256) -> CenterCrop(224) -> ToTensor
    images={THERMAL: ImageSpec("bilinear")},
)

MMWAVE_ONLY = FusionSpec(
    family="mmwave",
    model_filename="CoBas_mmWave_model.pth",
    modalities=(MMWAVE,),
    weights=(1.0,),
    fusion_norm=RAW255,
    tensor_norm=UNIT,  # ToTensor only: the raw 724x256 range-angle frame
    images={MMWAVE: ImageSpec("none")},
    tensor_pipeline=NATIVE,
)

VOICE_ONLY = FusionSpec(
    family="voice",
    model_filename="CoBas_voice_reflection_recognition_model.pth",
    modalities=(VOICE,),
    weights=(1.0,),
    fusion_norm=RAW255,
    tensor_norm=UNIT,  # Resize((224, 224)) -> ToTensor on the magma spectrogram
    voice=VoiceSpec(quantize="floor", flip_stage="before_resize", colormap="magma", save_size=None),
    tensor_pipeline=RESIZE224,
)

REGISTRY: dict[str, FusionSpec] = {
    spec.model_filename.lower(): spec
    for spec in (THREE_MODAL, MMWAVE_THERMAL, MMWAVE_VOICE, THERMAL_VOICE, THERMAL_ONLY, MMWAVE_ONLY, VOICE_ONLY)
}

FAMILIES: dict[str, FusionSpec] = {spec.family: spec for spec in REGISTRY.values()}

_RATIO_PATTERN = re.compile(r"^\s*(\d+)(?:\s*-\s*(\d+))+")
_META_KEY_TO_MODALITY = {"mmwave": MMWAVE, "thermal": THERMAL, "voice": VOICE}


def is_model_file(path: str | Path) -> bool:
    """True for a PyTorch checkpoint: the only model format CoBas uses."""
    return Path(path).suffix.lower() in MODEL_SUFFIXES


def spec_for_model_file(path: str | Path) -> FusionSpec:
    """Return the registry spec for a checkpoint filename (.pth or .pt)."""
    name = Path(path).name.lower()
    if name.endswith(".pt"):
        name = name[: -len(".pt")] + ".pth"  # the registry is keyed by the .pth names
    try:
        return REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(REGISTRY))
        raise KeyError(f"No preprocessing spec for {Path(path).name!r}. Known: {known}") from None


def weights_from_folder_name(name: str) -> tuple[int, ...] | None:
    """Parse '1 - 1 - 2 Fusions' into (1, 1, 2)."""
    match = _RATIO_PATTERN.match(name)
    if not match:
        return None
    numbers = re.findall(r"\d+", name[: match.end()])
    return tuple(int(n) for n in numbers)


def weights_from_metadata(spec: FusionSpec, metadata: Mapping[str, Any]) -> tuple[float, ...] | None:
    raw = metadata.get("fusion_weights")
    if not isinstance(raw, Mapping):
        return None
    by_modality: dict[str, float] = {}
    for key, value in raw.items():
        modality = _META_KEY_TO_MODALITY.get(str(key).lower())
        if modality is None:
            warnings.warn(f"Unknown fusion weight key {key!r} in checkpoint metadata")
            return None
        by_modality[modality] = float(value)
    if set(by_modality) != set(spec.modalities):
        warnings.warn(
            f"Checkpoint fusion weights {sorted(by_modality)} do not match the "
            f"{spec.family} modalities {spec.modalities}"
        )
        return None
    return tuple(by_modality[m] for m in spec.modalities)


def resolve_spec(
    model_path: str | Path,
    metadata: Mapping[str, Any] | None = None,
    *,
    weights_override: Sequence[float] | None = None,
) -> FusionSpec:
    """Build the spec for a model: registry, then checkpoint metadata, then folder name.

    Resolution order for the fusion weights:
    1. explicit ``weights_override`` (CLI),
    2. ``fusion_weights`` stored in the checkpoint (three-modal, mmWave-Thermal,
       mmWave-Voice families store it; Thermal-Voice does not),
    3. the ratio in the parent folder name, e.g. ``1 - 2 Fusions``,
    4. equal weights.
    """
    model_path = Path(model_path)
    spec = spec_for_model_file(model_path)
    metadata = metadata or {}

    if weights_override is not None:
        spec = spec.with_weights(weights_override, "command line")
    else:
        from_meta = weights_from_metadata(spec, metadata)
        if from_meta is not None:
            spec = spec.with_weights(from_meta, "checkpoint metadata")
        else:
            ratio = weights_from_folder_name(model_path.parent.name)
            if ratio is not None and len(ratio) == len(spec.modalities):
                spec = spec.with_weights(ratio, f"folder name {model_path.parent.name!r}")

    _cross_check_metadata(spec, metadata)
    return spec


def _cross_check_metadata(spec: FusionSpec, metadata: Mapping[str, Any]) -> None:
    """Warn when checkpoint metadata disagrees with the registry recipe."""
    mean = metadata.get("model_input_mean")
    std = metadata.get("model_input_std")
    if mean is not None and std is not None:
        is_unit = all(abs(float(m)) < 1e-9 for m in mean) and all(abs(float(s) - 1) < 1e-9 for s in std)
        expected = UNIT if is_unit else IMAGENET
        if expected != spec.tensor_norm:
            warnings.warn(
                f"Checkpoint says tensor normalization {expected!r} but the registry "
                f"recipe for {spec.family} uses {spec.tensor_norm!r}"
            )
    text = str(metadata.get("fusion_normalization", "")).lower()
    if text:
        uses_minmax = "min-max" in text and "no min-max" not in text
        if uses_minmax and spec.fusion_norm != PER_IMAGE_MINMAX:
            warnings.warn(f"Checkpoint describes min-max fusion; registry uses {spec.fusion_norm}")
        if "divided by 255" in text and spec.fusion_norm != DIV255:
            warnings.warn(f"Checkpoint describes /255 fusion; registry uses {spec.fusion_norm}")
        if text.strip() == "none" and spec.fusion_norm != RAW255:
            warnings.warn(f"Checkpoint describes raw fusion; registry uses {spec.fusion_norm}")
    band = metadata.get("voice_frequency_range_hz")
    if band and spec.voice is not None:
        lo, hi = (float(band[0]), float(band[1]))
        if abs(lo - spec.voice.fmin) > 1e-6 or abs(hi - spec.voice.fmax) > 1e-6:
            warnings.warn(
                f"Checkpoint voice band {lo:g}-{hi:g} Hz differs from registry "
                f"{spec.voice.fmin:g}-{spec.voice.fmax:g} Hz"
            )
