"""Pixel-parity tests against images saved by the Colab training notebooks.

``tests/fixtures/pair0_100pct`` holds the raw chirp-0 capture of the 100 %
battery session (Oct 21 2026) plus the intermediate and fused images the
notebooks wrote for that same pair (``pair_index`` 0 of class ``100_ Percent``
on Drive). If our preprocessing reproduces those bytes, the device feeds the
network exactly what it was trained on.
"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from cobas_inference import preprocess, specs

FIXTURES = Path(__file__).parent / "fixtures" / "pair0_100pct"
RAW = FIXTURES / "raw"
EXPECTED = FIXTURES / "expected"

RAW_PATHS = {
    specs.MMWAVE: RAW / "frame_000001.jpg",
    specs.THERMAL: RAW / "Thermal_Image_100_Percent_Position_1_001.png",
    specs.VOICE: RAW / "CoBas_V1_PulseVoice_20261021_184200_357070_Position_1_001.wav",
}


def _load(path: Path, mode: str | None = None) -> np.ndarray:
    if not path.is_file():
        pytest.skip(f"fixture missing: {path.name}")
    with Image.open(path) as image:
        if mode:
            image = image.convert(mode)
        return np.asarray(image)


def _assert_identical(
    got: np.ndarray,
    expected: np.ndarray,
    what: str,
    tolerance: int = 0,
    max_fraction_differing: float = 0.0,
) -> None:
    """Exact match by default; ``tolerance``/``max_fraction_differing`` relax it.

    The round-to-nearest spectrogram families reproduce the Colab images except
    for a handful of pixels that differ by exactly one grey level (8 of 65,536
    for the checked pair). No arithmetic variant removes them and none of the
    values sits near a rounding tie, so this is a last-bit floating-point
    difference between the Colab (x86) and Raspberry Pi (aarch64) FFT/mel
    stacks, not a recipe error. Those tests allow at most one grey level on at
    most 0.05 % of pixels; everything else must match byte for byte.
    """
    assert got.shape == expected.shape, f"{what}: shape {got.shape} != {expected.shape}"
    diff = np.abs(got.astype(int) - expected.astype(int))
    assert int(diff.max()) <= tolerance, (
        f"{what}: max difference {int(diff.max())} exceeds {tolerance}"
    )
    fraction = float((diff > 0).sum()) / diff.size
    assert fraction <= max_fraction_differing, (
        f"{what}: {fraction:.4%} of values differ (allowed {max_fraction_differing:.4%})"
    )


# Allowance for the round-to-nearest spectrogram families (see _assert_identical).
ULP_TOLERANCE = dict(tolerance=1, max_fraction_differing=0.0005)


@pytest.fixture(scope="module")
def raw_available():
    missing = [p.name for p in RAW_PATHS.values() if not p.is_file()]
    if missing:
        pytest.skip(f"raw fixtures missing: {missing}")


# --------------------------------------------------------------------------- #
# Three-modal 1-1-1 (per-image min-max, ImageNet normalization)
# --------------------------------------------------------------------------- #

class TestThreeModal:
    spec = specs.THREE_MODAL
    folder = EXPECTED / "three_modal_1-1-1"

    def test_mmwave_resize(self, raw_available):
        got = preprocess.modality_image_256(RAW_PATHS[specs.MMWAVE], self.spec.images[specs.MMWAVE])
        _assert_identical(got, _load(self.folder / "mmwave.png", "RGB"), "mmWave 256")

    def test_thermal_resize(self, raw_available):
        got = preprocess.modality_image_256(RAW_PATHS[specs.THERMAL], self.spec.images[specs.THERMAL])
        _assert_identical(got, _load(self.folder / "thermal.png", "RGB"), "thermal 256")

    def test_voice_spectrogram(self, raw_available):
        got = preprocess.voice_spectrogram_256(RAW_PATHS[specs.VOICE], self.spec.voice)
        _assert_identical(got, _load(self.folder / "voice.png", "L"), "voice spectrogram", **ULP_TOLERANCE)

    def test_fused_from_saved_voice_is_exact(self, raw_available):
        # Isolates the mmWave/thermal resizes and the fusion arithmetic from the spectrogram.
        mm = preprocess.modality_image_256(RAW_PATHS[specs.MMWAVE], self.spec.images[specs.MMWAVE])
        th = preprocess.modality_image_256(RAW_PATHS[specs.THERMAL], self.spec.images[specs.THERMAL])
        voice = preprocess.gray_to_rgb(_load(self.folder / "voice.png", "L"))
        got = preprocess.fuse([mm, th, voice], self.spec.weights, self.spec.fusion_norm)
        _assert_identical(got, _load(self.folder / "fused.png", "RGB"), "fused 1-1-1 (saved voice)")

    def test_fused(self, raw_available):
        got = preprocess.fused_image(RAW_PATHS, self.spec)
        _assert_identical(got, _load(self.folder / "fused.png", "RGB"), "fused 1-1-1", **ULP_TOLERANCE)


# --------------------------------------------------------------------------- #
# Thermal-Voice 1-1 (224 pre-crop quirk, floor quantization, flip after resize)
# --------------------------------------------------------------------------- #

class TestThermalVoice:
    spec = specs.THERMAL_VOICE
    folder = EXPECTED / "thermal_voice_1-1"

    def test_thermal_224_intermediate(self, raw_available):
        image = preprocess.resize(preprocess.load_rgb(RAW_PATHS[specs.THERMAL]), 256, "bilinear")
        got = np.asarray(preprocess.center_crop(image, 224))
        _assert_identical(got, _load(self.folder / "thermal224.png", "RGB"), "thermal 224 crop")

    def test_voice_spectrogram_before_flip(self, raw_available):
        # The notebook saved the spectrogram un-flipped and flipped it at fusion time.
        got = preprocess.voice_spectrogram_256(RAW_PATHS[specs.VOICE], self.spec.voice)
        _assert_identical(np.flipud(got), _load(self.folder / "voice.png", "L"), "voice spectrogram")

    def test_fused(self, raw_available):
        got = preprocess.fused_image(RAW_PATHS, self.spec)
        _assert_identical(got, _load(self.folder / "fused.png", "RGB"), "fused thermal-voice")


# --------------------------------------------------------------------------- #
# mmWave-Voice 1-1 (raw 0..255 average, rint quantization, flip after resize)
# --------------------------------------------------------------------------- #

class TestMmwaveVoice:
    spec = specs.MMWAVE_VOICE
    folder = EXPECTED / "mmwave_voice_1-1"

    def test_mmwave_resize(self, raw_available):
        got = preprocess.modality_image_256(RAW_PATHS[specs.MMWAVE], self.spec.images[specs.MMWAVE])
        _assert_identical(got, _load(self.folder / "mmwave.png", "RGB"), "mmWave 256")

    def test_voice_spectrogram_before_flip(self, raw_available):
        got = preprocess.voice_spectrogram_256(RAW_PATHS[specs.VOICE], self.spec.voice)
        _assert_identical(np.flipud(got), _load(self.folder / "voice.png", "L"), "voice spectrogram", **ULP_TOLERANCE)

    def test_fused_from_saved_voice_is_exact(self, raw_available):
        mm = preprocess.modality_image_256(RAW_PATHS[specs.MMWAVE], self.spec.images[specs.MMWAVE])
        voice = preprocess.gray_to_rgb(np.flipud(_load(self.folder / "voice.png", "L")))
        got = preprocess.fuse([mm, voice], self.spec.weights, self.spec.fusion_norm)
        _assert_identical(got, _load(self.folder / "fused.png", "RGB"), "fused mmwave-voice (saved voice)")

    def test_fused(self, raw_available):
        got = preprocess.fused_image(RAW_PATHS, self.spec)
        _assert_identical(got, _load(self.folder / "fused.png", "RGB"), "fused mmwave-voice", **ULP_TOLERANCE)


# --------------------------------------------------------------------------- #
# mmWave-Thermal 1-1 has no saved PNGs on Drive (it cached a uint8 array), so
# check the recipe against the checkpoint's own description instead.
# --------------------------------------------------------------------------- #

def test_mmwave_thermal_recipe_matches_checkpoint_description(raw_available):
    spec = specs.MMWAVE_THERMAL
    mm = preprocess.modality_image_256(RAW_PATHS[specs.MMWAVE], spec.images[specs.MMWAVE])
    th = preprocess.modality_image_256(RAW_PATHS[specs.THERMAL], spec.images[specs.THERMAL])
    # Notebook: float32 arrays, 'rint(clip(0.5*(mmWave/255) + 0.5*(Thermal/255), 0, 1)*255)'
    mm32, th32 = mm.astype(np.float32) / 255.0, th.astype(np.float32) / 255.0
    expected = np.rint(np.clip(0.5 * mm32 + 0.5 * th32, 0.0, 1.0) * 255.0).astype(np.uint8)
    got = preprocess.fuse([mm, th], spec.weights, spec.fusion_norm)
    _assert_identical(got, expected, "fused mmwave-thermal")
