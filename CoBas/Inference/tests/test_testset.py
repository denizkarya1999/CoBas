"""The test dataset builder: a few pictures per battery level for every model (no torch)."""

import csv
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from cobas_inference import preprocess, specs, testset
from cobas_inference.predictor import Prediction

RAW = Path(__file__).parent / "fixtures" / "pair0_100pct" / "raw"
THERMAL = RAW / "Thermal_Image_100_Percent_Position_1_001.png"
MMWAVE = RAW / "frame_000001.jpg"
VOICE = RAW / "CoBas_V1_PulseVoice_20261021_184200_357070_Position_1_001.wav"
CLASSES = specs.CLASS_NAMES


def _no_metadata(path):
    return {}


@pytest.fixture
def source(tmp_path):
    """Two battery levels with three chirps each, plus an aborted (empty) session."""
    if not THERMAL.is_file():
        pytest.skip("parity fixtures missing")
    root = tmp_path / "Source Dataset"
    for percent in (0, 100):
        session = root / f"{percent}_Percent_Battery" / "Session_20261021_183911_abc"
        for folder in ("Thermal Frames", "mmWave Frames", "Voices"):
            (session / folder).mkdir(parents=True)
        for chirp in (1, 2, 3):
            shutil.copyfile(THERMAL, session / "Thermal Frames" / f"Thermal_Image_{percent}_Percent_Position_1_{chirp:03d}.png")
            shutil.copyfile(MMWAVE, session / "mmWave Frames" / f"frame_{chirp:06d}.jpg")
            shutil.copyfile(VOICE, session / "Voices" / f"CoBas_V1_PulseVoice_x_Position_1_{chirp:03d}.wav")
    (root / "100_Percent_Battery" / "Session_20261021_180000_aborted" / "Thermal Frames").mkdir(parents=True)
    return root


@pytest.fixture
def models(tmp_path):
    root = tmp_path / "models"
    for folder, name in (
        ("Standalone", "CoBas_thermal_model.pth"),
        ("Standalone", "CoBas_mmWave_model.pth"),
        ("1 - 2 Fusions", "CoBas_thermal_and_voice_model.pth"),
        ("1 - 2 Fusions", "notes.pth"),
    ):
        (root / folder).mkdir(parents=True, exist_ok=True)
        (root / folder / name).write_bytes(b"x")
    return root


def test_pick_evenly_spreads_over_the_whole_list():
    chirps = list(range(1, 201))
    assert testset.pick_evenly(chirps, 10) == [11, 31, 51, 71, 91, 111, 131, 151, 171, 191]
    assert testset.pick_evenly([1, 2, 3], 10) == [1, 2, 3]
    assert testset.pick_evenly(chirps, 1) == [101]


def test_models_are_named_in_full_and_standalone_comes_first(models):
    found = testset.find_models(models, _no_metadata)
    assert [str(m.folder) for m in found] == [
        "Standalone/CoBas Thermal Model",
        "Standalone/CoBas mmWave Model",
        "1 - 2 Fusions/CoBas Thermal and Voice Model",
    ]
    assert found[2].spec.weights == pytest.approx((1 / 3, 2 / 3))  # the folder's own ratio
    assert set(testset.MODEL_NAMES) == set(specs.FAMILIES)


def test_build_writes_the_pictures_each_model_opens(source, models, tmp_path):
    out = tmp_path / "CoBas Test Dataset"
    rows = testset.build(source, models, out, per_class=2, read_metadata=_no_metadata, progress=lambda line: None)
    assert len(rows) == 3 * 2 * 2  # models x levels x pictures

    thermal = out / "Standalone" / "CoBas Thermal Model" / "100_Percent_Battery"
    assert sorted(p.name for p in thermal.iterdir()) == [
        "Thermal_Image_100_Percent_Position_1_001.png", "Thermal_Image_100_Percent_Position_1_003.png",
    ]
    assert (thermal / "Thermal_Image_100_Percent_Position_1_001.png").read_bytes() == THERMAL.read_bytes()

    mmwave = out / "Standalone" / "CoBas mmWave Model" / "0_Percent_Battery" / "mmWave_Frame_0_Percent_Position_1_001.jpg"
    assert mmwave.read_bytes() == MMWAVE.read_bytes()  # copied, never re-encoded

    fused = out / "1 - 2 Fusions" / "CoBas Thermal and Voice Model" / "100_Percent_Battery" / "Fused_Image_100_Percent_Position_1_001.png"
    spec = testset.find_models(models, _no_metadata)[2].spec
    expected = preprocess.fused_image({specs.THERMAL: THERMAL, specs.VOICE: VOICE}, spec)
    np.testing.assert_array_equal(np.asarray(Image.open(fused)), expected)
    np.testing.assert_array_equal(
        preprocess.image_model_input(fused, spec),
        preprocess.model_input({specs.THERMAL: THERMAL, specs.VOICE: VOICE}, spec),
    )

    with open(out / "manifest.csv", newline="", encoding="utf-8") as handle:
        manifest = list(csv.DictReader(handle))
    assert len(manifest) == len(rows)
    assert all((out / row["picture"]).is_file() for row in manifest)
    assert manifest[-1]["sources"].startswith("Thermal_Image_100_Percent_Position_1_003.png; CoBas_V1_PulseVoice")
    assert "3 models x 2 battery levels x 2 pictures = 4 pictures per model, 12 in total." in (out / "README.txt").read_text()

    with pytest.raises(FileExistsError):
        testset.build(source, models, out, read_metadata=_no_metadata)


def test_check_reads_the_expected_level_from_the_folder(source, models, tmp_path):
    out = tmp_path / "CoBas Test Dataset"
    testset.build(source, models, out, per_class=1, read_metadata=_no_metadata, progress=lambda line: None)

    class AlwaysFull:
        def predict_image(self, path):
            probabilities = np.zeros(len(CLASSES))
            probabilities[-1] = 1.0
            return Prediction(probabilities, CLASSES)

    lines = []
    results = testset.check(out, models, lambda path: AlwaysFull(), lines.append, _no_metadata)
    assert [(r["pictures"], r["correct"]) for r in results] == [(2, 1)] * 3
    assert results[0]["misses"] == [("Thermal_Image_0_Percent_Position_1_002.png", "0%", "100%")]
    assert "Standalone/CoBas Thermal Model: 1/2 correct" in lines
