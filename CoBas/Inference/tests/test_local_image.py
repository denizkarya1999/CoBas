"""Local Image Inference: one picture from disk in, battery level out (no Tk window, no torch)."""

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from cobas_inference import preprocess, specs
from cobas_inference.predictor import Prediction, Predictor

APP_DIRECTORY = Path(__file__).resolve().parents[2] / "CoBas Battery Reader V1.0 App"
if str(APP_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(APP_DIRECTORY))

FIXTURES = Path(__file__).parent / "fixtures" / "pair0_100pct"
RAW = FIXTURES / "raw"
EXPECTED = FIXTURES / "expected"
THERMAL = RAW / "Thermal_Image_100_Percent_Position_1_001.png"
MMWAVE = RAW / "frame_000001.jpg"
VOICE = RAW / "CoBas_V1_PulseVoice_20261021_184200_357070_Position_1_001.wav"
CLASSES = specs.CLASS_NAMES


@pytest.fixture
def fixtures_available():
    if not THERMAL.is_file():
        pytest.skip("parity fixtures missing")


# --------------------------------------------------------------------------- #
# cobas_inference: the picture is the model's own input image
# --------------------------------------------------------------------------- #

def test_thermal_image_takes_the_standalone_recipe(fixtures_available):
    got = preprocess.image_model_input(THERMAL, specs.THERMAL_ONLY)
    np.testing.assert_array_equal(got, preprocess.model_input({specs.THERMAL: THERMAL}, specs.THERMAL_ONLY))
    assert got.shape == (1, 3, 224, 224)


def test_mmwave_frame_keeps_its_native_size(fixtures_available):
    got = preprocess.image_model_input(MMWAVE, specs.MMWAVE_ONLY)
    assert got.shape == (1, 3, 256, 724)
    np.testing.assert_array_equal(got, preprocess.model_input({specs.MMWAVE: MMWAVE}, specs.MMWAVE_ONLY))


def test_saved_fused_image_equals_the_chirp_pipeline(fixtures_available):
    """Opening the notebook's fused PNG feeds the network what the raw chirp would."""
    from_image = preprocess.image_model_input(EXPECTED / "thermal_voice_1-1" / "fused.png", specs.THERMAL_VOICE)
    from_chirp = preprocess.model_input({specs.THERMAL: THERMAL, specs.VOICE: VOICE}, specs.THERMAL_VOICE)
    np.testing.assert_array_equal(from_image, from_chirp)


def test_saved_spectrogram_equals_the_voice_pipeline(fixtures_available):
    from_image = preprocess.image_model_input(EXPECTED / "voice_standalone" / "spectrogram.png", specs.VOICE_ONLY)
    from_chirp = preprocess.model_input({specs.VOICE: VOICE}, specs.VOICE_ONLY)
    # Same allowance as test_parity: the Colab colour table differs from the
    # Pi's by one grey level on a handful of pixels.
    difference = np.abs(from_image - from_chirp) * 255
    assert difference.max() <= 1.0 + 1e-4
    assert (difference > 0).mean() <= 0.0005


def test_fused_image_of_another_size_is_resized(tmp_path):
    path = tmp_path / "picture.png"
    Image.new("RGB", (300, 200), (10, 120, 240)).save(path)
    got = preprocess.image_model_input(path, specs.THREE_MODAL)
    assert got.shape == (1, 3, 224, 224)
    expected = (np.array([10, 120, 240]) / 255 - specs.IMAGENET_MEAN) / specs.IMAGENET_STD
    np.testing.assert_allclose(got[0, :, 0, 0], expected, rtol=1e-5)


def test_photo_is_refused_by_the_native_size_model(tmp_path):
    path = tmp_path / "photo.png"
    Image.new("RGB", (2000, 1500)).save(path)
    with pytest.raises(ValueError, match="too large"):
        preprocess.image_model_input(path, specs.MMWAVE_ONLY)
    # The same photo is fine for a model that resizes its input.
    assert preprocess.image_model_input(path, specs.THERMAL_ONLY).shape == (1, 3, 224, 224)


def test_predict_image_runs_the_network_on_the_picture(fixtures_available):
    seen = []

    def runner(x):
        seen.append(x)
        logits = np.zeros((1, len(CLASSES)), dtype=np.float32)
        logits[0, 6] = 9.0
        return logits

    predictor = Predictor(runner, specs.THERMAL_ONLY, CLASSES, Path("CoBas_thermal_model.pth"), {})
    prediction = predictor.predict_image(THERMAL)
    assert prediction.label == "60%"
    np.testing.assert_array_equal(seen[0], preprocess.image_model_input(THERMAL, specs.THERMAL_ONLY))


def _recording_predictor(spec, name):
    seen = []

    def runner(x):
        seen.append(x)
        return np.zeros((1, len(CLASSES)), dtype=np.float32)

    return Predictor(runner, spec, CLASSES, Path(name), {}), seen


def test_replay_as_images_opens_the_frame_or_a_saved_fused_image(fixtures_available, tmp_path):
    """``replay --as-images`` runs a dataset through the Local Image Inference recipe."""
    from cobas_inference.cli import build_parser, image_classifier
    from cobas_inference.session import ChirpTriplet

    chirp = ChirpTriplet(position=1, index=1, global_index=1, mmwave=MMWAVE, thermal=THERMAL, voice=VOICE)
    predictor, seen = _recording_predictor(specs.THERMAL_ONLY, "CoBas_thermal_model.pth")
    image_classifier(predictor, tmp_path)(chirp)
    np.testing.assert_array_equal(seen[0], preprocess.image_model_input(THERMAL, specs.THERMAL_ONLY))
    assert not list(tmp_path.iterdir())  # the saved frame itself was opened

    predictor, seen = _recording_predictor(specs.THERMAL_VOICE, "CoBas_thermal_and_voice_model.pth")
    image_classifier(predictor, tmp_path)(chirp)
    np.testing.assert_array_equal(seen[0], preprocess.model_input(chirp.paths(), specs.THERMAL_VOICE))
    assert [p.name for p in tmp_path.iterdir()] == ["picture.png"]

    args = build_parser().parse_args(["replay", "--model", "m.pth", "some/dir"])
    assert args.as_images is False
    assert build_parser().parse_args(["replay", "--model", "m.pth", "d", "--as-images"]).as_images is True


# --------------------------------------------------------------------------- #
# App screen logic
# --------------------------------------------------------------------------- #

LocalImageInference = pytest.importorskip("LocalImageInference")
InferenceMode = pytest.importorskip("InferenceMode")


def _models(tmp_path):
    for folder, name in (
        ("1 - 1 - 1 Fusions", "CoBas_mmwave_thermal_voice_fusion_model.pth"),
        ("Standalone", "CoBas_mmWave_model.pth"),
        ("Standalone", "CoBas_thermal_model.pth"),
        ("Standalone", "CoBas_voice_reflection_recognition_model.pth"),
    ):
        (tmp_path / folder).mkdir(exist_ok=True)
        (tmp_path / folder / name).write_bytes(b"x")
    return InferenceMode.discover_models(tmp_path)


def test_standalone_models_come_first_and_thermal_is_the_default(tmp_path):
    choices = LocalImageInference.image_models(_models(tmp_path))
    assert [c.family for c in choices] == ["mmwave", "thermal", "voice", "mmwave_thermal_voice"]
    assert LocalImageInference.default_image_model(choices).family == "thermal"
    assert LocalImageInference.default_image_model(choices[-1:]).family == "mmwave_thermal_voice"
    assert LocalImageInference.default_image_model([]) is None
    described = {c.family: LocalImageInference.expected_image(c) for c in choices}
    assert described == {
        "mmwave": "a mmWave range-angle frame",
        "thermal": "a thermal camera image",
        "voice": "a voice spectrogram image",
        "mmwave_thermal_voice": "a fused mmWave + Thermal + Voice image",
    }


def test_list_directory_shows_folders_and_images_in_natural_order(tmp_path):
    for name in ("Session_10", "Session_2", ".hidden"):
        (tmp_path / name).mkdir()
    for name in ("frame_000010.jpg", "frame_000002.JPG", "thermal.png", "notes.txt", "voice.wav", ".secret.png"):
        (tmp_path / name).write_bytes(b"x")
    folders, images = LocalImageInference.list_directory(tmp_path)
    assert [p.name for p in folders] == ["Session_2", "Session_10"]
    assert [p.name for p in images] == ["frame_000002.JPG", "frame_000010.jpg", "thermal.png"]
    with pytest.raises(OSError):
        LocalImageInference.list_directory(tmp_path / "missing")


def test_shorten_keeps_the_start_or_the_end():
    assert LocalImageInference.shorten("short", 10) == "short"
    assert LocalImageInference.shorten("abcdefghij", 6) == "abcde…"
    assert LocalImageInference.shorten("/home/tailab/Datasets", 9, keep_end=True) == "…Datasets"


class StubPredictor:
    def __init__(self, label):
        self.label = label
        self.images = []

    def predict_image(self, path):
        self.images.append(Path(path))
        if self.label is None:
            raise ValueError("not an image")
        probabilities = np.zeros(len(CLASSES))
        probabilities[CLASSES.index(self.label)] = 1.0
        return Prediction(probabilities, CLASSES)


def _runner(loaded, results, errors, labels):
    def load(choice):
        loaded.append(choice.family)
        return StubPredictor(labels[choice.family])

    def post_ui(callback, *args):
        callback(*args)

    return LocalImageInference.LocalImageRunner(
        post_ui, lambda request, label: results.append((request, label)),
        lambda request, message: errors.append((request, message)), load_predictor=load,
    )


def test_runner_loads_each_model_once_and_reports_the_label(tmp_path):
    choices = {c.family: c for c in _models(tmp_path)}
    loaded, results, errors = [], [], []
    runner = _runner(loaded, results, errors, {"thermal": "60%", "mmwave": "10%", "voice": None})
    runner.estimate(1, choices["thermal"], "a.png").join(10)
    runner.estimate(2, choices["thermal"], "b.png").join(10)
    runner.estimate(3, choices["mmwave"], "b.png").join(10)
    assert loaded == ["thermal", "mmwave"]
    assert results == [(1, "60%"), (2, "60%"), (3, "10%")]
    runner.estimate(4, choices["voice"], "b.png").join(10)
    assert errors == [(4, "not an image")]


def test_runner_skips_a_request_that_was_superseded(tmp_path):
    choices = {c.family: c for c in _models(tmp_path)}
    loaded, results, errors = [], [], []
    runner = _runner(loaded, results, errors, {"thermal": "60%"})
    with runner._work:  # A prediction is in flight while two more requests arrive.
        first = runner.estimate(1, choices["thermal"], "a.png")
        second = runner.estimate(2, choices["thermal"], "b.png")
    first.join(10)
    second.join(10)
    assert results == [(2, "60%")] and not errors
