"""Headless tests for the app's Inference Mode engine (no Tk window, no torch)."""

import sys
import threading
from pathlib import Path

import numpy as np
import pytest

APP_DIRECTORY = Path(__file__).resolve().parents[2] / "CoBas Battery Reader V1.0 App"
if str(APP_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(APP_DIRECTORY))

InferenceMode = pytest.importorskip("InferenceMode")
from cobas_inference.predictor import Prediction  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures" / "pair0_100pct" / "raw"
CLASSES = tuple(f"{v}%" for v in range(0, 101, 10))


class StubPredictor:
    """Stands in for cobas_inference.Predictor without loading torch."""

    class_names = CLASSES

    def __init__(self, label="60%"):
        self.label = label
        self.calls = []

    def requires(self):
        return ("mmwave", "thermal", "voice")

    def predict_paths(self, paths):
        self.calls.append(dict(paths))
        probabilities = np.full(len(CLASSES), 0.01)
        probabilities[CLASSES.index(self.label)] = 1.0 - 0.01 * (len(CLASSES) - 1)
        return Prediction(probabilities, CLASSES, 12.0)


def _immediate_post_ui(callback, *args, token=None):
    callback(*args)


def test_discover_models_reads_ratio_folders(tmp_path):
    (tmp_path / "1 - 1 - 1 Fusions").mkdir()
    (tmp_path / "1 - 1 - 1 Fusions" / "CoBas_mmwave_thermal_voice_fusion_model.pth").write_bytes(b"x")
    (tmp_path / "2 - 1 Fusions").mkdir()
    (tmp_path / "2 - 1 Fusions" / "CoBas_thermal_and_voice_model.pth").write_bytes(b"x")
    (tmp_path / "2 - 1 Fusions" / "unrelated.pth").write_bytes(b"x")
    choices = InferenceMode.discover_models(tmp_path)
    assert [c.family for c in choices] == ["mmwave_thermal_voice", "thermal_voice"]
    assert choices[0].modalities == ("mmwave", "thermal", "voice")
    assert "1 - 1 - 1 Fusions" in choices[0].label
    assert InferenceMode.default_model(choices) is choices[0]
    assert InferenceMode.discover_models(tmp_path / "missing") == []


def test_wait_for_stable_file_waits_for_complete_image(tmp_path):
    from PIL import Image

    target = tmp_path / "frame.jpg"
    full = Image.new("RGB", (64, 32), (10, 20, 30))
    import io

    buffer = io.BytesIO()
    full.save(buffer, format="JPEG")
    data = buffer.getvalue()

    def slow_writer():
        with open(target, "wb") as handle:
            handle.write(data[: len(data) // 2])
            handle.flush()
            threading.Event().wait(0.25)
            handle.write(data[len(data) // 2 :])

    threading.Thread(target=slow_writer).start()
    assert InferenceMode.wait_for_stable_file(target, timeout_seconds=3.0) == target
    assert InferenceMode.wait_for_stable_file(tmp_path / "never.jpg", timeout_seconds=0.2) is None


def test_engine_classifies_fixture_chirp_and_writes_csv(tmp_path):
    if not (FIXTURES / "frame_000001.jpg").is_file():
        pytest.skip("fixtures missing")
    results, statuses, errors = [], [], []
    engine = InferenceMode.InferenceEngine(
        _immediate_post_ui, results.append, statuses.append, errors.append
    )
    stub = StubPredictor("60%")
    with engine._lock:
        engine.predictor = stub
        engine.choice = InferenceMode.ModelChoice(Path("m/CoBas_mmwave_thermal_voice_fusion_model.pth"), "mmwave_thermal_voice", stub.requires(), "m")
    assert engine.is_ready
    session = tmp_path / "Session_test"
    engine.start_session(session)
    job = InferenceMode.ChirpJob(
        token=1, position=1, index=1, global_index=1,
        thermal_path=str(FIXTURES / "Thermal_Image_100_Percent_Position_1_001.png"),
        thermal_fallback_path=str(tmp_path / "missing.png"),
        voice_path=str(FIXTURES / "CoBas_V1_PulseVoice_20261021_184200_357070_Position_1_001.wav"),
        mmwave_path=str(FIXTURES / "frame_000001.jpg"),
    )
    engine.enqueue(job)
    engine.enqueue(InferenceMode.ChirpJob(**{**job.__dict__, "index": 2, "global_index": 2}))
    engine.stop()
    engine._worker.join(timeout=10)
    assert not errors, errors
    assert len(results) == 2
    assert results[-1].vote_label == "60%" and results[-1].count == 2
    assert results[-1].consistency == 1.0
    assert set(stub.calls[0]) == {"mmwave", "thermal", "voice"}
    # Inference leaves no prediction files behind.
    assert not list(session.rglob("*.csv"))
    assert InferenceMode.format_estimate(results[-1]).startswith("Estimate: 60%")


def test_engine_reports_missing_mmwave_frame(tmp_path):
    results, statuses, errors = [], [], []
    engine = InferenceMode.InferenceEngine(_immediate_post_ui, results.append, statuses.append, errors.append)
    stub = StubPredictor()
    with engine._lock:
        engine.predictor = stub
        engine.choice = None
    InferenceMode.MMWAVE_FRAME_TIMEOUT_SECONDS = 0.2
    engine.start_session(tmp_path / "S")
    engine.enqueue(InferenceMode.ChirpJob(
        token=1, position=1, index=1, global_index=1,
        thermal_path=str(FIXTURES / "Thermal_Image_100_Percent_Position_1_001.png"),
        thermal_fallback_path="", voice_path=str(FIXTURES / "CoBas_V1_PulseVoice_20261021_184200_357070_Position_1_001.wav"),
        mmwave_path=str(tmp_path / "frame_000001.jpg"),
    ))
    engine.stop()
    engine._worker.join(timeout=10)
    assert results == []
    assert errors and "mmWave" in errors[0]
