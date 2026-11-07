import warnings
from pathlib import Path

import pytest

from cobas_inference import specs


def test_registry_covers_the_seven_checkpoint_names():
    assert set(specs.REGISTRY) == {
        "cobas_mmwave_thermal_voice_fusion_model.pth",
        "cobas_mmwave_thermal_fusion_model.pth",
        "cobas_mmwave_and_voice_fusion_model.pth",
        "cobas_thermal_and_voice_model.pth",
        "cobas_thermal_model.pth",
        "cobas_mmwave_model.pth",
        "cobas_voice_reflection_recognition_model.pth",
    }


def test_standalone_specs_are_single_modality():
    thermal = specs.spec_for_model_file("Standalone/CoBas_thermal_model.pth")
    mmwave = specs.spec_for_model_file("Standalone/CoBas_mmWave_model.pth")
    voice = specs.spec_for_model_file("Standalone/CoBas_voice_reflection_recognition_model.pth")
    assert thermal.modalities == ("thermal",) and thermal.tensor_pipeline == specs.RESIZE256_CROP224
    assert mmwave.modalities == ("mmwave",) and mmwave.tensor_pipeline == specs.NATIVE
    assert mmwave.images["mmwave"].resize == "none"
    assert voice.modalities == ("voice",) and voice.tensor_pipeline == specs.RESIZE224
    assert voice.voice.colormap == "magma" and voice.voice.save_size is None
    # "Standalone" is not a ratio folder, so weights stay the single default.
    resolved = specs.resolve_spec("models/Standalone/CoBas_thermal_model.pth", {"label_to_index": {"0%": 0}})
    assert resolved.weights == (1.0,)


def test_spec_lookup_accepts_onnx_name():
    spec = specs.spec_for_model_file("x/CoBas_mmwave_thermal_fusion_model.onnx")
    assert spec.family == "mmwave_thermal"


def test_unknown_filename_raises():
    with pytest.raises(KeyError):
        specs.spec_for_model_file("something_else.pth")


@pytest.mark.parametrize(
    "name, expected",
    [
        ("1 - 1 Fusions", (1, 1)),
        ("1 - 2 Fusions", (1, 2)),
        ("2 - 1 - 1 Fusions", (2, 1, 1)),
        ("Standalone", None),
        ("CoBas Models", None),
    ],
)
def test_weights_from_folder_name(name, expected):
    assert specs.weights_from_folder_name(name) == expected


def test_three_modal_weights_from_checkpoint_metadata_win_over_folder():
    path = Path("1 - 2 - 1 Fusions") / "CoBas_mmwave_thermal_voice_fusion_model.pth"
    meta = {"fusion_weights": {"mmWave": 0.25, "Thermal": 0.25, "Voice": 0.5}}
    spec = specs.resolve_spec(path, meta)
    assert spec.weights == pytest.approx((0.25, 0.25, 0.5))
    assert spec.weights_source == "checkpoint metadata"


def test_thermal_voice_falls_back_to_folder_ratio():
    path = Path("2 - 1 Fusions") / "CoBas_thermal_and_voice_model.pth"
    spec = specs.resolve_spec(path, {"label_to_index": {"0%": 0}})
    assert spec.modalities == ("thermal", "voice")
    assert spec.weights == pytest.approx((2 / 3, 1 / 3))
    assert "folder name" in spec.weights_source


def test_command_line_override_wins():
    path = Path("1 - 1 Fusions") / "CoBas_mmwave_and_voice_fusion_model.pth"
    spec = specs.resolve_spec(path, {"fusion_weights": {"mmWave": 0.5, "Voice": 0.5}}, weights_override=[3, 1])
    assert spec.weights == pytest.approx((0.75, 0.25))


def test_metadata_cross_check_warns_on_normalization_mismatch():
    path = Path("1 - 1 Fusions") / "CoBas_mmwave_thermal_fusion_model.pth"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        specs.resolve_spec(path, {"model_input_mean": [0.485, 0.456, 0.406], "model_input_std": [0.229, 0.224, 0.225]})
    assert any("tensor normalization" in str(w.message) for w in caught)


def test_metadata_matching_registry_is_silent():
    path = Path("1 - 1 Fusions") / "CoBas_mmwave_thermal_fusion_model.pth"
    meta = {
        "fusion_weights": {"mmWave": 0.5, "Thermal": 0.5},
        "model_input_mean": [0.0, 0.0, 0.0],
        "model_input_std": [1.0, 1.0, 1.0],
        "fusion_normalization": "mmWave and Thermal each divided by 255; no min-max normalization",
    }
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        spec = specs.resolve_spec(path, meta)
    assert spec.fusion_norm == specs.DIV255 and spec.tensor_norm == specs.UNIT
