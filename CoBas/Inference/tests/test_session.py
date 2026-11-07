from pathlib import Path

import pytest

from cobas_inference import session


def _make_session(root: Path, positions: int, per_position: int, extra_mmwave: int = 0) -> Path:
    s = root / "60_Percent_Battery" / "Session_20261021_185500_k1glepg0"
    (s / "mmWave Frames").mkdir(parents=True)
    (s / "Thermal Frames").mkdir()
    (s / "Voices").mkdir()
    total = positions * per_position
    for k in range(1, total + 1 + extra_mmwave):
        (s / "mmWave Frames" / f"frame_{k:06d}.jpg").write_bytes(b"")
    for p in range(1, positions + 1):
        for k in range(1, per_position + 1):
            (s / "Thermal Frames" / f"Thermal_Image_60_Percent_Position_{p}_{k:03d}.png").write_bytes(b"")
            (s / "Voices" / f"CoBas_V1_PulseVoice_20261021_185500_000000_Position_{p}_{k:03d}.wav").write_bytes(b"")
    return s


def test_discover_pairs_by_position_and_global_frame(tmp_path):
    s = _make_session(tmp_path, positions=4, per_position=3, extra_mmwave=1)
    triplets = session.discover_session(s)
    assert len(triplets) == 12
    first, last = triplets[0], triplets[-1]
    assert (first.position, first.index, first.global_index) == (1, 1, 1)
    assert first.mmwave.name == "frame_000001.jpg"
    assert (last.position, last.index, last.global_index) == (4, 3, 12)
    assert last.mmwave.name == "frame_000012.jpg"
    assert last.thermal.name.endswith("Position_4_003.png")
    assert last.voice.name.endswith("Position_4_003.wav")
    assert set(first.paths()) == {"mmwave", "thermal", "voice"}


def test_label_from_session_path(tmp_path):
    s = _make_session(tmp_path, positions=1, per_position=1)
    assert session.label_from_path(s) == "60%"
    assert session.label_from_path(tmp_path) is None


def test_find_sessions_under_dataset_root(tmp_path):
    s = _make_session(tmp_path, positions=1, per_position=2)
    assert session.find_sessions(tmp_path) == [s]
    assert session.find_sessions(s) == [s]


def test_missing_modality_is_reported_as_none(tmp_path):
    s = _make_session(tmp_path, positions=1, per_position=2)
    (s / "mmWave Frames" / "frame_000002.jpg").unlink()
    triplets = session.discover_session(s)
    assert triplets[1].mmwave is None
    assert "mmwave" not in triplets[1].paths()


def test_empty_session_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        session.discover_session(tmp_path)
