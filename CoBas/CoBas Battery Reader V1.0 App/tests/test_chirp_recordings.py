"""Regression tests for one-file-per-chirp voice capture outputs."""

from __future__ import annotations

import tempfile
import unittest
import wave
from pathlib import Path

from CoBas_V1 import CoBasV1App


def write_test_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(8_000)
        stream.writeframes(b"\0\0" * 16)


class ChirpRecordingTests(unittest.TestCase):
    def test_two_hundred_chirps_remain_two_hundred_wav_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_directory = Path(temporary)
            recordings = []
            for pulse_number in range(1, 201):
                path = output_directory / f"chirp_{pulse_number:03d}.wav"
                write_test_wav(path)
                recordings.append(
                    {
                        "path": str(path),
                        "pulse_number": pulse_number,
                    }
                )

            app = object.__new__(CoBasV1App)
            app.pulse_recordings = recordings
            recording_paths = app.validate_chirp_voice_recordings()

            self.assertEqual(len(recording_paths), 200)
            self.assertEqual(len(list(output_directory.glob("*.wav"))), 200)
            self.assertTrue(all(Path(path).is_file() for path in recording_paths))
            self.assertFalse((output_directory / "Voice_Recording.wav").exists())

    def test_missing_chirp_recording_fails_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "chirp_001.wav"
            app = object.__new__(CoBasV1App)
            app.pulse_recordings = [
                {"path": str(path), "pulse_number": 1},
            ]

            with self.assertRaisesRegex(RuntimeError, "missing for chirp 1"):
                app.validate_chirp_voice_recordings()


if __name__ == "__main__":
    unittest.main()
