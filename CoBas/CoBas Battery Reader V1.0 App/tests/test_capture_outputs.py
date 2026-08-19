"""Artifact-level tests for the integrated CoBas capture outputs."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
import wave
from pathlib import Path

from CoBas_V1 import CoBasV1App
from iq_logic import IQSample
from Logic.range_angle_processor import RangeAngleProcessor
from Logic.raw_iq_source import IQFrame
from Logic.reference_frame_generator import generate_random_reference
from Logic.session_logger import RangeAngleSessionLogger
from MMWave.capture import _CleanFrameWriter


class CaptureOutputTests(unittest.TestCase):
    def test_mmwave_outputs_use_battery_mmwave_directory(self):
        samples = tuple(
            IQSample(
                frame_number=7,
                range_bin=range_bin,
                virtual_antenna=antenna,
                i=(range_bin + 1) * (antenna + 2),
                q=antenna - range_bin,
            )
            for range_bin in range(64)
            for antenna in range(4)
        )
        raw = IQFrame(
            frame_number=7,
            virtual_antenna_count=4,
            samples=samples,
        )
        processed = RangeAngleProcessor().process(raw)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "20_Percent_Battery" / "mmWave Data"
            with RangeAngleSessionLogger(
                20,
                session_name="20_Percent_Battery",
                log_directory=root / "Logs",
            ) as logger:
                logger.write_frame(raw, processed)

            writer = _CleanFrameWriter(root / "Frames")
            writer.validate_preconditions()
            self.assertTrue(writer.write_if_due(processed))
            writer.finish()
            generate_random_reference(
                20,
                frames_directory=root / "Frames",
                references_directory=root / "References",
            )

            expected = {
                "Logs/Raw IQ Signals/20_Percent_Battery.csv",
                "Logs/Range-Angle Responses/20_Percent_Battery.csv",
                "Frames/frame_000001.jpg",
                "References/20_Percent_Battery_Reference.jpg",
            }
            actual = {
                str(path.relative_to(root))
                for path in root.rglob("*")
                if path.is_file()
            }
            self.assertLessEqual(expected, actual)
            self.assertEqual(processed.power_db.shape, (5, 121))

    def test_voice_pulses_become_one_battery_recording(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            recordings = []
            for pulse_number in (1, 2, 3):
                path = root / f"pulse_{pulse_number}.wav"
                with wave.open(str(path), "wb") as stream:
                    stream.setnchannels(1)
                    stream.setsampwidth(2)
                    stream.setframerate(8_000)
                    stream.writeframes(b"\0\0" * 800)
                recordings.append({"path": str(path), "pulse_number": pulse_number})

            app = object.__new__(CoBasV1App)
            app.pulse_recordings = recordings
            output = root / "Voice_Recording.wav"
            app.build_battery_voice_recording(str(output))

            with wave.open(str(output), "rb") as stream:
                self.assertEqual(stream.getnframes(), 2_400)
            self.assertTrue(all(not Path(item["path"]).exists() for item in recordings))

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is required")
    def test_position_segments_become_one_thermal_video(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = []
            for filename, color in (("one.mp4", "red"), ("two.mp4", "blue")):
                path = root / filename
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-f",
                        "lavfi",
                        "-i",
                        f"color=c={color}:s=64x48:r=8:d=0.5",
                        "-an",
                        "-c:v",
                        "libx264",
                        "-pix_fmt",
                        "yuv420p",
                        str(path),
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                inputs.append(path)

            segments = [
                {"position": 1, "thermal_video_path": str(inputs[0])},
                {"position": 2, "thermal_video_path": str(inputs[1])},
            ]
            output = root / "Thermal_Video.mp4"
            CoBasV1App.concatenate_thermal_videos(segments, str(output))

            self.assertTrue(output.exists())
            self.assertTrue(all(not path.exists() for path in inputs))


if __name__ == "__main__":
    unittest.main()
