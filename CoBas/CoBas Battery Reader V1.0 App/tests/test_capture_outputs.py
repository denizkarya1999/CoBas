"""Artifact-level tests for the integrated CoBas capture outputs."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
import unittest
import wave
from pathlib import Path

from CoBas_V1 import CoBasV1App
from Camera.thermal_camera import ColoredThermalRenderer, ThermalCamera
from iq_logic import IQSample
from Logic.range_angle_processor import RangeAngleProcessor
from Logic.raw_iq_source import IQFrame
from Logic.reference_frame_generator import generate_random_reference
from Logic.session_logger import RangeAngleSessionLogger
from MMWave.capture import (
    PREVIEW_HEIGHT,
    PREVIEW_WIDTH,
    MMWaveCaptureService,
    _CleanFrameWriter,
)


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
            preview = writer.preview_image(processed)
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
            self.assertEqual(preview.size, (PREVIEW_WIDTH, PREVIEW_HEIGHT))
            self.assertEqual(preview.mode, "RGB")

    def test_mmwave_ready_requires_a_valid_iq_frame(self):
        class DelayedFrameSource:
            total_packets = 0

            def __init__(self):
                self.read_count = 0

            def read_frames(self):
                self.read_count += 1
                return [] if self.read_count < 3 else [object()]

        with tempfile.TemporaryDirectory() as temporary:
            capture = MMWaveCaptureService(20, Path(temporary) / "mmWave Data")
            source = DelayedFrameSource()
            self.assertTrue(capture._wait_until_streaming(source, 0.25))
            self.assertEqual(source.read_count, 3)

    def test_mmwave_ready_reports_an_incompatible_packet_stream(self):
        class IncompatibleSource:
            total_packets = 4

            @staticmethod
            def read_frames():
                return []

        with tempfile.TemporaryDirectory() as temporary:
            capture = MMWaveCaptureService(20, Path(temporary) / "mmWave Data")
            with self.assertRaisesRegex(RuntimeError, "no compatible complex"):
                capture._wait_until_streaming(IncompatibleSource(), 0.0)

    def test_thermal_mock_produces_and_clears_live_frames(self):
        with tempfile.TemporaryDirectory() as temporary:
            camera = ThermalCamera(output_dir=temporary, mock=True)
            try:
                self.assertTrue(camera.start_camera())
                deadline = time.monotonic() + 2.0
                while time.monotonic() < deadline and not camera.has_frame():
                    camera.poll_events()
                    time.sleep(0.02)

                self.assertTrue(camera.has_frame())
                preview = camera.get_preview_image(480, 320)
                self.assertIsNotNone(preview)
                self.assertEqual(preview.size, (480, 320))
            finally:
                camera.stop_camera()

            self.assertFalse(camera.has_frame())
            self.assertFalse(camera.is_tracking)

    def test_thermal_color_scale_uses_absolute_celsius(self):
        renderer = ColoredThermalRenderer(15, 30)
        frame = [15 * 50] * (32 * 24)
        frame[-1] = 30 * 50

        rgb = renderer.render_rgb(frame, 32, 24)
        cold_pixel = rgb[:3]
        hot_pixel = rgb[-3:]

        self.assertNotEqual(cold_pixel, hot_pixel)
        self.assertEqual(renderer.last_frame_min_celsius, 15.0)

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
