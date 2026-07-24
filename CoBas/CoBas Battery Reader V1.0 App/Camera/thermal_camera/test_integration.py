"""Integration coverage for the thermal camera bundled with CoBas V1."""

import importlib
import io
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock


APP_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = APP_ROOT.parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from Camera.Camera import Camera
from Camera.thermal_camera import CelsiusHeatMap, ThermalCamera
from CoBas_V1 import CoBasV1App


regular_camera_module = importlib.import_module("Camera.Camera")
thermal_camera_module = importlib.import_module("Camera.thermal_camera.gui")


class CoBasThermalCameraIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.output_directory = tempfile.TemporaryDirectory()
        self.camera = ThermalCamera(
            output_dir=self.output_directory.name,
            mock=True,
        )

    def tearDown(self):
        self.camera.stop_camera()
        self.output_directory.cleanup()

    def test_bundled_backend_matches_current_hardware_backend(self):
        hardware_root = REPO_ROOT / "Hardware"
        bundled_root = Path(__file__).resolve().parent

        for filename in ("celsius_heat_map.py", "thermal_camera_logic.py"):
            with self.subTest(filename=filename):
                self.assertEqual(
                    (bundled_root / filename).read_bytes(),
                    (hardware_root / filename).read_bytes(),
                )

    def test_public_camera_uses_high_resolution_colored_renderer(self):
        heat_map = CelsiusHeatMap()

        self.assertIs(self.camera.rgb_renderer.heat_map.__class__, CelsiusHeatMap)
        self.assertEqual(heat_map.COLOR_RESOLUTION_CELSIUS, 0.05)
        self.assertEqual(len(heat_map.mapping), 1201)
        self.assertEqual(len(set(heat_map.mapping.values())), 1201)

    def test_range_change_and_preview_use_fixed_celsius_colors(self):
        self.assertTrue(self.camera.set_temperature_range(15, 30))
        frame = [
            750 + index % 51
            for index in range(32 * 24)
        ]
        with self.camera.frame_lock:
            self.camera.latest_frame = frame

        preview = self.camera.get_preview_image(640, 480)
        colors = self.camera.rgb_renderer.frame_colors([750, 753, 755])

        self.assertEqual(preview.size, (640, 480))
        self.assertEqual(self.camera.get_temperature_range(), (15.0, 30.0))
        self.assertEqual(len(colors), len(set(colors)))

    def test_colored_and_grayscale_modes_share_selected_range(self):
        self.camera.set_temperature_range(12.5, 28.75)

        self.assertTrue(self.camera.set_display_mode("grayscale"))
        self.assertEqual(
            self.camera.renderer.legend_celsius_range(),
            (12.5, 28.75),
        )
        self.assertTrue(self.camera.set_display_mode("rgb"))
        self.assertEqual(
            self.camera.renderer.legend_celsius_range(),
            (12.5, 28.75),
        )

    def test_mock_worker_reaches_the_cobas_preview(self):
        self.camera.start_camera()
        deadline = time.monotonic() + 3.0

        while time.monotonic() < deadline and not self.camera.has_frame():
            self.camera.poll_events()
            time.sleep(0.02)

        preview = self.camera.get_preview_image(640, 480)
        self.assertTrue(self.camera.has_frame())
        self.assertEqual(self.camera.status, "Thermal live")
        self.assertEqual(preview.size, (640, 480))

    def test_regular_capture_logs_average_fps(self):
        camera = Camera(output_dir=self.output_directory.name)
        camera.is_recording = True
        camera.video_writer = mock.Mock()
        camera.record_start_monotonic = 10.0
        camera.record_frames_written = 40

        output = io.StringIO()
        with mock.patch.object(
            regular_camera_module.time,
            "monotonic",
            return_value=12.0,
        ), redirect_stdout(output):
            camera.stop_recording_capture_phase()

        self.assertIn(
            "Regular camera average FPS: 20.00 "
            "(40 frames over 2.00 seconds)",
            output.getvalue(),
        )
        self.assertEqual(camera.last_recording_average_fps, 20.0)
        self.assertEqual(camera.last_recording_duration, 2.0)
        self.assertEqual(camera.last_recording_frame_count, 40)

    def test_thermal_capture_logs_average_fps(self):
        recorder = mock.Mock(frames_written=16, frames_dropped=0)
        scale_recorder = mock.Mock(frames_dropped=0)
        self.camera.is_recording = True
        self.camera.recorder = recorder
        self.camera.scale_recorder = scale_recorder
        self.camera.record_start_monotonic = 10.0

        output = io.StringIO()
        with mock.patch.object(
            thermal_camera_module.time,
            "monotonic",
            return_value=12.0,
        ), redirect_stdout(output):
            self.camera.stop_recording()

        self.assertIn(
            "Thermal camera average FPS: 8.00 "
            "(16 frames over 2.00 seconds)",
            output.getvalue(),
        )
        self.assertEqual(self.camera.last_recording_average_fps, 8.0)
        self.assertEqual(self.camera.last_recording_duration, 2.0)
        self.assertEqual(self.camera.last_recording_frame_count, 16)

    def test_session_writes_both_average_fps_values_to_text_file(self):
        app = CoBasV1App.__new__(CoBasV1App)
        app.captures_dir = self.output_directory.name
        app.current_recording_timestamp = "20260724_120000_000000"
        app.average_fps_log_path = None
        app.camera = mock.Mock(
            last_recording_average_fps=20.0,
            last_recording_duration=2.0,
            last_recording_frame_count=40,
        )
        app.thermal_camera = mock.Mock(
            last_recording_average_fps=8.0,
            last_recording_duration=2.0,
            last_recording_frame_count=16,
        )

        log_path = app.write_average_fps_log(True, True)
        contents = Path(log_path).read_text(encoding="utf-8")

        self.assertIn("Regular camera average FPS: 20.00", contents)
        self.assertIn("Regular camera frames: 40", contents)
        self.assertIn("Thermal camera average FPS: 8.00", contents)
        self.assertIn("Thermal camera frames: 16", contents)


if __name__ == "__main__":
    unittest.main()
