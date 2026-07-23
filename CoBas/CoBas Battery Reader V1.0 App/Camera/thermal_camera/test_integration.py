"""Integration coverage for the thermal camera bundled with CoBas V1."""

import sys
import tempfile
import time
import unittest
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = APP_ROOT.parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from Camera.thermal_camera import CelsiusHeatMap, ThermalCamera


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


if __name__ == "__main__":
    unittest.main()
