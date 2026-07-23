"""Regression tests for fixed Celsius-to-color associations."""

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))

from celsius_heat_map import CelsiusHeatMap
from thermal_camera_logic import ThermalRenderer


class ThermalColorMappingTests(unittest.TestCase):
    def setUp(self):
        self.heat_map = CelsiusHeatMap()
        self.renderer = ThermalRenderer(15, 30)

    def test_selected_range_crops_fixed_zero_to_sixty_scale(self):
        self.assertEqual(
            self.renderer.scale_color(0.0),
            self.heat_map.rgb_for_celsius(15),
        )
        self.assertEqual(
            self.renderer.scale_color(0.5),
            self.heat_map.rgb_for_celsius(22.5),
        )
        self.assertEqual(
            self.renderer.scale_color(1.0),
            self.heat_map.rgb_for_celsius(30),
        )

    def test_every_point_zero_point_zero_five_interval_has_unique_color(self):
        expected_color_count = round(
            (
                self.heat_map.MAX_CELSIUS
                - self.heat_map.MIN_CELSIUS
            )
            / self.heat_map.COLOR_RESOLUTION_CELSIUS
        ) + 1
        colors = tuple(self.heat_map.COLORS_BY_CELSIUS.values())

        self.assertEqual(expected_color_count, 1201)
        self.assertEqual(len(colors), expected_color_count)
        self.assertEqual(len(set(colors)), expected_color_count)
        self.assertLessEqual(
            self.heat_map.COLOR_RESOLUTION_CELSIUS,
            self.heat_map.SENSOR_NETD_CELSIUS,
        )

    def test_fractional_temperatures_receive_progressive_unique_shades(self):
        temperatures = (15.0, 15.05, 15.1, 15.5, 16.0)
        colors = [
            self.heat_map.rgb_for_celsius(temperature)
            for temperature in temperatures
        ]

        self.assertEqual(len(colors), len(set(colors)))
        self.assertEqual(colors[0], (0, 74, 210))
        self.assertEqual(colors[-1], (0, 89, 216))

    def test_sensor_significant_raw_changes_cannot_share_a_color(self):
        # Raw camera values advance by 0.02 °C. Four raw counts equal 0.08 °C,
        # the first representable change above the specified 0.065 K NETD.
        raw_step_colors = [
            self.heat_map.rgb_for_celsius(raw_value / 50.0)
            for raw_value in range(60 * 50 + 1)
        ]

        self.assertTrue(
            all(
                first_color != second_color
                for first_color, second_color in zip(
                    raw_step_colors,
                    raw_step_colors[4:],
                )
            )
        )

    def test_whole_degree_anchor_colors_are_preserved(self):
        for temperature, expected_color in (
            self.heat_map.ANCHOR_COLORS_BY_CELSIUS.items()
        ):
            with self.subTest(temperature=temperature):
                self.assertEqual(
                    self.heat_map.rgb_for_celsius(temperature),
                    expected_color,
                )

    def test_pixels_outside_selected_range_use_fixed_endpoint_colors(self):
        colors = self.renderer.frame_colors(
            [
                10 * 50,
                15 * 50,
                30 * 50,
                35 * 50,
            ]
        )

        self.assertEqual(colors[0], colors[1])
        self.assertEqual(colors[2], colors[3])
        self.assertEqual(colors[1], "#004ad2")
        self.assertEqual(colors[2], "#00f596")

    def test_selected_range_must_stay_inside_color_scale(self):
        for minimum, maximum in ((-1, 30), (15, 61)):
            with self.subTest(minimum=minimum, maximum=maximum):
                with self.assertRaisesRegex(ValueError, "0–60"):
                    ThermalRenderer(minimum, maximum)


if __name__ == "__main__":
    unittest.main()
