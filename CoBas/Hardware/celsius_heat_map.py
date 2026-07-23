"""Map the default 15–30 °C display range to RGB heat-map colors."""

import math
from types import MappingProxyType


def _index_colors_by_temperature(colors, min_celsius, max_celsius):
    """Return an immutable temperature lookup for evenly spaced colors."""
    step = (max_celsius - min_celsius) / (len(colors) - 1)
    return MappingProxyType(
        {
            min_celsius + index * step: color
            for index, color in enumerate(colors)
        }
    )


class CelsiusHeatMap:
    """Map a configurable Celsius range across the complete RGB palette."""

    # These endpoints control display sensitivity only. Sensor conversion still
    # reports physical Celsius values; values outside this range are clamped by
    # the renderer to the nearest endpoint color.
    MIN_CELSIUS = 15
    MAX_CELSIUS = 30

    # Keep all 61 colors from the former 0–60 °C scale. They now act as evenly
    # spaced palette stops across 15–30 °C, so the visual color range remains
    # unchanged while a 0.25 °C change advances by one original color step.
    _COLORS_BY_PALETTE_INDEX = MappingProxyType(
        {
            # Palette stops 0–7: very dark navy blue to dark blue.
            0: (0, 0, 48),
            1: (0, 0, 59),
            2: (0, 0, 70),
            3: (0, 0, 81),
            4: (0, 0, 92),
            5: (0, 0, 103),
            6: (0, 0, 114),
            7: (0, 0, 125),
            # Palette stops 8–14: deep blue to bright blue.
            8: (0, 5, 135),
            9: (0, 15, 146),
            10: (0, 25, 157),
            11: (0, 35, 167),
            12: (0, 44, 178),
            13: (0, 54, 189),
            14: (0, 64, 199),
            # Palette stops 15–22: bright blue to cyan-blue.
            15: (0, 74, 210),
            16: (0, 89, 216),
            17: (0, 105, 222),
            18: (0, 120, 228),
            19: (0, 136, 234),
            20: (0, 151, 240),
            21: (0, 167, 246),
            22: (0, 182, 252),
            # Palette stops 23–29: cyan-blue to green-turquoise.
            23: (0, 194, 248),
            24: (0, 201, 234),
            25: (0, 208, 220),
            26: (0, 216, 206),
            27: (0, 223, 192),
            28: (0, 230, 178),
            29: (0, 238, 164),
            # Palette stops 30–37: green-turquoise to bright yellow.
            30: (0, 245, 150),
            31: (34, 245, 138),
            32: (67, 245, 126),
            33: (101, 245, 114),
            34: (134, 245, 102),
            35: (168, 245, 90),
            36: (202, 245, 78),
            37: (235, 245, 66),
            # Palette stops 38–44: yellow to yellow-orange.
            38: (252, 239, 58),
            39: (253, 227, 55),
            40: (253, 215, 52),
            41: (253, 203, 49),
            42: (254, 192, 46),
            43: (254, 180, 42),
            44: (255, 168, 39),
            # Palette stops 45–52: orange to hot red.
            45: (255, 156, 36),
            46: (252, 142, 35),
            47: (250, 127, 34),
            48: (247, 113, 33),
            49: (244, 98, 32),
            50: (242, 84, 31),
            51: (239, 70, 30),
            52: (236, 55, 29),
            # Palette stops 53–59: hot red through pale red toward white.
            53: (236, 62, 43),
            54: (239, 89, 73),
            55: (242, 117, 104),
            56: (244, 145, 134),
            57: (247, 172, 164),
            58: (250, 200, 194),
            59: (252, 227, 225),
            # Palette stop 60: white, representing the hottest value.
            60: (255, 255, 255),
        }
    )
    COLOR_STOPS = tuple(_COLORS_BY_PALETTE_INDEX.values())
    CELSIUS_PER_COLOR_STEP = (MAX_CELSIUS - MIN_CELSIUS) / (
        len(COLOR_STOPS) - 1
    )
    COLORS_BY_CELSIUS = _index_colors_by_temperature(
        COLOR_STOPS,
        MIN_CELSIUS,
        MAX_CELSIUS,
    )

    def __init__(self, min_celsius=None, max_celsius=None):
        """Create a heat map, using 15–30 °C when no range is supplied."""
        if min_celsius is None:
            min_celsius = self.MIN_CELSIUS
        if max_celsius is None:
            max_celsius = self.MAX_CELSIUS

        try:
            min_celsius = float(min_celsius)
            max_celsius = float(max_celsius)
        except (TypeError, ValueError) as exc:
            raise TypeError("temperature range values must be real numbers") from exc

        if not math.isfinite(min_celsius) or not math.isfinite(max_celsius):
            raise ValueError("temperature range values must be finite")
        if max_celsius <= min_celsius:
            raise ValueError("maximum temperature must be greater than minimum")

        # Store the selected endpoints on this instance. Class-level constants
        # remain available as defaults for existing callers and future dialogs.
        self.MIN_CELSIUS = min_celsius
        self.MAX_CELSIUS = max_celsius
        self.CELSIUS_PER_COLOR_STEP = (
            max_celsius - min_celsius
        ) / (len(self.COLOR_STOPS) - 1)
        self.COLORS_BY_CELSIUS = _index_colors_by_temperature(
            self.COLOR_STOPS,
            min_celsius,
            max_celsius,
        )

        # The public lookup maps each evenly spaced temperature stop to an RGB
        # tuple and remains immutable so callers cannot change shared colors.
        self.mapping = self.COLORS_BY_CELSIUS

    def rgb_for_celsius(self, celsius):
        """Return an ``(red, green, blue)`` tuple for a temperature in Celsius."""
        try:
            temperature = float(celsius)
        except (TypeError, ValueError) as exc:
            raise TypeError("temperature must be a real number") from exc

        if not math.isfinite(temperature):
            raise ValueError("temperature must be finite")
        if not self.MIN_CELSIUS <= temperature <= self.MAX_CELSIUS:
            raise ValueError(
                f"temperature must be between {self.MIN_CELSIUS:g} °C "
                f"and {self.MAX_CELSIUS:g} °C"
            )

        # Convert the requested temperature into a fractional palette index.
        # The selected minimum maps to stop 0 and the selected maximum maps to
        # stop 60 without changing any of the original RGB colors.
        palette_position = (
            (temperature - self.MIN_CELSIUS)
            / (self.MAX_CELSIUS - self.MIN_CELSIUS)
            * (len(self.COLOR_STOPS) - 1)
        )
        lower_index = math.floor(palette_position)
        upper_index = math.ceil(palette_position)
        lower_rgb = self.COLOR_STOPS[lower_index]
        if lower_index == upper_index:
            return lower_rgb

        # Temperatures between two stops blend their neighboring colors to keep
        # the output smooth rather than jumping abruptly between colors.
        upper_rgb = self.COLOR_STOPS[upper_index]
        weight = palette_position - lower_index
        return tuple(
            round(lower + (upper - lower) * weight)
            for lower, upper in zip(lower_rgb, upper_rgb)
        )

    def __getitem__(self, celsius):
        """Allow ``heat_map[temperature]`` lookup syntax."""
        return self.rgb_for_celsius(celsius)
