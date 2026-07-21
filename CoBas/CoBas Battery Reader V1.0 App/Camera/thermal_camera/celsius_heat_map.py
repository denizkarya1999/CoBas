"""Map temperatures from 0 °C through 60 °C to RGB heat-map colors."""

import math
from types import MappingProxyType


class CelsiusHeatMap:
    """Provide RGB colors for temperatures in the inclusive 0–60 °C range."""

    MIN_CELSIUS = 0
    MAX_CELSIUS = 60

    # Every whole Celsius degree has its own RGB color. The color families
    # progress from cold navy blue through cyan, green, yellow, orange, red,
    # pale red, and finally hot white.
    COLORS_BY_CELSIUS = MappingProxyType(
        {
            # 0–7 °C: very dark navy blue to dark blue.
            0: (0, 0, 48),
            1: (0, 0, 59),
            2: (0, 0, 70),
            3: (0, 0, 81),
            4: (0, 0, 92),
            5: (0, 0, 103),
            6: (0, 0, 114),
            7: (0, 0, 125),
            # 8–14 °C: deep blue to bright blue.
            8: (0, 5, 135),
            9: (0, 15, 146),
            10: (0, 25, 157),
            11: (0, 35, 167),
            12: (0, 44, 178),
            13: (0, 54, 189),
            14: (0, 64, 199),
            # 15–22 °C: bright blue to cyan-blue.
            15: (0, 74, 210),
            16: (0, 89, 216),
            17: (0, 105, 222),
            18: (0, 120, 228),
            19: (0, 136, 234),
            20: (0, 151, 240),
            21: (0, 167, 246),
            22: (0, 182, 252),
            # 23–29 °C: cyan-blue to green-turquoise.
            23: (0, 194, 248),
            24: (0, 201, 234),
            25: (0, 208, 220),
            26: (0, 216, 206),
            27: (0, 223, 192),
            28: (0, 230, 178),
            29: (0, 238, 164),
            # 30–37 °C: green-turquoise to bright yellow.
            30: (0, 245, 150),
            31: (34, 245, 138),
            32: (67, 245, 126),
            33: (101, 245, 114),
            34: (134, 245, 102),
            35: (168, 245, 90),
            36: (202, 245, 78),
            37: (235, 245, 66),
            # 38–44 °C: yellow to yellow-orange.
            38: (252, 239, 58),
            39: (253, 227, 55),
            40: (253, 215, 52),
            41: (253, 203, 49),
            42: (254, 192, 46),
            43: (254, 180, 42),
            44: (255, 168, 39),
            # 45–52 °C: orange to hot red.
            45: (255, 156, 36),
            46: (252, 142, 35),
            47: (250, 127, 34),
            48: (247, 113, 33),
            49: (244, 98, 32),
            50: (242, 84, 31),
            51: (239, 70, 30),
            52: (236, 55, 29),
            # 53–59 °C: hot red through pale red toward white.
            53: (236, 62, 43),
            54: (239, 89, 73),
            55: (242, 117, 104),
            56: (244, 145, 134),
            57: (247, 172, 164),
            58: (250, 200, 194),
            59: (252, 227, 225),
            # 60 °C: white, representing the hottest value.
            60: (255, 255, 255),
        }
    )

    def __init__(self):
        # Keep the public lookup table immutable so shared colors cannot be
        # changed accidentally by a caller.
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

        lower_celsius = math.floor(temperature)
        upper_celsius = math.ceil(temperature)
        lower_rgb = self.COLORS_BY_CELSIUS[lower_celsius]
        if lower_celsius == upper_celsius:
            return lower_rgb

        # Fractional temperatures blend only the two neighboring one-degree
        # colors; exact whole degrees always return their assigned RGB value.
        upper_rgb = self.COLORS_BY_CELSIUS[upper_celsius]
        weight = temperature - lower_celsius
        return tuple(
            round(lower + (upper - lower) * weight)
            for lower, upper in zip(lower_rgb, upper_rgb)
        )

    def __getitem__(self, celsius):
        """Allow ``heat_map[temperature]`` lookup syntax."""
        return self.rgb_for_celsius(celsius)
