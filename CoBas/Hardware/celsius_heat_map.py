"""Map temperatures from 0 °C through 60 °C to fixed RGB colors."""

import math
from types import MappingProxyType


def _nearest_available_rgb(ideal_rgb, used_colors, reserved_colors):
    """Return the closest unused 8-bit RGB value to a fractional RGB target."""
    rounded_rgb = tuple(round(channel) for channel in ideal_rgb)

    # Most interpolated colors are already unique. When 8-bit rounding causes
    # a collision, search the immediately surrounding RGB values. Reserving
    # the whole-degree anchors ensures those documented colors stay unchanged.
    for radius in range(256):
        channel_ranges = [
            range(
                max(0, channel - radius),
                min(255, channel + radius) + 1,
            )
            for channel in rounded_rgb
        ]
        candidates = []
        for red in channel_ranges[0]:
            for green in channel_ranges[1]:
                for blue in channel_ranges[2]:
                    candidate = (red, green, blue)
                    if candidate in used_colors or candidate in reserved_colors:
                        continue
                    distance = sum(
                        (actual - ideal) ** 2
                        for actual, ideal in zip(candidate, ideal_rgb)
                    )
                    candidates.append((distance, candidate))

        if candidates:
            return min(candidates)[1]

    raise RuntimeError("unable to allocate a unique thermal RGB color")


def _build_high_resolution_scale(
    anchors,
    min_celsius,
    max_celsius,
    resolution_celsius,
):
    """Build an immutable scale with one unique RGB value per interval."""
    steps_per_celsius = round(1.0 / resolution_celsius)
    total_steps = round(
        (max_celsius - min_celsius) / resolution_celsius
    )
    reserved_colors = frozenset(anchors.values())
    used_colors = set()
    scale = {}

    for scale_index in range(total_steps + 1):
        degree_offset, substep = divmod(scale_index, steps_per_celsius)
        lower_celsius = min_celsius + degree_offset

        if substep == 0:
            color = anchors[lower_celsius]
        else:
            upper_celsius = lower_celsius + 1
            weight = substep / steps_per_celsius
            lower_rgb = anchors[lower_celsius]
            upper_rgb = anchors[upper_celsius]
            ideal_rgb = tuple(
                lower + (upper - lower) * weight
                for lower, upper in zip(lower_rgb, upper_rgb)
            )
            color = _nearest_available_rgb(
                ideal_rgb,
                used_colors,
                reserved_colors,
            )

        if color in used_colors:
            raise RuntimeError(
                "thermal color scale contains a duplicate RGB value"
            )

        temperature = round(
            min_celsius + scale_index * resolution_celsius,
            10,
        )
        scale[temperature] = color
        used_colors.add(color)

    return MappingProxyType(scale)


class CelsiusHeatMap:
    """Provide unique fixed RGB colors at 0.05 °C intervals from 0–60 °C."""

    MIN_CELSIUS = 0
    MAX_CELSIUS = 60
    DEFAULT_DISPLAY_MIN_CELSIUS = 15
    DEFAULT_DISPLAY_MAX_CELSIUS = 30
    # The camera's specified NETD is 0.065 K RMS at 2 Hz. A 0.05 °C color
    # interval is slightly finer, so sensor-significant changes cannot collapse
    # into the same color interval.
    SENSOR_NETD_CELSIUS = 0.065
    COLOR_RESOLUTION_CELSIUS = 0.05

    # Every whole Celsius degree owns one RGB color. Selecting a narrower
    # display range must crop this scale, not stretch all colors over the
    # selected endpoints.
    ANCHOR_COLORS_BY_CELSIUS = MappingProxyType(
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
    COLORS_BY_CELSIUS = _build_high_resolution_scale(
        ANCHOR_COLORS_BY_CELSIUS,
        MIN_CELSIUS,
        MAX_CELSIUS,
        COLOR_RESOLUTION_CELSIUS,
    )
    COLOR_STOPS = tuple(COLORS_BY_CELSIUS.values())
    CELSIUS_PER_COLOR_STEP = COLOR_RESOLUTION_CELSIUS

    def __init__(self):
        # Keep the public lookup immutable so callers cannot change shared
        # Celsius-to-color associations.
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

        # Quantize to the nearest 0.05 °C color interval. This is finer than the
        # camera's specified 0.065 K NETD while remaining deterministic and
        # avoiding duplicate RGB colors caused by ordinary 8-bit interpolation.
        color_position = (
            (temperature - self.MIN_CELSIUS)
            / self.COLOR_RESOLUTION_CELSIUS
        )
        color_index = max(
            0,
            min(
                len(self.COLOR_STOPS) - 1,
                math.floor(color_position + 0.5),
            ),
        )
        return self.COLOR_STOPS[color_index]

    def __getitem__(self, celsius):
        """Allow ``heat_map[temperature]`` lookup syntax."""
        return self.rgb_for_celsius(celsius)
