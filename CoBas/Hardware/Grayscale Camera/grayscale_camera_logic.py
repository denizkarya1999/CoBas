"""Greyscale rendering for the MLX90642 camera."""

import sys
from pathlib import Path


# Find the shared camera code.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Add the parent folder once.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import the sensor size and base renderer.
from thermal_camera_logic import (  # noqa: E402
    SENSOR_HEIGHT,
    SENSOR_WIDTH,
    ThermalRenderer,
)


class GrayscaleThermalRenderer(ThermalRenderer):
    """Map cold to black and hot to white."""

    def _color(self, value, min_value, max_value, color_cache):
        """Convert one heat value to grey RGB."""
        # Use the raw value as a cache key.
        rounded = int(value)
        cached = color_cache.get(rounded)
        if cached is not None:
            # Reuse the saved color.
            return cached

        if max_value <= min_value:
            # Use grey for a flat frame.
            intensity = 127
        else:
            # Scale heat from 0 to 1.
            normalized = (value - min_value) / (max_value - min_value)
            # Keep the value in range.
            normalized = max(0.0, min(1.0, normalized))
            # Convert it to 8-bit grey.
            intensity = int(normalized * 255)

        # Equal RGB values make grey.
        color = (intensity, intensity, intensity)
        color_cache[rounded] = color
        return color

    def _get_nearest_map(self, width, height):
        """Map output pixels to sensor pixels."""
        # Keep this map separate from base maps.
        key = ("nearest", width, height)
        if key in self._maps:
            # Reuse the saved map.
            return self._maps[key]

        # Pick one source column per output column.
        x_map = [
            min(SENSOR_WIDTH - 1, x * SENSOR_WIDTH // width)
            for x in range(width)
        ]
        # Pick one source row per output row.
        y_map = [
            min(SENSOR_HEIGHT - 1, y * SENSOR_HEIGHT // height)
            for y in range(height)
        ]

        # Save the map for later frames.
        self._maps[key] = (x_map, y_map)
        return x_map, y_map

    def render_rgb(self, frame, width, height):
        """Create a large, blocky RGB frame."""
        # Make the output size safe.
        width = max(1, int(width))
        height = max(1, int(height))

        # Find the cold and hot values.
        min_value = min(frame)
        max_value = max(frame)
        # Build the nearest-pixel map.
        x_map, y_map = self._get_nearest_map(width, height)

        # Start a new color cache.
        color_cache = {}

        # Build each sensor row only once.
        source_rows = []
        for source_y in range(SENSOR_HEIGHT):
            source_row = source_y * SENSOR_WIDTH
            output_row = bytearray(width * 3)
            offset = 0

            # Repeat source pixels across the row.
            for source_x in x_map:
                red, green, blue = self._color(
                    frame[source_row + source_x],
                    min_value,
                    max_value,
                    color_cache,
                )
                # Write one RGB pixel.
                output_row[offset] = red
                output_row[offset + 1] = green
                output_row[offset + 2] = blue
                offset += 3

            # Save the finished row.
            source_rows.append(bytes(output_row))

        # Repeat rows without smoothing.
        output = bytearray(width * height * 3)
        row_size = width * 3
        offset = 0
        for source_y in y_map:
            output[offset : offset + row_size] = source_rows[source_y]
            offset += row_size

        # Return an immutable frame.
        return bytes(output)
