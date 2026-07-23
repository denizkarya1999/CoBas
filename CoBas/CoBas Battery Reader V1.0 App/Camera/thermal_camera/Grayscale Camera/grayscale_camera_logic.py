"""Grayscale rendering for the MLX90642 thermal camera."""

import math
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Import the shared thermal-camera implementation.
# ---------------------------------------------------------------------------

# Locate the project folder containing thermal_camera_logic.py.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Add the project folder to Python's module search path only once.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import the sensor dimensions and the common renderer base class.
from thermal_camera_logic import (  # noqa: E402
    SENSOR_HEIGHT,
    SENSOR_WIDTH,
    ThermalRenderer,
    raw_to_celsius,
)


class GrayscaleThermalRenderer(ThermalRenderer):
    """
    Convert MLX90642 temperature measurements into a grayscale RGB image.

    Lower temperatures are displayed as darker pixels, while higher
    temperatures are displayed as brighter pixels.

    Two temperature-scaling modes are supported:

    1. Automatic scaling:
       The coldest valid value in each frame becomes black, and the hottest
       valid value becomes white.

    2. Fixed scaling:
       A fixed temperature range can be provided when creating the renderer.
       This makes the same temperature appear with the same brightness across
       multiple frames.

    Examples
    --------
    Automatic per-frame scaling:

        renderer = GrayscaleThermalRenderer()

    Fixed 20-40 degree scaling:

        renderer = GrayscaleThermalRenderer(
            display_min=20.0,
            display_max=40.0,
        )
    """

    # Middle gray is used for invalid pixels and completely flat frames.
    INVALID_INTENSITY = 127

    def __init__(
        self,
        *args,
        display_min=None,
        display_max=None,
        **kwargs,
    ):
        """
        Initialize the grayscale thermal renderer.

        Parameters
        ----------
        display_min : float or None
            Fixed temperature mapped to black. When None, the minimum valid
            value from each frame is used.

        display_max : float or None
            Fixed temperature mapped to white. When None, the maximum valid
            value from each frame is used.

        Notes
        -----
        display_min and display_max must either both be provided or both be
        left as None.
        """
        # Initialize the shared ThermalRenderer implementation.
        super().__init__(*args, **kwargs)

        # A fixed range requires both endpoints.
        if (display_min is None) != (display_max is None):
            raise ValueError(
                "display_min and display_max must either both be provided "
                "or both be None."
            )

        if display_min is not None:
            try:
                display_min = float(display_min)
                display_max = float(display_max)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "display_min and display_max must be numeric values."
                ) from exc

            # Reject NaN and infinite configuration values.
            if not math.isfinite(display_min) or not math.isfinite(display_max):
                raise ValueError(
                    "display_min and display_max must be finite values."
                )

            if display_max <= display_min:
                raise ValueError(
                    "display_max must be greater than display_min."
                )

        self.display_min = display_min
        self.display_max = display_max

        # Cache only the most recently requested output-size mapping.
        # This avoids unlimited cache growth while the application window
        # is being resized.
        self._nearest_map_key = None
        self._nearest_map_value = None

    @staticmethod
    def _validate_frame(frame):
        """
        Validate and convert a sensor frame to a flat list of floats.

        The renderer expects one temperature value for each MLX90642 sensor
        pixel, arranged in row-major order:

            frame[y * SENSOR_WIDTH + x]

        Returns
        -------
        list[float]
            The validated thermal values.

        Raises
        ------
        ValueError
            If the frame has the wrong number of values, contains unsupported
            objects, or contains no finite temperature measurements.
        """
        expected_size = SENSOR_WIDTH * SENSOR_HEIGHT

        if frame is None:
            raise ValueError("The thermal frame cannot be None.")

        try:
            # Create a local list so the frame is safe to access repeatedly.
            values = [float(value) for value in frame]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "The thermal frame must be a flat sequence of numeric values."
            ) from exc

        if len(values) != expected_size:
            raise ValueError(
                f"Expected {expected_size} thermal values "
                f"({SENSOR_WIDTH} x {SENSOR_HEIGHT}), "
                f"but received {len(values)}."
            )

        # At least one finite value is required to determine a display range.
        if not any(math.isfinite(value) for value in values):
            raise ValueError(
                "The thermal frame contains no finite temperature values."
            )

        return values

    def _get_display_range(self, frame):
        """
        Determine which temperature range should be mapped to 0-255.

        A configured fixed range is preferred. Otherwise, the minimum and
        maximum finite values from the current frame are used.
        """
        if self.display_min is not None:
            return self.display_min, self.display_max

        # Ignore NaN and infinite values when calculating the frame range.
        finite_values = [
            value
            for value in frame
            if math.isfinite(value)
        ]

        return min(finite_values), max(finite_values)

    def legend_celsius_range(self, frame=None):
        """Return fixed or frame-derived grayscale endpoints in Celsius."""
        if self.display_min is not None:
            return (
                raw_to_celsius(self.display_min),
                raw_to_celsius(self.display_max),
            )

        if frame is None:
            return None

        values = self._validate_frame(frame)
        min_value, max_value = self._get_display_range(values)
        return raw_to_celsius(min_value), raw_to_celsius(max_value)

    def legend_extrema_celsius(self, frame=None):
        """Show extrema only when the grayscale spectrum has fixed endpoints."""
        if self.display_min is None:
            return None
        return super().legend_extrema_celsius(frame)

    def _color(
        self,
        value,
        min_value,
        max_value,
        color_cache,
    ):
        """
        Convert one temperature value into a grayscale RGB color.

        The final 8-bit intensity is used as the cache key. This avoids the
        precision loss caused by converting temperatures to integers before
        normalization.
        """
        if not math.isfinite(value):
            # Invalid sensor values are displayed as middle gray.
            intensity = self.INVALID_INTENSITY

        elif max_value <= min_value:
            # A frame with no temperature variation cannot be normalized.
            intensity = self.INVALID_INTENSITY

        else:
            # Normalize the temperature to a value between 0.0 and 1.0.
            normalized = (
                (value - min_value)
                / (max_value - min_value)
            )

            # Clamp values outside a fixed display range.
            normalized = max(0.0, min(1.0, normalized))

            # Convert the normalized value to an 8-bit intensity.
            # Adding 0.5 gives conventional rounding instead of truncation.
            intensity = int(normalized * 255.0 + 0.5)

        # Temperatures producing the same final intensity can safely reuse
        # the same RGB tuple.
        cached_color = color_cache.get(intensity)
        if cached_color is not None:
            return cached_color

        # Equal red, green, and blue values produce grayscale.
        color = (intensity, intensity, intensity)
        color_cache[intensity] = color

        return color

    def _get_nearest_map(self, width, height):
        """
        Map each output pixel to its nearest sensor pixel.

        Pixel-center mapping is used so that very small output images select
        central sensor pixels instead of being biased toward the top-left
        corner.
        """
        key = (width, height)

        # Reuse the most recent mapping when the output size has not changed.
        if key == self._nearest_map_key:
            return self._nearest_map_value

        # Map each output column to a source sensor column.
        x_map = [
            min(
                SENSOR_WIDTH - 1,
                int((output_x + 0.5) * SENSOR_WIDTH / width),
            )
            for output_x in range(width)
        ]

        # Map each output row to a source sensor row.
        y_map = [
            min(
                SENSOR_HEIGHT - 1,
                int((output_y + 0.5) * SENSOR_HEIGHT / height),
            )
            for output_y in range(height)
        ]

        self._nearest_map_key = key
        self._nearest_map_value = (x_map, y_map)

        return x_map, y_map

    def render_rgb(self, frame, width, height):
        """
        Convert a thermal frame into a blocky RGB image buffer.

        Nearest-neighbor enlargement is intentionally used so each thermal
        sensor cell remains visually distinct instead of being interpolated.

        Parameters
        ----------
        frame : sequence
            Flat row-major thermal frame containing one value per sensor cell.

        width : int
            Requested output width in pixels.

        height : int
            Requested output height in pixels.

        Returns
        -------
        bytes
            Raw RGB data containing width * height * 3 bytes.
        """
        try:
            width = int(width)
            height = int(height)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "The output width and height must be integer-compatible values."
            ) from exc

        if width <= 0 or height <= 0:
            raise ValueError(
                "The output width and height must be greater than zero."
            )

        # Validate the frame before attempting minimum, maximum, or indexing
        # operations.
        values = self._validate_frame(frame)

        # Select either the fixed display range or the current frame range.
        min_value, max_value = self._get_display_range(values)

        # Build or reuse the nearest-neighbor output mapping.
        x_map, y_map = self._get_nearest_map(width, height)

        # Cache RGB tuples by final grayscale intensity for this frame.
        color_cache = {}

        # Build the enlarged output representation of each sensor row once.
        # Those rows can then be reused when scaling vertically.
        source_rows = []

        for source_y in range(SENSOR_HEIGHT):
            source_row_start = source_y * SENSOR_WIDTH

            # Each RGB pixel requires three bytes.
            output_row = bytearray(width * 3)
            output_offset = 0

            for source_x in x_map:
                sensor_index = source_row_start + source_x
                sensor_value = values[sensor_index]

                red, green, blue = self._color(
                    sensor_value,
                    min_value,
                    max_value,
                    color_cache,
                )

                output_row[output_offset] = red
                output_row[output_offset + 1] = green
                output_row[output_offset + 2] = blue

                output_offset += 3

            source_rows.append(bytes(output_row))

        # Allocate the complete RGB image.
        output = bytearray(width * height * 3)
        row_size = width * 3
        output_offset = 0

        # Repeat the prebuilt sensor rows using nearest-neighbor scaling.
        for source_y in y_map:
            output[
                output_offset : output_offset + row_size
            ] = source_rows[source_y]

            output_offset += row_size

        # Return an immutable RGB buffer suitable for image or GUI libraries.
        return bytes(output)