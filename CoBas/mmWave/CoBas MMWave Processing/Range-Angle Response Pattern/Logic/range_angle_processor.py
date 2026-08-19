"""Convert complex range/antenna matrices into range-angle power maps."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import (
    ANGLE_POINT_COUNT,
    AOP_12_CHANNEL_AZIMUTH_CHANNELS,
    AOP_12_CHANNEL_POSITIONS_HALF_WAVELENGTH,
    AOP_8_CHANNEL_AZIMUTH_CHANNELS,
    AOP_8_CHANNEL_POSITIONS_HALF_WAVELENGTH,
    AOP_4_CHANNEL_AZIMUTH_CHANNELS,
    AOP_4_CHANNEL_POSITIONS_HALF_WAVELENGTH,
    AOP_VIRTUAL_CHANNEL_PHASE_ROTATION,
    DISPLAY_DYNAMIC_RANGE_DB,
    DISPLAY_RANGE_BIN_INDICES,
    MAXIMUM_ANGLE_DEGREES,
    MINIMUM_ANGLE_DEGREES,
    RANGE_BIN_SPACING_M,
    RANGE_FFT_SIZE,
    TEMPORAL_SMOOTHING_ALPHA,
)
from .raw_iq_source import IQFrame


@dataclass(frozen=True, slots=True)
class RangeAngleFrame:
    """One display-ready range-angle response expressed in relative decibels."""

    frame_number: int
    angles_degrees: np.ndarray
    ranges_meters: np.ndarray
    power_db: np.ndarray
    peak_angle_degrees: float
    peak_range_meters: float
    input_antenna_count: int
    beamforming_channel_count: int


class RangeAngleProcessor:
    """Apply calibrated conventional beamforming to each USB1 I/Q frame."""

    def __init__(self) -> None:
        self.angles_degrees = np.linspace(
            MINIMUM_ANGLE_DEGREES,
            MAXIMUM_ANGLE_DEGREES,
            ANGLE_POINT_COUNT,
            dtype=np.float64,
        )
        self._range_bin_indices = np.asarray(
            DISPLAY_RANGE_BIN_INDICES,
            dtype=np.intp,
        )
        self.ranges_meters = (
            self._range_bin_indices.astype(np.float64) * RANGE_BIN_SPACING_M
        )

        self._all_phase_rotations = np.asarray(
            AOP_VIRTUAL_CHANNEL_PHASE_ROTATION,
            dtype=np.float64,
        )
        self._smoothed_power: np.ndarray | None = None

    def reset(self) -> None:
        """Discard temporal state before a new live capture."""
        self._smoothed_power = None

    def process(self, frame: IQFrame) -> RangeAngleFrame:
        """Create a range-angle map from one range-major virtual-array frame."""
        channels, positions = self._layout_for_antenna_count(
            frame.virtual_antenna_count
        )

        # Rebuild X[range_bin, virtual_antenna] = I + jQ from the compact rows.
        iq_matrix = np.zeros(
            (RANGE_FFT_SIZE, frame.virtual_antenna_count),
            dtype=np.complex128,
        )
        for sample in frame.samples:
            if (
                0 <= sample.range_bin < RANGE_FFT_SIZE
                and 0 <= sample.virtual_antenna < frame.virtual_antenna_count
            ):
                iq_matrix[sample.range_bin, sample.virtual_antenna] = complex(
                    sample.i,
                    sample.q,
                )

        # Retain only range bins whose centers fall inside 20-50 centimeters.
        azimuth_iq = iq_matrix[self._range_bin_indices][:, channels]
        phase_rotation = self._all_phase_rotations[channels]
        calibrated_iq = azimuth_iq * phase_rotation[np.newaxis, :]

        # Assign the same taper to channels sharing a horizontal position. This
        # allows the eight-channel format to combine two elevation rows without
        # biasing one azimuth location over another.
        aperture_window = np.hamming(4).astype(np.float64)
        window = aperture_window[np.rint(positions).astype(np.intp)]
        window_gain = max(float(np.sum(window)), np.finfo(float).eps)
        weighted_iq = calibrated_iq * window[np.newaxis, :]

        # With positions measured in lambda/2, the spatial phase is
        # pi * position * sin(theta). Each column is one candidate direction.
        angle_radians = np.radians(self.angles_degrees)
        steering = np.exp(
            -1j
            * np.pi
            * positions[:, np.newaxis]
            * np.sin(angle_radians)[np.newaxis, :]
        )

        # Matched steering across antennas produces response[range, angle].
        response = weighted_iq @ np.conjugate(steering)
        power = np.square(np.abs(response) / window_gain)

        if self._smoothed_power is None:
            self._smoothed_power = power
        else:
            alpha = TEMPORAL_SMOOTHING_ALPHA
            self._smoothed_power = (
                alpha * power + (1.0 - alpha) * self._smoothed_power
            )

        # A relative scale is honest for uncalibrated ADC/FFT counts: the
        # strongest cell is 0 dB and the displayed floor is -dynamic_range dB.
        epsilon = np.finfo(np.float64).tiny
        reference_power = max(float(np.max(self._smoothed_power)), epsilon)
        normalized_power = self._smoothed_power / reference_power
        power_db = 10.0 * np.log10(np.maximum(normalized_power, epsilon))
        power_db = np.clip(power_db, -DISPLAY_DYNAMIC_RANGE_DB, 0.0)

        peak_flat_index = int(np.argmax(self._smoothed_power))
        peak_range_index, peak_angle_index = np.unravel_index(
            peak_flat_index,
            self._smoothed_power.shape,
        )
        return RangeAngleFrame(
            frame_number=frame.frame_number,
            angles_degrees=self.angles_degrees,
            ranges_meters=self.ranges_meters,
            power_db=power_db,
            peak_angle_degrees=float(self.angles_degrees[peak_angle_index]),
            peak_range_meters=float(self.ranges_meters[peak_range_index]),
            input_antenna_count=frame.virtual_antenna_count,
            beamforming_channel_count=len(channels),
        )

    @staticmethod
    def _layout_for_antenna_count(
        antenna_count: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Choose the best supported AOP azimuth layout for a TLV payload."""
        if antenna_count >= 12:
            channels = AOP_12_CHANNEL_AZIMUTH_CHANNELS
            positions = AOP_12_CHANNEL_POSITIONS_HALF_WAVELENGTH
        elif antenna_count >= 8:
            channels = AOP_8_CHANNEL_AZIMUTH_CHANNELS
            positions = AOP_8_CHANNEL_POSITIONS_HALF_WAVELENGTH
        elif antenna_count >= 4:
            channels = AOP_4_CHANNEL_AZIMUTH_CHANNELS
            positions = AOP_4_CHANNEL_POSITIONS_HALF_WAVELENGTH
        else:
            raise ValueError(
                "Range-angle processing needs at least four virtual antenna "
                f"symbols, but USB1 supplied {antenna_count}."
            )
        return (
            np.asarray(channels, dtype=np.intp),
            np.asarray(positions, dtype=np.float64),
        )
