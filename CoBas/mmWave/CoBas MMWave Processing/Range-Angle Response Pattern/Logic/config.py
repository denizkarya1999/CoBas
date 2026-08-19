"""Fixed signal-processing and display constants for the range-angle app."""

from __future__ import annotations


SPEED_OF_LIGHT_M_S = 299_792_458.0
ADC_SAMPLE_RATE_HZ = 2_000_000.0
FREQUENCY_SLOPE_HZ_S = 70e12
RANGE_FFT_SIZE = 64
RANGE_BIN_SPACING_M = (
    SPEED_OF_LIGHT_M_S
    * ADC_SAMPLE_RATE_HZ
    / (2.0 * FREQUENCY_SLOPE_HZ_S * RANGE_FFT_SIZE)
)

# Only this near-field window is processed, displayed, and logged. With the
# fixed profile it contains range bins 3 through 7 (approximately 20.1-46.8 cm).
MINIMUM_RANGE_METERS = 0.20
MAXIMUM_RANGE_METERS = 0.50
DISPLAY_RANGE_BIN_INDICES = tuple(
    range_bin
    for range_bin in range(RANGE_FFT_SIZE)
    if MINIMUM_RANGE_METERS
    <= range_bin * RANGE_BIN_SPACING_M
    <= MAXIMUM_RANGE_METERS
)
DISPLAY_RANGE_BIN_COUNT = len(DISPLAY_RANGE_BIN_INDICES)

# TI's IWR6843AOP virtual-channel geometry uses half-wavelength grid units.
# SDK/firmware variants can export 12, 8, or 4 complex antenna symbols:
#
# - For 12 symbols, indices 4, 6, 8, and 10 are one complete horizontal row.
# - For 8 symbols, all channels are usable for an elevation-zero azimuth slice;
#   pairs occupy the same horizontal positions on different elevation rows.
# - Four-symbol output is treated as an already selected horizontal row.
AOP_12_CHANNEL_AZIMUTH_CHANNELS = (4, 6, 8, 10)
AOP_12_CHANNEL_POSITIONS_HALF_WAVELENGTH = (0.0, 1.0, 2.0, 3.0)
AOP_8_CHANNEL_AZIMUTH_CHANNELS = (0, 1, 2, 3, 4, 5, 6, 7)
AOP_8_CHANNEL_POSITIONS_HALF_WAVELENGTH = (
    2.0,
    2.0,
    3.0,
    3.0,
    0.0,
    0.0,
    1.0,
    1.0,
)
AOP_4_CHANNEL_AZIMUTH_CHANNELS = (0, 1, 2, 3)
AOP_4_CHANNEL_POSITIONS_HALF_WAVELENGTH = (0.0, 1.0, 2.0, 3.0)

# Board feed directions introduce alternating 180-degree rotations. The chosen
# horizontal row uses the even channels, whose correction factors are all +1.
AOP_VIRTUAL_CHANNEL_PHASE_ROTATION = (
    1.0,
    -1.0,
    1.0,
    -1.0,
    1.0,
    -1.0,
    1.0,
    -1.0,
    1.0,
    -1.0,
    1.0,
    -1.0,
)

MINIMUM_ANGLE_DEGREES = -60.0
MAXIMUM_ANGLE_DEGREES = 60.0
ANGLE_POINT_COUNT = 121
DISPLAY_DYNAMIC_RANGE_DB = 50.0

# Weight assigned to the newest linear-power frame. Smoothing in linear power
# avoids the mathematical error of averaging decibel values directly.
TEMPORAL_SMOOTHING_ALPHA = 0.35

EVENT_QUEUE_SIZE = 4
GUI_REFRESH_INTERVAL_MS = 50

# Clean spectrogram-video settings. The selected response window is enlarged
# to a codec-safe image size while preserving the same format for all sessions.
SPECTROGRAM_FRAME_WIDTH = 724
SPECTROGRAM_FRAME_HEIGHT = 256
SPECTROGRAM_FRAME_RATE = 2.0
SPECTROGRAM_VIDEO_CODEC = "MJPG"
SPECTROGRAM_FRAME_QUEUE_SIZE = 64
FRAME_WINDOW_MARKER_FILENAME = ".window_20cm_50cm_-60deg_60deg"
