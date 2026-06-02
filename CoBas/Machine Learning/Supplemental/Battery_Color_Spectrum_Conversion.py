import numpy as np
from PIL import Image
from skimage import img_as_float
import matplotlib.pyplot as plt
import random


def process_battery_heat_spectrum(
    image_path,
    crop_size=120,
    spectrum_width=300,
    spectrum_height=40
):
    """
    Processes a thermal image and generates a heat-intensity spectrum.

    The spectrum represents battery heat from:
    - dark orange = lower heat intensity
    - yellow = higher heat intensity

    No denoising.
    No masking.
    No color exclusion.
    """

    # --------------------------------------------------
    # 1. Load image and center crop
    # --------------------------------------------------

    img = Image.open(image_path).convert("RGB")

    w, h = img.size

    left = int((w - crop_size) / 2)
    top = int((h - crop_size) / 2)
    right = left + crop_size
    bottom = top + crop_size

    cropped_img = img.crop((left, top, right, bottom))

    # --------------------------------------------------
    # 2. Convert crop to float array
    # --------------------------------------------------

    img_data = img_as_float(
        np.array(cropped_img)
    )

    # --------------------------------------------------
    # 3. Compute heat intensity from RGB image
    # --------------------------------------------------
    # Higher brightness means higher estimated heat intensity.

    heat_intensity = (
        0.299 * img_data[:, :, 0] +
        0.587 * img_data[:, :, 1] +
        0.114 * img_data[:, :, 2]
    )

    # Normalize heat intensity to 0-1
    min_intensity = np.min(heat_intensity)
    max_intensity = np.max(heat_intensity)

    if max_intensity > min_intensity:
        heat_intensity = (
            heat_intensity - min_intensity
        ) / (
            max_intensity - min_intensity
        )
    else:
        heat_intensity = np.zeros_like(heat_intensity)

    # --------------------------------------------------
    # 4. Build 1D heat profile across the crop
    # --------------------------------------------------
    # Average vertically, so each x-position gets one heat value.

    heat_profile = np.mean(
        heat_intensity,
        axis=0
    )

    # Resize heat profile to desired spectrum width
    original_x = np.linspace(
        0,
        1,
        heat_profile.shape[0]
    )

    target_x = np.linspace(
        0,
        1,
        spectrum_width
    )

    heat_profile_resized = np.interp(
        target_x,
        original_x,
        heat_profile
    )

    # --------------------------------------------------
    # 5. Convert heat intensity to dark-orange -> yellow colors
    # --------------------------------------------------

    dark_orange = np.array([0.80, 0.25, 0.00])  # dark orange
    yellow = np.array([1.00, 1.00, 0.00])       # yellow

    spectrum_colors = (
        dark_orange * (1.0 - heat_profile_resized[:, None]) +
        yellow * heat_profile_resized[:, None]
    )

    spectrum_colors = np.clip(
        spectrum_colors,
        0.0,
        1.0
    )

    # --------------------------------------------------
    # 6. Create horizontal spectrum strip
    # --------------------------------------------------

    spectrum_strip = np.tile(
        spectrum_colors[np.newaxis, :, :],
        (spectrum_height, 1, 1)
    )

    return cropped_img, spectrum_strip, heat_profile_resized