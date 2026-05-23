from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


TARGET_WIDTH = 400


def fix_time_width(spectrogram: np.ndarray) -> np.ndarray:
    if spectrogram.ndim != 2:
        raise ValueError(f"Expected 2D spectrogram, got shape {spectrogram.shape}")

    _, width = spectrogram.shape

    if width < TARGET_WIDTH:
        return np.pad(spectrogram, ((0, 0), (0, TARGET_WIDTH - width)), mode="constant")

    if width > TARGET_WIDTH:
        return spectrogram[:, :TARGET_WIDTH]

    return spectrogram


def normalize_spectrogram(spectrogram: np.ndarray) -> np.ndarray:
    spectrogram = spectrogram.astype(np.float32)
    spectrogram = np.nan_to_num(spectrogram, nan=0.0, posinf=0.0, neginf=0.0)
    return (spectrogram - spectrogram.mean()) / (spectrogram.std() + 1e-6)


def prepare_folder(input_dir: Path, output_dir: Path) -> int:
    npy_paths = sorted(input_dir.glob("*.npy"))
    if not npy_paths:
        print(f"[SKIP] No STFT .npy files found in: {input_dir}")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    processed = 0

    for input_path in npy_paths:
        spectrogram = np.load(input_path).astype(np.float32)
        spectrogram = fix_time_width(spectrogram)
        spectrogram = normalize_spectrogram(spectrogram)
        np.save(output_dir / input_path.name, spectrogram.astype(np.float32))
        processed += 1

    print(f"[OK] {input_dir} -> {output_dir}: {processed} prepared arrays")
    return processed


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare STFT .npy files before dataset creation.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    prepare_folder(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
