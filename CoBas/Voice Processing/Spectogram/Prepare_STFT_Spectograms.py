from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


VOICE_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUTS = {
    "beacon": VOICE_ROOT / "Spectogram_Output" / "Beacon_STFT_15k_19p2k",
    "old_code": VOICE_ROOT / "Spectogram_Output" / "Old_Code_STFT_15k_19p2k",
}

DEFAULT_OUTPUT_ROOT = VOICE_ROOT / "Spectrogram"
TARGET_WIDTH = 400


def fix_time_width(spectrogram: np.ndarray, target_width: int = TARGET_WIDTH) -> np.ndarray:
    if spectrogram.ndim != 2:
        raise ValueError(f"Expected 2D spectrogram, got shape {spectrogram.shape}")

    freq_bins, width = spectrogram.shape

    if width < target_width:
        pad = target_width - width
        return np.pad(spectrogram, ((0, 0), (0, pad)), mode="constant")

    if width > target_width:
        return spectrogram[:, :target_width]

    return spectrogram


def normalize_spectrogram(spectrogram: np.ndarray) -> np.ndarray:
    spectrogram = spectrogram.astype(np.float32)
    spectrogram = np.nan_to_num(spectrogram, nan=0.0, posinf=0.0, neginf=0.0)
    return (spectrogram - spectrogram.mean()) / (spectrogram.std() + 1e-6)


def prepare_spectrogram_file(input_path: Path, output_path: Path) -> None:
    spectrogram = np.load(input_path).astype(np.float32)
    spectrogram = fix_time_width(spectrogram)
    spectrogram = normalize_spectrogram(spectrogram)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, spectrogram.astype(np.float32))


def prepare_folder(input_dir: Path, output_dir: Path) -> int:
    npy_paths = sorted(input_dir.glob("*.npy"))
    if not npy_paths:
        print(f"[SKIP] No STFT .npy files found in: {input_dir}")
        return 0

    processed = 0
    for input_path in npy_paths:
        output_path = output_dir / input_path.name
        prepare_spectrogram_file(input_path, output_path)
        processed += 1

    print(f"[OK] {input_dir} -> {output_dir}: {processed} prepared arrays")
    return processed


def prepare_all(beacon_input: Path, old_code_input: Path, output_root: Path) -> None:
    print(f"Old-code final preprocessing: pad/crop width={TARGET_WIDTH}, z-score normalize")

    beacon_count = prepare_folder(
        beacon_input,
        output_root / "Beacon_Prepared",
    )
    old_count = prepare_folder(
        old_code_input,
        output_root / "Old_Code_Prepared",
    )

    print()
    print(f"Prepared beacon spectrograms: {beacon_count}")
    print(f"Prepared old-code spectrograms: {old_count}")
    print(f"Output root: {output_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply old-code final preprocessing to raw STFT .npy files."
    )
    parser.add_argument(
        "--beacon-input",
        type=Path,
        default=DEFAULT_INPUTS["beacon"],
        help="Folder containing beacon raw STFT .npy files.",
    )
    parser.add_argument(
        "--old-code-input",
        type=Path,
        default=DEFAULT_INPUTS["old_code"],
        help="Folder containing old-code raw STFT .npy files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root folder for prepared spectrogram .npy files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_all(args.beacon_input, args.old_code_input, args.output_root)


if __name__ == "__main__":
    main()
