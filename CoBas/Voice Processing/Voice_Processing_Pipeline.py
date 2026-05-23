from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import wave
from pathlib import Path


VOICE_ROOT = Path(__file__).resolve().parent

DEFAULT_INPUTS = (
    VOICE_ROOT / "5_15sPause_BeaconProtocol.wav",
    VOICE_ROOT / "5_15sPause_NeaconProtocol.wav",
    Path("5_15sPause_BeaconProtocol.wav"),
    Path("5_15sPause_NeaconProtocol.wav"),
)

INTERMEDIATE_FOLDERS = (
    VOICE_ROOT / "Seperated_Audios",
    VOICE_ROOT / "Spectogram_Output",
)


def find_default_input() -> Path:
    for path in DEFAULT_INPUTS:
        if path.exists():
            return path

    checked_paths = ", ".join(str(path) for path in DEFAULT_INPUTS)
    raise FileNotFoundError(f"Could not find protocol WAV. Checked: {checked_paths}")


def print_voice_file_info(path: Path) -> None:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frame_count = wav.getnframes()

    duration = frame_count / sample_rate

    print(f"Opened: {path}")
    print(f"Channels: {channels}")
    print(f"Sample width: {sample_width} bytes")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Frames: {frame_count}")
    print(f"Duration: {duration:.3f} seconds")


def delete_folder(folder_path: Path) -> None:
    if folder_path.exists() and folder_path.is_dir():
        shutil.rmtree(folder_path)


def run_step(label: str, command: list[str]) -> None:
    print(f"\nStarting {label}...")
    result = subprocess.run(command)

    if result.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {result.returncode}")

    print(f"Finished {label}.")


def clean_intermediates() -> None:
    for folder in INTERMEDIATE_FOLDERS:
        delete_folder(folder)


def default_output_folder_for(input_path: Path) -> Path:
    return input_path.parent / f"{input_path.stem}_Spectogram"


def run_voice_pipeline(input_path: Path, final_output_folder: Path) -> None:
    print_voice_file_info(input_path)

    clean_intermediates()
    delete_folder(final_output_folder)

    run_step(
        "beacon cropping",
        [
            sys.executable,
            str(VOICE_ROOT / "Cropping" / "Beacon_Cropping.py"),
            "--input",
            str(input_path),
        ],
    )

    run_step(
        "old-code fixed cropping",
        [
            sys.executable,
            str(VOICE_ROOT / "Cropping" / "Old_Code_Cropping.py"),
            "--input",
            str(input_path),
        ],
    )

    run_step(
        "STFT spectrogram generation",
        [
            sys.executable,
            str(VOICE_ROOT / "Spectogram" / "STFT_Spectogram.py"),
        ],
    )

    run_step(
        "old-code final spectrogram preprocessing",
        [
            sys.executable,
            str(VOICE_ROOT / "Spectogram" / "Prepare_STFT_Spectograms.py"),
            "--output-root",
            str(final_output_folder),
        ],
    )

    clean_intermediates()

    print("\nVoice processing finished successfully.")
    print(f"Final STFT spectrogram folder: {final_output_folder}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the full CoBas voice preprocessing pipeline and keep only the "
            "final prepared STFT spectrogram output folder."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Input protocol WAV. Defaults to Voice Processing/5_15sPause_BeaconProtocol.wav.",
    )
    parser.add_argument(
        "--output-folder",
        type=Path,
        default=None,
        help="Final folder for prepared STFT spectrogram .npy files. Defaults to <voice_name>_Spectogram.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input if args.input is not None else find_default_input()
    input_path = input_path.resolve()
    output_folder = args.output_folder if args.output_folder is not None else default_output_folder_for(input_path)
    run_voice_pipeline(input_path, output_folder)


if __name__ == "__main__":
    main()
