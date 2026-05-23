from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import wave
from pathlib import Path


VOICE_PIPELINE_ROOT = Path(__file__).resolve().parent


def run_step(label: str, command: list[str]) -> None:
    print(f"\nStarting {label}...")
    result = subprocess.run(command)

    if result.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {result.returncode}")

    print(f"Finished {label}.")


def delete_folder(folder_path: Path) -> None:
    if folder_path.exists() and folder_path.is_dir():
        shutil.rmtree(folder_path)


def print_voice_info(input_path: Path) -> None:
    with wave.open(str(input_path), "rb") as wav:
        sample_rate = wav.getframerate()
        frame_count = wav.getnframes()
        channels = wav.getnchannels()

    print(f"Voice input: {input_path}")
    print(f"Channels: {channels}")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Duration: {frame_count / sample_rate:.3f} seconds")


def process_voice(input_voice: Path, output_folder: Path) -> None:
    input_voice = input_voice.resolve()
    output_folder = output_folder.resolve()

    if not input_voice.exists():
        raise FileNotFoundError(f"Voice file not found: {input_voice}")

    work_folder = output_folder.parent / "_Voice_Processing_Work" / input_voice.stem
    cropped_folder = work_folder / "Beacon_Cropped"
    raw_stft_folder = work_folder / "Beacon_STFT_15k_19p2k"

    print_voice_info(input_voice)
    delete_folder(work_folder)
    delete_folder(output_folder)

    run_step(
        "voice beacon cropping",
        [
            sys.executable,
            str(VOICE_PIPELINE_ROOT / "Cropping" / "Beacon_Cropping.py"),
            str(input_voice),
            "--output-dir",
            str(cropped_folder),
        ],
    )

    run_step(
        "voice STFT spectrogram generation",
        [
            sys.executable,
            str(VOICE_PIPELINE_ROOT / "Spectogram" / "STFT_Spectogram.py"),
            "--input-dir",
            str(cropped_folder / "Segments_2s"),
            "--output-dir",
            str(raw_stft_folder),
        ],
    )

    run_step(
        "voice final spectrogram preprocessing",
        [
            sys.executable,
            str(VOICE_PIPELINE_ROOT / "Spectogram" / "Prepare_STFT_Spectograms.py"),
            "--input-dir",
            str(raw_stft_folder),
            "--output-dir",
            str(output_folder),
        ],
    )

    delete_folder(work_folder)
    parent_work = work_folder.parent
    if parent_work.exists() and not any(parent_work.iterdir()):
        parent_work.rmdir()

    print("\nVoice processing finished successfully.")
    print(f"Voice STFT spectrograms saved in: {output_folder}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CoBas beacon voice preprocessing.")
    parser.add_argument("input_voice")
    parser.add_argument("--output-folder", type=Path, required=True)
    args = parser.parse_args()

    process_voice(Path(args.input_voice), args.output_folder)


if __name__ == "__main__":
    main()
