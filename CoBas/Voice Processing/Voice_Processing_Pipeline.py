from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np


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


def read_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())

    if sample_width == 2:
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 3:
        raw = np.frombuffer(frames, dtype=np.uint8).reshape(-1, 3)
        signed = (
            raw[:, 0].astype(np.int32)
            | (raw[:, 1].astype(np.int32) << 8)
            | (raw[:, 2].astype(np.int32) << 16)
        )
        signed = np.where(signed & 0x800000, signed - 0x1000000, signed)
        audio = signed.astype(np.float32) / 8388608.0
    elif sample_width == 4:
        audio = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes")

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    return audio, sample_rate


def write_wav_mono(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.rint(np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)

    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())


def save_two_second_segments(input_voice: Path, output_dir: Path) -> int:
    audio, sample_rate = read_wav_mono(input_voice)
    chunk_samples = int(round(sample_rate * 2.0))
    segment_count = math.floor(audio.size / chunk_samples)

    for index in range(segment_count):
        start = index * chunk_samples
        end = start + chunk_samples
        write_wav_mono(output_dir / f"{input_voice.stem}_seg{index:03d}.wav", audio[start:end], sample_rate)

    return segment_count


def save_direct_segments(input_path: Path, output_dir: Path) -> int:
    input_path = input_path.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_dir():
        total = 0
        for wav_path in sorted(input_path.glob("*.wav")):
            total += save_two_second_segments(wav_path, output_dir)
        return total

    return save_two_second_segments(input_path, output_dir)


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


def run_already_cropped_voice_pipeline(input_path: Path, final_output_folder: Path) -> None:
    work_folder = VOICE_ROOT / "Seperated_Audios" / "Already_Beacon_Cropped"
    segment_folder = work_folder / "Segments_2s"
    empty_old_code_folder = work_folder / "Old_Code_Empty"
    stft_root = VOICE_ROOT / "Spectogram_Output"

    clean_intermediates()
    delete_folder(final_output_folder)
    segment_count = save_direct_segments(input_path, segment_folder)
    empty_old_code_folder.mkdir(parents=True, exist_ok=True)

    if segment_count == 0:
        raise RuntimeError(f"No complete 2-second voice segments were created from: {input_path}")

    run_step(
        "STFT spectrogram generation",
        [
            sys.executable,
            str(VOICE_ROOT / "Spectogram" / "STFT_Spectogram.py"),
            "--beacon-input",
            str(segment_folder),
            "--old-code-input",
            str(empty_old_code_folder),
            "--output-root",
            str(stft_root),
        ],
    )

    run_step(
        "final spectrogram preprocessing",
        [
            sys.executable,
            str(VOICE_ROOT / "Spectogram" / "Prepare_STFT_Spectograms.py"),
            "--beacon-input",
            str(stft_root / "Beacon_STFT_15k_19p2k"),
            "--old-code-input",
            str(stft_root / "Old_Code_STFT_15k_19p2k"),
            "--output-root",
            str(final_output_folder),
        ],
    )

    clean_intermediates()

    print("\nAlready beacon-cropped voice processing finished successfully.")
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
    parser.add_argument(
        "--already-cropped",
        action="store_true",
        help="Input voice was already segmented by raw-video beacon detection.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input if args.input is not None else find_default_input()
    input_path = input_path.resolve()
    output_folder = args.output_folder if args.output_folder is not None else default_output_folder_for(input_path)
    if args.already_cropped:
        run_already_cropped_voice_pipeline(input_path, output_folder)
    else:
        run_voice_pipeline(input_path, output_folder)


if __name__ == "__main__":
    main()
