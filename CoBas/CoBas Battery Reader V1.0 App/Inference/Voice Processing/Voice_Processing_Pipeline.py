from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np


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


def read_wav_mono(input_path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(input_path), "rb") as wav:
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


def write_wav_mono(output_path: Path, audio: np.ndarray, sample_rate: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.rint(np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)

    with wave.open(str(output_path), "wb") as wav:
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


def process_already_cropped_voice(input_path: Path, output_folder: Path) -> None:
    input_path = input_path.resolve()
    output_folder = output_folder.resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Voice input not found: {input_path}")

    work_folder = output_folder.parent / "_Voice_Processing_Work" / input_path.stem
    segment_folder = work_folder / "Segments_2s"
    raw_stft_folder = work_folder / "Beacon_STFT_15k_19p2k"

    delete_folder(work_folder)
    delete_folder(output_folder)

    segment_count = save_direct_segments(input_path, segment_folder)
    if segment_count == 0:
        raise RuntimeError(f"No complete 2-second voice segments were created from: {input_path}")

    run_step(
        "voice STFT spectrogram generation",
        [
            sys.executable,
            str(VOICE_PIPELINE_ROOT / "Spectogram" / "STFT_Spectogram.py"),
            "--input-dir",
            str(segment_folder),
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

    print("\nAlready beacon-cropped voice processing finished successfully.")
    print(f"Voice STFT spectrograms saved in: {output_folder}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CoBas beacon voice preprocessing.")
    parser.add_argument("input_voice")
    parser.add_argument("--output-folder", type=Path, required=True)
    parser.add_argument(
        "--already-cropped",
        action="store_true",
        help="Input voice was already segmented by raw-video beacon detection.",
    )
    args = parser.parse_args()

    if args.already_cropped:
        process_already_cropped_voice(Path(args.input_voice), args.output_folder)
    else:
        process_voice(Path(args.input_voice), args.output_folder)


if __name__ == "__main__":
    main()
