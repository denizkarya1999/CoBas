from __future__ import annotations

import argparse
import math
import wave
from pathlib import Path

import numpy as np


VOICE_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUTS = (
    VOICE_ROOT / "5_15sPause_BeaconProtocol.wav",
    VOICE_ROOT / "5_15sPause_NeaconProtocol.wav",
)

CHUNK_SECONDS = 2.0


def find_default_input() -> Path:
    for path in DEFAULT_INPUTS:
        if path.exists():
            return path

    checked_paths = ", ".join(str(path) for path in DEFAULT_INPUTS)
    raise FileNotFoundError(f"Could not find protocol WAV. Checked: {checked_paths}")


def read_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())

    if sample_width == 2:
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
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


def old_code_crop(input_path: Path, output_dir: Path) -> None:
    audio, sample_rate = read_wav_mono(input_path)
    chunk_samples = int(round(sample_rate * CHUNK_SECONDS))
    segment_count = math.floor(audio.size / chunk_samples)
    base_name = input_path.stem

    for index in range(segment_count):
        start = index * chunk_samples
        end = start + chunk_samples
        output_path = output_dir / f"{base_name}_old_seg{index:03d}.wav"
        write_wav_mono(output_path, audio[start:end], sample_rate)

    print(f"Input: {input_path}")
    print(f"Old-code crop size: {CHUNK_SECONDS:.1f}s")
    print(f"Saved 2s segments: {segment_count}")
    print(f"Output folder: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crop protocol audio into fixed 2s chunks.")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=VOICE_ROOT / "Seperated_Audios" / "Old_Code_2s_Cropped",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input if args.input is not None else find_default_input()
    old_code_crop(input_path, args.output_dir)


if __name__ == "__main__":
    main()
