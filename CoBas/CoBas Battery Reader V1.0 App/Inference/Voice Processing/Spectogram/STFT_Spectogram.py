from __future__ import annotations

import argparse
import wave
from pathlib import Path

import numpy as np
from scipy import signal


TARGET_SAMPLE_RATE = 48_000
NFFT = 2048
HOP = 512
FREQ_LOW = 15_000.0
FREQ_HIGH = 19_200.0


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


def resample_if_needed(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
    if sample_rate == TARGET_SAMPLE_RATE:
        return audio.astype(np.float32), sample_rate

    gcd = np.gcd(sample_rate, TARGET_SAMPLE_RATE)
    up = TARGET_SAMPLE_RATE // gcd
    down = sample_rate // gcd
    return signal.resample_poly(audio, up, down).astype(np.float32), TARGET_SAMPLE_RATE


def compute_stft_band(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    if audio.size < NFFT:
        audio = np.pad(audio, (0, NFFT - audio.size), mode="constant")

    freqs, _, stft = signal.stft(
        audio,
        fs=sample_rate,
        window="hann",
        nperseg=NFFT,
        noverlap=NFFT - HOP,
        nfft=NFFT,
        boundary="zeros",
        padded=True,
    )

    band_mask = (freqs >= FREQ_LOW) & (freqs <= FREQ_HIGH)
    if not np.any(band_mask):
        raise ValueError(f"No STFT bins found for {FREQ_LOW:.1f}-{FREQ_HIGH:.1f} Hz.")

    return np.abs(stft[band_mask, :]).astype(np.float32)


def process_folder(input_dir: Path, output_dir: Path) -> int:
    wav_paths = sorted(input_dir.glob("*.wav"))
    if not wav_paths:
        print(f"[SKIP] No WAV files found in: {input_dir}")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    processed = 0

    for wav_path in wav_paths:
        audio, sample_rate = read_wav_mono(wav_path)
        audio, sample_rate = resample_if_needed(audio, sample_rate)
        spectrogram = compute_stft_band(audio, sample_rate)
        np.save(output_dir / f"{wav_path.stem}.npy", spectrogram)
        processed += 1

    print(f"[OK] {input_dir} -> {output_dir}: {processed} STFT arrays")
    return processed


def main() -> None:
    parser = argparse.ArgumentParser(description="Create cropped 15-19.2 kHz STFT .npy files.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    process_folder(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
