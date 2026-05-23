from __future__ import annotations

import argparse
import wave
from pathlib import Path

import numpy as np
from scipy import signal


VOICE_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUTS = {
    "beacon": VOICE_ROOT / "Seperated_Audios" / "Beacon_Cropped" / "Segments_2s",
    "old_code": VOICE_ROOT / "Seperated_Audios" / "Old_Code_2s_Cropped",
}

DEFAULT_OUTPUT_ROOT = VOICE_ROOT / "Spectogram_Output"

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
    resampled = signal.resample_poly(audio, up, down).astype(np.float32)
    return resampled, TARGET_SAMPLE_RATE


def compute_stft_band(
    audio: np.ndarray,
    sample_rate: int,
    freq_low: float = FREQ_LOW,
    freq_high: float = FREQ_HIGH,
) -> np.ndarray:
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

    band_mask = (freqs >= freq_low) & (freqs <= freq_high)
    if not np.any(band_mask):
        raise ValueError(
            f"No STFT frequency bins found for {freq_low:.1f}-{freq_high:.1f} Hz "
            f"at sample rate {sample_rate}."
        )

    magnitude = np.abs(stft[band_mask, :]).astype(np.float32)
    return magnitude


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

        output_path = output_dir / f"{wav_path.stem}.npy"
        np.save(output_path, spectrogram)
        processed += 1

    print(f"[OK] {input_dir} -> {output_dir}: {processed} STFT arrays")
    return processed


def process_all(
    beacon_input: Path,
    old_code_input: Path,
    output_root: Path,
) -> None:
    print(f"STFT settings: n_fft={NFFT}, hop={HOP}, band={FREQ_LOW:.0f}-{FREQ_HIGH:.0f} Hz")

    beacon_count = process_folder(
        beacon_input,
        output_root / "Beacon_STFT_15k_19p2k",
    )
    old_count = process_folder(
        old_code_input,
        output_root / "Old_Code_STFT_15k_19p2k",
    )

    print()
    print(f"Beacon cropped spectrograms: {beacon_count}")
    print(f"Old-code cropped spectrograms: {old_count}")
    print(f"Output root: {output_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create cropped 15-19.2 kHz STFT spectrogram .npy files from voice crops."
    )
    parser.add_argument(
        "--beacon-input",
        type=Path,
        default=DEFAULT_INPUTS["beacon"],
        help="Folder containing beacon-cropped 2s WAV segments.",
    )
    parser.add_argument(
        "--old-code-input",
        type=Path,
        default=DEFAULT_INPUTS["old_code"],
        help="Folder containing old fixed 2s WAV segments.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root folder for generated STFT .npy arrays.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process_all(args.beacon_input, args.old_code_input, args.output_root)


if __name__ == "__main__":
    main()
