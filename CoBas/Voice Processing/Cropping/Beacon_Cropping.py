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

BEACON_FREQ = 10_000.0
BEACON_DURATION_SEC = 2.0
GUARD_SILENCE_SEC = 1.5
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


def beacon_scores(
    audio: np.ndarray,
    sample_rate: int,
    frame_sec: float = 0.10,
    hop_sec: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    frame_size = int(round(frame_sec * sample_rate))
    hop_size = int(round(hop_sec * sample_rate))

    if audio.size < frame_size:
        raise ValueError("Audio is shorter than the beacon detector frame size.")

    window = np.hanning(frame_size).astype(np.float32)
    t = np.arange(frame_size, dtype=np.float32) / sample_rate
    reference = np.exp(-2j * np.pi * BEACON_FREQ * t).astype(np.complex64)

    frame_count = 1 + (audio.size - frame_size) // hop_size
    centers = np.zeros(frame_count, dtype=np.float32)
    scores = np.zeros(frame_count, dtype=np.float32)

    for index in range(frame_count):
        start = index * hop_size
        frame = audio[start:start + frame_size] * window
        energy = np.sqrt(np.mean(frame * frame)) + 1e-9
        tone_strength = abs(np.dot(frame, reference)) / frame_size
        centers[index] = (start + frame_size / 2) / sample_rate
        scores[index] = tone_strength / energy

    return centers, scores


def contiguous_regions(mask: np.ndarray) -> list[tuple[int, int]]:
    regions: list[tuple[int, int]] = []
    start = None

    for index, enabled in enumerate(mask):
        if enabled and start is None:
            start = index
        elif not enabled and start is not None:
            regions.append((start, index))
            start = None

    if start is not None:
        regions.append((start, len(mask)))

    return regions


def detect_beacons(audio: np.ndarray, sample_rate: int) -> list[tuple[float, float]]:
    centers, scores = beacon_scores(audio, sample_rate)
    threshold = max(float(np.percentile(scores, 95)) * 0.40, float(scores.max()) * 0.30)
    active = scores >= threshold
    min_frames = max(1, int(round((BEACON_DURATION_SEC * 0.45) / 0.02)))

    candidates = []
    for start, end in contiguous_regions(active):
        if end - start < min_frames:
            continue
        candidates.append((
            float(centers[start]),
            float(centers[end - 1]),
            float(scores[start:end].mean()),
        ))

    if not candidates:
        raise RuntimeError(
            "Could not detect any beacon regions. "
            f"Max score={scores.max():.4f}, threshold={threshold:.4f}"
        )

    candidates.sort(key=lambda item: item[2], reverse=True)
    selected = sorted(candidates[:2], key=lambda item: item[0])
    return [(start, end) for start, end, _ in selected]


def save_segments(
    audio: np.ndarray,
    sample_rate: int,
    output_dir: Path,
    base_name: str,
) -> int:
    chunk_samples = int(round(sample_rate * CHUNK_SECONDS))
    segment_count = math.floor(audio.size / chunk_samples)

    for index in range(segment_count):
        start = index * chunk_samples
        end = start + chunk_samples
        write_wav_mono(output_dir / f"{base_name}_seg{index:03d}.wav", audio[start:end], sample_rate)

    return segment_count


def crop_by_beacon(input_path: Path, output_dir: Path) -> None:
    audio, sample_rate = read_wav_mono(input_path)
    beacons = detect_beacons(audio, sample_rate)
    first_beacon = beacons[0]
    second_beacon = beacons[1] if len(beacons) > 1 else None

    active_start_sec = first_beacon[1] + GUARD_SILENCE_SEC
    active_end_sec = (
        second_beacon[0] - GUARD_SILENCE_SEC
        if second_beacon is not None
        else audio.size / sample_rate
    )
    active_start = max(0, int(round(active_start_sec * sample_rate)))
    active_end = min(audio.size, int(round(active_end_sec * sample_rate)))

    if active_end <= active_start:
        raise RuntimeError("Beacon crop produced an empty active region.")

    cropped = audio[active_start:active_end]
    base_name = input_path.stem
    crop_path = output_dir / f"{base_name}_beacon_active_region.wav"
    segment_dir = output_dir / "Segments_2s"

    write_wav_mono(crop_path, cropped, sample_rate)
    segment_count = save_segments(cropped, sample_rate, segment_dir, f"{base_name}_beacon")

    print(f"Input: {input_path}")
    print(f"Detected beacon 1: {first_beacon[0]:.3f}s to {first_beacon[1]:.3f}s")
    if second_beacon is not None:
        print(f"Detected beacon 2: {second_beacon[0]:.3f}s to {second_beacon[1]:.3f}s")
    else:
        print("Detected beacon 2: not found; using recording end as crop boundary")
    print(f"Beacon active crop: {active_start_sec:.3f}s to {active_end_sec:.3f}s")
    print(f"Saved crop: {crop_path}")
    print(f"Saved 2s segments: {segment_count}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crop protocol audio using the beacon markers.")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=VOICE_ROOT / "Seperated_Audios" / "Beacon_Cropped",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input if args.input is not None else find_default_input()
    crop_by_beacon(input_path, args.output_dir)


if __name__ == "__main__":
    main()
