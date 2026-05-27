from __future__ import annotations

import argparse
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np


BEACON_FREQ = 10_000.0
BEACON_DURATION_SEC = 2.0
GUARD_SILENCE_SEC = 1.5


def run_command(command: list[str]) -> bool:
    result = subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


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


def beacon_scores(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    frame_sec = 0.10
    hop_sec = 0.02
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
    return [(start, end) for start, end, _ in sorted(candidates[:2], key=lambda item: item[0])]


def format_seconds(seconds: float) -> str:
    if float(seconds).is_integer():
        return f"{int(seconds):03d}s"

    return f"{seconds:06.2f}s".replace(".", "p")


def extract_audio_for_detection(input_video: Path, wav_path: Path) -> bool:
    return run_command([
        "ffmpeg",
        "-y",
        "-i", str(input_video),
        "-vn",
        "-ar", "48000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        str(wav_path),
    ])


def cut_video(input_video: Path, output_video: Path, start_sec: float, end_sec: float) -> bool:
    output_video.parent.mkdir(parents=True, exist_ok=True)
    duration_sec = end_sec - start_sec
    return run_command([
        "ffmpeg",
        "-y",
        "-ss", f"{start_sec:.3f}",
        "-t", f"{duration_sec:.3f}",
        "-i", str(input_video),
        "-c:v", "libx264",
        "-c:a", "aac",
        "-reset_timestamps", "1",
        str(output_video),
    ])


def segment_video_by_beacon(input_video: Path, output_folder: Path) -> bool:
    input_video = Path(input_video)
    output_folder = Path(output_folder)

    if not input_video.exists():
        print(f"Error: Input video file does not exist: {input_video}")
        return False

    with tempfile.TemporaryDirectory() as temp_dir:
        detection_wav = Path(temp_dir) / f"{input_video.stem}_beacon_detection.wav"
        if not extract_audio_for_detection(input_video, detection_wav):
            print(f"Beacon audio extraction failed: {input_video}")
            return False

        audio, sample_rate = read_wav_mono(detection_wav)
        beacons = detect_beacons(audio, sample_rate)

    first_beacon = beacons[0]
    second_beacon = beacons[1] if len(beacons) > 1 else None
    active_start_sec = first_beacon[1] + GUARD_SILENCE_SEC
    active_end_sec = second_beacon[0] - GUARD_SILENCE_SEC if second_beacon else audio.size / sample_rate

    if active_end_sec <= active_start_sec:
        print("Beacon video segmentation produced an empty active region.")
        return False

    interval = f"{format_seconds(active_start_sec)}-{format_seconds(active_end_sec)}"
    output_path = output_folder / f"{input_video.stem}_beacon_{interval}.mp4"

    if not cut_video(input_video, output_path, active_start_sec, active_end_sec):
        print(f"Beacon video cut failed: {input_video}")
        return False

    print(f"Detected beacon 1: {first_beacon[0]:.3f}s to {first_beacon[1]:.3f}s")
    if second_beacon:
        print(f"Detected beacon 2: {second_beacon[0]:.3f}s to {second_beacon[1]:.3f}s")
    else:
        print("Detected beacon 2: not found; using recording end as crop boundary")
    print(f"Beacon video segment saved: {output_path}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Segment raw video using audio beacon markers.")
    parser.add_argument("input_video")
    parser.add_argument("--output-folder", type=Path, required=True)
    args = parser.parse_args()

    if not segment_video_by_beacon(Path(args.input_video), args.output_folder):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
