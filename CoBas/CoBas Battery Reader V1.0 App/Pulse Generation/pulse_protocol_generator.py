"""Generate, play, and optionally record one two-second CoBas chirp pulse."""

import argparse
import os
import time
import wave

import numpy as np
import sounddevice as sd


SAMPLE_RATE = 48_000
PULSE_DURATION_SECONDS = 2.0
START_FREQUENCY = 15_000.0
END_FREQUENCY = 19_200.0
AMPLITUDE = 0.85
FADE_MILLISECONDS = 5.0

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(SCRIPT_DIR, "Inputs")
OUTPUT_PATH = os.path.join(INPUT_DIR, "2_second_pulse.wav")


def apply_fade(signal, fade_milliseconds=FADE_MILLISECONDS):
    """Fade both ends of a pulse to prevent playback clicks."""
    fade_samples = int(
        round(fade_milliseconds * 1e-3 * SAMPLE_RATE)
    )

    if fade_samples == 0 or 2 * fade_samples >= signal.size:
        return signal

    window = np.ones_like(signal)
    window[:fade_samples] = np.linspace(
        0.0,
        1.0,
        fade_samples,
        dtype=np.float32,
    )
    window[-fade_samples:] = np.linspace(
        1.0,
        0.0,
        fade_samples,
        dtype=np.float32,
    )
    return signal * window


def build_pulse():
    """Build one linear chirp lasting exactly two seconds."""
    sample_count = int(round(SAMPLE_RATE * PULSE_DURATION_SECONDS))
    time_axis = np.arange(sample_count, dtype=np.float32) / SAMPLE_RATE
    sweep_rate = (
        END_FREQUENCY - START_FREQUENCY
    ) / PULSE_DURATION_SECONDS
    phase = 2.0 * np.pi * (
        START_FREQUENCY * time_axis
        + 0.5 * sweep_rate * time_axis * time_axis
    )
    signal = (AMPLITUDE * np.sin(phase)).astype(np.float32)
    return apply_fade(signal)


def write_wav(path, signal):
    """Write a floating-point mono signal as a 16-bit WAV."""
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    pcm = np.rint(
        np.clip(signal, -1.0, 1.0) * 32767.0
    ).astype(np.int16)

    with wave.open(path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(pcm.tobytes())


def read_wav(path):
    """Read a mono 16-bit WAV into a floating-point signal."""
    with wave.open(path, "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        rate = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())

    if channels != 1 or sample_width != 2:
        raise ValueError("The pulse WAV must be mono and 16-bit.")

    signal = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
    return signal / 32768.0, rate


def generate_pulse():
    """Generate and save a fresh copy of the two-second pulse."""
    signal = build_pulse()
    write_wav(OUTPUT_PATH, signal)
    print(f"Wrote: {OUTPUT_PATH}")
    print(f"Pulse duration: {signal.size / SAMPLE_RATE:.2f} seconds")
    return signal


def play_pulse(signal=None, rate=SAMPLE_RATE):
    """Play one pulse without opening an input stream."""
    if signal is None:
        signal, rate = read_wav(OUTPUT_PATH)

    try:
        playback_started_at = time.time()
        sd.play(signal, samplerate=rate, blocking=False)
        print(
            f"PLAYBACK_STARTED {playback_started_at:.9f}",
            flush=True,
        )
        sd.wait()
        print("PULSE_FINISHED", flush=True)
    finally:
        sd.stop()


def play_and_record_pulse(
    signal,
    recording_path,
    input_device=None,
):
    """
    Play and record one pulse through a single duplex stream.

    sounddevice creates the input and output streams together, records exactly
    the number of samples in the pulse, and closes both when playback finishes.
    """
    playback = np.asarray(signal, dtype=np.float32).reshape(-1, 1)

    try:
        playback_started_at = time.time()
        recording = sd.playrec(
            playback,
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            device=(input_device, None),
            blocking=False,
        )
        print(
            f"PLAYBACK_STARTED {playback_started_at:.9f}",
            flush=True,
        )
        sd.wait()
        write_wav(recording_path, recording.reshape(-1))
        print(f"PULSE_FINISHED {recording_path}", flush=True)
    finally:
        sd.stop()


def main():
    parser = argparse.ArgumentParser(
        description="Generate and use one two-second CoBas chirp pulse."
    )
    parser.add_argument(
        "--mode",
        choices=[
            "generate-only",
            "play-existing",
            "generate-and-play",
            "generate-play-record",
        ],
        default="generate-and-play",
    )
    parser.add_argument(
        "--record-output",
        help="WAV path for microphone audio captured during the pulse.",
    )
    parser.add_argument(
        "--input-device",
        type=int,
        default=None,
        help="sounddevice input-device index; defaults to the system input.",
    )
    args = parser.parse_args()

    if args.mode == "generate-only":
        generate_pulse()
        return

    if args.mode == "play-existing":
        play_pulse()
        return

    signal = generate_pulse()

    if args.mode == "generate-play-record":
        if not args.record_output:
            parser.error("--record-output is required for generate-play-record")
        play_and_record_pulse(
            signal,
            args.record_output,
            input_device=args.input_device,
        )
        return

    play_pulse(signal)


if __name__ == "__main__":
    main()
